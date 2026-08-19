import json
import time

import httpx
import psutil
from pydantic import ValidationError

from .config import EMBEDDING_OPTIONS, OLLAMA_URL, REQUEST_OPTIONS
from .observability import current_trace_id, traced
from .schemas import (
    GroundedAnswer,
    MetricSnapshot,
    ModelResult,
    RetrievedDocument,
    RetrievalTrace,
)

SYSTEM_PROMPT = """Você é um assistente interno de RH da empresa fictícia Aurora Labs.
Responda EXCLUSIVAMENTE com base nos DOCUMENTOS fornecidos nesta requisição.
Não use conhecimento externo, não complete lacunas e não invente políticas.
Preserve exatamente números, valores, prazos, datas, limites e condições relevantes.
Se os documentos não contiverem informação suficiente, defina encontrado=false,
use fontes=[] e responda que a informação não foi encontrada nos documentos fornecidos.
Quando responder, cite em fontes apenas os IDs dos documentos realmente utilizados.
A resposta deve ser curta, objetiva e em português do Brasil.
Retorne somente o objeto no schema solicitado."""


def _build_prompt(
    question: str,
    documents: list[RetrievedDocument],
) -> str:
    if documents:
        context = "\n\n".join(
            (
                f"[DOCUMENTO {doc.id}]\n"
                f"Título: {doc.title}\n"
                f"Categoria: {doc.category}\n"
                f"Conteúdo: {doc.content}"
            )
            for doc in documents
        )
    else:
        context = "Nenhum documento foi recuperado para esta consulta."

    return f"""DOCUMENTOS RECUPERADOS:
{context}

PERGUNTA DO FUNCIONÁRIO:
{question}

Responda apenas com base nos documentos acima."""


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL) -> None:
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=5.0)
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.base_url}/api/version"
            )
            return response.is_success
        except httpx.HTTPError:
            return False

    async def installed_models(self) -> set[str]:
        response = await self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return {
            item.get("name", "")
            for item in data.get("models", [])
            if item.get("name")
        }

    async def embed_texts(
        self,
        model: str,
        texts: list[str],
    ) -> tuple[list[list[float]], float]:
        if not texts:
            return [], 0.0

        with traced("ollama.embed", model=model, batch_size=len(texts)):
            wall_start = time.perf_counter()
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": model,
                        "input": texts,
                        "truncate": True,
                        "options": EMBEDDING_OPTIONS,
                    },
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Falha ao gerar embeddings com {model}: {exc}"
                ) from exc

            data = response.json()
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    "Ollama retornou uma quantidade inesperada de embeddings."
                )

            wall_ms = (time.perf_counter() - wall_start) * 1000
            total_ns = int(data.get("total_duration", 0) or 0)
            total_ms = total_ns / 1_000_000 if total_ns else wall_ms
            return embeddings, round(total_ms, 2)

    async def _vram_for(self, model: str) -> int | None:
        try:
            response = await self.client.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
            for item in response.json().get("models", []):
                names = {item.get("name"), item.get("model")}
                if model in names:
                    return int(item.get("size_vram", 0) or 0)
        except (httpx.HTTPError, ValueError, TypeError):
            return None
        return None

    async def run_grounded_query(
        self,
        model: str,
        question: str,
        documents: list[RetrievedDocument],
        retrieval: RetrievalTrace | None = None,
    ) -> ModelResult:
        schema = GroundedAnswer.model_json_schema()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_prompt(question, documents),
                },
            ],
            "stream": False,
            "think": False,
            "format": schema,
            "options": REQUEST_OPTIONS,
            "keep_alive": "2m",
        }

        rss_before = psutil.Process().memory_info().rss
        wall_start = time.perf_counter()

        with traced(
            "ollama.chat",
            model=model,
            document_count=len(documents),
        ):
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                wall_ms = (time.perf_counter() - wall_start) * 1000
                raw = data.get("message", {}).get("content", "")

                try:
                    parsed_json = json.loads(raw)
                    answer = GroundedAnswer.model_validate(parsed_json)
                except (json.JSONDecodeError, ValidationError) as exc:
                    return ModelResult(
                        model=model,
                        ok=False,
                        question=question,
                        raw_output=raw,
                        retrieved_documents=documents,
                        retrieval=retrieval,
                        trace_id=current_trace_id(),
                        error=f"Saída inválida: {exc}",
                    )

                vram = await self._vram_for(model)
                rss_after = psutil.Process().memory_info().rss
                eval_count = int(data.get("eval_count", 0) or 0)
                eval_duration = int(data.get("eval_duration", 0) or 0)
                tps = (
                    eval_count / (eval_duration / 1_000_000_000)
                    if eval_duration
                    else 0.0
                )

                metrics = MetricSnapshot(
                    total_ms=round(
                        int(
                            data.get(
                                "total_duration",
                                int(wall_ms * 1_000_000),
                            )
                            or int(wall_ms * 1_000_000)
                        )
                        / 1_000_000,
                        2,
                    ),
                    load_ms=round(
                        int(data.get("load_duration", 0) or 0)
                        / 1_000_000,
                        2,
                    ),
                    prompt_tokens=int(
                        data.get("prompt_eval_count", 0) or 0
                    ),
                    output_tokens=eval_count,
                    tokens_per_second=round(tps, 2),
                    process_rss_mb=round(
                        max(rss_before, rss_after) / 1024 / 1024,
                        2,
                    ),
                    vram_bytes=vram,
                    cpu_only_verified=(vram == 0) if vram is not None else None,
                )

                return ModelResult(
                    model=model,
                    ok=True,
                    question=question,
                    answer=answer,
                    raw_output=raw,
                    retrieved_documents=documents,
                    retrieval=retrieval,
                    metrics=metrics,
                    trace_id=current_trace_id(),
                )
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:500]
                return ModelResult(
                    model=model,
                    ok=False,
                    question=question,
                    retrieved_documents=documents,
                    retrieval=retrieval,
                    trace_id=current_trace_id(),
                    error=(
                        f"Ollama respondeu "
                        f"{exc.response.status_code}: {detail}"
                    ),
                )
            except httpx.HTTPError as exc:
                return ModelResult(
                    model=model,
                    ok=False,
                    question=question,
                    retrieved_documents=documents,
                    retrieval=retrieval,
                    trace_id=current_trace_id(),
                    error=f"Falha de conexão com Ollama: {exc}",
                )
