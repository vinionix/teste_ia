import json
import time

import httpx
import psutil
from pydantic import ValidationError

from .config import OLLAMA_URL, REQUEST_OPTIONS
from .evaluator import evaluate_output
from .schemas import FormattedText, MetricSnapshot, ModelResult, SourceRecord

SYSTEM_PROMPT = """Você é um formatador de texto determinístico.
Use SOMENTE os dados recebidos. Não invente nomes, valores, datas, status ou fatos.
Escreva em português do Brasil, de forma curta e natural.
A categoria deve ser uma destas: financeiro, informativo, suporte, outro.
Para pagamentos pendentes, use financeiro.
Retorne somente o objeto no schema solicitado."""


def _build_prompt(record: SourceRecord) -> str:
    return (
        "Transforme o registro abaixo em uma mensagem curta para o cliente. "
        "Preserve explicitamente o nome, o valor e o vencimento na mensagem.\n\n"
        f"REGISTRO:\n{record.model_dump_json(indent=2)}"
    )


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL) -> None:
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0))

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/api/version")
            return response.is_success
        except httpx.HTTPError:
            return False

    async def installed_models(self) -> set[str]:
        response = await self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return {item.get("name", "") for item in data.get("models", [])}

    async def _vram_for(self, model: str) -> int:
        try:
            response = await self.client.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
            for item in response.json().get("models", []):
                names = {item.get("name"), item.get("model")}
                if model in names:
                    return int(item.get("size_vram", 0) or 0)
        except (httpx.HTTPError, ValueError, TypeError):
            pass
        return 0

    async def run_model(
        self,
        model: str,
        record: SourceRecord,
        expected_category: str | None = None,
    ) -> ModelResult:
        schema = FormattedText.model_json_schema()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(record)},
            ],
            "stream": False,
            "think": False,
            "format": schema,
            "options": REQUEST_OPTIONS,
            "keep_alive": "2m",
        }

        rss_before = psutil.Process().memory_info().rss
        wall_start = time.perf_counter()

        try:
            response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            raw = data.get("message", {}).get("content", "")

            try:
                parsed_json = json.loads(raw)
                output = FormattedText.model_validate(parsed_json)
            except (json.JSONDecodeError, ValidationError) as exc:
                return ModelResult(
                    model=model,
                    ok=False,
                    raw_output=raw,
                    error=f"Saída inválida: {exc}",
                )

            vram = await self._vram_for(model)
            rss_after = psutil.Process().memory_info().rss
            eval_count = int(data.get("eval_count", 0) or 0)
            eval_duration = int(data.get("eval_duration", 0) or 0)
            tps = (eval_count / (eval_duration / 1_000_000_000)) if eval_duration else 0.0

            metrics = MetricSnapshot(
                total_ms=round(int(data.get("total_duration", 0) or int(wall_ms * 1_000_000)) / 1_000_000, 2),
                load_ms=round(int(data.get("load_duration", 0) or 0) / 1_000_000, 2),
                prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
                output_tokens=eval_count,
                tokens_per_second=round(tps, 2),
                process_rss_mb=round(max(rss_before, rss_after) / 1024 / 1024, 2),
                vram_bytes=vram,
                cpu_only_verified=vram == 0,
            )

            return ModelResult(
                model=model,
                ok=True,
                output=output,
                raw_output=raw,
                metrics=metrics,
                evaluation=evaluate_output(record, output, expected_category),
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            return ModelResult(model=model, ok=False, error=f"Ollama respondeu {exc.response.status_code}: {detail}")
        except httpx.HTTPError as exc:
            return ModelResult(model=model, ok=False, error=f"Falha de conexão com Ollama: {exc}")
