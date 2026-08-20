import hashlib
import math
import time
from pathlib import Path

from .config import DEFAULT_EMBEDDING_MODEL, HYBRID_EMBEDDING_WEIGHT
from .database import list_documents, rank_documents_lexical
from .observability import traced
from .ollama_client import OllamaClient
from .schemas import (
    Document,
    RetrievedDocument,
    RetrievalMode,
    RetrievalTrace,
)


def _document_text(document: Document) -> str:
    return (
        f"Título: {document.title}\n"
        f"Categoria: {document.category}\n"
        f"Conteúdo: {document.content}"
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class Retriever:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client
        self._embedding_cache: dict[
            tuple[str, int, str],
            list[float],
        ] = {}

    def clear_embedding_cache(self, model: str | None = None) -> int:
        if model is None:
            removed = len(self._embedding_cache)
            self._embedding_cache.clear()
            return removed

        keys = [
            key
            for key in self._embedding_cache
            if key[0] == model
        ]
        for key in keys:
            self._embedding_cache.pop(key, None)
        return len(keys)

    @staticmethod
    def _document_cache_key(
        model: str,
        document: Document,
    ) -> tuple[str, int, str]:
        digest = hashlib.sha256(
            _document_text(document).encode("utf-8")
        ).hexdigest()
        return model, document.id, digest

    async def _document_embeddings(
        self,
        model: str,
        documents: list[Document],
    ) -> tuple[dict[int, list[float]], float]:
        vectors: dict[int, list[float]] = {}
        missing: list[Document] = []

        for document in documents:
            key = self._document_cache_key(model, document)
            cached = self._embedding_cache.get(key)
            if cached is None:
                missing.append(document)
            else:
                vectors[document.id] = cached

        embedding_ms = 0.0
        if missing:
            generated, embedding_ms = await self.client.embed_texts(
                model,
                [_document_text(document) for document in missing],
            )
            for document, vector in zip(missing, generated):
                key = self._document_cache_key(model, document)
                self._embedding_cache[key] = vector
                vectors[document.id] = vector

        return vectors, embedding_ms

    async def _embedding_scores(
        self,
        documents: list[Document],
        query: str,
        embedding_model: str,
    ) -> tuple[dict[int, float], float]:
        document_vectors, document_ms = await self._document_embeddings(
            embedding_model,
            documents,
        )
        query_vectors, query_ms = await self.client.embed_texts(
            embedding_model,
            [query],
        )
        query_vector = query_vectors[0]
        scores = {
            document.id: _cosine_similarity(
                query_vector,
                document_vectors[document.id],
            )
            for document in documents
        }
        return scores, round(document_ms + query_ms, 2)

    @staticmethod
    def _rank_from_scores(
        documents: list[Document],
        scores: dict[int, float],
    ) -> list[RetrievedDocument]:
        ranked = [
            RetrievedDocument(
                **document.model_dump(),
                score=round(float(scores.get(document.id, 0.0)), 6),
            )
            for document in documents
        ]
        ranked.sort(key=lambda item: (-item.score, item.id))
        return ranked

    async def retrieve(
        self,
        db_path: Path,
        query: str,
        mode: RetrievalMode = "lexical",
        limit: int = 3,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> tuple[list[RetrievedDocument], RetrievalTrace]:
        with traced(
            "retrieval.search",
            mode=mode,
            top_k=limit,
            embedding_model=embedding_model if mode != "lexical" else "",
        ):
            started = time.perf_counter()
            embedding_ms = 0.0

            if mode == "lexical":
                ranked = rank_documents_lexical(db_path, query)
            else:
                documents = list_documents(db_path)
                embedding_scores, embedding_ms = await self._embedding_scores(
                    documents,
                    query,
                    embedding_model,
                )

                if mode == "embedding":
                    ranked = self._rank_from_scores(
                        documents,
                        embedding_scores,
                    )
                else:
                    lexical_ranked = rank_documents_lexical(
                        db_path,
                        query,
                    )
                    lexical_scores = {
                        document.id: document.score
                        for document in lexical_ranked
                    }
                    lexical_max = max(lexical_scores.values(), default=1.0)
                    hybrid_scores: dict[int, float] = {}

                    for document in documents:
                        lexical_normalized = (
                            lexical_scores.get(document.id, 0.0)
                            / lexical_max
                        )
                        semantic_normalized = (
                            embedding_scores.get(document.id, 0.0) + 1.0
                        ) / 2.0
                        hybrid_scores[document.id] = (
                            HYBRID_EMBEDDING_WEIGHT * semantic_normalized
                            + (1.0 - HYBRID_EMBEDDING_WEIGHT)
                            * lexical_normalized
                        )

                    ranked = self._rank_from_scores(
                        documents,
                        hybrid_scores,
                    )

            selected = ranked[:limit]
            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )
            trace = RetrievalTrace(
                mode=mode,
                latency_ms=latency_ms,
                embedding_ms=embedding_ms,
                ranked_document_ids=[item.id for item in selected],
                ranked_scores=[item.score for item in selected],
            )
            return selected, trace
