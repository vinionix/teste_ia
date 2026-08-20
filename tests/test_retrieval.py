import asyncio
import json

import pytest

from cpu_llm_lab.benchmark import (
    _median,
    _percentile,
    _recall_at_k,
    _reciprocal_rank,
)
from cpu_llm_lab.database import init_database
from cpu_llm_lab.retrieval import Retriever, _cosine_similarity


class FakeEmbeddingClient:
    async def embed_texts(self, model, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "home office" in lowered or "trabalha de casa" in lowered:
                vectors.append([1.0, 0.0])
            elif "ferias" in lowered or "férias" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.2, 0.2])
        return vectors, 1.0


def _seed_documents(tmp_path):
    seed = tmp_path / "documents.json"
    database = tmp_path / "documents.db"
    seed.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "title": "Política de Férias",
                    "category": "ferias",
                    "content": "Regras para férias.",
                },
                {
                    "id": 4,
                    "title": "Auxílio Home Office",
                    "category": "trabalho remoto",
                    "content": "A empresa oferece ajuda para home office.",
                },
            ]
        ),
        encoding="utf-8",
    )
    init_database(database, seed)
    return database


def test_cosine_similarity():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_retrieval_metrics():
    expected = {4}
    ranked = [2, 4, 1, 3, 5]

    assert _recall_at_k(expected, ranked, 1) == 0.0
    assert _recall_at_k(expected, ranked, 3) == 1.0
    assert _reciprocal_rank(expected, ranked) == pytest.approx(0.5)


def test_distribution_statistics():
    values = [10.0, 20.0, 30.0, 40.0]

    assert _median(values) == 25.0
    assert _percentile(values, 95) == pytest.approx(38.5)
    assert _median([]) == 0.0
    assert _percentile([], 95) == 0.0


def test_embedding_retriever_handles_semantic_paraphrase(tmp_path):
    database = _seed_documents(tmp_path)
    retriever = Retriever(FakeEmbeddingClient())

    documents, trace = asyncio.run(retriever.retrieve(
        database,
        "A empresa ajuda quem trabalha de casa?",
        mode="embedding",
        limit=1,
        embedding_model="fake-embedding",
    ))

    assert documents[0].id == 4
    assert trace.mode == "embedding"
    assert trace.embedding_ms == 2.0


def test_embedding_cache_can_be_cleared_for_cold_warm_runs(tmp_path):
    database = _seed_documents(tmp_path)
    retriever = Retriever(FakeEmbeddingClient())

    _, cold = asyncio.run(retriever.retrieve(
        database,
        "A empresa ajuda quem trabalha de casa?",
        mode="embedding",
        limit=1,
        embedding_model="fake-embedding",
    ))
    _, warm = asyncio.run(retriever.retrieve(
        database,
        "A empresa ajuda quem trabalha de casa?",
        mode="embedding",
        limit=1,
        embedding_model="fake-embedding",
    ))

    removed = retriever.clear_embedding_cache("fake-embedding")
    _, cold_again = asyncio.run(retriever.retrieve(
        database,
        "A empresa ajuda quem trabalha de casa?",
        mode="embedding",
        limit=1,
        embedding_model="fake-embedding",
    ))

    assert cold.embedding_ms == 2.0
    assert warm.embedding_ms == 1.0
    assert removed == 2
    assert cold_again.embedding_ms == 2.0
