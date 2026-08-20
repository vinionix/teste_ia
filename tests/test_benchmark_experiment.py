import asyncio
import json

from cpu_llm_lab.benchmark import run_benchmark
from cpu_llm_lab.schemas import (
    GroundedAnswer,
    MetricSnapshot,
    ModelResult,
    RetrievedDocument,
    RetrievalTrace,
)


class FakeRetriever:
    def __init__(self):
        self.cold_next = False
        self.clear_calls = 0

    def clear_embedding_cache(self, model=None):
        self.cold_next = True
        self.clear_calls += 1
        return 1

    async def retrieve(
        self,
        db_path,
        query,
        mode="lexical",
        limit=3,
        embedding_model="embeddinggemma:latest",
    ):
        cold = self.cold_next and mode in {"embedding", "hybrid"}
        self.cold_next = False
        latency = 100.0 if cold else 30.0
        embedding = 80.0 if cold else 20.0
        document = RetrievedDocument(
            id=1,
            title="Auxílio Home Office",
            category="trabalho_hibrido",
            content="O auxílio é de R$ 180,00 mensais.",
            score=1.0,
        )
        return [document], RetrievalTrace(
            mode=mode,
            latency_ms=latency,
            embedding_ms=embedding,
            ranked_document_ids=[1],
            ranked_scores=[1.0],
        )


class FakeOllamaClient:
    async def run_grounded_query(
        self,
        model,
        question,
        documents,
        retrieval=None,
    ):
        return ModelResult(
            model=model,
            ok=True,
            question=question,
            answer=GroundedAnswer(
                resposta="O auxílio é de R$ 180,00 mensais.",
                fontes=[1],
                encontrado=True,
            ),
            retrieved_documents=documents,
            retrieval=retrieval,
            metrics=MetricSnapshot(
                total_ms=200.0,
                load_ms=10.0,
                prompt_tokens=100,
                output_tokens=20,
                tokens_per_second=25.0,
                process_rss_mb=100.0,
                vram_bytes=0,
                cpu_only_verified=True,
            ),
        )


def test_repeated_benchmark_controls_cold_and_warm_runs(tmp_path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps([
            {
                "id": "home_office",
                "question": "Existe auxílio para trabalhar de casa?",
                "expected_document_ids": [1],
                "required_facts": ["R$ 180,00"],
                "forbidden_facts": ["R$ 300,00"],
                "should_answer": True,
            }
        ]),
        encoding="utf-8",
    )

    retriever = FakeRetriever()
    response = asyncio.run(run_benchmark(
        FakeOllamaClient(),
        retriever,
        ["fake-model"],
        ["embedding", "hybrid"],
        cases_path,
        tmp_path / "unused.db",
        top_k=1,
        repetitions=3,
        order_seed=7,
    ))

    assert retriever.clear_calls == 2
    assert response.environment["planned_llm_executions"] == 6
    assert response.environment["repetitions"] == 3
    assert len(response.rows) == 2

    for row in response.rows:
        assert row.cases == 1
        assert row.successful_cases == 1
        assert row.executions == 3
        assert row.successful_executions == 3
        assert row.avg_cold_retrieval_ms == 100.0
        assert row.avg_warm_retrieval_ms == 30.0
        assert row.avg_cold_embedding_ms == 80.0
        assert row.avg_warm_embedding_ms == 20.0
        assert row.median_total_ms == 200.0
        assert row.p95_total_ms == 200.0
        assert row.recall_at_1 == 100.0
        assert row.retrieval_hit_rate == 100.0
