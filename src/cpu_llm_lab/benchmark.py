import json
import platform
from pathlib import Path

import psutil

from .evaluator import evaluate_query
from .observability import traced
from .ollama_client import OllamaClient
from .retrieval import Retriever
from .schemas import (
    BenchmarkResponse,
    BenchmarkRow,
    ModelResult,
    RetrievalMode,
    RetrievalTrace,
    TestCase,
)


def load_cases(path: Path) -> list[TestCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TestCase.model_validate(item) for item in raw]


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values) * 100, 2) if values else 0.0


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _recall_at_k(
    expected_ids: set[int],
    ranked_ids: list[int],
    k: int,
) -> float:
    if not expected_ids:
        return 0.0
    found = expected_ids & set(ranked_ids[:k])
    return len(found) / len(expected_ids)


def _reciprocal_rank(
    expected_ids: set[int],
    ranked_ids: list[int],
) -> float:
    if not expected_ids:
        return 0.0
    for position, document_id in enumerate(ranked_ids, start=1):
        if document_id in expected_ids:
            return 1.0 / position
    return 0.0


async def run_benchmark(
    client: OllamaClient,
    retriever: Retriever,
    models: list[str],
    retrieval_modes: list[RetrievalMode],
    cases_path: Path,
    database_path: Path,
    top_k: int = 3,
    embedding_model: str = "embeddinggemma",
) -> BenchmarkResponse:
    cases = load_cases(cases_path)
    details: dict[str, list[ModelResult]] = {
        f"{mode}::{model}": []
        for mode in retrieval_modes
        for model in models
    }
    retrieval_stats = {
        mode: {
            "recall_1": [],
            "recall_3": [],
            "recall_5": [],
            "mrr": [],
            "latency_ms": [],
            "embedding_ms": [],
        }
        for mode in retrieval_modes
    }

    with traced(
        "benchmark.run",
        case_count=len(cases),
        model_count=len(models),
        retrieval_mode_count=len(retrieval_modes),
        top_k=top_k,
    ):
        for case in cases:
            with traced("benchmark.case", case_id=case.id):
                for mode in retrieval_modes:
                    ranking_k = max(top_k, 5)
                    try:
                        ranked, retrieval_trace = await retriever.retrieve(
                            database_path,
                            case.question,
                            mode=mode,
                            limit=ranking_k,
                            embedding_model=embedding_model,
                        )
                    except RuntimeError as exc:
                        for model in models:
                            details[f"{mode}::{model}"].append(
                                ModelResult(
                                    model=model,
                                    ok=False,
                                    question=case.question,
                                    error=(
                                        f"Falha no retrieval {mode}: {exc}"
                                    ),
                                )
                            )
                        continue

                    ranked_ids = [
                        document.id
                        for document in ranked
                    ]
                    stats = retrieval_stats[mode]
                    stats["latency_ms"].append(
                        retrieval_trace.latency_ms
                    )
                    stats["embedding_ms"].append(
                        retrieval_trace.embedding_ms
                    )

                    expected_ids = set(case.expected_document_ids)
                    if expected_ids:
                        stats["recall_1"].append(
                            _recall_at_k(expected_ids, ranked_ids, 1)
                        )
                        stats["recall_3"].append(
                            _recall_at_k(expected_ids, ranked_ids, 3)
                        )
                        stats["recall_5"].append(
                            _recall_at_k(expected_ids, ranked_ids, 5)
                        )
                        stats["mrr"].append(
                            _reciprocal_rank(expected_ids, ranked_ids)
                        )

                    generation_documents = ranked[:top_k]
                    generation_trace = RetrievalTrace(
                        mode=retrieval_trace.mode,
                        latency_ms=retrieval_trace.latency_ms,
                        embedding_ms=retrieval_trace.embedding_ms,
                        ranked_document_ids=ranked_ids,
                        ranked_scores=[
                            document.score
                            for document in ranked
                        ],
                    )

                    for model in models:
                        result = await client.run_grounded_query(
                            model,
                            case.question,
                            generation_documents,
                            retrieval=generation_trace,
                        )
                        if result.ok and result.answer:
                            result.evaluation = evaluate_query(
                                case,
                                result.answer,
                                generation_documents,
                            )
                        details[f"{mode}::{model}"].append(result)

    rows: list[BenchmarkRow] = []
    for mode in retrieval_modes:
        mode_stats = retrieval_stats[mode]
        for model in models:
            results = details[f"{mode}::{model}"]
            successful = [
                result
                for result in results
                if result.ok and result.evaluation and result.metrics
            ]
            evaluations = [
                result.evaluation
                for result in successful
                if result.evaluation
            ]
            retrieval_values = [
                bool(item.retrieval_hit)
                for item in evaluations
            ]
            source_values = [
                item.source_accuracy
                for item in evaluations
                if item.source_accuracy is not None
            ]
            abstention_values = [
                item.abstention_ok
                for item in evaluations
                if item.abstention_ok is not None
            ]
            cpu_values = [
                result.metrics.cpu_only_verified
                for result in successful
                if result.metrics
            ]
            if not cpu_values or any(value is None for value in cpu_values):
                cpu_only_all_runs = None
            else:
                cpu_only_all_runs = all(bool(value) for value in cpu_values)

            rows.append(
                BenchmarkRow(
                    model=model,
                    retrieval_mode=mode,
                    embedding_model=(
                        embedding_model
                        if mode in {"embedding", "hybrid"}
                        else None
                    ),
                    cases=len(results),
                    successful_cases=len(successful),
                    avg_factual_score=(
                        round(
                            sum(
                                item.factual_score
                                for item in evaluations
                            )
                            / len(evaluations),
                            2,
                        )
                        if evaluations
                        else 0.0
                    ),
                    retrieval_hit_rate=_rate(retrieval_values),
                    recall_at_1=round(
                        _average(mode_stats["recall_1"]) * 100,
                        2,
                    ),
                    recall_at_3=round(
                        _average(mode_stats["recall_3"]) * 100,
                        2,
                    ),
                    recall_at_5=round(
                        _average(mode_stats["recall_5"]) * 100,
                        2,
                    ),
                    mrr=round(_average(mode_stats["mrr"]), 4),
                    source_accuracy_rate=_rate(
                        [bool(value) for value in source_values]
                    ),
                    hallucination_free_rate=_rate(
                        [
                            item.hallucination_free
                            for item in evaluations
                        ]
                    ),
                    abstention_accuracy=_rate(
                        [bool(value) for value in abstention_values]
                    ),
                    avg_retrieval_ms=_average(
                        mode_stats["latency_ms"]
                    ),
                    avg_embedding_ms=_average(
                        mode_stats["embedding_ms"]
                    ),
                    avg_total_ms=(
                        round(
                            sum(
                                result.metrics.total_ms
                                for result in successful
                            )
                            / len(successful),
                            2,
                        )
                        if successful
                        else 0.0
                    ),
                    avg_tokens_per_second=(
                        round(
                            sum(
                                result.metrics.tokens_per_second
                                for result in successful
                            )
                            / len(successful),
                            2,
                        )
                        if successful
                        else 0.0
                    ),
                    cpu_only_all_runs=cpu_only_all_runs,
                    errors=len(results) - len(successful),
                )
            )

    environment = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(
            psutil.virtual_memory().total / 1024**3,
            2,
        ),
        "top_k": top_k,
        "retrieval_modes": retrieval_modes,
        "embedding_model": embedding_model,
        "benchmark_cases": len(cases),
    }

    return BenchmarkResponse(
        rows=rows,
        details=details,
        environment=environment,
    )
