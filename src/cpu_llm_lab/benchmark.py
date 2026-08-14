import json
import platform
from pathlib import Path

import psutil

from .database import search_documents
from .evaluator import evaluate_query
from .ollama_client import OllamaClient
from .schemas import BenchmarkResponse, BenchmarkRow, ModelResult, TestCase


def load_cases(path: Path) -> list[TestCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TestCase.model_validate(item) for item in raw]


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values) * 100, 2) if values else 0.0


async def run_benchmark(
    client: OllamaClient,
    models: list[str],
    cases_path: Path,
    database_path: Path,
    top_k: int = 3,
) -> BenchmarkResponse:
    cases = load_cases(cases_path)
    details: dict[str, list[ModelResult]] = {model: [] for model in models}

    for case in cases:
        documents = search_documents(database_path, case.question, limit=top_k)
        for model in models:
            result = await client.run_grounded_query(model, case.question, documents)
            if result.ok and result.answer:
                result.evaluation = evaluate_query(case, result.answer, documents)
            details[model].append(result)

    rows: list[BenchmarkRow] = []
    for model, results in details.items():
        successful = [
            result
            for result in results
            if result.ok and result.evaluation and result.metrics
        ]
        evaluations = [result.evaluation for result in successful if result.evaluation]
        retrieval_values = [bool(item.retrieval_hit) for item in evaluations]
        source_values = [
            item.source_accuracy for item in evaluations if item.source_accuracy is not None
        ]
        abstention_values = [
            item.abstention_ok for item in evaluations if item.abstention_ok is not None
        ]

        rows.append(
            BenchmarkRow(
                model=model,
                cases=len(results),
                successful_cases=len(successful),
                avg_factual_score=round(
                    sum(item.factual_score for item in evaluations) / len(evaluations), 2
                ) if evaluations else 0.0,
                retrieval_hit_rate=_rate(retrieval_values),
                source_accuracy_rate=_rate([bool(value) for value in source_values]),
                hallucination_free_rate=_rate(
                    [item.hallucination_free for item in evaluations]
                ),
                abstention_accuracy=_rate([bool(value) for value in abstention_values]),
                avg_total_ms=round(
                    sum(result.metrics.total_ms for result in successful) / len(successful), 2
                ) if successful else 0.0,
                avg_tokens_per_second=round(
                    sum(result.metrics.tokens_per_second for result in successful) / len(successful),
                    2,
                ) if successful else 0.0,
                cpu_only_all_runs=bool(successful)
                and all(result.metrics.cpu_only_verified for result in successful),
                errors=len(results) - len(successful),
            )
        )

    environment = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "top_k": top_k,
    }

    return BenchmarkResponse(rows=rows, details=details, environment=environment)
