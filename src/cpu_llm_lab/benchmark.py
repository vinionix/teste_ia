import json
import platform
from pathlib import Path

import psutil

from .ollama_client import OllamaClient
from .schemas import BenchmarkResponse, BenchmarkRow, ModelResult, TestCase


def load_cases(path: Path) -> list[TestCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TestCase.model_validate(item) for item in raw]


async def run_benchmark(
    client: OllamaClient,
    models: list[str],
    cases_path: Path,
) -> BenchmarkResponse:
    cases = load_cases(cases_path)
    details: dict[str, list[ModelResult]] = {model: [] for model in models}

    for model in models:
        for case in cases:
            result = await client.run_model(
                model=model,
                record=case.record,
                expected_category=case.expected_category,
            )
            details[model].append(result)

    rows: list[BenchmarkRow] = []
    for model, results in details.items():
        successful = [r for r in results if r.ok and r.evaluation and r.metrics]
        rows.append(
            BenchmarkRow(
                model=model,
                cases=len(results),
                successful_cases=len(successful),
                schema_success_rate=round(len(successful) / len(results) * 100, 2) if results else 0,
                avg_fidelity=round(sum(r.evaluation.fidelity_score for r in successful) / len(successful), 2) if successful else 0,
                avg_total_ms=round(sum(r.metrics.total_ms for r in successful) / len(successful), 2) if successful else 0,
                avg_tokens_per_second=round(sum(r.metrics.tokens_per_second for r in successful) / len(successful), 2) if successful else 0,
                cpu_only_all_runs=bool(successful) and all(r.metrics.cpu_only_verified for r in successful),
                errors=len(results) - len(successful),
            )
        )

    environment = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }

    return BenchmarkResponse(rows=rows, details=details, environment=environment)
