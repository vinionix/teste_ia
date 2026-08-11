from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceRecord(BaseModel):
    cliente: str = Field(min_length=1)
    plano: str = Field(min_length=1)
    valor: float = Field(ge=0)
    status: str = Field(min_length=1)
    vencimento: str = Field(min_length=1)


class FormattedText(BaseModel):
    titulo: str = Field(min_length=1, max_length=100)
    mensagem: str = Field(min_length=1, max_length=600)
    categoria: Literal["financeiro", "informativo", "suporte", "outro"]


class CompareRequest(BaseModel):
    record: SourceRecord
    models: list[str] = Field(default_factory=list)


class MetricSnapshot(BaseModel):
    total_ms: float
    load_ms: float
    prompt_tokens: int
    output_tokens: int
    tokens_per_second: float
    process_rss_mb: float
    vram_bytes: int = 0
    cpu_only_verified: bool


class Evaluation(BaseModel):
    json_valid: bool
    schema_valid: bool
    name_preserved: bool
    amount_preserved: bool
    due_date_preserved: bool
    expected_category: bool | None = None
    fidelity_score: float
    notes: list[str] = Field(default_factory=list)


class ModelResult(BaseModel):
    model: str
    ok: bool
    output: FormattedText | None = None
    raw_output: str = ""
    metrics: MetricSnapshot | None = None
    evaluation: Evaluation | None = None
    error: str | None = None


class TestCase(BaseModel):
    id: str
    record: SourceRecord
    expected_category: str


class BenchmarkRow(BaseModel):
    model: str
    cases: int
    successful_cases: int
    schema_success_rate: float
    avg_fidelity: float
    avg_total_ms: float
    avg_tokens_per_second: float
    cpu_only_all_runs: bool
    errors: int


class BenchmarkResponse(BaseModel):
    rows: list[BenchmarkRow]
    details: dict[str, list[ModelResult]]
    environment: dict[str, Any]
