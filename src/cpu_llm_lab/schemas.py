from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: int
    title: str
    category: str
    content: str


class RetrievedDocument(Document):
    score: int = 0


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    models: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=10)


class GroundedAnswer(BaseModel):
    resposta: str = Field(min_length=1, max_length=1600)
    fontes: list[int] = Field(default_factory=list)
    encontrado: bool


class MetricSnapshot(BaseModel):
    total_ms: float
    load_ms: float
    prompt_tokens: int
    output_tokens: int
    tokens_per_second: float
    process_rss_mb: float
    vram_bytes: int = 0
    cpu_only_verified: bool


class QueryEvaluation(BaseModel):
    retrieval_hit: bool | None = None
    required_facts_total: int = 0
    required_facts_preserved: int = 0
    factual_score: float = 0.0
    source_accuracy: bool | None = None
    hallucination_free: bool = True
    abstention_ok: bool | None = None
    notes: list[str] = Field(default_factory=list)


class ModelResult(BaseModel):
    model: str
    ok: bool
    question: str
    answer: GroundedAnswer | None = None
    raw_output: str = ""
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    metrics: MetricSnapshot | None = None
    evaluation: QueryEvaluation | None = None
    error: str | None = None


class TestCase(BaseModel):
    id: str
    question: str
    expected_document_ids: list[int] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    should_answer: bool = True


class BenchmarkRow(BaseModel):
    model: str
    cases: int
    successful_cases: int
    avg_factual_score: float
    retrieval_hit_rate: float
    source_accuracy_rate: float
    hallucination_free_rate: float
    abstention_accuracy: float
    avg_total_ms: float
    avg_tokens_per_second: float
    cpu_only_all_runs: bool
    errors: int


class BenchmarkResponse(BaseModel):
    rows: list[BenchmarkRow]
    details: dict[str, list[ModelResult]]
    environment: dict[str, Any]
