from fastapi.testclient import TestClient

import cpu_llm_lab.app as app_module
from cpu_llm_lab.schemas import BenchmarkResponse


INSTALLED_MODELS = {
    "gemma3:270m",
    "qwen3:0.6b",
    "gemma3:1b",
    "llama3.2:1b",
    "qwen3:1.7b",
    "embeddinggemma",
}


class FakeOllamaClient:
    async def health(self) -> bool:
        return True

    async def installed_models(self) -> set[str]:
        return INSTALLED_MODELS


class FakeRetriever:
    pass


async def fake_run_benchmark(
    client,
    retriever,
    models,
    retrieval_modes,
    cases_path,
    database_path,
    top_k=3,
    embedding_model="embeddinggemma",
) -> BenchmarkResponse:
    return BenchmarkResponse(
        rows=[],
        details={
            f"{mode}::{model}": []
            for mode in retrieval_modes
            for model in models
        },
        environment={
            "top_k": top_k,
            "received_models": models,
            "retrieval_modes": retrieval_modes,
            "embedding_model": embedding_model,
        },
    )


def make_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(
        app_module,
        "client",
        FakeOllamaClient(),
    )
    monkeypatch.setattr(
        app_module,
        "retriever",
        FakeRetriever(),
    )
    monkeypatch.setattr(
        app_module,
        "run_benchmark",
        fake_run_benchmark,
    )
    return TestClient(app_module.app)


def test_benchmark_accepts_repeated_model_query_parameters(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post(
        "/api/benchmark"
        "?models=gemma3%3A270m"
        "&models=qwen3%3A0.6b"
        "&models=gemma3%3A1b"
        "&models=llama3.2%3A1b"
        "&models=qwen3%3A1.7b"
        "&top_k=3"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["environment"]["received_models"] == [
        "gemma3:270m",
        "qwen3:0.6b",
        "gemma3:1b",
        "llama3.2:1b",
        "qwen3:1.7b",
    ]
    assert body["environment"]["retrieval_modes"] == ["lexical"]
    assert body["environment"]["top_k"] == 3


def test_benchmark_accepts_repeated_retrieval_modes(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post(
        "/api/benchmark"
        "?models=qwen3%3A0.6b"
        "&retrieval_modes=lexical"
        "&retrieval_modes=embedding"
        "&retrieval_modes=hybrid"
        "&top_k=3"
    )

    assert response.status_code == 200
    assert response.json()["environment"]["retrieval_modes"] == [
        "lexical",
        "embedding",
        "hybrid",
    ]


def test_benchmark_requires_at_least_one_model(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post("/api/benchmark?top_k=3")

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Selecione pelo menos um modelo instalado."
    )


def test_benchmark_rejects_models_not_installed(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post(
        "/api/benchmark?models=nao-instalado%3A1b&top_k=3"
    )

    assert response.status_code == 400
    assert "nao-instalado:1b" in response.json()["detail"]


def test_benchmark_validates_top_k_lower_bound(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post(
        "/api/benchmark?models=qwen3%3A0.6b&top_k=0"
    )

    assert response.status_code == 422


def test_benchmark_validates_top_k_upper_bound(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post(
        "/api/benchmark?models=qwen3%3A0.6b&top_k=11"
    )

    assert response.status_code == 422
