import os
from pathlib import Path

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "hr_documents.db"
DOCUMENTS_SEED_PATH = DATA_DIR / "hr_documents.json"
CASES_PATH = DATA_DIR / "test_cases.json"

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:latest")
HYBRID_EMBEDDING_WEIGHT = min(
    1.0,
    max(0.0, float(os.getenv("HYBRID_EMBEDDING_WEIGHT", "0.5"))),
)

RECOMMENDED_MODELS = [
    "gemma3:270m",
    "smollm2:135m",
    "smollm2:360m",
    "qwen3:0.6b",
    "gemma3:1b",
    "llama3.2:1b",
    "deepseek-r1:1.5b",
    "qwen3:1.7b",
    "smollm2:1.7b",
    "llama3.2:3b",
    "ministral-3:3b",
    "qwen3:4b",
]

REQUEST_OPTIONS = {
    "temperature": 0,
    "num_ctx": 4096,
    "num_predict": 320,
    "num_gpu": 0,
}

EMBEDDING_OPTIONS = {
    "num_gpu": 0,
}
