import os
from pathlib import Path

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "hr_documents.db"
DOCUMENTS_SEED_PATH = DATA_DIR / "hr_documents.json"
CASES_PATH = DATA_DIR / "test_cases.json"

RECOMMENDED_MODELS = [
    "gemma3:270m",
    "smollm2:135m",
    "smollm2:360m",
    "qwen3:0.6b",
    "gemma3:1b",
    "llama3.2:1b",
    "qwen3:1.7b",
    "smollm2:1.7b",
    "llama3.2:3b",
    "qwen3:4b",
]

REQUEST_OPTIONS = {
    "temperature": 0,
    "num_ctx": 4096,
    "num_predict": 320,
    "num_gpu": 0,
}
