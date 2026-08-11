import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

DEFAULT_MODELS = [
    "qwen3:0.6b",
    "gemma3:1b",
    "qwen3:1.7b",
]

REQUEST_OPTIONS = {
    "temperature": 0,
    "num_ctx": 2048,
    "num_predict": 220,
    "num_gpu": 0,
}
