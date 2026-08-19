import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from .schemas import Document, RetrievedDocument

STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos",
    "e", "em", "eu", "me", "meu", "minha", "na", "nas", "no", "nos", "o",
    "os", "ou", "para", "por", "que", "se", "um", "uma", "tem", "tenho",
    "qual", "quais", "quanto", "quando", "onde", "posso", "sobre",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _normalize(text).split()
        if len(token) > 2 and token not in STOPWORDS
    }


def init_database(db_path: Path, seed_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    documents = json.loads(seed_path.read_text(encoding="utf-8"))

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO documents (id, title, category, content)
            VALUES (:id, :title, :category, :content)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                category = excluded.category,
                content = excluded.content
            """,
            documents,
        )
        connection.commit()


def list_documents(db_path: Path) -> list[Document]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, title, category, content FROM documents ORDER BY id"
        ).fetchall()
    return [Document.model_validate(dict(row)) for row in rows]


def _lexical_score(query_tokens: set[str], document: Document) -> int:
    title_tokens = _tokens(document.title)
    category_tokens = _tokens(document.category)
    content_tokens = _tokens(document.content)
    return (
        len(query_tokens & title_tokens) * 5
        + len(query_tokens & category_tokens) * 3
        + len(query_tokens & content_tokens)
    )


def rank_documents_lexical(db_path: Path, query: str) -> list[RetrievedDocument]:
    query_tokens = _tokens(query)
    scored: list[RetrievedDocument] = []

    for document in list_documents(db_path):
        score = _lexical_score(query_tokens, document)
        if score > 0:
            scored.append(
                RetrievedDocument(**document.model_dump(), score=float(score))
            )

    scored.sort(key=lambda item: (-item.score, item.id))
    return scored


def search_documents(
    db_path: Path,
    query: str,
    limit: int = 3,
) -> list[RetrievedDocument]:
    return rank_documents_lexical(db_path, query)[:limit]
