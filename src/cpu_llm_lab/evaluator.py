import re
import unicodedata

from .observability import traced
from .schemas import GroundedAnswer, QueryEvaluation, RetrievedDocument, TestCase


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def evaluate_query(
    case: TestCase,
    answer: GroundedAnswer,
    retrieved_documents: list[RetrievedDocument],
) -> QueryEvaluation:
    with traced("evaluation.query", case_id=case.id):
        normalized_answer = _normalized(answer.resposta)
        retrieved_ids = {document.id for document in retrieved_documents}
        expected_ids = set(case.expected_document_ids)

        retrieval_hit = (
            expected_ids.issubset(retrieved_ids) if expected_ids else True
        )
        preserved = sum(
            1
            for fact in case.required_facts
            if _normalized(fact) in normalized_answer
        )
        total = len(case.required_facts)
        factual_score = (
            round((preserved / total) * 100, 2) if total else 100.0
        )

        cited_ids = set(answer.fontes)
        source_accuracy = (
            bool(expected_ids & cited_ids)
            if case.should_answer and expected_ids
            else None
        )
        forbidden_found = [
            fact
            for fact in case.forbidden_facts
            if _normalized(fact) in normalized_answer
        ]
        hallucination_free = not forbidden_found
        abstention_ok = answer.encontrado == case.should_answer

        notes: list[str] = []
        if not retrieval_hit:
            notes.append(
                "O recuperador não trouxe todos os documentos esperados."
            )
        if preserved < total:
            notes.append(
                f"Preservou {preserved}/{total} fatos obrigatórios."
            )
        if source_accuracy is False:
            notes.append(
                "A resposta não citou uma das fontes esperadas."
            )
        if forbidden_found:
            notes.append(
                "Incluiu informação proibida/alterada: "
                + ", ".join(forbidden_found)
            )
        if not abstention_ok:
            notes.append(
                "Errou ao decidir se havia informação suficiente para responder."
            )

        return QueryEvaluation(
            retrieval_hit=retrieval_hit,
            required_facts_total=total,
            required_facts_preserved=preserved,
            factual_score=factual_score,
            source_accuracy=source_accuracy,
            hallucination_free=hallucination_free,
            abstention_ok=abstention_ok,
            notes=notes,
        )
