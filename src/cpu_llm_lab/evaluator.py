import re
from datetime import datetime

from .schemas import Evaluation, FormattedText, SourceRecord


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _amount_variants(value: float) -> set[str]:
    fixed = f"{value:.2f}"
    integer = str(int(value)) if value.is_integer() else ""
    variants = {
        fixed,
        fixed.replace(".", ","),
        f"r$ {fixed}",
        f"r${fixed}",
        f"r$ {fixed.replace('.', ',')}",
        f"r${fixed.replace('.', ',')}",
    }
    if integer:
        variants.add(integer)
        variants.add(f"r$ {integer}")
    return {_normalized(v) for v in variants if v}


def _date_variants(date_text: str) -> set[str]:
    variants = {date_text}
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(date_text, fmt)
            variants.update(
                {
                    parsed.strftime("%d/%m/%Y"),
                    parsed.strftime("%d-%m-%Y"),
                    parsed.strftime("%Y-%m-%d"),
                }
            )
            break
        except ValueError:
            continue
    return {_normalized(v) for v in variants}


def evaluate_output(
    source: SourceRecord,
    output: FormattedText,
    expected_category: str | None = None,
) -> Evaluation:
    combined = _normalized(f"{output.titulo} {output.mensagem} {output.categoria}")
    name_ok = _normalized(source.cliente) in combined
    amount_ok = any(v in combined for v in _amount_variants(source.valor))
    date_ok = any(v in combined for v in _date_variants(source.vencimento))
    category_ok = None if expected_category is None else output.categoria == expected_category

    checks = [name_ok, amount_ok, date_ok]
    if category_ok is not None:
        checks.append(category_ok)

    notes: list[str] = []
    if not name_ok:
        notes.append("Nome do cliente não foi preservado.")
    if not amount_ok:
        notes.append("Valor não foi preservado de forma reconhecível.")
    if not date_ok:
        notes.append("Data de vencimento não foi preservada.")
    if category_ok is False:
        notes.append("Categoria diferente da esperada.")

    return Evaluation(
        json_valid=True,
        schema_valid=True,
        name_preserved=name_ok,
        amount_preserved=amount_ok,
        due_date_preserved=date_ok,
        expected_category=category_ok,
        fidelity_score=round(sum(checks) / len(checks) * 100, 2),
        notes=notes,
    )
