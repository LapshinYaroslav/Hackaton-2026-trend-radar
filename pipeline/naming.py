"""След названия кандидата в научных источниках: работы и организации по фразе в окне обучения.

Название кандидата — термин шага 4 (name_en = term_en); здесь только проверка следа: OpenAlex, фраза,
type:article, 2020-09 … 2026-08, дешёвый режим group_by (meta.count тот же). Нулевой след — no_trace,
источник не ответил — trace_unknown. Нормализатор обучения (три варианта названия) удалён в задаче Л:
в продукте он не выбирался.

DEVIATIONS уходит в выход (normalizer_deviations) без изменений:
- company_stoplist_off: стоп-лист компаний в проверке названий выключен. В нём лежат термины самих
  сигналов (sovereign ai, physical intelligence), а при обучении он отклонил 3 варианта из 549.
"""
from __future__ import annotations

from collector.constants import COLLECTION_START, CUTOFF_DATE
from collector.exceptions import AdapterError
from collector.models import build_search_terms

DEVIATIONS = ["company_stoplist_off"]
MAX_WORKERS = 10


def trace(name: str, openalex) -> dict:
    """Работы и организации по фразе в окне обучения. None — источник не ответил."""
    try:
        answer = openalex.count_institutions(build_search_terms([name], []), COLLECTION_START, CUTOFF_DATE)
    except AdapterError:
        answer = None
    answer = answer or {}
    return {"name": name, "n_works": answer.get("works"), "n_institutions": answer.get("institutions"),
            "n_institutions_capped": answer.get("capped")}
