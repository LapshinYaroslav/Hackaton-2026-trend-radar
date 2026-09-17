"""
Интерфейс Streamlit на мок-данных (docs/contracts/api_response.json).

По pipeline.md шаг 10: дашборд + страница инсайта.
Пока API нет — «Найти сигналы» читает тот же JSON.
Streamlit при каждом клике заново выполняет файл → состояние в st.session_state.
"""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "docs" / "contracts" / "api_response.json"

# Подписи типов документов из pipeline.md (для человека в UI)
SOURCE_TYPE_RU = {
    "paper": "статья",
    "preprint": "препринт",
    "patent": "патент",
    "news": "новость",
    "press_release": "пресс-релиз",
    "product": "продукт",
    "report": "отчёт",
    "standard": "стандарт",
    "blog": "блог",
}


def load_mock_response() -> dict:
    with CONTRACT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def init_state() -> None:
    if "insight_id" not in st.session_state:
        st.session_state.insight_id = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "error" not in st.session_state:
        st.session_state.error = None


def find_signal(signals: list, candidate_id: Optional[str]) -> Optional[dict]:
    if not candidate_id:
        return None
    for signal in signals:
        if signal.get("candidate_id") == candidate_id:
            return signal
    return None


def run_search(query: str) -> None:
    """
    Имитация долгого запроса (в pipeline риск: поиск №2 на каждого кандидата).
    Позже здесь будет HTTP к FastAPI; адрес API — из переменной окружения.
    """
    st.session_state.insight_id = None
    st.session_state.error = None
    query = (query or "").strip()
    if not query:
        st.session_state.error = "Введите направление поиска."
        st.session_state.result = None
        return

    # spinner = индикатор загрузки; без красного traceback при сбое
    with st.spinner(
        "Идёт поиск и оценка кандидатов. Это может занять несколько минут…"
    ):
        try:
            time.sleep(1.0)  # только для демо ощущения «идёт работа»
            data = load_mock_response()
            data = dict(data)
            data["query"] = query
            st.session_state.result = data
        except Exception as exc:  # noqa: BLE001 — показываем человеку, не traceback
            st.session_state.result = None
            st.session_state.error = (
                f"Не удалось получить результат. Попробуйте ещё раз. ({exc})"
            )


def render_source_card(source: dict) -> None:
    trust = (source.get("trust_level") or "").lower()
    low_trust = trust in {"low", "низкий"}
    raw_type = source.get("source_type") or "—"
    type_ru = SOURCE_TYPE_RU.get(raw_type, raw_type)

    title = source.get("title") or "Без названия"
    with st.container(border=True):
        if low_trust:
            st.warning("Источник с низким уровнем доверия")
        st.markdown(f"**{'⚠️ ' if low_trust else ''}{title}**")
        st.write(f"Ссылка: {source.get('url', '—')}")
        st.write(f"Дата публикации: {source.get('published_at', '—')}")
        st.write(f"Тип: {type_ru}")
        st.write(f"Язык оригинала: {source.get('language', '—')}")
        st.write(f"Уровень доверия: {source.get('trust_level', '—')}")
        summary = source.get("summary_ru")
        if summary:
            label = "Резюме"
            if source.get("summary_generated"):
                label += " (сгенерированное резюме)"
            st.write(f"{label}: {summary}")


def render_insight(signal: dict) -> None:
    if st.button("← Назад к списку"):
        st.session_state.insight_id = None
        st.rerun()

    insight = signal.get("insight") or {}
    st.title(signal.get("name_ru", "Инсайт"))
    st.write(f"**Уверенность модели (скоринг):** {signal.get('score', '—')}")

    st.subheader("Описание технологии")
    st.write(insight.get("description_ru") or "Нет описания")

    st.subheader("Преимущества")
    for item in insight.get("advantages_ru") or []:
        st.write(f"- {item}")

    st.subheader("Кейс-примеры")
    for item in insight.get("cases_ru") or []:
        st.write(f"- {item}")

    st.subheader("Оценки в аналитических отчётах")
    st.write(insight.get("analyst_notes_ru") or "—")

    st.subheader("Почему это слабый сигнал")
    st.write(insight.get("weak_signal_explanation_ru") or "—")

    st.subheader("Почему модель уверена")
    st.caption("Предикторы = вклады признаков (вес × стандартизованное значение), см. pipeline.md")
    for feature in signal.get("top_features") or []:
        name = feature.get("name")
        contrib = feature.get("contribution")
        text = feature.get("explanation_ru", "")
        if name is not None and contrib is not None:
            st.write(f"- **{name}** (вклад {contrib:+}): {text}")
        else:
            st.write(f"- {text}")

    st.subheader("Источники")
    for source in signal.get("sources") or []:
        render_source_card(source)


def render_signals_tab(signals: list) -> None:
    if not signals:
        st.info("Сигналов пока нет. Нажмите «Найти сигналы».")
        return

    for signal in signals:
        predictors = "; ".join(
            f.get("explanation_ru", "")
            for f in (signal.get("top_features") or [])
            if f.get("explanation_ru")
        )
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{signal.get('name_ru', '')}**")
            st.write(f"Скоринг: {signal.get('score', '—')}")
            st.write(f"Ключевые предикторы: {predictors or '—'}")
        with right:
            if st.button("Смотреть инсайт", key=f"insight_{signal.get('candidate_id')}"):
                st.session_state.insight_id = signal.get("candidate_id")
                st.rerun()
        st.divider()


def render_rejected_tab(rejected: list) -> None:
    if not rejected:
        st.info("Отклонённых кандидатов нет.")
        return
    for item in rejected:
        st.markdown(f"**{item.get('name_ru', 'Без названия')}**")
        st.write(f"Причина: {item.get('reason_ru', '—')}")
        st.divider()


st.set_page_config(page_title="Trend Radar", layout="wide")
init_state()

data = st.session_state.result
signals = (data or {}).get("signals") or []

selected = find_signal(signals, st.session_state.insight_id)
if selected is not None:
    render_insight(selected)
    st.stop()

st.title("Радар слабых сигналов")
st.caption("Демо на фейковых данных (пока без живого API). План: docs/pipeline.md")

st.subheader("Поисковый запрос")
default_q = (data or {}).get("query") or "технологии в медицине"
query = st.text_input("Направление поиска", value=default_q)

if st.button("Найти сигналы", type="primary"):
    run_search(query)
    st.rerun()

if st.session_state.error:
    st.error(st.session_state.error)

if data is None:
    st.info("Введите тему и нажмите «Найти сигналы», чтобы увидеть ТОП и отклонённые.")
    st.stop()

stats = data.get("stats") or {}
c1, c2, c3 = st.columns(3)
c1.metric("Обработано источников", stats.get("sources_processed", "—"))
c2.metric("Найдено кандидатов", stats.get("candidates_found", "—"))
c3.metric("Сигналы > 75%", stats.get("signals_above_75", "—"))

tab_top, tab_rejected = st.tabs(["ТОП-15", "Отклонённые"])
with tab_top:
    render_signals_tab(signals)
with tab_rejected:
    render_rejected_tab(data.get("rejected") or [])
