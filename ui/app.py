"""
Интерфейс Streamlit под финальные требования ТЗ (Газпромбанк.Тех).

Данные:
1) API_URL → GET /search (в Docker: http://api:8000);
2) иначе мок docs/contracts/api_response.json.

Streamlit перезапускает скрипт при каждом клике → состояние в st.session_state.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import time
from typing import Optional

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "docs" / "contracts" / "api_response.json"
TOP_N = 15
SCORE_HIGH = 0.75

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

TAB_TOP = "ТОП-15"
TAB_CANDIDATES = "Кандидаты"
TAB_REJECTED = "Отклонённые"


def load_mock_response() -> dict:
    with CONTRACT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def fetch_from_api(query: str, api_url: str) -> dict:
    url = api_url.rstrip("/") + "/search"
    response = requests.get(url, params={"q": query}, timeout=120)
    response.raise_for_status()
    return response.json()


def init_state() -> None:
    defaults = {
        "insight_id": None,
        "result": None,
        "error": None,
        "active_tab": TAB_TOP,
        "searching": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def find_signal(signals: list, candidate_id: Optional[str]) -> Optional[dict]:
    if not candidate_id:
        return None
    for signal in signals:
        if signal.get("candidate_id") == candidate_id:
            return signal
    return None


def top_signals(signals: list) -> list:
    """Сортировка по score убыв., не больше 15 — как требует ТЗ."""
    ranked = sorted(
        signals,
        key=lambda s: float(s.get("score") or 0.0),
        reverse=True,
    )
    return ranked[:TOP_N]


def run_search(query: str) -> None:
    st.session_state.insight_id = None
    st.session_state.error = None
    st.session_state.active_tab = TAB_TOP
    query = (query or "").strip()
    if not query:
        st.session_state.error = "Введите направление поиска."
        st.session_state.result = None
        return

    api_url = (os.getenv("API_URL") or "").strip()

    with st.spinner(
        "Идёт поиск по открытым источникам и оценка кандидатов. "
        "Это может занять несколько минут…"
    ):
        try:
            if api_url:
                data = fetch_from_api(query, api_url)
            else:
                time.sleep(0.4)
                data = dict(load_mock_response())
                data["query"] = query

            status = (data.get("status") or "done").lower()
            if status == "error":
                st.session_state.result = data
                st.session_state.error = data.get("error_ru") or (
                    "Поиск завершился с ошибкой. Попробуйте другой запрос."
                )
                return
            if status == "running":
                st.session_state.result = data
                st.session_state.error = None
                return

            st.session_state.result = data
        except requests.RequestException as exc:
            st.session_state.result = None
            st.session_state.error = (
                "Не удалось связаться с API. Проверьте, что сервис api запущен "
                f"и API_URL верный. ({exc})"
            )
        except Exception as exc:  # noqa: BLE001
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
        url = source.get("url")
        if url:
            st.markdown(f"Ссылка: [{url}]({url})")
        else:
            st.write("Ссылка: —")
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
    score = signal.get("score")
    st.title(signal.get("name_ru", "Инсайт"))
    st.write(f"**Уверенность модели (скоринг):** {score if score is not None else '—'}")

    st.subheader("Описание технологии")
    st.write(insight.get("description_ru") or "Нет описания")

    st.subheader("Преимущества")
    advantages = insight.get("advantages_ru") or []
    if advantages:
        for item in advantages:
            st.write(f"- {item}")
    else:
        st.write("—")

    st.subheader("Кейс-примеры")
    cases = insight.get("cases_ru") or []
    if cases:
        for item in cases:
            st.write(f"- {item}")
    else:
        st.write("—")

    st.subheader("Оценки в аналитических отчётах")
    st.write(insight.get("analyst_notes_ru") or "—")

    st.subheader("Почему это слабый сигнал (зарождающийся тренд)")
    st.write(insight.get("weak_signal_explanation_ru") or "—")

    st.subheader("Почему модель присвоила такую уверенность")
    st.caption(
        "Ключевые предикторы = вклады признаков модели "
        "(вес × стандартизованное значение)."
    )
    features = signal.get("top_features") or []
    if not features:
        st.write("—")
    for feature in features:
        name = feature.get("name")
        contrib = feature.get("contribution")
        text = feature.get("explanation_ru", "")
        if name is not None and contrib is not None:
            st.write(f"- **{name}** (вклад {contrib:+}): {text}")
        else:
            st.write(f"- {text}")

    st.subheader("Источники")
    sources = signal.get("sources") or []
    if not sources:
        st.write("Источники не переданы.")
    for source in sources:
        render_source_card(source)


def render_signals_tab(signals: list) -> None:
    ranked = top_signals(signals)
    if not ranked:
        st.info("В выдаче пока нет слабых сигналов. Нажмите «Найти сигналы».")
        return

    st.caption(
        f"Показано {len(ranked)} из максимум {TOP_N} (сортировка по уверенности модели)."
    )
    for index, signal in enumerate(ranked, start=1):
        predictors = "; ".join(
            f.get("explanation_ru", "")
            for f in (signal.get("top_features") or [])
            if f.get("explanation_ru")
        )
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{index}. {signal.get('name_ru', '')}**")
            score = signal.get("score")
            score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
            st.write(f"Скоринг (уверенность модели): {score_txt}")
            st.write(f"Ключевые предикторы: {predictors or '—'}")
        with right:
            cid = signal.get("candidate_id") or f"row_{index}"
            if st.button("Смотреть инсайт", key=f"insight_{cid}"):
                st.session_state.insight_id = signal.get("candidate_id")
                st.rerun()
        st.divider()


def render_candidates_tab(candidates: list, stats: dict) -> None:
    st.caption(
        "Плюс по ТЗ: список технологий-кандидатов на слабый сигнал "
        "(до фильтра мейнстрима / порога)."
    )
    if not candidates:
        # Фоллбек: если API ещё не отдаёт candidates — хотя бы число из stats
        found = stats.get("candidates_found")
        st.info(
            "Список кандидатов не передан API. "
            f"В статистике указано найдено: {found if found is not None else '—'}."
        )
        return

    rows = [
        {
            "ID": c.get("candidate_id", "—"),
            "Технология": c.get("name_ru", "—"),
            "Статус": c.get("stage_ru", "—"),
        }
        for c in candidates
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_rejected_tab(rejected: list) -> None:
    st.caption(
        "Логика исключения зрелых трендов, стандартов, хайпа и шума — требование ТЗ."
    )
    if not rejected:
        st.info("Отклонённых кандидатов нет.")
        return
    for item in rejected:
        with st.container(border=True):
            st.markdown(f"**{item.get('name_ru', 'Без названия')}**")
            st.write(f"Причина исключения: {item.get('reason_ru', '—')}")


def render_stats(data: dict) -> None:
    stats = data.get("stats") or {}
    signals = data.get("signals") or []
    candidates = data.get("candidates") or []

    sources = stats.get("sources_processed", "—")
    found = stats.get("candidates_found")
    if found is None:
        found = len(candidates) if candidates else "—"
    above = stats.get("signals_above_75")
    if above is None:
        above = sum(
            1
            for s in signals
            if isinstance(s.get("score"), (int, float)) and s["score"] >= SCORE_HIGH
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Обработано источников", sources)
    c2.metric("Найдено кандидатов", found)
    c3.metric("Сигналы > 75%", above)

    # Переход к списку кандидатов (плюс ТЗ)
    if st.button("Перейти к списку кандидатов", key="goto_candidates"):
        st.session_state.active_tab = TAB_CANDIDATES
        st.rerun()


st.set_page_config(page_title="Радар слабых сигналов", layout="wide")
init_state()

data = st.session_state.result
signals = (data or {}).get("signals") or []

selected = find_signal(signals, st.session_state.insight_id)
if selected is not None:
    render_insight(selected)
    st.stop()

st.title("Радар слабых сигналов")
api_configured = bool((os.getenv("API_URL") or "").strip())
mode = (
    "API: " + os.getenv("API_URL", "")
    if api_configured
    else "мок-файл (API_URL не задан)"
)
st.caption(
    f"Режим данных: {mode}. Интерфейс на русском · Streamlit (допускается ТЗ)."
)

st.subheader("Поисковый запрос")
default_q = (data or {}).get("query") or "технологии в медицине"
query = st.text_input(
    "Технологическое направление (свободная форма)",
    value=default_q,
    placeholder="например: технологии в ИИ",
)

if st.button("Найти сигналы", type="primary"):
    run_search(query)
    st.rerun()

if st.session_state.error:
    st.error(st.session_state.error)

if data is None:
    st.info(
        "Введите направление и нажмите «Найти сигналы». "
        "Система покажет ТОП-15, кандидатов и причины отклонения."
    )
    st.stop()

status = (data.get("status") or "done").lower()
if status == "running":
    st.warning("Запрос ещё обрабатывается. Обновите страницу или повторите поиск позже.")
elif status == "done":
    st.success(f"Запрос выполнен: «{data.get('query', '')}»")

render_stats(data)

# st.tabs не умеет выбрать вкладку программно → radio как переключатель разделов
tab_labels = [TAB_TOP, TAB_CANDIDATES, TAB_REJECTED]
current = st.session_state.active_tab
if current not in tab_labels:
    current = TAB_TOP
chosen = st.radio(
    "Разделы выдачи",
    tab_labels,
    index=tab_labels.index(current),
    horizontal=True,
    label_visibility="collapsed",
)
if chosen != st.session_state.active_tab:
    st.session_state.active_tab = chosen
    st.rerun()

if st.session_state.active_tab == TAB_TOP:
    render_signals_tab(signals)
elif st.session_state.active_tab == TAB_CANDIDATES:
    render_candidates_tab(data.get("candidates") or [], data.get("stats") or {})
else:
    render_rejected_tab(data.get("rejected") or [])
