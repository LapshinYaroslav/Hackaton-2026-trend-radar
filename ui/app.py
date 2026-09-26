"""
Дашборд по контракту docs/contracts/query_result.example.json.

Без Docker: если API_URL не задан, читаем файл примера локально.
Если API_URL задан — POST /queries и опрос GET, пока status != done.
Старый api_response.json больше не используем.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import time

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = ROOT / "docs" / "contracts" / "query_result.example.json"

AREAS = [
    "Edge",
    "Защита ИИ",
    "Индустриальный ИИ",
    "Инфраструктура ИИ",
    "Роботы",
    "Финтех",
    "Другое",
]
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

TRUST_RU = {"high": "высокий", "medium": "средний", "low": "низкий"}

VIEW_TOP = "ТОП"
VIEW_LIST = "Кандидаты"
VIEW_EXCLUDED = "Исключённые"
VIEW_CARD = "Карточка"


def load_example() -> dict:
    data = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    data.pop("_note", None)
    data["status"] = "done"
    return data


def api_base() -> str:
    return (os.getenv("API_URL") or "").strip().rstrip("/")


def fetch_history(limit: int = 20) -> list[dict]:
    base = api_base()
    if not base:
        return []
    try:
        response = requests.get(f"{base}/queries", params={"limit": limit}, timeout=10)
        response.raise_for_status()
        return list(response.json().get("items") or [])
    except requests.RequestException:
        return []


def fetch_balance() -> list[dict]:
    base = api_base()
    if not base:
        return []
    try:
        response = requests.get(f"{base}/catalog/balance", timeout=10)
        response.raise_for_status()
        return list(response.json().get("areas") or [])
    except requests.RequestException:
        return []


def open_history_item(query_id: str) -> None:
    response = requests.get(f"{api_base()}/queries/{query_id}", timeout=30)
    response.raise_for_status()
    data = response.json()
    status = (data.get("status") or "").lower()
    st.session_state.query_id = query_id
    st.session_state.error = None
    st.session_state.view = VIEW_TOP
    st.session_state.rank = None
    if status == "done":
        st.session_state.result = data
        st.session_state.polling = False
    elif status == "error":
        st.session_state.result = None
        st.session_state.error = data.get("error") or "Ошибка расчёта"
        st.session_state.polling = False
    else:
        st.session_state.result = None
        st.session_state.polling = True
        st.session_state.progress_stage = data.get("progress_stage") or ""
        st.session_state.progress_done = int(data.get("progress_done") or 0)
        st.session_state.progress_total = int(data.get("progress_total") or 6)


def init_state() -> None:
    defaults = {
        "result": None,
        "error": None,
        "view": VIEW_TOP,
        "rank": None,
        "query_id": None,
        "progress_stage": "",
        "progress_done": 0,
        "progress_total": 6,
        "polling": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_local(topic: str, area: str | None) -> None:
    data = load_example()
    data["topic"] = topic
    data["area"] = area
    st.session_state.result = data
    st.session_state.query_id = data.get("query_id")
    st.session_state.polling = False
    st.session_state.error = None
    st.session_state.view = VIEW_TOP
    st.session_state.rank = None


def start_remote(topic: str, area: str | None) -> None:
    response = requests.post(
        api_base() + "/queries",
        json={"topic": topic, "area": area},
        timeout=30,
    )
    response.raise_for_status()
    st.session_state.query_id = response.json()["query_id"]
    st.session_state.result = None
    st.session_state.polling = True
    st.session_state.error = None
    st.session_state.view = VIEW_TOP
    st.session_state.rank = None


def poll_remote() -> None:
    query_id = st.session_state.query_id
    response = requests.get(f"{api_base()}/queries/{query_id}", timeout=30)
    response.raise_for_status()
    data = response.json()
    st.session_state.progress_stage = data.get("progress_stage") or ""
    st.session_state.progress_done = int(data.get("progress_done") or 0)
    st.session_state.progress_total = int(data.get("progress_total") or 6)
    status = (data.get("status") or "").lower()
    if status == "done":
        st.session_state.result = data
        st.session_state.polling = False
    elif status == "error":
        st.session_state.error = data.get("error") or "Ошибка расчёта"
        st.session_state.polling = False


def render_sources(sources: list) -> None:
    for source in sources:
        trust = (source.get("trust_level") or "").lower()
        low = trust == "low"
        title = source.get("title") or "Без названия"
        with st.container(border=True):
            if low:
                st.warning("Источник с низким уровнем доверия")
            st.markdown(f"**{'⚠️ ' if low else ''}{title}**")
            url = source.get("url")
            if url:
                st.markdown(f"Ссылка: [{url}]({url})")
            st.write(f"Дата: {source.get('published_at', '—')}")
            raw_type = source.get("source_type") or "—"
            st.write(f"Тип: {SOURCE_TYPE_RU.get(raw_type, raw_type)}")
            st.write(f"Язык оригинала: {source.get('language', '—')}")
            st.write(f"Доверие: {TRUST_RU.get(trust, trust or '—')}")


def render_card(item: dict) -> None:
    if st.button("← Назад к ТОП"):
        st.session_state.view = VIEW_TOP
        st.session_state.rank = None
        st.rerun()
    st.title(item.get("name_ru") or "Карточка")
    st.write(f"**Английское название:** {item.get('name_en', '—')}")
    score = item.get("score")
    st.write(f"**Уверенность модели:** {score if score is not None else '—'}")
    st.subheader("Почему это слабый сигнал")
    for line in item.get("explanation_ru") or []:
        st.write(f"- {line}")
    st.caption(
        "Полный отчёт (описание, преимущество, кейс) появится после генерации инсайта. "
        "Числа модели уже показаны выше и не пересказываются языковой моделью."
    )
    st.subheader("Источники")
    render_sources(item.get("sources") or [])


def render_top(items: list) -> None:
    if not items:
        st.info("В ТОП пока нет технологий выше порога.")
        return
    st.caption(f"Показано {len(items)}. Если меньше 15 — столько прошло порог модели.")
    for item in items:
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{item.get('rank', '—')}. {item.get('name_ru', '')}**")
            st.write(item.get("name_en") or "")
            score = item.get("score")
            score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
            st.write(f"Скоринг: {score_txt}")
            preds = "; ".join(item.get("explanation_ru") or [])
            st.write(f"Ключевые предикторы: {preds or '—'}")
        with right:
            rank = item.get("rank")
            if st.button("Смотреть", key=f"card_{rank}"):
                st.session_state.rank = rank
                st.session_state.view = VIEW_CARD
                st.rerun()
        st.divider()


def render_excluded(items: list) -> None:
    if not items:
        st.info("Исключённых кандидатов нет.")
        return
    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item.get('name_ru', 'Без названия')}**")
            st.write(item.get("name_en") or "")
            score = item.get("score")
            if score is None:
                st.write("Скоринг: не оценён")
            else:
                st.write(f"Скоринг: {score:.2f}")
            st.write(f"Причина исключения: {item.get('reason_ru', '—')}")


def render_candidates(data: dict) -> None:
    top = data.get("top") or []
    excluded = data.get("excluded") or []
    st.caption(
        "В контракте пока нет полного списка из 64 имён. "
        "Показываем тех, кто попал в ТОП, и тех, кого исключили."
    )
    rows = []
    for item in top:
        rows.append(
            {
                "Где": "ТОП",
                "Название": item.get("name_ru"),
                "English": item.get("name_en"),
                "Скоринг": item.get("score"),
            }
        )
    for item in excluded:
        rows.append(
            {
                "Где": "исключён",
                "Название": item.get("name_ru"),
                "English": item.get("name_en"),
                "Скоринг": item.get("score"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_stats(data: dict) -> None:
    stats = data.get("stats") or {}
    above = stats.get("above_075")
    if above is None:
        above = sum(
            1
            for item in (data.get("top") or [])
            if isinstance(item.get("score"), (int, float)) and item["score"] >= SCORE_HIGH
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("Обработано источников", stats.get("documents_total", "—"))
    c2.metric("Найдено кандидатов", stats.get("candidates_found", "—"))
    c3.metric("Сигналы > 75%", above)
    if st.button("Перейти к списку кандидатов"):
        st.session_state.view = VIEW_LIST
        st.rerun()


st.set_page_config(page_title="Радар слабых сигналов", layout="wide")
init_state()

if st.session_state.polling and st.session_state.query_id and api_base():
    try:
        poll_remote()
    except requests.RequestException as exc:
        st.session_state.error = f"Не удалось опросить API. ({exc})"
        st.session_state.polling = False

data = st.session_state.result
if (
    data
    and st.session_state.view == VIEW_CARD
    and st.session_state.rank is not None
):
    chosen = next(
        (item for item in (data.get("top") or []) if item.get("rank") == st.session_state.rank),
        None,
    )
    if chosen:
        render_card(chosen)
        st.stop()

st.title("Радар слабых сигналов")
mode = f"API: {api_base()}" if api_base() else "локальный пример контракта (без Docker)"
st.caption(mode)

if api_base():
    with st.sidebar:
        st.subheader("История запросов")
        history = fetch_history()
        if not history:
            st.caption("Пока пусто — выполните поиск.")
        for item in history:
            label = item.get("topic") or item.get("query_id")
            area = item.get("area") or "без области"
            status = item.get("status") or ""
            caption = f"{area} · {status}"
            if st.button(f"{label}", key=f"hist_{item['query_id']}", help=caption):
                try:
                    open_history_item(item["query_id"])
                except requests.RequestException as exc:
                    st.session_state.error = f"Не удалось открыть запрос. ({exc})"
                st.rerun()
        balance = fetch_balance()
        if balance:
            st.subheader("Обучающая выборка")
            st.caption("100 сигналов + 60 негативов")
            st.dataframe(balance, use_container_width=True, hide_index=True)

st.subheader("Поисковый запрос")
topic = st.text_input(
    "Технологическое направление",
    value=(data or {}).get("topic") or "роботы для промышленности",
)
area_label = st.selectbox("Область", AREAS, index=AREAS.index("Роботы"))
area = None if area_label == "Другое" else area_label

if st.button("Найти сигналы", type="primary"):
    st.session_state.error = None
    if not topic.strip():
        st.session_state.error = "Введите тему."
    elif api_base():
        try:
            start_remote(topic.strip(), area)
        except requests.RequestException as exc:
            st.session_state.error = (
                "API недоступен. Запустите его локально или уберите API_URL. "
                f"({exc})"
            )
    else:
        with st.spinner("Считаем пример контракта…"):
            time.sleep(0.4)
        start_local(topic.strip(), area)
    st.rerun()

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.polling:
    total = max(int(st.session_state.progress_total or 1), 1)
    done = int(st.session_state.progress_done or 0)
    st.progress(min(done / total, 1.0))
    st.info(
        f"Стадия: {st.session_state.progress_stage or 'запуск'} "
        f"({done}/{total}). Расчёт может занять несколько минут."
    )
    time.sleep(0.6)
    st.rerun()

if data is None:
    st.info("Введите тему, выберите область и нажмите «Найти сигналы».")
    st.stop()

st.success(f"Запрос: «{data.get('topic', '')}» · область: {data.get('area') or 'не задана'}")
render_stats(data)

views = [VIEW_TOP, VIEW_LIST, VIEW_EXCLUDED]
current = st.session_state.view if st.session_state.view in views else VIEW_TOP
chosen_view = st.radio("Разделы", views, index=views.index(current), horizontal=True)
if chosen_view != st.session_state.view:
    st.session_state.view = chosen_view
    st.rerun()

if st.session_state.view == VIEW_LIST:
    render_candidates(data)
elif st.session_state.view == VIEW_EXCLUDED:
    render_excluded(data.get("excluded") or [])
else:
    render_top(data.get("top") or [])
