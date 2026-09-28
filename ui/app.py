"""
Дашборд по контракту docs/contracts/query_result.example.json.

Без Docker: если API_URL не задан, читаем файл примера локально.
Если API_URL задан — POST /queries и опрос GET, пока status != done.
Старый api_response.json больше не используем.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import base64
import html
import json
import os
import sys
import time

import requests
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
# streamlit run ui/app.py не кладёт корень репозитория в путь импорта
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLE_PATH = ROOT / "docs" / "contracts" / "query_result.example.json"

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
    """Адрес API. Внутри Docker это http://api:8000. На Mac такого имени нет."""
    url = (os.getenv("API_URL") or "").strip().rstrip("/")
    if not url:
        return ""
    host = url.split("://", 1)[-1].split("/")[0].split(":")[0]
    if host == "api" and not Path("/.dockerenv").exists():
        return ""
    return url


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
        "owned": [],
        "history_open": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_local(topic: str) -> None:
    from search.area import detect_area

    data = load_example()
    data["topic"] = topic
    data["area"] = detect_area(topic, use_llm=False)
    st.session_state.result = data
    st.session_state.query_id = data.get("query_id")
    st.session_state.polling = False
    st.session_state.error = None
    st.session_state.view = VIEW_TOP
    st.session_state.rank = None
    st.session_state.history_open = False
    st.session_state.topic_text = topic
    _remember_local(data)


def start_remote(topic: str) -> None:
    response = requests.post(
        api_base() + "/queries",
        json={"topic": topic},
        timeout=30,
    )
    response.raise_for_status()
    st.session_state.query_id = response.json()["query_id"]
    st.session_state.result = None
    st.session_state.polling = True
    st.session_state.error = None
    st.session_state.view = VIEW_TOP
    st.session_state.rank = None
    owned = list(st.session_state.owned)
    if st.session_state.query_id not in owned:
        owned.insert(0, st.session_state.query_id)
    st.session_state.owned = owned
    st.session_state.nav_token = st.session_state.query_id
    st.session_state.history_open = False
    st.session_state.topic_text = topic
    st.query_params["open"] = st.session_state.query_id


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


STATUS_RU = {
    "done": "готово",
    "running": "считается",
    "queued": "в очереди",
    "error": "ошибка",
}


def _when(value: str | None) -> str:
    if not value:
        return ""
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return ""
    if moment.tzinfo is None:
        return moment.strftime("%d.%m.%Y, %H:%M")
    return moment.astimezone().strftime("%d.%m.%Y, %H:%M")


def _remember_local(data: dict) -> None:
    saved = st.session_state.setdefault("local_saved", {})
    query_id = data.get("query_id")
    if not query_id:
        return
    saved[query_id] = data
    order = st.session_state.setdefault("local_order", [])
    if query_id in order:
        order.remove(query_id)
    order.insert(0, query_id)


def load_history() -> list[dict]:
    if api_base():
        try:
            response = requests.get(f"{api_base()}/queries", params={"limit": 30}, timeout=5)
            response.raise_for_status()
            return list(response.json().get("items") or [])
        except requests.RequestException:
            return []
    order = st.session_state.get("local_order") or []
    saved = st.session_state.get("local_saved") or {}
    items = []
    for query_id in order:
        data = saved.get(query_id) or {}
        items.append(
            {
                "query_id": query_id,
                "topic": data.get("topic") or query_id,
                "area": data.get("area"),
                "status": "done",
                "created_at": None,
                "candidates_found": (data.get("stats") or {}).get("candidates_found"),
                "above_075": (data.get("stats") or {}).get("above_075"),
            }
        )
    return items


def open_saved(query_id: str) -> None:
    st.session_state.insight_reports = {}
    st.session_state.rank = None
    st.session_state.view = VIEW_TOP
    if not api_base():
        data = (st.session_state.get("local_saved") or {}).get(query_id)
        if not data:
            st.session_state.error = "Этот запрос не сохранился в сессии."
            return
        st.session_state.result = data
        st.session_state.query_id = query_id
        st.session_state.polling = False
        st.session_state.error = None
        return
    response = requests.get(f"{api_base()}/queries/{query_id}", timeout=30)
    response.raise_for_status()
    data = response.json()
    status = (data.get("status") or "").lower()
    st.session_state.query_id = query_id
    st.session_state.progress_stage = data.get("progress_stage") or ""
    st.session_state.progress_done = int(data.get("progress_done") or 0)
    st.session_state.progress_total = int(data.get("progress_total") or 6)
    if status == "done":
        st.session_state.result = data
        st.session_state.polling = False
        st.session_state.error = None
        return
    st.session_state.result = None
    st.session_state.polling = status == "running" and query_id in (st.session_state.owned or [])
    if status == "error":
        st.session_state.error = data.get("error") or "Ошибка расчёта"
    elif st.session_state.polling:
        st.session_state.error = None
    else:
        st.session_state.error = "Этот запрос не был завершён. Запустите тему ещё раз."


def reset_query() -> None:
    st.session_state.result = None
    st.session_state.query_id = None
    st.session_state.polling = False
    st.session_state.error = None
    st.session_state.view = VIEW_TOP
    st.session_state.rank = None
    st.session_state.insight_reports = {}
    st.session_state.topic_text = ""


def consume_nav() -> None:
    """Клик по строке истории приходит как ?open=id и открывает сохранённый запрос."""
    token = st.query_params.get("open")
    if not token or token == st.session_state.get("nav_token"):
        return
    st.session_state.nav_token = token
    st.session_state.history_open = False
    st.session_state.collapse_sidebar = True
    if token == "new":
        reset_query()
        return
    try:
        open_saved(str(token))
    except requests.RequestException as exc:
        st.session_state.error = f"Не удалось открыть запрос. ({exc})"


def _history_row(item: dict, current: str | None) -> str:
    query_id = str(item.get("query_id") or "")
    topic = html.escape((item.get("topic") or "без темы").strip())
    status = (item.get("status") or "").lower()
    when = html.escape(_when(item.get("created_at")))
    active = " active" if query_id and query_id == current else ""
    dot = "run" if status == "running" else "err" if status == "error" else "done"
    hint = STATUS_RU.get(status, status)
    if when:
        hint = f"{hint}, {when}" if hint else when
    return (
        f"<a class='hist-item{active}' href='?open={html.escape(query_id)}' title='{html.escape(hint)}'>"
        f"<span class='dot {dot}'></span>"
        "<span class='body'>"
        f"<span class='title'>{topic}</span>"
        f"<span class='when'>{when}</span>"
        "</span>"
        "</a>"
    )


def render_history() -> None:
    items = load_history()
    current = st.session_state.query_id
    rows = "".join(_history_row(item, current) for item in items)
    empty = "" if rows else "<p class='hist-empty'>Пока пусто. Первый поиск появится здесь.</p>"
    with st.sidebar:
        st.markdown(
            "<nav class='side-nav'>"
            "<a class='side-action' href='?open=new'>"
            "<svg viewBox='0 0 16 16' aria-hidden='true'>"
            "<path d='M8 3.2v9.6M3.2 8h9.6' fill='none' stroke='currentColor' "
            "stroke-width='1.4' stroke-linecap='round'/></svg>"
            "<span>Новый запрос</span>"
            "</a>"
            "<div class='side-label'>История</div>"
            f"{rows}{empty}"
            "</nav>",
            unsafe_allow_html=True,
        )


def keep_sidebar_closed() -> None:
    """Стрелка остаётся. После выбора запроса в истории панель сразу скрывается."""
    collapse_now = "true" if st.session_state.pop("collapse_sidebar", False) else "false"
    components.html(
        f"""
        <script>
        (() => {{
          const parent = window.parent;
          const doc = parent.document;
          if ({collapse_now}) {{
            try {{ parent.sessionStorage.setItem("horizonShut", "1"); }} catch (e) {{}}
          }}
          let shut = false;
          try {{ shut = parent.sessionStorage.getItem("horizonShut") === "1"; }} catch (e) {{}}
          if (shut) doc.documentElement.setAttribute("data-horizon-shut", "1");
          if (parent.__horizonSidebarWatch) return;
          parent.__horizonSidebarWatch = true;
          doc.addEventListener("click", (event) => {{
            if (!doc.documentElement.hasAttribute("data-horizon-shut")) return;
            const target = event.target;
            if (!target || !target.closest) return;
            const opener = target.closest(
              '[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"]'
            );
            if (!opener) return;
            doc.documentElement.removeAttribute("data-horizon-shut");
            try {{ parent.sessionStorage.removeItem("horizonShut"); }} catch (e) {{}}
            event.preventDefault();
            event.stopPropagation();
          }}, true);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def score_bar(score) -> str:
    if not isinstance(score, (int, float)):
        return "<span>скоринг не посчитан</span>"
    step = max(0, min(10, int(round(score * 10))))
    hot = " hot" if score >= SCORE_HIGH else ""
    return (
        f"<div class='score{hot}'><i class='w{step}'></i></div>"
        f"<span>уверенность {score:.0%}</span>"
    )


def trust_chip(level: str) -> str:
    key = (level or "").lower()
    label = TRUST_RU.get(key, key or "—")
    css = key if key in {"high", "medium", "low"} else "medium"
    return f"<span class='chip {css}'>{label}</span>"


def render_sources(sources: list) -> None:
    for source in sources:
        trust = (source.get("trust_level") or "").lower()
        low = trust == "low"
        title = source.get("title") or "Без названия"
        with st.container(border=True):
            if low:
                st.warning("Источник с низким уровнем доверия")
            st.markdown(f"**{'⚠️ ' if low else ''}{title}**")
            url = source.get("url") or ""
            placeholder = "example.org" in url or "example.com" in url
            if placeholder:
                st.warning(
                    "Это учебная ссылка из файла-примера, не настоящая статья. "
                    "По ней нет текста: оркестратор ещё не подставил реальные документы."
                )
            elif url:
                st.markdown(f"Ссылка: [{url}]({url})")
            st.write(f"Дата: {source.get('published_at', '—')}")
            raw_type = source.get("source_type") or "—"
            st.write(f"Тип: {SOURCE_TYPE_RU.get(raw_type, raw_type)}")
            st.write(f"Язык оригинала: {source.get('language', '—')}")
            st.markdown(trust_chip(trust), unsafe_allow_html=True)
            if source.get("trust_reason"):
                st.caption(source["trust_reason"])
            if source.get("summary_ru"):
                note = source.get("summary_note") or "сгенерированное резюме"
                st.write(f"Резюме ({note}): {source['summary_ru']}")


def _show_report(content: dict) -> None:
    st.subheader("Описание технологии")
    block = content.get("description") or {}
    st.write(block.get("text") or "—")
    st.subheader("Преимущество")
    st.write((content.get("advantage") or {}).get("text") or "—")
    if content.get("low_trust_warning"):
        st.warning(content["low_trust_warning"])
    st.subheader("Источники")
    render_sources(content.get("sources") or [])


def _local_report(item: dict) -> None:
    from search.insights import generate_insight

    rank = item.get("rank")
    store = st.session_state.setdefault("insight_reports", {})
    if rank not in store:
        with st.spinner("Готовим отчёт по найденным документам…"):
            try:
                store[rank] = {"status": "ready", "content": generate_insight(item)}
            except Exception as exc:  # noqa: BLE001
                store[rank] = {"status": "error", "error": str(exc)}
    report = store[rank]
    if report.get("status") == "ready":
        _show_report(report["content"])
        return
    st.error(
        "Отчёт не собрался. Для живого текста в .env нужны "
        "YANDEX_FOLDER_ID, YANDEX_API_KEY и модель yandexgpt-5-pro. "
        f"({report.get('error')})"
    )
    st.subheader("Источники")
    render_sources(item.get("sources") or [])


def _remote_report(item: dict) -> None:
    query_id = st.session_state.query_id
    rank = item.get("rank")
    try:
        response = requests.get(
            f"{api_base()}/queries/{query_id}/insights/{rank}",
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        st.error(f"Не удалось запросить инсайт. ({exc})")
        return
    status = payload.get("status")
    if status == "ready" and payload.get("content"):
        _show_report(payload["content"])
        return
    if status == "error":
        st.error(payload.get("error") or "Ошибка генерации отчёта")
        return
    st.info("Готовим отчёт…")
    time.sleep(2)
    st.rerun()


def render_card(item: dict) -> None:
    if st.button("← Назад к ТОП"):
        st.session_state.view = VIEW_TOP
        st.session_state.rank = None
        st.rerun()
    st.title(item.get("name_ru") or "Инсайт")
    st.write(f"**Английское название:** {item.get('name_en', '—')}")
    st.markdown(score_bar(item.get("score")), unsafe_allow_html=True)
    if api_base():
        _remote_report(item)
    else:
        _local_report(item)


def _technology_description(item: dict) -> str:
    """Короткое описание технологии по документам, без долей и вкладов модели."""
    explicit = str(item.get("description_ru") or "").strip()
    if explicit:
        return explicit
    titles = []
    for source in item.get("sources") or []:
        title = str(source.get("title") or "").strip()
        summary = str(source.get("summary_ru") or "").strip()
        line = summary or title
        if line and line not in titles:
            titles.append(line)
        if len(titles) == 2:
            break
    if not titles:
        return "Описание по найденным документам появится в карточке."
    return "По найденным документам: " + "; ".join(titles) + "."


def render_top(items: list) -> None:
    if not items:
        st.info("В ТОП пока нет технологий выше порога.")
        return
    st.caption(f"Показано {len(items)}. Если меньше 15 — столько прошло порог модели.")
    for item in items:
        left, right = st.columns([4, 1])
        with left:
            st.markdown(
                f"<div class='signal-name'>{item.get('rank', '—')}. {item.get('name_ru', '')}</div>",
                unsafe_allow_html=True,
            )
            st.caption(item.get("name_en") or "")
            st.markdown(score_bar(item.get("score")), unsafe_allow_html=True)
            st.write(_technology_description(item))
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


def inject_style() -> None:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(900px 420px at 100% -10%, rgba(61, 220, 151, 0.16), transparent 55%),
            radial-gradient(700px 380px at -10% 0%, rgba(88, 166, 255, 0.10), transparent 50%),
            #101614;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stDecoration"],
        #MainMenu,
        [data-testid="stMainMenu"],
        [data-testid="stToolbarActions"],
        [data-testid="stAppDeployButton"],
        .stDeployButton {
          display: none !important;
          visibility: hidden !important;
        }
        /* Вместо бегущего человечка Streamlit — спокойное зелёное кольцо. */
        [data-testid="stStatusWidget"] svg,
        [data-testid="stStatusWidget"] img {
          display: none !important;
        }
        [data-testid="stStatusWidget"]:has(svg)::after,
        [data-testid="stStatusWidget"]:has(img)::after {
          content: "";
          display: block;
          width: 16px;
          height: 16px;
          margin: 8px 10px 0 0;
          border-radius: 50%;
          border: 2px solid rgba(61, 220, 151, 0.25);
          border-top-color: #3DDC97;
          animation: sweep 0.9s linear infinite;
        }
        [data-testid="stMain"] {
          justify-content: flex-start !important;
        }
        [data-testid="stMain"] .block-container {
          max-width: 1100px;
          width: 100%;
          margin-left: 0 !important;
          margin-right: auto !important;
          padding-top: 1.15rem !important;
          padding-left: 3.25rem !important;
          padding-right: 2rem !important;
          padding-bottom: 2rem !important;
        }
        [data-testid="stCustomComponentV1"],
        iframe[title="streamlit_components_v1.html"] {
          position: absolute !important;
          width: 0 !important;
          height: 0 !important;
          min-height: 0 !important;
          border: 0 !important;
          overflow: hidden !important;
          pointer-events: none !important;
        }
        [data-testid="stSidebar"] {
          background: #181818 !important;
          border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        html[data-horizon-shut="1"] [data-testid="stSidebar"] {
          width: 0 !important;
          min-width: 0 !important;
          max-width: 0 !important;
          flex-basis: 0 !important;
          border: 0 !important;
          padding: 0 !important;
          overflow: hidden !important;
        }
        html[data-horizon-shut="1"] [data-testid="stSidebar"] .side-nav {
          display: none !important;
        }
        html[data-horizon-shut="1"] [data-testid="stSidebarCollapseButton"] {
          position: fixed !important;
          left: 12px !important;
          top: 14px !important;
          z-index: 1000001 !important;
          display: inline-flex !important;
          visibility: visible !important;
          transform: scaleX(-1);
        }
        html[data-horizon-shut="1"] [data-testid="stSidebarCollapseButton"] button {
          visibility: visible !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
          padding: 12px 10px 18px;
        }
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] {
          color: #d7e6de;
        }
        .side-nav { display: flex; flex-direction: column; gap: 2px; }
        [data-testid="stSidebar"] a.side-action,
        [data-testid="stSidebar"] a.hist-item {
          display: flex;
          align-items: flex-start;
          gap: 8px;
          min-height: 28px;
          padding: 6px 8px;
          border-radius: 8px;
          color: #e6e6e6 !important;
          text-decoration: none !important;
          font-size: 13.5px;
          line-height: 1.3;
        }
        [data-testid="stSidebar"] a.side-action { margin: 2px 0 6px; color: #f3f3f3 !important; }
        .side-action svg { width: 15px; height: 15px; flex: none; margin-top: 2px; opacity: 0.9; }
        [data-testid="stSidebar"] a.side-action:hover,
        [data-testid="stSidebar"] a.hist-item:hover {
          background: rgba(255, 255, 255, 0.06);
          color: #fff !important;
        }
        .side-label {
          margin: 10px 8px 4px;
          color: #8d8d8d;
          font-size: 12px;
        }
        .hist-item .body {
          display: flex;
          flex-direction: column;
          min-width: 0;
          flex: 1;
          gap: 1px;
        }
        .hist-item .title {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        [data-testid="stSidebar"] a.hist-item .when {
          color: #8d8d8d !important;
          font-size: 12px;
        }
        .hist-item .dot { margin-top: 5px; }
        [data-testid="stSidebar"] a.hist-item.active {
          background: #2a2a2a;
          color: #fff !important;
        }
        .dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          flex: none;
          background: #5c5c5c;
        }
        .dot.run { background: #4c8dff; box-shadow: 0 0 0 3px rgba(76, 141, 255, 0.16); }
        .dot.err { background: #ff8b7a; }
        .dot.done { background: #6f6f6f; }
        .hist-item.active .dot { background: #4c8dff; }
        .hist-empty {
          margin: 6px 8px 0;
          color: #8d8d8d;
          font-size: 12.5px;
          line-height: 1.4;
        }
        h1 { letter-spacing: -0.03em; }
        div[data-testid="stMetric"] {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(61, 220, 151, 0.22);
          border-radius: 16px;
          padding: 12px 14px;
        }
        .stButton > button[kind="primary"] {
          background: #3DDC97;
          color: #102018;
          border: none;
          border-radius: 12px;
          font-weight: 650;
        }
        .team-wrap {
          margin-top: 2.8rem;
          padding-top: 1.1rem;
          border-top: 1px solid rgba(255, 255, 255, 0.08);
        }
        .team-head { margin: 0 0 14px; }
        .team-head p {
          margin: 6px 0 0;
          max-width: 40rem;
          color: #C9DAD2;
          font-size: 0.95rem;
          line-height: 1.45;
        }
        .logo { width: 92px; height: 92px; flex: none; display: block; }
        @keyframes sweep { to { transform: rotate(360deg); } }
        .team-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; align-items: stretch; }
        .person {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(61, 220, 151, 0.22);
          border-radius: 16px;
          padding: 16px 16px 14px;
          min-width: 0;
          display: flex;
          flex-direction: column;
        }
        .who { display: flex; align-items: flex-start; gap: 12px; min-height: 4.6rem; }
        .mark {
          width: 36px;
          height: 36px;
          border-radius: 12px;
          display: grid;
          place-items: center;
          flex: none;
          color: #8EE7BE;
          background: rgba(61, 220, 151, 0.12);
          font-weight: 650;
        }
        .person b { display: block; font-size: 1.05rem; font-weight: 650; line-height: 1.2; }
        .person span { display: block; margin-top: 3px; color: #9FB3A9; font-size: 0.82rem; line-height: 1.35; }
        .person ul {
          margin: 14px 0 0;
          padding-left: 1.05rem;
          color: #C9DAD2;
          font-size: 0.86rem;
          line-height: 1.4;
        }
        .person li { margin: 0 0 8px; }
        .person li:last-child { margin-bottom: 0; }
        .person .links { margin-top: auto; padding-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
        .person a {
          color: #C9DAD2;
          text-decoration: none;
          font-size: 0.82rem;
          padding: 3px 8px;
          border-radius: 99px;
          border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .person a:hover { color: #E8F3EE; border-color: rgba(61, 220, 151, 0.45); }
        .hero {
          display: flex;
          align-items: center;
          gap: 16px;
          margin: 0.2rem 0 1rem;
        }
        .hero h1 { margin: -8px 0 0; font-size: 2.4rem; line-height: 1.05; }
        .kicker { color: #8EE7BE; letter-spacing: 0.14em; font-size: 0.75rem; text-transform: uppercase; }
        .score {
          height: 8px;
          border-radius: 99px;
          background: rgba(255,255,255,0.08);
          overflow: hidden;
          margin: 6px 0 4px;
          max-width: 280px;
        }
        .score > i {
          display: block;
          height: 100%;
          border-radius: 99px;
          background: linear-gradient(90deg, #1f8a62, #3DDC97);
        }
        .w0 { width: 4%; } .w1 { width: 10%; } .w2 { width: 20%; } .w3 { width: 30%; }
        .w4 { width: 40%; } .w5 { width: 50%; } .w6 { width: 60%; } .w7 { width: 70%; }
        .w8 { width: 80%; } .w9 { width: 90%; } .w10 { width: 100%; }
        .score.hot > i { background: linear-gradient(90deg, #3DDC97, #d6ff6b); }
        .chip {
          display: inline-block;
          padding: 2px 8px;
          border-radius: 99px;
          font-size: 0.78rem;
          margin-right: 6px;
          border: 1px solid transparent;
        }
        .chip.high { color: #b6ffd8; background: rgba(61,220,151,0.16); border-color: rgba(61,220,151,0.35); }
        .chip.medium { color: #ffe7a3; background: rgba(255,196,87,0.12); border-color: rgba(255,196,87,0.35); }
        .chip.low { color: #ffc1b5; background: rgba(255,107,87,0.12); border-color: rgba(255,107,87,0.35); }
        .signal-name { font-size: 1.15rem; font-weight: 650; }
        @media (max-width: 800px) { .team-grid { grid-template-columns: 1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Контакты команды. GitHub взят из авторов репозитория."""
    people = (
        (
            "Ярослав",
            "руководитель проекта, данные, скоринговая модель, работа с LLM",
            (
                "Архитектура решения, координация команды",
                "Подготовка данных и разработка признаков с проверкой на смещения выборки",
                "Скоринговая модель на логистической регрессии: валидация, калибровка вероятностей, подбор порога с приоритетом precision, интерпретируемость решений",
                "Открытый поиск: оценка кандидатов, отсечение зрелых технологий и хайпа с обоснованием",
            ),
            "+79219708813",
            "Yaroslvlapshin",
            "https://github.com/LapshinYaroslav",
        ),
        (
            "Слава",
            "сбор данных, база и бэкенд",
            (
                "Парсеры пяти открытых источников: научные публикации, препринты, новости",
                "Свежие документы по подтемам и счётчики публикаций по годам с 2020-го",
                "Итоги корпуса источников по окнам, чтобы числа технологий было с чем сравнить",
                "Схема PostgreSQL: документы, признаки, результаты скоринга, кэш счётчиков",
                "FastAPI: обработка запроса и фоновые задачи",
            ),
            "+79248216440",
            "Vyacheslav_1282",
            "https://github.com/slavachlystov",
        ),
        (
            "Егор",
            "интерфейс и поставка",
            (
                "Веб-интерфейс: дашборд ТОП-15, карточки сигналов, страницы инсайтов",
                "Источники на карточке: ссылка, дата, тип, язык, уровень доверия",
                "Прогресс запроса, ошибки на экране и история прошлых поисков",
                "ETL: единая схема документа, дедупликация, уровни доверия источников",
                "Отказоустойчивость: таймауты, повторы, изоляция сбоев, параллельный сбор",
                "Docker Compose, демо-стенд, README и техническая документация",
            ),
            "+79113833519",
            "Cbeezy0",
            "https://github.com/sboevegor7-spec",
        ),
    )
    cards = []
    for name, role, points, phone, telegram, github in people:
        pretty = phone
        digits = phone.removeprefix("+")
        if len(digits) == 11:
            pretty = f"+{digits[0]} {digits[1:4]} {digits[4:7]}-{digits[7:9]}-{digits[9:11]}"
        items = "".join(f"<li>{html.escape(point)}</li>" for point in points)
        cards.append(
            "<div class='person'>"
            "<div class='who'>"
            f"<div class='mark'>{name[:1]}</div>"
            f"<div><b>{html.escape(name)}</b><span>{html.escape(role)}</span></div>"
            "</div>"
            f"<ul>{items}</ul>"
            "<div class='links'>"
            f"<a href='tel:{phone}'>{pretty}</a>"
            f"<a href='https://t.me/{telegram}' target='_blank'>Telegram</a>"
            f"<a href='{github}' target='_blank'>GitHub</a>"
            "</div></div>"
        )
    st.markdown(
        "<div class='team-wrap'>"
        "<div class='team-head'>"
        "<div class='kicker'>команда</div>"
        "<p>Наша команда собирает сервис для Газпромбанк. "
        "По введённой теме мы ищем статьи и новости, выделяем из них ранние технологии "
        "и оцениваем, насколько это ещё слабый сигнал. "
        "На экране остаётся список таких технологий и карточка с описанием и преимуществом, "
        "а прошлые запросы сохраняются в истории.</p>"
        "</div>"
        f"<div class='team-grid'>{''.join(cards)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


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


st.set_page_config(page_title="Горизонт", layout="wide", initial_sidebar_state="collapsed")
inject_style()
init_state()
consume_nav()
render_history()
keep_sidebar_closed()

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
        render_footer()
        st.stop()

logo_src = base64.b64encode(Path(__file__).with_name("logo.svg").read_bytes()).decode("ascii")
st.markdown(
    "<div class='hero'>"
    f"<img class='logo' alt='' src='data:image/svg+xml;base64,{logo_src}'/>"
    "<div>"
    "<div class='kicker'>поиск слабых сигналов</div>"
    "<h1>Горизонт</h1>"
    "</div></div>",
    unsafe_allow_html=True,
)

st.subheader("Поисковый запрос")
locked = bool(st.session_state.polling or data)
shown_topic = (
    (data or {}).get("topic")
    or st.session_state.get("topic_text")
    or ""
)
topic = st.text_input(
    "Технологическое направление",
    value=shown_topic,
    placeholder="Введите свою тему",
    key=f"topic-{st.session_state.query_id or 'new'}",
    disabled=locked,
)

if not locked and st.button("Найти сигналы", type="primary"):
    st.session_state.error = None
    if not topic.strip():
        st.session_state.error = "Введите тему."
    elif api_base():
        try:
            start_remote(topic.strip())
        except requests.RequestException as exc:
            st.session_state.error = (
                "API недоступен. Запустите его локально или уберите API_URL. "
                f"({exc})"
            )
    else:
        with st.spinner("Считаем пример контракта…"):
            time.sleep(0.4)
        start_local(topic.strip())
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
    render_footer()
    st.stop()

st.success(f"Запрос: «{data.get('topic', '')}» · область: {data.get('area') or 'Другое'}")
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

render_footer()
