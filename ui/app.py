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
import sys
import time

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
# streamlit run ui/app.py не кладёт корень репозитория в путь импорта
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
    st.subheader("Кейс-пример")
    st.write((content.get("case") or {}).get("text") or "—")
    st.subheader("Оценки в аналитических отчётах")
    st.write((content.get("analyst_assessment") or {}).get("text") or "—")
    st.subheader("Почему модель так уверена")
    st.write(content.get("status_explanation") or "—")
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
    st.subheader("Почему модель так уверена")
    for line in item.get("explanation_ru") or []:
        st.write(f"- {line}")
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
            preds = "; ".join(item.get("explanation_ru") or [])
            st.write(preds or "—")
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
        .radar {
          width: 14px;
          height: 14px;
          border-radius: 50%;
          border: 2px solid #3DDC97;
          box-shadow: 0 0 0 4px rgba(61, 220, 151, 0.15);
          position: relative;
        }
        .radar::after {
          content: "";
          position: absolute;
          inset: 1px;
          border-radius: 50%;
          background: conic-gradient(from 0deg, transparent 0 68%, #3DDC97 80%, transparent 81%);
          animation: sweep 2.8s linear infinite;
        }
        @keyframes sweep { to { transform: rotate(360deg); } }
        .team-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; align-items: stretch; }
        .person {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(61, 220, 151, 0.22);
          border-radius: 16px;
          padding: 14px 14px 12px;
          min-height: 132px;
          display: flex;
          flex-direction: column;
        }
        .who { display: flex; align-items: center; gap: 12px; }
        .mark {
          width: 38px;
          height: 38px;
          border-radius: 12px;
          display: grid;
          place-items: center;
          flex: none;
          color: #8EE7BE;
          background: rgba(61, 220, 151, 0.12);
          font-weight: 650;
        }
        .person b { display: block; font-size: 1.05rem; font-weight: 650; }
        .person span { color: #9FB3A9; font-size: 0.85rem; }
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
        .hero h1 { margin: 0; font-size: 2.4rem; }
        .kicker { color: #8EE7BE; letter-spacing: 0.14em; font-size: 0.75rem; text-transform: uppercase; }
        .radar.lg { width: 42px; height: 42px; border-width: 2px; }
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
            "модель и оркестратор",
            "+79219708813",
            "Yaroslvlapshin",
            "https://github.com/LapshinYaroslav",
        ),
        (
            "Слава",
            "сбор документов и API",
            "+79248216440",
            "Vyacheslav_1282",
            "https://github.com/slavachlystov",
        ),
        (
            "Егор",
            "интерфейс и отчёты",
            "+79113833519",
            "Cbeezy0",
            "https://github.com/sboevegor7-spec",
        ),
    )
    cards = []
    for name, role, phone, telegram, github in people:
        pretty = phone
        digits = phone.removeprefix("+")
        if len(digits) == 11:
            pretty = f"+{digits[0]} {digits[1:4]} {digits[4:7]}-{digits[7:9]}-{digits[9:11]}"
        cards.append(
            "<div class='person'>"
            "<div class='who'>"
            f"<div class='mark'>{name[:1]}</div>"
            f"<div><b>{name}</b><span>{role}</span></div>"
            "</div>"
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
        "<p>Trend Radar для Газпромбанк.Tech: модель, сбор документов и экран, на котором сигнал становится понятным.</p>"
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


st.set_page_config(page_title="Радар слабых сигналов", layout="wide")
inject_style()
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
        render_footer()
        st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="radar lg"></div>
      <div>
        <div class="kicker">поиск слабых сигналов</div>
        <h1>Радар</h1>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
mode = "живой API" if api_base() else "пример контракта, без Docker"
st.caption(mode)

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
    render_footer()
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

render_footer()
