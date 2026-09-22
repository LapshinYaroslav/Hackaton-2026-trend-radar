"""Признаки по счётчикам поиска №2 (pipeline.md 0.2).

Формулы те же, что в версии 0.1. Изменилось, откуда берутся числа: не из выгруженных
документов, а из счётчиков источника. Потолок выдачи на счётчики не влияет, поэтому
volume, growth и recency больше не зависят от того, сколько документов удалось скачать.

Доли считаются по сырым числам документов (задача 6). Нормировка на корпус источника
отменена: она вводилась под многотермовый запрос и после перехода на один термин стала
вредна. Подробности и числа — в _shares.

Ожидаемый знак долей, проверенный на 26 технологиях Edge: у сигнала share_research
НИЖЕ, а не выше, share_market — выше. Зрелая технология копила статьи годами при
умершем новостном потоке; зарождающаяся имеет десяток статей и всплеск новостей о
раундах — в датасете организаторов 43 сигнала из 100 обоснованы инвестраундом.
Медианы: сигналы 0.5276, негативы 0.9882.
"""
import math

import pandas as pd

from collector.constants import AGGREGATE_WINDOWS, YEAR_WINDOWS

FEATURE_NAMES = ["volume", "growth", "recency", "share_research", "share_market",
                 "share_prev6"]

# Источник целиком относится к научным или к рыночным: у arXiv тип всегда preprint,
# у новостных источников практически всегда news (pipeline.md 0.2).
RESEARCH_SOURCES = {"openalex", "arxiv"}
MARKET_SOURCES = {"techcrunch"}
KNOWN_SOURCES = RESEARCH_SOURCES | MARKET_SOURCES


def _check_windows(counters: pd.DataFrame) -> None:
    """Ловит годовые счётчики, которые забыли сложить в рабочие окна.

    Без этой проверки окна all бы просто не нашлось: volume стал бы нулём, growth и
    доли — NaN, и ошибка выглядела бы как «технология, про которую никто не пишет».
    """
    years = sorted(set(counters["window"].astype(str)) & set(YEAR_WINDOWS))
    if years:
        raise ValueError(
            f"На входе годовые окна {years}. Сложи их в рабочие: "
            f"model.counters.aggregate_windows(counters)."
        )


def check_coverage(counters: pd.DataFrame, expected_sources: set[str]) -> None:
    """Проверяет, что каждый ожидаемый источник ответил по всем рабочим окнам.

    Неполное покрытие — отказ считать, а не предупреждение. Повод из этапа 2: arXiv
    под троттлингом ответил на 25 запросов из 182, признаки посчитались почти без него,
    и n_sources_growth показал ROC AUC 0.750 — он мерил, каким технологиям повезло
    получить ответ стороннего API. На полном покрытии тот же признак даёт ровно 0.500.
    """
    have = set(zip(counters["source"].astype(str), counters["window"].astype(str)))
    missing = sorted((source, window) for source in expected_sources
                     for window in AGGREGATE_WINDOWS if (source, window) not in have)
    if missing:
        got = len(expected_sources) * len(AGGREGATE_WINDOWS) - len(missing)
        raise ValueError(
            f"неполное покрытие: получено {got} из "
            f"{len(expected_sources) * len(AGGREGATE_WINDOWS)} пар источник-окно, "
            f"не хватает {missing}. Признаки по неполным данным не считаются."
        )


def _check_sources(counters: pd.DataFrame) -> None:
    """Проверяет, что каждый источник отнесён к научным или рыночным."""
    unknown = sorted(set(counters["source"]) - KNOWN_SOURCES)
    if unknown:
        raise ValueError(
            f"Источники без отнесения к научным или рыночным: {unknown}. "
            f"Добавь их в RESEARCH_SOURCES или MARKET_SOURCES, иначе доли посчитаются "
            f"мимо них и молча разойдутся с volume."
        )


def _window_sum(counters: pd.DataFrame, window: str, sources: set[str] | None = None) -> int:
    """Сумма счётчиков по окну, при необходимости по подмножеству источников."""
    rows = counters.loc[counters["window"] == window]
    if sources is not None:
        rows = rows.loc[rows["source"].isin(sources)]
    return int(rows["n"].sum())


def _totals_by_window(source_totals: pd.DataFrame, window: str) -> pd.Series:
    """Корпусные итоги источников за окно, индекс — источник."""
    rows = source_totals.loc[source_totals["window"] == window]
    return rows.set_index("source")["n_total"]


def _volume(counters: pd.DataFrame) -> float:
    """ln(1 + число документов за весь период)."""
    return math.log1p(_window_sum(counters, "all"))


def _recency(counters: pd.DataFrame) -> float:
    """Доля документов за последние 24 месяца. NaN, если документов нет."""
    n_all = _window_sum(counters, "all")
    if n_all == 0:
        return math.nan
    return _window_sum(counters, "recent24") / n_all


def _share_prev6(counters: pd.DataFrame) -> float:
    """Какая доля двенадцатилетнего следа технологии пришлась на предыдущую шестилетку.

    Возраст темы, посчитанный суммой, а не минимумом. Дату первого документа брать
    нельзя: после ретро-индексации OpenAlex отдаёт книгу 1934 года с датой публикации
    2026 (находка 014), и один такой выброс унёс бы возраст на десятилетия. Сумма к
    единичным записям устойчива, а книга 1934 года не попадает ни в одно из двух окон.

    Окна равной длины — шесть лет против шести, — поэтому дробь сравнима между
    источниками с разной историей. Нормировать не на что и не нужно: числитель и
    знаменатель — числа из одних и тех же источников. Оговорка: корпуса источников
    со временем растут, поэтому доля занижена у всех технологий разом. Сравнивать
    значения между технологиями можно, читать как настоящую долю старых работ — нет.
    """
    n_prev = _window_sum(counters, "prev6")
    n_all = _window_sum(counters, "all")
    if n_prev + n_all == 0:
        return math.nan
    return n_prev / (n_prev + n_all)


def growth_by_source(counters: pd.DataFrame, source_totals: pd.DataFrame) -> pd.DataFrame:
    """Рост по каждому источнику отдельно: колонки source, g, w.

    g = ln((n_now+1)/(n_before+1)) - ln(N_now/N_before) — рост технологии за вычетом
    роста самого источника. w = n_now + n_before — сколько документов за этим ростом
    стоит. Источник без корпусных итогов за оба окна или с нулевым корпусом в строки
    не попадает: поправку для него взять неоткуда.
    """
    totals_now = _totals_by_window(source_totals, "now")
    totals_before = _totals_by_window(source_totals, "before")
    rows: list[dict[str, float]] = []
    for source in sorted(set(counters["source"]) & set(totals_now.index) & set(totals_before.index)):
        corpus_now = int(totals_now.loc[source])
        corpus_before = int(totals_before.loc[source])
        if corpus_now <= 0 or corpus_before <= 0:
            # Нулевой корпус — это «источник не ответил», а не «документов не было».
            continue
        n_now = _window_sum(counters, "now", {source})
        n_before = _window_sum(counters, "before", {source})
        rows.append({
            "source": source,
            "g": math.log((n_now + 1) / (n_before + 1)) - math.log(corpus_now / corpus_before),
            "w": float(n_now + n_before),
        })
    return pd.DataFrame(rows, columns=["source", "g", "w"])


def _growth(counters: pd.DataFrame, source_totals: pd.DataFrame) -> float:
    """Средний рост по источникам, взвешенный числом документов за окна роста.

    Поправка на фон берётся у каждого источника своя. Складывать счётчики всех
    источников в один числитель, а корпуса — в один знаменатель нельзя: сумма корпусов
    на 96% состоит из OpenAlex, он в 25 раз больше arXiv и в 1400 раз больше TechCrunch.
    Фон OpenAlex тогда применялся бы и к технологии, которой в OpenAlex нет. Разброс
    фона измерен 20.09.2026: openalex +0.3395, arxiv +0.3575, techcrunch -0.3594 —
    TechCrunch сжимается, пока научные базы растут. Для технологии, живущей в новостях,
    пуловая поправка занижала рост примерно на 0.70, то есть вдвое. А технология без
    научного следа и с рыночным шумом — это профиль маркетингового хайпа, поэтому
    ошибка коррелировала бы с меткой класса, и модель выучила бы артефакт.

    Вес — число документов, а не единица на источник: источник, нашедший три новости,
    не должен весить столько же, сколько источник с двумя тысячами статей.
    """
    detail = growth_by_source(counters, source_totals)
    if detail.empty:
        return math.nan
    weight = float(detail["w"].sum())
    if weight == 0:
        # Ни одного документа в окнах роста ни у одного источника: взвешивать нечем,
        # но фон известен, и ответ «рост равен минус фону» содержательнее пропуска.
        return float(detail["g"].mean())
    return float((detail["g"] * detail["w"]).sum() / weight)


def growth_by_sources(counters: pd.DataFrame, source_totals: pd.DataFrame,
                      sources: set[str]) -> float:
    """Рост по подмножеству источников, взвешенный числом документов.

    growth считает средний рост по всем источникам; здесь то же самое, но по
    заданному подмножеству. Признак модели growth_research — это вызов с
    RESEARCH_SOURCES: рост научного следа отдельно от рыночного.
    """
    detail = growth_by_source(counters, source_totals)
    part = detail.loc[detail["source"].isin(sources)]
    if part.empty:
        return math.nan
    weight = float(part["w"].sum())
    if weight == 0:
        return float(part["g"].mean())
    return float((part["g"] * part["w"]).sum() / weight)


def _shares(counters: pd.DataFrame) -> tuple[float, float]:
    """Доли научных и рыночных источников по сырым числам документов.

    share_research = n_research / (n_research + n_market). Корпусные итоги здесь больше
    не участвуют: для growth они на месте, а долям только мешали.

    Почему решение пересмотрено. Нормировка r = n / N вводилась при многотермовом
    запросе: OpenAlex отдавал тысячи документов на технологию, и сырая доля науки
    вырождалась в 0.9986 у кого угодно. После перехода на один канонический термин
    научные счётчики упали в десятки раз, и посылка перестала выполняться. Измерено
    на 26 технологиях Edge 20.09.2026, вырожденных значений (выше 0.99 или ниже 0.01):
      сырые доли            9 из 26   ROC AUC 0.250
      нормировка на корпус 13 из 26   ROC AUC 0.312
      нормировка на медиану 9 из 26   ROC AUC 0.287
    Корпус OpenAlex в 667 раз больше корпуса TechCrunch, поэтому нормировка давала
    одной новости вес сотен статей и схлопывала долю науки у всех, про кого TechCrunch
    написал хоть раз: у edge to cloud routing 7 статей против 56 новостей давали не
    0.111, а 0.0002.

    Ноль документов в рыночном источнике — это не пропуск: share_research станет ровно 1,
    share_market ровно 0. NaN только когда документов нет вовсе.
    """
    rows = counters.loc[counters["window"] == "all"]
    by_source = rows.groupby("source")["n"].sum()
    research = int(by_source.reindex(sorted(RESEARCH_SOURCES)).fillna(0).sum())
    market = int(by_source.reindex(sorted(MARKET_SOURCES)).fillna(0).sum())
    total = research + market
    if total == 0:
        return math.nan, math.nan
    share_research = research / total
    # Источник либо научный, либо рыночный, третьего не дано (_check_sources), поэтому
    # доля рынка — это остаток. Так записано в pipeline.md 0.2, и так видно, что два
    # признака связаны точно: их сумма всегда 1. Это надо помнить при обучении.
    return share_research, 1.0 - share_research


def compute_features(counters: pd.DataFrame, source_totals: pd.DataFrame,
                     *, expected_sources: set[str] | None) -> dict[str, float]:
    """Шесть признаков по счётчикам технологии, в порядке FEATURE_NAMES.

    counters: колонки source, window, n. source_totals: колонки source, window, n_total.
    Окна счётчиков: all, now, before, recent24, prev6; окна итогов: all, now, before.
    И те и другие получаются из годовых сложением в model.counters.aggregate_windows.

    expected_sources обязателен и передаётся всегда: признаки по неполным данным считать
    нельзя (check_coverage). Явный None означает осознанный пропуск проверки — так
    сделаны юнит-тесты формул, которые подают один источник и одно окно.
    """
    _check_windows(counters)
    if expected_sources:
        check_coverage(counters, expected_sources)
    _check_sources(counters)
    if _window_sum(counters, "all") == 0:
        # За период сбора документов нет: объём равен нулю, остальное считать не из чего.
        # Кроме share_prev6: если в прошлую шестилетку документы были, а в эту не
        # стало, доля равна единице, и это содержательный ответ, а не пропуск.
        return {"volume": 0.0, "growth": math.nan, "recency": math.nan,
                "share_research": math.nan, "share_market": math.nan,
                "share_prev6": _share_prev6(counters)}

    share_research, share_market = _shares(counters)
    values = {
        "volume": _volume(counters),
        "growth": _growth(counters, source_totals),
        "recency": _recency(counters),
        "share_research": share_research,
        "share_market": share_market,
        "share_prev6": _share_prev6(counters),
    }
    return {name: values[name] for name in FEATURE_NAMES}
