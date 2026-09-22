"""Загрузка размеченных технологий: сигналы из датасета организаторов, негативы из labels.

Обе стороны приводятся к одной форме одним кодом. Из датасета берутся только название,
область и метка: остальные колонки («Компании», «Почему это слабый сигнал», «Стадия
развития») в пайплайн не идут — их нет у негативов, и любое их использование стало бы
разным способом разметки у двух классов.

Исключение — колонка «Компании»: из неё собирается стоп-лист, чтобы названия компаний
не попали в поисковые термины. Это фильтр, а не признак.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIGNALS_XLSX = ROOT / "data" / "raw" / "dataset.xlsx"
NEGATIVES_CSV = ROOT / "labels" / "negatives.csv"

# Хвостовые скобки: только в самом конце строки и без вложенных скобок внутри.
TAIL_PARENS = re.compile(r"\s*\(([^()]*)\)\s*$")
# Скобки в любой другой позиции. Их содержимое тоже снимается: у сигналов внутри
# почти всегда глосса или примеры — (OCS), (brain-inspired), (метро, геотермия, НПЗ).
# На s8 именно внутренняя скобка ломала нормализатор: английское слово посреди русской
# фразы, и модель отвечала смесью алфавитов пять попыток подряд.
INLINE_PARENS = re.compile(r"\s*\(([^()]*)\)")
WORD = re.compile(r"[^\W_]+", re.UNICODE)
LATIN = re.compile(r"[a-zA-Z]")
MIN_COMPANY_LEN = 4


def split_tail_parens(name: str) -> tuple[str, str]:
    """Название без хвостовых скобок и содержимое скобок. Скобок нет — вторая строка пустая."""
    text = " ".join(str(name).split())
    match = TAIL_PARENS.search(text)
    if not match:
        return text, ""
    return text[: match.start()].strip(), match.group(1).strip()


def strip_inline_parens(name: str) -> tuple[str, str]:
    """Название без внутренних скобок и всё, что в них было, через точку с запятой."""
    inside = [item.strip() for item in INLINE_PARENS.findall(name) if item.strip()]
    text = " ".join(INLINE_PARENS.sub(" ", name).split())
    return text, "; ".join(inside)


def latin_word_share(text: str) -> float:
    """Доля слов с латинскими буквами. Пустая строка — ноль."""
    words = WORD.findall(str(text))
    if not words:
        return 0.0
    return sum(1 for word in words if LATIN.search(word)) / len(words)


def word_count(text: str) -> int:
    """Число слов в строке."""
    return len(WORD.findall(str(text)))


def load_signals() -> pd.DataFrame:
    """Сто слабых сигналов. Заголовок во второй строке листа."""
    raw = pd.read_excel(SIGNALS_XLSX, header=1)
    table = pd.DataFrame({
        "tech_id": ["s" + str(int(number)) for number in raw["№"]],
        "name": raw["Технология (слабый сигнал)"].astype(str).str.strip(),
        "area": raw["Область"].astype(str).str.strip(),
    })
    table["label"] = 1
    table["negative_type"] = ""
    table["search_terms_manual"] = ""
    return table


def load_negatives() -> pd.DataFrame:
    """Шестьдесят не-сигналов из нашей разметки."""
    raw = pd.read_csv(NEGATIVES_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    table = pd.DataFrame({
        "tech_id": raw["id"].str.strip(),
        "name": raw["name"].str.strip(),
        "area": raw["area"].str.strip(),
    })
    table["label"] = 0
    table["negative_type"] = raw["negative_type"].str.strip()
    table["search_terms_manual"] = raw["search_terms"].str.strip()
    return table


def load_all() -> pd.DataFrame:
    """Сигналы и негативы в одной таблице, с отделёнными хвостовыми скобками.

    name_en_manual — рукописное английское название негатива; в пайплайн не идёт,
    служит эталоном для проверки нормализатора. name_gloss — то же место у сигнала,
    но там лежит пояснение, а не название. name_inline_gloss — содержимое скобок из
    середины строки, тоже только для разбора.
    """
    table = pd.concat([load_signals(), load_negatives()], ignore_index=True)
    split = [split_tail_parens(name) for name in table["name"]]
    tails = [item[1] for item in split]
    inline = [strip_inline_parens(item[0]) for item in split]
    table["name_ru"] = [item[0] for item in inline]
    table["name_inline_gloss"] = [item[1] for item in inline]
    table["name_en_manual"] = [tail if label == 0 else ""
                               for tail, label in zip(tails, table["label"])]
    table["name_gloss"] = [tail if label == 1 else ""
                           for tail, label in zip(tails, table["label"])]
    return table


def company_stoplist() -> list[str]:
    """Названия компаний из датасета сигналов: у негативов такой колонки нет.

    Остаются только многословные элементы, они сопоставляются как целая фраза.
    Однословные выброшены целиком: среди них orbital, modular, tempo, width, asia,
    cambridge — обычные слова, которые оказались ещё и названиями компаний. Отличить
    одно от другого стоп-лист не может, информации для этого у него нет, поэтому на
    таких элементах он давал только ложные срабатывания: «small modular reactor» —
    каноническое название SMR — отвергался из-за «Mojo/Modular» в колонке.

    Дефект был класс-асимметричен по построению: колонка есть только у сигналов,
    поэтому блокировались названия сигналов и именно в их предметных областях.
    Измеренный ущерб — три строки из 152, все сигналы.

    Различимые однословные бренды (innatera, akida) теряем сознательно: нормализатор
    колонку «Компании» не видит, а ручной просмотр всех названий поймает бренд сразу.
    """
    raw = pd.read_excel(SIGNALS_XLSX, header=1)
    names: set[str] = set()
    for cell in raw["Компании"].astype(str):
        for part in re.split(r"[,;()/]", cell):
            item = " ".join(part.split()).casefold().strip(".")
            # В колонке рядом с компаниями встречаются суммы раундов: «$12.5m series a».
            # Это не названия, в стоп-листе им делать нечего.
            if "$" in item or item == "nan" or len(item) < MIN_COMPANY_LEN:
                continue
            if " " not in item:
                continue
            names.add(item)
    return sorted(names)
