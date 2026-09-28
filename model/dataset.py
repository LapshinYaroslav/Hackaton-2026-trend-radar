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
# Многословная часть стоп-листа компаний, собранная из датасета один раз
# (scripts/build_company_stoplist.py). Лежит в git: у жюри датасета организаторов нет.
COMPANY_STOPLIST_TXT = ROOT / "labels" / "company_stoplist.txt"

MIN_COMPANY_LEN = 4


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


def company_stoplist() -> list[str]:
    """Стоп-лист компаний из labels/company_stoplist.txt: по фразе на строку.

    Читается только файл, датасет не нужен. Нет файла — ошибка с подсказкой, как его собрать.
    """
    if not COMPANY_STOPLIST_TXT.exists():
        raise FileNotFoundError(
            f"нет {COMPANY_STOPLIST_TXT.relative_to(ROOT)}: собери его командой "
            "python -m scripts.build_company_stoplist (нужен data/raw/dataset.xlsx)")
    lines = COMPANY_STOPLIST_TXT.read_text(encoding="utf-8").splitlines()
    return sorted({line.strip() for line in lines if line.strip()})


def company_stoplist_from_xlsx() -> list[str]:
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
