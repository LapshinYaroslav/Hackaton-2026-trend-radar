"""Загрузка датасета организаторов в чистую таблицу."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "dataset.xlsx"

COLUMNS = {
    "№": "id",
    "Технология (слабый сигнал)": "name",
    "Область": "area",
    "Компании": "companies",
    "Почему это слабый сигнал": "rationale",
    "Стадия развития": "stage_raw",
    "Тренд упоминаний": "trend_raw",
    "Балл (стадия+тренд)": "score",
    "Источники": "sources_raw",
}
LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
NEGATIVES_FILE = ROOT / "labels" / "negatives.csv"
GROUPS_FILE = ROOT / "labels" / "signal_groups.csv"

AREAS = {"Edge", "Защита ИИ", "Индустриальный ИИ", "Инфраструктура ИИ", "Роботы", "Финтех"}
NEGATIVE_TYPES = {"mature", "hype", "fading"}

NEGATIVE_COLUMNS = ["id", "name", "area", "negative_type", "pair_with", "criterion", "evidence_url"]
GROUP_COLUMNS = ["id", "group", "reason"]
TABLE_COLUMNS = ["tech_id", "name", "area", "label", "source", "negative_type", "pair_with", "group"]


def parse_links(text: str) -> list[dict]:
    """Markdown-ссылки вида [Название](url) -> список словарей."""
    return [{"title": title, "url": url} for title, url in LINK_RE.findall(text)]


def load_signals(path: Path = RAW_FILE) -> pd.DataFrame:
    """Читает датасет организаторов: 100 слабых сигналов, label = 1."""
    # Первая строка листа — название таблицы, заголовки во второй
    df = pd.read_excel(path, header=1)
    df = df.loc[:, list(COLUMNS)].rename(columns=COLUMNS)
    df["id"] = df["id"].astype(int)
    df["score"] = df["score"].astype(int)
    df["companies"] = df["companies"].str.split(r"\s*[,;]\s*")
    df["sources"] = df["sources_raw"].map(parse_links)
    df["label"] = 1  # в датасете только слабые сигналы
    return df


def _normalize_name(name: str) -> str:
    """Название без регистра и лишних пробелов — для сравнения на совпадение."""
    return re.sub(r"\s+", " ", name).strip().casefold()


def _read_csv(path: Path) -> pd.DataFrame:
    """Читает CSV как текст: utf-8-sig, все ячейки — строки, пустые — "", без NaN."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return df.apply(lambda col: col.str.strip())


def _row_numbers(df: pd.DataFrame, mask: pd.Series) -> list[int]:
    """Номер строки в файле = индекс DataFrame + 2 (первая строка — заголовок)."""
    return (df.index[mask] + 2).tolist()


def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Есть ли все обязательные колонки; падает сразу, до любых проверок по строкам."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")


def _check_not_empty(df: pd.DataFrame, column: str) -> list[str]:
    """Значение колонки не должно быть пустой строкой."""
    mask = df[column] == ""
    if not mask.any():
        return []
    return [f"{column}: пустое значение в строках {_row_numbers(df, mask)}"]


def _check_unique(df: pd.DataFrame, column: str, key=None) -> list[str]:
    """Значения колонки уникальны (после key(), если задан); пустые строки не считаются."""
    values = df[column].map(key) if key else df[column]
    mask = values.duplicated(keep=False) & (values != "")
    if not mask.any():
        return []
    return [f"{column}: повтор значения в строках {_row_numbers(df, mask)}"]


def _check_values_in_set(df: pd.DataFrame, column: str, allowed: set[str]) -> list[str]:
    """Значения колонки — только из allowed."""
    mask = ~df[column].isin(allowed)
    if not mask.any():
        return []
    return [f"{column}: недопустимое значение в строках {_row_numbers(df, mask)}"]


def _check_regex(df: pd.DataFrame, column: str, pattern: str, rule: str) -> list[str]:
    """Значения колонки целиком соответствуют pattern."""
    mask = ~df[column].str.fullmatch(pattern)
    if not mask.any():
        return []
    return [f"{rule} — строки {_row_numbers(df, mask)}"]


def _to_nullable_int(series: pd.Series) -> pd.Series:
    """Пустая строка -> <NA>, иначе целое число; тип Int64."""
    return pd.to_numeric(series.replace("", pd.NA), errors="raise").astype("Int64")


def load_negatives(path: Path = NEGATIVES_FILE) -> pd.DataFrame:
    """Читает и проверяет ручную разметку не-сигналов, добавляет label = 0."""
    df = _read_csv(path)
    _check_columns(df, NEGATIVE_COLUMNS)

    errors: list[str] = []
    errors += _check_regex(df, "id", r"n\d+", "id должен быть в формате n<число>")
    errors += _check_unique(df, "id")
    errors += _check_not_empty(df, "name")
    errors += _check_unique(df, "name", key=_normalize_name)
    errors += _check_values_in_set(df, "area", AREAS)
    errors += _check_values_in_set(df, "negative_type", NEGATIVE_TYPES)
    errors += _check_regex(df, "evidence_url", r"https?://.*",
                            "evidence_url должен начинаться с http:// или https://")
    errors += _check_regex(df, "pair_with", r"([1-9]\d*)?",
                            "pair_with должен быть пустым или положительным числом")
    if errors:
        raise ValueError("\n".join(errors))

    result = df[NEGATIVE_COLUMNS].copy()
    result["pair_with"] = _to_nullable_int(result["pair_with"])
    result["label"] = 0
    return result


def load_groups(path: Path = GROUPS_FILE) -> dict[int, str]:
    """Читает ручную разметку групп сигналов: {id сигнала: название группы}."""
    df = _read_csv(path)
    _check_columns(df, GROUP_COLUMNS)

    errors: list[str] = []
    errors += _check_regex(df, "id", r"\d+", "id должен быть целым числом")
    errors += _check_unique(df, "id")
    errors += _check_not_empty(df, "group")
    if errors:
        raise ValueError("\n".join(errors))

    return {int(row_id): group for row_id, group in zip(df["id"], df["group"])}


def _check_ids_known(ids, known_ids: set, rule: str) -> list[str]:
    """Каждое значение из ids должно быть среди known_ids."""
    unknown = sorted(set(ids) - known_ids)
    if not unknown:
        return []
    return [f"{rule}: неизвестные id {unknown}"]


def _check_no_name_overlap(signals: pd.DataFrame, negatives: pd.DataFrame) -> list[str]:
    """Название не-сигнала не должно совпадать (без регистра) с названием сигнала."""
    signal_names = set(signals["name"].map(_normalize_name))
    overlap = sorted(set(negatives["name"].map(_normalize_name)) & signal_names)
    if not overlap:
        return []
    return [f"Названия не-сигналов совпадают с сигналами: {overlap}"]


def _build_signal_rows(signals: pd.DataFrame, groups: dict[int, str]) -> pd.DataFrame:
    """Строки сигналов: tech_id='s'+id, label=1, source='dataset', group из словаря или 's'+id."""
    rows = signals[["id", "name", "area"]].copy()
    rows["tech_id"] = "s" + rows["id"].astype(str)
    rows["label"] = 1
    rows["source"] = "dataset"
    rows["negative_type"] = pd.NA
    rows["pair_with"] = pd.array([pd.NA] * len(rows), dtype="Int64")
    rows["group"] = rows["id"].map(lambda i: groups.get(i, f"s{i}"))
    return rows[TABLE_COLUMNS]


def _build_negative_rows(negatives: pd.DataFrame) -> pd.DataFrame:
    """Строки не-сигналов: tech_id=id из файла, label=0, source='negatives', group=tech_id."""
    rows = negatives[["id", "name", "area", "negative_type", "pair_with"]].copy()
    rows["tech_id"] = rows["id"]
    rows["label"] = 0
    rows["source"] = "negatives"
    rows["group"] = rows["id"]
    return rows[TABLE_COLUMNS]


def build_training_table(signals: pd.DataFrame, negatives: pd.DataFrame, groups: dict[int, str]) -> pd.DataFrame:
    """Собирает сигналы (label=1) и не-сигналы (label=0) в одну таблицу для обучения."""
    signal_ids = set(signals["id"])

    errors: list[str] = []
    errors += _check_ids_known(groups.keys(), signal_ids, "id в group")
    errors += _check_ids_known(negatives["pair_with"].dropna(), signal_ids, "pair_with")
    errors += _check_no_name_overlap(signals, negatives)
    if errors:
        raise ValueError("\n".join(errors))

    return pd.concat(
        [_build_signal_rows(signals, groups), _build_negative_rows(negatives)],
        ignore_index=True,
    )


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    signals = load_signals()
    negatives = load_negatives()
    groups = load_groups()
    table = build_training_table(signals, negatives, groups)

    print("Размер таблицы:", table.shape)
    print(pd.crosstab(table["area"], table["label"]))

    group_sizes = table.groupby("group").size()
    print("Групп:", len(group_sizes), "из нескольких технологий:", int((group_sizes > 1).sum()))

    n_negatives = int((table["label"] == 0).sum())
    if n_negatives != 60:
        print(f"ПРЕДУПРЕЖДЕНИЕ: не-сигналов {n_negatives}, ожидалось 60")

    per_area = table.loc[table["label"] == 0].groupby("area").size().reindex(sorted(AREAS), fill_value=0)
    bad_areas = per_area[per_area != 10]
    if not bad_areas.empty:
        print(f"ПРЕДУПРЕЖДЕНИЕ: не по 10 не-сигналов в областях: {bad_areas.to_dict()}")