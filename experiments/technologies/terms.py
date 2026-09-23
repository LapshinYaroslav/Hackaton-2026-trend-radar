"""Нормализатор названий: русское название технологии -> каноническое английское.

Единственный вызов модели в пайплайне. На вход идут только name_ru и область: ни метки
класса, ни negative_type, ни рукописного английского имени негатива, ни колонок датасета
про компании, стадию и обоснование.

Шага синонимов больше нет (задача 3В). Вклад синонима в volume зависит от того, есть ли
у технологии устоявшиеся альтернативные названия, а это и есть зрелость, то есть метка
класса. У негатива три термина дали бы три вклада, у сигнала 2025 года — один вклад и
два нуля, и разница ушла бы в признак. Текст удалённого промпта лежит рядом в
prompt_synonyms_diagnostic.md: на этапе 2 утверждение будет проверено числом.

Модель прибита явно: yandexgpt-5-pro, не алиас. Повтор после отказа идёт с температурой
0.8 и приписанным нарушением: на 0.3 модель повторяет тот же ответ слово в слово, это
видно по прогонам s8 и n33 от 20.09.2026.
"""
import os
import re

from experiments.subqueries import llm

MODEL_VERSION = "yandexgpt-5-pro"
PROMPT_VERSION_NORM = "v4"
FAMILY_NORM = "tech_norm"
CANDIDATES = 3

TEMPERATURE_FIRST = 0.3
TEMPERATURE_RETRY = 0.8
MAX_RETRIES = 2

MAX_WORDS_NAME = 5
BAD_CHARS_NAME = set('"\'(),:;/{}[]')
YEAR = re.compile(r"\b(19\d{2}|20\d{2}|2100)\b")
WORD = re.compile(r"[^\W_]+", re.UNICODE)
LATIN = re.compile(r"[a-z]", re.IGNORECASE)
# Промпт просит английский ответ, но на входе без латиницы модель отвечает по-русски
# или смешивает алфавиты внутри слова («neuroморфный chip»). Для поиска это мусор:
# в OpenAlex и TechCrunch русских записей нет.
CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def pin_model() -> None:
    """Фиксирует модель для всех вызовов этого прогона."""
    os.environ["YANDEX_GPT_MODEL"] = MODEL_VERSION


def stop_pattern(stoplist: list[str]) -> re.Pattern:
    """Регулярка стоп-листа компаний: совпадение по целому слову."""
    parts = sorted((re.escape(item) for item in stoplist), key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


def _words(text: str) -> list[str]:
    return WORD.findall(text)


def check_candidates(text: str, stop: re.Pattern) -> tuple[list[str], list[str]]:
    """Разбирает ответ нормализатора: три различных кандидата и список нарушений.

    Выбирать из них код не будет: критерия, который не коррелировал бы с меткой класса,
    не существует. Максимум по объёму берёт самое широкое понятие, минимум — выдуманную
    фразу, середина — произвол. Выбирает человек, вслепую (build_candidates_file).
    """
    names = [line.strip()[len("VARIANT:"):].strip().strip(".").lower()
             for line in text.strip().splitlines()
             if line.strip().upper().startswith("VARIANT:")]
    if len(names) != CANDIDATES:
        return names, [f"в предыдущем ответе было строк VARIANT {len(names)}, "
                       f"нужно {CANDIDATES}"]
    if len(set(names)) != CANDIDATES:
        return names, ["в предыдущем ответе варианты повторялись, нужны три разных"]
    problems: list[str] = []
    for name in names:
        problems.extend(f"{name!r}: {item}" for item in check_one(name, stop))
    return names, problems


def check_one(name: str, stop: re.Pattern) -> list[str]:
    """Проверки одного варианта названия."""
    problems: list[str] = []
    if len(_words(name)) > MAX_WORDS_NAME:
        problems.append(f"в предыдущем ответе было слов {len(_words(name))}, "
                        f"потолок {MAX_WORDS_NAME}")
    bad = sorted(set(name) & BAD_CHARS_NAME)
    if bad:
        problems.append(f"в предыдущем ответе были запрещённые символы {bad}")
    if YEAR.search(name):
        problems.append("в предыдущем ответе был год")
    found = stop.search(name)
    if found:
        problems.append(f"в предыдущем ответе было название компании {found.group(0)!r}")
    if not LATIN.search(name):
        problems.append("в предыдущем ответе не было латинских букв")
    if CYRILLIC.search(name):
        problems.append("в предыдущем ответе была кириллица")
    return problems


def _with_note(topic: str, note: str) -> str:
    """Дописывает к входу указание, что именно было нарушено в прошлый раз."""
    return topic if not note else f"{topic}\n\n{note}"


def ask_name(name_ru: str, area: str, repeat: int = 0, note: str = "") -> tuple[str, bool]:
    """Каноническое английское название. Второй элемент — был ли сетевой вызов."""
    topic = _with_note(f"название: {name_ru}\nобласть: {area}", note)
    answer, called = llm.ask(topic, PROMPT_VERSION_NORM, repeat_index=repeat,
                             family=FAMILY_NORM, json_object=False,
                             temperature=TEMPERATURE_FIRST if not repeat else TEMPERATURE_RETRY)
    return (answer.get("text") or ""), called
