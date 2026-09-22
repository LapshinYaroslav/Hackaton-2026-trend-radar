"""Предобработка названий: отделение хвостовых скобок и подсчёт латиницы."""
import pytest

from model.dataset import (latin_word_share, split_tail_parens, strip_inline_parens,
                           word_count)


@pytest.mark.parametrize(
    "name, expected_ru, expected_tail",
    [
        ("Web application firewall для веб-приложений и API (WAF)",
         "Web application firewall для веб-приложений и API", "WAF"),
        ("Дексterity-кисти как самостоятельный компонентный рынок (не часть гуманоида)",
         "Дексterity-кисти как самостоятельный компонентный рынок", "не часть гуманоида"),
        ("Сети LoRaWAN для датчиков на батарейном питании (LoRaWAN)",
         "Сети LoRaWAN для датчиков на батарейном питании", "LoRaWAN"),
    ],
)
def test_tail_parens_are_separated(name, expected_ru, expected_tail):
    assert split_tail_parens(name) == (expected_ru, expected_tail)


def test_tail_split_ignores_parens_in_the_middle():
    """Хвостовое правило трогает только конец строки: середину снимает другая функция."""
    name = "Нейроморфные (brain-inspired) чипы для ИИ-нагрузок"

    assert split_tail_parens(name) == (name, "")


def test_inline_parens_are_removed_with_content():
    """Скобка в середине уходит вместе с содержимым: это глосса, а не часть названия.

    На s8 английское слово посреди русской фразы ломало нормализатор пять попыток
    подряд — он отвечал смесью алфавитов.
    """
    text, gloss = strip_inline_parens("Нейроморфные (brain-inspired) чипы для ИИ-нагрузок")

    assert text == "Нейроморфные чипы для ИИ-нагрузок"
    assert gloss == "brain-inspired"


def test_several_inline_parens_are_all_removed():
    text, gloss = strip_inline_parens("Тепловые сети (метро, геотермия) и накопители (ТЭС)")

    assert text == "Тепловые сети и накопители"
    assert gloss == "метро, геотермия; ТЭС"


def test_name_without_inline_parens_is_unchanged():
    assert strip_inline_parens("Частные сети 5G") == ("Частные сети 5G", "")


def test_nested_parens_are_not_split():
    """Вложенные скобки не разбираем: правило одно и простое, спорное идёт в ручной разбор."""
    name = "Технология (часть (внутри))"

    assert split_tail_parens(name)[1] == ""


def test_name_without_parens_is_unchanged():
    assert split_tail_parens("  Частные сети 5G  ") == ("Частные сети 5G", "")


def test_latin_share_counts_words_with_latin_letters():
    assert latin_word_share("Сети LoRaWAN для датчиков") == pytest.approx(0.25, abs=1e-9)
    assert latin_word_share("web application firewall") == 1.0
    assert latin_word_share("датчики") == 0.0
    assert latin_word_share("") == 0.0


def test_word_count_ignores_punctuation():
    assert word_count("Частные сети 5G на промышленных площадках") == 6
    assert word_count("ISO 20022, ISO 8583") == 4
