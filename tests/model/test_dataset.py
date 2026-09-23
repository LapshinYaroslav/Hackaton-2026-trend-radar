"""Предобработка названий: отделение хвостовых и внутренних скобок."""
import pytest

from model.dataset import split_tail_parens, strip_inline_parens


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


def test_company_stoplist_file_matches_xlsx():
    """Файл в git совпадает со стоп-листом из датасета. Без датасета тест пропускается."""
    from model import dataset

    if not dataset.SIGNALS_XLSX.exists():
        pytest.skip("нет data/raw/dataset.xlsx")
    assert set(dataset.company_stoplist()) == set(dataset.company_stoplist_from_xlsx())


def test_company_stoplist_without_file_explains(tmp_path, monkeypatch):
    """Нет файла стоп-листа — FileNotFoundError с командой сборки, а не падение на xlsx."""
    from model import dataset

    monkeypatch.setattr(dataset, "COMPANY_STOPLIST_TXT", tmp_path / "company_stoplist.txt")
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="build_company_stoplist"):
        dataset.company_stoplist()
