"""Стоп-лист компаний: файл в git и сверка с датасетом."""
import pytest


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
