"""Собирает labels/company_stoplist.txt из колонки «Компании» датасета организаторов.

Запускается один раз, когда меняется датасет: у жюри датасета нет, поэтому сервис
читает готовый файл из git (model.dataset.company_stoplist). Правила отбора — те же,
что в model.dataset.company_stoplist_from_xlsx: только многословные названия.

Запуск: python -m scripts.build_company_stoplist
"""
from model.dataset import COMPANY_STOPLIST_TXT, ROOT, company_stoplist_from_xlsx


def main() -> None:
    """Пишет стоп-лист по фразе на строку, отсортированно."""
    names = company_stoplist_from_xlsx()
    COMPANY_STOPLIST_TXT.write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")
    print(f"{COMPANY_STOPLIST_TXT.relative_to(ROOT)}: {len(names)} фраз")


if __name__ == "__main__":
    main()
