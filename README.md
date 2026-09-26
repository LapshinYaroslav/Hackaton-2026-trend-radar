# Hackaton-2026-trend-radar
Сервис для автоматизированного сбора и анализа зарождающихся трендов (слабые сигналы) в научно-технологических отраслях

Требования: [`docs/pipeline.md`](docs/pipeline.md).

## Сборщик

Модуль `collector/`: поиск №1 (свежие документы по подзапросам) и счётчики поиска №2; вызывается из оркестратора `python -m pipeline`. Источники: OpenAlex, arXiv, TechCrunch;
для счётчика патентов боевой модели `s2a2-v1` — поисковая платформа Роспатента (`collector/rospatent.py`, только число
патентных документов по фразе за 2020-09-01…2026-08-31, мировой фонд — список датасетов в `model/config.py`).

```bash
pip install -r requirements-dev.txt
pytest tests/collector
```

Командный план пайплайна: [`docs/pipeline.md`](docs/pipeline.md). Методология модели: [`docs/methodology.md`](docs/methodology.md).

## Что уже можно поднять

Актуальный пример ответа оркестратора: [`docs/contracts/query_result.example.json`](docs/contracts/query_result.example.json). Пока оркестратора нет, API через несколько секунд отдаёт этот пример. PostgreSQL для этого мока не нужен.

Сервисы Docker Compose:

| Сервис | Роль | Порт на хосте |
|--------|------|----------------|
| `db` | PostgreSQL 16 | 5432 |
| `api` | FastAPI (пока заглушка на моке) | 8000 |
| `ui` | Streamlit | 8501 |

**Важно:** внутри Docker сервисы ходят друг к другу по **имени сервиса** (`http://api:8000`), не по `localhost`.

## Быстрый старт (macOS + Docker Desktop)

1. Установи и запусти [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Клонируй репозиторий и перейди в него:

```bash
git clone https://github.com/LapshinYaroslav/Hackaton-2026-trend-radar.git
cd Hackaton-2026-trend-radar
```

3. Создай `.env` из шаблона:

```bash
cp .env.example .env
```

4. Подними стек:

```bash
docker compose up --build -d
```

5. Открой в браузере:

- UI: http://localhost:8501  
- API health: http://localhost:8000/health  
- Пример поиска API: http://localhost:8000/search?q=технологии%20в%20медицине  

6. Остановка:

```bash
docker compose down
```

Схема — `db/schema.sql`: кэш сборщика, обучающая выборка (100 сигналов + 60 негативов), история запросов. Посев из `labels/`, `data/interim/technologies.csv` и при наличии — `data/raw/dataset.xlsx`.

```bash
# локально, когда Postgres уже запущен
python -m db
```

API при старте с `DATABASE_URL` сам применяет схему и заливает справочники. Если меняли `db/schema.sql` и нужно пересоздать volume с нуля:

```bash
docker compose down -v
docker compose up --build -d
```

## Локально без Docker

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ui/requirements.txt -r api/requirements.txt
```

Только интерфейс (читает файл примера):

```bash
streamlit run ui/app.py
```

Интерфейс через API (прогресс по стадиям, тот же пример):

```bash
.venv/bin/python -m uvicorn api.main:app --app-dir . --host 127.0.0.1 --port 8000
```

Во втором окне:

```bash
export API_URL=http://127.0.0.1:8000
streamlit run ui/app.py
```

## Структура (зоны ответственности)

- `ui/` — интерфейс (Егор)
- `api/` — FastAPI (Слава; сейчас заглушка)
- `db/` — схема Postgres
- `collector/`, `search/` — сбор документов
- `model/` — признаки и скоринг
- `docs/` — пайплайн, методология, контракты

## Секреты

Ключи и пароли только в `.env` (в `.gitignore`). В репозитории — `.env.example` без боевых значений.

Переменные, которые читает пайплайн:

| переменная | зачем |
|---|---|
| `YANDEX_FOLDER_ID`, `YANDEX_API_KEY`, `YANDEX_GPT_MODEL` | YandexGPT: подзапросы и извлечение кандидатов |
| `OPENALEX_API_KEY` (или `OPEN_ALEX`) | OpenAlex: поиск и счётчики научных работ |
| `ROSPATENT` | Роспатент: число патентов по фразе, признак `share_patent` модели `s2a2-v1` |

Без `ROSPATENT` пайплайн не падает: патентный признак у всех кандидатов недоступен, модель подставляет медиану
обучения, в `warnings` прогона об этом запись. Для отладки Роспатент выключается флагом `--no-rospatent`.
