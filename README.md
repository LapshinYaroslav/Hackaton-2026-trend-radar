# Hackaton-2026-trend-radar
Сервис для автоматизированного сбора и анализа зарождающихся трендов (слабые сигналы) в научно-технологических отраслях

Требования: [`docs/pipeline.md`](docs/pipeline.md).

## Сборщик

Модуль `collector/`: режим обучения (шаг 3) и запрос (поиск №1 и поиск №2). Источники: OpenAlex, arXiv, TechCrunch.

```bash
pip install -r requirements-dev.txt
pytest tests/collector
python -m collector history --candidate candidate.json --output search2.json
```

Командный план пайплайна: [`docs/pipeline.md`](docs/pipeline.md). Методология модели: [`docs/methodology.md`](docs/methodology.md).

## Что уже можно поднять

Сейчас UI и заглушка API работают на мок-контракте [`docs/contracts/api_response.json`](docs/contracts/api_response.json). Живой пайплайн (поиск, модель) подключается командой позже — формат ответа тот же.

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

Если меняли `db/schema.sql` и нужно пересоздать БД с нуля (удалит данные volume):

```bash
docker compose down -v
docker compose up --build -d
```

## Локально только UI (без Docker)

Нужны Python 3.9+ (лучше 3.11) и venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Без `API_URL` интерфейс читает мок-файл. Чтобы ходить в API на хосте:

```bash
export API_URL=http://localhost:8000
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
