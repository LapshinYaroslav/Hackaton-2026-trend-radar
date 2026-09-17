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
