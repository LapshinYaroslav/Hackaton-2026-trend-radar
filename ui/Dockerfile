FROM python:3.11-slim

WORKDIR /app

COPY ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Структура как в репо: ui/ (app.py, view.py, logo.svg) и тема .streamlit/
COPY ui /app/ui
COPY .streamlit /app/.streamlit

EXPOSE 8501

# 0.0.0.0 — слушаем все интерфейсы контейнера (иначе с хоста не открыть)
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
