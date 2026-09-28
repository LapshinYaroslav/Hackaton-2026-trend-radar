FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/api/requirements.txt

COPY api /app/api
COPY db /app/db
COPY labels /app/labels
COPY data/interim /app/data/interim
COPY evidence /app/evidence
COPY docs /app/docs
COPY collector /app/collector
COPY model /app/model
COPY pipeline /app/pipeline
COPY search /app/search

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
