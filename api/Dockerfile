FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt \
    pandas==2.2.3 numpy==2.1.3 scikit-learn==1.5.2 joblib==1.4.2 httpx==0.28.1

COPY collector /app/collector
COPY model /app/model
COPY search /app/search
COPY pipeline /app/pipeline
COPY evidence/source_totals_training.json /app/evidence/source_totals_training.json
COPY api /app/api
COPY docs/contracts /app/docs/contracts

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
