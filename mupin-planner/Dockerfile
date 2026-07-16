FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm_config.yaml .
COPY module_registry.yaml .
COPY src/ ./src/

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]