# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps de sistem (hnswlib/healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# dependențe Python
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# codul aplicației
COPY . .

# utilizator non-root
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/metrics || exit 1

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
