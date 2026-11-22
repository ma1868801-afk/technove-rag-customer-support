# TechNova WhatsApp support assistant -- production container.
#
# Build:  docker build -t technova-rag .
# Run:    docker run -p 8080:8080 --env-file .env -v technova_chroma:/app/chroma_db technova-rag
#
# The embedding + reranker models (~200MB total) download from
# huggingface.co on first startup and are cached in the image layer /
# mounted volume afterwards -- make sure the container has outbound
# network access at least once.

FROM python:3.12-slim

WORKDIR /app

# Minimal build deps for sentence-transformers/torch wheels on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()" || exit 1

# One worker: local-mode Chroma + the SQLite checkpointer are both
# single-writer. Scale horizontally behind a load balancer with
# CHROMA_MODE=server (a standalone `chroma run` instance) once you need
# more throughput than one process gives you -- see README "Scaling past
# one process".
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
