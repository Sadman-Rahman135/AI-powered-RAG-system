# syntax=docker/dockerfile:1.7
# DocIntel backend image: API (uvicorn) and worker (celery) share it.
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    EMBEDDING_CACHE_DIR=/opt/models/fastembed \
    STORAGE_DIR=/data/storage \
    PYTHONPATH=/app

# tesseract-ocr: optional OCR engine (OCR_ENGINE=tesseract); libgomp1/libglib2.0-0: onnxruntime / opencv runtime deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

WORKDIR /app
COPY requirements.lock ./
RUN uv pip install --system --no-cache -r requirements.lock

# Bake the OCR and embedding models into the image (no model downloads at runtime).
RUN python -c "from rapidocr import RapidOCR; RapidOCR()" \
    && python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/models/fastembed')"

COPY pyproject.toml alembic.ini ./
COPY alembic ./alembic
COPY app ./app

RUN useradd --create-home --uid 10001 docintel \
    && mkdir -p /data/storage \
    && chown -R docintel:docintel /data /opt/models
USER docintel

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
