FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    TIKTOKEN_CACHE_DIR=/opt/models/tiktoken \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed

# Tesseract: primary OCR engine for scanned pages
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake models into the image: no network downloads at startup
COPY scripts/prefetch_models.py scripts/
RUN python scripts/prefetch_models.py

COPY app ./app

EXPOSE 8000

# /health is 503 until ingestion finishes; allow the full 10-minute window
HEALTHCHECK --interval=10s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
