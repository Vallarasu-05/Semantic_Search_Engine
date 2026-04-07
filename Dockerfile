# ─────────────────────────────────────────────────────────────────
# Semantic Search Engine — Dockerfile
# Base: python:3.11-slim  (keeps image lean for Render free tier)
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data during build (avoids runtime download)
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True)"

# Copy app source
COPY app/     ./app/
COPY static/  ./static/

# Render injects PORT env var; default 8000
ENV PORT=8000

# Artifacts cache dir (persisted between Render deploys via disk)
ENV ARTIFACTS_DIR=/tmp/semantic_engine

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120"]
