# ================================================================
# US Visa Approval Predictor — Production Docker Image
# ================================================================
# Multi-stage aware: uses python:3.10-slim for a lean image.
# Runs as a non-root user for security best practices.
# ================================================================

FROM python:3.10-slim

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ─────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser

# ── Set working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ──────────────────────────────────────
COPY . .

# ── Re-install the package in editable mode ───────────────────────
RUN pip install --no-cache-dir -e .

# ── Create necessary runtime directories ─────────────────────────
RUN mkdir -p saved_models logs artifact static && \
    chown -R appuser:appuser /app

# ── Switch to non-root user ──────────────────────────────────────
USER appuser

# ── Expose port ──────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Start server ─────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
