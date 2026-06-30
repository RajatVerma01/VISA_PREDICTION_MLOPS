# ================================================================
# US Visa MLOps — Makefile
# ================================================================
# Usage: make <target>
# ================================================================

.PHONY: help install install-dev train serve test lint \
        docker-build docker-run mlflow clean

PYTHON      := python
PIP         := pip
APP         := app:app
IMAGE_NAME  := usvisa-predictor
PORT        := 8000

# ── Default target ───────────────────────────────────────────────
help:
	@echo ""
	@echo "US Visa MLOps — Available Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install dev + test + lint deps"
	@echo "  make train         Run the full 6-stage training pipeline"
	@echo "  make serve         Start the FastAPI server (hot-reload)"
	@echo "  make test          Run unit tests with coverage"
	@echo "  make lint          Run flake8 linter"
	@echo "  make mlflow        Launch MLflow tracking UI"
	@echo "  make docker-build  Build the Docker image"
	@echo "  make docker-run    Run the Docker container locally"
	@echo "  make clean         Remove __pycache__ and .pyc files"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""

# ── Dependencies ─────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

# ── ML Pipeline ──────────────────────────────────────────────────
train:
	$(PYTHON) demo.py

# ── Web App ──────────────────────────────────────────────────────
serve:
	uvicorn $(APP) --host 0.0.0.0 --port $(PORT) --reload

# ── Testing ──────────────────────────────────────────────────────
test:
	pytest tests/ -v \
		--cov=USvisa \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-x

# ── Linting ──────────────────────────────────────────────────────
lint:
	flake8 USvisa/ app.py \
		--max-line-length=150 \
		--extend-ignore=F403,F405 \
		--exclude=__pycache__,*.egg-info,.git \
		--count \
		--statistics

# ── Observability ────────────────────────────────────────────────
mlflow:
	mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 0.0.0.0 --port 5000

# ── Docker ───────────────────────────────────────────────────────
docker-build:
	docker build -t $(IMAGE_NAME):latest .

docker-run:
	docker run --rm \
		-p $(PORT):$(PORT) \
		--env-file .env \
		-v $(shell pwd)/saved_models:/app/saved_models \
		-v $(shell pwd)/logs:/app/logs \
		$(IMAGE_NAME):latest

# ── Cleanup ──────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	@echo "✅ Clean complete"
