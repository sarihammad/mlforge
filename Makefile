.PHONY: install train register serve test drift-check docker-up docker-down clean mlflow-ui lint format

PYTHON := python
DATA_DIR := data
EXPERIMENT := adult-income

# ── Environment ────────────────────────────────────────────────────────────

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed."

# ── Training ───────────────────────────────────────────────────────────────

train:
	PYTHONPATH=. $(PYTHON) -m src.pipelines.train_pipeline \
		--experiment $(EXPERIMENT) \
		--output-dir $(DATA_DIR)

train-register:
	PYTHONPATH=. $(PYTHON) -m src.pipelines.train_pipeline \
		--experiment $(EXPERIMENT) \
		--output-dir $(DATA_DIR) \
		--register \
		--promote-staging

register:
	@echo "Re-run with --register flag to register the last trained model."
	PYTHONPATH=. $(PYTHON) -m src.pipelines.train_pipeline \
		--experiment $(EXPERIMENT) \
		--data-dir $(DATA_DIR) \
		--load-splits \
		--register \
		--promote-staging

# ── Serving ────────────────────────────────────────────────────────────────

serve:
	PYTHONPATH=. uvicorn src.serving.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

# ── Drift & Retraining ─────────────────────────────────────────────────────

drift-check:
	PYTHONPATH=. $(PYTHON) -m src.pipelines.retrain_pipeline \
		--reference-data $(DATA_DIR)/X_train.parquet \
		--current-data $(DATA_DIR)/X_val.parquet

retrain:
	PYTHONPATH=. $(PYTHON) -m src.pipelines.retrain_pipeline \
		--reference-data $(DATA_DIR)/X_train.parquet \
		--current-data $(DATA_DIR)/X_val.parquet \
		--auto-promote \
		--data-dir $(DATA_DIR)

# ── Testing ────────────────────────────────────────────────────────────────

test:
	PYTHONPATH=. pytest tests/ -v --tb=short

test-coverage:
	PYTHONPATH=. pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── MLflow UI ──────────────────────────────────────────────────────────────

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000 &
	@echo "MLflow UI available at http://localhost:5000"

# ── Docker ─────────────────────────────────────────────────────────────────

docker-up:
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@echo "MLflow:     http://localhost:5000"
	@echo "API:        http://localhost:8000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000 (admin/admin)"

docker-down:
	docker compose down

docker-down-clean:
	docker compose down -v --remove-orphans

docker-logs:
	docker compose logs -f api

# ── Code quality ───────────────────────────────────────────────────────────

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

format:
	black src/ tests/ --line-length=100
	isort src/ tests/

# ── Clean ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
	@echo "Cleaned."
