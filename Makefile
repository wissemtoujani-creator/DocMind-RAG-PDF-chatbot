.PHONY: help install install-dev run dev test test-cov lint fmt typecheck smoke clean

PY := venv/Scripts/python.exe
ifeq ($(OS),Windows_N/A)
PY := .venv/bin/python
endif

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev:  ## Install runtime + development dependencies
	$(PY) -m pip install -r requirements-dev.txt

run:  ## Start the production server
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:  ## Start the server with autoreload
	$(PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:  ## Run the test suite
	$(PY) -m pytest

test-cov:  ## Run the test suite with a coverage report
	$(PY) -m pytest --cov=app --cov-report=term-missing --cov-report=html

lint:  ## Lint
	$(PY) -m ruff check app tests scripts
	$(PY) -m ruff format --check app tests scripts

fmt:  ## Auto-fix lint and formatting
	$(PY) -m ruff check --fix app tests scripts
	$(PY) -m ruff format app tests scripts

typecheck:  ## Static type check
	$(PY) -m mypy app

smoke:  ## End-to-end check against the live provider stack
	$(PY) scripts/smoke.py

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './venv/*' -exec rm -rf {} +
