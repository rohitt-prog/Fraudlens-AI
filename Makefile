# ==============================================================================
# FraudLens AI - Development Automation
# ==============================================================================

.PHONY: setup format lint test clean docker-build docker-up docker-down help

.DEFAULT_GOAL := help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup: ## Create virtual environment and install dependencies
	@echo "Setting up python virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete. To activate the environment run: source $(VENV)/bin/activate"

format: ## Format source files using Black and isort
	@echo "Running Black formatter..."
	$(PYTHON) -m black .
	@echo "Running isort..."
	$(PYTHON) -m isort .

lint: ## Lint files using Ruff and mypy
	@echo "Running Ruff check..."
	$(PYTHON) -m ruff check .
	@echo "Running mypy type check..."
	$(PYTHON) -m mypy .

test: ## Run unit tests with pytest
	@echo "Running pytest..."
	$(PYTHON) -m pytest

clean: ## Remove caches, build assets, and log files
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Caches cleaned."

docker-build: ## Build docker image
	docker compose build

docker-up: ## Start docker container services
	docker compose up -d

docker-down: ## Stop docker container services
	docker compose down

help: ## Show this help message
	@echo "Available Makefile commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
