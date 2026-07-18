# FraudLens AI - Backend REST API Service

This is the FastAPI backend service for the FraudLens AI system. It provides REST API endpoints to perform real-time fraud transaction scoring and return explainable model features.

## Directory Structure

```
backend/
├── app/                 # FastAPI application module
│   ├── api/             # API routes and request controllers
│   ├── core/            # App configurations, secrets, and DB setups
│   ├── schemas/         # Request and response models (Pydantic)
│   ├── services/        # Scoring orchestrations and business services
│   ├── utils/           # Shared API helpers (loggers, responses)
│   └── main.py          # FastAPI application entry point
├── tests/               # API endpoint unit and integration tests
├── Dockerfile           # Backend container blueprint
├── requirements.txt     # Python requirements for backend
└── pyproject.toml       # Linter and test tool configurations
```

## Setup and Run

To set up the development environment, navigate to the `backend/` subproject directory:

```bash
cd backend

# Virtual environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the FastAPI server locally:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Running Checks

```bash
# Format check
black --check .
isort --check-only .

# Linting
ruff check .

# Type checking
mypy .

# Run tests
pytest
```
