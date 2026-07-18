# FraudLens AI - Machine Learning Service

This directory contains the machine learning components for FraudLens AI, including data preprocessing, feature engineering, model training, evaluation, explainability, and local batch/online inference.

## Directory Structure

```
ml/
├── artifacts/           # Serialized encoders, scalers, metadata (Gitignored)
│   ├── scalers/         # Fitted feature scaling objects
│   ├── thresholds/      # Tuned classification thresholds
│   ├── encoders/        # Categorical encoders (e.g. Target / One-Hot)
│   └── metadata/        # Model configuration and training run metadata
├── config/              # Centralized configuration (Pydantic Settings)
├── data/                # Data storage directories (Gitignored)
│   ├── raw/             # Unprocessed raw transaction data
│   ├── processed/       # Cleaned datasets and engineered features
│   └── external/        # External side-car data (GeoIP, IP logs)
├── models/              # Saved model weights (Gitignored)
├── notebooks/           # Research, EDA, and validation Jupyter notebooks
├── reports/             # Generated training reports and figures
├── mlruns/              # Local MLflow experiments tracker (Gitignored)
├── scripts/             # Infrastructure and deployment automation
├── src/                 # Main Python package source code
│   ├── preprocessing/   # Data loading and raw data cleaning pipelines
│   ├── feature_engineering/ # Feature extraction, selection, and generation
│   ├── training/        # ML model training loops and hyperparameters tuning
│   ├── explainability/  # XAI interpretability calculations (SHAP)
│   ├── inference/       # Offline/Online inference wrapper APIs
│   ├── evaluation/      # Model scoring, validation, and testing metrics
│   ├── visualization/   # EDA plots and model evaluation figures
│   ├── services/        # ML workflow and orchestration services
│   └── utils/           # Shared logging, seeding, and path utilities
├── tests/               # Pytest suite for ML workflows
├── Dockerfile           # ML training container blueprint
├── requirements.txt     # Python requirements for ML
└── pyproject.toml       # Linter and test tool configurations
```

## Setup and Run

To set up the development environment, navigate to the `ml/` subproject directory:

```bash
cd ml

# Virtual environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
