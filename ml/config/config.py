"""Configuration module for FraudLens AI.

This module defines the project settings and directory paths using Pydantic
Settings. It loads environment variables from a local `.env` file if present,
providing a centralized and type-safe configuration system.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the ml subproject (ml/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# Preprocessing Constants
# ==============================================================================

# Dataset
DATASET_FILENAME: str = "creditcard.csv"
TARGET_COLUMN: str = "Class"

# Columns to scale — V1–V28 are already PCA-transformed, leave untouched
FEATURES_TO_SCALE: list[str] = ["Amount", "Time"]

# All 30 feature columns (Time + V1–V28 + Amount)
ALL_FEATURE_COLUMNS: list[str] = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)

# Data split ratios (must sum to 1.0)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Reproducibility
RANDOM_SEED: int = 42

# SMOTE configuration
SMOTE_SAMPLING_STRATEGY: str = "minority"
SMOTE_K_NEIGHBORS: int = 5

# Scaler identifier saved in metadata
SCALER_TYPE: str = "StandardScaler"

# Dataset version for metadata tracking
DATASET_VERSION: str = "1.0.0"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Attributes:
        PROJECT_NAME: Name of the project.
        VERSION: Version of the project.
        ENV: Execution environment (e.g., development, production).
        RANDOM_SEED: Random seed for reproducibility.
        API_HOST: Host address for the FastAPI server.
        API_PORT: Port number for the FastAPI server.
        API_DEBUG: Flag to enable debug mode in the API.
        DASHBOARD_HOST: Host address for the dashboard.
        DASHBOARD_PORT: Port number for the dashboard.
        LOG_LEVEL: Logging level (e.g., INFO, WARNING, ERROR).
    """

    # Project Metadata
    PROJECT_NAME: str = Field(default="FraudLens AI", description="Name of the project")
    VERSION: str = Field(default="0.1.0", description="Version of the project")
    ENV: str = Field(default="development", description="Execution environment")
    RANDOM_SEED: int = Field(default=42, description="Random seed for reproducibility")

    # API Settings
    API_HOST: str = Field(default="0.0.0.0", description="API Host address")
    API_PORT: int = Field(default=8000, description="API Port number")
    API_DEBUG: bool = Field(default=True, description="API Debug flag")

    # Dashboard Settings
    DASHBOARD_HOST: str = Field(
        default="localhost", description="Dashboard host address"
    )
    DASHBOARD_PORT: int = Field(default=3000, description="Dashboard port number")

    # Logging Settings
    LOG_LEVEL: str = Field(default="INFO", description="Global logging level")

    # Configuration for loading from .env (check both ml/.env and root .env)
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), str(PROJECT_ROOT.parent / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Dynamic Path Resolutions (Read-Only Properties)
    # -------------------------------------------------------------------------

    @property
    def root_dir(self) -> Path:
        """Get the absolute path to the ml subproject root directory.

        Returns:
            Path: The ml project root directory path.
        """
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        """Get the absolute path to the data directory.

        Returns:
            Path: The data directory path.
        """
        return PROJECT_ROOT / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Get the absolute path to the raw data directory.

        Returns:
            Path: The raw data directory path.
        """
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Get the absolute path to the processed data directory.

        Returns:
            Path: The processed data directory path.
        """
        return self.data_dir / "processed"

    @property
    def external_data_dir(self) -> Path:
        """Get the absolute path to the external data directory.

        Returns:
            Path: The external data directory path.
        """
        return self.data_dir / "external"

    @property
    def model_dir(self) -> Path:
        """Get the absolute path to the models directory.

        Returns:
            Path: The models directory path.
        """
        return PROJECT_ROOT / "models"

    @property
    def log_dir(self) -> Path:
        """Get the absolute path to the logs directory.

        Returns:
            Path: The logs directory path.
        """
        return PROJECT_ROOT / "logs"

    @property
    def log_file(self) -> Path:
        """Get the absolute path to the main log file.

        Returns:
            Path: The log file path.
        """
        return self.log_dir / "fraudlens.log"

    @property
    def mlruns_dir(self) -> Path:
        """Get the absolute path to the MLflow runs directory.

        Returns:
            Path: The MLflow runs directory path.
        """
        return PROJECT_ROOT / "mlruns"

    @property
    def reports_dir(self) -> Path:
        """Get the absolute path to the reports directory.

        Returns:
            Path: The reports directory path.
        """
        return PROJECT_ROOT / "reports"

    @property
    def figures_dir(self) -> Path:
        """Get the absolute path to the EDA figures directory.

        Returns:
            Path: The figures output directory path.
        """
        return self.reports_dir / "figures"

    @property
    def artifacts_dir(self) -> Path:
        """Get the absolute path to the ML artifacts directory.

        Returns:
            Path: The artifacts directory path.
        """
        return PROJECT_ROOT / "artifacts"

    @property
    def scalers_dir(self) -> Path:
        """Get the absolute path to the fitted scalers directory.

        Returns:
            Path: The scalers storage directory path.
        """
        return self.artifacts_dir / "scalers"

    @property
    def raw_dataset_path(self) -> Path:
        """Get the absolute path to the raw creditcard CSV dataset.

        Returns:
            Path: The dataset CSV file path.
        """
        return self.raw_data_dir / DATASET_FILENAME


# Global settings singleton
settings = Settings()
