"""Configuration module for FraudLens AI.

This module defines the project settings and directory paths using Pydantic
Settings. It loads environment variables from a local `.env` file if present,
providing a centralized and type-safe configuration system.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

    # Dynamic Path Resolutions (Read-Only Properties)
    @property
    def root_dir(self) -> Path:
        """Get the absolute path to the project root directory.

        Returns:
            Path: The project root directory path.
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


# Global settings instance
settings = Settings()
