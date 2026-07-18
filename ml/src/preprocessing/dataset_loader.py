"""Data ingestion module for FraudLens AI preprocessing pipeline.

This module provides the DatasetLoader class responsible for safely loading
the credit card fraud CSV dataset from disk with comprehensive validation.
"""

import pandas as pd

from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetLoadError(Exception):
    """Raised when the dataset cannot be loaded due to validation or I/O failure."""


class DatasetLoader:
    """Loads the credit card fraud dataset from a CSV file with full validation.

    Validates file existence, extension, readability, and parsability before
    returning a typed DataFrame. Raises descriptive exceptions on any failure.

    Attributes:
        path: Absolute path to the CSV dataset file.
    """

    SUPPORTED_EXTENSION: str = ".csv"

    def __init__(self, path: Path) -> None:
        """Initialises the DatasetLoader with a target file path.

        Args:
            path: The absolute path to the CSV dataset file.
        """
        self.path = path

    def load(self) -> pd.DataFrame:
        """Loads and returns the dataset as a DataFrame.

        Sequentially runs all pre-load checks before attempting to parse the
        CSV. On success, logs dataset shape and returns the DataFrame.

        Returns:
            pd.DataFrame: The loaded credit card fraud dataset.

        Raises:
            DatasetLoadError: If any validation step or parsing fails.
        """
        logger.info(f"Loading dataset from: {self.path}")
        self._check_exists()
        self._check_extension()
        self._check_readable()
        df = self._parse_csv()
        self._check_not_empty(df)
        logger.info(
            f"Dataset loaded successfully — shape: {df.shape}, "
            f"memory: {df.memory_usage(deep=True).sum() / 1_048_576:.2f} MB"
        )
        return df

    # -------------------------------------------------------------------------
    # Private validation helpers
    # -------------------------------------------------------------------------

    def _check_exists(self) -> None:
        """Verifies the file exists at the configured path.

        Raises:
            DatasetLoadError: If the file does not exist.
        """
        if not self.path.exists():
            raise DatasetLoadError(
                f"Dataset file not found: '{self.path}'. "
                "Please download creditcard.csv from Kaggle and place it in ml/data/raw/."
            )
        if not self.path.is_file():
            raise DatasetLoadError(
                f"Path exists but is not a file: '{self.path}'."
            )

    def _check_extension(self) -> None:
        """Verifies the file has the expected .csv extension.

        Raises:
            DatasetLoadError: If the extension is not .csv.
        """
        if self.path.suffix.lower() != self.SUPPORTED_EXTENSION:
            raise DatasetLoadError(
                f"Invalid file extension '{self.path.suffix}'. "
                f"Expected '{self.SUPPORTED_EXTENSION}'."
            )

    def _check_readable(self) -> None:
        """Verifies the file can be opened for reading.

        Raises:
            DatasetLoadError: If the file cannot be read (permissions error).
        """
        try:
            with open(self.path, "r", encoding="utf-8"):
                pass
        except PermissionError as exc:
            raise DatasetLoadError(
                f"Permission denied when reading: '{self.path}'."
            ) from exc
        except OSError as exc:
            raise DatasetLoadError(
                f"OS error when accessing file '{self.path}': {exc}"
            ) from exc

    def _parse_csv(self) -> pd.DataFrame:
        """Parses the CSV file into a DataFrame.

        Returns:
            pd.DataFrame: The parsed DataFrame.

        Raises:
            DatasetLoadError: If the CSV cannot be parsed (malformed content).
        """
        try:
            df: pd.DataFrame = pd.read_csv(self.path)
            return df
        except pd.errors.EmptyDataError as exc:
            raise DatasetLoadError(
                f"Dataset file is empty or contains no data: '{self.path}'."
            ) from exc
        except pd.errors.ParserError as exc:
            raise DatasetLoadError(
                f"CSV parsing failed for '{self.path}'. "
                f"File may be corrupted or malformed: {exc}"
            ) from exc
        except Exception as exc:
            raise DatasetLoadError(
                f"Unexpected error loading '{self.path}': {exc}"
            ) from exc

    def _check_not_empty(self, df: pd.DataFrame) -> None:
        """Verifies the loaded DataFrame contains at least one row.

        Args:
            df: The DataFrame to check.

        Raises:
            DatasetLoadError: If the DataFrame has zero rows.
        """
        if df.empty:
            raise DatasetLoadError(
                f"Dataset loaded from '{self.path}' contains no rows."
            )
