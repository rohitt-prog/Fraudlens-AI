"""Data preprocessing module for FraudLens AI.

This module provides the DataPreprocessor class, which fits a StandardScaler
on the Amount and Time columns only. V1–V28 features are PCA-transformed by
the dataset provider and must not be re-scaled.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.config import FEATURES_TO_SCALE, TARGET_COLUMN, settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Filename for the persisted scaler object
SCALER_FILENAME: str = "amount_time_scaler.pkl"


class DataPreprocessor:
    """Fits and applies StandardScaler to Amount and Time columns.

    V1–V28 are already PCA-transformed by Kaggle and must remain unmodified.
    The fitted scaler is persisted to ml/artifacts/scalers/ for use during
    inference without re-fitting.

    Attributes:
        scaler_path: Path where the fitted scaler will be saved.
        _scaler: The fitted StandardScaler instance (None before fitting).
    """

    def __init__(self, scaler_path: Path | None = None) -> None:
        """Initialises the DataPreprocessor.

        Args:
            scaler_path: Optional path override for the saved scaler file.
                Defaults to ml/artifacts/scalers/amount_time_scaler.pkl.
        """
        self.scaler_path = scaler_path or (settings.scalers_dir / SCALER_FILENAME)
        self._scaler: StandardScaler | None = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the scaler on Amount and Time, transforms them in-place.

        Returns a new DataFrame with Amount and Time standardised. All other
        columns (V1–V28, Class) remain unchanged.

        Args:
            df: The raw or validated DataFrame containing all feature columns.

        Returns:
            pd.DataFrame: A copy of the DataFrame with scaled Amount and Time.

        Raises:
            ValueError: If any of the required scaling columns are not found.
        """
        self._validate_columns(df)
        logger.info(
            f"Fitting StandardScaler on columns: {FEATURES_TO_SCALE}. "
            f"V1–V28 left untouched."
        )
        df_out = df.copy()
        self._scaler = StandardScaler()
        df_out[FEATURES_TO_SCALE] = self._scaler.fit_transform(df[FEATURES_TO_SCALE])

        self._log_scaling_stats(df, df_out)
        self._save_scaler()
        return df_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the already-fitted scaler to new data (e.g., at inference).

        Args:
            df: DataFrame with Amount and Time columns to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.

        Raises:
            RuntimeError: If called before fit_transform.
            ValueError: If required columns are missing.
        """
        if self._scaler is None:
            raise RuntimeError(
                "Scaler has not been fitted. Call fit_transform() first "
                "or load a persisted scaler with load_scaler()."
            )
        self._validate_columns(df)
        df_out = df.copy()
        df_out[FEATURES_TO_SCALE] = self._scaler.transform(df[FEATURES_TO_SCALE])
        return df_out

    @classmethod
    def load_scaler(cls, scaler_path: Path | None = None) -> "DataPreprocessor":
        """Loads a previously saved scaler from disk.

        Args:
            scaler_path: Optional path override for the scaler file.

        Returns:
            DataPreprocessor: Instance with a pre-loaded scaler ready for transform().

        Raises:
            FileNotFoundError: If the scaler file is not found.
        """
        path = scaler_path or (settings.scalers_dir / SCALER_FILENAME)
        if not path.exists():
            raise FileNotFoundError(
                f"Scaler file not found at '{path}'. "
                "Run the preprocessing pipeline first to generate it."
            )
        instance = cls(scaler_path=path)
        with open(path, "rb") as f:
            instance._scaler = pickle.load(f)
        logger.info(f"Scaler loaded from: {path}")
        return instance

    def get_scaler_params(self) -> dict[str, list[float]]:
        """Returns the fitted scaler's mean and standard deviation parameters.

        Returns:
            dict: Dictionary with 'mean' and 'scale' lists (one per feature).

        Raises:
            RuntimeError: If called before fit_transform.
        """
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted yet.")
        return {
            "features": FEATURES_TO_SCALE,
            "mean": self._scaler.mean_.tolist(),
            "scale": self._scaler.scale_.tolist(),
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validates that all columns to scale exist in the DataFrame.

        Args:
            df: The DataFrame to check.

        Raises:
            ValueError: If any required column is missing.
        """
        missing = [c for c in FEATURES_TO_SCALE if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns required for scaling not found in DataFrame: {missing}"
            )

    def _log_scaling_stats(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
        """Logs before/after statistics for scaled columns.

        Args:
            df_before: DataFrame before scaling.
            df_after: DataFrame after scaling.
        """
        for col in FEATURES_TO_SCALE:
            logger.info(
                f"  {col}: "
                f"before=[mean={df_before[col].mean():.4f}, std={df_before[col].std():.4f}] → "
                f"after=[mean={df_after[col].mean():.6f}, std={df_after[col].std():.6f}]"
            )

    def _save_scaler(self) -> None:
        """Serialises the fitted StandardScaler to disk using pickle.

        Raises:
            RuntimeError: If the scaler has not been fitted.
            PermissionError: If the target directory is not writable.
        """
        if self._scaler is None:
            raise RuntimeError("Cannot save — scaler has not been fitted.")
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info(f"Fitted scaler saved to: {self.scaler_path}")
