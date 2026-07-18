"""Data splitting module for FraudLens AI preprocessing pipeline.

This module provides the DataSplitter class, which creates stratified
train/validation/test splits and logs class distributions at every stage.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    ALL_FEATURE_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SplitResult:
    """Container for the six arrays produced by the stratified split.

    Attributes:
        X_train: Feature matrix for training.
        X_val: Feature matrix for validation.
        X_test: Feature matrix for testing.
        y_train: Target vector for training.
        y_val: Target vector for validation.
        y_test: Target vector for testing.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


class DataSplitter:
    """Creates stratified train/validation/test splits from the processed DataFrame.

    Uses sklearn's train_test_split with stratify=y to maintain the original
    class ratio in every split, preventing data leakage between phases.

    Attributes:
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        random_seed: Seed for reproducibility.
    """

    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        """Initialises the DataSplitter with configurable ratios.

        Args:
            train_ratio: Fraction of data for training. Default 0.70.
            val_ratio: Fraction of data for validation. Default 0.15.
            test_ratio: Fraction of data for testing. Default 0.15.
            random_seed: Random seed for reproducible splits. Default 42.

        Raises:
            ValueError: If ratios do not sum to approximately 1.0.
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self._validate_ratios()

    def split(self, df: pd.DataFrame) -> SplitResult:
        """Performs a stratified two-stage split into train/val/test sets.

        Stage 1: Split full data into train and temp (val + test).
        Stage 2: Split temp into val and test.

        Args:
            df: Processed DataFrame containing feature and target columns.

        Returns:
            SplitResult: Dataclass holding the six NumPy arrays.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        self._validate_dataframe(df)
        logger.info(
            f"Splitting dataset — train: {self.train_ratio:.0%}, "
            f"val: {self.val_ratio:.0%}, test: {self.test_ratio:.0%} "
            f"(stratified, seed={self.random_seed})"
        )

        feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
        X: np.ndarray = df[feature_cols].values
        y: np.ndarray = df[TARGET_COLUMN].values

        # Stage 1: train vs (val + test)
        temp_size = self.val_ratio + self.test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=temp_size,
            stratify=y,
            random_state=self.random_seed,
        )

        # Stage 2: val vs test from temp
        test_relative = self.test_ratio / temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=test_relative,
            stratify=y_temp,
            random_state=self.random_seed,
        )

        result = SplitResult(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
        self._log_distributions(result)
        return result

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _validate_ratios(self) -> None:
        """Validates that the provided split ratios sum to 1.0.

        Raises:
            ValueError: If the sum deviates by more than 1e-6.
        """
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.6f}. "
                f"Provided: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}."
            )

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validates that the DataFrame contains the required columns.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If target column or feature columns are missing.
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")
        feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
        if len(feature_cols) == 0:
            raise ValueError(
                "No feature columns found in DataFrame. "
                f"Expected columns from ALL_FEATURE_COLUMNS."
            )

    def _log_distributions(self, result: SplitResult) -> None:
        """Logs class distributions and row counts for all three splits.

        Args:
            result: The SplitResult containing all six arrays.
        """
        for name, X, y in [
            ("Train", result.X_train, result.y_train),
            ("Validation", result.X_val, result.y_val),
            ("Test", result.X_test, result.y_test),
        ]:
            total = len(y)
            fraud = int(y.sum())
            normal = total - fraud
            logger.info(
                f"  {name:12s}: {total:>7,} rows | "
                f"Normal={normal:,} ({normal / total:.2%}), "
                f"Fraud={fraud:,} ({fraud / total:.2%})"
            )
