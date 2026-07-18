"""Tests for the DataSplitter class.

Covers: ratio validation, correct split sizes, stratification preservation,
SplitResult shape, and error handling for bad inputs.
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing.splitter import DataSplitter, SplitResult
from config.config import ALL_FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def scaled_df() -> pd.DataFrame:
    """Returns a scaled synthetic DataFrame suitable for splitting.

    Returns:
        pd.DataFrame: Synthetic creditcard-like DataFrame with 1000 rows.
    """
    n = 1000
    rng = np.random.default_rng(99)
    data: dict[str, object] = {
        "Time": rng.standard_normal(n),
        "Amount": rng.standard_normal(n),
        "Class": [1 if i < 20 else 0 for i in range(n)],  # 2% fraud
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)
    return pd.DataFrame(data)


@pytest.fixture()
def splitter() -> DataSplitter:
    """Returns a default DataSplitter instance.

    Returns:
        DataSplitter: A splitter with default ratios.
    """
    return DataSplitter()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataSplitterRatioValidation:
    """Tests for split ratio validation."""

    def test_raises_when_ratios_do_not_sum_to_one(self) -> None:
        """DataSplitter raises ValueError when ratios don't sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            DataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.3)

    def test_valid_ratios_do_not_raise(self) -> None:
        """DataSplitter accepts valid ratios that sum to 1.0."""
        splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert splitter is not None


class TestDataSplitterOutputShapes:
    """Tests for output array shapes from split()."""

    def test_returns_split_result_dataclass(
        self, scaled_df: pd.DataFrame, splitter: DataSplitter
    ) -> None:
        """split() returns a SplitResult dataclass instance."""
        result = splitter.split(scaled_df)
        assert isinstance(result, SplitResult)

    def test_train_val_test_cover_full_dataset(
        self, scaled_df: pd.DataFrame, splitter: DataSplitter
    ) -> None:
        """Total rows across all splits must equal the original dataset size."""
        result = splitter.split(scaled_df)
        total = (
            len(result.y_train) + len(result.y_val) + len(result.y_test)
        )
        assert total == len(scaled_df)

    def test_train_split_is_largest(
        self, scaled_df: pd.DataFrame, splitter: DataSplitter
    ) -> None:
        """Training split must have more rows than val and test."""
        result = splitter.split(scaled_df)
        assert len(result.y_train) > len(result.y_val)
        assert len(result.y_train) > len(result.y_test)

    def test_val_test_are_approximately_equal(
        self, scaled_df: pd.DataFrame, splitter: DataSplitter
    ) -> None:
        """Validation and test splits should have approximately the same size."""
        result = splitter.split(scaled_df)
        size_diff = abs(len(result.y_val) - len(result.y_test))
        assert size_diff <= 5  # Allow small rounding differences

    def test_feature_matrix_shapes_match_labels(
        self, scaled_df: pd.DataFrame, splitter: DataSplitter
    ) -> None:
        """X and y arrays must have the same number of rows for each split."""
        result = splitter.split(scaled_df)
        assert result.X_train.shape[0] == result.y_train.shape[0]
        assert result.X_val.shape[0] == result.y_val.shape[0]
        assert result.X_test.shape[0] == result.y_test.shape[0]


class TestDataSplitterStratification:
    """Tests that stratification preserves class ratios across splits."""

    def test_fraud_is_present_in_all_splits(
        self, scaled_df: pd.DataFrame, splitter: DataSplitter
    ) -> None:
        """All three splits must contain at least one fraud sample (stratified)."""
        result = splitter.split(scaled_df)
        assert result.y_train.sum() > 0, "No fraud in train split"
        assert result.y_val.sum() > 0, "No fraud in val split"
        assert result.y_test.sum() > 0, "No fraud in test split"

    def test_fraud_ratio_is_similar_across_splits(
        self, scaled_df: pd.DataFrame, splitter: DataSplitter
    ) -> None:
        """Fraud percentage should be approximately equal across all splits."""
        result = splitter.split(scaled_df)
        train_rate = result.y_train.sum() / len(result.y_train)
        val_rate = result.y_val.sum() / len(result.y_val)
        test_rate = result.y_test.sum() / len(result.y_test)
        # All rates should be within 2% of each other
        assert abs(train_rate - val_rate) < 0.03
        assert abs(train_rate - test_rate) < 0.03


class TestDataSplitterErrorHandling:
    """Tests for error conditions in DataSplitter."""

    def test_raises_on_missing_target_column(self, splitter: DataSplitter) -> None:
        """DataSplitter raises ValueError if the target column is absent."""
        df = pd.DataFrame({"V1": [1.0, 2.0], "Amount": [10.0, 20.0]})
        with pytest.raises(ValueError, match=TARGET_COLUMN):
            splitter.split(df)

    def test_raises_on_empty_feature_columns(self, splitter: DataSplitter) -> None:
        """DataSplitter raises ValueError if no feature columns are found."""
        df = pd.DataFrame({"Class": [0, 1, 0, 1, 0]})
        with pytest.raises(ValueError, match="feature columns"):
            splitter.split(df)
