"""Tests for the DataPreprocessor class.

Covers: correct columns are scaled, V1-V28 left untouched,
scaler is saved to disk, transform raises before fit, load_scaler works.
"""

import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing.preprocessor import DataPreprocessor, SCALER_FILENAME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Returns a synthetic DataFrame with all expected columns.

    Returns:
        pd.DataFrame: A synthetic creditcard-like DataFrame.
    """
    n = 300
    rng = np.random.default_rng(0)
    data: dict[str, object] = {
        "Time": rng.uniform(0, 172800, n),
        "Amount": rng.uniform(0.01, 25691.0, n),
        "Class": rng.integers(0, 2, n),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)
    return pd.DataFrame(data)


@pytest.fixture()
def preprocessor(tmp_path: Path) -> DataPreprocessor:
    """Returns a DataPreprocessor with a temp scaler path.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        DataPreprocessor: Configured instance with temp scaler path.
    """
    return DataPreprocessor(scaler_path=tmp_path / "scalers" / SCALER_FILENAME)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataPreprocessorFitTransform:
    """Tests for the fit_transform method."""

    def test_scales_amount_and_time_columns(
        self, sample_df: pd.DataFrame, preprocessor: DataPreprocessor
    ) -> None:
        """After fit_transform, Amount and Time have approximately 0 mean and 1 std."""
        df_out = preprocessor.fit_transform(sample_df)
        assert abs(df_out["Amount"].mean()) < 0.1
        assert abs(df_out["Amount"].std() - 1.0) < 0.1
        assert abs(df_out["Time"].mean()) < 0.1
        assert abs(df_out["Time"].std() - 1.0) < 0.1

    def test_v_features_are_untouched(
        self, sample_df: pd.DataFrame, preprocessor: DataPreprocessor
    ) -> None:
        """V1–V28 columns must have identical values before and after preprocessing."""
        df_out = preprocessor.fit_transform(sample_df)
        for i in range(1, 29):
            col = f"V{i}"
            pd.testing.assert_series_equal(
                sample_df[col].reset_index(drop=True),
                df_out[col].reset_index(drop=True),
                check_names=False,
                obj=f"Column {col} was incorrectly modified",
            )

    def test_class_column_is_untouched(
        self, sample_df: pd.DataFrame, preprocessor: DataPreprocessor
    ) -> None:
        """The Class column must remain unchanged after preprocessing."""
        df_out = preprocessor.fit_transform(sample_df)
        pd.testing.assert_series_equal(
            sample_df["Class"].reset_index(drop=True),
            df_out["Class"].reset_index(drop=True),
        )

    def test_returns_dataframe_with_same_shape(
        self, sample_df: pd.DataFrame, preprocessor: DataPreprocessor
    ) -> None:
        """fit_transform must return a DataFrame with the same shape as input."""
        df_out = preprocessor.fit_transform(sample_df)
        assert df_out.shape == sample_df.shape

    def test_raises_on_missing_scale_columns(
        self, preprocessor: DataPreprocessor
    ) -> None:
        """fit_transform raises ValueError if Amount or Time columns are absent."""
        df_bad = pd.DataFrame({"V1": [1.0, 2.0], "Class": [0, 1]})
        with pytest.raises(ValueError, match="required for scaling"):
            preprocessor.fit_transform(df_bad)


class TestDataPreprocessorPersistence:
    """Tests for scaler saving and loading."""

    def test_scaler_is_saved_to_disk(
        self, sample_df: pd.DataFrame, preprocessor: DataPreprocessor
    ) -> None:
        """After fit_transform, a scaler .pkl file exists at the configured path."""
        preprocessor.fit_transform(sample_df)
        assert preprocessor.scaler_path.exists()

    def test_saved_scaler_is_valid_pickle(
        self, sample_df: pd.DataFrame, preprocessor: DataPreprocessor
    ) -> None:
        """The saved scaler file must be a loadable StandardScaler pickle."""
        from sklearn.preprocessing import StandardScaler
        preprocessor.fit_transform(sample_df)
        with open(preprocessor.scaler_path, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, StandardScaler)

    def test_load_scaler_raises_if_not_found(self, tmp_path: Path) -> None:
        """load_scaler raises FileNotFoundError when no scaler file is present."""
        with pytest.raises(FileNotFoundError, match="not found"):
            DataPreprocessor.load_scaler(scaler_path=tmp_path / "missing.pkl")

    def test_transform_raises_before_fit(
        self, sample_df: pd.DataFrame, preprocessor: DataPreprocessor
    ) -> None:
        """transform raises RuntimeError when called without prior fitting."""
        with pytest.raises(RuntimeError, match="not been fitted"):
            preprocessor.transform(sample_df)


class TestDataPreprocessorParams:
    """Tests for scaler parameter introspection."""

    def test_get_scaler_params_returns_expected_keys(
        self, sample_df: pd.DataFrame, preprocessor: DataPreprocessor
    ) -> None:
        """get_scaler_params returns 'features', 'mean', and 'scale' keys."""
        preprocessor.fit_transform(sample_df)
        params = preprocessor.get_scaler_params()
        assert "features" in params
        assert "mean" in params
        assert "scale" in params
        assert len(params["mean"]) == 2  # Amount and Time only
