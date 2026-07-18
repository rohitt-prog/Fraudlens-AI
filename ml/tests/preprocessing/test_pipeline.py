"""Integration tests for the PreprocessingPipeline class.

Uses a synthetic CSV fixture (no real dataset required) to verify that all
8 pipeline steps execute without error, produce the correct output files,
and return the expected NumPy arrays.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing.pipeline import PreprocessingPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_csv(tmp_path: Path) -> Path:
    """Creates a realistic synthetic creditcard.csv for pipeline testing.

    Generates 2000 rows with the correct 31-column schema (Time, V1–V28,
    Amount, Class) with a realistic ~1% fraud rate.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Path: Path to the generated CSV file.
    """
    n_normal = 1980
    n_fraud = 20
    n = n_normal + n_fraud
    rng = np.random.default_rng(123)

    data: dict[str, object] = {
        "Time": rng.uniform(0, 172800, n),
        "Amount": rng.uniform(0.01, 5000.0, n),
        "Class": [0] * n_normal + [1] * n_fraud,
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)

    df = pd.DataFrame(data)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / "creditcard.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def pipeline_dirs(tmp_path: Path) -> dict[str, Path]:
    """Returns a dictionary of temp directories for pipeline output.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        dict: Contains 'processed', 'reports', 'figures', 'scalers' paths.
    """
    return {
        "processed": tmp_path / "data" / "processed",
        "reports": tmp_path / "reports",
        "figures": tmp_path / "reports" / "figures",
        "scalers": tmp_path / "artifacts" / "scalers",
    }


@pytest.fixture()
def pipeline(
    synthetic_csv: Path, pipeline_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> PreprocessingPipeline:
    """Returns a PreprocessingPipeline instance wired to temp directories.

    Monkeypatches settings paths to avoid touching the real ml/data/ directory.

    Args:
        synthetic_csv: Path to the synthetic CSV fixture.
        pipeline_dirs: Dictionary of output directories.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        PreprocessingPipeline: Pipeline instance with temp directories.
    """
    from config import config as cfg_module

    monkeypatch.setattr(cfg_module.settings, "reports_dir", pipeline_dirs["reports"], raising=False)
    monkeypatch.setattr(cfg_module.settings, "figures_dir", pipeline_dirs["figures"], raising=False)
    monkeypatch.setattr(cfg_module.settings, "scalers_dir", pipeline_dirs["scalers"], raising=False)

    return PreprocessingPipeline(
        dataset_path=synthetic_csv,
        processed_dir=pipeline_dirs["processed"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreprocessingPipelineE2E:
    """End-to-end integration tests for PreprocessingPipeline.run()."""

    def test_pipeline_runs_without_errors(self, pipeline: PreprocessingPipeline) -> None:
        """Pipeline completes successfully on a valid synthetic dataset."""
        arrays = pipeline.run()
        assert arrays is not None

    def test_returns_all_six_arrays(self, pipeline: PreprocessingPipeline) -> None:
        """Pipeline returns a dict with exactly 6 NumPy array keys."""
        arrays = pipeline.run()
        expected_keys = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
        assert set(arrays.keys()) == expected_keys

    def test_all_returned_values_are_numpy_arrays(
        self, pipeline: PreprocessingPipeline
    ) -> None:
        """All returned values must be NumPy ndarrays."""
        arrays = pipeline.run()
        for key, arr in arrays.items():
            assert isinstance(arr, np.ndarray), f"Expected ndarray for '{key}', got {type(arr)}"

    def test_y_arrays_contain_only_binary_labels(
        self, pipeline: PreprocessingPipeline
    ) -> None:
        """Target arrays y_train, y_val, y_test must only contain 0 and 1."""
        arrays = pipeline.run()
        for key in ("y_train", "y_val", "y_test"):
            unique = set(arrays[key].tolist())
            assert unique.issubset({0, 1}), f"{key} contains unexpected labels: {unique}"

    def test_x_feature_count_is_correct(self, pipeline: PreprocessingPipeline) -> None:
        """X arrays must have 30 feature columns (Time + V1-V28 + Amount)."""
        arrays = pipeline.run()
        for key in ("X_train", "X_val", "X_test"):
            assert arrays[key].shape[1] == 30, (
                f"{key} has {arrays[key].shape[1]} features, expected 30"
            )

    def test_train_is_larger_than_val_and_test(
        self, pipeline: PreprocessingPipeline
    ) -> None:
        """Training split (post-SMOTE) must be larger than val and test."""
        arrays = pipeline.run()
        assert len(arrays["y_train"]) > len(arrays["y_val"])
        assert len(arrays["y_train"]) > len(arrays["y_test"])

    def test_fraud_class_present_in_all_splits(
        self, pipeline: PreprocessingPipeline
    ) -> None:
        """Fraud class must be present in all splits after stratified split."""
        arrays = pipeline.run()
        assert arrays["y_train"].sum() > 0, "No fraud in train"
        assert arrays["y_val"].sum() > 0, "No fraud in val"
        assert arrays["y_test"].sum() > 0, "No fraud in test"


class TestPreprocessingPipelineOutputFiles:
    """Tests that the pipeline saves all expected output files."""

    def test_processed_numpy_files_are_saved(
        self,
        pipeline: PreprocessingPipeline,
        pipeline_dirs: dict[str, Path],
    ) -> None:
        """Pipeline saves 6 .npy files to the processed data directory."""
        pipeline.run()
        processed = pipeline_dirs["processed"]
        expected_files = [
            "X_train.npy", "X_val.npy", "X_test.npy",
            "y_train.npy", "y_val.npy", "y_test.npy",
        ]
        for filename in expected_files:
            assert (processed / filename).exists(), f"Missing: {filename}"

    def test_scaler_file_is_saved(
        self,
        pipeline: PreprocessingPipeline,
        pipeline_dirs: dict[str, Path],
    ) -> None:
        """Pipeline saves the fitted scaler to ml/artifacts/scalers/."""
        pipeline.run()
        scaler_path = pipeline_dirs["scalers"] / "amount_time_scaler.pkl"
        assert scaler_path.exists(), "Scaler file not saved"

    def test_npy_files_are_loadable(
        self,
        pipeline: PreprocessingPipeline,
        pipeline_dirs: dict[str, Path],
    ) -> None:
        """Saved .npy files can be loaded back and match the returned arrays."""
        arrays = pipeline.run()
        processed = pipeline_dirs["processed"]
        for name, arr in arrays.items():
            loaded = np.load(processed / f"{name}.npy")
            np.testing.assert_array_equal(loaded, arr, err_msg=f"Mismatch for {name}")
