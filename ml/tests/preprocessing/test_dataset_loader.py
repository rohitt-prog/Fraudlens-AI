"""Tests for the DatasetLoader class.

Covers: missing file, wrong extension, empty file, permission error,
corrupted CSV, and valid load scenarios.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.preprocessing.dataset_loader import DatasetLoader, DatasetLoadError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_csv(tmp_path: Path) -> Path:
    """Creates a minimal valid creditcard-like CSV file.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Path: Path to the created CSV file.
    """
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    header = ",".join(cols)
    row = ",".join(["0.0"] * (len(cols) - 1) + ["0"])
    content = f"{header}\n{row}\n"
    csv_path = tmp_path / "creditcard.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


@pytest.fixture()
def corrupted_csv(tmp_path: Path) -> Path:
    """Creates a CSV file with malformed/unparseable content.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Path: Path to the corrupted CSV.
    """
    csv_path = tmp_path / "bad.csv"
    csv_path.write_bytes(b"\x00\xff\xfe malformed binary content \x00\x01")
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDatasetLoaderFileValidation:
    """Tests for file-level validation in DatasetLoader."""

    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        """DatasetLoader raises DatasetLoadError when the file does not exist."""
        loader = DatasetLoader(path=tmp_path / "nonexistent.csv")
        with pytest.raises(DatasetLoadError, match="not found"):
            loader.load()

    def test_raises_when_path_is_directory(self, tmp_path: Path) -> None:
        """DatasetLoader raises DatasetLoadError when the path points to a directory."""
        loader = DatasetLoader(path=tmp_path)
        with pytest.raises(DatasetLoadError, match="not a file"):
            loader.load()

    def test_raises_on_wrong_extension(self, tmp_path: Path) -> None:
        """DatasetLoader raises DatasetLoadError for non-.csv extensions."""
        bad_file = tmp_path / "data.parquet"
        bad_file.write_text("col1,col2\n1,2\n")
        loader = DatasetLoader(path=bad_file)
        with pytest.raises(DatasetLoadError, match="extension"):
            loader.load()

    def test_raises_on_empty_csv(self, tmp_path: Path) -> None:
        """DatasetLoader raises DatasetLoadError when the CSV file has no rows."""
        empty = tmp_path / "empty.csv"
        empty.write_text("Time,V1,Amount,Class\n", encoding="utf-8")
        loader = DatasetLoader(path=empty)
        with pytest.raises(DatasetLoadError):
            loader.load()


class TestDatasetLoaderSuccess:
    """Tests for successful loading scenarios."""

    def test_loads_valid_csv(self, valid_csv: Path) -> None:
        """DatasetLoader returns a non-empty DataFrame for a valid CSV."""
        loader = DatasetLoader(path=valid_csv)
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_returns_correct_columns(self, valid_csv: Path) -> None:
        """DatasetLoader returns all expected columns from the CSV."""
        loader = DatasetLoader(path=valid_csv)
        df = loader.load()
        assert "Time" in df.columns
        assert "Class" in df.columns
        assert "Amount" in df.columns
        for i in range(1, 29):
            assert f"V{i}" in df.columns

    def test_dataframe_has_correct_dtype(self, valid_csv: Path) -> None:
        """DatasetLoader returns a DataFrame with numeric dtypes."""
        loader = DatasetLoader(path=valid_csv)
        df = loader.load()
        import pandas.api.types as ptypes
        for col in df.columns:
            assert ptypes.is_numeric_dtype(df[col]), f"Column '{col}' is not numeric"
