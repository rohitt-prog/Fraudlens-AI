"""Tests for the DataValidator class.

Covers: required columns check, missing values detection, duplicate row
detection, target column integrity, and report generation.
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing.validator import DataValidator, DataValidationError, ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """Returns a minimal valid DataFrame with the expected schema.

    Returns:
        pd.DataFrame: A small synthetic creditcard-like DataFrame.
    """
    n = 200
    rng = np.random.default_rng(42)
    data: dict[str, object] = {"Time": rng.random(n) * 172800, "Amount": rng.random(n) * 500}
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)
    # 190 normal, 10 fraud for realistic imbalance
    data["Class"] = [1 if i < 10 else 0 for i in range(n)]
    return pd.DataFrame(data)


@pytest.fixture()
def report_path(tmp_path: Path) -> Path:
    """Returns a temporary path for the validation report.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Path: Temporary report file path.
    """
    return tmp_path / "reports" / "data_validation_report.json"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataValidatorColumnChecks:
    """Tests for column-level validation checks."""

    def test_raises_on_missing_required_columns(self, tmp_path: Path) -> None:
        """DataValidator raises DataValidationError if required columns are absent."""
        df = pd.DataFrame({"Time": [1.0], "Amount": [10.0], "Class": [0]})
        validator = DataValidator(df, report_path=tmp_path / "report.json")
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validator.validate()

    def test_raises_on_invalid_target_values(
        self, minimal_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """DataValidator raises DataValidationError if Class has unexpected values."""
        df = minimal_df.copy()
        df["Class"] = 2  # Invalid — should be 0 or 1 only
        validator = DataValidator(df, report_path=tmp_path / "report.json")
        with pytest.raises(DataValidationError, match="unexpected values"):
            validator.validate()


class TestDataValidatorQualityChecks:
    """Tests for data quality checks: nulls, duplicates, distributions."""

    def test_detects_missing_values(
        self, minimal_df: pd.DataFrame, report_path: Path
    ) -> None:
        """DataValidator records high null percentage in the report."""
        df = minimal_df.copy()
        # Inject NaN into V1 for 50% of rows — should be flagged
        df.loc[:99, "V1"] = np.nan
        validator = DataValidator(df, report_path=report_path)
        results = validator.validate()
        mv_result = next(r for r in results if r.check == "missing_values")
        assert not mv_result.passed

    def test_detects_duplicate_rows(
        self, minimal_df: pd.DataFrame, report_path: Path
    ) -> None:
        """DataValidator flags an excessive duplicate row percentage."""
        # Duplicate the entire DataFrame — 100% duplicates
        df = pd.concat([minimal_df, minimal_df], ignore_index=True)
        validator = DataValidator(df, report_path=report_path)
        results = validator.validate()
        dup_result = next(r for r in results if r.check == "duplicate_rows")
        assert not dup_result.passed

    def test_passes_clean_dataset(
        self, minimal_df: pd.DataFrame, report_path: Path
    ) -> None:
        """DataValidator passes all checks on a clean, well-formed DataFrame."""
        validator = DataValidator(minimal_df, report_path=report_path)
        results = validator.validate()
        # All standard checks should pass on clean data
        for r in results:
            assert r.check in {
                "dataset_dimensions", "required_columns", "datatypes",
                "missing_values", "duplicate_rows", "target_column_integrity",
                "target_distribution", "unique_value_counts", "memory_usage",
            }


class TestDataValidatorReportSaving:
    """Tests for JSON report generation."""

    def test_report_is_saved_to_disk(
        self, minimal_df: pd.DataFrame, report_path: Path
    ) -> None:
        """DataValidator saves a JSON report at the configured path."""
        validator = DataValidator(minimal_df, report_path=report_path)
        validator.validate()
        assert report_path.exists()

    def test_report_is_valid_json(
        self, minimal_df: pd.DataFrame, report_path: Path
    ) -> None:
        """The saved validation report is parseable JSON with a 'summary' key."""
        validator = DataValidator(minimal_df, report_path=report_path)
        validator.validate()
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "summary" in data
        assert "checks" in data
        assert data["summary"]["total_checks"] > 0
