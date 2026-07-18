"""Data validation module for FraudLens AI preprocessing pipeline.

This module provides the DataValidator class, which performs comprehensive
schema, quality, and distribution checks on the raw dataset and produces
a structured JSON validation report.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config.config import ALL_FEATURE_COLUMNS, TARGET_COLUMN, settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum acceptable null percentage per column before raising a warning
NULL_THRESHOLD_PCT: float = 5.0
# Maximum acceptable duplicate row percentage before raising a warning
DUPLICATE_THRESHOLD_PCT: float = 1.0


@dataclass
class ValidationResult:
    """Holds the outcome of a single validation check.

    Attributes:
        check: Name of the validation check performed.
        passed: Whether the check passed.
        details: Human-readable detail string.
        data: Optional dictionary of structured metrics.
    """

    check: str
    passed: bool
    details: str
    data: dict[str, Any] = field(default_factory=dict)


class DataValidationError(Exception):
    """Raised when a critical validation check fails."""


class DataValidator:
    """Validates the raw credit card fraud dataset for quality and correctness.

    Performs schema validation, missing-value analysis, duplicate detection,
    dtype verification, target distribution checks, and memory profiling.
    Saves a structured JSON report to ml/reports/.

    Attributes:
        df: The DataFrame to validate.
        report_path: Path to write the JSON validation report.
    """

    REQUIRED_COLUMNS: list[str] = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]

    def __init__(self, df: pd.DataFrame, report_path: Path | None = None) -> None:
        """Initialises the DataValidator.

        Args:
            df: The DataFrame to validate.
            report_path: Optional path override for the validation report.
                Defaults to ml/reports/data_validation_report.json.
        """
        self.df = df
        self.report_path = report_path or (
            settings.reports_dir / "data_validation_report.json"
        )

    def validate(self) -> list[ValidationResult]:
        """Runs all validation checks and saves the JSON report.

        Returns:
            list[ValidationResult]: Ordered list of validation results.

        Raises:
            DataValidationError: If a critical validation check fails.
        """
        logger.info("Starting data validation...")
        results: list[ValidationResult] = [
            self._check_dimensions(),
            self._check_required_columns(),
            self._check_datatypes(),
            self._check_missing_values(),
            self._check_duplicates(),
            self._check_target_column(),
            self._check_target_distribution(),
            self._check_unique_values(),
            self._check_memory_usage(),
        ]

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)
        logger.info(f"Validation complete — {passed_count}/{total} checks passed.")

        self._save_report(results)
        return results

    # -------------------------------------------------------------------------
    # Individual validation checks
    # -------------------------------------------------------------------------

    def _check_dimensions(self) -> ValidationResult:
        """Validates that the DataFrame has non-zero rows and columns.

        Returns:
            ValidationResult: Dimension check result.
        """
        rows, cols = self.df.shape
        passed = rows > 0 and cols > 0
        return ValidationResult(
            check="dataset_dimensions",
            passed=passed,
            details=f"Rows: {rows:,}, Columns: {cols}",
            data={"rows": rows, "columns": cols},
        )

    def _check_required_columns(self) -> ValidationResult:
        """Validates all required columns are present in the DataFrame.

        Returns:
            ValidationResult: Required columns check result.

        Raises:
            DataValidationError: If required columns are missing.
        """
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        passed = len(missing) == 0
        if not passed:
            raise DataValidationError(
                f"Missing required columns: {missing}. "
                "Ensure the dataset is the Kaggle creditcard.csv file."
            )
        return ValidationResult(
            check="required_columns",
            passed=True,
            details=f"All {len(self.REQUIRED_COLUMNS)} required columns present.",
            data={"required": self.REQUIRED_COLUMNS},
        )

    def _check_datatypes(self) -> ValidationResult:
        """Validates that all feature and target columns have numeric dtypes.

        Returns:
            ValidationResult: Dtype check result.
        """
        non_numeric = [
            col
            for col in self.REQUIRED_COLUMNS
            if col in self.df.columns
            and not pd.api.types.is_numeric_dtype(self.df[col])
        ]
        passed = len(non_numeric) == 0
        detail = (
            "All columns are numeric."
            if passed
            else f"Non-numeric columns: {non_numeric}"
        )
        return ValidationResult(
            check="datatypes",
            passed=passed,
            details=detail,
            data={"non_numeric_columns": non_numeric},
        )

    def _check_missing_values(self) -> ValidationResult:
        """Analyses null values per column and flags columns exceeding the threshold.

        Returns:
            ValidationResult: Missing values check result.
        """
        null_counts = self.df.isnull().sum()
        null_pct = (null_counts / len(self.df) * 100).round(4)
        problematic = null_pct[null_pct > NULL_THRESHOLD_PCT].to_dict()
        total_nulls = int(null_counts.sum())
        passed = len(problematic) == 0
        detail = (
            f"No nulls found."
            if total_nulls == 0
            else f"Total nulls: {total_nulls:,}. Columns over {NULL_THRESHOLD_PCT}%: {list(problematic.keys())}"
        )
        return ValidationResult(
            check="missing_values",
            passed=passed,
            details=detail,
            data={
                "total_nulls": total_nulls,
                "null_pct_per_column": null_pct.to_dict(),
                "problematic_columns": problematic,
            },
        )

    def _check_duplicates(self) -> ValidationResult:
        """Detects duplicate rows and flags if they exceed the threshold.

        Returns:
            ValidationResult: Duplicate rows check result.
        """
        dup_count = int(self.df.duplicated().sum())
        dup_pct = round(dup_count / len(self.df) * 100, 4)
        passed = dup_pct <= DUPLICATE_THRESHOLD_PCT
        return ValidationResult(
            check="duplicate_rows",
            passed=passed,
            details=f"Duplicate rows: {dup_count:,} ({dup_pct}%)",
            data={"duplicate_count": dup_count, "duplicate_pct": dup_pct},
        )

    def _check_target_column(self) -> ValidationResult:
        """Validates the target column contains only binary values (0 and 1).

        Returns:
            ValidationResult: Target column integrity check result.

        Raises:
            DataValidationError: If the target column contains unexpected values.
        """
        if TARGET_COLUMN not in self.df.columns:
            raise DataValidationError(f"Target column '{TARGET_COLUMN}' not found.")

        unique_vals = sorted(self.df[TARGET_COLUMN].unique().tolist())
        expected = [0, 1]
        passed = unique_vals == expected
        if not passed:
            raise DataValidationError(
                f"Target column '{TARGET_COLUMN}' contains unexpected values: {unique_vals}. "
                f"Expected: {expected}."
            )
        return ValidationResult(
            check="target_column_integrity",
            passed=True,
            details=f"Target column '{TARGET_COLUMN}' contains only binary values: {unique_vals}.",
            data={"unique_values": unique_vals},
        )

    def _check_target_distribution(self) -> ValidationResult:
        """Computes class distribution and fraud rate for the target column.

        Returns:
            ValidationResult: Target distribution result with counts and ratios.
        """
        counts = self.df[TARGET_COLUMN].value_counts().sort_index()
        total = len(self.df)
        fraud_count = int(counts.get(1, 0))
        normal_count = int(counts.get(0, 0))
        fraud_pct = round(fraud_count / total * 100, 4)
        return ValidationResult(
            check="target_distribution",
            passed=True,
            details=(
                f"Normal (0): {normal_count:,} ({100 - fraud_pct:.4f}%), "
                f"Fraud (1): {fraud_count:,} ({fraud_pct:.4f}%)"
            ),
            data={
                "normal_count": normal_count,
                "fraud_count": fraud_count,
                "fraud_pct": fraud_pct,
                "imbalance_ratio": round(normal_count / max(fraud_count, 1), 2),
            },
        )

    def _check_unique_values(self) -> ValidationResult:
        """Counts unique values per column to detect constant or high-cardinality columns.

        Returns:
            ValidationResult: Unique value count result.
        """
        unique_counts = self.df.nunique().to_dict()
        constant_cols = [col for col, count in unique_counts.items() if count == 1]
        return ValidationResult(
            check="unique_value_counts",
            passed=len(constant_cols) == 0,
            details=(
                "No constant columns detected."
                if not constant_cols
                else f"Constant columns (only one unique value): {constant_cols}"
            ),
            data={"unique_counts": unique_counts, "constant_columns": constant_cols},
        )

    def _check_memory_usage(self) -> ValidationResult:
        """Reports DataFrame memory usage.

        Returns:
            ValidationResult: Memory usage check result.
        """
        mem_bytes = self.df.memory_usage(deep=True).sum()
        mem_mb = round(mem_bytes / 1_048_576, 2)
        return ValidationResult(
            check="memory_usage",
            passed=True,
            details=f"Total memory usage: {mem_mb} MB",
            data={"memory_mb": mem_mb, "memory_bytes": int(mem_bytes)},
        )

    # -------------------------------------------------------------------------
    # Report generation
    # -------------------------------------------------------------------------

    def _save_report(self, results: list[ValidationResult]) -> None:
        """Serialises validation results to a JSON file.

        Args:
            results: List of ValidationResult objects to serialise.
        """
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "summary": {
                "total_checks": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
            },
            "checks": [
                {
                    "check": r.check,
                    "passed": r.passed,
                    "details": r.details,
                    "data": r.data,
                }
                for r in results
            ],
        }
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report saved to: {self.report_path}")
