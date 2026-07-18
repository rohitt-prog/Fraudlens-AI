"""Metadata generation module for FraudLens AI preprocessing pipeline.

This module provides the MetadataGenerator class, which assembles a structured
JSON metadata record documenting every aspect of the preprocessing run for
reproducibility, auditing, and model tracking.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.config import (
    DATASET_VERSION,
    RANDOM_SEED,
    SCALER_TYPE,
    SMOTE_K_NEIGHBORS,
    SMOTE_SAMPLING_STRATEGY,
    TARGET_COLUMN,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    settings,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MetadataGenerator:
    """Assembles and saves pipeline run metadata to a JSON file.

    Captures: dataset version, creation timestamp, row counts per split,
    feature count, class distributions before and after SMOTE, scaler type,
    SMOTE parameters, split ratios, and the global random seed.

    Attributes:
        output_path: Path for the generated metadata.json file.
    """

    def __init__(self, output_path: Path | None = None) -> None:
        """Initialises the MetadataGenerator.

        Args:
            output_path: Optional override for the metadata output path.
                Defaults to ml/reports/metadata.json.
        """
        self.output_path = output_path or (settings.reports_dir / "metadata.json")

    def generate_and_save(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train_before_smote: np.ndarray,
        y_train_after_smote: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        scaler_params: dict[str, Any],
        smote_params: dict[str, Any],
        original_row_count: int,
    ) -> dict[str, Any]:
        """Generates the metadata record and saves it to disk.

        Args:
            X_train: Training feature array (post-SMOTE).
            X_val: Validation feature array.
            X_test: Test feature array.
            y_train_before_smote: Training labels before SMOTE.
            y_train_after_smote: Training labels after SMOTE.
            y_val: Validation labels.
            y_test: Test labels.
            scaler_params: Output of DataPreprocessor.get_scaler_params().
            smote_params: Output of SMOTEResampler.get_params().
            original_row_count: Total rows in the raw dataset.

        Returns:
            dict: The complete metadata dictionary that was saved.
        """
        logger.info("Generating pipeline metadata...")

        metadata: dict[str, Any] = {
            "dataset": {
                "version": DATASET_VERSION,
                "target_column": TARGET_COLUMN,
                "original_row_count": original_row_count,
                "feature_count": X_train.shape[1],
            },
            "pipeline_run": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "random_seed": RANDOM_SEED,
            },
            "splits": {
                "ratios": {
                    "train": TRAIN_RATIO,
                    "val": VAL_RATIO,
                    "test": TEST_RATIO,
                },
                "row_counts": {
                    "train_before_smote": int(len(y_train_before_smote)),
                    "train_after_smote": int(len(y_train_after_smote)),
                    "val": int(len(y_val)),
                    "test": int(len(y_test)),
                },
                "class_distributions": {
                    "train_before_smote": self._class_dist(y_train_before_smote),
                    "train_after_smote": self._class_dist(y_train_after_smote),
                    "val": self._class_dist(y_val),
                    "test": self._class_dist(y_test),
                },
                "feature_shapes": {
                    "X_train": list(X_train.shape),
                    "X_val": list(X_val.shape),
                    "X_test": list(X_test.shape),
                },
            },
            "preprocessing": {
                "scaler": {
                    "type": SCALER_TYPE,
                    "params": scaler_params,
                },
                "smote": {
                    "applied_to": "training_split_only",
                    **smote_params,
                },
            },
            "artifacts": {
                "scaler_path": str(settings.scalers_dir / "amount_time_scaler.pkl"),
                "processed_data_dir": str(settings.processed_data_dir),
                "reports_dir": str(settings.reports_dir),
            },
        }

        self._save(metadata)
        return metadata

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _class_dist(self, y: np.ndarray) -> dict[str, Any]:
        """Computes class counts and percentages for a target array.

        Args:
            y: 1-D NumPy array of class labels (0 or 1).

        Returns:
            dict: Contains 'normal_count', 'fraud_count', 'fraud_pct', 'total'.
        """
        unique, counts = np.unique(y, return_counts=True)
        dist = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
        total = int(len(y))
        normal = dist.get(0, 0)
        fraud = dist.get(1, 0)
        return {
            "total": total,
            "normal_count": normal,
            "fraud_count": fraud,
            "fraud_pct": round(fraud / total * 100, 4) if total > 0 else 0.0,
        }

    def _save(self, metadata: dict[str, Any]) -> None:
        """Writes the metadata dictionary to a JSON file.

        Args:
            metadata: The metadata record to serialise.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to: {self.output_path}")
