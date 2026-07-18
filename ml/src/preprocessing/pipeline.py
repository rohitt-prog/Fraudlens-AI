"""Preprocessing pipeline orchestrator for FraudLens AI.

This module provides the PreprocessingPipeline class and a CLI entry point.
It coordinates all data engineering steps from raw CSV ingestion through to
saved NumPy arrays and auto-generated reports.

Run from the ml/ directory:
    python -m src.preprocessing.pipeline
"""

import sys
from pathlib import Path

import numpy as np

from config.config import settings
from src.preprocessing.dataset_loader import DatasetLoader
from src.preprocessing.eda import ExploratoryAnalyzer
from src.preprocessing.metadata import MetadataGenerator
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.smote import SMOTEResampler
from src.preprocessing.splitter import DataSplitter
from src.preprocessing.validator import DataValidator
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Filenames for the persisted NumPy array artefacts
ARRAY_FILES: dict[str, str] = {
    "X_train": "X_train.npy",
    "X_val": "X_val.npy",
    "X_test": "X_test.npy",
    "y_train": "y_train.npy",
    "y_val": "y_val.npy",
    "y_test": "y_test.npy",
}


class PreprocessingPipeline:
    """Orchestrates the full data engineering pipeline for FraudLens AI.

    Steps executed in order:
        1. Load — Read raw CSV from ml/data/raw/
        2. Validate — Schema and quality checks → data_validation_report.json
        3. EDA — Generate 8 plots and eda_report.md
        4. Preprocess — Scale Amount and Time, save scaler artefact
        5. Split — Stratified 70/15/15 train/val/test split
        6. SMOTE — Oversample training minority class
        7. Save — Persist 6 NumPy arrays to ml/data/processed/
        8. Metadata — Save pipeline run metadata.json

    Attributes:
        dataset_path: Path to the raw creditcard.csv file.
        processed_dir: Directory where NumPy arrays will be saved.
    """

    def __init__(
        self,
        dataset_path: Path | None = None,
        processed_dir: Path | None = None,
    ) -> None:
        """Initialises the PreprocessingPipeline.

        Args:
            dataset_path: Optional path override for the raw dataset.
                Defaults to ml/data/raw/creditcard.csv.
            processed_dir: Optional path override for saving processed arrays.
                Defaults to ml/data/processed/.
        """
        self.dataset_path = dataset_path or settings.raw_dataset_path
        self.processed_dir = processed_dir or settings.processed_data_dir

    def run(self) -> dict[str, np.ndarray]:
        """Executes the full preprocessing pipeline end-to-end.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys X_train, X_val, X_test,
                y_train, y_val, y_test pointing to the final NumPy arrays.

        Raises:
            Any exception raised by individual pipeline steps propagates up,
            with descriptive error messages logged before re-raising.
        """
        logger.info("=" * 70)
        logger.info("FraudLens AI — Preprocessing Pipeline Starting")
        logger.info("=" * 70)

        # ------------------------------------------------------------------
        # Step 1 — Load
        # ------------------------------------------------------------------
        logger.info("[Step 1/8] Loading dataset...")
        loader = DatasetLoader(path=self.dataset_path)
        df = loader.load()

        # ------------------------------------------------------------------
        # Step 2 — Validate
        # ------------------------------------------------------------------
        logger.info("[Step 2/8] Validating dataset...")
        validator = DataValidator(df)
        validator.validate()

        # ------------------------------------------------------------------
        # Step 3 — EDA
        # ------------------------------------------------------------------
        logger.info("[Step 3/8] Running Exploratory Data Analysis...")
        analyzer = ExploratoryAnalyzer(df)
        analyzer.run()

        # ------------------------------------------------------------------
        # Step 4 — Preprocess (Scale Amount & Time)
        # ------------------------------------------------------------------
        logger.info("[Step 4/8] Scaling Amount and Time features...")
        preprocessor = DataPreprocessor()
        df_scaled = preprocessor.fit_transform(df)
        scaler_params = preprocessor.get_scaler_params()

        # ------------------------------------------------------------------
        # Step 5 — Split
        # ------------------------------------------------------------------
        logger.info("[Step 5/8] Splitting into train/validation/test sets...")
        splitter = DataSplitter()
        split = splitter.split(df_scaled)

        # Capture y_train before SMOTE for metadata comparison
        y_train_before_smote = split.y_train.copy()

        # ------------------------------------------------------------------
        # Step 6 — SMOTE (training only)
        # ------------------------------------------------------------------
        logger.info("[Step 6/8] Applying SMOTE to training split...")
        resampler = SMOTEResampler()
        X_train_resampled, y_train_resampled = resampler.resample(
            split.X_train, split.y_train, split_name="train"
        )
        smote_params = resampler.get_params()

        # ------------------------------------------------------------------
        # Step 7 — Save processed arrays
        # ------------------------------------------------------------------
        logger.info("[Step 7/8] Saving processed NumPy arrays...")
        arrays: dict[str, np.ndarray] = {
            "X_train": X_train_resampled,
            "X_val": split.X_val,
            "X_test": split.X_test,
            "y_train": y_train_resampled,
            "y_val": split.y_val,
            "y_test": split.y_test,
        }
        self._save_arrays(arrays)

        # ------------------------------------------------------------------
        # Step 8 — Metadata
        # ------------------------------------------------------------------
        logger.info("[Step 8/8] Generating pipeline metadata...")
        meta_gen = MetadataGenerator()
        meta_gen.generate_and_save(
            X_train=X_train_resampled,
            X_val=split.X_val,
            X_test=split.X_test,
            y_train_before_smote=y_train_before_smote,
            y_train_after_smote=y_train_resampled,
            y_val=split.y_val,
            y_test=split.y_test,
            scaler_params=scaler_params,
            smote_params=smote_params,
            original_row_count=len(df),
        )

        logger.info("=" * 70)
        logger.info("FraudLens AI — Preprocessing Pipeline Complete ✓")
        logger.info(f"  Processed arrays saved to : {self.processed_dir}")
        logger.info(f"  Reports saved to          : {settings.reports_dir}")
        logger.info(f"  Scaler saved to           : {settings.scalers_dir}")
        logger.info("=" * 70)

        return arrays

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _save_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        """Persists all split NumPy arrays to the processed data directory.

        Args:
            arrays: Dictionary mapping array name to NumPy array.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            filename = ARRAY_FILES[name]
            output_path = self.processed_dir / filename
            np.save(output_path, array)
            logger.info(f"  Saved {name}: shape={array.shape} → {output_path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> None:
    """CLI entry point for the preprocessing pipeline.

    Runs the full pipeline and exits with code 0 on success, 1 on failure.
    """
    try:
        pipeline = PreprocessingPipeline()
        pipeline.run()
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
