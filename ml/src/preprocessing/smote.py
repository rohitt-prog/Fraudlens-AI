"""SMOTE resampling module for FraudLens AI preprocessing pipeline.

This module provides the SMOTEResampler class, which applies Synthetic Minority
Over-sampling Technique (SMOTE) exclusively to the training split to address the
severe class imbalance in the credit card fraud dataset.

IMPORTANT: SMOTE must NEVER be applied to validation or test data. Doing so
causes data leakage and renders evaluation metrics unreliable.
"""

import numpy as np
from imblearn.over_sampling import SMOTE

from config.config import RANDOM_SEED, SMOTE_K_NEIGHBORS, SMOTE_SAMPLING_STRATEGY
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SMOTEError(Exception):
    """Raised when SMOTE is misused (e.g. applied to validation/test data)."""


class SMOTEResampler:
    """Applies SMOTE to the training feature and target arrays only.

    SMOTE generates synthetic minority-class (Fraud) samples by interpolating
    between existing fraud examples and their k nearest neighbours. This
    prevents the model from being biased toward the majority class without
    duplicating real fraud records.

    The resampler enforces training-only use via a split_name guard.

    Attributes:
        sampling_strategy: SMOTE sampling strategy. Default 'minority'.
        k_neighbors: Number of nearest neighbours used by SMOTE.
        random_seed: Seed for reproducibility.
    """

    ALLOWED_SPLIT_NAMES: frozenset[str] = frozenset({"train", "training"})

    def __init__(
        self,
        sampling_strategy: str = SMOTE_SAMPLING_STRATEGY,
        k_neighbors: int = SMOTE_K_NEIGHBORS,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        """Initialises the SMOTEResampler.

        Args:
            sampling_strategy: How to resample. 'minority' oversamples the
                minority class to match the majority. Default 'minority'.
            k_neighbors: Number of neighbours used to generate synthetic samples.
                Default 5.
            random_seed: Seed for reproducibility. Default 42.
        """
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_seed = random_seed
        self._smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_seed,
        )

    def resample(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        split_name: str = "train",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Applies SMOTE to the provided training arrays.

        Logs class distributions before and after resampling.

        Args:
            X_train: Feature matrix of the training split.
            y_train: Target vector of the training split.
            split_name: Name of the split being resampled. Must be 'train' or
                'training'. Raises SMOTEError for any other value.

        Returns:
            tuple[np.ndarray, np.ndarray]: Resampled (X_resampled, y_resampled).

        Raises:
            SMOTEError: If split_name is not in ALLOWED_SPLIT_NAMES.
            ValueError: If X_train and y_train have incompatible shapes.
        """
        self._enforce_training_only(split_name)
        self._validate_inputs(X_train, y_train)

        logger.info(
            f"Applying SMOTE — strategy='{self.sampling_strategy}', "
            f"k_neighbors={self.k_neighbors}, seed={self.random_seed}"
        )
        self._log_distribution("Before SMOTE", y_train)

        X_resampled, y_resampled = self._smote.fit_resample(X_train, y_train)

        self._log_distribution("After  SMOTE", y_resampled)
        logger.info(
            f"SMOTE complete — rows: {len(y_train):,} → {len(y_resampled):,} "
            f"(+{len(y_resampled) - len(y_train):,} synthetic samples)"
        )
        return X_resampled, y_resampled

    def get_params(self) -> dict[str, object]:
        """Returns the SMOTE configuration as a serialisable dictionary.

        Returns:
            dict: SMOTE parameters for inclusion in the pipeline metadata.
        """
        return {
            "sampling_strategy": self.sampling_strategy,
            "k_neighbors": self.k_neighbors,
            "random_seed": self.random_seed,
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _enforce_training_only(self, split_name: str) -> None:
        """Raises SMOTEError if the split name is not a training split.

        Args:
            split_name: The name of the data split being passed.

        Raises:
            SMOTEError: If split_name is not 'train' or 'training'.
        """
        if split_name.lower() not in self.ALLOWED_SPLIT_NAMES:
            raise SMOTEError(
                f"SMOTE must only be applied to training data. "
                f"Got split_name='{split_name}'. "
                "Applying SMOTE to validation or test data causes data leakage "
                "and produces invalid evaluation metrics."
            )

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validates input array shapes and non-emptiness.

        Args:
            X: Feature array.
            y: Target array.

        Raises:
            ValueError: If arrays are empty or have mismatched row counts.
        """
        if X.shape[0] == 0:
            raise ValueError("X_train is empty. Cannot apply SMOTE.")
        if y.shape[0] == 0:
            raise ValueError("y_train is empty. Cannot apply SMOTE.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_train ({X.shape[0]} rows) and y_train ({y.shape[0]} rows) "
                "have different number of samples."
            )

    def _log_distribution(self, label: str, y: np.ndarray) -> None:
        """Logs class count and proportion for a target array.

        Args:
            label: A prefix label for the log line.
            y: Target array to analyse.
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        dist = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
        normal = dist.get(0, 0)
        fraud = dist.get(1, 0)
        logger.info(
            f"  {label}: total={total:,} | "
            f"Normal={normal:,} ({normal / total:.2%}), "
            f"Fraud={fraud:,} ({fraud / total:.2%})"
        )
