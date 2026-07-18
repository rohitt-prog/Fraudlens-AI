"""Tests for the SMOTEResampler class.

Covers: training-only enforcement, class balancing effect, shape consistency,
invalid split names, and empty/mismatched inputs.
"""

import pytest
import numpy as np

from src.preprocessing.smote import SMOTEResampler, SMOTEError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def imbalanced_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Returns a small imbalanced training dataset: 200 normal, 10 fraud.

    Returns:
        tuple[np.ndarray, np.ndarray]: X_train (210, 30) and y_train (210,).
    """
    rng = np.random.default_rng(7)
    n_normal = 200
    n_fraud = 10
    X_normal = rng.standard_normal((n_normal, 30))
    X_fraud = rng.standard_normal((n_fraud, 30))
    X = np.vstack([X_normal, X_fraud])
    y = np.array([0] * n_normal + [1] * n_fraud)
    return X, y


@pytest.fixture()
def resampler() -> SMOTEResampler:
    """Returns a default SMOTEResampler instance.

    Returns:
        SMOTEResampler: An instance with default parameters.
    """
    return SMOTEResampler(k_neighbors=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSMOTEResamplerSafetyGuard:
    """Tests for the training-only enforcement mechanism."""

    def test_raises_on_val_split_name(
        self,
        imbalanced_training_data: tuple[np.ndarray, np.ndarray],
        resampler: SMOTEResampler,
    ) -> None:
        """SMOTEResampler raises SMOTEError when split_name='val'."""
        X, y = imbalanced_training_data
        with pytest.raises(SMOTEError, match="data leakage"):
            resampler.resample(X, y, split_name="val")

    def test_raises_on_test_split_name(
        self,
        imbalanced_training_data: tuple[np.ndarray, np.ndarray],
        resampler: SMOTEResampler,
    ) -> None:
        """SMOTEResampler raises SMOTEError when split_name='test'."""
        X, y = imbalanced_training_data
        with pytest.raises(SMOTEError, match="data leakage"):
            resampler.resample(X, y, split_name="test")

    def test_raises_on_validation_split_name(
        self,
        imbalanced_training_data: tuple[np.ndarray, np.ndarray],
        resampler: SMOTEResampler,
    ) -> None:
        """SMOTEResampler raises SMOTEError when split_name='validation'."""
        X, y = imbalanced_training_data
        with pytest.raises(SMOTEError):
            resampler.resample(X, y, split_name="validation")

    def test_accepts_train_split_name(
        self,
        imbalanced_training_data: tuple[np.ndarray, np.ndarray],
        resampler: SMOTEResampler,
    ) -> None:
        """SMOTEResampler succeeds when split_name='train'."""
        X, y = imbalanced_training_data
        X_res, y_res = resampler.resample(X, y, split_name="train")
        assert len(X_res) > 0

    def test_accepts_training_split_name(
        self,
        imbalanced_training_data: tuple[np.ndarray, np.ndarray],
        resampler: SMOTEResampler,
    ) -> None:
        """SMOTEResampler succeeds when split_name='training'."""
        X, y = imbalanced_training_data
        X_res, y_res = resampler.resample(X, y, split_name="training")
        assert len(X_res) > 0


class TestSMOTEResamplerBalancing:
    """Tests verifying correct class balancing behaviour."""

    def test_minority_class_is_increased(
        self,
        imbalanced_training_data: tuple[np.ndarray, np.ndarray],
        resampler: SMOTEResampler,
    ) -> None:
        """After SMOTE, the fraud count must be higher than before."""
        X, y = imbalanced_training_data
        fraud_before = int(y.sum())
        _, y_res = resampler.resample(X, y)
        fraud_after = int(y_res.sum())
        assert fraud_after > fraud_before

    def test_output_has_more_rows_than_input(
        self,
        imbalanced_training_data: tuple[np.ndarray, np.ndarray],
        resampler: SMOTEResampler,
    ) -> None:
        """SMOTE must produce more rows than the original training set."""
        X, y = imbalanced_training_data
        X_res, y_res = resampler.resample(X, y)
        assert len(X_res) > len(X)

    def test_output_feature_count_unchanged(
        self,
        imbalanced_training_data: tuple[np.ndarray, np.ndarray],
        resampler: SMOTEResampler,
    ) -> None:
        """The number of features must be identical before and after SMOTE."""
        X, y = imbalanced_training_data
        X_res, _ = resampler.resample(X, y)
        assert X_res.shape[1] == X.shape[1]

    def test_labels_still_binary_after_smote(
        self,
        imbalanced_training_data: tuple[np.ndarray, np.ndarray],
        resampler: SMOTEResampler,
    ) -> None:
        """SMOTE should only produce labels 0 and 1 — no new classes."""
        X, y = imbalanced_training_data
        _, y_res = resampler.resample(X, y)
        assert set(y_res.tolist()).issubset({0, 1})


class TestSMOTEResamplerInputValidation:
    """Tests for input shape and emptiness validation."""

    def test_raises_on_mismatched_shapes(self, resampler: SMOTEResampler) -> None:
        """SMOTEResampler raises ValueError when X and y row counts differ."""
        X = np.zeros((100, 30))
        y = np.zeros(50)
        with pytest.raises(ValueError, match="different number of samples"):
            resampler.resample(X, y)

    def test_raises_on_empty_X(self, resampler: SMOTEResampler) -> None:
        """SMOTEResampler raises ValueError when X_train is empty."""
        X = np.zeros((0, 30))
        y = np.zeros(0)
        with pytest.raises(ValueError, match="empty"):
            resampler.resample(X, y)


class TestSMOTEResamplerParams:
    """Tests for parameter export."""

    def test_get_params_returns_expected_keys(self, resampler: SMOTEResampler) -> None:
        """get_params returns a dictionary with the expected configuration keys."""
        params = resampler.get_params()
        assert "sampling_strategy" in params
        assert "k_neighbors" in params
        assert "random_seed" in params
