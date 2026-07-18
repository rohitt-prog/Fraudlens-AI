"""Preprocessing package for FraudLens AI.

Exposes the public API for the data engineering pipeline components.
"""

from src.preprocessing.dataset_loader import DatasetLoader, DatasetLoadError
from src.preprocessing.eda import ExploratoryAnalyzer
from src.preprocessing.metadata import MetadataGenerator
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.smote import SMOTEResampler, SMOTEError
from src.preprocessing.splitter import DataSplitter, SplitResult
from src.preprocessing.validator import DataValidator, DataValidationError, ValidationResult

__all__ = [
    "DatasetLoader",
    "DatasetLoadError",
    "DataValidator",
    "DataValidationError",
    "ValidationResult",
    "ExploratoryAnalyzer",
    "DataPreprocessor",
    "DataSplitter",
    "SplitResult",
    "SMOTEResampler",
    "SMOTEError",
    "MetadataGenerator",
    "PreprocessingPipeline",
]
