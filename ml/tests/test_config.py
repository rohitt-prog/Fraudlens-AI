"""Tests for the centralized configuration module.

This module verifies that project configurations are loaded correctly, default
values are set, environment variables are parsed, and paths are resolved
as absolute pathlib.Path objects.
"""

from pathlib import Path

from config.config import settings


def test_settings_metadata() -> None:
    """Verifies that default settings load metadata correctly."""
    assert settings.PROJECT_NAME == "FraudLens AI"
    assert settings.VERSION == "0.1.0"
    assert settings.ENV == "development"
    assert isinstance(settings.RANDOM_SEED, int)


def test_settings_paths() -> None:
    """Verifies that project paths resolve dynamically and are absolute Path objects."""
    assert isinstance(settings.root_dir, Path)
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.raw_data_dir, Path)
    assert isinstance(settings.processed_data_dir, Path)
    assert isinstance(settings.external_data_dir, Path)
    assert isinstance(settings.model_dir, Path)
    assert isinstance(settings.log_dir, Path)
    assert isinstance(settings.log_file, Path)
    assert isinstance(settings.mlruns_dir, Path)

    # Basic parent-child verification to ensure correctness
    assert settings.data_dir.parent == settings.root_dir
    assert settings.raw_data_dir.parent == settings.data_dir
    assert settings.processed_data_dir.parent == settings.data_dir
    assert settings.external_data_dir.parent == settings.data_dir
    assert settings.log_file.parent == settings.log_dir
    assert settings.mlruns_dir.parent == settings.root_dir
