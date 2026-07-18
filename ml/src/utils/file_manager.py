"""File and directory management utilities for FraudLens AI.

This module contains functions to check, create, and manage files and directories
safely using the standard `pathlib.Path` objects.
"""

from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensures that a directory exists.

    If the directory does not exist, it creates it along with any necessary
    parent directories. If the path already exists but is not a directory,
    a ValueError is raised.

    Args:
        path: The pathlib.Path representing the target directory.

    Returns:
        Path: The validated/created directory path.

    Raises:
        ValueError: If the path exists but is not a directory.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path.resolve()}")
    elif not path.is_dir():
        raise ValueError(f"Specified path exists but is not a directory: {path}")
    return path


def verify_file_exists(path: Path) -> bool:
    """Verifies that a file exists and is indeed a file.

    Args:
        path: The pathlib.Path representing the file to verify.

    Returns:
        bool: True if the file exists and is a file, False otherwise.
    """
    is_valid = path.exists() and path.is_file()
    if not is_valid:
        logger.warning(f"File not found or is invalid: {path.resolve()}")
    return is_valid
