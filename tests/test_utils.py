"""Tests for the utility modules.

This module validates loggers, random seeding functions, and file manager utility
methods to ensure reliable execution of workflows.
"""

import logging
import random
from pathlib import Path
from src.utils.file_manager import ensure_directory, verify_file_exists
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def test_get_logger() -> None:
    """Verifies that logging utility configures a logger without duplicate handlers."""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert len(logger.handlers) >= 1

    handler_count = len(logger.handlers)
    logger_duplicate = get_logger("test_logger")
    assert len(logger_duplicate.handlers) == handler_count


def test_set_seed() -> None:
    """Verifies that set_seed locks random generation behavior for reproducibility."""
    seed_value = 42
    set_seed(seed_value)
    val1 = random.random()

    set_seed(seed_value)
    val2 = random.random()
    assert val1 == val2


def test_file_manager_ensure_directory(tmp_path: Path) -> None:
    """Verifies that ensure_directory creates a directory and validates existence."""
    target_dir = tmp_path / "sub_folder"
    assert not target_dir.exists()

    result_dir = ensure_directory(target_dir)
    assert result_dir.exists()
    assert result_dir.is_dir()
    assert result_dir == target_dir


def test_file_manager_verify_file_exists(tmp_path: Path) -> None:
    """Verifies that verify_file_exists detects files correctly."""
    target_file = tmp_path / "temp.txt"
    assert not verify_file_exists(target_file)

    target_file.write_text("test data", encoding="utf-8")
    assert verify_file_exists(target_file)
