"""Logging utility for FraudLens AI.

This module provides a centralized function to configure and retrieve logger
instances. Loggers write messages to both standard output and a file inside
the logs directory.
"""

import logging
import sys

from config.config import settings


def get_logger(name: str) -> logging.Logger:
    """Configures and retrieves a logger instance.

    Configures the logger level based on settings and adds handlers for console
    and file output. Standard format: Timestamp | Log Level | Module | Message.

    Args:
        name: The name of the module or class requesting the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)

    # Return if handlers are already configured to prevent duplicate handlers
    if logger.handlers:
        return logger

    # Set logger level from global settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Standard log format: Timestamp | Log Level | Module | Message
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console output handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File output handler
    log_file = settings.log_file
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to stderr if file logging setup fails
        print(
            "Warning: Logging to file failed. Initializing console-only logging. "
            f"Error: {e}",
            file=sys.stderr,
        )

    # Prevent logging propagation to root logger
    logger.propagate = False

    return logger
