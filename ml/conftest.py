"""Pytest configuration for the ml/ subproject.

Adds the ml/ root to sys.path so that imports like `from config.config import settings`
and `from src.preprocessing.xxx import ...` resolve correctly when running pytest
from the ml/ directory.
"""

import sys
from pathlib import Path

# Add the ml/ project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
