"""Reproducibility utilities for FraudLens AI.

This module provides functions to initialize random seeds across various libraries,
ensuring consistent behavior across training and evaluation runs.
"""

import os
import random

from config.config import settings


def set_seed(seed: int | None = None) -> int:
    """Sets random seeds for reproducibility.

    Applies the seed to python's built-in `random` module, sets the environment
    `PYTHONHASHSEED`, and attempts to set seeds for NumPy and PyTorch if they are
    available in the runtime environment.

    Args:
        seed: An optional integer seed. If None, the seed specified in the
          centralized configurations will be used.

    Returns:
        int: The seed value that was set.
    """
    if seed is None:
        seed = settings.RANDOM_SEED

    # Python built-in random
    random.seed(seed)

    # Environment variable for python hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Conditionally set numpy seed if numpy is installed
    try:
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(seed)
    except ImportError:
        pass

    # Conditionally set torch seeds if torch is installed
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
