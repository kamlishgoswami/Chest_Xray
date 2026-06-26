"""Global determinism / reproducibility setup.

Must be imported and called BEFORE any TensorFlow ops are constructed.
Honors the reproducibility protocol in METHODS_SECTION.md §3.2.3 (TRIPOD-AI).
"""
from __future__ import annotations

import os
import random

SEED = 42


def set_global_determinism(seed: int = SEED) -> None:
    """Seed all RNGs and enable deterministic ops.

    Call this at the very top of every entrypoint, before importing/using
    TensorFlow. Full bitwise GPU reproducibility is not guaranteed (cuDNN),
    but run-to-run variance is minimized.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
