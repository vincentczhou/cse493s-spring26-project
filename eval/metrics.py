"""Metrics on cached token-id streams, per PLAN.md §5.

Operates on numpy arrays produced by eval/tokenize_test.py.
"""

from pathlib import Path

import numpy as np


def compression_ratio(test_path: Path, token_ids: np.ndarray) -> float:
    """UTF-8 bytes (raw, pre-NFKC) per token. Option B per PLAN.md §5.1.

    Numerator: raw bytes on disk (the file is not NFKC-normalized at save time
    per PLAN.md §3). Denominator: number of tokens produced by the tokenizer.
    Higher = more compressive.
    """
    return test_path.stat().st_size / len(token_ids)
