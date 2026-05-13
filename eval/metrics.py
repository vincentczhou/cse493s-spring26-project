"""Metrics on cached token-id streams, per PLAN.md §5.

Operates on numpy arrays produced by eval/tokenize_test.py.
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from nltk.util import ngrams
from scipy.stats import entropy


def compression_ratio(test_path: Path, token_ids: np.ndarray) -> float:
    """UTF-8 bytes (raw, pre-NFKC) per token. Option B per PLAN.md §5.1.

    Numerator: raw bytes on disk (the file is not NFKC-normalized at save time
    per PLAN.md §3). Denominator: number of tokens produced by the tokenizer.
    Higher = more compressive.
    """
    return test_path.stat().st_size / len(token_ids)


def kgram_entropy(token_ids: np.ndarray, k: int) -> float:
    """Empirical (k-1)-th order conditional entropy H(T | C) in bits/token.

    For k = 1: H_1 = -Σ_t p̂(t) log_2 p̂(t).
    For k ≥ 2:
        H_k = Σ_c p̂(c) Σ_t [-p̂(t | c) log_2 p̂(t | c)]
            = Σ_c p̂(c) · H(T | C = c)
    where C is the (k-1)-gram context and T is the next token.

    scipy.stats.entropy normalizes its input internally, so we pass raw counts.
    """
    if k == 1:
        _, counts = np.unique(token_ids, return_counts=True)
        return float(entropy(counts, base=2))

    # Count k-grams and aggregate (k-1)-gram context counts from them.

    # 1. Convert token ID array to list of ints for use with nltk.util.ngrams.
    tokens = token_ids.tolist()
    # 2. Create k-gram : counts dictionary
    kgram_counts = Counter(ngrams(tokens, k))
    # 3. Create (k-1)-gram context : counts dictionary by summing k-gram counts that share the same (k-1)-gram prefix.
    ctx_counts: dict = defaultdict(int)
    for kgram, c in kgram_counts.items():
        ctx_counts[kgram[:-1]] += c

    # Algebraic rearrangement of H(T|C) = Σ_c p̂(c) Σ_t [-p̂(t|c) log p̂(t|c)]
    # into a single sum over (context, next-token) pairs:
    #   H(T|C) = (1/N) Σ_{(c,t)} c(c,t) · log_2( c(c) / c(c,t) )
    # where N = total k-grams = Σ c(c, t).

    # List of k-gram counts
    c_ct = np.array(list(kgram_counts.values()), dtype=np.float64)
    # List of corresponding (k-1)-gram context counts
    c_c = np.array([ctx_counts[kg[:-1]] for kg in kgram_counts], dtype=np.float64)

    h_k = (1 / c_ct.sum()) * np.sum(c_ct * np.log2(c_c / c_ct))
    return float(h_k)
