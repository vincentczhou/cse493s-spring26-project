"""Metrics on cached token-id streams, per PLAN.md §5.

Operates on numpy arrays produced by eval/tokenize_test.py.
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from nltk.util import ngrams
from scipy.stats import entropy
from tokenizers import Tokenizer


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


def capacity_utilization(token_ids: np.ndarray, vocab_size: int) -> dict[str, float]:
    """Capacity utilization metrics (PLAN.md §5.3, Erdogan eq. 1).

    Both are unigram-distribution entropies divided by log_2(|V|); values in
    [0, 1] measure how "evenly" the tokenizer uses its vocabulary.

        eta   = H_1        / log_2(|V|)     (Shannon — uniform → 1.0)
        eta_2 = H_2^Rényi  / log_2(|V|)     (collision — places weight on the head)

    Rényi entropy of order α (α > 0, α ≠ 1):
        H_α(p) = (1 / (1 - α)) · log_2 Σ_t p̂(t)^α
    At α=2 this collapses to the collision entropy -log_2 Σ p̂(t)².

    Note: H_2^Rényi is the Rényi-α=2 entropy of the *unigram* distribution,
    not the bigram conditional entropy returned by kgram_entropy(ids, 2).
    """
    h1 = kgram_entropy(token_ids, 1)  # Shannon unigram entropy

    _, counts = np.unique(token_ids, return_counts=True)
    p = counts / len(token_ids)
    alpha = 2.0
    h_renyi = float((1 / (1 - alpha)) * np.log2(np.sum(p**alpha)))

    log2_vocab = float(np.log2(vocab_size))
    return {
        "eta": h1 / log2_vocab,
        "eta_2": h_renyi / log2_vocab,
    }


def cross_boundary_token_fraction(token_ids: np.ndarray, tokenizer_path: Path) -> float:
    """Fraction of tokens in the stream that contain a space not at position 0.

    In byte-level BPE, space is encoded as 'Ġ'. Standard phase-1 merges respect
    whitespace boundaries, so Ġ can only appear at position 0 of a token. A Ġ at
    any later position means the token crosses a word boundary — a phase-2-only
    artifact. This fraction goes to 0 as t → vocab_size (all phase 1).
    """
    # Note: this also counts pure-whitespace tokens like ĠĠ (consecutive spaces),
    # which are valid phase-1 merges within a whitespace-only pretokenizer chunk.
    # These are rare and negligible in practice.
    tok = Tokenizer.from_file(str(tokenizer_path))
    vocab = tok.get_vocab()  # {token_str: id}
    is_cross = np.zeros(tok.get_vocab_size(), dtype=bool)
    for token_str, idx in vocab.items():
        if "Ġ" in token_str[1:]:
            is_cross[idx] = True
    return float(is_cross[token_ids].sum()) / len(token_ids)
