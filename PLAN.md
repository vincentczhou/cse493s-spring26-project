# Project Plan

This document captures the methodology, experimental design, and code architecture for the project. It supersedes the high-level PROPOSAL.md for implementation purposes.

## 1. Goals

Reproduce three experiments from Erdogan et al. (2026), "An Information-Theoretic Perspective on LLM Tokenizers" (arXiv:2601.09039), using SuperBPE (Liu et al. 2025, arXiv:2503.13423) as the tokenizer family, and ablate SuperBPE's transition point `t`.

The three metrics, each plotted vs tokenizer training corpus size, are:

1. **Compression ratio** (Erdogan Fig 2/3)
2. **k-gram entropy** H_1, ..., H_5 and entropy rate (Erdogan Fig 4)
3. **Capacity utilization** η = H_1 / log_2(K) (Erdogan Fig 8)

## 2. Scope

- **Vocabulary size:** `K = 16000` only (matches Erdogan Figs 4 and 8).
- **Domain:** English only, from C4 (`allenai/c4`, `en` split). No code, no multilingual.
- **Tokenizer family:** SuperBPE with five values of the transition point `t`. The endpoints recover standard BPE and a pretok-free BPE, so no separate BPE implementation is needed.

## 3. Datasets

- **Source:** C4 English, streamed from HuggingFace `allenai/c4` (`en`).
- **Stream construction:** treat C4 as a single concatenated character stream, following Erdogan §IV. Concatenate documents in stream order with a single `\n` separator between them. "Characters" means Unicode codepoints (`len(str)`), not bytes and not tokens.
- **Training stream:** the first `10^8` characters of the concatenated `train` split, saved once to disk as `_data/c4_en_train_1e8.txt`. Sweep training sizes by taking prefixes of this single file.
- **Test slice:** a fixed `10^7` (10M) character slice taken from the `validation` split (same concatenation rule), saved as `_data/c4_en_test_10m.txt`. Using `validation` rather than the tail of `train` guarantees zero overlap with any training prefix and is materially equivalent for English C4. The same slice is reused across every (tokenizer, training_size) cell.
- **Pre-cleaning:** minimal. Drop documents shorter than ~20 characters. **Do not** collapse internal whitespace (`" ".join(text.split())` is forbidden — it destroys exactly the whitespace structure SuperBPE phase 2 is supposed to bridge). Do not NFKC-normalize the saved files; NFKC happens inside the tokenizer at encode time.

## 4. Tokenizers

All five tokenizers are produced by a single SuperBPE training algorithm parameterized by transition point `t`. Vocabulary size `T = 16000` is fixed.

| Name | `t` | `t/T` | Description |
|------|----:|------:|-------------|
| `bpe_pretok` | 16000 | 1.00 | Standard BPE with whitespace pretokenization (Erdogan baseline) |
| `sbpe_t14400` | 14400 | 0.90 | SuperBPE, mostly phase-1; mirrors Liu et al.'s main 8B model setting |
| `sbpe_t12800` | 12800 | 0.80 | SuperBPE, middle ablation |
| `sbpe_t6400` | 6400 | 0.40 | SuperBPE near Liu et al.'s efficiency optimum |
| `bpe_no_pretok` | 0 | 0.00 | BPE with no whitespace pretokenization (naive variant) |

The `t` values mirror Liu et al.'s ratios (`t/T ∈ {0.9, 0.8, 0.4}` from their `T=200k` ablation). Subject to revision once we look at our own preliminary compression-ratio curves.

**Pipeline (matches Erdogan where compatible):**
- Normalizer: NFKC
- Pre-tokenizer: whitespace-based during phase 1; *disabled* during phase 2 (`t < T`)
- Special tokens: `<pad>`, `<s>`, `</s>`, `<unk>`
- Byte-level fallback: yes (consistent with both papers' practice)

Deviation from Erdogan: SuperBPE phase 2 by construction operates without the whitespace pretokenizer. This is the point of the ablation, not a bug.

## 5. Metrics (exact definitions)

All metrics are computed on the **same fixed 10M-char test slice**, after tokenizing it with the trained tokenizer to produce a single token-id sequence `T = (t_1, ..., t_n)`.

### 5.1 Compression ratio (Erdogan eq. in §IV.A)

```
CR(T; D) = Σ_{x ∈ D} |x|_UTF-8 / Σ_{x ∈ D} |T(x)|
```

UTF-8 bytes per token, averaged over the test corpus. Higher = more compressive.

**Asymmetric byte/token measurement (matches Erdogan's likely intent — "Option B"):** the numerator `|x|_UTF-8` is measured on the **raw** test text as it sits on disk (no NFKC). The denominator `|T(x)|` is the number of tokens after the tokenizer's normalizer + pretokenizer + BPE merges. The paper is not explicit, but adjacent passages in §III ("the raw UTF-8 bytes ... and then tokenize the same text") and the placement of NFKC under "training pipeline" rather than "data preparation" both support this reading. For English C4 the practical difference between this and the NFKC-on-disk alternative is well under 0.1%; the more important property is that all 5 tokenizers within one run measure the same way.

### 5.2 k-gram entropy (Erdogan §IV.B)

Empirical unigram distribution: `p̂(t) = count(t) / n`.

Unigram entropy:

```
H_1 = -Σ_t p̂(t) log_2 p̂(t)
```

For `k = 2, ..., 5`: count all length-k tuples in `T`; derive empirical conditional `p̂(t_i | t_{i-k+1}^{i-1})`; then

```
H_k = (1/n) Σ_{i=k}^{n} -p̂(t_i | t_{i-k+1}^{i-1}) log_2 p̂(t_i | t_{i-k+1}^{i-1})
```

Also report the per-character entropy rate `H_k × (tokens / char)` (Erdogan Fig 4b).

**Expected behavior:** `H_1` rises with training size; `H_k` for `k ≥ 2` falls.

### 5.3 Capacity utilization (Erdogan §VI, eq. 1)

```
η(T; D) = H_1(T; D) / log_2(K)
```

with `K = |V| = 16000`. Also compute the Rényi-2 (collision) variant:

```
η_2(T; D) = H_2_Rényi(T; D) / log_2(K)
```

where `H_2_Rényi = -log_2 Σ_t p̂(t)^2`.

## 6. Experimental grid

- **Training sizes (chars, log scale):** `N ∈ {10^3, 10^4, 10^5, 10^6, 10^7, 10^8}` (6 points)
- **Tokenizers:** 5 (see §4)
- **Total tokenizers to train:** 30
- **Test slice:** 1 (fixed)
- **Total evaluations:** 30, each producing all three metric families

## 7. Code architecture

```
conf/
  config.yaml             # Hydra root config
  data/
    data.yaml             # data group: HF kwargs + char budgets + output paths

data/
  data.py                 # one-shot: download + save train/test character streams

_data/
  c4_en_train_1e8.txt     # (generated, ~100 MB)
  c4_en_test_10m.txt      # (generated, ~10 MB)

train/
  train_tokenizer.py      # one trainer; CLI: --t <int> --train-chars <int> --out <path>
                          # t=T → standard BPE; t=0 → no-pretok BPE; 0<t<T → SuperBPE
                          # Wraps the official SuperBPE repo as the underlying engine.

tokenizers/
  {name}_n{N}.json        # 30 files (one per cell of the grid)

eval/
  tokenize_test.py        # tokenizes the fixed test slice; caches token-id stream
  metrics.py              # compression_ratio, kgram_entropy(k_max=5), capacity_utilization
  test_streams/{name}_n{N}.npy   # cached token id arrays

results/
  results.jsonl           # one line per (tokenizer, train_size, metric)

run_sweep.py              # orchestrates the 5 × 6 grid; trains, evals, appends results

plot/
  figures.py              # produces compression-ratio plot (milestone 1),
                          # k-gram entropy plot, capacity utilization plot
```

## 8. Dependencies

- **`alisawuffles/tokenizers-superbpe`** — a fork of `huggingface/tokenizers` that adds SuperBPE support. **Conflicts with vanilla `tokenizers`** (same package name), so vanilla `tokenizers` must not be installed. Included as a Rust-built editable install via the SuperBPE submodule. Requires a Rust compiler to build.
- **`PythonNut/superbpe`** — the SuperBPE training repo, added as a git submodule. The tokenizers fork lives inside it as a nested submodule (`tokenizers_superbpe/`). Clone with `git clone --recurse-submodules` or run `git submodule update --init --recursive` after cloning.
- `datasets` (HF) — for C4 streaming
- `hydra-core`, `omegaconf` — config management
- `numpy`, `matplotlib` — metrics and plots
- `tqdm` — progress bars

**uv setup:** override the `tokenizers` dependency to point at the fork via `[tool.uv.sources]` (path = `superbpe/tokenizers_superbpe/bindings/python`, editable). The fork exposes the same `tokenizers` package API so all import statements stay unchanged.

The existing `scripts/train_paper_smallest.py` and `scripts/compute_bytes_per_token.py` will be deleted or moved to `scratch/` — they don't match this plan (wrong pre-tokenizer, no training-size sweep).

## 9. Milestones

### Milestone 1: compression-ratio figure

Implement, in order:
1. `data/prep_c4.py` — produces `train_1e8.txt` and `test_10m.txt`.
2. `train/train_tokenizer.py` — works for `t = T` (standard BPE) end-to-end first.
3. `eval/tokenize_test.py` + `eval/metrics.py::compression_ratio`.
4. `run_sweep.py` for just `t = T` over the 6 training sizes.
5. `plot/figures.py::compression_ratio` — reproduce an Erdogan-Fig-3-style line.
6. Extend `train_tokenizer.py` to handle arbitrary `t` (wire up SuperBPE).
7. Re-run sweep over all 5 tokenizers; redraw the figure with 5 lines.

### Milestone 2: k-gram entropy

Add `eval/metrics.py::kgram_entropy` and corresponding plotting code. Reuse the cached token streams from milestone 1 — no retraining needed.

### Milestone 3: capacity utilization

Add `eval/metrics.py::capacity_utilization` (η and η_2). Trivial given the unigram counts already computed for `H_1`.

## 10. Open decisions

- **Final `t` values:** the `{0, 6400, 12800, 14400, 16000}` set is provisional. Revisit after the first compression-ratio curve to make sure we span the interesting regime.
- **Whether `10^8` is feasible end-to-end:** SuperBPE phase 2 is slower without whitespace pretok (Liu et al. §2.2 note). If `10^8` is too slow, cap the largest cell at `10^7`.
