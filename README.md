# CSE 493S Spring 2026 Project

Extends the information-theoretic tokenizer analysis of Erdogan et al. (2026) to SuperBPE (Liu et al. 2025), with an ablation over SuperBPE's phase-2 vocabulary budget. See `PLAN.md` for the full methodology and `PROPOSAL.md` for the project proposal.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Rust (for building the SuperBPE tokenizers fork)

## Installation

```bash
# Clone with submodules (pulls in superbpe + its tokenizers fork)
git clone --recurse-submodules <repo-url>
cd cse493s-spring26-project

# If already cloned without submodules
git submodule update --init --recursive

# Install dependencies (builds the SuperBPE tokenizers fork)
uv sync

# Register the pre-commit hook (dev only)
uv run pre-commit install
```

## Running individual stages

The pipeline has four stages, each driven by Hydra. Run them in order, or use `run_sweep.py` (below) to orchestrate the full grid.

### 1. Build the data streams

Download and concatenate the C4 English train + test streams (~110 MB total). Outputs to `_data/`, or the path specified in [conf/data/default.yaml](conf/data/default.yaml).

```bash
uv run data/data.py                     # default: 1e8 train chars, 1e7 test chars
uv run data/data.py data.overwrite=true # force regenerate
```

### 2. Train a tokenizer

Trains one tokenizer for a given (transition_point, train_chars) cell. Outputs to `_tokenizers/<name>.json`, or the path specified in [conf/tokenizer/default.yaml](conf/tokenizer/default.yaml).

```bash
# Run a single cell of the sweep grid
uv run train/train_tokenizer.py tokenizer=t_8000_n1e5

# Or override fields directly on top of the default config
uv run train/train_tokenizer.py tokenizer.t=8000 tokenizer.train_chars=1e5 tokenizer.output_file=t8000_n1e5.json
```

### 3. Tokenize the test set

Encodes the 10M-char validation slice with a trained tokenizer and caches the token-id array as a `.npy`. Outputs to `_test_streams/<name>.npy`, or the path specified in [conf/tokenize_test.yaml](conf/tokenize_test.yaml).

```bash
uv run eval/tokenize_test.py tokenizer=t_8000_n1e5
```

### 4. Compute metrics

`eval/metrics.py` is a library (not a CLI). It exposes three functions that take a cached token-id array:

```python
from eval.metrics import compression_ratio, kgram_entropy, capacity_utilization
import numpy as np

ids = np.load("_test_streams/t8000_n1e5.npy")
cr = compression_ratio(Path("_data/c4_en_validation_1e7.txt"), ids)
H1..H5 = [kgram_entropy(ids, k) for k in range(1, 6)]
util = capacity_utilization(ids, vocab_size=16000)
```

`run_sweep.py` calls these directly, so you usually don't invoke them by hand.

## Running the full sweep

`run_sweep.py` orchestrates the 5×6 grid (5 transition points × 6 training sizes = 30 cells) in two phases:

1. **Phase 1 — train all tokenizers in parallel.** Each training subprocess uses the Rust BPE trainer, which is CPU-heavy via rayon. To avoid over-subscription, parallelism is capped by `sweep.train_workers`, and each subprocess gets `RAYON_NUM_THREADS=sweep.rayon_num_threads` cores. The product of the two should roughly equal your CPU core count.
2. **Phase 2 — tokenize the test slice + compute metrics in parallel.** These steps are single-threaded per cell, so we can run more concurrently — controlled by `sweep.eval_workers`. After each cell, a row is appended to `results/results.jsonl`, or the path specified in [conf/run_sweep.yaml](conf/run_sweep.yaml).

```bash
uv run run_sweep.py                                       # defaults (16-core machine)
uv run run_sweep.py sweep.overwrite=true                  # delete results.jsonl, start fresh
uv run run_sweep.py sweep.train_workers=2 sweep.rayon_num_threads=8  # 2×8 = 16 active threads
uv run run_sweep.py sweep.eval_workers=16                 # more phase-2 parallelism
uv run run_sweep.py sweep.t_values=[0,16000]              # subset of t values
uv run run_sweep.py sweep.n_exponents=[3,4]               # subset of training sizes
```

The sweep is **idempotent**: re-running it skips cells already in `results.jsonl`, and the train/tokenize substeps skip their work if the cached output files exist. Interrupted sweeps resume cleanly. Use `sweep.overwrite=true` to force a clean restart.

**Tuning parallelism.** Defaults assume a 16-core machine: `train_workers=4 × rayon_num_threads=4 = 16` active CPU threads in phase 1. The product should match your core count to saturate without over-subscription. Rayon doesn't scale linearly past ~4–8 threads on this BPE workload (algorithmic bottlenecks), so `4 × 4` typically beats `1 × 16` on total throughput. Phase 2's `eval_workers` can be set higher independently since each eval cell is single-threaded.

## Configs

All configs live in [conf/](conf/). See [conf/README.md](conf/README.md) for the layout, how cell configs (ablations) inherit from the template, and how to customize the sweep grid.
