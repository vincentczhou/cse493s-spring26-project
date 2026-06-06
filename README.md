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

## Experiments

The pipeline is parameterized by an **experiment**, which bundles a dataset, vocab size,
sweep grid, and output paths. Two are defined (see [conf/experiment/](conf/experiment/)):

| Experiment   | Dataset                         | Vocab  | Transition points `t`                |
|--------------|---------------------------------|--------|--------------------------------------|
| `c4_16k`     | C4 English (`allenai/c4`)       | 16k    | 0, 4k, 8k, 12k, 16k (steps of 4k)    |
| `olmo_200k`  | OLMo 2 mix (`UW/olmo-mix-...`)  | 200k   | 0, 20k, 40k, …, 200k (steps of 20k)  |

Every command below takes an `experiment=` selector. There is no usable default — running
without one errors loudly on purpose, to force an explicit choice.

## Running individual stages

The pipeline has four stages, each driven by Hydra. Run them in order, or use `run_sweep.py` (below) to orchestrate the full grid.

### 1. Build the data streams

Download and concatenate the train + test character streams for the experiment's dataset. Outputs to `_data/`, with filenames set by the chosen [conf/data/](conf/data/) config.

```bash
uv run data/data.py experiment=c4_16k                     # 1e8 train chars, 1e7 test chars
uv run data/data.py experiment=c4_16k data.overwrite=true # force regenerate
```

### 2. Train a tokenizer

Trains one tokenizer for a given (transition_point, train_chars) cell. Outputs to `_tokenizers/<experiment>/<name>.json`. The `tokenizer:` defaults live in [conf/experiment/default.yaml](conf/experiment/default.yaml); override `t`, `train_chars`, and `output_file` to pick a cell.

```bash
uv run train/train_tokenizer.py experiment=c4_16k \
  tokenizer.t=8000 tokenizer.train_chars=1e5 tokenizer.output_file=t8000_n1e5.json
```

### 3. Tokenize the test set

Encodes the validation slice with a trained tokenizer and caches the token-id array as a `.npy`. Outputs to `_test_streams/<experiment>/<name>.npy`. Pass the same cell overrides as in step 2 so it loads the matching tokenizer.

```bash
uv run eval/tokenize_test.py experiment=c4_16k \
  tokenizer.t=8000 tokenizer.train_chars=1e5 tokenizer.output_file=t8000_n1e5.json
```

### 4. Compute metrics

`eval/metrics.py` is a library (not a CLI). It exposes four functions that take a cached token-id array:

```python
from eval.metrics import compression_ratio, kgram_entropy, capacity_utilization, cross_boundary_token_fraction
import numpy as np
from pathlib import Path

ids = np.load("_test_streams/c4_16k/t8000_n1e5.npy")
cr   = compression_ratio(Path("_data/c4_en_validation_1e7.txt"), ids)
H1_5 = [kgram_entropy(ids, k) for k in range(1, 6)]
util = capacity_utilization(ids, vocab_size=16000)
cb   = cross_boundary_token_fraction(ids, Path("_tokenizers/c4_16k/t8000_n1e5.json"))
```

`run_sweep.py` calls these directly, so you usually don't invoke them by hand.

## Running the full sweep

`run_sweep.py` orchestrates the experiment's grid (the cross-product of `experiment.sweep_grid.t × experiment.sweep_grid.train_chars` — 30 cells for `c4_16k`) in two phases:

1. **Phase 1 — train all tokenizers in parallel.** Each training subprocess uses the Rust BPE trainer, which is CPU-heavy via rayon. To avoid over-subscription, parallelism is capped by `sweep.train_workers`, and each subprocess gets `RAYON_NUM_THREADS=sweep.rayon_num_threads` cores. The product of the two should roughly equal your CPU core count.
2. **Phase 2 — tokenize the test slice + compute metrics in parallel.** These steps are single-threaded per cell, so we can run more concurrently — controlled by `sweep.eval_workers`. After each cell, a row is appended to `results/<experiment>.jsonl`.

```bash
uv run run_sweep.py experiment=c4_16k                                    # defaults (16-core machine)
uv run run_sweep.py experiment=olmo_200k                                 # the larger experiment
uv run run_sweep.py experiment=c4_16k sweep.overwrite=true               # delete results, start fresh
uv run run_sweep.py experiment=c4_16k sweep.train_workers=2 sweep.rayon_num_threads=8  # 2×8 = 16 threads
uv run run_sweep.py experiment=c4_16k sweep.eval_workers=16              # more phase-2 parallelism
uv run run_sweep.py experiment=c4_16k 'experiment.sweep_grid.t=[0,16000]'  # subset of t values
uv run run_sweep.py experiment=c4_16k 'experiment.sweep_grid.train_chars=[1000,10000]'  # subset of training sizes
```

The sweep is **idempotent**: re-running it skips cells already in `results/<experiment>.jsonl`, and the train/tokenize substeps skip their work if the cached output files exist. Interrupted sweeps resume cleanly. Use `sweep.overwrite=true` to force a clean restart.

**Tuning parallelism.** Defaults assume a 16-core machine: `train_workers=4 × rayon_num_threads=4 = 16` active CPU threads in phase 1. The product should match your core count to saturate without over-subscription. Rayon doesn't scale linearly past ~4–8 threads on this BPE workload (algorithmic bottlenecks), so `4 × 4` typically beats `1 × 16` on total throughput. Phase 2's `eval_workers` can be set higher independently since each eval cell is single-threaded.

## Configs

All configs live in [conf/](conf/). See [conf/README.md](conf/README.md) for the layout, how cell configs (ablations) inherit from the template, and how to customize the sweep grid.
