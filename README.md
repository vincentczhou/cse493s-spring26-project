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
```

## Data

Download the C4 English train and test streams (~110 MB total):

```bash
uv run data/data.py
```

Outputs land in `_data/`. Override any config value on the CLI, e.g.:

```bash
uv run data/data.py data.overwrite=true
```
