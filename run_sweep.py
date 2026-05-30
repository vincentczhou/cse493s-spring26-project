"""Run the grid sweep for the selected experiment.

The active experiment (conf/experiment/<name>.yaml) defines the data source,
vocab size, and the sweep grid. Each cell is one (t, train_chars) point.

For each cell:
  1. Train tokenizer  (subprocess: train/train_tokenizer.py)
  2. Tokenize test    (subprocess: eval/tokenize_test.py)
  3. Compute metrics  (in-process: eval.metrics functions)
  4. Append row to the experiment's results file

This is done in two phases:
  Phase 1: train all tokenizers in parallel. Each training subprocess is CPU-heavy
           via the Rust BPE trainer's rayon parallelism. To avoid over-subscription,
           we cap rayon per subprocess via RAYON_NUM_THREADS and bound the number
           of parallel subprocesses by sweep.train_workers.
  Phase 2: tokenize the test slice + compute metrics in parallel. These steps are
           single-threaded, so we can run more cells concurrently here.
           After each cell, append one row to the results file.

Re-running the sweep is idempotent — cells already in the results file are skipped
in phase 2, and the train/tokenize subprocesses skip when their output files exist
(overwrite=false). Set sweep.overwrite=true to start fresh. Partial sweeps resume
cleanly.

  uv run run_sweep.py                       # default experiment (c4_16k)
  uv run run_sweep.py experiment=olmo_200k  # switch experiment
"""

import itertools
import json
import logging
import math
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from eval.metrics import capacity_utilization, compression_ratio, kgram_entropy

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cell:
    """All paths and metadata for one (t, train_size) cell of the sweep."""

    name: str
    experiment_name: str
    tokenizer_file: str
    npy_path: Path
    test_path: Path
    t: int
    vocab_size: int
    train_chars: int
    test_chars: int

    def overrides(self) -> list[str]:
        """Hydra CLI overrides that pin a subprocess to this cell."""
        return [
            f"experiment={self.experiment_name}",
            f"tokenizer.t={self.t}",
            f"tokenizer.train_chars={self.train_chars}",
            f"tokenizer.output_file={self.tokenizer_file}",
        ]


def build_cells(cfg: DictConfig, root: Path) -> list[Cell]:
    """Cross-product the experiment's sweep_grid into Cell objects."""
    grid = cfg.experiment.sweep_grid
    streams_dir = root / cfg.experiment.streams_dir
    test_path = root / cfg.data.output_dir / cfg.data.test_file
    vocab_size = int(cfg.experiment.vocab_size)
    test_chars = int(float(cfg.data.test_chars))

    cells = []
    for t, train_chars in itertools.product(grid.t, grid.train_chars):
        t = int(t)
        train_chars = int(float(train_chars))
        name = f"t{t}_n1e{int(round(math.log10(train_chars)))}"
        tok_file = f"{name}.json"
        cells.append(
            Cell(
                name=name,
                experiment_name=cfg.experiment.name,
                tokenizer_file=tok_file,
                npy_path=streams_dir / f"{name}.npy",
                test_path=test_path,
                t=t,
                vocab_size=vocab_size,
                train_chars=train_chars,
                test_chars=test_chars,
            )
        )
    return cells


def train_cell(cell: Cell, root: Path, env: dict) -> None:
    """Phase 1: train one tokenizer (subprocess skips if cached)."""
    # 1. Train tokenizer (subprocess skips if output exists & overwrite=false).
    subprocess.run(
        ["uv", "run", "train/train_tokenizer.py", *cell.overrides()],
        cwd=root,
        check=True,
        capture_output=True,
        env=env,
    )


def eval_cell(
    cell: Cell,
    root: Path,
    results_path: Path,
    write_lock: threading.Lock,
) -> dict:
    """Phase 2: tokenize the test slice and compute metrics for one cell."""
    # 2. Tokenize test set (subprocess skips if output exists & overwrite=false).
    subprocess.run(
        ["uv", "run", "eval/tokenize_test.py", *cell.overrides()],
        cwd=root,
        check=True,
        capture_output=True,
    )

    # 3. Compute metrics in-process.
    token_ids = np.load(cell.npy_path)
    cr = compression_ratio(cell.test_path, token_ids)
    hk = {f"H_{k}": kgram_entropy(token_ids, k) for k in range(1, 6)}
    cap = capacity_utilization(token_ids, vocab_size=cell.vocab_size)

    result = {
        "cell": cell.name,
        "t": cell.t,
        "train_chars": cell.train_chars,
        "test_chars": cell.test_chars,
        "vocab_size": cell.vocab_size,
        "n_tokens": int(len(token_ids)),
        "compression_ratio": cr,
        **hk,
        **cap,
    }

    with write_lock, results_path.open("a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def completed_cells(results_path: Path) -> set[str]:
    """Cell names already present in the results file."""
    if not results_path.exists():
        return set()
    done = set()
    with results_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["cell"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


@hydra.main(version_base=None, config_path="conf", config_name="run_sweep")
def main(cfg: DictConfig) -> None:
    # Hydra leaves cwd unchanged by default (hydra.job.chdir=False), so Path.cwd() is the project root.
    root = Path.cwd()
    results_path = root / cfg.experiment.results_file
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.sweep.overwrite and results_path.exists():
        log.info(f"[overwrite] removing {results_path}")
        results_path.unlink()

    cells = build_cells(cfg, root)
    done = completed_cells(results_path)
    todo = [c for c in cells if c.name not in done]
    log.info(
        f"[sweep] experiment={cfg.experiment.name}: {len(done)} done, {len(todo)} todo"
    )

    # Phase 1: train tokenizers in parallel with capped rayon to prevent over-subscription.
    train_env = {**os.environ, "RAYON_NUM_THREADS": str(cfg.sweep.rayon_num_threads)}
    log.info(
        f"[phase 1] training: train_workers={cfg.sweep.train_workers}, "
        f"RAYON_NUM_THREADS={cfg.sweep.rayon_num_threads} "
        f"(≈ {cfg.sweep.train_workers * cfg.sweep.rayon_num_threads} active CPU threads)"
    )
    with ThreadPoolExecutor(max_workers=cfg.sweep.train_workers) as ex:
        futures = {ex.submit(train_cell, c, root, train_env): c for c in todo}
        pbar = tqdm(as_completed(futures), total=len(todo), desc="train", unit="cell")
        for fut in pbar:
            cell = futures[fut]
            try:
                fut.result()
            except subprocess.CalledProcessError as e:
                pbar.write(f"[train fail] {cell.name}: {e.stderr.decode()[:500]}")
            except Exception as e:
                pbar.write(f"[train fail] {cell.name}: {e}")

    # Phase 2: tokenize + metrics in parallel (each cell is single-threaded).
    log.info(f"[phase 2] eval: eval_workers={cfg.sweep.eval_workers}")
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=cfg.sweep.eval_workers) as ex:
        futures = {
            ex.submit(eval_cell, c, root, results_path, write_lock): c for c in todo
        }
        pbar = tqdm(as_completed(futures), total=len(todo), desc="eval", unit="cell")
        for fut in pbar:
            cell = futures[fut]
            try:
                r = fut.result()
                pbar.write(
                    f"[done] {cell.name}: CR={r['compression_ratio']:.3f}  "
                    f"H_1={r['H_1']:.2f}  eta={r['eta']:.3f}"
                )
            except subprocess.CalledProcessError as e:
                pbar.write(f"[eval fail] {cell.name}: {e.stderr.decode()[:500]}")
            except Exception as e:
                pbar.write(f"[eval fail] {cell.name}: {e}")


if __name__ == "__main__":
    main()
