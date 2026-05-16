"""Run the grid sweep over (transition_point, train_chars) cells.

For each cell:
  1. Train tokenizer  (subprocess: train/train_tokenizer.py)
  2. Tokenize test    (subprocess: eval/tokenize_test.py)
  3. Compute metrics  (in-process: eval.metrics functions)
  4. Append row to results.jsonl

Re-running the sweep is idempotent — cells already in results.jsonl are skipped.
The train/tokenize subprocesses also skip if their output files exist
(overwrite=false), so partial sweeps resume cleanly.
"""

import itertools
import json
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
from hydra import compose
from omegaconf import DictConfig
from tqdm import tqdm

from eval.metrics import capacity_utilization, compression_ratio, kgram_entropy

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cell:
    """All paths and metadata for one (tokenizer, train_size) cell, resolved from its config."""

    name: str
    tokenizer_path: Path
    npy_path: Path
    test_path: Path
    t: int
    vocab_size: int
    train_chars: int


def resolve_cell(name: str, root: Path) -> Cell:
    """Compose conf/tokenize_test.yaml with tokenizer=<name> to derive paths + values."""
    cfg = compose(config_name="tokenize_test", overrides=[f"tokenizer={name}"])
    tok_file = cfg.tokenizer.output_file
    return Cell(
        name=name,
        tokenizer_path=root / cfg.tokenizer.output_dir / tok_file,
        npy_path=root / cfg.eval.output_dir / Path(tok_file).with_suffix(".npy").name,
        test_path=root / cfg.data.output_dir / cfg.data.test_file,
        t=int(cfg.tokenizer.t),
        vocab_size=int(cfg.tokenizer.vocab_size),
        train_chars=int(float(cfg.tokenizer.train_chars)),
    )


def run_cell(
    cell: Cell, root: Path, results_path: Path, write_lock: threading.Lock
) -> dict:
    """Train, tokenize, and compute metrics for one cell."""
    # 1. Train tokenizer (subprocess skips if output exists & overwrite=false).
    subprocess.run(
        ["uv", "run", "train/train_tokenizer.py", f"tokenizer={cell.name}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    # 2. Tokenize test set (subprocess skips if output exists & overwrite=false).
    subprocess.run(
        ["uv", "run", "eval/tokenize_test.py", f"tokenizer={cell.name}"],
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
        "vocab_size": cell.vocab_size,
        "compression_ratio": cr,
        **hk,
        **cap,
    }

    with write_lock, results_path.open("a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def completed_cells(results_path: Path) -> set[str]:
    """Cell names already present in results.jsonl."""
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
    results_path = root / cfg.sweep.results_file
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve every cell's paths upfront (single-threaded — compose isn't thread-safe).
    cell_names = [
        f"t_{t}_n1e{n}"
        for t, n in itertools.product(cfg.sweep.t_values, cfg.sweep.n_exponents)
    ]
    cells = [resolve_cell(name, root) for name in cell_names]

    done = completed_cells(results_path)
    todo = [c for c in cells if c.name not in done]
    log.info(
        f"[sweep] {len(done)} done, {len(todo)} todo, "
        f"max_workers={cfg.sweep.max_workers}"
    )

    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=cfg.sweep.max_workers) as ex:
        futures = {
            ex.submit(run_cell, c, root, results_path, write_lock): c for c in todo
        }
        pbar = tqdm(as_completed(futures), total=len(todo), desc="sweep", unit="cell")
        for fut in pbar:
            cell = futures[fut]
            try:
                r = fut.result()
                pbar.write(
                    f"[done] {cell.name}: CR={r['compression_ratio']:.3f}  "
                    f"H_1={r['H_1']:.2f}  eta={r['eta']:.3f}"
                )
            except subprocess.CalledProcessError as e:
                pbar.write(f"[fail] {cell.name}: {e.stderr.decode()[:500]}")
            except Exception as e:
                pbar.write(f"[fail] {cell.name}: {e}")


if __name__ == "__main__":
    main()
