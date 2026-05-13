"""Encode the fixed test slice with a trained tokenizer and cache token IDs as .npy."""

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tokenizers import Tokenizer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="tokenize_test")
def main(cfg: DictConfig) -> None:
    tok = cfg.tokenizer
    ev = cfg.eval

    tokenizer_path = Path(tok.output_dir) / tok.output_file
    test_path = Path(cfg.data.output_dir) / cfg.data.test_file
    out_path = Path(ev.output_dir) / tokenizer_path.with_suffix(".npy").name

    if out_path.exists() and not ev.overwrite:
        log.info(f"[skip] {out_path} exists (overwrite=true to regenerate)")
        return

    log.info(
        f"[encode] {tokenizer_path} × {test_path} ({test_path.stat().st_size:,} bytes)"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    text = test_path.read_text(encoding="utf-8")
    token_ids = np.array(tokenizer.encode(text).ids, dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, token_ids)
    log.info(f"[save] {len(token_ids):,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
