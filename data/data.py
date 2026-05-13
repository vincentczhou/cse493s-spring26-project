"""Build C4 English character streams per PLAN.md §3.

Streams documents from C4 (en), concatenates them with a single newline
separator, and writes exactly `train_chars` Unicode codepoints from the
train split and `test_chars` from the validation split.

Docs shorter than `min_doc_chars` are dropped. No NFKC or whitespace
collapsing happens here — both belong to the tokenizer pipeline.
"""

import logging
from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


def build_stream(
    dataset_iter,
    target_chars: int,
    min_doc_chars: int,
    separator: str,
    desc: str,
) -> str:
    parts: list[str] = []
    written = 0
    sep_len = len(separator)
    pbar = tqdm(total=target_chars, unit="char", unit_scale=True, desc=desc)
    for ex in dataset_iter:
        text = ex.get("text") or ""
        # Drop short samples
        if len(text) < min_doc_chars:
            continue

        # If this isn't the first doc, we need to write a separator before it.
        # If the separator would put us over the target, we write only the part of it that fits, then stop (kinda scuffed!). Otherwise we write the whole separator and continue.
        if parts:
            remaining = target_chars - written
            # Separator longer then remaining chars
            if remaining <= sep_len:
                parts.append(separator[:remaining])
                pbar.update(remaining)
                written = target_chars
                break
            # Separator fits, write it and continue
            parts.append(separator)
            pbar.update(sep_len)
            written += sep_len

        remaining = target_chars - written
        # If the text is longer than the remaining characters, we only write the part that fits.
        if len(text) >= remaining:
            parts.append(text[:remaining])
            pbar.update(remaining)
            written = target_chars
            break

        # Normal case: the whole text fits, write it and continue
        parts.append(text)
        pbar.update(len(text))
        written += len(text)

    pbar.close()
    if written < target_chars:
        raise RuntimeError(
            f"Stream exhausted at {written:,} chars before reaching {target_chars:,}"
        )
    # The separator is part of parts, so we don't join on it.
    return "".join(parts)


def write_stream(
    out_path: Path,
    path: str,
    name: str,
    split: str,
    target_chars: int,
    min_doc_chars: int,
    separator: str,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        log.info(f"[skip] {out_path} exists (overwrite=true to regenerate)")
        return
    log.info(f"[stream] {path}/{name} split={split} -> {out_path}")
    ds = load_dataset(path, name, split=split, streaming=True)
    text = build_stream(
        iter(ds), target_chars, min_doc_chars, separator, desc=out_path.name
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    log.info(f"[write] {len(text):,} chars -> {out_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="data")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    out_dir = Path(d.output_dir)
    # Write train stream
    write_stream(
        out_dir / d.train_file,
        path=d.path,
        name=d.name,
        split=d.train_split,
        target_chars=int(d.train_chars),
        min_doc_chars=d.min_doc_chars,
        separator=d.doc_separator,
        overwrite=d.overwrite,
    )
    # Write test stream
    write_stream(
        out_dir / d.test_file,
        path=d.path,
        name=d.name,
        split=d.test_split,
        target_chars=int(d.test_chars),
        min_doc_chars=d.min_doc_chars,
        separator=d.doc_separator,
        overwrite=d.overwrite,
    )


if __name__ == "__main__":
    main()
