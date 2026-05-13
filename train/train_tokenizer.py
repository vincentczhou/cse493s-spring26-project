"""Train a tokenizer with transition point t per PLAN.md §4.

Up to two phases:
  Phase 1 (vocab 0  -> t):  fresh BPE w/ whitespace-respecting regex pretokenization
  Phase 2 (vocab t  -> T):  BPE extension permitting cross-whitespace merges

Special cases:
  t == T -> only Phase 1 runs (standard BPE with whitespace pretokenization)
  t == 0 -> only Phase 2 runs (BPE without whitespace pretokenization)

Vocabulary math:
  ByteLevel pre-tokenization seeds the vocab with all 256 byte tokens.
  Phase 1 learns (t - 256) merges  →  merges.txt has (t - 256) entries.
  Phase 2 inherits those and learns (T - t) new merges.
  Final tokenizer: T tokens = 256 byte tokens + (T - 256) merges total.
  Since both phases use the same corpus and phase-2 regex is more permissive,
  all inherited merges are always applicable (no merges from merges.txt are skipped).

The SuperBPE fork's BpeTrainer detects a merges.txt file in the current
working directory at train time and switches to extend mode automatically.
Both phases run inside a temp workdir to keep this CWD-coupling contained.

The phase 1 / phase 2 regexes are copied verbatim from SuperBPE's
scripts/train_tokenizer.sh and scripts/extend_tokenizer.sh respectively.
_train_or_extend_tokenizer is copied from superbpe/utils.py::train_or_extend_tokenizer.
"""

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tokenizers import Regex, Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel, Split
from tokenizers.trainers import BpeTrainer

log = logging.getLogger(__name__)


# Whitespace-respecting (GPT-4-style); never merges across whitespace. Copied from SuperBPE's scripts/train_tokenizer.sh.
PHASE1_REGEX = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
# Permits cross-whitespace merges (only splits on long whitespace, digits, punctuation runs). Copied from SuperBPE's scripts/extend_tokenizer.sh.
PHASE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


@contextmanager
def chdir(path: Path):
    """Context manager to temporarily change the current working directory."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _train_or_extend_tokenizer(
    train_files: list[str], vocab_size: int, regex_string: str
) -> Tokenizer:
    """Copy of superbpe/utils.py::train_or_extend_tokenizer, plus an NFKC
    normalizer (PLAN.md §4 / Erdogan §IV) which SuperBPE's original omits.

    If a merges.txt is present in cwd, the fork's BpeTrainer detects it and
    switches to extend mode; otherwise it trains fresh.
    """
    # Stubs in the SuperBPE fork are out of sync with its runtime API; ignore here.
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = NFKC()  # type: ignore[assignment]
    trainer = BpeTrainer(show_progress=True, vocab_size=vocab_size)  # type: ignore[call-arg]
    pretokenizers = [
        Split(pattern=Regex(regex_string), behavior="isolated", invert=False),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),  # type: ignore[call-arg]
    ]
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pretokenizers)  # type: ignore[assignment]
    tokenizer.train(train_files, trainer)
    return tokenizer


@hydra.main(version_base=None, config_path="../conf", config_name="train_tokenizer")
def main(cfg: DictConfig) -> None:
    c = cfg.tokenizer
    # Transition point
    t = int(c.t)
    # Vocabulary Size
    T = int(c.vocab_size)
    # Training Size (in chars)
    train_chars = int(float(c.train_chars))

    if not 0 <= t <= T:
        raise ValueError(f"Require 0 <= t <= vocab_size; got t={t}, T={T}")

    input_file = Path(c.input_file)
    out_path = Path(c.output_dir) / c.output_file

    if out_path.exists() and not c.overwrite:
        log.info(f"[skip] {out_path} exists (overwrite=true to regenerate)")
        return

    log.info(f"[train] t={t} vocab_size={T} train_chars={train_chars:,}")
    log.info(f"[train] input={input_file} -> output={out_path}")

    text = input_file.read_text(encoding="utf-8")[:train_chars]
    log.info(f"[train] loaded {len(text):,} chars")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The SuperBPE fork's BpeTrainer detects a merges.txt file in the current working directory at train time and switches to extend mode automatically if it is present. Both phases run inside a temp workdir to keep this CWD-coupling contained.
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        train_file = work / "train.txt"
        # Write the training text to a file in the temp workdir
        train_file.write_text(text, encoding="utf-8")
        train_files = [str(train_file)]

        tokenizer: Tokenizer | None = None

        # Phase 1: fresh BPE up to vocab t, whitespace-respecting regex.
        if t > 0:
            log.info(f"[phase1] training BPE to vocab {t}")
            with chdir(work):
                # Ensure no leftover merges.txt (would silently flip to extend mode).
                (work / "merges.txt").unlink(missing_ok=True)
                tokenizer = _train_or_extend_tokenizer(train_files, t, PHASE1_REGEX)
                tokenizer.model.save(".")  # writes merges.txt for phase 2 to read

        # Phase 2: extend (or train fresh if t == 0) up to vocab T, cross-whitespace regex.
        if t < T:
            log.info(
                f"[phase2] {'extending' if t > 0 else 'training'} BPE to vocab {T}"
            )
            with chdir(work):
                if t == 0:
                    (work / "merges.txt").unlink(missing_ok=True)
                tokenizer = _train_or_extend_tokenizer(train_files, T, PHASE2_REGEX)

        assert tokenizer is not None
        tokenizer.save(str(out_path))
        log.info(f"[save] {out_path}  (vocab size = {tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
