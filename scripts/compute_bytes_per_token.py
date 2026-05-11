from datasets import load_dataset
from tokenizers import Tokenizer

# ============================================================
# Config
# ============================================================

TOKENIZER_PATH = "tokenizers/paper_bpe_16k.json"

MAX_DOCS = 10000

# ============================================================
# Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# ============================================================
# Load dataset
# ============================================================

print("Loading C4 stream...")

dataset = load_dataset(
    "allenai/c4",
    "en",
    split="validation",
    streaming=True,
)

# ============================================================
# Compute statistics
# ============================================================

total_bytes = 0
total_tokens = 0

print("Evaluating tokenizer...")

for i, example in enumerate(dataset):
    text = example["text"].strip()

    if len(text) < 20:
        continue

    # Normalize whitespace
    text = " ".join(text.split())

    # UTF-8 byte count
    byte_count = len(text.encode("utf-8"))

    # Tokenize
    encoding = tokenizer.encode(text)

    token_count = len(encoding.ids)

    # Skip pathological edge cases
    if token_count == 0:
        continue

    total_bytes += byte_count
    total_tokens += token_count

    if (i + 1) % 1000 == 0:
        current_bpt = total_bytes / total_tokens

        print(
            f"Docs: {i+1:,} | "
            f"Bytes/token: {current_bpt:.4f}"
        )

    if i >= MAX_DOCS:
        break

# ============================================================
# Final metric
# ============================================================

bytes_per_token = total_bytes / total_tokens

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

print(f"Total bytes:   {total_bytes:,}")
print(f"Total tokens:  {total_tokens:,}")

print(f"\nBytes per token: {bytes_per_token:.4f}")