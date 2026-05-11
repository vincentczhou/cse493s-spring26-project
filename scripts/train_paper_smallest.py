from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.processors import ByteLevel as ByteLevelProcessor

# ============================================================
# Configuration
# ============================================================

VOCAB_SIZE = 16000

# Approximate training scale:
# ~50k docs is usually around tens of MB of text
MAX_DOCS = 50000

OUTPUT_PATH = "tokenizers/paper_bpe_16k.json"

# ============================================================
# Dataset
# ============================================================

print("Loading C4 stream...")

dataset = load_dataset(
    "allenai/c4",
    "en",
    split="train",
    streaming=True,
)

# ============================================================
# Iterator
# ============================================================

def batch_iterator(batch_size=1000):
    batch = []

    for i, example in enumerate(dataset):
        text = example["text"].strip()

        # Skip tiny samples
        if len(text) < 20:
            continue

        # Normalize whitespace
        text = " ".join(text.split())

        batch.append(text)

        if len(batch) == batch_size:
            yield batch
            batch = []

        if i >= MAX_DOCS:
            break

# ============================================================
# Tokenizer
# ============================================================

print("Building tokenizer...")

tokenizer = Tokenizer(BPE(unk_token="<unk>"))

tokenizer.normalizer = NFKC()

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<s>",
        "</s>",
        "<unk>",
    ],
)

# ============================================================
# Training
# ============================================================

print("Training tokenizer...")

tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
)

tokenizer.post_processor = ByteLevelProcessor(
    trim_offsets=False
)

# ============================================================
# Save
# ============================================================

print("Saving tokenizer...")

tokenizer.save(OUTPUT_PATH)

print(f"Saved tokenizer to: {OUTPUT_PATH}")

# ============================================================
# Inspect
# ============================================================

print("\nTokenizer statistics:")

print("Vocab size:", tokenizer.get_vocab_size())

sample = """
Tokenization is a form of structured compression.
"""

encoding = tokenizer.encode(sample)

print("\nSample tokens:")
print(encoding.tokens[:50])

print("\nNumber of tokens:")
print(len(encoding.tokens))