# Config layout

All configs live in `conf/`, structured as [Hydra](https://hydra.cc) config groups.

```
conf/
├── data.yaml                  # primary config for data/data.py
├── data/
│   └── default.yaml           # data group: C4 paths, char budgets, output dir
├── train_tokenizer.yaml       # primary config for train/train_tokenizer.py
├── tokenize_test.yaml         # primary config for eval/tokenize_test.py
├── run_sweep.yaml             # primary config for run_sweep.py
└── tokenizer/                 # tokenizer group: 1 template + 30 cell configs
    ├── default.yaml           # template: vocab=16000, train_chars=1e8, regexes, etc.
    ├── t_0_n1e3.yaml          # ┐
    ├── t_0_n1e4.yaml          # │ 30 cell configs (5 t × 6 train_chars).
    ├── ...                    # │ Each inherits default.yaml and overrides
    └── t_16000_n1e8.yaml      # ┘ t, train_chars, and output_file.
```

## How a cell config inherits the template

Each of the 30 cell configs starts with a Hydra `defaults` list:

```yaml
# conf/tokenizer/t_8000_n1e5.yaml
defaults:
  - default       # 1. load conf/tokenizer/default.yaml first
  - _self_        # 2. then apply the overrides below

t: 8000
train_chars: 1e5
output_file: t8000_n1e5.json
```

Hydra processes the `defaults:` list top-to-bottom, with later entries overriding earlier ones. So the cell file gets the full set of fields from `default.yaml`, then overwrites the three it cares about.

## How a primary config selects a tokenizer

The primary configs (`train_tokenizer.yaml`, `tokenize_test.yaml`, etc.) select which file to use from each config group:

```yaml
# conf/train_tokenizer.yaml
defaults:
  - data: default          # from conf/data/, pick default.yaml
  - tokenizer: default     # from conf/tokenizer/, pick default.yaml
  - _self_
```

To switch to a different cell, override on the CLI:

```bash
uv run train/train_tokenizer.py tokenizer=t_8000_n1e5
```

That replaces `tokenizer: default` with `tokenizer: t_8000_n1e5`, which itself includes `default` via its own `defaults:` block — so all the default fields are still present, just with `t`, `train_chars`, and `output_file` overridden.

## Customizing the sweep grid

`conf/run_sweep.yaml` defines the grid as two lists:

```yaml
sweep:
  t_values: [0, 4000, 8000, 12000, 16000]
  n_exponents: [3, 4, 5, 6, 7, 8]
  results_file: results/results.jsonl
  max_workers: 2
```

`run_sweep.py` takes the cross-product and looks up `conf/tokenizer/t_<t>_n1e<n>.yaml` for each cell. If you change these lists, ensure the corresponding cell config files exist.
