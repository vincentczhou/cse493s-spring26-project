# Config layout

All configs live in `conf/`, structured as [Hydra](https://hydra.cc) config groups.

```
conf/
├── data.yaml                  # primary config for data/data.py
├── train_tokenizer.yaml       # primary config for train/train_tokenizer.py
├── tokenize_test.yaml         # primary config for eval/tokenize_test.py
├── run_sweep.yaml             # primary config for run_sweep.py
├── experiment/                # experiment group: bundles data + vocab + grid + paths
│   ├── default.yaml           # base: sweep workers, path templates, tokenizer/eval blocks
│   ├── c4_16k.yaml            # C4 English,  vocab 16k  (+ /data: c4)
│   └── olmo_200k.yaml         # OLMo 2 mix,  vocab 200k (+ /data: olmo)
└── data/                      # data group: where to pull the char streams from
    ├── defaults.yaml          # template of shared fields (??? = must be filled in)
    ├── c4.yaml                # allenai/c4, en
    └── olmo.yaml              # UW/olmo-mix-1124-subset-p99
```

The **experiment** is the top-level knob. Each experiment file bundles everything that
distinguishes one run of the pipeline from another:

- which dataset to use (via `- /data: <name>` in its defaults list),
- the vocab size,
- the sweep grid (`t × train_chars`),
- and the output paths (tokenizers, token streams, results), namespaced by experiment name.

Every primary config selects `experiment: default`, which is intentionally incomplete
(`name`, `vocab_size`, and `sweep_grid` are unset) so it errors loudly — forcing you to
pick a real experiment on the CLI:

```bash
uv run run_sweep.py experiment=c4_16k
uv run run_sweep.py experiment=olmo_200k
```

## How an experiment is composed

Each experiment file is merged into the global config namespace (`# @package _global_`)
and pulls in its dataset and the shared defaults:

```yaml
# conf/experiment/c4_16k.yaml
# @package _global_
defaults:
  - default          # 1. load experiment/default.yaml (path templates, tokenizer block)
  - /data: c4        # 2. load data/c4.yaml into the global `data:` group
  - _self_           # 3. apply the overrides below

experiment:
  name: c4_16k
  vocab_size: 16000
  sweep_grid:
    t: [0, 4000, 8000, 12000, 16000]
    train_chars: [1000, 10000, 100000, 1000000, 10000000, 100000000]
```

`experiment/default.yaml` holds everything shared across experiments — the sweep worker
counts, the path templates (all keyed off `${experiment.name}`), and the `tokenizer:` /
`eval:` blocks that the per-stage scripts read. Because it's `# @package _global_`, those
blocks land at `cfg.tokenizer`, `cfg.eval`, `cfg.sweep`, etc. — not nested under
`cfg.experiment`.

## How a primary config selects an experiment

The primary configs all share the same two-line defaults list:

```yaml
# conf/run_sweep.yaml
defaults:
  - experiment: default      # from conf/experiment/, pick one (default errors loudly)
  - _self_

hydra:
  run:
    dir: outputs/${experiment.name}/run_sweep/${now:%Y-%m-%d_%H-%M-%S}
```

Override `experiment=` on the CLI to choose a real experiment. The Hydra output dir is
namespaced by experiment name so logs from different experiments don't collide.

## Adding a new dataset

Create `conf/data/<name>.yaml` inheriting the template, filling in the `???` fields:

```yaml
# conf/data/mydata.yaml
defaults:
  - defaults
  - _self_
path: org/dataset           # HF dataset id
name: subset                # HF "config" / subset
train_file: mydata_train_1e8.txt
test_file: mydata_test_1e7.txt
```

Then point an experiment at it with `- /data: mydata` in that experiment's defaults list.

## Customizing the sweep grid

The grid lives in the experiment file as two lists, cross-producted by `run_sweep.py`:

```yaml
experiment:
  sweep_grid:
    t: [0, 4000, 8000, 12000, 16000]
    train_chars: [1000, 10000, 100000, 1000000, 10000000, 100000000]
```

`run_sweep.py` takes the cross-product and builds one `Cell` per `(t, train_chars)` pair
programmatically — there are no per-cell config files to maintain. To run a subset, edit
these lists or override on the CLI:

```bash
uv run run_sweep.py experiment=c4_16k 'experiment.sweep_grid.t=[0,16000]'
```
