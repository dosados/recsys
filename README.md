# RecSys MVP

Modular end-to-end recommendation pipeline with unified contracts:
- cores: `ema`, `gcn`, `online_kmeans`, `two_tower`, `popular`, `random`;
- retrieval modes: `bruteforce_full`, `bruteforce_subset`, `ann_full`, `ann_subset`;
- eval policy: positives-only holdout;
- metrics artifacts: standardized JSON files on disk.

## Installation

```bash
pip install -r requirements.txt
```

## Config-first CLI

All service commands use JSON configs.

### Train

```bash
python run.py train --core ema --config configs/ema.json --model-dir ./artifacts
```

### Update

```bash
python run.py update --core ema --config configs/ema.json --model-dir ./artifacts
```

### Recommend

```bash
python run.py recommend \
  --core ema \
  --config configs/ema.json \
  --retrieval-config configs/retrieval_bruteforce_full.json \
  --model-dir ./artifacts \
  --user-ids 1,2,3 \
  --k 20
```

### Evaluate

```bash
python run.py evaluate \
  --core ema \
  --config configs/ema.json \
  --eval-config configs/eval.json \
  --model-dir ./artifacts
```

## Data protocol

- dataset: VK-LSVD;
- temporal split defaults: train `week_00..24`, validation `week_25`;
- positives policy: `like|share|bookmark|click_on_author|open_comments|timespent>30`.

## Project structure

- `core/contracts` - shared core/retrieval contracts and datatypes;
- `core/ema`, `core/gcn`, `core/online_kmeans`, `core/two_tower` - recommendation cores;
- `retrieval/` - brute-force and ANN retrievers + mode factory;
- `eval/` - holdout, metrics, standardized metrics writer;
- `config/` + `configs/` - JSON config loading and templates;
- `app/cli.py` - unified CLI entrypoint;
- `tests/` - unit + smoke tests.
