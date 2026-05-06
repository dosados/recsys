#!/usr/bin/env python3
"""
Read validation interaction parquet in batches (pyarrow) and summarize
how EvaluationRunner builds the holdout and how ranking metrics are defined.

Usage (from repo root):
  conda activate recsys
  python scripts/inspect_eval_data.py
  python scripts/inspect_eval_data.py --data-root datasets/VK-LSVD
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.dataset_defaults import vk_layout  # noqa: E402


def _scan_parquet_batched(path: Path, batch_size: int = 262_144) -> dict:
    """Stream file; row counts + VK holdout = all unique item_id per user (same as eval)."""
    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    n_rows = pf.metadata.num_rows
    user_row_count: Counter = Counter()
    user_items: dict[int, set] = defaultdict(set)
    total = 0
    cols = ["user_id", "item_id"]
    for b in pf.iter_batches(batch_size=batch_size, columns=cols):
        u = b.column("user_id").to_numpy(zero_copy_only=False)
        it = b.column("item_id").to_numpy(zero_copy_only=False)
        for i in range(len(u)):
            uid = int(u[i])
            user_row_count[uid] += 1
            user_items[uid].add(int(it[i]))
        total += len(u)
    assert total == n_rows, (total, n_rows)
    multi = sum(1 for c in user_row_count.values() if c > 1)
    sizes = np.array([len(user_items[u]) for u in user_items], dtype=np.int64)
    sample = []
    for uid in list(user_items.keys())[:3]:
        its = sorted(user_items[uid])
        sample.append({"user_id": uid, "n_unique_items": len(its), "first_items": its[:8]})
    return {
        "path": str(path),
        "schema_columns": names,
        "num_rows": n_rows,
        "num_unique_users": len(user_row_count),
        "users_with_multiple_rows": multi,
        "holdout_users": len(user_items),
        "unique_items_per_user_min": int(sizes.min()),
        "unique_items_per_user_max": int(sizes.max()),
        "unique_items_per_user_mean": float(sizes.mean()),
        "unique_items_per_user_median": float(np.median(sizes)),
        "sample_holdout": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect VK-LSVD validation parquet vs eval holdout logic")
    parser.add_argument("--data-root", default=None, help="Dataset root (default: <repo>/datasets/VK-LSVD)")
    parser.add_argument("--batch-size", type=int, default=262_144)
    args = parser.parse_args()

    layout = vk_layout(args.data_root)
    val_path = Path(layout["validation_interactions"])

    print("=== Paths ===")
    print("validation (default for `evaluate`):", val_path)
    print()

    explanation = """
=== Что считается «позитивом» (ground truth) в текущем коде ===

Файл: eval/runner.py, метод _build_holdout().

1) Legacy (есть колонки score И timestamp):
   - Берутся только строки с score >= min_score_positive (по умолчанию 2).
   - Для каждого user_id выбирается ОДИН item_id — у того события,
     у которого максимальный timestamp (последнее по времени среди «позитивных»).
   - Позитив = этот один item_id (массив длины 1).

2) VK-LSVD (нет score/timestamp в parquet, как в week_25):
   - Используется _build_holdout_vk_all_interactions().
   - Весь файл читается батчами; для каждого user_id собирается множество
     всех уникальных item_id из всех строк с этим пользователем.
   - Позитивы = отсортированный массив этих item_id (релевантных может быть много).

=== Как считаются метрики (eval/metrics.py), для каждого пользователя ===

- Модель выдаёт pred_items[:k] (top-k item_id).
- true_items — массив всех релевантных id из валидации для этого пользователя.

  precision@k = (число позиций в top-k, чей item_id входит в true_items) / k

  recall@k    = (число уникальных релевантных из true_items, попавших в top-k) / len(true_items)

  MAP@k, nDCG@k — бинарная релевантность по позициям (релевантен ли pred[i] относительно множества true).

- Усреднение: по всем пользователям из holdout.

  coverage@k  = |union всех pred[:k]| / total_items (строки items_metadata.parquet).

=== Этот скрипт ===

Потоково читает validation и печатает статистику, согласованную с VK-holdout (все уникальные item на пользователя).
"""
    print(explanation)

    print("=== VALIDATION ===")
    if not val_path.exists():
        print(f"MISSING: {val_path}\n")
        return
    stats = _scan_parquet_batched(val_path, batch_size=args.batch_size)
    print(json.dumps(stats, indent=2))
    print()


if __name__ == "__main__":
    main()
