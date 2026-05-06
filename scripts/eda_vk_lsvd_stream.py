#!/usr/bin/env python3
"""
Stream EDA for VK-LSVD using pyarrow (iter_batches) + numpy (HLL, npz mmap).

Does not materialize full parquet files. Cardinality of users/items per week
uses HyperLogLog (p=14) for bounded memory on ~billions of rows.

Positives follow specs.txt: like|share|bookmark|click_on_author|open_comments|timespent>30
(bool columns: null treated as false).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass
class ColumnNullAgg:
    nulls: Dict[str, int] = field(default_factory=dict)

    def update(self, table: pa.Table) -> None:
        n = table.num_rows
        if n == 0:
            return
        for name in table.column_names:
            col = table.column(name)
            nn = int(pc.sum(pc.cast(pc.is_null(col), pa.int64())).as_py())
            self.nulls[name] = self.nulls.get(name, 0) + nn


class NumpyHLL:
    """HyperLogLog (standard estimator, 64-bit hashes). p in [4..16]."""

    def __init__(self, p: int = 14) -> None:
        if not (4 <= p <= 16):
            raise ValueError("p must be in [4, 16]")
        self.p = p
        self.m = 1 << p
        self.width = 64 - p
        self.M = np.zeros(self.m, dtype=np.uint8)

    @staticmethod
    def mix64(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.uint64, copy=False)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51_AFD7_ED55_8CCD)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC4CE_B9FE_1A85_EC53)
        x ^= x >> np.uint64(33)
        return x

    def add_u32(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        h = self.mix64(arr.astype(np.uint64, copy=False))
        idx = (h >> np.uint64(64 - self.p)).astype(np.int64, copy=False)
        mask = (np.uint64(1) << np.uint64(self.width)) - np.uint64(1)
        w = h & mask
        wf = w.astype(np.float64)
        rho = np.where(
            wf == 0.0,
            np.int64(self.width + 1),
            np.int64(self.width) - np.floor(np.log2(wf)).astype(np.int64),
        ).astype(np.uint8, copy=False)
        np.maximum.at(self.M, idx, rho)

    def count(self) -> float:
        reg = self.M.astype(np.float64)
        Z = float(np.sum(np.power(2.0, -reg)))
        m = float(self.m)
        if self.m == 16:
            alpha = 0.673
        elif self.m == 32:
            alpha = 0.697
        elif self.m == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1.0 + 0.939 / m)
        E = alpha * m * m / Z
        V = int(np.sum(self.M == 0))
        if E <= 2.5 * m and V > 0:
            E = m * math.log(m / V)
        H = float(1 << 32)
        if E > H / 30.0:
            E = -H * math.log(1.0 - E / H)
        return float(E)


def _positive_mask(table: pa.Table) -> pa.Array:
    like = pc.fill_null(table.column("like"), False)
    share = pc.fill_null(table.column("share"), False)
    bookmark = pc.fill_null(table.column("bookmark"), False)
    coa = pc.fill_null(table.column("click_on_author"), False)
    oc = pc.fill_null(table.column("open_comments"), False)
    ts = table.column("timespent")
    ts_valid = pc.is_valid(ts)
    ts_gt = pc.and_(ts_valid, pc.greater(ts, 30))
    out = like
    for f in (share, bookmark, coa, oc, ts_gt):
        out = pc.or_(out, f)
    return out


def _col_to_numpy_uint32(col: pa.ChunkedArray) -> np.ndarray:
    arr = col.combine_chunks()
    if pa.types.is_uint32(arr.type):
        return np.asarray(arr, dtype=np.uint32)
    return np.asarray(pc.cast(arr, pa.uint32()), dtype=np.uint32)


def scan_parquet_interactions(
    path: str,
    dataset_root: str,
    batch_rows: int,
    hll_p: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    pf = pq.ParquetFile(path)
    n_rows_meta = int(pf.metadata.num_rows)
    schema = pf.schema_arrow
    names = schema.names

    col_agg = ColumnNullAgg()
    hll_users = NumpyHLL(p=hll_p)
    hll_items = NumpyHLL(p=hll_p)

    n_pos = 0
    n_seen = 0

    for batch in pf.iter_batches(batch_size=batch_rows, columns=names):
        table = pa.Table.from_batches([batch])
        n = table.num_rows
        if n == 0:
            continue
        n_seen += n
        col_agg.update(table)
        pos = _positive_mask(table)
        n_pos += int(pc.sum(pc.cast(pos, pa.int64())).as_py())

        uid = table.column("user_id")
        iid = table.column("item_id")
        mu = pc.is_valid(uid)
        mi = pc.is_valid(iid)
        if pc.any(mu).as_py():
            u = _col_to_numpy_uint32(pc.filter(uid, mu))
            hll_users.add_u32(u)
        if pc.any(mi).as_py():
            it = _col_to_numpy_uint32(pc.filter(iid, mi))
            hll_items.add_u32(it)

    null_frac = {c: col_agg.nulls.get(c, 0) / max(n_seen, 1) for c in names}
    est_u = hll_users.count()
    est_i = hll_items.count()
    dt = time.perf_counter() - t0

    return {
        "path": os.path.relpath(path, dataset_root),
        "rows_meta": n_rows_meta,
        "rows_scanned": n_seen,
        "positives": n_pos,
        "positive_rate": n_pos / max(n_seen, 1),
        "unique_users_hll_p": hll_p,
        "unique_users_est": est_u,
        "unique_items_hll_p": hll_p,
        "unique_items_est": est_i,
        "avg_rows_per_user_est": n_seen / max(est_u, 1e-9),
        "avg_rows_per_item_est": n_seen / max(est_i, 1e-9),
        "null_fraction": null_frac,
        "seconds": round(dt, 3),
        "throughput_rows_per_sec": int(n_seen / dt) if dt > 0 else None,
    }


def scan_metadata_parquet_exact_unique(path: str, batch_rows: int, id_col: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    pf = pq.ParquetFile(path)
    n_rows_meta = int(pf.metadata.num_rows)
    names = pf.schema_arrow.names
    col_agg = ColumnNullAgg()
    ids: set[int] = set()
    n_seen = 0

    for batch in pf.iter_batches(batch_size=batch_rows, columns=names):
        table = pa.Table.from_batches([batch])
        n_seen += table.num_rows
        col_agg.update(table)
        col = table.column(id_col)
        m = pc.is_valid(col)
        if pc.any(m).as_py():
            u = pc.unique(pc.filter(col, m)).to_pylist()
            ids.update(int(x) for x in u)

    null_frac = {c: col_agg.nulls.get(c, 0) / max(n_seen, 1) for c in names}
    dt = time.perf_counter() - t0
    return {
        "path": path,
        "rows_meta": n_rows_meta,
        "rows_scanned": n_seen,
        "id_column": id_col,
        "unique_ids_exact": len(ids),
        "null_fraction": null_frac,
        "seconds": round(dt, 3),
    }


def npz_item_embeddings_summary(path: str, chunk: int) -> Dict[str, Any]:
    z = np.load(path, mmap_mode="r")
    out: Dict[str, Any] = {"path": path, "arrays": {}, "item_id_duplicates": None}
    for k in z.files:
        a = z[k]
        entry: Dict[str, Any] = {"shape": list(a.shape), "dtype": str(a.dtype)}
        if k == "embedding" and a.dtype == np.float16 and a.ndim == 2:
            n = min(1 << 20, a.shape[0])
            chunk_arr = np.asarray(a[:n, :], dtype=np.float16)
            entry["nan_fraction_sample_first_rows"] = float(
                np.isnan(chunk_arr.astype(np.float32)).mean()
            )
        out["arrays"][k] = entry

    ids = z["item_id"]
    n = int(ids.shape[0])
    seen: set[int] = set()
    dup = 0
    step = max(1, chunk)
    for i in range(0, n, step):
        block = np.asarray(ids[i : i + step], dtype=np.uint32)
        for v in block.tolist():
            vi = int(v)
            if vi in seen:
                dup += 1
            else:
                seen.add(vi)
    out["item_id_duplicates"] = dup
    out["item_id_unique_exact"] = len(seen)
    out["item_id_rows"] = n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/home/mants/Documents/own/recsys/datasets/VK-LSVD",
    )
    ap.add_argument("--batch-rows", type=int, default=2_000_000)
    ap.add_argument("--hll-p", type=int, default=14)
    ap.add_argument("--npz-chunk", type=int, default=1_000_000)
    ap.add_argument(
        "--out",
        default="/home/mants/Documents/own/recsys/tmp/eda_vk_lsvd.json",
        help="Write full JSON report here",
    )
    ap.add_argument(
        "--max-interaction-files",
        type=int,
        default=0,
        help="If >0, only scan first N interaction files (sorted path order).",
    )
    args = ap.parse_args()
    root = args.root

    interaction_paths: List[str] = []
    for split in ("train", "validation"):
        d = os.path.join(root, "subsamples", "ur0.1", split)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".parquet"):
                interaction_paths.append(os.path.join(d, fn))
    if args.max_interaction_files > 0:
        interaction_paths = interaction_paths[: args.max_interaction_files]

    results: Dict[str, Any] = {
        "config": {
            "root": root,
            "batch_rows": args.batch_rows,
            "hll_p": args.hll_p,
            "positive_rule": "like|share|bookmark|click_on_author|open_comments|timespent>30; null bools=false",
        },
        "interaction_files": [],
        "metadata": {},
        "embeddings_npz": None,
    }

    for ip in interaction_paths:
        results["interaction_files"].append(
            scan_parquet_interactions(
                ip,
                dataset_root=root,
                batch_rows=args.batch_rows,
                hll_p=args.hll_p,
            )
        )

    um = os.path.join(root, "metadata", "users_metadata.parquet")
    im = os.path.join(root, "metadata", "items_metadata.parquet")
    if os.path.isfile(um):
        results["metadata"]["users_metadata"] = scan_metadata_parquet_exact_unique(
            um, args.batch_rows, "user_id"
        )
    if os.path.isfile(im):
        results["metadata"]["items_metadata"] = scan_metadata_parquet_exact_unique(
            im, args.batch_rows, "item_id"
        )

    npz = os.path.join(root, "metadata", "item_embeddings.npz")
    if os.path.isfile(npz):
        results["embeddings_npz"] = npz_item_embeddings_summary(npz, chunk=args.npz_chunk)

    tot_rows = sum(x["rows_scanned"] for x in results["interaction_files"])
    tot_pos = sum(x["positives"] for x in results["interaction_files"])
    tot_sec = sum(x["seconds"] for x in results["interaction_files"])
    results["rollup_interactions"] = {
        "files": len(results["interaction_files"]),
        "total_rows_scanned": tot_rows,
        "total_positives": tot_pos,
        "overall_positive_rate": tot_pos / max(tot_rows, 1),
        "sum_file_seconds": round(tot_sec, 3),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results["rollup_interactions"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
