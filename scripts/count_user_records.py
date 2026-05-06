#!/usr/bin/env python3
"""Count per-user records in each parquet file with optional filters."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SUPPORTED_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not_in", "is_null", "is_not_null"}
DEFAULT_ROOT = Path("/home/mants/Documents/own/recsys/datasets")
DEFAULT_GLOB = "week_*.parquet"
DEFAULT_USER_COLUMN = "user_id"
DEFAULT_COMBINE = "or"
DEFAULT_FILTERS: list[list[str]] = [
    ["timespent", ">", "20"],
    ["like", "==", "1"],
    ["share", "==", "1"],
    ["bookmark", "==", "1"],
    ["click_on_author", "==", "1"],
    ["open_comments", "==", "1"],
]


def parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_filters(raw_filters: list[list[str]]) -> list[tuple[str, str, Any]]:
    parsed: list[tuple[str, str, Any]] = []
    for raw_filter in raw_filters:
        if len(raw_filter) != 3:
            raise ValueError("Each --filter must be: <column> <op> <value>")
        column, op, value_raw = raw_filter
        if op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported operator '{op}'. Allowed: {sorted(SUPPORTED_OPS)}")
        parsed.append((column, op, parse_value(value_raw)))
    return parsed


def required_columns_for_filters(filters: list[tuple[str, str, Any]]) -> set[str]:
    return {column for column, _, _ in filters}


def coerce_value_for_field(field_type: pa.DataType, op: str, value: Any) -> Any:
    if pa.types.is_boolean(field_type):
        if op in {"in", "not_in"} and isinstance(value, list):
            coerced = []
            for item in value:
                if isinstance(item, bool):
                    coerced.append(item)
                elif item in (0, 1):
                    coerced.append(bool(item))
                else:
                    coerced.append(item)
            return coerced
        if isinstance(value, bool):
            return value
        if value in (0, 1):
            return bool(value)
    return value


def build_expression(filters: list[tuple[str, str, Any]], combine: str, schema: pa.Schema | None = None):
    if not filters:
        return None

    expressions = []
    for column, op, value in filters:
        field = ds.field(column)
        if schema is not None:
            value = coerce_value_for_field(schema.field(column).type, op, value)
        if op == "==":
            expr = field == value
        elif op == "!=":
            expr = field != value
        elif op == ">":
            expr = field > value
        elif op == ">=":
            expr = field >= value
        elif op == "<":
            expr = field < value
        elif op == "<=":
            expr = field <= value
        elif op == "in":
            expr = field.isin(value)
        elif op == "not_in":
            expr = ~field.isin(value)
        elif op == "is_null":
            expr = field.is_null()
        elif op == "is_not_null":
            expr = ~field.is_null()
        else:
            raise ValueError(f"Unexpected operator: {op}")
        expressions.append(expr)

    result = expressions[0]
    for expr in expressions[1:]:
        result = (result & expr) if combine == "and" else (result | expr)
    return result


def schema_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def count_users_in_file(path: Path, user_column: str, expression):
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(columns=[user_column], filter=expression)
    if table.num_rows == 0:
        return []

    counts = pc.value_counts(table[user_column]).to_pylist()
    rows = []
    for row in counts:
        rows.append((row["values"], int(row["counts"])))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def rows_to_counter(rows: list[tuple[Any, int]]) -> Counter:
    return Counter({user: count for user, count in rows})


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def distribution_stats(counter: Counter) -> dict[str, float]:
    values = sorted(counter.values())
    if not values:
        return {
            "users": 0,
            "rows": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    return {
        "users": float(len(values)),
        "rows": float(sum(values)),
        "min": float(values[0]),
        "p25": percentile(values, 0.25),
        "median": float(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "p75": percentile(values, 0.75),
        "max": float(values[-1]),
    }


def print_distribution_block(title: str, counter: Counter) -> None:
    stats = distribution_stats(counter)
    print(f"  {title}:")
    print(
        "    "
        f"users={int(stats['users'])} rows={int(stats['rows'])} "
        f"min={stats['min']:.2f} p25={stats['p25']:.2f} median={stats['median']:.2f} "
        f"mean={stats['mean']:.2f} p75={stats['p75']:.2f} max={stats['max']:.2f}"
    )


def print_stats_block(title: str, stats: dict[str, float]) -> None:
    print(f"  {title}:")
    print(
        "    "
        f"users={int(stats['users'])} rows={int(stats['rows'])} "
        f"min={stats['min']:.2f} p25={stats['p25']:.2f} median={stats['median']:.2f} "
        f"mean={stats['mean']:.2f} p75={stats['p75']:.2f} max={stats['max']:.2f}"
    )


def average_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
    if not stats_list:
        return {
            "users": 0.0,
            "rows": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    keys = ["users", "rows", "min", "p25", "median", "mean", "p75", "max"]
    return {key: statistics.mean(item[key] for item in stats_list) for key in keys}


def collect_parquet_files(root: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(pattern)
        if path.is_file() and path.suffix == ".parquet" and path.name.startswith("week_")
    )


def print_schema_summary(files: list[Path]) -> None:
    print("=== Columns in parquet files ===")
    for file_path in files:
        cols = schema_columns(file_path)
        print(f"{file_path}")
        print("  columns:", ", ".join(cols))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root folder with parquet files.",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="Glob pattern for parquet files (used recursively). Only week_*.parquet files are processed.",
    )
    parser.add_argument("--user-column", default=DEFAULT_USER_COLUMN, help="Column containing user id.")
    parser.add_argument(
        "--filter",
        action="append",
        nargs=3,
        default=None,
        metavar=("COLUMN", "OP", "VALUE"),
        help='Filter triplet, e.g. --filter like == 1 or --filter platform in "[1,2]". Repeatable.',
    )
    parser.add_argument(
        "--combine",
        choices=["and", "or"],
        default=DEFAULT_COMBINE,
        help="How to combine multiple filters.",
    )
    parser.add_argument("--show-schema-only", action="store_true", help="Only print file columns.")
    args = parser.parse_args()

    # If filters are not passed via CLI, use the task-specific defaults.
    if args.filter is None:
        args.filter = [row[:] for row in DEFAULT_FILTERS]

    files = collect_parquet_files(args.root, args.glob)
    if not files:
        raise SystemExit(f"No parquet files found in {args.root} with pattern '{args.glob}'")

    if args.show_schema_only:
        print_schema_summary(files)
        return

    filters = parse_filters(args.filter)
    required_columns = required_columns_for_filters(filters)

    if filters:
        print(f"filters={filters}, combine={args.combine}")
    else:
        print("filters=<none>")

    global_total_by_user: Counter = Counter()
    global_matched_by_user: Counter = Counter()
    per_file_total_stats: list[dict[str, float]] = []
    per_file_matched_stats: list[dict[str, float]] = []
    skipped_files = 0
    for file_path in files:
        cols = schema_columns(file_path)
        if args.user_column not in cols:
            skipped_files += 1
            continue
        missing_filter_cols = sorted(required_columns - set(cols))
        if missing_filter_cols:
            skipped_files += 1
            continue

        file_schema = pq.ParquetFile(file_path).schema_arrow
        expression = build_expression(filters, args.combine, schema=file_schema)
        matched_rows = count_users_in_file(file_path, args.user_column, expression)
        total_rows = count_users_in_file(file_path, args.user_column, None)
        matched_by_user = rows_to_counter(matched_rows)
        total_by_user = rows_to_counter(total_rows)

        global_total_by_user.update(total_by_user)
        global_matched_by_user.update(matched_by_user)
        per_file_total_stats.append(distribution_stats(total_by_user))
        per_file_matched_stats.append(distribution_stats(matched_by_user))

    processed_files = len(per_file_total_stats)
    print("\n=== Average per-file statistics ===")
    print(f"files_processed={processed_files} files_skipped={skipped_files}")
    print_stats_block("rows_per_user_total", average_stats(per_file_total_stats))
    print_stats_block("rows_per_user_matched", average_stats(per_file_matched_stats))

    print("\n=== Global statistics across all files ===")
    print(f"files_processed={processed_files}")
    print_distribution_block("rows_per_user_total", global_total_by_user)
    print_distribution_block("rows_per_user_matched", global_matched_by_user)


if __name__ == "__main__":
    main()
