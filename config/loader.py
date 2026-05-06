import json
from pathlib import Path
from typing import Any

from .schemas import CORE_SCHEMA, EVAL_SCHEMA, RETRIEVAL_SCHEMA, validate_config


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be an object: {p}")
    return payload


def load_core_config(path: str) -> dict[str, Any]:
    return validate_config(_read_json(path), CORE_SCHEMA, "core_config")


def load_retrieval_config(path: str) -> dict[str, Any]:
    cfg = validate_config(_read_json(path), RETRIEVAL_SCHEMA, "retrieval_config")
    cfg.setdefault("ann_enabled", False)
    cfg.setdefault("random_seed", 42)
    cfg.setdefault("extra", {})
    return cfg


def load_eval_config(path: str) -> dict[str, Any]:
    cfg = validate_config(_read_json(path), EVAL_SCHEMA, "eval_config")
    cfg.setdefault("positive_policy", "positives_only")
    cfg.setdefault("retrieval", {})
    cfg.setdefault("recommend_chunk_size", 10_000)
    cfg.setdefault("max_eval_users", 10_000)
    return cfg
