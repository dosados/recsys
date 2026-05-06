import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.contracts import EvalArtifacts


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_metrics(
    core_name: str,
    metrics: dict[str, float],
    output_dir: str,
    config: dict[str, Any],
    data_slice: str,
    k: int,
) -> EvalArtifacts:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_meta = {
        "core_name": core_name,
        "config_hash": _stable_hash(config),
        "data_slice": data_slice,
        "k": int(k),
        "timestamp": now,
    }
    payload = {"run_meta": run_meta, "metrics": metrics}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{core_name}_{now}_{run_meta['config_hash']}"
    out_path = out_dir / f"{run_id}.metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return EvalArtifacts(
        metrics_path=str(out_path),
        run_id=run_id,
        core_name=core_name,
        metrics=metrics,
        metadata=run_meta,
    )
