import numpy as np


def precision_at_k(pred_items: np.ndarray, true_items: np.ndarray, k: int) -> float:
    if k <= 0 or pred_items.size == 0:
        return 0.0
    pred_k = pred_items[:k]
    if pred_k.size == 0:
        return 0.0
    hit = np.isin(pred_k, true_items)
    return float(hit.mean())


def recall_at_k(pred_items: np.ndarray, true_items: np.ndarray, k: int) -> float:
    if true_items.size == 0:
        return 0.0
    pred_k = pred_items[:k]
    hit = np.isin(pred_k, true_items)
    return float(hit.sum() / true_items.size)


def map_at_k(pred_items: np.ndarray, true_items: np.ndarray, k: int) -> float:
    pred_k = pred_items[:k]
    if pred_k.size == 0 or true_items.size == 0:
        return 0.0
    hit = np.isin(pred_k, true_items).astype(np.float32)
    if hit.sum() == 0:
        return 0.0
    cumsum = np.cumsum(hit)
    positions = np.arange(1, hit.size + 1, dtype=np.float32)
    precision_per_hit = (cumsum / positions) * hit
    denom = min(k, true_items.size)
    return float(precision_per_hit.sum() / denom)


def ndcg_at_k(pred_items: np.ndarray, true_items: np.ndarray, k: int) -> float:
    pred_k = pred_items[:k]
    if pred_k.size == 0 or true_items.size == 0:
        return 0.0
    rel = np.isin(pred_k, true_items).astype(np.float32)
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2, dtype=np.float32))
    dcg = float((rel * discounts).sum())
    ideal_rel = np.ones(min(k, true_items.size), dtype=np.float32)
    idcg = float((ideal_rel * discounts[: ideal_rel.size]).sum())
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def coverage_at_k(batch_pred_items: list[np.ndarray], total_items: int, k: int) -> float:
    if total_items <= 0 or not batch_pred_items:
        return 0.0
    merged = np.concatenate([items[:k] for items in batch_pred_items if items.size > 0])
    if merged.size == 0:
        return 0.0
    return float(np.unique(merged).size / total_items)

