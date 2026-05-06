from typing import FrozenSet, List

USERS_REQUIRED: FrozenSet[str] = frozenset({"user_id"})
ITEMS_WITH_EMBEDDING: FrozenSet[str] = frozenset({"item_id", "embedding"})
ITEMS_METADATA_ONLY: FrozenSet[str] = frozenset({"item_id"})
INTERACTIONS_LEGACY: FrozenSet[str] = frozenset({"user_id", "item_id", "score"})
INTERACTIONS_VK: FrozenSet[str] = frozenset({"user_id", "item_id"})


def _validate_columns(name: str, columns: FrozenSet[str], required: FrozenSet[str]) -> None:
    missing = required - columns
    if missing:
        raise ValueError(f"{name} schema mismatch. Missing columns: {sorted(missing)}")


def validate_users_schema(column_names: List[str]) -> None:
    _validate_columns("users", frozenset(column_names), USERS_REQUIRED)


def validate_items_schema(column_names: List[str], *, embeddings_in_parquet: bool) -> None:
    req = ITEMS_WITH_EMBEDDING if embeddings_in_parquet else ITEMS_METADATA_ONLY
    _validate_columns("items", frozenset(column_names), req)


def validate_interactions_schema(column_names: List[str]) -> str:
    cols = frozenset(column_names)
    if INTERACTIONS_LEGACY <= cols:
        return "legacy"
    if INTERACTIONS_VK <= cols:
        return "vk"
    missing_legacy = INTERACTIONS_LEGACY - cols
    missing_vk = INTERACTIONS_VK - cols
    raise ValueError(
        "interactions schema mismatch. "
        f"Need either legacy {sorted(INTERACTIONS_LEGACY)} or VK {sorted(INTERACTIONS_VK)}. "
        f"Missing for legacy: {sorted(missing_legacy)}; for VK: {sorted(missing_vk)}"
    )
