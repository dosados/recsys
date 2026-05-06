from dataclasses import dataclass
from typing import Any


class ConfigValidationError(ValueError):
    pass


@dataclass
class SchemaRule:
    key: str
    expected_type: type
    required: bool = True


CORE_SCHEMA = [
    SchemaRule("core_name", str, required=True),
    SchemaRule("params", dict, required=True),
]

RETRIEVAL_SCHEMA = [
    SchemaRule("mode", str, required=True),
    SchemaRule("ann_enabled", bool, required=False),
    SchemaRule("subset_size", int, required=False),
    SchemaRule("random_seed", int, required=False),
    SchemaRule("extra", dict, required=False),
]

EVAL_SCHEMA = [
    SchemaRule("k", int, required=True),
    SchemaRule("positive_policy", str, required=True),
    SchemaRule("metrics_output_dir", str, required=True),
    SchemaRule("retrieval", dict, required=False),
    SchemaRule("recommend_chunk_size", int, required=False),
    SchemaRule("max_eval_users", int, required=False),
]


def validate_config(payload: dict[str, Any], schema: list[SchemaRule], name: str) -> dict[str, Any]:
    validated = dict(payload)
    for rule in schema:
        if rule.key not in validated:
            if rule.required:
                raise ConfigValidationError(f"{name}: missing required key '{rule.key}'")
            continue
        value = validated[rule.key]
        if value is not None and not isinstance(value, rule.expected_type):
            raise ConfigValidationError(
                f"{name}: key '{rule.key}' must be {rule.expected_type.__name__}, got {type(value).__name__}"
            )
    return validated
