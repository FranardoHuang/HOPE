"""Dependency-light receipts for the effective reward recipe.

The input is the *already composed* environment configuration, not a task YAML
or a list of Hydra overrides.  This module deliberately does not import Hydra,
OmegaConf, Isaac Lab, or Torch.  It only records active reward terms (non-None
terms whose finite numeric weight is non-zero) and hashes their effective
callable, weight, and parameters.

The SHA-256 covers the canonical JSON encoding of::

    {"schema_version": 1, "terms": [...]}

It does not cover the convenience ``sha256`` field returned in the receipt.
Values which cannot be represented without guessing at their semantics fail
closed instead of falling back to ``repr()``, whose output may contain process
addresses or other unstable state.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping, Sequence


EFFECTIVE_REWARD_RECIPE_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()


class RewardRecipeError(ValueError):
    """The effective reward configuration cannot produce a stable receipt."""


class RewardRecipeMismatchError(RewardRecipeError):
    """The effective reward recipe does not match its expected SHA-256."""


def _get_member(value, name, default=_MISSING):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _extract_rewards_node(cfg):
    """Accept an env cfg, ``task`` wrapper, or the rewards node itself."""

    rewards = _get_member(cfg, "rewards")
    if rewards is not _MISSING:
        if rewards is None:
            raise RewardRecipeError("cfg.rewards is None")
        return rewards

    task = _get_member(cfg, "task")
    if task is not _MISSING:
        rewards = _get_member(task, "rewards")
        if rewards is _MISSING:
            raise RewardRecipeError("cfg.task has no rewards node")
        if rewards is None:
            raise RewardRecipeError("cfg.task.rewards is None")
        return rewards

    return cfg


def _object_items(value, *, context):
    if isinstance(value, Mapping):
        return list(value.items())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [(field.name, getattr(value, field.name)) for field in dataclasses.fields(value)]
    try:
        attributes = vars(value)
    except TypeError as exc:
        raise RewardRecipeError(
            f"{context} must be a mapping, dataclass, or attribute object"
        ) from exc
    return [(name, item) for name, item in attributes.items() if not name.startswith("_")]


def _callable_identity(func, *, term_name):
    if isinstance(func, str):
        if not func or func.strip() != func or any(char.isspace() for char in func):
            raise RewardRecipeError(
                f"reward term {term_name!r} has an invalid callable identity string"
            )
        return func

    # Bound methods and arbitrary callable instances can hide mutable instance
    # state which a module/qualname string would not capture.
    if inspect.ismethod(func) and getattr(func, "__self__", None) is not None:
        raise RewardRecipeError(
            f"reward term {term_name!r} uses a bound method; callable state is not stable"
        )
    if not (inspect.isfunction(func) or inspect.isbuiltin(func) or inspect.isclass(func)):
        raise RewardRecipeError(
            f"reward term {term_name!r} func must be a named function, class, or identity string"
        )

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if (
        not isinstance(module, str)
        or not module
        or not isinstance(qualname, str)
        or not qualname
        or "<locals>" in qualname
        or "<lambda>" in qualname
    ):
        raise RewardRecipeError(
            f"reward term {term_name!r} callable has no stable module-qualified identity"
        )
    return f"{module}.{qualname}"


def _type_identity(value):
    cls = value if isinstance(value, type) else type(value)
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        raise RewardRecipeError(f"{cls!r} has no stable type identity")
    if "<locals>" in qualname:
        raise RewardRecipeError(f"{module}.{qualname} is a local type and is not stable")
    return f"{module}.{qualname}"


def _stable_json_value(value, *, context, active_ids):
    """Normalize supported values without ever consulting an unstable repr."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RewardRecipeError(f"{context} contains a non-finite float")
        return value
    if isinstance(value, enum.Enum):
        return {"__enum__": f"{_type_identity(value)}.{value.name}"}
    if isinstance(value, slice):
        return {
            "__slice__": [
                _stable_json_value(value.start, context=f"{context}.start", active_ids=active_ids),
                _stable_json_value(value.stop, context=f"{context}.stop", active_ids=active_ids),
                _stable_json_value(value.step, context=f"{context}.step", active_ids=active_ids),
            ]
        }

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in active_ids:
            raise RewardRecipeError(f"{context} contains a reference cycle")
        active_ids.add(object_id)
        try:
            normalized = {}
            for key, item in value.items():
                if type(key) is not str or not key:
                    raise RewardRecipeError(f"{context} contains a non-string or empty key")
                normalized[key] = _stable_json_value(
                    item, context=f"{context}.{key}", active_ids=active_ids
                )
            return normalized
        finally:
            active_ids.remove(object_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        object_id = id(value)
        if object_id in active_ids:
            raise RewardRecipeError(f"{context} contains a reference cycle")
        active_ids.add(object_id)
        try:
            return [
                _stable_json_value(
                    item, context=f"{context}[{index}]", active_ids=active_ids
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_ids.remove(object_id)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        object_id = id(value)
        if object_id in active_ids:
            raise RewardRecipeError(f"{context} contains a reference cycle")
        active_ids.add(object_id)
        try:
            fields = {
                field.name: _stable_json_value(
                    getattr(value, field.name),
                    context=f"{context}.{field.name}",
                    active_ids=active_ids,
                )
                for field in dataclasses.fields(value)
            }
            return {"__config_type__": _type_identity(value), "fields": fields}
        finally:
            active_ids.remove(object_id)

    raise RewardRecipeError(
        f"{context} has unsupported type {_type_identity(value)}; "
        "stable reward receipts do not fall back to repr()"
    )


def _normalized_term(term_name, term):
    func = _get_member(term, "func")
    weight = _get_member(term, "weight")
    params = _get_member(term, "params", {})

    if func is _MISSING:
        raise RewardRecipeError(f"reward term {term_name!r} has no func")
    if weight is _MISSING:
        raise RewardRecipeError(f"reward term {term_name!r} has no weight")
    if type(weight) not in (int, float) or isinstance(weight, bool):
        raise RewardRecipeError(f"reward term {term_name!r} weight must be a finite number")
    normalized_weight = float(weight)
    if not math.isfinite(normalized_weight):
        raise RewardRecipeError(f"reward term {term_name!r} weight must be finite")
    if normalized_weight == 0.0:
        return None

    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise RewardRecipeError(f"reward term {term_name!r} params must be a mapping")

    return {
        "name": term_name,
        "callable": _callable_identity(func, term_name=term_name),
        "weight": normalized_weight,
        "params": _stable_json_value(
            params, context=f"reward term {term_name!r} params", active_ids=set()
        ),
    }


def effective_reward_recipe(cfg):
    """Return the normalized, hashable recipe payload from a composed cfg."""

    rewards = _extract_rewards_node(cfg)
    terms = []
    seen_names = set()
    for raw_name, term in _object_items(rewards, context="rewards node"):
        if type(raw_name) is not str or not raw_name or raw_name.strip() != raw_name:
            raise RewardRecipeError("reward term names must be non-empty strings without padding")
        if raw_name in seen_names:
            raise RewardRecipeError(f"duplicate reward term name {raw_name!r}")
        seen_names.add(raw_name)
        if term is None:
            continue
        normalized = _normalized_term(raw_name, term)
        if normalized is not None:
            terms.append(normalized)
    terms.sort(key=lambda item: item["name"])
    return {
        "schema_version": EFFECTIVE_REWARD_RECIPE_SCHEMA_VERSION,
        "terms": terms,
    }


def canonical_effective_reward_recipe_json(recipe):
    """Encode a normalized recipe payload as stable, compact JSON."""

    if not isinstance(recipe, Mapping):
        raise RewardRecipeError("recipe must be a mapping")
    if set(recipe) != {"schema_version", "terms"}:
        raise RewardRecipeError("recipe must contain exactly schema_version and terms")
    try:
        return json.dumps(
            recipe,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RewardRecipeError("recipe is not canonical JSON data") from exc


def effective_reward_recipe_sha256(cfg):
    """Return SHA-256 of the effective recipe's canonical JSON bytes."""

    recipe = effective_reward_recipe(cfg)
    encoded = canonical_effective_reward_recipe_json(recipe).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_expected_reward_recipe_sha256(actual_sha256, expected_sha256):
    """Fail loudly unless an expected SHA-256 exactly matches the actual recipe."""

    if not isinstance(actual_sha256, str) or not _SHA256_RE.fullmatch(actual_sha256):
        raise RewardRecipeError("actual reward recipe SHA-256 must be 64 lowercase hex characters")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise RewardRecipeError("expected reward recipe SHA-256 must be 64 lowercase hex characters")
    if actual_sha256 != expected_sha256:
        raise RewardRecipeMismatchError(
            "effective reward recipe SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def build_effective_reward_receipt(cfg, expected_sha256=None):
    """Build the normalized receipt and optionally enforce a preregistered SHA."""

    recipe = effective_reward_recipe(cfg)
    canonical_json = canonical_effective_reward_recipe_json(recipe)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    if expected_sha256 is not None:
        validate_expected_reward_recipe_sha256(digest, expected_sha256)
    return {
        "schema_version": recipe["schema_version"],
        "terms": recipe["terms"],
        "sha256": digest,
    }
