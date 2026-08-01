#!/usr/bin/env python3
"""Materialize one code-owned N1 fixed-initial-domain receipt.

The operator supplies only a repository root and an action ID.  The output
path, action identity, domain, masks, and cell mixture come from the current
tracked action registry and its pinned contact bundle.  The selected manifest
is parsed through the production schema-v3 validator; the bundle's frozen N=5
source manifest is exact-byte bound and checked only for the historical
identity fields consumed here.  The 32 arm widths come from
``action_ball_sampling._arm_support_parameters`` after the production profile
adapter has validated the selected profile.  This producer therefore does not
maintain a second domain algorithm.

The registry's planned ``fixed_domain_initial_receipt`` path must have no digest
yet.  Publication uses that exact non-content-addressed path with O_EXCL.  The
receipt contains its canonical content SHA; stdout reports the file SHA that a
later registry-only commit may pin.  The registry action-source identity
excludes materialized digests, avoiding a registry/receipt hash cycle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
import sys
import types
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
RECEIPT_KIND = "n1_fixed_domain_initial_receipt_v1"
MATERIALIZATION_KIND = "n1_fixed_domain_initial_materialization_v1"
VERIFICATION_KIND = "n1_fixed_domain_initial_verification_v1"
PRODUCER_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_n1_fixed_domain_initial_receipt.py"
)
REGISTRY_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/a3_vendor_action_registry.py"
)
TRAIN_REPO_PATH = "hope_training/whole_body_tracking/scripts/train.py"
MDP_DIR = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
MDP_MODULES = (
    "counter_rally",
    "action_ball_curriculum",
    "action_ball_sampling",
    "action_ball_manifest",
    "action_ball_profile_adapter",
)
SUPPORTED_ACTIONS = frozenset(("bh_loop_c", "bh_block"))
AUTHORIZED_LANES = {
    "bh_loop_c": (
        "bh_loop_c_static_v1",
        "bh_loop_c_monotonic_fresh_canary_v1",
    ),
    "bh_block": ("bh_block_static_v1",),
}
EXPECTED_CELL_MIXTURE = {
    "center_slots": 1,
    "interior_slots": 3,
    "frontier_slots": 1,
    "interior_level_scale": 0.8,
    "frontier_band_fraction": 0.2,
    "schedule": ["interior", "center", "interior", "frontier", "interior"],
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class FixedDomainReceiptRefused(RuntimeError):
    """Raised before publication when a source or invariant differs."""


@dataclass(frozen=True)
class MaterializedReceipt:
    path: Path
    repo_path: str
    content_sha256: str
    file_sha256: str


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FixedDomainReceiptRefused(
            f"value is not finite canonical ASCII JSON: {exc}"
        ) from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixedDomainReceiptRefused(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FixedDomainReceiptRefused(
                    f"{name} contains non-finite JSON {token!r}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixedDomainReceiptRefused(f"{name} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise FixedDomainReceiptRefused(f"{name} must be one JSON object")
    return value


def _exact(value: object, keys: Sequence[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixedDomainReceiptRefused(f"{name} must be a mapping")
    actual, expected = set(value), set(keys)
    if actual != expected:
        raise FixedDomainReceiptRefused(
            f"{name} keys differ (missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)})"
        )
    return value


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise FixedDomainReceiptRefused(f"{name} must be 64 lowercase hex")
    return value


def _finite(value: object, *, name: str, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise FixedDomainReceiptRefused(f"{name} must be a plain number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise FixedDomainReceiptRefused(
            f"{name} must be finite and >= {minimum}"
        )
    return result


def _repo_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise FixedDomainReceiptRefused(f"{name} must be a normalized repo path")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != value
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise FixedDomainReceiptRefused(f"{name} must be a normalized repo path")
    return value


def _root(value: object) -> Path:
    try:
        root = Path(value).resolve(strict=True)  # type: ignore[arg-type]
    except (OSError, RuntimeError, TypeError) as exc:
        raise FixedDomainReceiptRefused("repo root does not resolve") from exc
    if not root.is_dir():
        raise FixedDomainReceiptRefused("repo root must be a directory")
    return root


def _stable_read(root: Path, relative: str, *, name: str) -> bytes:
    relative = _repo_path(relative, name=f"{name} path")
    path = root / relative
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FixedDomainReceiptRefused(f"cannot inspect {name}: {relative}") from exc
    if not stat.S_ISREG(before.st_mode) or resolved != path:
        raise FixedDomainReceiptRefused(f"{name} must be a real regular repo file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FixedDomainReceiptRefused(f"cannot read {name}") from exc
    after = path.lstat()
    identity = lambda info: (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(raw) != after.st_size:
        raise FixedDomainReceiptRefused(f"{name} changed while being read")
    return raw


def _git(root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise FixedDomainReceiptRefused("cannot invoke git") from exc


def _tracked_clean(root: Path, paths: Sequence[str]) -> None:
    unique = tuple(dict.fromkeys(paths))
    for path in unique:
        result = _git(root, ("ls-files", "--error-unmatch", "--", path))
        if result.returncode != 0 or result.stdout.strip() != path:
            raise FixedDomainReceiptRefused(f"input is not tracked: {path}")
    for args, label in (
        (("diff", "--quiet", "--", *unique), "working tree"),
        (("diff", "--cached", "--quiet", "--", *unique), "index"),
    ):
        if _git(root, args).returncode != 0:
            raise FixedDomainReceiptRefused(f"inputs differ from HEAD in {label}")


def _load_file_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FixedDomainReceiptRefused(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise FixedDomainReceiptRefused(f"cannot import {path}: {exc}") from exc
    return module


def _load_mdp(root: Path, salt: str) -> dict[str, types.ModuleType]:
    package_name = f"_fixed_domain_mdp_{salt}_{_sha(str(root).encode())[:10]}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root / MDP_DIR)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    result: dict[str, types.ModuleType] = {}
    for name in MDP_MODULES:
        full_name = f"{package_name}.{name}"
        result[name] = _load_file_module(
            root / MDP_DIR / f"{name}.py", full_name
        )
    return result


def _pin(value: object, *, name: str, allow_pending: bool = False) -> tuple[str, Optional[str]]:
    if not hasattr(value, "path") or not hasattr(value, "sha256"):
        if not isinstance(value, Mapping) or not {"path", "sha256"}.issubset(value):
            raise FixedDomainReceiptRefused(f"{name} must contain path and sha256")
        path, digest = value["path"], value["sha256"]
    else:
        path, digest = value.path, value.sha256
    path = _repo_path(path, name=f"{name}.path")
    if digest is None and allow_pending:
        return path, None
    return path, _digest(digest, name=f"{name}.sha256")


def _pinned_json(
    root: Path, pin: tuple[str, Optional[str]], *, name: str
) -> tuple[bytes, dict[str, Any]]:
    path, expected = pin
    if expected is None:
        raise FixedDomainReceiptRefused(f"{name} digest is not materialized")
    raw = _stable_read(root, path, name=name)
    if _sha(raw) != expected:
        raise FixedDomainReceiptRefused(f"{name} SHA-256 mismatch")
    return raw, _strict_json(raw, name=name)


def _validate_source_manifest_identity(
    document: Mapping[str, Any], *, action_id: str
) -> None:
    """Validate only the immutable identity fields of a historical source.

    A contact bundle may intentionally retain an older, content-addressed
    source manifest whose diagnostic holdout predates the current formal
    admission policy.  Reapplying today's full manifest validator would make
    that historical identity retroactively invalid.  The selected training
    manifest still passes the complete production validator below; this check
    is deliberately limited to the two source fields consumed by this receipt.
    """

    if document.get("mobility_mode") != "no_move":
        raise FixedDomainReceiptRefused(
            "source manifest mobility_mode must be exactly no_move"
        )
    action_order = document.get("action_order")
    if (
        type(action_order) is not list
        or any(type(value) is not str for value in action_order)
        or action_order.count(action_id) != 1
    ):
        raise FixedDomainReceiptRefused(
            "source manifest action_order must contain the selected action exactly once"
        )


def _axis_labels(arm: str) -> tuple[str, str, str, str]:
    """Human-readable labels only; widths remain runtime-owned."""

    if arm.startswith("time_to_contact_"):
        return "time_to_contact", "scalar", arm.rsplit("_", 1)[1], "s"
    if arm.startswith("incoming_speed_"):
        return "incoming_speed", "scalar", arm.rsplit("_", 1)[1], "mps"
    if arm.startswith("spin_magnitude_"):
        return "spin_magnitude", "scalar", arm.rsplit("_", 1)[1], "radps"
    if arm.startswith("contact_"):
        _, coordinate, side = arm.split("_")
        return "contact", coordinate, side, "m"
    for prefix in ("base_spawn", "base_travel", "landing_aim"):
        if arm.startswith(prefix + "_"):
            _, _, coordinate, side = arm.split("_")
            return prefix, coordinate, side, "m"
    for prefix in ("incoming_direction", "spin_direction"):
        if arm.startswith(prefix + "_"):
            _, _, coordinate, side = arm.split("_")
            return prefix, coordinate, side, "rad"
    raise FixedDomainReceiptRefused(f"unknown runtime arm {arm!r}")


def _input(path: str, raw: bytes) -> dict[str, str]:
    return {"path": path, "sha256": _sha(raw)}


def build_receipt(
    repo_root: object,
    action_id: object,
    *,
    require_materialized_output: bool = False,
) -> dict[str, Any]:
    """Build and validate one receipt without publishing it."""

    root = _root(repo_root)
    if type(action_id) is not str or action_id not in SUPPORTED_ACTIONS:
        raise FixedDomainReceiptRefused(
            "action_id must be exactly bh_loop_c or bh_block"
        )
    source_paths = (
        PRODUCER_REPO_PATH,
        REGISTRY_REPO_PATH,
        TRAIN_REPO_PATH,
        *(f"{MDP_DIR}/{name}.py" for name in MDP_MODULES),
    )
    source_raw = {
        path: _stable_read(root, path, name=path) for path in source_paths
    }
    train_source = source_raw[TRAIN_REPO_PATH]
    if (
        train_source.count(b"sampler = ActionBallSampler(") != 1
        or train_source.count(b"sampling_mixture=SamplingMixture(),") != 1
    ):
        raise FixedDomainReceiptRefused(
            "production train wiring must construct ActionBallSampler with "
            "exactly one default SamplingMixture()"
        )
    registry = _load_file_module(
        root / REGISTRY_REPO_PATH, f"_fixed_domain_registry_{action_id}_{id(root)}"
    )
    try:
        config = registry.get_action_config(action_id)
        bundle_pin = _pin(
            registry.require_materialized_pin(
                config.contact_bundle,
                action_id=action_id,
                layer="contact bundle",
            ),
            name="contact bundle",
        )
        output_pin = _pin(
            config.fixed_domain_initial_receipt,
            name="fixed-domain initial receipt",
            allow_pending=True,
        )
        registry_identity_value = registry.action_source_identity(config)
        registry_identity = registry.action_source_identity_sha256(config)
    except Exception as exc:
        raise FixedDomainReceiptRefused(f"action registry refused: {exc}") from exc
    if require_materialized_output and output_pin[1] is None:
        raise FixedDomainReceiptRefused(
            "fixed-domain initial receipt is not registry-materialized"
        )
    if not require_materialized_output and output_pin[1] is not None:
        raise FixedDomainReceiptRefused(
            "fixed-domain initial receipt is already registry-materialized"
        )
    expected_output = (
        f"configs/n1_fixed_domain_initial_20260802_r9/"
        f"{action_id}.fixed_domain_initial.v1.json"
    )
    if output_pin[0] != expected_output:
        raise FixedDomainReceiptRefused("registry fixed-domain planned path differs")

    bundle_raw, bundle = _pinned_json(root, bundle_pin, name="contact bundle")
    if (
        bundle.get("schema_version") != 2
        or bundle.get("artifact_type") != "n1_contact_training_bundle_v2"
        or bundle.get("action_id") != action_id
        or bundle.get("scope") != "upper"
    ):
        raise FixedDomainReceiptRefused("contact bundle identity differs")
    manifest_pin = _pin(bundle.get("manifest"), name="selected manifest")
    source_manifest_pin = _pin(bundle.get("source_manifest"), name="source manifest")
    manifest_raw, _ = _pinned_json(root, manifest_pin, name="selected manifest")
    source_manifest_raw, source_manifest_document = _pinned_json(
        root, source_manifest_pin, name="source manifest"
    )
    _validate_source_manifest_identity(
        source_manifest_document, action_id=action_id
    )

    modules = _load_mdp(root, action_id)
    manifest_module = modules["action_ball_manifest"]
    adapter = modules["action_ball_profile_adapter"]
    sampling = modules["action_ball_sampling"]
    try:
        loaded = manifest_module.load_action_ball_manifest(
            root / manifest_pin[0], expected_sha256=manifest_pin[1]
        )
        adapted = adapter.adapt_action_ball_manifest(loaded.manifest)
        curriculum = adapter.build_curriculum_config(loaded.manifest)
    except Exception as exc:
        raise FixedDomainReceiptRefused(
            f"production manifest/profile validation refused: {exc}"
        ) from exc
    if (
        loaded.manifest.action_order != (action_id,)
        or adapted.action_order != (action_id,)
        or loaded.manifest.mobility_mode != "no_move"
    ):
        raise FixedDomainReceiptRefused("selected action order or mobility differs")
    profile = adapted.profiles[0]
    if bundle.get("action_uid") != profile.action_uid:
        raise FixedDomainReceiptRefused("bundle/profile action_uid differs")
    if tuple(sampling.ARM_KEYS) != tuple(modules["action_ball_curriculum"].ARM_KEYS):
        raise FixedDomainReceiptRefused("sampling/curriculum arm order differs")
    levels = sampling.DomainLevels()
    mixture = sampling.SamplingMixture()
    if mixture.as_dict() != EXPECTED_CELL_MIXTURE:
        raise FixedDomainReceiptRefused(
            "production default SamplingMixture is not adopted 1:3:1"
        )
    active = curriculum.active_arm_keys(mobility="no_move")
    axes = []
    for arm in sampling.ARM_KEYS:
        try:
            initial, maximum, cap = sampling._arm_support_parameters(profile, arm)
            width = sampling._arm_physical_width(profile, levels, arm)
        except Exception as exc:
            raise FixedDomainReceiptRefused(
                f"runtime support helper refused {arm}: {exc}"
            ) from exc
        family, coordinate, side, unit = _axis_labels(arm)
        axes.append(
            {
                "arm": arm,
                "family": family,
                "coordinate": coordinate,
                "side": side,
                "unit": unit,
                "initial": initial,
                "maximum": maximum,
                "width": width,
                "cap": cap,
                "mask": arm in active,
            }
        )
    for row in axes:
        if row["family"] == "base_travel" and (
            row["mask"] is not False
            or any(row[key] != 0.0 for key in ("initial", "maximum", "width", "cap"))
        ):
            raise FixedDomainReceiptRefused(
                "no_move base_travel support must be masked and exactly zero"
            )

    registry_identity_payload = _strict_json(
        _canonical_bytes(registry_identity_value),
        name="registry action source identity",
    )
    if canonical_sha256(registry_identity_payload) != registry_identity:
        raise FixedDomainReceiptRefused(
            "registry action source identity payload/SHA differ"
        )
    input_paths = {
        "producer_source": PRODUCER_REPO_PATH,
        "runtime_sampling_wiring": TRAIN_REPO_PATH,
        **{f"mdp_{name}": f"{MDP_DIR}/{name}.py" for name in MDP_MODULES},
        "contact_bundle": bundle_pin[0],
        "action_manifest": manifest_pin[0],
        "source_manifest": source_manifest_pin[0],
    }
    input_raw = {
        "producer_source": source_raw[PRODUCER_REPO_PATH],
        "runtime_sampling_wiring": source_raw[TRAIN_REPO_PATH],
        **{
            f"mdp_{name}": source_raw[f"{MDP_DIR}/{name}.py"]
            for name in MDP_MODULES
        },
        "contact_bundle": bundle_raw,
        "action_manifest": manifest_raw,
        "source_manifest": source_manifest_raw,
    }
    _tracked_clean(
        root,
        (
            *source_paths,
            bundle_pin[0],
            manifest_pin[0],
            source_manifest_pin[0],
        ),
    )
    for path in source_paths:
        if _stable_read(root, path, name=f"replayed source {path}") != source_raw[path]:
            raise FixedDomainReceiptRefused(
                f"source changed during materialization: {path}"
            )
    for role, path in input_paths.items():
        if _stable_read(root, path, name=f"replayed {role}") != input_raw[role]:
            raise FixedDomainReceiptRefused(f"{role} changed during materialization")

    payload = {
        "action_id": action_id,
        "action_uid": profile.action_uid,
        "action_order": [action_id],
        "authorized_lane_ids": list(AUTHORIZED_LANES[action_id]),
        "scope": "upper",
        "planned_output_path": output_pin[0],
        "authority": {
            "mode": "fixed_initial_domain",
            "curriculum_promotion": False,
            "diagnostic_unauthorized": True,
        },
        "mobility": {"mode": "no_move", "base_travel_zero": True},
        "domain_epoch": 0,
        "domain_levels": levels.as_dict(),
        "domain_axes": axes,
        "active_arm_keys": list(active),
        "cell_mixture": {
            "value": mixture.as_dict(),
            "canonical_sha256": mixture.sha256,
        },
        "sampling_profile_sha256": profile.sha256,
        "profile_adapter_contract_sha256": adapted.contract_sha256,
        "registry_action_source_identity": registry_identity_payload,
        "registry_action_source_identity_sha256": _digest(
            registry_identity, name="registry action source identity"
        ),
        "inputs": {
            role: _input(input_paths[role], input_raw[role])
            for role in input_paths
        },
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "content": payload,
        "content_sha256": canonical_sha256(payload),
    }
    validate_receipt_document(document)
    return document


def validate_receipt_document(value: object) -> dict[str, Any]:
    """Validate a detached receipt without trusting dict permissiveness."""

    row = _exact(
        value,
        ("schema_version", "kind", "content", "content_sha256"),
        name="fixed-domain receipt",
    )
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != RECEIPT_KIND:
        raise FixedDomainReceiptRefused("receipt schema/kind differs")
    content = _exact(
        row["content"],
        (
            "action_id",
            "action_uid",
            "action_order",
            "authorized_lane_ids",
            "scope",
            "planned_output_path",
            "authority",
            "mobility",
            "domain_epoch",
            "domain_levels",
            "domain_axes",
            "active_arm_keys",
            "cell_mixture",
            "sampling_profile_sha256",
            "profile_adapter_contract_sha256",
            "registry_action_source_identity",
            "registry_action_source_identity_sha256",
            "inputs",
        ),
        name="fixed-domain content",
    )
    if _digest(row["content_sha256"], name="content_sha256") != canonical_sha256(content):
        raise FixedDomainReceiptRefused("receipt content SHA differs")
    action = content["action_id"]
    if action not in SUPPORTED_ACTIONS or content["action_order"] != [action]:
        raise FixedDomainReceiptRefused("receipt action/order differs")
    if content["authorized_lane_ids"] != list(AUTHORIZED_LANES[action]):
        raise FixedDomainReceiptRefused("receipt lane reuse contract differs")
    expected_path = (
        f"configs/n1_fixed_domain_initial_20260802_r9/"
        f"{action}.fixed_domain_initial.v1.json"
    )
    if content["planned_output_path"] != expected_path or content["scope"] != "upper":
        raise FixedDomainReceiptRefused("receipt output path/scope differs")
    if content["authority"] != {
        "mode": "fixed_initial_domain",
        "curriculum_promotion": False,
        "diagnostic_unauthorized": True,
    }:
        raise FixedDomainReceiptRefused("receipt is not fixed/non-promotable")
    if content["mobility"] != {"mode": "no_move", "base_travel_zero": True}:
        raise FixedDomainReceiptRefused("receipt mobility differs")
    if content["domain_epoch"] != 0:
        raise FixedDomainReceiptRefused("fixed initial domain epoch must be zero")
    levels = content["domain_levels"]
    axes = content["domain_axes"]
    if not isinstance(levels, Mapping) or not isinstance(axes, list):
        raise FixedDomainReceiptRefused("receipt domain levels/axes are invalid")
    arms = frozenset(levels)
    if len(arms) != 32 or set(levels.values()) != {0.0}:
        raise FixedDomainReceiptRefused("receipt must have 32 zero domain levels")
    active = content["active_arm_keys"]
    if (
        not isinstance(active, list)
        or len(active) != len(set(active))
        or not set(active).issubset(arms)
        or len(axes) != len(arms)
    ):
        raise FixedDomainReceiptRefused("receipt active arms/axis count differ")
    axis_arms = [axis.get("arm") if isinstance(axis, Mapping) else None for axis in axes]
    if len(set(axis_arms)) != len(axis_arms) or set(axis_arms) != set(arms):
        raise FixedDomainReceiptRefused("receipt axis identities differ from levels")
    for axis in axes:
        arm = axis["arm"]
        axis = _exact(
            axis,
            (
                "arm",
                "family",
                "coordinate",
                "side",
                "unit",
                "initial",
                "maximum",
                "width",
                "cap",
                "mask",
            ),
            name=f"axis {arm}",
        )
        initial = _finite(axis["initial"], name=f"{arm}.initial")
        maximum = _finite(axis["maximum"], name=f"{arm}.maximum")
        width = _finite(axis["width"], name=f"{arm}.width")
        cap = _finite(axis["cap"], name=f"{arm}.cap")
        if (
            axis["arm"] != arm
            or type(axis["mask"]) is not bool
            or axis["mask"] is not (arm in active)
            or width != min(initial, cap)
            or initial > maximum + 1.0e-12
            or maximum > cap + 1.0e-12
        ):
            raise FixedDomainReceiptRefused(f"axis {arm} support/mask differs")
        if arm.startswith("base_travel_") and (
            axis["mask"] is not False
            or any(item != 0.0 for item in (initial, maximum, width, cap))
        ):
            raise FixedDomainReceiptRefused("base_travel must be masked and zero")
    mixture = _exact(
        content["cell_mixture"],
        ("value", "canonical_sha256"),
        name="cell mixture",
    )
    if _digest(mixture["canonical_sha256"], name="mixture SHA") != canonical_sha256(
        mixture["value"]
    ):
        raise FixedDomainReceiptRefused("cell mixture SHA differs")
    if mixture["value"] != EXPECTED_CELL_MIXTURE:
        raise FixedDomainReceiptRefused("cell mixture is not adopted 1:3:1")
    registry_identity_payload = content["registry_action_source_identity"]
    if not isinstance(registry_identity_payload, Mapping):
        raise FixedDomainReceiptRefused(
            "registry action source identity must be a mapping"
        )
    if canonical_sha256(registry_identity_payload) != content[
        "registry_action_source_identity_sha256"
    ]:
        raise FixedDomainReceiptRefused(
            "registry action source identity payload/SHA differ"
        )
    for field in (
        "sampling_profile_sha256",
        "profile_adapter_contract_sha256",
        "registry_action_source_identity_sha256",
    ):
        _digest(content[field], name=field)
    inputs = content["inputs"]
    if not isinstance(inputs, Mapping) or not inputs:
        raise FixedDomainReceiptRefused("receipt inputs must be a non-empty mapping")
    for role, pin in inputs.items():
        if type(role) is not str:
            raise FixedDomainReceiptRefused("receipt input role must be text")
        _exact(pin, ("path", "sha256"), name=f"input {role}")
        _pin(pin, name=f"input {role}")
    return dict(row)


def materialize(repo_root: object, action_id: object) -> MaterializedReceipt:
    root = _root(repo_root)
    document = build_receipt(root, action_id)
    relative = document["content"]["planned_output_path"]
    path = root / relative
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FixedDomainReceiptRefused("cannot create output parent") from exc
    if parent != path.parent:
        raise FixedDomainReceiptRefused("output parent must not contain symlinks")
    raw = _canonical_bytes(document) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FixedDomainReceiptRefused(f"refusing to overwrite {path}") from exc
    except OSError as exc:
        raise FixedDomainReceiptRefused(f"cannot create {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return MaterializedReceipt(
        path=path,
        repo_path=relative,
        content_sha256=document["content_sha256"],
        file_sha256=_sha(raw),
    )


def verify_materialized(repo_root: object, action_id: object) -> MaterializedReceipt:
    """Rebuild one receipt after registry SHA backfill and require exact bytes."""

    root = _root(repo_root)
    expected = build_receipt(
        root,
        action_id,
        require_materialized_output=True,
    )
    registry = _load_file_module(
        root / REGISTRY_REPO_PATH,
        f"_fixed_domain_verify_registry_{action_id}_{id(root)}",
    )
    try:
        config = registry.get_action_config(action_id)
        output_pin = _pin(
            registry.require_materialized_pin(
                config.fixed_domain_initial_receipt,
                action_id=action_id,
                layer="fixed-domain initial receipt",
            ),
            name="fixed-domain initial receipt",
        )
    except Exception as exc:
        raise FixedDomainReceiptRefused(f"action registry refused: {exc}") from exc
    raw = _stable_read(root, output_pin[0], name="fixed-domain initial receipt")
    if _sha(raw) != output_pin[1]:
        raise FixedDomainReceiptRefused(
            "fixed-domain initial receipt SHA-256 mismatch"
        )
    document = validate_receipt_document(
        _strict_json(raw, name="fixed-domain initial receipt")
    )
    if raw != _canonical_bytes(document) + b"\n":
        raise FixedDomainReceiptRefused(
            "fixed-domain initial receipt is not canonical JSON plus newline"
        )
    if document != expected:
        raise FixedDomainReceiptRefused(
            "fixed-domain initial receipt differs from live pinned inputs"
        )
    return MaterializedReceipt(
        path=root / output_pin[0],
        repo_path=output_pin[0],
        content_sha256=document["content_sha256"],
        file_sha256=output_pin[1],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--action-id", required=True, choices=sorted(SUPPORTED_ACTIONS))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = (
            materialize(args.repo_root, args.action_id)
            if args.materialize
            else verify_materialized(args.repo_root, args.action_id)
        )
    except FixedDomainReceiptRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        _canonical_bytes(
            {
                "schema_version": 1,
                "kind": (
                    MATERIALIZATION_KIND
                    if args.materialize
                    else VERIFICATION_KIND
                ),
                "path": result.repo_path,
                "content_sha256": result.content_sha256,
                "file_sha256": result.file_sha256,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
