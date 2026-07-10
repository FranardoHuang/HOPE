"""Dependency-light helpers for the evaluator-owned Isaac BankExam adapter.

The runtime entry point lives in :mod:`isaac_bank_exam`; this module contains the parts that can
and should be audited without launching Isaac: the nominal evaluation profile, common-random-
number action noise, all-attempt ledger validation, strict JSON/CSV output and ready-state hashes.
It imports neither Isaac nor Torch at module import time.
"""

from __future__ import annotations

import csv
import copy
import dataclasses
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "hope.isaac-bank-exam.v1"
TRACKING_GUARD_FUNCTIONS = {
    "anchor_pos": ("bad_anchor_pos_z_only", "bad_anchor_pos_z_only_hold_aware"),
    "anchor_ori": ("bad_anchor_ori", "bad_anchor_ori_hold_aware"),
    "ee_body_pos": (
        "bad_motion_body_pos_z_only",
        "bad_motion_body_pos_z_only_hold_aware",
    ),
}


class IsaacBankExamError(RuntimeError):
    """Fail-closed adapter/ledger error."""


def hydrate_missing_dataclass_defaults(
    objects: Mapping[str, Any], *, allow_legacy_diagnostic: bool
) -> dict[str, Any]:
    """Fill fields absent from historical pickles using only current declared defaults."""

    hydrated: dict[str, list[str]] = {}
    for label, obj in objects.items():
        if not dataclasses.is_dataclass(obj):
            raise IsaacBankExamError(f"{label} is not a dataclass/configclass instance")
        present = vars(obj)
        added = []
        for field in dataclasses.fields(obj):
            if field.name in present:
                continue
            if field.default is not dataclasses.MISSING:
                value = copy.deepcopy(field.default)
            elif field.default_factory is not dataclasses.MISSING:
                value = field.default_factory()
            else:
                continue
            if not allow_legacy_diagnostic:
                raise IsaacBankExamError(
                    f"formal BankExam refuses config pickle missing field {label}.{field.name}"
                )
            setattr(obj, field.name, value)
            added.append(field.name)
        if added:
            hydrated[label] = sorted(added)
    return {
        "schema": "hope.historical-config-default-hydration.v1",
        "hydrated_fields": hydrated,
        "used": bool(hydrated),
    }


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite_or_none(value: Any) -> Any:
    """Convert NumPy scalars/arrays and replace non-finite floats with JSON ``null``."""

    if isinstance(value, np.ndarray):
        return [finite_or_none(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return finite_or_none(value.item())
    if isinstance(value, Mapping):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def policy_observation_tensor(raw: Any, *, device: str | None = None):
    """Normalize Isaac-Lab wrapper observation return variants to the actor tensor.

    Isaac Lab versions in the two pods return either a tensor, ``(tensor, extras)``, or a mapping
    with a ``policy`` entry.  The old Phase-1 prototype assumed a tensor and left two Kit processes
    spinning after ``AttributeError: 'tuple' object has no attribute 'to'``.
    """

    value = raw
    if isinstance(value, tuple):
        if not value:
            raise IsaacBankExamError("empty observation tuple")
        value = value[0]
    if isinstance(value, Mapping):
        if "policy" not in value:
            raise IsaacBankExamError(
                f"observation mapping has no 'policy' entry: {sorted(value)}"
            )
        value = value["policy"]
    if not hasattr(value, "shape") or not hasattr(value, "to"):
        raise IsaacBankExamError(
            f"policy observation is not a tensor-like value: {type(value).__name__}"
        )
    return value.to(device) if device is not None else value


def _set_if_present(obj: Any, name: str, value: Any, changes: dict[str, Any], prefix: str) -> None:
    if obj is not None and hasattr(obj, name):
        setattr(obj, name, value)
        changes[f"{prefix}.{name}"] = finite_or_none(value)


def apply_nominal_eval_profile(env_cfg: Any, *, num_envs: int) -> dict[str, Any]:
    """Disable unmanifested randomness and force one nominal-stand, native-speed swing.

    This runs only on the evaluator's in-memory copy of ``params/env.pkl``.  It never edits the
    saved training config and never points the training command at an exam bank.
    """

    if int(num_envs) <= 0:
        raise IsaacBankExamError("num_envs must be positive")
    changes: dict[str, Any] = {}
    scene = getattr(env_cfg, "scene", None)
    if scene is None or not hasattr(scene, "num_envs"):
        raise IsaacBankExamError("saved env cfg has no scene.num_envs")
    scene.num_envs = int(num_envs)
    changes["scene.num_envs"] = int(num_envs)

    events = getattr(env_cfg, "events", None)
    disabled_events = []
    preserved_events = []
    if events is not None:
        for name, value in vars(events).items():
            if name.startswith("_") or value is None:
                continue
            # EventTermCfg instances expose ``mode``; do not null unrelated bookkeeping fields.
            if hasattr(value, "mode"):
                if name == "add_joint_default_pos":
                    params = getattr(value, "params", None)
                    if not isinstance(params, Mapping) or "pos_distribution_params" not in params:
                        raise IsaacBankExamError(
                            "add_joint_default_pos must expose pos_distribution_params so nominal "
                            "default joints can be captured without randomization"
                        )
                    params = dict(params)
                    params["pos_distribution_params"] = None
                    value.params = params
                    preserved_events.append(name)
                else:
                    setattr(events, name, None)
                    disabled_events.append(name)
    changes["events.disabled"] = sorted(disabled_events)
    changes["events.preserved_nominal_capture"] = sorted(preserved_events)

    observations = getattr(env_cfg, "observations", None)
    policy = getattr(observations, "policy", None)
    _set_if_present(policy, "enable_corruption", False, changes, "observations.policy")

    commands = getattr(env_cfg, "commands", None)
    motion = getattr(commands, "motion", None)
    if motion is None:
        raise IsaacBankExamError("saved env cfg has no commands.motion")
    for name, value in (
        ("stand_start_prob", 1.0),
        ("post_swing_start_prob", 0.0),
        ("stand_start_min_hold", 0),
        ("stand_start_yaw_range", (0.0, 0.0)),
        ("hold_steps_range", (0, 0)),
        ("wrap_teleport", False),
        ("clip_switch_prob", 0.0),
        ("stagger_initial_clock", False),
        ("resampling_time_range", (1.0e9, 1.0e9)),
    ):
        _set_if_present(motion, name, value, changes, "commands.motion")

    speed_range = tuple(float(v) for v in getattr(motion, "speed_scale_range", (1.0, 1.0)))
    speed_per_clip = getattr(motion, "speed_scale_per_clip", None)
    if speed_range != (1.0, 1.0) or speed_per_clip is not None:
        raise IsaacBankExamError(
            "BankExam v1 requires native one-frame-per-step motion; bake retiming into the motion "
            "asset or add a versioned schedule timing contract"
        )

    racket = getattr(commands, "racket_target", None)
    if racket is None:
        raise IsaacBankExamError("saved env cfg has no commands.racket_target")
    for name, value in (
        ("midswing_resample_prob", 0.0),
        ("achieved_target_mix_prob", 0.0),
        ("target_delay_steps", 0),
        ("target_jitter_pos_per_s", 0.0),
        ("target_jitter_vel_per_s", 0.0),
        ("target_noise_white", 0.0),
        ("target_noise_ar1_sigma", 0.0),
        ("target_dropout_prob", 0.0),
        ("target_post_strike_dropout_s", 0.0),
        ("target_bias_per_swing", 0.0),
        ("shadow_ball", False),
        ("shadow_table", False),
        ("physical_ball", False),
        ("resampling_time_range", (1.0e9, 1.0e9)),
    ):
        _set_if_present(racket, name, value, changes, "commands.racket_target")

    profile = {
        "schema": "hope.isaac-bank-exam.nominal-profile.v1",
        "changes": changes,
        "exam_bank_injected_into_training_cfg": False,
        "one_environment_per_schedule_item": True,
        "ready_state": "isaac-default-stand",
    }
    profile["sha256"] = canonical_sha256(profile)
    return profile


def apply_hold_aware_tracking_guard_profile(
    env_cfg: Any,
    *,
    hold_aware_functions: Mapping[str, Any],
    allow_legacy_override: bool,
) -> dict[str, Any]:
    """Require current hold-aware guards or explicitly upgrade a historical diagnostic cfg.

    The saved YAML/pickle parity check happens before this evaluator-only mutation. Current exact
    runs must already use the hold-aware functions. Historical Phase-1 canaries may replace the
    three legacy tracking guards so nominal ready-stand holds have the same semantics as MuJoCo;
    the caller records this replacement as an inexact reason.
    """

    terms = getattr(env_cfg, "terminations", None)
    if terms is None:
        raise IsaacBankExamError("saved env cfg has no terminations")
    changes: dict[str, Any] = {"schema": "hope.hold-aware-tracking-guards.v1", "terms": {}}
    for term_name, (legacy_name, aware_name) in TRACKING_GUARD_FUNCTIONS.items():
        term = getattr(terms, term_name, None)
        if term is None:
            changes["terms"][term_name] = {"active": False}
            continue
        function = getattr(term, "func", None)
        function_name = (
            function if isinstance(function, str) else getattr(function, "__name__", None)
        )
        params = getattr(term, "params", None)
        if not isinstance(params, Mapping):
            raise IsaacBankExamError(
                f"termination {term_name} has no mutable parameter mapping"
            )
        if function_name == aware_name:
            if params.get("ignore_hold") is not True:
                raise IsaacBankExamError(
                    f"hold-aware termination {term_name} must set ignore_hold=true"
                )
            changes["terms"][term_name] = {
                "active": True,
                "function": aware_name,
                "ignore_hold": True,
                "overridden": False,
            }
            continue
        if function_name != legacy_name:
            raise IsaacBankExamError(
                f"termination {term_name} has unsupported function {function_name!r}"
            )
        if not allow_legacy_override:
            raise IsaacBankExamError(
                f"formal BankExam requires hold-aware termination {term_name}; "
                "legacy replacement is diagnostic-only"
            )
        replacement = hold_aware_functions.get(term_name)
        if getattr(replacement, "__name__", None) != aware_name:
            raise IsaacBankExamError(
                f"missing trusted hold-aware replacement {aware_name!r} for {term_name}"
            )
        updated = dict(params)
        updated["ignore_hold"] = True
        term.func = replacement
        term.params = updated
        changes["terms"][term_name] = {
            "active": True,
            "function": aware_name,
            "ignore_hold": True,
            "overridden": True,
        }
    changes["overridden_terms"] = sorted(
        name for name, value in changes["terms"].items() if value.get("overridden") is True
    )
    return changes


def configure_runtime_train_bank_loader(
    env_cfg: Any, *, allow_legacy_diagnostic: bool
) -> dict[str, Any]:
    """Inspect the saved train-bank file before gym construction and set its loader boundary."""

    commands = getattr(env_cfg, "commands", None)
    racket = getattr(commands, "racket_target", None)
    if racket is None:
        raise IsaacBankExamError("saved env cfg has no commands.racket_target")
    raw_path = str(getattr(racket, "question_bank", "") or "").strip()
    if not raw_path:
        return {"configured": False, "schema_version": None, "legacy_override": False}
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise IsaacBankExamError(f"saved runtime train bank does not exist: {path}")
    metadata = None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "meta_json" in payload:
                metadata = json.loads(
                    bytes(np.asarray(payload["meta_json"], dtype=np.uint8)).decode("utf-8")
                )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IsaacBankExamError(f"cannot inspect saved runtime train bank {path}: {exc}") from exc
    schema = metadata.get("schema_version") if isinstance(metadata, Mapping) else None
    if schema == 3:
        if metadata.get("split") != "train":
            raise IsaacBankExamError(
                f"saved runtime schema-v3 bank split={metadata.get('split')!r}, expected train"
            )
        setattr(racket, "question_bank_allow_legacy", False)
        legacy_override = False
    else:
        if not allow_legacy_diagnostic:
            raise IsaacBankExamError(
                "formal BankExam requires a schema-v3 saved runtime train bank; "
                "legacy loader override is diagnostic-only"
            )
        setattr(racket, "question_bank_allow_legacy", True)
        legacy_override = True
    return {
        "configured": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": schema,
        "split": metadata.get("split") if isinstance(metadata, Mapping) else None,
        "legacy_override": legacy_override,
    }


def configure_runtime_motion_loader(
    env_cfg: Any, *, allow_legacy_diagnostic: bool
) -> dict[str, Any]:
    """Open the decisive legacy link-origin velocity signature only for inexact replay."""

    commands = getattr(env_cfg, "commands", None)
    motion = getattr(commands, "motion", None)
    if motion is None:
        raise IsaacBankExamError("saved env cfg has no commands.motion")
    raw_files = getattr(motion, "motion_file", None)
    files = [raw_files] if isinstance(raw_files, str) else list(raw_files or ())
    if not files:
        raise IsaacBankExamError("saved motion command has no motion files")
    audited = []
    legacy_paths = []
    for raw_path in files:
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.is_file():
            raise IsaacBankExamError(f"saved runtime motion does not exist: {path}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                declared = "kinematics_schema_version" in payload.files
                suspected = False
                residual = None
                max_ang = None
                if not declared:
                    pos = np.asarray(payload["body_pos_w"], dtype=np.float64)
                    lin = np.asarray(payload["body_lin_vel_w"], dtype=np.float64)
                    ang = np.asarray(payload["body_ang_vel_w"], dtype=np.float64)
                    fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
                    link_fd = np.gradient(pos, 1.0 / fps, axis=0)
                    residual = float(np.max(np.abs(lin - link_fd)))
                    max_ang = float(np.max(np.linalg.norm(ang, axis=-1)))
                    suspected = max_ang > 0.2 and residual <= 1.0e-4
        except (OSError, KeyError, ValueError) as exc:
            raise IsaacBankExamError(f"cannot audit saved runtime motion {path}: {exc}") from exc
        if suspected:
            legacy_paths.append(str(path))
        audited.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "kinematics_declared": declared,
            "legacy_link_origin_signature": suspected,
            "link_fd_max_abs_mps": residual,
            "max_ang_radps": max_ang,
        })
    if legacy_paths and not allow_legacy_diagnostic:
        raise IsaacBankExamError(
            "formal BankExam refuses untagged link-origin velocity motions: "
            + ", ".join(legacy_paths)
        )
    setattr(motion, "allow_legacy_link_origin_velocity", bool(legacy_paths))
    return {
        "schema": "hope.runtime-motion-loader-profile.v1",
        "legacy_override": bool(legacy_paths),
        "legacy_paths": legacy_paths,
        "motions": audited,
    }


def per_attempt_action_noise(
    schedule: Sequence[Any], *, n_steps: int, action_dim: int
) -> np.ndarray:
    """Precompute one independent standard-Gaussian stream per immutable attempt seed."""

    if int(n_steps) <= 0 or int(action_dim) <= 0:
        raise IsaacBankExamError("positive n_steps/action_dim required")
    rows = []
    seen = set()
    for item in schedule:
        seed = int(item.attempt_seed)
        if seed in seen:
            raise IsaacBankExamError("attempt seeds must be unique within an exam schedule")
        seen.add(seed)
        rows.append(
            np.random.default_rng(seed).standard_normal((int(n_steps), int(action_dim)))
        )
    return np.asarray(rows, dtype=np.float64)


def ready_state_sha256(root_state: Any, joint_pos: Any, joint_vel: Any) -> str:
    """Hash one complete nominal ready-state snapshot using canonical little-endian float64."""

    digest = hashlib.sha256(b"hope-isaac-ready-state-v1\0")
    for name, value in (
        ("root_state", root_state),
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
    ):
        array = np.asarray(value, dtype="<f8")
        if not np.isfinite(array).all():
            raise IsaacBankExamError(f"ready-state {name} contains NaN/Inf")
        array = array.copy()
        array[array == 0.0] = 0.0
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    count = lambda key: sum(bool(row.get(key, False)) for row in rows)
    no_fall = n - count("physical_fall")
    return {
        "n_attempts": n,
        "n_no_fall": no_fall,
        "n_reached_exact": count("reached_exact"),
        "n_hit": count("hit"),
        "n_returned": count("returned"),
        "n_guard_reset": count("guard_reset"),
        "no_fall_rate": (no_fall / n) if n else None,
        "exact_reach_rate": (count("reached_exact") / n) if n else None,
        "hit_rate": (count("hit") / n) if n else None,
        "return_rate": (count("returned") / n) if n else None,
        "guard_reset_rate": (count("guard_reset") / n) if n else None,
        "finalize_reason_counts": dict(Counter(str(row.get("finalize_reason")) for row in rows)),
    }


def summarize_attempts(
    records: Sequence[Mapping[str, Any]], clip_names: Sequence[str]
) -> dict[str, Any]:
    summary = _group(records)
    summary["per_clip"] = {
        str(name): _group([row for row in records if int(row["clip"]) == clip])
        for clip, name in enumerate(clip_names)
    }
    return summary


def validate_complete_ledger(records: Sequence[Mapping[str, Any]], schedule: Sequence[Any]) -> None:
    if len(records) != len(schedule):
        raise IsaacBankExamError(
            f"attempt ledger has {len(records)} rows, expected schedule K={len(schedule)}"
        )
    ordered = sorted(records, key=lambda row: int(row["schedule_index"]))
    expected_indices = list(range(len(schedule)))
    got_indices = [int(row["schedule_index"]) for row in ordered]
    if got_indices != expected_indices:
        raise IsaacBankExamError(
            f"attempt ledger schedule indices are not the immutable 0..K-1 order: {got_indices}"
        )
    for row, item in zip(ordered, schedule):
        expected = {
            "clip": int(item.clip),
            "bank_row": int(item.bank_row),
            "question_id": str(item.question_id),
            "repeat": int(item.repeat),
            "hold_steps": int(item.hold_steps),
            "attempt_seed": int(item.attempt_seed),
        }
        mismatch = {key: (row.get(key), value) for key, value in expected.items()
                    if row.get(key) != value}
        if mismatch:
            raise IsaacBankExamError(
                f"attempt {item.schedule_index} no longer matches its schedule item: {mismatch}"
            )
        if not bool(row.get("finalized", False)):
            raise IsaacBankExamError(f"attempt {item.schedule_index} was not finalized")
        if bool(row.get("censored", False)):
            raise IsaacBankExamError(
                f"attempt {item.schedule_index} is externally truncated/censored; whole cell invalid"
            )


ATTEMPT_COLUMNS = (
    "schedule_index", "env_id", "clip", "side", "bank_row", "question_id", "repeat",
    "hold_steps", "attempt_seed", "ready_state_sha256", "start_step", "end_step",
    "finalize_reason", "finalized", "censored", "physical_fall", "guard_reset",
    "reached_exact", "hit", "returned", "pos_error_m", "vel_error_mps",
    "normal_error_deg", "landing_x", "landing_y", "net_clear",
)


def write_scorecard(
    *,
    output_json: str | os.PathLike[str],
    output_csv: str | os.PathLike[str],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    schedule: Sequence[Any],
    clip_names: Sequence[str],
) -> dict[str, Any]:
    """Validate and atomically write a strict all-attempt scorecard pair."""

    validate_complete_ledger(records, schedule)
    ordered = sorted((dict(row) for row in records), key=lambda row: int(row["schedule_index"]))
    document = {
        "schema": SCHEMA,
        **dict(metadata),
        "status": "valid",
        "summary": summarize_attempts(ordered, clip_names),
        "attempts": ordered,
    }
    strict = finite_or_none(document)
    out_json = Path(output_json).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = out_json.with_name(out_json.name + ".tmp")
    tmp_csv = out_csv.with_name(out_csv.name + ".tmp")
    with open(tmp_json, "w", encoding="utf-8") as stream:
        json.dump(strict, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with open(tmp_csv, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ATTEMPT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in strict["attempts"]:
            writer.writerow({key: row.get(key) for key in ATTEMPT_COLUMNS})
    os.replace(tmp_json, out_json)
    os.replace(tmp_csv, out_csv)
    return strict
