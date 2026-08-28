#!/usr/bin/env python3
"""Shared fixed-action tape and evidence format for FullMDP engine diagnosis.

Backends own scene construction and stepping.  This module owns only the
tracked SplitMix tape, a small portable state surface, durable no-clobber
records, and a comparison that reports differences without granting a physics
or readiness verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fixtures/action_ball_full_mdp_cross_engine_tape_v3.json"
CONFIG_SHA256 = "c4c393b5ff4e56d25a96e90a3b04af832405666e1a5196356b865a7ff156e954"
ARRAYS_NAME = "portable_state.npz"
SUMMARY_NAME = "summary.json"
MASK64 = (1 << 64) - 1

INITIAL_FLOAT_FIELDS = (
    "initial_root_pos",
    "initial_root_quat",
    "initial_root_lin_vel",
    "initial_joint_pos",
    "initial_joint_vel",
    "initial_racket_pos",
    "initial_racket_lin_vel",
    "initial_racket_normal",
    "initial_racket_long_axis",
)
TICK_FLOAT_FIELDS = (
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "joint_pos",
    "joint_vel",
    "racket_pos",
    "racket_lin_vel",
    "racket_normal",
    "racket_long_axis",
)
TICK_CONTROL_FLOAT_FIELDS = ("joint_qdes",)
TICK_RECORDED_FLOAT_FIELDS = (*TICK_FLOAT_FIELDS, *TICK_CONTROL_FLOAT_FIELDS)
DISCRETE_FIELDS = ("done", "time_out")


class CrossEngineTapeError(RuntimeError):
    pass


def _stable_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or path.resolve(strict=True) != path
        ):
            raise CrossEngineTapeError(f"{label} is not one canonical regular file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise CrossEngineTapeError(f"cannot read {label}") from exc
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise CrossEngineTapeError(f"{label} changed while reading")
    return payload


def load_config() -> tuple[dict, str]:
    payload = _stable_bytes(CONFIG, "cross-engine tape config")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONFIG_SHA256:
        raise CrossEngineTapeError("cross-engine tape config SHA differs")
    try:
        config = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrossEngineTapeError("cross-engine tape config is not JSON") from exc
    if (
        config.get("schema_version") != 3
        or config.get("probe_id") != "action_ball_full_mdp_cross_engine_tape_v3"
        or config.get("num_envs") != 512
        or config.get("num_ticks") != 48
        or config.get("action_width") != 31
        or config.get("environment_seed") != 0
        or config.get("action_tape", {}).get("generator")
        != "splitmix64_u24_uniform_about_pinned_dynamic_ready_v2"
        or config.get("scientific_scope")
        != {
            "diagnostic_unauthorized": True,
            "training_authorized": False,
            "checkpoint_authority": False,
            "promotion_authority": False,
            "physics_parity_authority": False,
        }
    ):
        raise CrossEngineTapeError("cross-engine tape contract differs")
    initial = config.get("initial_state")
    if initial != {
        "mode": "production_full_mdp_deterministic_reset",
        "joint_position_noise_rad": 0.0,
        "joint_velocity_noise_rad_s": 0.0,
        "root_xy_noise_m": 0.0,
        "root_yaw_noise_rad": 0.0,
    }:
        raise CrossEngineTapeError("cross-engine initial-state contract differs")
    tape = config["action_tape"]
    if (
        type(tape.get("seed")) is not int
        or not 0 <= tape["seed"] <= MASK64
        or any(
            type(tape.get(name)) not in (int, float)
            or not math.isfinite(float(tape[name]))
            for name in ("delta_low", "delta_high")
        )
        or not -1.0
        <= float(tape["delta_low"])
        < float(tape["delta_high"])
        <= 1.0
    ):
        raise CrossEngineTapeError("cross-engine action tape parameters differ")
    _action_center(config)
    return config, digest


def _action_center(config: dict) -> tuple[float, ...]:
    """Read the tape centre from the one pinned runtime birth artifact.

    The fixture owns the perturbation seed and envelope, while the dynamic-ready
    artifact owns the actor mean.  Keeping a second copied vector in the fixture
    made the diagnostic stale whenever the physical birth changed.
    """

    center = config.get("action_tape", {}).get("center")
    if (
        not isinstance(center, dict)
        or set(center)
        != {"kind", "dynamic_ready_artifact_path", "dynamic_ready_artifact_sha256"}
        or center.get("kind") != "pinned_dynamic_ready_normalized_actor_action_v2"
    ):
        raise CrossEngineTapeError("cross-engine action centre contract differs")
    relative = center.get("dynamic_ready_artifact_path")
    expected_sha = center.get("dynamic_ready_artifact_sha256")
    if (
        type(relative) is not str
        or not relative
        or Path(relative).is_absolute()
        or Path(relative).parts != tuple(part for part in relative.split("/") if part)
        or any(part in (".", "..") for part in Path(relative).parts)
        or type(expected_sha) is not str
        or len(expected_sha) != 64
    ):
        raise CrossEngineTapeError("cross-engine action centre pin differs")
    artifact_path = ROOT / relative
    payload = _stable_bytes(artifact_path, "dynamic-ready action centre artifact")
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise CrossEngineTapeError("dynamic-ready action centre artifact SHA differs")
    try:
        artifact = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrossEngineTapeError(
            "dynamic-ready action centre artifact is not JSON"
        ) from exc
    values = artifact.get("hold_candidate", {}).get("normalized_actor_action")
    if (
        artifact.get("schema_version") != 2
        or artifact.get("kind") != "agibot_a3_action_dynamic_ready_candidate_v2"
        or artifact.get("action_id") != "take_061_unit04_bh"
        or not isinstance(values, list)
        or len(values) != config.get("action_width")
        or any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in values
        )
        or not any(float(value) != 0.0 for value in values)
    ):
        raise CrossEngineTapeError("dynamic-ready action centre payload differs")
    return tuple(float(value) for value in values)


def generate_action_bytes(config: dict) -> tuple[bytes, str]:
    tape = config["action_tape"]
    state = tape["seed"]
    low = float(tape["delta_low"])
    span = float(tape["delta_high"]) - low
    center = _action_center(config)
    count = config["num_ticks"] * config["num_envs"] * config["action_width"]
    payload = bytearray()
    for flat_index in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        value ^= value >> 31
        unit = (value >> 40) / float(1 << 24)
        payload.extend(
            struct.pack(
                "<f",
                center[flat_index % config["action_width"]] + low + span * unit,
            )
        )
    header = json.dumps(
        {
            "dtype": "<f4",
            "shape": [
                config["num_ticks"],
                config["num_envs"],
                config["action_width"],
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii") + b"\n"
    return bytes(payload), hashlib.sha256(header + payload).hexdigest()


def action_tape_numpy():
    import numpy as np

    config, config_sha = load_config()
    payload, tape_sha = generate_action_bytes(config)
    array = np.frombuffer(payload, dtype="<f4").reshape(
        config["num_ticks"], config["num_envs"], config["action_width"]
    )
    return array.copy(), config, config_sha, tape_sha


def require_live_action_center(live_action, config: dict):
    """Require the backend's actual fresh actor mean to equal the tape center."""

    import numpy as np

    expected = np.asarray(_action_center(config), dtype="<f4")
    actual = np.asarray(live_action, dtype="<f4")
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise CrossEngineTapeError(
            "backend live dynamic-ready actor mean differs from tape center"
        )
    return expected.copy()


def source_identity() -> dict:
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "dirty": dirty.returncode != 0 or bool(dirty.stdout.strip()),
    }


def canonicalize_wxyz(array):
    """Make the quaternion double cover deterministic without changing rotation."""

    import numpy as np

    value = np.asarray(array).copy()
    if value.shape[-1:] != (4,):
        raise CrossEngineTapeError("quaternion array must end in width four")
    sign = np.where(value[..., :1] < 0.0, -1.0, 1.0)
    return value * sign


def _fresh_root(path: Path) -> Path:
    if (
        not path.is_absolute()
        or path.name in ("", ".", "..")
        or os.path.lexists(path)
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise CrossEngineTapeError("probe output root must be one absent absolute path")
    os.mkdir(path, 0o700)
    return path


def _write_json_x(path: Path, value: dict) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        if os.write(descriptor, payload) != len(payload):
            raise CrossEngineTapeError("short JSON write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _required_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    n, h, a = config["num_envs"], config["num_ticks"], config["action_width"]
    widths = {
        "root_pos": 3,
        "root_quat": 4,
        "root_lin_vel": 3,
        "joint_pos": a,
        "joint_vel": a,
        "racket_pos": 3,
        "racket_lin_vel": 3,
        "racket_normal": 3,
        "racket_long_axis": 3,
    }
    shapes = {"actions": (h, n, a)}
    for name, width in widths.items():
        shapes[name] = (h, n, width)
        shapes["initial_" + name] = (n, width)
    shapes["joint_qdes"] = (h, n, a)
    shapes.update({name: (h, n) for name in DISCRETE_FIELDS})
    return shapes


def write_probe_record(
    output_root: Path,
    *,
    backend: str,
    arrays: dict,
    joint_names: list[str] | tuple[str, ...],
    runtime_identity: dict,
    source_identity_at_start: dict,
) -> dict:
    import numpy as np

    if backend not in ("isaac", "mujoco"):
        raise CrossEngineTapeError("backend must be isaac or mujoco")
    if (
        not isinstance(source_identity_at_start, dict)
        or set(source_identity_at_start) != {"commit", "dirty"}
        or type(source_identity_at_start.get("commit")) is not str
        or not source_identity_at_start["commit"]
        or type(source_identity_at_start.get("dirty")) is not bool
    ):
        raise CrossEngineTapeError("portable probe start-source identity differs")
    tape, config, config_sha, tape_sha = action_tape_numpy()
    expected = _required_shapes(config)
    if set(arrays) != set(expected) or any(
        np.asarray(arrays[name]).shape != shape for name, shape in expected.items()
    ):
        raise CrossEngineTapeError("portable probe array surface differs")
    owned = {name: np.asarray(value).copy() for name, value in arrays.items()}
    if owned["actions"].dtype != np.dtype("<f4") or not np.array_equal(
        owned["actions"], tape
    ):
        raise CrossEngineTapeError("portable probe action tape differs")
    for name in (*INITIAL_FLOAT_FIELDS, *TICK_RECORDED_FLOAT_FIELDS):
        if not np.issubdtype(owned[name].dtype, np.floating) or not np.isfinite(
            owned[name]
        ).all():
            raise CrossEngineTapeError(f"portable probe {name} is not finite floating data")
    for name in DISCRETE_FIELDS:
        if not (
            np.issubdtype(owned[name].dtype, np.bool_)
            or np.issubdtype(owned[name].dtype, np.integer)
        ):
            raise CrossEngineTapeError(f"portable probe {name} is not discrete")
    names = [str(value) for value in joint_names]
    if len(names) != config["action_width"] or len(set(names)) != len(names):
        raise CrossEngineTapeError("portable probe joint order differs")
    owned["initial_root_quat"] = canonicalize_wxyz(owned["initial_root_quat"])
    owned["root_quat"] = canonicalize_wxyz(owned["root_quat"])
    root = _fresh_root(Path(output_root))
    descriptor = os.open(
        root / ARRAYS_NAME,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            np.savez_compressed(stream, **owned)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    payload = _stable_bytes(root / ARRAYS_NAME, "portable probe arrays")
    summary = {
        "schema_version": 3,
        "record_type": "action_ball_full_mdp_cross_engine_tape_probe_v3",
        "diagnostic_unauthorized": True,
        "training_authorized": False,
        "checkpoint_authority": False,
        "promotion_authority": False,
        "physics_parity_authority": False,
        "backend": backend,
        "source": dict(source_identity_at_start),
        "runtime_identity": runtime_identity,
        "config_sha256": config_sha,
        "action_tape_sha256": tape_sha,
        "joint_names": names,
        "shape": {
            "num_envs": config["num_envs"],
            "num_ticks": config["num_ticks"],
            "action_width": config["action_width"],
        },
        "arrays_sha256": hashlib.sha256(payload).hexdigest(),
        "done_rows": int(np.asarray(owned["done"], dtype=np.int64).sum()),
        "time_out_rows": int(np.asarray(owned["time_out"], dtype=np.int64).sum()),
    }
    _write_json_x(root / SUMMARY_NAME, summary)
    return summary


def _load_record(root: Path):
    import numpy as np

    summary = json.loads(_stable_bytes(root / SUMMARY_NAME, "probe summary"))
    payload = _stable_bytes(root / ARRAYS_NAME, "probe arrays")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    config, config_sha = load_config()
    _payload, tape_sha = generate_action_bytes(config)
    if (
        summary.get("schema_version") != 3
        or summary.get("record_type")
        != "action_ball_full_mdp_cross_engine_tape_probe_v3"
        or summary.get("backend") not in ("isaac", "mujoco")
        or summary.get("config_sha256") != config_sha
        or summary.get("action_tape_sha256") != tape_sha
        or summary.get("arrays_sha256") != hashlib.sha256(payload).hexdigest()
        or set(arrays) != set(_required_shapes(config))
    ):
        raise CrossEngineTapeError("portable probe record identity differs")
    for name, shape in _required_shapes(config).items():
        if arrays[name].shape != shape:
            raise CrossEngineTapeError(f"portable probe shape differs: {name}")
    return summary, arrays


def _first_exact_difference(left, right):
    import numpy as np

    mismatch = np.asarray(left != right)
    if not mismatch.any():
        return None
    index = tuple(int(value) for value in np.argwhere(mismatch)[0])
    return index


def compare_records(isaac_root: Path, mujoco_root: Path, output: Path) -> dict:
    import numpy as np

    isaac, left = _load_record(isaac_root)
    mujoco, right = _load_record(mujoco_root)
    if isaac["backend"] != "isaac" or mujoco["backend"] != "mujoco":
        raise CrossEngineTapeError("comparison backend roles differ")
    if isaac["joint_names"] != mujoco["joint_names"] or not np.array_equal(
        left["actions"], right["actions"]
    ):
        raise CrossEngineTapeError("comparison inputs do not share joint order/tape")
    numeric = {}
    exact_first = []
    for name in (*INITIAL_FLOAT_FIELDS, *TICK_RECORDED_FLOAT_FIELDS):
        a = left[name].astype(np.float64)
        b = right[name].astype(np.float64)
        delta = np.abs(a - b)
        first = _first_exact_difference(left[name], right[name])
        numeric[name] = {
            "exact": first is None,
            "first_exact_difference_index": first,
            "max_abs": float(delta.max()) if delta.size else 0.0,
            "mean_abs": float(delta.mean()) if delta.size else 0.0,
        }
        if first is not None:
            tick = -1 if name.startswith("initial_") else first[0]
            exact_first.append((tick, name, first))
    discrete = {}
    for name in DISCRETE_FIELDS:
        first = _first_exact_difference(left[name], right[name])
        discrete[name] = {
            "mismatch_cells": int(np.count_nonzero(left[name] != right[name])),
            "first_mismatch_index": first,
        }
        if first is not None:
            exact_first.append((first[0], name, first))
    first = None
    if exact_first:
        tick, name, index = sorted(exact_first, key=lambda row: (row[0], row[1]))[0]
        first = {"phase": "initial" if tick < 0 else "post_step", "tick": tick, "field": name, "index": index}
    record = {
        "schema_version": 3,
        "record_type": "action_ball_full_mdp_cross_engine_tape_comparison_v3",
        "diagnostic_unauthorized": True,
        "promotion_authority": False,
        "physics_parity_authority": False,
        "same_exact_action_tape": True,
        "same_joint_order": True,
        "isaac_source": isaac["source"],
        "mujoco_source": mujoco["source"],
        "first_exact_difference": first,
        "numeric_difference_envelope": numeric,
        "discrete_difference": discrete,
    }
    _write_json_x(output, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--isaac-root", type=Path, required=True)
    compare.add_argument("--mujoco-root", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = compare_records(args.isaac_root, args.mujoco_root, args.output)
    except (CrossEngineTapeError, OSError, ValueError) as exc:
        print("FULLMDP_CROSS_ENGINE_TAPE_ERROR=" + str(exc), flush=True)
        return 2
    print(
        "FULLMDP_CROSS_ENGINE_TAPE_COMPARISON_JSON="
        + json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
