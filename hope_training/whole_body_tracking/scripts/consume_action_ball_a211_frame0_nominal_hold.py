#!/usr/bin/env python3
"""Consume one exact live Isaac hold into an A211 frame-0 receipt.

This consumer never runs Isaac and never upgrades a failed or incomplete live
receipt.  It closes the provenance gap between the generic
``isaac_action_ball_nominal_hold_v1`` producer and the A211 lineage by
revalidating the exact frame-0 candidate, the deterministically derived probe
input, all live safety telemetry, and both relevant Git source commits before
publishing one canonical no-clobber receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER_FILE = SCRIPT_DIR / "launch_action_ball_a211_four_arm_diagnostic.py"
PROBE_FILE = SCRIPT_DIR / "check_table_obstacle_scene.py"
WRAPPER_FILE = SCRIPT_DIR / "run_action_ball_a211_frame0_nominal_hold.py"
GENERIC_RECEIPT_KIND = "isaac_action_ball_nominal_hold_v1"
PROBE_INPUT_KIND = "action_ball_a211_frame0_nominal_hold_probe_input_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
REQUIRED_TERMINATIONS = (
    "robot_hit_table",
    "base_fell_tilt",
    "base_too_low",
    "joint_qdes_forbidden",
    "joint_actual_forbidden",
)


class ReceiptError(RuntimeError):
    """An input or publication did not close the exact live receipt."""


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "_a211_frame0_receipt_launcher", LAUNCHER_FILE
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot import A211 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_L = _load_launcher()


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReceiptError("value is not finite canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object, *, name: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ReceiptError("%s must be one lowercase SHA-256" % name)
    return value


def _commit(value: object, *, name: str) -> str:
    if type(value) is not str or COMMIT_RE.fullmatch(value) is None:
        raise ReceiptError("%s must be one full lowercase Git commit" % name)
    return value


def _relative(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ReceiptError("%s must be a non-empty POSIX relative path" % name)
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ReceiptError("%s must be a normalized relative path" % name)
    return pure.as_posix()


def _regular(path: Path, *, name: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReceiptError("cannot inspect %s: %s" % (name, exc)) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ReceiptError("%s must be a regular non-symlink file" % name)


def _strict_json(
    path: Path, *, name: str, newline: Optional[bool], canonical: bool = True
) -> tuple[dict[str, Any], bytes]:
    _regular(path, name=name)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReceiptError("%s is not strict JSON" % name) from exc
    if type(value) is not dict:
        raise ReceiptError("%s must be one JSON object" % name)
    if not canonical:
        canonical_bytes(value)  # validate finite, serializable JSON values
        return value, raw
    expected = canonical_bytes(value)
    allowed = (expected, expected + b"\n") if newline is None else (
        (expected + b"\n",) if newline else (expected,)
    )
    if raw not in allowed:
        raise ReceiptError("%s is not canonical JSON" % name)
    return value, raw


def _verify_seal(document: Mapping[str, Any], *, name: str) -> str:
    seal = _sha(document.get("content_sha256"), name=name + ".content_sha256")
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    if canonical_sha256(unsigned) != seal:
        raise ReceiptError("%s content seal is not reproducible" % name)
    return seal


def _git(root: Path, args: Sequence[str], *, binary: bool = False):
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not binary,
            check=False,
        )
    except OSError as exc:
        raise ReceiptError("cannot execute Git: %s" % exc) from exc


def verify_exact_clean_source(root: Path, source_commit: str) -> None:
    """Require a clean exact checkout before a live receipt is consumed."""

    source_commit = _commit(source_commit, name="probe_source_commit")
    head = _git(root, ("rev-parse", "HEAD"))
    if head.returncode or head.stdout.strip() != source_commit:
        raise ReceiptError("probe source commit is not the checkout HEAD")
    dirty = _git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    if dirty.returncode or dirty.stdout.strip():
        raise ReceiptError("exact Pod probe requires a clean checkout")
    for path in (PROBE_FILE, WRAPPER_FILE, Path(__file__).resolve()):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ReceiptError("probe source escaped repository root") from exc
        current = path.read_bytes()
        committed = _git(root, ("show", source_commit + ":" + relative), binary=True)
        if committed.returncode or committed.stdout != current:
            raise ReceiptError("probe source differs from exact source commit: %s" % relative)


def _tracked_input(
    root: Path,
    relative: str,
    expected_sha: str,
    source_commit: str,
    *,
    name: str,
) -> Path:
    relative = _relative(relative, name=name + " path")
    expected_sha = _sha(expected_sha, name=name + " SHA-256")
    path = root / relative
    _regular(path, name=name)
    if path.resolve(strict=True) != path:
        raise ReceiptError("%s path traverses a symlink" % name)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ReceiptError("%s file SHA-256 differs" % name)
    committed = _git(root, ("show", source_commit + ":" + relative), binary=True)
    if committed.returncode or committed.stdout != raw:
        raise ReceiptError("%s differs from probe source commit" % name)
    return path


def _finite_vector(value: object, count: int, *, name: str) -> list[float]:
    if type(value) is not list or len(value) != count:
        raise ReceiptError("%s must contain %d values" % (name, count))
    result = []
    for item in value:
        if type(item) not in (int, float) or isinstance(item, bool):
            raise ReceiptError("%s must be numeric" % name)
        number = float(item)
        if not math.isfinite(number):
            raise ReceiptError("%s must be finite" % name)
        result.append(number)
    return result


def validate_frame0_artifact(
    document: Mapping[str, Any], *, motion_sha256: str
) -> tuple[str, Mapping[str, Any]]:
    expected_keys = {
        "schema_version", "kind", "diagnostic_unauthorized", "source_kind",
        "action_id", "motion_sha256", "policy_dt_s",
        "wait_schedule_canonical_sha256", "timing_receipt", "birth_horizon",
        "frame0", "content_sha256",
    }
    if set(document) != expected_keys:
        raise ReceiptError("frame0 artifact keys differ")
    seal = _verify_seal(document, name="frame0 artifact")
    if (
        document["schema_version"] != 2
        or document["kind"] != _L.FRAME0_EXACT_ARTIFACT_KIND
        or document["diagnostic_unauthorized"] is not True
        or document["source_kind"] != _L.FRAME0_EXACT_SOURCE_KIND
        or document["motion_sha256"] != motion_sha256
        or document["policy_dt_s"] != _L.POLICY_DT_S
        or document["wait_schedule_canonical_sha256"]
        != _L.WAIT_SCHEDULE["canonical_sha256"]
    ):
        raise ReceiptError("frame0 artifact contract differs")
    timing_pin = document["timing_receipt"]
    horizon = document["birth_horizon"]
    horizon_keys = {
        "schema_version", "kind", "derivation",
        "timing_receipt_canonical_sha256", "policy_dt_s",
        "post_reset_coverage_policy_ticks", "max_reset_wait_policy_ticks",
        "pre_swing_wait_s", "pre_swing_wait_policy_ticks_ceil",
        "required_policy_ticks",
    }
    if (
        type(timing_pin) is not dict
        or set(timing_pin) != {"path", "sha256"}
        or type(timing_pin["path"]) is not str
        or not timing_pin["path"]
        or SHA256_RE.fullmatch(str(timing_pin["sha256"])) is None
        or type(horizon) is not dict
        or set(horizon) != horizon_keys
    ):
        raise ReceiptError("frame0 birth-horizon authority is incomplete")
    try:
        pre_wait = float(horizon["pre_swing_wait_s"])
        pre_wait_ticks = int(math.ceil(pre_wait / _L.POLICY_DT_S))
        required_ticks = (
            1 + int(_L.WAIT_SCHEDULE["max_wait_ticks"]) + pre_wait_ticks
        )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ReceiptError("frame0 birth-horizon timing is malformed") from exc
    if (
        not math.isfinite(pre_wait)
        or pre_wait < 0.0
        or horizon["schema_version"] != 1
        or horizon["kind"] != "action_ball_frame0_dynamic_birth_horizon_v1"
        or horizon["derivation"]
        != "post_reset_coverage_plus_max_reset_wait_plus_ceil_pre_swing_wait"
        or SHA256_RE.fullmatch(
            str(horizon["timing_receipt_canonical_sha256"])
        ) is None
        or horizon["policy_dt_s"] != _L.POLICY_DT_S
        or horizon["post_reset_coverage_policy_ticks"] != 1
        or horizon["max_reset_wait_policy_ticks"]
        != _L.WAIT_SCHEDULE["max_wait_ticks"]
        or horizon["pre_swing_wait_policy_ticks_ceil"] != pre_wait_ticks
        or horizon["required_policy_ticks"] != required_ticks
    ):
        raise ReceiptError("frame0 birth horizon is not the sealed timing derivation")
    frame0 = document["frame0"]
    if type(frame0) is not dict or set(frame0) != {
        "root_pos_w_m", "root_quat_wxyz", "root_lin_vel_w_mps",
        "root_ang_vel_w_radps", "joint_pos_rad", "joint_vel_radps",
    }:
        raise ReceiptError("frame0 exact-state keys differ")
    _finite_vector(frame0["root_pos_w_m"], 3, name="frame0 root position")
    quat = _finite_vector(frame0["root_quat_wxyz"], 4, name="frame0 root quaternion")
    if not math.isclose(sum(value * value for value in quat), 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise ReceiptError("frame0 root quaternion is not unit length")
    _finite_vector(frame0["joint_pos_rad"], 31, name="frame0 joint position")
    for key, count in (
        ("root_lin_vel_w_mps", 3),
        ("root_ang_vel_w_radps", 3),
        ("joint_vel_radps", 31),
    ):
        values = _finite_vector(frame0[key], count, name="frame0 " + key)
        if values != [0.0] * count:
            raise ReceiptError("frame0 exact candidate must have zero velocity")
    return seal, frame0


def derive_probe_input(
    *,
    frame0_artifact: Mapping[str, Any],
    frame0_file_sha256: str,
    artifact_source_commit: str,
    plant_template: Mapping[str, Any],
    plant_template_file_sha256: str,
    motion_path: Path,
    motion_sha256: str,
    probe_source_commit: str,
) -> dict[str, Any]:
    """Build the sole dynamic-ready-shaped input accepted for this exact probe."""

    artifact_content_sha, frame0 = validate_frame0_artifact(
        frame0_artifact, motion_sha256=motion_sha256
    )
    template_content_sha = _verify_seal(plant_template, name="plant template")
    try:
        robot = plant_template["robot"]
        runtime = plant_template["runtime_plant"]
        default = _finite_vector(runtime["default_joint_pos_rad"], 31, name="default q")
        scale = _finite_vector(runtime["action_scale_rad"], 31, name="action scale")
        limits = runtime["qdes_joint_pos_limits"]
        inset = float(runtime["finite_projection_soft_envelope_inset_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReceiptError("plant template core fields are invalid") from exc
    names = robot.get("joint_names") if isinstance(robot, Mapping) else None
    if (
        plant_template.get("schema_version") != 2
        or plant_template.get("kind") != "agibot_a3_action_dynamic_ready_candidate_v2"
        or plant_template.get("action_id") != frame0_artifact["action_id"]
        or not isinstance(names, list)
        or len(names) != 31
        or len(set(names)) != 31
        or not isinstance(limits, list)
        or len(limits) != 31
        or not math.isfinite(inset)
        or inset < 0.0
        or any(value == 0.0 for value in scale)
    ):
        raise ReceiptError("plant template is not the exact A3 N=1 action plant")
    q = _finite_vector(frame0["joint_pos_rad"], 31, name="frame0 q")
    normalized = []
    for index, (target, base, gain, pair) in enumerate(zip(q, default, scale, limits)):
        bounds = _finite_vector(pair, 2, name="qdes bounds")
        lower, upper = bounds
        soft_lower = lower + inset * (upper - lower)
        soft_upper = upper - inset * (upper - lower)
        if not lower < upper or not soft_lower < target < soft_upper:
            raise ReceiptError("frame0 q is outside the plant qdes envelope at joint %d" % index)
        action = (target - base) / gain
        if not math.isfinite(action) or not math.isclose(
            base + gain * action, target, rel_tol=0.0, abs_tol=2.0e-7
        ):
            raise ReceiptError("frame0 hold action does not decode at joint %d" % index)
        normalized.append(action)
    unsigned = {
        "schema_version": 2,
        "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
        "action_id": frame0_artifact["action_id"],
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "isaac_nominal_hold_validated": False,
        },
        "robot": robot,
        "runtime_plant": runtime,
        "physical_ready": {
            "root_pos_w_m": frame0["root_pos_w_m"],
            "root_quat_wxyz": frame0["root_quat_wxyz"],
            "root_lin_vel_w_mps": frame0["root_lin_vel_w_mps"],
            "root_ang_vel_w_radps": frame0["root_ang_vel_w_radps"],
            "joint_pos_rad": frame0["joint_pos_rad"],
            "joint_vel_radps": frame0["joint_vel_radps"],
        },
        "hold_candidate": {
            "semantics": "hold_exact_frame0_qdes_with_zero_initial_velocity",
            "hold_qdes_joint_pos_rad": frame0["joint_pos_rad"],
            "normalized_actor_action": normalized,
        },
        "ready_source": {
            "schema_version": 1,
            "kind": PROBE_INPUT_KIND,
            "diagnostic_unauthorized": True,
            "same_exact_teacher_and_physical_frame0": True,
        },
        "sources": {
            "stable_motion": {
                "path": str(motion_path),
                "sha256": motion_sha256,
                "frame_index": 0,
            },
            "frame0_exact_artifact": {
                "file_sha256": frame0_file_sha256,
                "content_sha256": artifact_content_sha,
                "artifact_source_commit": artifact_source_commit,
                "birth_horizon": frame0_artifact["birth_horizon"],
                "timing_receipt": frame0_artifact["timing_receipt"],
            },
            "plant_template": {
                "file_sha256": plant_template_file_sha256,
                "content_sha256": template_content_sha,
            },
            "probe_source_commit": probe_source_commit,
        },
        "required_next_gate": {
            "kind": GENERIC_RECEIPT_KIND,
            "exact_policy_steps": frame0_artifact["birth_horizon"][
                "required_policy_ticks"
            ],
            "zero_terminal_required": list(REQUIRED_TERMINATIONS),
        },
        "non_claims": [
            "not training deployment promotion export or hardware authorization",
            "not a PASS until the exact live Isaac receipt is consumed",
        ],
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


# [已删除 2026-08-06 过期结构清理] _same_numbers(5 行):零调用点。
# 它是"逐向量比对"的旧写法;现役校验是下面 validate_probe_input 的 ``probe != expected``
# 整字典相等,覆盖面严格更大(连非数值字段一起管),所以删它不放宽任何门。
def validate_probe_input(
    probe: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> str:
    seal = _verify_seal(probe, name="probe input")
    if probe != expected:
        raise ReceiptError("probe input is not the deterministic exact frame0 derivation")
    return seal


def validate_live_receipt(
    document: Mapping[str, Any],
    *,
    raw_path: Path,
    probe_path: Path,
    probe_file_sha256: str,
    probe_content_sha256: str,
    frame0_artifact: Mapping[str, Any],
    joint_names: Sequence[str],
    control_decimation: int,
    control_step_action_delay: Mapping[str, Any],
) -> tuple[str, str]:
    content_sha = _verify_seal(document, name="live safety evidence")
    raw_file_sha = sha256_file(raw_path)
    artifact = document.get("artifact")
    ticks = frame0_artifact["birth_horizon"]["required_policy_ticks"]
    policy_dt = frame0_artifact["policy_dt_s"]
    expected_duration = ticks * policy_dt
    joint = document.get("joint_safety_telemetry")
    delay = document.get("control_step_action_delay_runtime")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != GENERIC_RECEIPT_KIND
        or document.get("verdict") != "PASS"
        or document.get("action_id") != frame0_artifact["action_id"]
        or document.get("motion_sha256") != frame0_artifact["motion_sha256"]
        or document.get("teacher_reference_unchanged") is not True
        or document.get("teacher_physical_birth_separated") is not False
        or document.get("candidate_physical_birth_written") is not True
        or document.get("candidate_hold_qdes_and_delay_history_installed") is not True
        or document.get("plant_contract_match") is not True
        or document.get("terminal_reasons") != []
        or document.get("generic_terminated") is not False
        or document.get("generic_truncated") is not False
        or type(document.get("completed_policy_steps")) is not int
        or document.get("completed_policy_steps") != ticks
        or type(document.get("completed_physics_steps")) is not int
        or document.get("completed_physics_steps") != ticks * control_decimation
        or not math.isclose(float(document.get("requested_duration_s", math.nan)), expected_duration, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(float(document.get("completed_duration_s", math.nan)), expected_duration, rel_tol=0.0, abs_tol=1.0e-12)
        or not isinstance(artifact, Mapping)
        or artifact.get("path") != str(probe_path)
        or artifact.get("sha256") != probe_file_sha256
        or artifact.get("content_sha256") != probe_content_sha256
        or not isinstance(document.get("active_terminations"), list)
        or any(name not in document["active_terminations"] for name in REQUIRED_TERMINATIONS)
        or not isinstance(joint, Mapping)
        or joint.get("schema_version") != 1
        or joint.get("complete") is not True
        or joint.get("joint_order") != list(joint_names)
        or joint.get("current_actual_hard_edge_joint_count") != 0
        or joint.get("current_actual_hard_edge_joint_names") != []
        or joint.get("substep_actual_hard_edge_joint_count") != 0
        or joint.get("substep_actual_hard_edge_joint_names") != []
        or not isinstance(delay, Mapping)
        or delay.get("schema_version") != 1
        or delay.get("kind") != "whole_body_tracking.policy_control_step_action_delay_receipt"
        or delay.get("num_envs") != 1
        or delay.get("initialized_env_count") != 1
        or delay.get("contract") != control_step_action_delay
    ):
        raise ReceiptError("live receipt does not prove the exact dynamic birth horizon")
    for key in (
        "minimum_root_z_m",
        "maximum_root_tilt_rad",
        "both_feet_contact_fraction",
    ):
        value = document.get(key)
        if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ReceiptError("live receipt %s is not finite actual telemetry" % key)
    minimum_gap = joint.get("final_minimum_hard_gap_rad")
    if (
        type(minimum_gap) not in (int, float)
        or isinstance(minimum_gap, bool)
        or not math.isfinite(float(minimum_gap))
        or float(minimum_gap) <= 0.0
    ):
        raise ReceiptError("live receipt lacks a positive final hard-limit gap")
    for key in (
        "preterminal_joint_pos_rad", "preterminal_joint_vel_radps",
        "final_joint_pos_rad", "final_joint_vel_radps", "hard_lower_rad", "hard_upper_rad",
    ):
        _finite_vector(joint.get(key), 31, name="live joint safety " + key)
    screenshots = document.get("screenshots")
    labels = (
        "raw_env_reset", "physical_ready_after_reset_write",
        "after_step_1", "after_step_10", "final",
    )
    if not isinstance(screenshots, list) or tuple(row.get("label") for row in screenshots if isinstance(row, Mapping)) != labels:
        raise ReceiptError("live receipt requires the five exact hold screenshots")
    for row in screenshots:
        path = Path(str(row.get("path")))
        _regular(path, name="live screenshot")
        if sha256_file(path) != _sha(row.get("sha256"), name="screenshot SHA-256"):
            raise ReceiptError("live screenshot bytes differ")
    return raw_file_sha, content_sha


def _dynamic_birth_gate_evidence(
    live: Mapping[str, Any], *, policy_dt_s: float
) -> dict[str, Any]:
    """Project velocity one policy tick forward inside the observed hard limits."""

    joint = live["joint_safety_telemetry"]
    names = joint["joint_order"]
    lower = joint["hard_lower_rad"]
    upper = joint["hard_upper_rad"]
    rows = []
    for label in ("preterminal", "final"):
        q = joint[label + "_joint_pos_rad"]
        dq = joint[label + "_joint_vel_radps"]
        projected = [p + policy_dt_s * v for p, v in zip(q, dq)]
        gaps = [
            min(value - lo, hi - value)
            for value, lo, hi in zip(projected, lower, upper)
        ]
        minimum = min(gaps)
        index = gaps.index(minimum)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ReceiptError(
                "%s velocity projection has no forward hard-gap headroom" % label
            )
        rows.append(
            {
                "state": label,
                "minimum_forward_hard_gap_rad": minimum,
                "minimum_forward_hard_gap_joint_name": names[index],
            }
        )
    terminations = set(live["terminal_reasons"])
    return {
        "schema_version": 1,
        "kind": "action_ball_frame0_dynamic_birth_gate_evidence_v1",
        "thresholds_preregistered": {
            "table_contact_count_max": 0,
            "nonfinite_count_max": 0,
            "actual_hard_edge_joint_count_max": 0,
            "minimum_forward_hard_gap_rad_exclusive_min": 0.0,
        },
        "observed": {
            "table_contact_count": int("robot_hit_table" in terminations),
            "nonfinite_count": 0,
            "current_actual_hard_edge_joint_count": joint[
                "current_actual_hard_edge_joint_count"
            ],
            "substep_actual_hard_edge_joint_count": joint[
                "substep_actual_hard_edge_joint_count"
            ],
            "forward_headroom": rows,
        },
        "nominal_scope": {
            "actor_bias": "exact_frame0_normalized_action",
            "per_env_joint_default_offset_dr_preserved": True,
            "per_env_joint_default_offset_range_rad": [-0.01, 0.01],
            "full_dr_distribution_hold_pass_claimed": False,
        },
    }


def _write_new(root: Path, relative: str, payload: bytes) -> dict[str, str]:
    relative = _relative(relative, name="output")
    output = root / relative
    if output.exists() or output.is_symlink():
        raise ReceiptError("no-clobber output already exists: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent:
        raise ReceiptError("output parent traverses a symlink")
    descriptor = os.open(
        str(output),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    created = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(output, 0o444)
        readback = output.read_bytes()
        if readback != payload:
            raise ReceiptError("output durable readback differs")
        return {"path": relative, "sha256": hashlib.sha256(readback).hexdigest()}
    except Exception:
        if created:
            try:
                os.chmod(output, 0o644)
                output.unlink()
            except OSError:
                pass
        raise


def consume(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve(strict=True)
    probe_source_commit = _commit(args.probe_source_commit, name="probe_source_commit")
    artifact_source_commit = _commit(args.artifact_source_commit, name="artifact_source_commit")
    verify_exact_clean_source(root, probe_source_commit)
    ancestor = _git(
        root,
        ("merge-base", "--is-ancestor", artifact_source_commit, probe_source_commit),
    )
    if ancestor.returncode != 0:
        raise ReceiptError("artifact source commit is not a probe source ancestor")
    frame_path = _tracked_input(
        root, args.frame0_artifact_path, args.expected_frame0_artifact_sha256,
        probe_source_commit, name="frame0 artifact",
    )
    template_path = _tracked_input(
        root, args.plant_template_path, args.expected_plant_template_sha256,
        probe_source_commit, name="plant template",
    )
    motion_path = _tracked_input(
        root, args.motion_path, args.expected_motion_sha256,
        probe_source_commit, name="motion",
    )
    frame, _frame_raw = _strict_json(frame_path, name="frame0 artifact", newline=True)
    template, _template_raw = _strict_json(template_path, name="plant template", newline=None)
    validate_frame0_artifact(frame, motion_sha256=args.expected_motion_sha256)
    timing_pin = frame["timing_receipt"]
    timing_path = _tracked_input(
        root,
        timing_pin["path"],
        timing_pin["sha256"],
        probe_source_commit,
        name="sealed timing receipt",
    )
    timing, timing_raw = _strict_json(
        timing_path, name="sealed timing receipt", newline=True
    )
    timing_unsigned = dict(timing)
    timing_seal = timing_unsigned.pop("canonical_sha256", None)
    if (
        timing_raw != canonical_bytes(timing) + b"\n"
        or timing_seal != frame["birth_horizon"]["timing_receipt_canonical_sha256"]
        or timing_seal != canonical_sha256(timing_unsigned)
        or timing.get("schema_version") != 5
        or timing.get("motion_sha256") != frame["motion_sha256"]
        or timing.get("contact_time_step_s") != frame["policy_dt_s"]
        or timing.get("pre_swing_wait_s")
        != frame["birth_horizon"]["pre_swing_wait_s"]
    ):
        raise ReceiptError("sealed timing receipt differs from the birth horizon")
    committed_artifact = _git(
        root,
        ("show", artifact_source_commit + ":" + _relative(args.frame0_artifact_path, name="frame0 artifact path")),
        binary=True,
    )
    if committed_artifact.returncode or committed_artifact.stdout != frame_path.read_bytes():
        raise ReceiptError("artifact source commit does not contain the exact frame0 bytes")
    probe_path = Path(args.probe_input).resolve(strict=True)
    raw_path = Path(args.live_receipt).resolve(strict=True)
    probe, _probe_raw = _strict_json(probe_path, name="probe input", newline=True)
    live, _live_raw = _strict_json(raw_path, name="live safety evidence", newline=False)
    expected_probe = derive_probe_input(
        frame0_artifact=frame,
        frame0_file_sha256=sha256_file(frame_path),
        artifact_source_commit=artifact_source_commit,
        plant_template=template,
        plant_template_file_sha256=sha256_file(template_path),
        motion_path=motion_path,
        motion_sha256=args.expected_motion_sha256,
        probe_source_commit=probe_source_commit,
    )
    probe_content_sha = validate_probe_input(probe, expected=expected_probe)
    runtime = template["runtime_plant"]
    live_file_sha, live_content_sha = validate_live_receipt(
        live,
        raw_path=raw_path,
        probe_path=probe_path,
        probe_file_sha256=sha256_file(probe_path),
        probe_content_sha256=probe_content_sha,
        frame0_artifact=frame,
        joint_names=template["robot"]["joint_names"],
        control_decimation=int(runtime["control_decimation"]),
        control_step_action_delay=runtime["control_step_action_delay"],
    )
    unsigned = {
        "schema_version": 2,
        "kind": _L.FRAME0_EXACT_RECEIPT_KIND,
        "diagnostic_unauthorized": True,
        "source_kind": _L.FRAME0_EXACT_SOURCE_KIND,
        "verdict": "PASS",
        "action_id": frame["action_id"],
        "motion_sha256": frame["motion_sha256"],
        "artifact_file_sha256": sha256_file(frame_path),
        "artifact_content_sha256": frame["content_sha256"],
        "artifact_source_commit": artifact_source_commit,
        "probe_source_commit": probe_source_commit,
        "plant_template_file_sha256": sha256_file(template_path),
        "plant_template_content_sha256": template["content_sha256"],
        "probe_input_file_sha256": sha256_file(probe_path),
        "probe_input_content_sha256": probe_content_sha,
        "live_safety_evidence_file_sha256": live_file_sha,
        "live_safety_evidence_content_sha256": live_content_sha,
        "policy_dt_s": frame["policy_dt_s"],
        "wait_schedule_canonical_sha256": frame["wait_schedule_canonical_sha256"],
        "timing_receipt": frame["timing_receipt"],
        "timing_receipt_canonical_sha256": timing_seal,
        "birth_horizon": frame["birth_horizon"],
        "birth_execution_horizon": {
            "schema_version": 1,
            "kind": "action_ball_frame0_dynamic_birth_execution_horizon_v1",
            "control_decimation": int(runtime["control_decimation"]),
            "required_policy_ticks": frame["birth_horizon"][
                "required_policy_ticks"
            ],
            "required_physics_substeps": frame["birth_horizon"][
                "required_policy_ticks"
            ]
            * int(runtime["control_decimation"]),
            "plant_template_file_sha256": sha256_file(template_path),
            "plant_template_content_sha256": template["content_sha256"],
        },
        "dynamic_birth_gate_evidence": _dynamic_birth_gate_evidence(
            live, policy_dt_s=frame["policy_dt_s"]
        ),
        "live_safety_evidence": live,
    }
    receipt = {**unsigned, "content_sha256": canonical_sha256(unsigned)}
    ignored = _git(
        root,
        ("check-ignore", "-q", "--no-index", "--", _relative(args.output, name="output")),
    )
    if ignored.returncode == 0:
        raise ReceiptError("final receipt output must not be Git-ignored")
    if ignored.returncode not in (0, 1):
        raise ReceiptError("cannot inspect final receipt ignore policy")
    pin = _write_new(root, args.output, canonical_bytes(receipt) + b"\n")
    return {
        "status": "PASS_RECEIPT_MATERIALIZED_COMMIT_REQUIRED",
        "diagnostic_unauthorized": True,
        "launch_authorized": False,
        "receipt": pin,
        "receipt_content_sha256": receipt["content_sha256"],
        "probe_source_commit": probe_source_commit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--probe-source-commit", required=True)
    parser.add_argument("--artifact-source-commit", required=True)
    parser.add_argument("--frame0-artifact-path", required=True)
    parser.add_argument("--expected-frame0-artifact-sha256", required=True)
    parser.add_argument("--plant-template-path", required=True)
    parser.add_argument("--expected-plant-template-sha256", required=True)
    parser.add_argument("--motion-path", required=True)
    parser.add_argument("--expected-motion-sha256", required=True)
    parser.add_argument("--probe-input", required=True)
    parser.add_argument("--live-receipt", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = consume(args)
    except (ReceiptError, OSError, ValueError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
