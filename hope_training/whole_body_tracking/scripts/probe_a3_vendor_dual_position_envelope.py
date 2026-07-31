#!/usr/bin/env python3
"""Run the source-bound A3 waist dual-position-envelope stress probe.

This is a simulator-only mechanical test, never a training or deployment
authorization.  Exactly eight environments share four state tapes:

``waist_roll/waist_pitch x lower/upper x H_ctrl ON/OFF``.

For each side, ``R`` is the distance from the live two-percent control edge to
the unchanged mechanical edge.  The initial state is 0.1 R inside H_ctrl and
its exact 5 ms kinematic projection is 0.6 R outside H_ctrl (therefore 0.4 R
inside H_mech).  The ON/OFF pair has byte-identical q0, qdot and q_des; only
the live PhysX limit row differs.  The existing action diagnostic retains its
20 ms ``capture_proxy`` meaning.  This receipt records the separate 5 ms
kinematic crossing and never calls either measurement a constraint impulse.

The probe writes one JSON receipt with O_EXCL outside the source and Isaac Lab
trees.  It requires an exact clean source HEAD both before Kit startup and
immediately before publication.  Live limits are restored to all-environment
H_ctrl in ``finally`` and exact readback is mandatory; a restoration failure
can only publish FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
KIND = "a3_vendor_dual_position_envelope_stress_v1"
CONFIRM = "SIM_ONLY_A3_DUAL_POSITION_ENVELOPE_8ENV_ONE_TICK"
EXACT_NUM_ENVS = 8
EXACT_PHYSICS_DT_S = 0.005
CONTROL_INSET_FRACTION = 0.02
Q0_INNER_CAGE_FRACTION = 0.1
KINEMATIC_OUTER_CAGE_FRACTION = 0.6
MECHANICAL_REMAINING_CAGE_FRACTION = 0.4
STRESSED_JOINTS = ("waist_roll_joint", "waist_pitch_joint")
SIDES = ("lower", "upper")
CONDITIONS = ("on", "off")


class DualEnvelopeProbeError(RuntimeError):
    """The strict dual-envelope probe contract was not satisfied."""


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _verify_clean_exact_checkout(
    root: Path,
    expected_commit: str,
    *,
    script_path: Path | None = None,
) -> str:
    root = root.resolve(strict=True)
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != root:
        raise DualEnvelopeProbeError(f"source root is not exact Git top-level: {top}")
    head = _git(root, "rev-parse", "HEAD")
    if head != expected_commit or len(head) != 40:
        raise DualEnvelopeProbeError(
            f"source HEAD mismatch: expected={expected_commit!r} actual={head!r}"
        )
    if _git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise DualEnvelopeProbeError("source checkout must be exactly clean")
    if script_path is not None:
        resolved = script_path.resolve(strict=True)
        if not _inside(resolved, root):
            raise DualEnvelopeProbeError("probe script is outside the reviewed source root")
        relative = resolved.relative_to(root).as_posix()
        tracked = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        if tracked != resolved.read_bytes():
            raise DualEnvelopeProbeError("probe script bytes differ from exact source HEAD")
    return head


def _validate_output_path(output: Path, forbidden_roots: Sequence[Path]) -> Path:
    output = output.expanduser()
    if not output.is_absolute():
        raise DualEnvelopeProbeError("--output must be an absolute JSON path")
    parent = output.parent.resolve(strict=True)
    resolved = parent / output.name
    if resolved.suffix != ".json":
        raise DualEnvelopeProbeError("--output must end in .json")
    for raw_root in forbidden_roots:
        root = raw_root.resolve(strict=True)
        if _inside(resolved, root):
            raise DualEnvelopeProbeError(
                f"output must remain outside protected root: {root}"
            )
    if resolved.exists() or resolved.is_symlink():
        raise DualEnvelopeProbeError(f"no-clobber output already exists: {resolved}")
    return resolved


def _installed_isaaclab_root() -> Path:
    """Resolve the editable/install tree that the live probe will import."""

    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        raise DualEnvelopeProbeError("the selected Python cannot resolve isaaclab")
    package_file = Path(spec.origin).resolve(strict=True)
    try:
        return Path(_git(package_file.parent, "rev-parse", "--show-toplevel")).resolve(
            strict=True
        )
    except (OSError, subprocess.CalledProcessError):
        return package_file.parent


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DualEnvelopeProbeError(f"{label} must be one finite number")
    return float(value)


def _float32_round(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def build_stress_tape(
    joint_names: Sequence[str],
    mechanical_limits: Sequence[Sequence[float]],
    control_limits: Sequence[Sequence[float]],
    *,
    physics_dt_s: float = EXACT_PHYSICS_DT_S,
) -> list[dict[str, Any]]:
    """Build the exact 8-row same-state ON/OFF tape without simulator imports."""

    if type(physics_dt_s) is not float or physics_dt_s != EXACT_PHYSICS_DT_S:
        raise DualEnvelopeProbeError("stress tape requires exact physics_dt_s=0.005")
    names = tuple(str(name) for name in joint_names)
    if len(names) != len(set(names)) or any(names.count(name) != 1 for name in STRESSED_JOINTS):
        raise DualEnvelopeProbeError("joint order must contain each stressed waist exactly once")
    if len(mechanical_limits) != len(names) or len(control_limits) != len(names):
        raise DualEnvelopeProbeError("limit row count must equal joint-name count")

    rows: list[dict[str, Any]] = []
    for joint_name in STRESSED_JOINTS:
        joint_index = names.index(joint_name)
        mechanical = tuple(
            _finite_number(value, f"H_mech[{joint_name}]")
            for value in mechanical_limits[joint_index]
        )
        control = tuple(
            _finite_number(value, f"H_ctrl[{joint_name}]")
            for value in control_limits[joint_index]
        )
        if len(mechanical) != 2 or len(control) != 2:
            raise DualEnvelopeProbeError("every limit row must be [lower, upper]")
        hard_lower, hard_upper = mechanical
        ctrl_lower, ctrl_upper = control
        span = hard_upper - hard_lower
        if not span > 0.0:
            raise DualEnvelopeProbeError(f"{joint_name} has non-positive H_mech span")
        lower_reserve = ctrl_lower - hard_lower
        upper_reserve = hard_upper - ctrl_upper
        expected_reserve = CONTROL_INSET_FRACTION * span
        if not math.isclose(lower_reserve, expected_reserve, rel_tol=0.0, abs_tol=1.0e-7):
            raise DualEnvelopeProbeError(f"{joint_name} lower H_ctrl reserve is not exact 2%")
        if not math.isclose(upper_reserve, expected_reserve, rel_tol=0.0, abs_tol=1.0e-7):
            raise DualEnvelopeProbeError(f"{joint_name} upper H_ctrl reserve is not exact 2%")

        for side in SIDES:
            direction = -1.0 if side == "lower" else 1.0
            mechanical_edge = hard_lower if side == "lower" else hard_upper
            control_edge = ctrl_lower if side == "lower" else ctrl_upper
            reserve = lower_reserve if side == "lower" else upper_reserve
            q0 = control_edge - direction * Q0_INNER_CAGE_FRACTION * reserve
            kinematic_q_5ms = (
                control_edge
                + direction * KINEMATIC_OUTER_CAGE_FRACTION * reserve
            )
            qdot = (kinematic_q_5ms - q0) / physics_dt_s
            if not math.isclose(
                abs(qdot),
                (Q0_INNER_CAGE_FRACTION + KINEMATIC_OUTER_CAGE_FRACTION)
                * reserve
                / physics_dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise AssertionError("internal qdot reserve arithmetic drifted")
            mechanical_gap = direction * (mechanical_edge - kinematic_q_5ms)
            if not math.isclose(
                mechanical_gap,
                MECHANICAL_REMAINING_CAGE_FRACTION * reserve,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise AssertionError("internal H_mech remaining-reserve arithmetic drifted")
            for condition in CONDITIONS:
                rows.append(
                    {
                        "env_id": len(rows),
                        "joint": joint_name,
                        "joint_index": joint_index,
                        "side": side,
                        "condition": condition,
                        "direction": int(direction),
                        "h_mech_edge_rad": mechanical_edge,
                        "h_ctrl_edge_rad": control_edge,
                        "cage_reserve_rad": reserve,
                        "q0_rad": q0,
                        "qdot0_rad_s": qdot,
                        "qdes_rad": q0,
                        "kinematic_q_5ms_rad": kinematic_q_5ms,
                        "kinematic_crosses_h_ctrl": True,
                        "kinematic_mechanical_gap_rad": mechanical_gap,
                    }
                )
    validate_stress_tape(rows)
    return rows


def validate_stress_tape(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXACT_NUM_ENVS:
        raise DualEnvelopeProbeError("stress tape must contain exactly 8 rows")
    expected = [
        (joint, side, condition)
        for joint in STRESSED_JOINTS
        for side in SIDES
        for condition in CONDITIONS
    ]
    observed = []
    for env_id, row in enumerate(rows):
        if row.get("env_id") != env_id:
            raise DualEnvelopeProbeError("stress tape env ids are not exact identity order")
        observed.append((row.get("joint"), row.get("side"), row.get("condition")))
        for key in (
            "h_mech_edge_rad",
            "h_ctrl_edge_rad",
            "cage_reserve_rad",
            "q0_rad",
            "qdot0_rad_s",
            "qdes_rad",
            "kinematic_q_5ms_rad",
            "kinematic_mechanical_gap_rad",
        ):
            _finite_number(row.get(key), f"stress_tape[{env_id}].{key}")
        if row.get("kinematic_crosses_h_ctrl") is not True:
            raise DualEnvelopeProbeError("every tape row must cross H_ctrl kinematically in 5 ms")
        if row["qdes_rad"] != row["q0_rad"]:
            raise DualEnvelopeProbeError("q_des must equal q0 exactly")
    if observed != expected:
        raise DualEnvelopeProbeError("stress tape joint/side/condition order drifted")
    for pair_start in range(0, EXACT_NUM_ENVS, 2):
        on = dict(rows[pair_start])
        off = dict(rows[pair_start + 1])
        for key in ("env_id", "condition"):
            on.pop(key)
            off.pop(key)
        if on != off:
            raise DualEnvelopeProbeError("ON/OFF pair does not share the exact same state tape")


def _side_gaps(row: Mapping[str, Any], q_after: float) -> tuple[float, float]:
    direction = int(row["direction"])
    control_gap = -direction * (q_after - float(row["h_ctrl_edge_rad"]))
    mechanical_gap = -direction * (q_after - float(row["h_mech_edge_rad"]))
    return control_gap, mechanical_gap


def validate_runtime_result(
    tape: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
    *,
    physics_dt_s: float,
    live_limits_restored_exact: bool,
) -> dict[str, Any]:
    """Validate strict ON/OFF outcomes and return receipt aggregates.

    No tolerance weakens a safety edge: ON must finish strictly inside H_ctrl;
    OFF must finish on/outside H_ctrl and strictly inside H_mech.  Exact q_des
    equality is checked independently from solver state.
    """

    validate_stress_tape(tape)
    if physics_dt_s != EXACT_PHYSICS_DT_S:
        raise DualEnvelopeProbeError("live physics dt is not exact 0.005 s")
    if live_limits_restored_exact is not True:
        raise DualEnvelopeProbeError("all-environment H_ctrl restoration/readback failed")
    if len(observations) != EXACT_NUM_ENVS:
        raise DualEnvelopeProbeError("runtime must return exactly 8 observation rows")

    normalized: list[dict[str, Any]] = []
    pair_counts: dict[tuple[str, str], dict[str, int]] = {
        (joint, side): {
            "strict_5ms_kinematic_attempt_count": 0,
            "on_capture_count": 0,
            "off_post_ctrl_penetration_count": 0,
            "mechanical_penetration_count": 0,
        }
        for joint in STRESSED_JOINTS
        for side in SIDES
    }
    for env_id, (tape_row, raw) in enumerate(zip(tape, observations, strict=True)):
        if raw.get("env_id") != env_id:
            raise DualEnvelopeProbeError("runtime observation env ids drifted")
        if raw.get("joint") != tape_row["joint"] or raw.get("side") != tape_row["side"]:
            raise DualEnvelopeProbeError("runtime observation identity differs from tape")
        if raw.get("condition") != tape_row["condition"]:
            raise DualEnvelopeProbeError("runtime condition differs from tape")
        q_after = _finite_number(raw.get("q_after_rad"), f"observation[{env_id}].q_after")
        qdot_after = _finite_number(
            raw.get("qdot_after_rad_s"), f"observation[{env_id}].qdot_after"
        )
        q0_live = _finite_number(
            raw.get("q0_live_rad"), f"observation[{env_id}].q0_live"
        )
        qdes = _finite_number(raw.get("qdes_rad"), f"observation[{env_id}].qdes")
        if q0_live != _float32_round(float(tape_row["q0_rad"])):
            raise DualEnvelopeProbeError("live q0 differs from the float32 exact stress tape")
        if qdes != q0_live:
            raise DualEnvelopeProbeError("live q_des differs from exact live q0")
        control_gap, mechanical_gap = _side_gaps(tape_row, q_after)
        condition = str(tape_row["condition"])
        if condition == "on" and not control_gap > 0.0:
            raise DualEnvelopeProbeError("H_ctrl ON did not capture strictly inside H_ctrl")
        if condition == "off" and not control_gap <= 0.0:
            raise DualEnvelopeProbeError("H_ctrl OFF did not enter [H_ctrl,H_mech)")
        if not mechanical_gap > 0.0:
            raise DualEnvelopeProbeError("stress row touched/crossed H_mech")
        key = (str(tape_row["joint"]), str(tape_row["side"]))
        pair_counts[key]["strict_5ms_kinematic_attempt_count"] += 1
        if condition == "on":
            pair_counts[key]["on_capture_count"] += 1
        else:
            pair_counts[key]["off_post_ctrl_penetration_count"] += 1
        normalized.append(
            {
                **dict(raw),
                "q_after_rad": q_after,
                "qdot_after_rad_s": qdot_after,
                "q0_live_rad": q0_live,
                "qdes_rad": qdes,
                "post_signed_ctrl_gap_rad": control_gap,
                "post_signed_mechanical_gap_rad": mechanical_gap,
                "strict_5ms_kinematic_crossing": True,
            }
        )

    diag = diagnostic.get("physx_control_position_limits")
    if not isinstance(diag, Mapping) or diag.get("enabled") is not True:
        raise DualEnvelopeProbeError("verified 20 ms H_ctrl diagnostic is absent")
    if diag.get("semantics") != "kinematic H_ctrl proxy; not a PhysX constraint impulse getter":
        raise DualEnvelopeProbeError("diagnostic proxy semantics drifted")
    if float(diag.get("ballistic_horizon_s", -1.0)) != 0.02:
        raise DualEnvelopeProbeError("existing capture_proxy horizon must remain 20 ms")
    by_joint = diag.get("by_joint")
    if not isinstance(by_joint, list):
        raise DualEnvelopeProbeError("diagnostic by_joint rows are absent")
    diag_rows = {row.get("joint"): row.get("sides") for row in by_joint if isinstance(row, Mapping)}
    aggregate_rows = []
    for joint in STRESSED_JOINTS:
        sides = diag_rows.get(joint)
        if not isinstance(sides, Mapping):
            raise DualEnvelopeProbeError(f"diagnostic is missing {joint}")
        for side in SIDES:
            values = sides.get(side)
            if not isinstance(values, Mapping):
                raise DualEnvelopeProbeError(f"diagnostic is missing {joint}/{side}")
            attempt = values.get("ballistic_attempt_proxy")
            capture = values.get("capture_proxy")
            penetration = values.get("ctrl_penetration_readback")
            if attempt != 2 or capture != 1 or penetration != 1:
                raise DualEnvelopeProbeError(
                    f"{joint}/{side} expected attempt=2 capture=1 penetration=1, "
                    f"got {attempt}/{capture}/{penetration}"
                )
            local = pair_counts[(joint, side)]
            if local != {
                "strict_5ms_kinematic_attempt_count": 2,
                "on_capture_count": 1,
                "off_post_ctrl_penetration_count": 1,
                "mechanical_penetration_count": 0,
            }:
                raise AssertionError("internal per-pair aggregate drifted")
            aggregate_rows.append(
                {
                    "joint": joint,
                    "side": side,
                    **local,
                    "existing_20ms_ballistic_attempt_proxy_count": attempt,
                    "existing_20ms_capture_proxy_count": capture,
                    "post_ctrl_penetration_readback_count": penetration,
                }
            )

    return {
        "observations": normalized,
        "aggregate_by_joint_side": aggregate_rows,
        "mechanical_penetration_count": 0,
        "all_rows_finite": True,
        "all_qdes_equal_q0_exact": True,
        "all_live_limits_restored_to_hctrl_exact": True,
        "existing_diagnostic": dict(diagnostic),
    }


def build_receipt(
    *,
    source_commit: str,
    source_script_sha256: str,
    task: str,
    motion_files: Sequence[Mapping[str, str]],
    tape: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any] | None,
    live_limit_identity: Mapping[str, Any] | None,
    restore: Mapping[str, Any],
    error: str | None,
) -> dict[str, Any]:
    restore_exact = restore.get("attempted") is True and restore.get("exact_readback") is True
    passed = error is None and runtime is not None and restore_exact
    content: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "PASS" if passed else "FAIL",
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
        "source_commit": source_commit,
        "producer": {
            "path": "hope_training/whole_body_tracking/scripts/"
            "probe_a3_vendor_dual_position_envelope.py",
            "sha256": source_script_sha256,
        },
        "task": task,
        "motion_files": list(motion_files),
        "contract": {
            "num_envs": EXACT_NUM_ENVS,
            "physics_ticks": 1,
            "physics_dt_s": EXACT_PHYSICS_DT_S,
            "stressed_joints": list(STRESSED_JOINTS),
            "sides": list(SIDES),
            "conditions": list(CONDITIONS),
            "q0_inner_cage_fraction": Q0_INNER_CAGE_FRACTION,
            "kinematic_outer_cage_fraction": KINEMATIC_OUTER_CAGE_FRACTION,
            "kinematic_remaining_mechanical_cage_fraction": (
                MECHANICAL_REMAINING_CAGE_FRACTION
            ),
            "qdot_formula": "direction*(0.1+0.6)*R/0.005",
            "qdes_contract": "qdes=q0 exact",
            "existing_capture_proxy_horizon_s": 0.02,
            "strict_kinematic_crossing_horizon_s": 0.005,
            "measurement_semantics": (
                "kinematic positions/velocities and H_ctrl proxies; no constraint impulse getter"
            ),
            "disabled_randomization": [
                "push",
                "pd_gain",
                "mass",
                "base_com",
                "physics_material",
                "joint_default_pos",
            ],
        },
        "stress_tape": [dict(row) for row in tape],
        "live_limit_identity": None if live_limit_identity is None else dict(live_limit_identity),
        "runtime": None if runtime is None else dict(runtime),
        "restore": dict(restore),
        "error": error,
    }
    return {**content, "content_sha256": _sha256_bytes(_canonical_json_bytes(content))}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="HOPE-PingPong-ActionBall-AgibotA3-v0")
    parser.add_argument("--motion-file", action="append", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args(argv)


def _disable_randomization(env_cfg: Any) -> list[str]:
    events = getattr(env_cfg, "events", None)
    if events is None:
        raise DualEnvelopeProbeError("task exposes no EventCfg")
    names = (
        "push_robot",
        "force_push",
        "force_push_sweep",
        "combined_push",
        "combined_push_sweep",
        "randomize_pd_gains",
        "randomize_link_mass",
        "base_com",
        "physics_material",
        "add_joint_default_pos",
    )
    disabled = []
    for name in names:
        if hasattr(events, name):
            setattr(events, name, None)
            disabled.append(name)
    required = {
        "push_robot",
        "randomize_pd_gains",
        "randomize_link_mass",
        "base_com",
        "physics_material",
        "add_joint_default_pos",
    }
    if not required.issubset(disabled):
        raise DualEnvelopeProbeError(
            f"task is missing required randomization switches: {sorted(required - set(disabled))}"
        )
    return disabled


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _run_live(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Launch Kit, run one physics tick, and always restore all-environment H_ctrl."""

    # Isaac imports belong after CLI/source validation so host tests can import this module.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=bool(args.headless),
        device=str(args.device),
        enable_cameras=False,
    )
    simulation_app = launcher.app
    env = None
    action_term = None
    robot = None
    all_hctrl_cpu = None
    restore = {"attempted": False, "exact_readback": False, "error": None}
    tape: list[dict[str, Any]] = []
    live_identity: dict[str, Any] = {}
    try:
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        import whole_body_tracking.tasks  # noqa: F401

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=EXACT_NUM_ENVS)
        if float(env_cfg.sim.dt) != EXACT_PHYSICS_DT_S or int(env_cfg.decimation) != 4:
            raise DualEnvelopeProbeError("task must retain dt=0.005 and decimation=4")
        disabled_events = _disable_randomization(env_cfg)
        motion_values = [str(path.resolve(strict=True)) for path in args.motion_file]
        env_cfg.commands.motion.motion_file = (
            motion_values[0] if len(motion_values) == 1 else motion_values
        )
        env_cfg.commands.racket_target.action_ball_diagnostic_unauthorized = True
        joint_action_cfg = env_cfg.actions.joint_pos
        joint_action_cfg.physx_control_position_limit_inset_fraction = CONTROL_INSET_FRACTION
        joint_action_cfg.control_step_action_delay_min = 0
        joint_action_cfg.control_step_action_delay_max = 0

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        base = env.unwrapped
        reset = env.reset()
        if not isinstance(reset, tuple) or len(reset) != 2:
            raise DualEnvelopeProbeError("Gym reset did not return (observation, info)")
        if int(base.num_envs) != EXACT_NUM_ENVS:
            raise DualEnvelopeProbeError("live task did not construct exactly 8 envs")
        if float(base.physics_dt) != EXACT_PHYSICS_DT_S:
            raise DualEnvelopeProbeError("live physics_dt is not exact 0.005")
        action_term = base.action_manager.get_term("joint_pos")
        robot = base.scene["robot"]
        action_term.verify_physx_control_position_limit_readback()
        # Reset may legitimately exercise the normal update diagnostic before
        # this one-tick tape starts.  Clear both its published counters and its
        # temporal proxy history so the receipt contains only these 8 envs.
        action_term.consume_actual_joint_forbidden_diagnostic()
        for name in (
            "_physx_control_diagnostic_previous_attempt",
            "_physx_control_diagnostic_current_penetration_dwell",
            "_physx_control_diagnostic_previous_qdot",
            "_physx_control_diagnostic_previous_qdot_valid",
        ):
            tensor = getattr(action_term, name, None)
            if tensor is None:
                raise DualEnvelopeProbeError(
                    f"H_ctrl diagnostic history buffer disappeared: {name}"
                )
            tensor.zero_()
        mechanical = robot.data.joint_pos_limits.detach().cpu()
        all_hctrl_cpu = action_term._physx_control_joint_pos_limits_snapshot.detach().clone()
        names = tuple(str(name) for name in robot.joint_names)
        tape = build_stress_tape(
            names,
            mechanical[0].tolist(),
            all_hctrl_cpu[0].tolist(),
        )

        mixed = all_hctrl_cpu.clone()
        for row in tape:
            if row["condition"] == "off":
                mixed[row["env_id"]] = mechanical[row["env_id"]]
        indices = robot._ALL_INDICES.detach().cpu()
        root_view = robot.root_physx_view
        root_view.set_dof_limits(mixed, indices=indices)
        mixed_readback = root_view.get_dof_limits()
        if not torch.equal(mixed_readback, mixed):
            raise DualEnvelopeProbeError("mixed H_ctrl ON/OFF live-limit readback is not exact")

        q0 = robot.data.default_joint_pos.detach().clone()
        qdot0 = torch.zeros_like(q0)
        if q0.dtype != torch.float32:
            raise DualEnvelopeProbeError("stress probe requires the shipped float32 articulation")
        for row in tape:
            q0[row["env_id"], row["joint_index"]] = row["q0_rad"]
            qdot0[row["env_id"], row["joint_index"]] = row["qdot0_rad_s"]
        if not torch.all(torch.isfinite(q0)) or not torch.all(torch.isfinite(qdot0)):
            raise DualEnvelopeProbeError("constructed q0/qdot tape is non-finite")
        robot.write_joint_state_to_sim(q0, qdot0)
        robot.set_joint_position_target(q0)
        base.scene.write_data_to_sim()

        action_term._record_physx_control_position_limit_diagnostic(
            joint_pos=q0,
            joint_vel=qdot0,
        )
        base.sim.step(render=False)
        base.scene.update(EXACT_PHYSICS_DT_S)
        q_after = robot.data.joint_pos.detach().clone()
        qdot_after = robot.data.joint_vel.detach().clone()
        qdes_source = getattr(robot.data, "joint_pos_target", None)
        if not torch.is_tensor(qdes_source):
            qdes_source = getattr(robot, "_joint_pos_target", None)
        if not torch.is_tensor(qdes_source) or tuple(qdes_source.shape) != tuple(q0.shape):
            raise DualEnvelopeProbeError(
                "live articulation exposes no exact joint-position target buffer"
            )
        qdes_live = qdes_source.detach().clone()
        action_term._record_physx_control_position_limit_diagnostic(
            joint_pos=q_after,
            joint_vel=qdot_after,
        )
        diagnostic = action_term.consume_actual_joint_forbidden_diagnostic()

        observations = []
        for row in tape:
            env_id = row["env_id"]
            joint_index = row["joint_index"]
            observations.append(
                {
                    "env_id": env_id,
                    "joint": row["joint"],
                    "side": row["side"],
                    "condition": row["condition"],
                    "q_after_rad": float(q_after[env_id, joint_index].detach().cpu()),
                    "qdot_after_rad_s": float(qdot_after[env_id, joint_index].detach().cpu()),
                    "q0_live_rad": float(q0[env_id, joint_index].detach().cpu()),
                    "qdes_rad": float(qdes_live[env_id, joint_index].detach().cpu()),
                }
            )
        live_identity = {
            "run_specific_live_limit_sha256": (
                action_term._physx_control_position_limit_readback_sha256
            ),
            "setter_no_mutation_sha256": (
                action_term._physx_control_setter_no_mutation_sha256
            ),
            "mixed_live_limit_sha256": _sha256_bytes(
                mixed.contiguous().numpy().tobytes()
            ),
            "mixed_readback_exact": True,
            "disabled_event_terms": disabled_events,
        }
        # Validate only after finally restored the live plant.  Preserve raw rows meanwhile.
        pending = {
            "observations": observations,
            "diagnostic": diagnostic,
            "physics_dt_s": float(base.physics_dt),
        }
    finally:
        if robot is not None and all_hctrl_cpu is not None:
            restore["attempted"] = True
            try:
                indices = robot._ALL_INDICES.detach().cpu()
                robot.root_physx_view.set_dof_limits(all_hctrl_cpu, indices=indices)
                readback = robot.root_physx_view.get_dof_limits()
                import torch

                restore["exact_readback"] = bool(torch.equal(readback, all_hctrl_cpu))
                if not restore["exact_readback"]:
                    restore["error"] = "restored live-limit readback differs from H_ctrl"
            except Exception as exc:  # noqa: BLE001 - preserve honest FAIL evidence
                restore["error"] = f"{type(exc).__name__}: {exc}"
        if env is not None:
            env.close()
        simulation_app.close()

    if restore["exact_readback"] is not True:
        raise DualEnvelopeProbeError(str(restore["error"] or "H_ctrl restoration failed"))
    runtime = validate_runtime_result(
        tape,
        pending["observations"],
        pending["diagnostic"],
        physics_dt_s=pending["physics_dt_s"],
        live_limits_restored_exact=True,
    )
    return runtime, tape, {"identity": live_identity, "restore": restore}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.execute is not True or args.confirm != CONFIRM:
        raise SystemExit(f"refusing simulator probe without --execute --confirm {CONFIRM}")
    source_root = args.source_root.resolve(strict=True)
    script_path = Path(__file__).resolve(strict=True)
    source_commit = _verify_clean_exact_checkout(
        source_root,
        args.expected_source_commit,
        script_path=script_path,
    )
    output = _validate_output_path(
        args.output,
        (source_root, _installed_isaaclab_root()),
    )
    motion_files = []
    for raw in args.motion_file:
        path = raw.resolve(strict=True)
        if not path.is_file():
            raise DualEnvelopeProbeError(f"motion input is not a file: {path}")
        motion_files.append({"path": str(path), "sha256": _sha256_file(path)})

    error = None
    runtime = None
    tape: list[dict[str, Any]] = []
    identity = None
    restore: dict[str, Any] = {
        "attempted": False,
        "exact_readback": False,
        "error": "runtime did not reach restoration",
    }
    try:
        runtime, tape, sidecar = _run_live(args)
        identity = sidecar["identity"]
        restore = sidecar["restore"]
    except Exception as exc:  # noqa: BLE001 - FAIL receipt is intentional
        error = f"{type(exc).__name__}: {exc}"

    # Re-attest the source immediately before the only publication side effect.
    _verify_clean_exact_checkout(
        source_root,
        source_commit,
        script_path=script_path,
    )
    for motion in motion_files:
        if _sha256_file(Path(motion["path"])) != motion["sha256"]:
            raise DualEnvelopeProbeError("motion input changed during the probe")
    payload = build_receipt(
        source_commit=source_commit,
        source_script_sha256=_sha256_file(script_path),
        task=args.task,
        motion_files=motion_files,
        tape=tape,
        runtime=runtime,
        live_limit_identity=identity,
        restore=restore,
        error=error,
    )
    _write_json_exclusive(output, payload)
    print(
        "A3_DUAL_POSITION_ENVELOPE_STRESS="
        + json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "content_sha256": payload["content_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
