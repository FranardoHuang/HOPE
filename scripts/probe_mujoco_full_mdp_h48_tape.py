#!/usr/bin/env python3
"""Replay one immutable H48 tape through the real portable Full-A env.

``probe`` writes raw arrays plus compact per-tick evidence.  ``compare``
measures two such records without inventing a tolerance or a readiness verdict.
Natural H48 rows that never reach launch/contact/outcome/recovery are reported
as ``未测``.  This diagnostic never trains, checkpoints, or fabricates a
lifecycle transition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fixtures/mujoco_full_mdp_h48_tape_v1.json"
CONFIG_SHA256 = "4dbd3168fd24ac12f544509b136b2c4c1457c260f1e829de84c155a2f60479e8"
LANE = ROOT / "hope_training/whole_body_tracking/mjlab_lane"
RUNNER = LANE / "mujoco_gpu_ac_full_mdp_wait_rsl3.py"
ENV_SOURCE = LANE / "mujoco_gpu_ac_full_mdp_initial_wait_env.py"
REWARD_CONTRACT_SOURCE = (
    ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/"
    "action_ball_full_mdp_reward_contract.py"
)
ARRAYS_NAME = "arrays.npz"
SUMMARY_NAME = "summary.json"
FLOAT_FIELDS = (
    "initial_qpos", "initial_qvel", "initial_actor", "initial_critic",
    "reward_terms", "actor203", "critic219", "qpos", "qvel",
)
DISCRETE_FIELDS = (
    "done", "termination_bits", "time_out", "backend_table_contact",
    "reset_generation", "epoch_phase", "outcome", "action_slot",
    "action_uid", "mount_normal_sign",
)
STRATA_EVENTS = {"reveal": "full_a_reveal_event", "launch": "full_a_launch_event",
    "contact": "full_a_racket_contact_event",
    "outcome": "full_a_flight_terminal_event", "recovery": "full_a_r07_present_event"}
MASK64 = (1 << 64) - 1

class ProbeError(RuntimeError):
    pass

def _stable_bytes(path: Path, label: str) -> bytes:
    try:
        row = path.lstat()
        if (not stat.S_ISREG(row.st_mode) or row.st_nlink != 1
                or path.resolve(strict=True) != path):
            raise ProbeError(f"{label} is not one canonical regular file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ProbeError(f"cannot read {label}") from exc
    if (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ProbeError(f"{label} changed while reading")
    return payload

def load_config() -> tuple[dict, str]:
    payload = _stable_bytes(CONFIG, "H48 tape config")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONFIG_SHA256:
        raise ProbeError("tracked H48 tape config SHA differs")
    try:
        cfg = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("H48 tape config is not JSON") from exc
    if (cfg.get("schema_version") != 1 or cfg.get("num_envs") != 64
            or cfg.get("num_ticks") != 48 or cfg.get("action_width") != 31
            or cfg.get("environment_seed") != 0
            or cfg.get("action_tape", {}).get("generator")
            != "splitmix64_u24_uniform_v1"):
        raise ProbeError("H48 tape config contract differs")
    tape = cfg["action_tape"]
    if (type(tape.get("seed")) is not int or not 0 <= tape["seed"] <= MASK64
            or not all(type(tape.get(k)) in (int, float) and math.isfinite(tape[k])
                       for k in ("low", "high"))
            or not -1.0 <= tape["low"] < tape["high"] <= 1.0):
        raise ProbeError("H48 action tape parameters differ")
    return cfg, digest

def generate_action_bytes(cfg: dict) -> tuple[bytes, str]:
    spec = cfg["action_tape"]
    state, low, span = spec["seed"], float(spec["low"]), float(spec["high"] - spec["low"])
    payload = bytearray()
    count = cfg["num_ticks"] * cfg["num_envs"] * cfg["action_width"]
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        value ^= value >> 31
        unit = (value >> 40) / float(1 << 24)
        payload.extend(struct.pack("<f", low + span * unit))
    header = json.dumps({"dtype": "<f4", "shape": [cfg["num_ticks"],
        cfg["num_envs"], cfg["action_width"]]}, sort_keys=True,
        separators=(",", ":")).encode("ascii") + b"\n"
    return bytes(payload), hashlib.sha256(header + payload).hexdigest()

def _load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProbeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module

reward_contract = _load_source(
    REWARD_CONTRACT_SOURCE, "_hope_h48_reward_contract"
)
REWARD_TERM_COUNT = reward_contract.REWARD_TERM_COUNT

def _fresh_root(path: Path) -> Path:
    if not path.is_absolute() or path.name in ("", ".", "..") or os.path.lexists(path):
        raise ProbeError("output root must be one absent absolute path")
    if path.parent.resolve(strict=True) != path.parent:
        raise ProbeError("output root parent is not canonical")
    os.mkdir(path, 0o700)
    return path

def _write_json_x(path: Path, record: dict) -> None:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        if os.write(fd, payload) != len(payload):
            raise ProbeError("short JSON write")
        os.fsync(fd)
    finally:
        os.close(fd)

def _source_identity() -> dict:
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=False)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                            "--untracked-files=all"], text=True,
                           capture_output=True, check=False)
    return {"commit": result.stdout.strip() if result.returncode == 0 else "unknown",
            "dirty": dirty.returncode != 0 or bool(dirty.stdout.strip())}

def _probe(output_root: Path, ready_pose: Path) -> dict:
    cfg, config_sha = load_config()
    action_bytes, tape_sha = generate_action_bytes(cfg)
    root = _fresh_root(output_root)
    runner = _load_source(RUNNER, "_hope_h48_tape_runner")
    runtime = runner._bind_full_a_runtime(str(root / "runtime_site"))
    if str(LANE) not in sys.path:
        sys.path.insert(0, str(LANE))
    import numpy as np
    import torch
    wait = runner._wait_module()
    prior = os.environ.get("ACTIONBALL_READY_POSE")
    os.environ["ACTIONBALL_READY_POSE"] = str(ready_pose)
    try:
        ready_payload, ready_source = runner._ready_pose_input()
    finally:
        if prior is None:
            os.environ.pop("ACTIONBALL_READY_POSE", None)
        else:
            os.environ["ACTIONBALL_READY_POSE"] = prior
    torch.manual_seed(cfg["environment_seed"])
    initial = cfg["initial_state"]
    task = wait.TaskCfg(episode_length_s=30.0, action_scale_mode="vendor",
        reset_joint_noise_rad=initial["joint_position_noise_rad"], reset_joint_vel_noise=initial["joint_velocity_noise_rad_s"],
        reset_root_xy_noise_m=initial["root_xy_noise_m"], reset_root_yaw_noise_rad=initial["root_yaw_noise_rad"])
    env = wait.FullMdpInitialWaitVecEnv(wait.SimCfg(nworld=cfg["num_envs"]), task,
        device="cuda:0", seed=cfg["environment_seed"],
        ready_pose_payload=ready_payload, ready_pose_source=ready_source,
        full_a_mode=True)
    tape = torch.frombuffer(bytearray(action_bytes), dtype=torch.float32).reshape(
        cfg["num_ticks"], cfg["num_envs"], cfg["action_width"])
    observations = env.get_observations()
    arrays = {"actions": tape.numpy().copy(),
        "initial_qpos": env.sim.data.qpos.detach().cpu().numpy().copy(),
        "initial_qvel": env.sim.data.qvel.detach().cpu().numpy().copy(),
        "initial_actor": observations["policy"].detach().cpu().numpy().copy(),
        "initial_critic": observations["critic"].detach().cpu().numpy().copy()}
    rows, event_names = {}, None
    for tick in range(cfg["num_ticks"]):
        observations, reward, done, extras = env.step(tape[tick].to(env.device))
        current_events = tuple(sorted(name for name in extras
            if name.startswith("full_a_") and name.endswith("_event")))
        event_names = current_events if event_names is None else event_names
        if current_events != event_names:
            raise ProbeError("Full-A event surface changed during H48")
        mapping = {"done": done, "termination_bits": extras["termination_bits"],
            "time_out": extras["time_outs"],
            "backend_table_contact": extras["backend_resolved_table_contact"],
            "reset_generation": extras["reset_generation"],
            "epoch_phase": extras["full_a_phase_before_reset"],
            "outcome": extras["full_a_outcome_code"],
            "action_slot": extras["full_a_action_slot"],
            "action_uid": extras["full_a_action_uid"],
            "mount_normal_sign": extras["full_a_mount_normal_sign"],
            "reward_terms": extras["reward_terms"], "actor203": observations["policy"],
            "critic219": observations["critic"], "qpos": env.sim.data.qpos,
            "qvel": env.sim.data.qvel}
        mapping.update({"event__" + name: extras[name] for name in event_names})
        if (tuple(mapping["reward_terms"].shape)
                != (cfg["num_envs"], REWARD_TERM_COUNT)
                or tuple(mapping["actor203"].shape) != (cfg["num_envs"], 203)
                or tuple(mapping["critic219"].shape) != (cfg["num_envs"], 219)
                or not torch.equal(reward, mapping["reward_terms"].sum(dim=1))
                or not torch.equal(mapping["actor203"], mapping["critic219"][:, :203])
                or any(not bool(torch.isfinite(mapping[name]).all()) for name in
                       ("reward_terms", "actor203", "critic219", "qpos", "qvel"))):
            raise ProbeError("Full-A H48 numeric/ABI surface differs")
        current = {name: value.detach().cpu().numpy().copy()
                   for name, value in mapping.items()}
        for name, value in current.items():
            rows.setdefault(name, []).append(value)
    arrays.update({name: np.stack(values) for name, values in rows.items()})
    if any(not np.isfinite(arrays[name]).all() for name in FLOAT_FIELDS):
        raise ProbeError("Full-A H48 stored numeric evidence is nonfinite")
    fd = os.open(root / ARRAYS_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)
    totals = {name: int(arrays["event__" + name].sum()) for name in event_names}
    arrays_payload = _stable_bytes(root / ARRAYS_NAME, "probe arrays")
    summary = {"schema_version": 1, "record_type": "mujoco_full_mdp_h48_fixed_tape",
        "diagnostic_unauthorized": True, "training_authorized": False,
        "checkpoint_authority": False, "source": _source_identity(),
        "probe_source_sha256": hashlib.sha256(_stable_bytes(Path(__file__).resolve(), "probe source")).hexdigest(),
        "environment_source_sha256": hashlib.sha256(_stable_bytes(ENV_SOURCE, "Full-A env source")).hexdigest(),
        "runtime": runtime, "config_sha256": config_sha, "action_tape_sha256": tape_sha,
        "ready_pose_sha256": hashlib.sha256(ready_payload).hexdigest(),
        "shape": {"num_envs": cfg["num_envs"], "num_ticks": cfg["num_ticks"],
                  "action_width": cfg["action_width"]},
        "arrays_npz_sha256": hashlib.sha256(arrays_payload).hexdigest(),
        "event_totals": totals,
        "natural_h48_strata": {name: (totals[event] if totals[event] else "未测")
                               for name, event in STRATA_EVENTS.items()},
        "natural_h48_tick_rows": cfg["num_envs"] * cfg["num_ticks"]}
    _write_json_x(root / SUMMARY_NAME, summary)
    return summary

def _load_record(root: Path):
    import numpy as np
    try:
        summary = json.loads(_stable_bytes(root / SUMMARY_NAME, "probe summary"))
        payload = _stable_bytes(root / ARRAYS_NAME, "probe arrays")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProbeError("cannot load probe record") from exc
    cfg, config_sha = load_config()
    action_bytes, tape_sha = generate_action_bytes(cfg)
    if (summary.get("record_type") != "mujoco_full_mdp_h48_fixed_tape"
            or summary.get("config_sha256") != config_sha
            or summary.get("action_tape_sha256") != tape_sha
            or summary.get("arrays_npz_sha256") != hashlib.sha256(payload).hexdigest()):
        raise ProbeError("probe record identity differs")
    base = {"actions", *FLOAT_FIELDS, *DISCRETE_FIELDS}
    events = {name for name in arrays if name.startswith("event__")}
    required_events = {"event__" + name for name in STRATA_EVENTS.values()}
    if set(arrays) != base | events or not required_events <= events:
        raise ProbeError("probe record array surface differs")
    expected = {"actions": (48, 64, 31), "initial_actor": (64, 203),
        "initial_critic": (64, 219),
        "reward_terms": (48, 64, REWARD_TERM_COUNT),
        "actor203": (48, 64, 203), "critic219": (48, 64, 219)}
    expected.update({name: (48, 64) for name in (*DISCRETE_FIELDS, *events)})
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ProbeError("probe record array shape differs: " + name)
    for initial, series in (("initial_qpos", "qpos"), ("initial_qvel", "qvel")):
        if (arrays[initial].ndim != 2 or arrays[initial].shape[0] != 64
                or arrays[initial].shape[1] <= 0
                or arrays[series].shape != (48, *arrays[initial].shape)):
            raise ProbeError("probe record array shape differs: " + series)
    if (arrays["actions"].dtype != np.dtype("<f4")
            or arrays["actions"].tobytes(order="C") != action_bytes
            or any(not np.issubdtype(arrays[name].dtype, np.floating)
                   or not np.isfinite(arrays[name]).all() for name in FLOAT_FIELDS)
            or any(not (np.issubdtype(arrays[name].dtype, np.integer)
                        or np.issubdtype(arrays[name].dtype, np.bool_))
                   for name in (*DISCRETE_FIELDS, *events))):
        raise ProbeError("probe record array dtype/content differs")
    return summary, arrays

def _compare(baseline_root: Path, candidate_root: Path, output: Path) -> dict:
    import numpy as np
    baseline, left = _load_record(baseline_root)
    candidate, right = _load_record(candidate_root)
    if set(left) != set(right) or any(left[name].shape != right[name].shape for name in left):
        raise ProbeError("comparison inputs do not share one exact tape/schema")
    discrete = list(DISCRETE_FIELDS) + sorted(name for name in left if name.startswith("event__"))
    discrete_diffs = {name: int(np.count_nonzero(left[name] != right[name])) for name in discrete}
    numeric = {}
    for name in FLOAT_FIELDS:
        if left[name].shape != right[name].shape:
            raise ProbeError("numeric comparison shape differs: " + name)
        delta = np.abs(left[name].astype(np.float64) - right[name].astype(np.float64))
        numeric[name] = {"exact": bool(np.array_equal(left[name], right[name])),
            "max_abs": float(delta.max()) if delta.size else 0.0,
            "mean_abs": float(delta.mean()) if delta.size else 0.0}
    record = {"schema_version": 1, "record_type": "mujoco_full_mdp_h48_tape_comparison",
        "diagnostic_unauthorized": True, "promotion_authority": False,
        "same_exact_tape": True, "baseline_source": baseline["source"],
        "candidate_source": candidate["source"],
        "discrete_mismatch_cells": discrete_diffs,
        "all_discrete_exact": not any(discrete_diffs.values()),
        "numeric_difference_envelope": numeric,
        "missing_natural_strata": {name: "未测" for name, event in STRATA_EVENTS.items()
            if not bool(right["event__" + event].any())}}
    _write_json_x(output, record)
    return record

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    probe = sub.add_parser("probe", help="run one real-env H48 tape")
    probe.add_argument("--output-root", type=Path, required=True)
    probe.add_argument("--ready-pose", type=Path, required=True)
    compare = sub.add_parser("compare", help="measure two saved tape records")
    compare.add_argument("--baseline-root", type=Path, required=True)
    compare.add_argument("--candidate-root", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = (_probe(args.output_root, args.ready_pose)
                  if args.mode == "probe" else
                  _compare(args.baseline_root, args.candidate_root, args.output))
    except (ProbeError, OSError, RuntimeError, ValueError) as exc:
        print("H48_TAPE_ERROR=" + str(exc), file=sys.stderr, flush=True)
        return 2
    print("ACTION_BALL_MUJOCO_H48_TAPE_JSON=" + json.dumps(record,
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
