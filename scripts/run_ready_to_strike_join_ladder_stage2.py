#!/usr/bin/env python3
"""Run the four registered d=12 ready-to-strike ladder cells, fail closed.

This is a host/CPU-only screening runner.  It never opens SSH, sends a signal,
retries a failed cell, starts a trainer, or addresses a robot.  Dry-run is the
default.  Execute mode needs an explicit token and an absolute, nonexistent
result root; every input snapshot and result is published with O_EXCL.

Stage 2 is intentionally downstream of the historical Stage-1 attestation.
The activation document must bind the exact receipt bytes and keep the honest
``screening-only`` formal claims.  A missing or edited receipt therefore blocks
before a namespace or subprocess is created.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np


SCHEMA_VERSION = 1
CONFIRM_TOKEN = "RUN_READY_TO_STRIKE_STAGE2_ONCE"
CHILD_TIMEOUT_S = 3600
EXPECTED_ACTIVATION_ID = "ready_to_strike_join_ladder_stage2_20260717"
EXPECTED_EXPERIMENT_ID = "ready_to_strike_join_ladder_20260717"
EXPECTED_QUEUE_SHA256 = "cfa112f799dab9af33914fdfb5bfff90d21b4692e38b16a4627393936a527b8b"
EXPECTED_PREREG_COMMIT = "8d74025e88fee832fae0ac2f672ec0eb9b2d3d5a"
EXPECTED_EVIDENCE_STATUS = "historical_stage1_attested_screening_only"

EXPECTED_STAGE1_CELLS = {
    "fh_rf_d17": ("forehand", "forehand", 17),
    "fh_rb_d06": ("forehand", "backhand", 6),
    "fh_rb_d17": ("forehand", "backhand", 17),
    "bh_rf_d17": ("backhand", "forehand", 17),
    "bh_rb_d06": ("backhand", "backhand", 6),
    "bh_rb_d17": ("backhand", "backhand", 17),
}
EXPECTED_STAGE2_CELLS = {
    "fh_rf_d12": ("forehand", "forehand", 12),
    "fh_rb_d12": ("forehand", "backhand", 12),
    "bh_rf_d12": ("backhand", "forehand", 12),
    "bh_rb_d12": ("backhand", "backhand", 12),
}
EXPECTED_OBSERVATIONS = {
    "fh_rf_d06": ("forehand", "forehand", 6, "b0350f4d8db600344651135673afc99c95d43a2c27432d131a8ed8a08253ac4d", "c16c4dc74f9ac254ab22a4987ad23ad3bc45fc462040a1a37f8eb98c213474ed", 0.64),
    "fh_rf_d17": ("forehand", "forehand", 17, "ff11d738dfe8fe629fbb796f322c51a47017918643d8da55f697cbf137ad9122", "5d939ec781858919611f0191447d0699499c1e3ccffc38c9d3b33f07ab371e60", 1.28),
    "fh_rb_d06": ("forehand", "backhand", 6, "469ef828855151854f425dda66928454395d6ab84098d2de469dc05dd73e1113", "71964c47d6c340d4f2268394b4009fa295cbb46c4095f52996f4ca10134e4b37", 0.70),
    "fh_rb_d17": ("forehand", "backhand", 17, "733e63b5c7cad7406d551f210f9ea1ddd5b8b4b5701ee2a93a972148fe0e91a1", "fe76360795483cdd4397d903eeeef085f82021dce5f7e8b1ced446de94f5512a", 1.54),
    "bh_rf_d06": ("backhand", "forehand", 6, "0f017dc58918aa1e80846d4dfec81668efbb3fecabf546727750a8c2717dfc89", "0d0e45e336d3aa02e3bb781fd252a5b2b5453d4e49caa6049b3b1d0c33f0c720", 0.94),
    "bh_rf_d17": ("backhand", "forehand", 17, "21fc80a57b214f78f9e4a8e9e6afa062e290b28bf876f7da34d2d8009fae9046", "0b9243d023173a2710fa5bd7b4ba27d97d9216e61b0bf16803caed41ab18d333", 1.94),
    "bh_rb_d06": ("backhand", "backhand", 6, "18b27e4b530f1145f0e8fd8cea50902c8559d2d037e0cbc023bd73d0b4187eec", "5482539bb4302a18b0868239a39ddf1e7ebcbff29e96d5223f8ba68050df8810", 0.78),
    "bh_rb_d17": ("backhand", "backhand", 17, "a842a87d0557dfc6e814e09ff466336ed14e294885304935316bc94d3410725e", "e8e7adac56ba5225061437a03cc26e1cb671b8b6bb099c5f0d034dc5c6dd1d6f", 1.42),
}

RUNTIME_RELATIVE_PATHS = {
    "generator_sha256": Path("scripts/build_ready_to_strike_motion.py"),
    "topp_sha256": Path("hope_training/whole_body_tracking/scripts/topp_mintime.py"),
    "mjcf_sha256": Path("agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"),
    "urdf_sha256": Path("agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"),
    "body_order_sha256": Path("configs/a3_runtime_body_order.txt"),
}
TOPP_CLOSURE_PATHS = (
    Path("hope_training/whole_body_tracking/scripts/topp_mintime.py"),
    Path("hope_training/whole_body_tracking/scripts/synthesize_timing.py"),
    Path("hope_training/whole_body_tracking/scripts/synthesize_timing_v2.py"),
    Path("hope_training/whole_body_tracking/scripts/audit_motion_npz.py"),
    Path("hope_training/whole_body_tracking/scripts/csv_to_npz_mujoco.py"),
    Path("hope_training/whole_body_tracking/scripts/racket_geometry_contract.py"),
    Path("hope_training/whole_body_tracking/scripts/motion_kinematics_contract.py"),
    Path("hope_training/whole_body_tracking/scripts/hope_frame_utils.py"),
    Path("hope_training/whole_body_tracking/scripts/a3_joint_order_contract.py"),
    Path("scripts/feasibility_oracle.py"),
)
TOPP_CERTIFICATE_KEYS = {
    "tool", "algorithm_scope", "search_objective", "generated_utc", "verdict",
    "direction", "chosen_scale", "feasible_reason", "budget", "acceptance",
    "durations", "oracle_before", "oracle_after", "kin", "source", "answer",
    "baseline_law", "stretch", "output", "timing_bound", "fidelity",
    "outer_trace", "inner_trace_best", "files", "budget_provenance",
    "runtime_provenance",
}
TOPP_TOOL = "topp_mintime.py v3 (unified-budget min-time bidirectional retiming)"
TOPP_ALGORITHM_SCOPE = (
    "heuristic upper bound within the sampled gamma ladder plus greedy local repair; "
    "not strict TOPP and not a global minimum proof"
)


class Stage2Error(ValueError):
    """Activation, evidence, namespace, subprocess, or result contract failed."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    payload: bytes
    mode: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage2Error(message)


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False,
                       separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _reject_constant(value: str) -> None:
    raise Stage2Error(f"JSON contains non-finite constant {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), parse_constant=_reject_constant,
                          object_pairs_hook=_pairs_no_duplicates)
    except Stage2Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2Error(f"cannot parse {label}: {exc}") from exc


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    actual = set(value)
    _require(actual == expected,
             f"{label} keys changed: missing={sorted(expected - actual)} "
             f"unexpected={sorted(actual - expected)}")
    return value


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    _require(type(value) in (int, float), f"{label} must be a number, not bool")
    number = float(value)
    _require(math.isfinite(number), f"{label} must be finite")
    if minimum is not None:
        _require(number >= minimum, f"{label} must be >= {minimum}")
    return number


def _ensure_no_symlink_components(path: Path, label: str,
                                  *, leaf_may_be_missing: bool = False) -> None:
    absolute = _absolute(path)
    current = Path(absolute.parts[0])
    for index, part in enumerate(absolute.parts[1:], start=1):
        current /= part
        leaf = index == len(absolute.parts) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            if leaf and leaf_may_be_missing:
                return
            raise Stage2Error(f"{label} component is missing: {current}")
        _require(not stat.S_ISLNK(info.st_mode), f"{label} contains symlink: {current}")


def _read_snapshot(path: Path | str, label: str) -> Snapshot:
    absolute = _absolute(path)
    _ensure_no_symlink_components(absolute, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(absolute, flags)
    except OSError as exc:
        raise Stage2Error(f"cannot open {label}: {absolute}: {exc}") from exc
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size,
                           before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns, after.st_ctime_ns)
        _require(identity_before == identity_after, f"{label} changed while reading")
        payload = b"".join(chunks)
        _require(len(payload) == before.st_size, f"{label} short read")
        return Snapshot(absolute, payload, stat.S_IMODE(before.st_mode))
    finally:
        os.close(fd)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    _ensure_no_symlink_components(path.parent, "output parent")
    _ensure_no_symlink_components(path, "output", leaf_may_be_missing=True)
    _require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, mode)
    except OSError as exc:
        raise Stage2Error(f"cannot publish no-clobber output {path}: {exc}") from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_queue(queue: Any) -> Mapping[str, Any]:
    queue = _exact_keys(queue, {
        "schema_version", "experiment_id", "created_utc", "human_owner", "executor",
        "purpose", "runtime", "assets", "fixed_contract",
        "observed_baseline_not_to_rerun", "staged_cells", "derived_cell_rule",
        "acceptance", "selection",
    }, "queue")
    _require(queue["schema_version"] == 1, "queue schema changed")
    _require(queue["experiment_id"] == EXPECTED_EXPERIMENT_ID, "queue experiment changed")
    runtime = _exact_keys(queue["runtime"], {
        "checkout_path", "checkout_commit", "generator_source_commit", "generator_sha256",
        "topp_sha256", "mjcf_sha256", "urdf_sha256", "body_order_sha256",
    }, "queue.runtime")
    _require(Path(runtime["checkout_path"]).is_absolute(), "runtime checkout must be absolute")
    for key in RUNTIME_RELATIVE_PATHS:
        _require(_is_sha256(runtime[key]), f"runtime {key} SHA is malformed")
    assets = _exact_keys(queue["assets"], {"forehand", "backhand"}, "queue.assets")
    for name, asset in assets.items():
        asset = _exact_keys(asset, {"path", "sha256", "contact_frame"}, f"asset {name}")
        _require(Path(asset["path"]).is_absolute(), f"asset {name} path must be absolute")
        _require(_is_sha256(asset["sha256"]), f"asset {name} SHA is malformed")
        _require(type(asset["contact_frame"]) is int and asset["contact_frame"] > 17,
                 f"asset {name} contact frame is invalid")
    fixed_expected = {
        "fps": 50, "ready_frame": 0, "ready_velocity": "bitwise_zero",
        "hold_frames": 4, "output_contact_frame": 25,
        "protected_precontact_seconds": 0.1, "delta_plus_blend_intervals": 22,
        "topp_objective": "runup", "body_mode": "fk", "automatic_retry": False,
        "gpu_or_trainer_signals": False, "training_authorized": False,
        "deployment_authorized": False, "hardware_authorized": False,
    }
    _require(queue["fixed_contract"] == fixed_expected, "queue fixed contract changed")
    _require(queue["derived_cell_rule"] == {
        "join_frame": "contact_frame - delta", "blend_intervals": "22 - delta",
        "output_contact_frame": 25,
    }, "queue derived-cell rule changed")
    staged = _exact_keys(queue["staged_cells"], {
        "stage1_endpoint_factorial", "stage2_midpoint_rule", "stage3_refinement_rule",
    }, "queue.staged_cells")
    stage1 = staged["stage1_endpoint_factorial"]
    _require(isinstance(stage1, list) and len(stage1) == 6, "queue must register six Stage1 cells")
    actual_stage1: dict[str, tuple[str, str, int]] = {}
    for row in stage1:
        row = _exact_keys(row, {"cell_id", "action", "ready_source", "delta"}, "Stage1 cell")
        _require(row["cell_id"] not in actual_stage1, "duplicate Stage1 cell")
        actual_stage1[row["cell_id"]] = (row["action"], row["ready_source"], row["delta"])
    _require(actual_stage1 == EXPECTED_STAGE1_CELLS, "Stage1 cell registry changed")
    midpoint = staged["stage2_midpoint_rule"]
    _require(isinstance(midpoint, Mapping) and midpoint.get("delta") == 12,
             "Stage2 midpoint rule changed")
    return queue


def _validate_activation(activation: Any, queue: Mapping[str, Any], *,
                         runner_sha256: str) -> tuple[Mapping[str, Any], list[dict[str, Any]], Path, str]:
    activation = _exact_keys(activation, {
        "schema_version", "activation_id", "created_utc", "parent_queue",
        "stage1_namespace", "observations", "decision", "evidence_status",
        "required_attestation_receipt", "required_attestation_receipt_sha256",
        "stage2_runner", "stage2_namespace", "launch_authorized",
        "authorized_stage2_cells",
        "runtime_authority",
    }, "activation")
    _require(activation["schema_version"] == 1, "activation schema changed")
    _require(activation["activation_id"] == EXPECTED_ACTIVATION_ID, "unknown activation id")
    _require(activation["parent_queue"] == {
        "path": "configs/ready_to_strike_join_ladder_20260717.yaml",
        "sha256": EXPECTED_QUEUE_SHA256,
        "main_commit": EXPECTED_PREREG_COMMIT,
    }, "activation parent queue changed")
    _require(activation["evidence_status"] == EXPECTED_EVIDENCE_STATUS,
             "Stage1 evidence is not historically attested screening-only")
    _require(activation["launch_authorized"] is True, "Stage2 launch is not authorized")
    _require(activation["stage2_runner"] == {
        "path": "scripts/run_ready_to_strike_join_ladder_stage2.py",
        "sha256": runner_sha256,
    }, "activation does not bind the executing Stage2 runner bytes")
    stage2_namespace = _absolute(activation["stage2_namespace"])
    _require(stage2_namespace.parent == _absolute(activation["stage1_namespace"]).parent,
             "Stage2 namespace is not a sibling of the attested Stage1 namespace")
    runtime_authority = {
        "cpu_only": True, "automatic_retry": False, "trainer_signal": False,
        "robot_command": False, "training_authorized": False,
        "deployment_authorized": False,
    }
    _require(activation["runtime_authority"] == runtime_authority,
             "activation runtime authority changed")
    expected_decision = {
        "ready_by_side_crossover": True, "forehand_prefers": "forehand",
        "backhand_prefers": "backhand",
        "delta17_strictly_worse_than_delta6_all_four_ready_by_side_pairs": True,
        "activate_both_ready_sources_at_midpoint": True,
        "shared_ready_not_yet_selected": True,
    }
    _require(activation["decision"] == expected_decision, "activation decision changed")
    observations = activation["observations"]
    _require(isinstance(observations, list) and len(observations) == 8,
             "activation must contain eight Stage1 observations")
    observed: dict[str, tuple[str, str, int, str, str, float]] = {}
    for row in observations:
        row = _exact_keys(row, {"cell_id", "action", "ready_source", "delta",
                                "candidate_sha256", "topp_certificate_sha256",
                                "start_to_contact_s"}, "activation observation")
        identity = (row["action"], row["ready_source"], row["delta"],
                    row["candidate_sha256"], row["topp_certificate_sha256"],
                    _finite(row["start_to_contact_s"], "observation timing", minimum=0.0))
        _require(row["cell_id"] not in observed, "duplicate activation observation")
        observed[row["cell_id"]] = identity
    _require(observed == EXPECTED_OBSERVATIONS, "activation Stage1 observations changed")
    cells = activation["authorized_stage2_cells"]
    _require(isinstance(cells, list) and len(cells) == 4,
             "activation must contain four Stage2 cells")
    actual_cells: dict[str, tuple[str, str, int]] = {}
    normalized: list[dict[str, Any]] = []
    for row in cells:
        row = _exact_keys(row, {"cell_id", "action", "ready_source", "delta"}, "Stage2 cell")
        identity = (row["action"], row["ready_source"], row["delta"])
        _require(row["cell_id"] not in actual_cells, "duplicate Stage2 cell")
        actual_cells[row["cell_id"]] = identity
        normalized.append(dict(row))
    _require(actual_cells == EXPECTED_STAGE2_CELLS, "Stage2 cell set changed")
    receipt_path = _absolute(activation["required_attestation_receipt"])
    stage1_root = _absolute(activation["stage1_namespace"])
    _require(receipt_path == stage1_root / "stage1_historical_attestation.json",
             "attestation receipt path is not inside the declared Stage1 namespace")
    receipt_sha = activation["required_attestation_receipt_sha256"]
    _require(_is_sha256(receipt_sha), "attestation receipt SHA is missing or malformed")
    return activation, normalized, receipt_path, receipt_sha


def _validate_stage1_receipt(document: Any, *, receipt_sha: str,
                             receipt_payload: bytes, queue: Mapping[str, Any],
                             activation: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(_sha256(receipt_payload) == receipt_sha, "Stage1 receipt bytes do not match activation SHA")
    receipt = _exact_keys(document, {
        "schema_version", "artifact_kind", "status", "experiment_id", "inputs",
        "cells", "attestor", "formal_claims", "runtime_authority",
    }, "Stage1 receipt")
    _require(receipt["schema_version"] == 1, "Stage1 receipt schema changed")
    _require(receipt["artifact_kind"] == "ready_to_strike_stage1_historical_attestation",
             "Stage1 receipt kind changed")
    _require(receipt["status"] == "historical_evidence_attested_no_runtime_authority",
             "Stage1 receipt status changed")
    _require(receipt["experiment_id"] == EXPECTED_EXPERIMENT_ID,
             "Stage1 receipt experiment changed")
    inputs = _exact_keys(receipt["inputs"], {
        "root", "queue_path", "queue_sha256", "generator_sha256", "summary_sha256",
        "runtime_source_commit",
    }, "Stage1 receipt inputs")
    _require(inputs["root"] == activation["stage1_namespace"], "Stage1 receipt root is misbound")
    _require(inputs["queue_sha256"] == EXPECTED_QUEUE_SHA256, "Stage1 receipt queue is misbound")
    _require(inputs["generator_sha256"] == queue["runtime"]["generator_sha256"],
             "Stage1 receipt generator is misbound")
    _require(inputs["runtime_source_commit"] == queue["runtime"]["checkout_commit"],
             "Stage1 receipt runtime commit is misbound")
    _require(_is_sha256(inputs["summary_sha256"]), "Stage1 summary SHA is malformed")
    _require(receipt["formal_claims"] == {
        "physics_replay_exact": False, "source_closure_exact": False,
        "mjcf_closure_exact": False, "screening_activation_evidence_only": True,
    }, "Stage1 receipt formal claims changed")
    _require(receipt["runtime_authority"] == {
        "read_only_historical": True, "ssh": False, "process_signal": False,
        "automatic_retry": False, "simulator": False, "trainer": False,
        "deployment": False, "robot_command": False,
    }, "Stage1 receipt runtime authority changed")
    attestor = _exact_keys(receipt["attestor"], {
        "source_sha256", "launch_snapshot", "launch_snapshot_sha256",
        "source_and_inputs_unchanged_before_publish",
    }, "Stage1 receipt attestor")
    _require(_is_sha256(attestor["source_sha256"]), "attestor source SHA is malformed")
    _require(attestor["source_and_inputs_unchanged_before_publish"] is True,
             "Stage1 inputs were not stable before receipt publication")
    _require(attestor["launch_snapshot_sha256"] == _sha256(_canonical_json(attestor["launch_snapshot"])),
             "Stage1 attestor launch snapshot is misbound")
    observation_by_id = {row["cell_id"]: row for row in activation["observations"]}
    cells = receipt["cells"]
    _require(isinstance(cells, list) and len(cells) == 6, "Stage1 receipt must bind six cells")
    actual: dict[str, tuple[str, str, int]] = {}
    for row in cells:
        row = _exact_keys(row, {
            "cell_id", "action", "ready_source", "delta", "join_frame",
            "blend_intervals", "candidate_sha256", "generator_contract_sha256",
            "output_sha256", "certificate_sha256", "candidate_start_to_contact_s",
            "within_0p5_s", "doses",
        }, "Stage1 receipt cell")
        cell_id = row["cell_id"]
        _require(cell_id in EXPECTED_STAGE1_CELLS and cell_id not in actual,
                 f"unexpected or duplicate Stage1 receipt cell {cell_id!r}")
        identity = (row["action"], row["ready_source"], row["delta"])
        _require(identity == EXPECTED_STAGE1_CELLS[cell_id], f"Stage1 receipt cell {cell_id} changed")
        actual[cell_id] = identity
        observation = observation_by_id[cell_id]
        _require(row["candidate_sha256"] == observation["candidate_sha256"],
                 f"Stage1 receipt candidate SHA for {cell_id} is misbound")
        _require(row["certificate_sha256"] == observation["topp_certificate_sha256"],
                 f"Stage1 receipt certificate SHA for {cell_id} is misbound")
        for key in ("generator_contract_sha256", "output_sha256"):
            _require(_is_sha256(row[key]), f"Stage1 receipt {cell_id}.{key} is malformed")
        contact = queue["assets"][row["action"]]["contact_frame"]
        _require(row["join_frame"] == contact - row["delta"], "Stage1 join frame is misbound")
        _require(row["blend_intervals"] == 22 - row["delta"], "Stage1 blend is misbound")
        timing = _finite(row["candidate_start_to_contact_s"], "Stage1 receipt timing", minimum=0.0)
        _require(type(row["within_0p5_s"]) is bool and row["within_0p5_s"] == (timing <= 0.5),
                 "Stage1 within-0.5 flag is misbound")
        doses = _exact_keys(row["doses"], {"cop", "friction", "torque"}, "Stage1 doses")
        for name, value in doses.items():
            _finite(value, f"Stage1 {cell_id} {name} dose", minimum=0.0)
    _require(actual == EXPECTED_STAGE1_CELLS, "Stage1 receipt cell set is incomplete")
    return receipt


def _load_npz(payload: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)), f"{label} contains duplicate ZIP entries")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise Stage2Error(f"cannot parse {label}: {exc}") from exc
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number):
            _require(bool(np.all(np.isfinite(value))), f"{label}.{key} contains non-finite values")
    return arrays


def _scalar_unicode(array: np.ndarray, label: str) -> str:
    value = np.asarray(array)
    _require(value.shape == () and value.dtype.kind == "U" and not value.dtype.hasobject,
             f"{label} must be one canonical Unicode scalar")
    return str(value.item())


def _validate_schema2(arrays: Mapping[str, np.ndarray], *, label: str,
                      body_order: Sequence[str], allow_migration: bool) -> int:
    time_keys = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                 "body_lin_vel_w", "body_ang_vel_w"}
    metadata_keys = {"fps", "kinematics_schema_version", "body_pos_point",
                     "body_lin_vel_point", "body_names"}
    migration_keys = {"kinematics_migration_source_sha256",
                      "kinematics_migration_source_point", "kinematics_migration_tool"}
    actual = set(arrays)
    core = time_keys | metadata_keys
    allowed = {frozenset(core)}
    if allow_migration:
        allowed.add(frozenset(core | migration_keys))
    _require(frozenset(actual) in allowed,
             f"{label} schema keys changed: {sorted(actual)}")
    fps = np.asarray(arrays["fps"])
    schema = np.asarray(arrays["kinematics_schema_version"])
    _require(fps.dtype == np.int64 and fps.shape == (1,) and int(fps[0]) == 50,
             f"{label}.fps changed")
    _require(schema.dtype == np.int64 and schema.shape == (1,) and int(schema[0]) == 2,
             f"{label}.kinematics_schema_version changed")
    _require(_scalar_unicode(arrays["body_pos_point"], f"{label}.body_pos_point") == "link_origin",
             f"{label}.body_pos_point changed")
    _require(_scalar_unicode(arrays["body_lin_vel_point"], f"{label}.body_lin_vel_point") == "center_of_mass",
             f"{label}.body_lin_vel_point changed")
    names = np.asarray(arrays["body_names"])
    _require(names.dtype.kind == "U" and names.shape == (len(body_order),)
             and tuple(str(value) for value in names.tolist()) == tuple(body_order),
             f"{label}.body_names differs from runtime order")
    q = np.asarray(arrays["joint_pos"])
    _require(q.dtype == np.float32 and q.ndim == 2 and q.shape[0] > 26 and q.shape[1] == 31,
             f"{label}.joint_pos shape/dtype changed")
    frames, bodies = q.shape[0], len(body_order)
    expected_shapes = {
        "joint_pos": (frames, 31), "joint_vel": (frames, 31),
        "body_pos_w": (frames, bodies, 3), "body_quat_w": (frames, bodies, 4),
        "body_lin_vel_w": (frames, bodies, 3), "body_ang_vel_w": (frames, bodies, 3),
    }
    for key, shape in expected_shapes.items():
        value = np.asarray(arrays[key])
        _require(value.dtype == np.float32 and value.shape == shape,
                 f"{label}.{key} shape/dtype changed")
    expected_velocity = np.gradient(q.astype(np.float64), 1.0 / 50.0, axis=0).astype(np.float32)
    _require(np.array_equal(np.asarray(arrays["joint_vel"]), expected_velocity),
             f"{label}.joint_vel is not the canonical position gradient")
    quaternion_norm = np.linalg.norm(np.asarray(arrays["body_quat_w"], dtype=np.float64), axis=-1)
    _require(float(np.max(np.abs(quaternion_norm - 1.0))) <= 2.0e-5,
             f"{label}.body_quat_w is not normalized")
    if migration_keys <= actual:
        _require(_is_sha256(_scalar_unicode(arrays["kinematics_migration_source_sha256"],
                                             f"{label}.migration SHA")),
                 f"{label} migration SHA is malformed")
        _require(_scalar_unicode(arrays["kinematics_migration_source_point"],
                                 f"{label}.migration point") in {"link_origin", "center_of_mass"},
                 f"{label} migration point changed")
        _require(_scalar_unicode(arrays["kinematics_migration_tool"],
                                 f"{label}.migration tool") == "migrate_motion_kinematics.py/v2",
                 f"{label} migration tool changed")
    return frames


def _validate_candidate(candidate: Snapshot, contract: Snapshot, *, cell: Mapping[str, Any],
                        queue: Mapping[str, Any], generator_path: Path,
                        asset_paths: Mapping[str, Path],
                        body_order: Sequence[str]) -> dict[str, Any]:
    arrays = _load_npz(candidate.payload, f"{cell['cell_id']} candidate")
    frames = _validate_schema2(arrays, label=f"{cell['cell_id']} candidate",
                               body_order=body_order, allow_migration=True)
    q = np.asarray(arrays["joint_pos"])
    qd = np.asarray(arrays["joint_vel"])
    for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        _require(np.array_equal(np.asarray(arrays[key])[:3], np.zeros_like(arrays[key][:3])),
                 f"candidate {key} does not start bitwise zero")
    document = _exact_keys(_load_json(contract.payload, "candidate contract"), {
        "schema_version", "artifact_kind", "status", "inputs", "tool", "request",
        "synthesis", "proof", "output", "authorization", "required_next_gates",
        "explicit_non_claims",
    }, "candidate contract")
    _require(document["schema_version"] == 1, "candidate contract schema changed")
    inputs = _exact_keys(document["inputs"], {
        "source_schema2_npz", "shared_ready_schema2_npz", "shared_ready_frame",
    }, "candidate inputs")
    _require(inputs["shared_ready_frame"] == 0, "candidate ready frame changed")
    for evidence, asset_name, evidence_label in (
        (inputs["source_schema2_npz"], cell["action"], "source"),
        (inputs["shared_ready_schema2_npz"], cell["ready_source"], "ready"),
    ):
        evidence = _exact_keys(evidence, {
            "path", "bytes", "sha256", "device", "inode", "mtime_ns", "ctime_ns",
        }, f"candidate {evidence_label} evidence")
        asset = queue["assets"][asset_name]
        _require(evidence["path"] == str(asset_paths[asset_name])
                 and evidence["sha256"] == asset["sha256"],
                 f"candidate {evidence_label} asset is misbound")
    proof = document["proof"]
    required_proof = {
        "fps": 50, "source_contact_frame": queue["assets"][cell["action"]]["contact_frame"],
        "source_join_frame": queue["assets"][cell["action"]]["contact_frame"] - 12,
        "output_contact_frame": 25, "protected_frames_before_contact": 5,
        "protected_window_bitwise_equal": True,
        "pose_and_body_velocity_source_suffix_bitwise_equal": True,
        "frame0_shared_ready_pose_bitwise_equal": True,
        "ready_source_velocity_channels_ignored": True,
        "ready_velocity_definition": "explicit_bitwise_zero",
        "initial_zero_velocity_frames": 3,
        "joint_position_continuous_quintic_endpoint_c2": True,
        "finite": True, "contact_time_from_frame0_s": 0.5,
    }
    for key, value in required_proof.items():
        _require(proof.get(key) == value, f"candidate proof {key} changed")
    _require(document["request"] == {
        "source_contact_frame": queue["assets"][cell["action"]]["contact_frame"],
        "source_join_frame": queue["assets"][cell["action"]]["contact_frame"] - 12,
        "ready_hold_frames": 4, "quintic_blend_intervals": 10,
        "protected_precontact_seconds": 0.1,
    }, "candidate request changed")
    _require(document["tool"].get("path") == str(generator_path)
             and document["tool"].get("sha256") == queue["runtime"]["generator_sha256"],
             "candidate generator evidence is misbound")
    _require(document["output"]["npz"].get("sha256") == candidate.sha256,
             "candidate contract does not bind candidate bytes")
    authorization = document["authorization"]
    _require(authorization.get("training_authorized") is False
             and authorization.get("deployment_authorized") is False
             and authorization.get("hardware_authorized") is False,
             "candidate authorization escaped screening")
    segment = np.diff(q[:26].astype(np.float64), axis=0)
    second = np.diff(q[:26].astype(np.float64), n=2, axis=0)
    return {
        "candidate_sha256": candidate.sha256,
        "generator_contract_sha256": contract.sha256,
        "frames": frames,
        "phase": 25.0 / float(frames - 1),
        "joint_path_l2": float(np.linalg.norm(segment, axis=1).sum()),
        "joint_curvature_l2": float(np.linalg.norm(second, axis=1).sum()),
        "max_joint_step_rad": float(np.abs(segment).max()),
    }


def _validate_topp(certificate: Snapshot, output: Snapshot, markdown: Snapshot, *,
                   candidate: Snapshot, candidate_info: Mapping[str, Any],
                   queue: Mapping[str, Any], paths: Mapping[str, Path],
                   body_order: Sequence[str]) -> dict[str, Any]:
    document = _exact_keys(_load_json(certificate.payload, "TOPP certificate"),
                           TOPP_CERTIFICATE_KEYS, "TOPP certificate")
    _require(document["tool"] == TOPP_TOOL, "TOPP tool/schema identity changed")
    _require(document["algorithm_scope"] == TOPP_ALGORITHM_SCOPE,
             "TOPP algorithm-scope honesty changed")
    _require(document["search_objective"] == "runup", "TOPP objective changed")
    files = _exact_keys(document["files"], {"input", "output", "report_path", "markdown_path"},
                       "TOPP files")
    _require(files["input"].get("sha256") == candidate.sha256, "TOPP input is misbound")
    _require(files["output"].get("sha256") == output.sha256, "TOPP output is misbound")
    _require(files["report_path"] == str(certificate.path), "TOPP report path is misbound")
    _require(files["markdown_path"] == str(markdown.path), "TOPP markdown path is misbound")
    arrays = _load_npz(output.payload, "TOPP output")
    output_frames = _validate_schema2(arrays, label="TOPP output", body_order=body_order,
                                      allow_migration=False)
    candidate_arrays = _load_npz(candidate.payload, "TOPP candidate input")
    acceptance = _exact_keys(document["acceptance"], {
        "cop_dose_final", "fric_dose_final", "tau_dose_final", "within_budget",
        "kin_out_window_clean", "kin_lock_window_clean", "kinematic_hard_limits_clean",
    }, "TOPP acceptance")
    for key in ("within_budget", "kin_out_window_clean", "kin_lock_window_clean",
                "kinematic_hard_limits_clean"):
        _require(acceptance[key] is True, f"TOPP {key} failed")
    source_doc = _exact_keys(document["source"], {
        "frames", "fps", "contact_frame", "phase", "runup_s", "duration_s",
        "clean_blade_speed_mps", "mean_abs_acc",
    }, "TOPP source")
    output_doc = _exact_keys(document["output"], {
        "frames", "fps", "contact_frame", "phase_out", "runup_s", "duration_s",
        "runup_change_x", "duration_change_x", "wait_s", "body_mode", "mean_abs_acc",
    }, "TOPP output")
    _require(output_doc.get("body_mode") == "fk", "TOPP did not use production FK")
    _require(output_doc.get("frames") == output_frames and output_doc.get("fps") == 50.0,
             "TOPP output frame/fps tuple is misbound")
    _require(source_doc["frames"] == candidate_info["frames"]
             and source_doc["fps"] == 50 and source_doc["contact_frame"] == 25
             and math.isclose(_finite(source_doc["phase"], "TOPP source phase"),
                              candidate_info["phase"], rel_tol=0.0, abs_tol=1e-12),
             "TOPP source tuple is misbound")
    source_runup = _finite(source_doc["runup_s"], "TOPP source runup", minimum=0.0)
    _require(math.isclose(source_runup,
                          round(source_doc["contact_frame"] / source_doc["fps"], 4),
                          rel_tol=0.0, abs_tol=5.1e-5),
             "TOPP source runup disagrees with contact frame/fps")
    output_contact = output_doc["contact_frame"]
    _require(type(output_contact) is int and 0 <= output_contact < output_frames,
             "TOPP output contact frame is invalid")
    _require(math.isclose(_finite(output_doc["phase_out"], "TOPP output phase"),
                          output_contact / float(output_frames - 1),
                          rel_tol=0.0, abs_tol=5.1e-7), "TOPP output phase is misbound")
    output_runup = _finite(output_doc["runup_s"], "TOPP output runup", minimum=0.0)
    _require(math.isclose(output_runup,
                          round(output_contact / float(output_doc["fps"]), 4),
                          rel_tol=0.0, abs_tol=5.1e-5),
             "TOPP output runup disagrees with contact frame/fps")
    _require(np.array_equal(np.asarray(arrays["joint_pos"])[output_contact],
                            np.asarray(candidate_arrays["joint_pos"])[25]),
             "TOPP output contact row differs from candidate")
    fidelity = _exact_keys(document["fidelity"], {
        "contact_row_bitwise", "blade_speed_clean_out_mps", "blade_speed_dev_frac",
        "face_normal_diff_deg", "first_frame_max_joint_vel",
    }, "TOPP fidelity")
    _require(fidelity["contact_row_bitwise"] is True, "TOPP contact row changed")
    _require(_finite(fidelity["blade_speed_dev_frac"], "TOPP blade-speed deviation", minimum=0.0) <= 0.02,
             "TOPP blade-speed deviation exceeds 2 percent")
    _require(_finite(fidelity["first_frame_max_joint_vel"], "TOPP first velocity", minimum=0.0) == 0.0,
             "TOPP output does not start at rest")
    _require(float(np.max(np.abs(np.asarray(arrays["joint_vel"])[0]))) == 0.0,
             "TOPP NPZ does not start at rest")
    timing = _exact_keys(document["timing_bound"], {
        "candidate_start_to_contact_s", "bound_semantics", "strict_global_minimum_proven",
    }, "TOPP timing bound")
    timing_s = _finite(timing["candidate_start_to_contact_s"], "TOPP timing", minimum=0.0)
    _require(timing["strict_global_minimum_proven"] is False, "TOPP falsely claims global minimum")
    _require(math.isclose(timing_s, output_runup, rel_tol=0.0, abs_tol=1e-12),
             "TOPP timing disagrees with output")
    provenance = document["runtime_provenance"]
    for name in ("mjcf", "urdf", "body_order"):
        _require(provenance[name].get("sha256") == queue["runtime"][f"{name}_sha256"],
                 f"TOPP {name} evidence is misbound")
    _require(provenance["tool"]["topp_mintime"].get("sha256") == queue["runtime"]["topp_sha256"],
             "TOPP tool evidence is misbound")
    budget_provenance = _exact_keys(document["budget_provenance"],
                                    {"clips", "scale", "envelope"},
                                    "TOPP budget provenance")
    _require(_finite(budget_provenance["scale"], "TOPP budget scale", minimum=0.0) == 1.5,
             "TOPP budget scale changed from the source-pinned default 1.5")
    _require(isinstance(budget_provenance["envelope"], list)
             and budget_provenance["envelope"], "TOPP budget envelope is empty")
    for index, value in enumerate(budget_provenance["envelope"]):
        _finite(value, f"TOPP budget envelope {index}", minimum=0.0)
    budget_clips = budget_provenance["clips"]
    _require(isinstance(budget_clips, list) and len(budget_clips) == 2,
             "TOPP budget clip count changed")
    for evidence, name in zip(budget_clips, ("forehand", "backhand")):
        _require(evidence.get("sha256") == queue["assets"][name]["sha256"],
                 f"TOPP budget clip {name} is misbound")
    doses = {
        "cop": _finite(acceptance["cop_dose_final"], "CoP dose", minimum=0.0),
        "friction": _finite(acceptance["fric_dose_final"], "friction dose", minimum=0.0),
        "torque": _finite(acceptance["tau_dose_final"], "torque dose", minimum=0.0),
    }
    budget = _exact_keys(document["budget"], {
        "cop_gate", "fric_gate", "tau_gate", "vel_limit_frac", "kin_vel_target",
        "kin_acc_target", "note",
    }, "TOPP budget")
    gates = {
        "cop": _finite(budget["cop_gate"], "CoP gate", minimum=0.0),
        "friction": _finite(budget["fric_gate"], "friction gate", minimum=0.0),
        "torque": _finite(budget["tau_gate"], "torque gate", minimum=0.0),
    }
    for name in doses:
        _require(doses[name] <= gates[name] + 5e-5,
                 f"TOPP {name} dose exceeds its recorded gate")
    return {
        "output_sha256": output.sha256,
        "topp_certificate_sha256": certificate.sha256,
        "topp_markdown_sha256": markdown.sha256,
        "candidate_start_to_contact_s": timing_s,
        "within_0p5_s": timing_s <= 0.5,
        "doses": doses,
    }


def _validate_inputs(*, activation_path: Path | str, queue_path: Path | str,
                     root: Path | str, runner_source: Path | str | None = None) -> dict[str, Any]:
    root = _absolute(root)
    _require(root.is_absolute(), "Stage2 root must be absolute")
    _ensure_no_symlink_components(root.parent, "Stage2 root parent")
    _ensure_no_symlink_components(root, "Stage2 root", leaf_may_be_missing=True)
    _require(not root.exists() and not root.is_symlink(), f"Stage2 root already exists: {root}")
    source_snapshot = _read_snapshot(runner_source or Path(__file__), "Stage2 runner source")
    activation_snapshot = _read_snapshot(activation_path, "activation")
    queue_snapshot = _read_snapshot(queue_path, "queue")
    _require(queue_snapshot.sha256 == EXPECTED_QUEUE_SHA256, "queue bytes differ from preregistration")
    queue = _validate_queue(_load_json(queue_snapshot.payload, "queue"))
    activation, cells, receipt_path, receipt_sha = _validate_activation(
        _load_json(activation_snapshot.payload, "activation"), queue,
        runner_sha256=source_snapshot.sha256)
    _require(root == _absolute(activation["stage2_namespace"]),
             "requested root differs from the one-shot Stage2 activation namespace")
    receipt_snapshot = _read_snapshot(receipt_path, "Stage1 attestation receipt")
    receipt = _validate_stage1_receipt(
        _load_json(receipt_snapshot.payload, "Stage1 receipt"),
        receipt_sha=receipt_sha, receipt_payload=receipt_snapshot.payload,
        queue=queue, activation=activation)
    runtime_root = _absolute(queue["runtime"]["checkout_path"])
    _require(runtime_root.is_dir(), "runtime checkout is missing")
    snapshots: dict[str, Snapshot] = {
        "runner": source_snapshot, "activation": activation_snapshot,
        "queue": queue_snapshot, "receipt": receipt_snapshot,
    }
    for key, relative in RUNTIME_RELATIVE_PATHS.items():
        snapshot = _read_snapshot(runtime_root / relative, f"runtime {key}")
        _require(snapshot.sha256 == queue["runtime"][key], f"runtime {key} SHA changed")
        snapshots[f"runtime:{relative}"] = snapshot
    for relative in TOPP_CLOSURE_PATHS:
        key = f"runtime:{relative}"
        if key not in snapshots:
            snapshots[key] = _read_snapshot(runtime_root / relative, f"TOPP closure {relative}")
    for name, asset in queue["assets"].items():
        snapshot = _read_snapshot(asset["path"], f"asset {name}")
        _require(snapshot.sha256 == asset["sha256"], f"asset {name} SHA changed")
        snapshots[f"asset:{name}"] = snapshot
    try:
        body_order = tuple(
            line.strip()
            for line in snapshots[f"runtime:{RUNTIME_RELATIVE_PATHS['body_order_sha256']}"]
            .payload.decode("utf-8").splitlines()
            if line.strip()
        )
    except UnicodeDecodeError as exc:
        raise Stage2Error("runtime body order is not UTF-8") from exc
    _require(body_order and len(body_order) == len(set(body_order)),
             "runtime body order must contain unique non-empty names")
    return {
        "root": root, "queue": queue, "activation": activation, "receipt": receipt,
        "cells": cells, "snapshots": snapshots, "runtime_root": runtime_root,
        "body_order": body_order,
    }


def plan_stage2(*, activation_path: Path | str, queue_path: Path | str,
                root: Path | str, runner_source: Path | str | None = None) -> dict[str, Any]:
    context = _validate_inputs(activation_path=activation_path, queue_path=queue_path,
                               root=root, runner_source=runner_source)
    queue = context["queue"]
    plans: list[dict[str, Any]] = []
    for cell in context["cells"]:
        contact = queue["assets"][cell["action"]]["contact_frame"]
        plans.append({**cell, "join_frame": contact - 12, "blend_intervals": 10,
                      "output_contact_frame": 25})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run_passed_no_namespace_created",
        "activation_id": EXPECTED_ACTIVATION_ID,
        "root": str(context["root"]),
        "cells": plans,
        "receipt_sha256": context["snapshots"]["receipt"].sha256,
        "runner_sha256": context["snapshots"]["runner"].sha256,
        "runtime_authority": context["activation"]["runtime_authority"],
    }


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _default_command_runner(command: Sequence[str], *, cwd: Path,
                            env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=cwd, env=dict(env), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, check=False,
                          timeout=CHILD_TIMEOUT_S)


def _run_stage2_impl(*, activation_path: Path | str, queue_path: Path | str,
                     root: Path | str, execute: bool = False,
                     confirm: str | None = None,
                     runner_source: Path | str | None = None,
                     command_runner: CommandRunner = _default_command_runner,
                     _ownership: dict[str, bool] | None = None) -> dict[str, Any]:
    if not execute:
        _require(confirm is None, "--confirm is only valid with --execute")
        return plan_stage2(activation_path=activation_path, queue_path=queue_path,
                           root=root, runner_source=runner_source)
    _require(confirm == CONFIRM_TOKEN, f"--execute requires --confirm {CONFIRM_TOKEN}")
    context = _validate_inputs(activation_path=activation_path, queue_path=queue_path,
                               root=root, runner_source=runner_source)
    root_path: Path = context["root"]
    root_path.mkdir(mode=0o700)
    if _ownership is not None:
        _ownership["created"] = True
    snapshots: Mapping[str, Snapshot] = context["snapshots"]
    snapshot_root = root_path / "snapshots"
    snapshot_root.mkdir(mode=0o700)
    snapshot_destinations: dict[str, Path] = {
        "runner": snapshot_root / "run_ready_to_strike_join_ladder_stage2.py",
        "activation": snapshot_root / "activation.json",
        "queue": snapshot_root / "queue.json",
        "receipt": snapshot_root / "stage1_historical_attestation.json",
    }
    for key in ("runner", "activation", "queue", "receipt"):
        _write_exclusive(snapshot_destinations[key], snapshots[key].payload)
    runtime_snapshot_root = snapshot_root / "runtime"
    runtime_snapshot_root.mkdir(mode=0o700)
    for key, snapshot in snapshots.items():
        if not key.startswith("runtime:"):
            continue
        relative = Path(key.split(":", 1)[1])
        destination = runtime_snapshot_root / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_exclusive(destination, snapshot.payload)
    asset_snapshot_root = snapshot_root / "assets"
    asset_snapshot_root.mkdir(mode=0o700)
    asset_snapshot_paths: dict[str, Path] = {}
    for name in ("forehand", "backhand"):
        destination = asset_snapshot_root / f"{name}.npz"
        _write_exclusive(destination, snapshots[f"asset:{name}"].payload)
        asset_snapshot_paths[name] = destination

    queue: Mapping[str, Any] = context["queue"]
    runtime_root: Path = context["runtime_root"]
    generator = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["generator_sha256"]
    topp = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["topp_sha256"]
    mjcf = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["mjcf_sha256"]
    urdf = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["urdf_sha256"]
    body_order = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["body_order_sha256"]
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    rows: list[dict[str, Any]] = []
    candidate_records: dict[str, tuple[Path, Snapshot, dict[str, Any]]] = {}
    for cell in context["cells"]:
        cell_root = root_path / cell["cell_id"]
        cell_root.mkdir(mode=0o700)
        action = queue["assets"][cell["action"]]
        join = action["contact_frame"] - 12
        candidate_path = cell_root / "candidate.npz"
        contract_path = cell_root / "candidate.contract.json"
        command = [
            sys.executable, "-B", str(generator), "--source",
            str(asset_snapshot_paths[cell["action"]]),
            "--ready-source", str(asset_snapshot_paths[cell["ready_source"]]),
            "--ready-frame", "0",
            "--contact-frame", str(action["contact_frame"]), "--join-frame", str(join),
            "--hold-frames", "4", "--blend-intervals", "10",
            "--output-npz", str(candidate_path), "--output-contract", str(contract_path),
        ]
        row: dict[str, Any] = {**cell, "join_frame": join, "blend_intervals": 10,
                               "generator_rc": None}
        rows.append(row)
        try:
            completed = command_runner(command, cwd=runtime_root, env=env)
        except Exception as exc:
            row["terminal_error"] = f"generator_spawn_or_wait_failed:{type(exc).__name__}:{exc}"
            continue
        _write_exclusive(cell_root / "generator.log", completed.stdout.encode("utf-8"))
        row["generator_rc"] = completed.returncode
        if completed.returncode != 0:
            continue
        try:
            candidate_snapshot = _read_snapshot(candidate_path, f"{cell['cell_id']} candidate")
            contract_snapshot = _read_snapshot(contract_path, f"{cell['cell_id']} contract")
            info = _validate_candidate(
                candidate_snapshot, contract_snapshot, cell=cell, queue=queue,
                generator_path=generator, asset_paths=asset_snapshot_paths,
                body_order=context["body_order"])
        except Stage2Error as exc:
            row["terminal_error"] = f"candidate_validation_failed:{exc}"
            continue
        row.update(info)
        candidate_records[cell["cell_id"]] = (candidate_path, candidate_snapshot, info)

    def run_topp(cell: Mapping[str, Any], row: dict[str, Any]) -> tuple[str, subprocess.CompletedProcess[str] | None, str | None]:
        candidate_path, _candidate_snapshot, info = candidate_records[cell["cell_id"]]
        topp_root = root_path / cell["cell_id"] / "topp"
        topp_root.mkdir(mode=0o700)
        command = [
            sys.executable, "-B", str(topp), "--input", str(candidate_path),
            "--phase", format(info["phase"], ".17g"), "--objective", "runup",
            "--budget-clips", str(asset_snapshot_paths["forehand"]),
            str(asset_snapshot_paths["backhand"]), "--mjcf", str(mjcf),
            "--urdf", str(urdf), "--body-order", str(body_order), "--body-mode", "fk",
            "--output", str(topp_root / "motion.npz"),
            "--report", str(topp_root / "certificate.json"),
            "--md", str(topp_root / "certificate.md"),
        ]
        try:
            completed = command_runner(
                command, cwd=runtime_root / "hope_training/whole_body_tracking", env=env)
        except Exception as exc:
            return cell["cell_id"], None, f"topp_spawn_or_wait_failed:{type(exc).__name__}:{exc}"
        return cell["cell_id"], completed, None

    row_by_id = {row["cell_id"]: row for row in rows}
    futures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for cell in context["cells"]:
            if cell["cell_id"] in candidate_records:
                futures.append(pool.submit(run_topp, cell, row_by_id[cell["cell_id"]]))
        for future in as_completed(futures):
            cell_id, completed, terminal_error = future.result()
            topp_root = root_path / cell_id / "topp"
            row = row_by_id[cell_id]
            if terminal_error is not None or completed is None:
                row["terminal_error"] = terminal_error or "TOPP child failed without result"
                continue
            _write_exclusive(topp_root / "run.log", completed.stdout.encode("utf-8"))
            row["topp_rc"] = completed.returncode
            if completed.returncode != 0:
                continue
            try:
                certificate = _read_snapshot(topp_root / "certificate.json", f"{cell_id} certificate")
                output = _read_snapshot(topp_root / "motion.npz", f"{cell_id} TOPP output")
                markdown = _read_snapshot(topp_root / "certificate.md", f"{cell_id} TOPP markdown")
                candidate_path, candidate_snapshot, info = candidate_records[cell_id]
                row.update(_validate_topp(
                    certificate, output, markdown, candidate=candidate_snapshot,
                    candidate_info=info, queue=queue,
                    paths={"candidate": candidate_path, "topp": topp, "mjcf": mjcf,
                           "urdf": urdf, "body_order": body_order},
                    body_order=context["body_order"],
                ))
            except Stage2Error as exc:
                row["terminal_error"] = f"topp_validation_failed:{exc}"

    input_stability_errors: list[str] = []
    for key, original in snapshots.items():
        try:
            current = _read_snapshot(original.path, f"unchanged input {key}")
            _require(current.payload == original.payload,
                     f"input changed during Stage2: {original.path}")
        except Stage2Error as exc:
            input_stability_errors.append(str(exc))
    execution_complete = (
        len(rows) == 4 and not input_stability_errors
        and all(row.get("generator_rc") == 0 and row.get("topp_rc") == 0
                and "topp_certificate_sha256" in row and "terminal_error" not in row
                for row in rows)
    )
    accepted_cells = sorted(
        row["cell_id"] for row in rows if row.get("within_0p5_s") is True
    )
    timing_by_cell = {
        row["cell_id"]: row.get("candidate_start_to_contact_s") for row in rows
        if "candidate_start_to_contact_s" in row
    }
    shared_ready_pass = {
        ready: all(
            row_by_id[f"{side}_{'rf' if ready == 'forehand' else 'rb'}_d12"].get("within_0p5_s") is True
            for side in ("fh", "bh")
        )
        for ready in ("forehand", "backhand")
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "ready_to_strike_join_ladder_stage2_screening_result",
        "status": ("stage2_execution_complete_no_retry"
                   if execution_complete else "stage2_terminal_failure_no_retry"),
        "activation_sha256": snapshots["activation"].sha256,
        "queue_sha256": snapshots["queue"].sha256,
        "stage1_receipt_sha256": snapshots["receipt"].sha256,
        "runner_sha256": snapshots["runner"].sha256,
        "runtime_snapshot_shas": {
            key.split(":", 1)[1]: value.sha256
            for key, value in sorted(snapshots.items()) if key.startswith("runtime:")
        },
        "asset_snapshot_shas": {
            key.split(":", 1)[1]: value.sha256
            for key, value in sorted(snapshots.items()) if key.startswith("asset:")
        },
        "rows": rows,
        "screening_acceptance": {
            "at_or_below_0p5_cells": accepted_cells,
            "timing_by_cell_s": timing_by_cell,
            "shared_ready_two_side_at_or_below_0p5": shared_ready_pass,
            "any_shared_ready_pass": any(shared_ready_pass.values()),
        },
        "input_stability_errors": input_stability_errors,
        "formal_claims": {
            "physics_replay_exact": False, "source_closure_exact": False,
            "mjcf_closure_exact": False, "screening_evidence_only": True,
            "strict_global_minimum_proven": False,
        },
        "runtime_authority": context["activation"]["runtime_authority"],
        "automatic_retry": False,
        "reviewed_child_timeout_s": CHILD_TIMEOUT_S,
        "trainer_or_robot_signals": [],
    }
    _write_exclusive(root_path / "stage2_summary.json", _canonical_json(summary))
    _require(execution_complete,
             "one or more Stage2 cells failed; evidence preserved and no retry attempted")
    return summary


def run_stage2(*, activation_path: Path | str, queue_path: Path | str,
               root: Path | str, execute: bool = False, confirm: str | None = None,
               runner_source: Path | str | None = None,
               command_runner: CommandRunner = _default_command_runner) -> dict[str, Any]:
    """Run once and preserve an honest terminal marker after namespace creation."""
    ownership = {"created": False}
    try:
        return _run_stage2_impl(
            activation_path=activation_path, queue_path=queue_path, root=root,
            execute=execute, confirm=confirm, runner_source=runner_source,
            command_runner=command_runner, _ownership=ownership,
        )
    except Exception as exc:
        root_path = _absolute(root)
        summary_path = root_path / "stage2_summary.json"
        terminal_path = root_path / "stage2_unexpected_terminal_failure.json"
        if (ownership["created"] and root_path.is_dir()
                and not summary_path.exists() and not terminal_path.exists()):
            terminal = {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "ready_to_strike_join_ladder_stage2_terminal_failure",
                "status": "stage2_unexpected_terminal_failure_no_retry",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "automatic_retry": False,
                "reviewed_child_timeout_s": CHILD_TIMEOUT_S,
                "trainer_or_robot_signals": [],
            }
            try:
                _write_exclusive(terminal_path, _canonical_json(terminal))
            except Exception:
                pass
        if isinstance(exc, Stage2Error):
            raise
        raise Stage2Error(
            f"unexpected Stage2 failure after evidence preservation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="Absolute, nonexistent no-clobber Stage2 result namespace")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_stage2(
            activation_path=args.activation, queue_path=args.queue, root=args.root,
            execute=args.execute, confirm=args.confirm, runner_source=Path(__file__),
        )
    except Stage2Error as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
