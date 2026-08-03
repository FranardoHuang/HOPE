#!/usr/bin/env python3
"""Run a deterministic exact-Pod qdes repair sweep for A211 physical ready.

The sweep never changes the physical birth, teacher, plant, table, controller,
delay contract, or hard terminations.  It derives diagnostic-only candidate
artifacts outside the checkout, changes only the waist-roll hold qdes/action,
short-screens each candidate, then runs every short PASS for an exact 200
policy / 800 physics steps.  A full PASS is selected by the preregistered
lexicographic rule documented in ``SELECTION_RULE`` below.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROBE_FILE = SCRIPT_DIR / "check_table_obstacle_scene.py"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
ARTIFACT_KIND = "agibot_a3_action_dynamic_ready_candidate_v2"
RECEIPT_KIND = "isaac_action_ball_nominal_hold_v1"
RESULT_KIND = "action_ball_a211_physical_ready_qdes_sweep_v1"
JOINT_NAME = "waist_roll_joint"
POLICY_DT_S = 0.02
CONTROL_DECIMATION = 4
SHORT_POLICY_STEPS = 80
FULL_POLICY_STEPS = 200
OFFSETS_RAD = (0.0, 0.04, 0.08, 0.12, 0.16, 0.20)
HARD_TERMINATIONS = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)
SELECTION_RULE = (
    "among exact 200/800 PASS candidates: maximize final minimum actual-hard "
    "gap; then minimize maximum root tilt; then maximize minimum root z; then "
    "minimize maximum initial PD effort ratio; then lexical candidate id"
)
WBT_SOURCE_RELATIVE = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking"
)


class SweepError(RuntimeError):
    """The sweep source, candidate, live evidence, or publication was invalid."""


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
        raise SweepError("value is not finite canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, args: Sequence[str], *, binary: bool = False):
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SweepError("cannot inspect %s: %s" % (name, exc)) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SweepError("%s must be a regular non-symlink file" % name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SweepError("%s is not strict JSON" % name) from exc
    if type(value) is not dict:
        raise SweepError("%s must be one JSON object" % name)
    canonical_bytes(value)
    return value


def _verify_seal(document: Mapping[str, Any], *, name: str) -> str:
    seal = document.get("content_sha256")
    if type(seal) is not str or SHA256_RE.fullmatch(seal) is None:
        raise SweepError("%s content SHA-256 is malformed" % name)
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    if canonical_sha256(unsigned) != seal:
        raise SweepError("%s content seal differs" % name)
    return seal


def _finite_vector(value: object, count: int, *, name: str) -> list[float]:
    if type(value) is not list or len(value) != count:
        raise SweepError("%s must contain %d values" % (name, count))
    result = []
    for item in value:
        if type(item) not in (int, float) or type(item) is bool:
            raise SweepError("%s must be numeric" % name)
        number = float(item)
        if not math.isfinite(number):
            raise SweepError("%s must be finite" % name)
        result.append(number)
    return result


def _write_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise SweepError("no-clobber output already exists: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    if path.read_bytes() != payload:
        raise SweepError("durable readback differs: %s" % path)
    return hashlib.sha256(payload).hexdigest()


def verify_exact_source(root: Path, source_commit: str) -> None:
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise SweepError("source commit must be one full lowercase Git commit")
    head = _git(root, ("rev-parse", "HEAD"))
    dirty = _git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    if head.returncode or head.stdout.strip() != source_commit:
        raise SweepError("source commit is not checkout HEAD")
    if dirty.returncode or dirty.stdout.strip():
        raise SweepError("exact Pod sweep requires a clean checkout")
    for path in (Path(__file__).resolve(), PROBE_FILE):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise SweepError("sweep source escaped repository root") from exc
        committed = _git(root, ("show", source_commit + ":" + relative), binary=True)
        if committed.returncode or committed.stdout != path.read_bytes():
            raise SweepError("source differs from exact commit: %s" % relative)


def load_base_artifact(
    root: Path, relative: str, expected_sha256: str, source_commit: str
) -> tuple[Path, dict[str, Any]]:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise SweepError("base artifact path must be repository-relative")
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise SweepError("base artifact SHA-256 is malformed")
    path = root / relative
    artifact = _strict_json(path, name="base dynamic-ready artifact")
    if sha256_file(path) != expected_sha256:
        raise SweepError("base artifact file SHA-256 differs")
    committed = _git(root, ("show", source_commit + ":" + relative), binary=True)
    if committed.returncode or committed.stdout != path.read_bytes():
        raise SweepError("base artifact differs from exact source commit")
    _verify_seal(artifact, name="base dynamic-ready artifact")
    try:
        names = artifact["robot"]["joint_names"]
        physical = artifact["physical_ready"]
        runtime = artifact["runtime_plant"]
        hold = artifact["hold_candidate"]
    except (KeyError, TypeError) as exc:
        raise SweepError("base artifact core fields are missing") from exc
    if (
        artifact.get("schema_version") != 2
        or artifact.get("kind") != ARTIFACT_KIND
        or type(names) is not list
        or len(names) != 31
        or names.count(JOINT_NAME) != 1
        or artifact.get("authorization", {}).get("training_authorized") is not False
    ):
        raise SweepError("base artifact is not the unauthorized exact A3 candidate")
    _finite_vector(physical.get("joint_pos_rad"), 31, name="physical ready q")
    if _finite_vector(
        physical.get("joint_vel_radps"), 31, name="physical ready qdot"
    ) != [0.0] * 31:
        raise SweepError("physical-ready birth must have zero joint velocity")
    for key in (
        "default_joint_pos_rad",
        "action_scale_rad",
        "joint_stiffness",
        "joint_effort_limits",
    ):
        _finite_vector(runtime.get(key), 31, name="runtime " + key)
    _finite_vector(hold.get("hold_qdes_joint_pos_rad"), 31, name="base hold qdes")
    return path, artifact


def derive_candidate(
    base: Mapping[str, Any], offset_rad: float
) -> tuple[dict[str, Any], dict[str, float | str]]:
    candidate = copy.deepcopy(base)
    names = candidate["robot"]["joint_names"]
    index = names.index(JOINT_NAME)
    runtime = candidate["runtime_plant"]
    physical = candidate["physical_ready"]
    hold = candidate["hold_candidate"]
    qdes = _finite_vector(hold["hold_qdes_joint_pos_rad"], 31, name="hold qdes")
    default = _finite_vector(runtime["default_joint_pos_rad"], 31, name="default q")
    scale = _finite_vector(runtime["action_scale_rad"], 31, name="action scale")
    stiffness = _finite_vector(runtime["joint_stiffness"], 31, name="stiffness")
    effort = _finite_vector(runtime["joint_effort_limits"], 31, name="effort")
    ready = _finite_vector(physical["joint_pos_rad"], 31, name="ready q")
    limits = runtime.get("qdes_joint_pos_limits")
    inset = runtime.get("finite_projection_soft_envelope_inset_fraction")
    if type(limits) is not list or len(limits) != 31:
        raise SweepError("qdes limits must be [31,2]")
    if type(inset) not in (int, float) or type(inset) is bool or not math.isfinite(float(inset)):
        raise SweepError("qdes soft inset is invalid")
    qdes[index] += offset_rad
    normalized = []
    for joint_index, (target, base_q, action_scale, pair) in enumerate(
        zip(qdes, default, scale, limits)
    ):
        bounds = _finite_vector(pair, 2, name="qdes limit")
        lower = bounds[0] + float(inset) * (bounds[1] - bounds[0])
        upper = bounds[1] - float(inset) * (bounds[1] - bounds[0])
        if action_scale <= 0.0 or not lower < target < upper:
            raise SweepError("candidate qdes leaves soft envelope at joint %d" % joint_index)
        normalized.append((target - base_q) / action_scale)
    ratios = []
    for kp, target, birth, limit in zip(stiffness, qdes, ready, effort):
        if limit <= 0.0:
            raise SweepError("runtime effort limit must be positive")
        ratios.append(abs(kp * (target - birth)) / limit)
    hold["hold_qdes_joint_pos_rad"] = qdes
    hold["normalized_actor_action"] = normalized
    hold["hold_qdes_mode"] = "a211_exact_pod_waist_roll_sweep_candidate"
    hold["selected_hold_authority"] = {
        "inherited_hold_claim": False,
        "semantics": "diagnostic_only_exact_pod_sweep_pending_live_receipt",
        "source_physical_birth_seed_sha256": None,
    }
    hold["semantics"] = (
        "base artifact physical birth and plant unchanged; only waist_roll_joint "
        "qdes/action offset is swept; exact live Isaac receipt is sole authority"
    )
    hold["solver_report_role"] = "stale_static_seed_not_hold_authority"
    unsigned = dict(candidate)
    unsigned.pop("content_sha256")
    candidate["content_sha256"] = canonical_sha256(unsigned)
    candidate_id = "waist_roll_%+0.2f" % offset_rad
    return candidate, {
        "candidate_id": candidate_id,
        "waist_roll_offset_rad": offset_rad,
        "waist_roll_qdes_rad": qdes[index],
        "maximum_initial_pd_effort_ratio": max(ratios),
    }


def _command(
    *, python: str, device: str, artifact: Path, artifact_sha: str,
    receipt: Path, screenshot_dir: Path, policy_steps: int,
) -> list[str]:
    return [
        python,
        str(PROBE_FILE),
        "--task", "HOPE-PingPong-ActionBall-AgibotA3-v0",
        "--num-envs", "1",
        "--device", device,
        "--table-obstacle", "on",
        "--nominal-hold", str(artifact),
        "--nominal-hold-sha256", artifact_sha,
        "--nominal-hold-receipt-out", str(receipt),
        "--duration-s", format(policy_steps * POLICY_DT_S, ".17g"),
        "--screenshot-dir", str(screenshot_dir),
    ]


def _probe_child_env(root: Path) -> dict[str, str]:
    """Bind the live probe to this exact checkout's Python package source."""

    source = (root / WBT_SOURCE_RELATIVE).resolve()
    if not source.is_dir() or not (source / "whole_body_tracking").is_dir():
        raise SweepError(
            "whole_body_tracking source package is missing from exact checkout"
        )
    child_env = dict(os.environ)
    inherited = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        str(source)
        if not inherited
        else str(source) + os.pathsep + inherited
    )
    return child_env


def validate_receipt(
    receipt: Mapping[str, Any], *, artifact_sha: str,
    artifact_content_sha: str, policy_steps: int,
) -> bool:
    _verify_seal(receipt, name="live nominal-hold receipt")
    joint = receipt.get("joint_safety_telemetry")
    binding = receipt.get("artifact")
    passed = receipt.get("verdict") == "PASS"
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != RECEIPT_KIND
        or receipt.get("verdict") not in ("PASS", "FAIL")
        or type(binding) is not dict
        or binding.get("sha256") != artifact_sha
        or binding.get("content_sha256") != artifact_content_sha
        or receipt.get("candidate_physical_birth_written") is not True
        or receipt.get("candidate_hold_qdes_and_delay_history_installed") is not True
        or receipt.get("teacher_reference_unchanged") is not True
        or receipt.get("teacher_physical_birth_separated") is not True
        or receipt.get("plant_contract_match") is not True
        or type(receipt.get("active_terminations")) is not list
        or any(name not in receipt["active_terminations"] for name in HARD_TERMINATIONS)
        or type(receipt.get("completed_policy_steps")) is not int
        or receipt.get("completed_policy_steps") < 0
        or receipt.get("completed_policy_steps") > policy_steps
        or receipt.get("completed_physics_steps")
        != receipt.get("completed_policy_steps") * CONTROL_DECIMATION
        or type(joint) is not dict
        or joint.get("schema_version") != 1
        or joint.get("complete") is not True
    ):
        raise SweepError("live receipt structural binding differs")
    if passed and (
        receipt.get("completed_policy_steps") != policy_steps
        or receipt.get("completed_physics_steps")
        != policy_steps * CONTROL_DECIMATION
        or receipt.get("terminal_reasons") != []
        or receipt.get("generic_terminated") is not False
        or receipt.get("generic_truncated") is not False
        or joint.get("current_actual_hard_edge_joint_count") != 0
        or joint.get("substep_actual_hard_edge_joint_count") != 0
        or type(joint.get("final_minimum_hard_gap_rad")) not in (int, float)
        or type(joint.get("final_minimum_hard_gap_rad")) is bool
        or not math.isfinite(float(joint["final_minimum_hard_gap_rad"]))
        or float(joint["final_minimum_hard_gap_rad"]) <= 0.0
    ):
        raise SweepError("PASS receipt lacks the requested exact hard-safe horizon")
    return passed


def _consume_probe_result(
    completed: subprocess.CompletedProcess,
    *,
    receipt_path: Path,
    stage: str,
    candidate_id: str,
    artifact_sha: str,
    artifact_content_sha: str,
    policy_steps: int,
) -> tuple[dict[str, Any], bool]:
    """Consume one probe receipt, distinguishing candidate FAIL from infra failure.

    ``check_table_obstacle_scene.py`` deliberately exits 2 after publishing a
    structurally valid nominal-hold FAIL receipt.  That is evidence about this
    candidate, not a reason to abandon later offsets.  Missing/malformed
    receipts and process/receipt verdict disagreement remain fail-loud because
    they cannot be interpreted as a candidate outcome.
    """

    if not receipt_path.is_file():
        raise SweepError(
            "%s live probe produced no receipt for %s (exit %d)"
            % (stage, candidate_id, completed.returncode)
        )
    receipt = _strict_json(
        receipt_path, name="%s live receipt" % stage
    )
    passed = validate_receipt(
        receipt,
        artifact_sha=artifact_sha,
        artifact_content_sha=artifact_content_sha,
        policy_steps=policy_steps,
    )
    expected_returncode = 0 if passed else 2
    if completed.returncode != expected_returncode:
        raise SweepError(
            "%s live probe process/receipt verdict differs for %s: "
            "exit=%d verdict=%s"
            % (
                stage,
                candidate_id,
                completed.returncode,
                receipt["verdict"],
            )
        )
    return receipt, passed


def _rank(row: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    receipt = row["full_receipt"]
    joint = receipt["joint_safety_telemetry"]
    return (
        -float(joint["final_minimum_hard_gap_rad"]),
        float(receipt["maximum_root_tilt_rad"]),
        -float(receipt["minimum_root_z_m"]),
        float(row["maximum_initial_pd_effort_ratio"]),
        str(row["candidate_id"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve(strict=True)
    source_commit = str(args.source_commit)
    verify_exact_source(root, source_commit)
    child_env = _probe_child_env(root)
    base_path, base = load_base_artifact(
        root, args.base_artifact_path,
        args.expected_base_artifact_sha256, source_commit,
    )
    work_dir = Path(args.work_dir).expanduser().absolute()
    try:
        work_dir.relative_to(root)
    except ValueError:
        pass
    else:
        raise SweepError("work directory must stay outside the exact checkout")
    if work_dir.exists() or work_dir.is_symlink():
        raise SweepError("fresh no-clobber work directory already exists")
    os.mkdir(work_dir, 0o755)
    rows = []
    for offset in OFFSETS_RAD:
        candidate, metadata = derive_candidate(base, offset)
        candidate_dir = work_dir / str(metadata["candidate_id"])
        candidate_dir.mkdir()
        artifact_path = candidate_dir / "candidate.dynamic_ready.v2.json"
        artifact_sha = _write_new(artifact_path, candidate)
        row: dict[str, Any] = {
            **metadata,
            "artifact": {
                "path": str(artifact_path),
                "sha256": artifact_sha,
                "content_sha256": candidate["content_sha256"],
            },
        }
        short_path = candidate_dir / "short.receipt.v1.json"
        short_command = _command(
            python=args.python, device=args.device, artifact=artifact_path,
            artifact_sha=artifact_sha, receipt=short_path,
            screenshot_dir=candidate_dir / "short_screenshots",
            policy_steps=SHORT_POLICY_STEPS,
        )
        completed = subprocess.run(
            short_command, cwd=str(root), check=False, env=child_env
        )
        short, short_pass = _consume_probe_result(
            completed,
            receipt_path=short_path,
            stage="short",
            candidate_id=str(metadata["candidate_id"]),
            artifact_sha=artifact_sha,
            artifact_content_sha=candidate["content_sha256"],
            policy_steps=SHORT_POLICY_STEPS,
        )
        row["short_receipt"] = short
        row["short_pass"] = short_pass
        if short_pass:
            full_path = candidate_dir / "full.receipt.v1.json"
            full_command = _command(
                python=args.python, device=args.device, artifact=artifact_path,
                artifact_sha=artifact_sha, receipt=full_path,
                screenshot_dir=candidate_dir / "full_screenshots",
                policy_steps=FULL_POLICY_STEPS,
            )
            completed = subprocess.run(
                full_command, cwd=str(root), check=False, env=child_env
            )
            full, full_pass = _consume_probe_result(
                completed,
                receipt_path=full_path,
                stage="full",
                candidate_id=str(metadata["candidate_id"]),
                artifact_sha=artifact_sha,
                artifact_content_sha=candidate["content_sha256"],
                policy_steps=FULL_POLICY_STEPS,
            )
            row["full_pass"] = full_pass
            row["full_receipt"] = full
        else:
            row["full_pass"] = False
            row["full_receipt"] = None
        rows.append(row)
    verify_exact_source(root, source_commit)
    eligible = [row for row in rows if row["full_pass"]]
    selected = min(eligible, key=_rank) if eligible else None
    unsigned = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "diagnostic_unauthorized": True,
        "verdict": "PASS" if selected is not None else "FAIL",
        "source_commit": source_commit,
        "base_artifact": {
            "path": str(base_path),
            "sha256": args.expected_base_artifact_sha256,
            "content_sha256": base["content_sha256"],
        },
        "immutable_contract": {
            "physical_ready_unchanged": True,
            "teacher_reference_unchanged": True,
            "runtime_plant_unchanged": True,
            "table_obstacle": "on",
            "controller_and_action_delay_from_base_artifact": True,
            "hard_terminations": list(HARD_TERMINATIONS),
            "control_decimation": CONTROL_DECIMATION,
        },
        "search": {
            "joint_name": JOINT_NAME,
            "offsets_rad": list(OFFSETS_RAD),
            "short_policy_steps": SHORT_POLICY_STEPS,
            "full_policy_steps": FULL_POLICY_STEPS,
            "selection_rule": SELECTION_RULE,
        },
        "candidates": rows,
        "selected_candidate_id": (
            selected["candidate_id"] if selected is not None else None
        ),
        "selected_artifact": (
            selected["artifact"] if selected is not None else None
        ),
        "selected_full_receipt_content_sha256": (
            selected["full_receipt"]["content_sha256"]
            if selected is not None else None
        ),
    }
    result = {**unsigned, "content_sha256": canonical_sha256(unsigned)}
    result_path = work_dir / "a211_physical_ready_qdes_sweep.result.v1.json"
    _write_new(result_path, result)
    return {
        "verdict": result["verdict"],
        "diagnostic_unauthorized": True,
        "selected_candidate_id": result["selected_candidate_id"],
        "selected_artifact": result["selected_artifact"],
        "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--expected-base-artifact-sha256", required=True)
    parser.add_argument("--python", required=True, help="exact Pod Isaac Python")
    parser.add_argument("--device", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
    except (SweepError, OSError, ValueError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
