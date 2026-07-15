#!/usr/bin/env python3
"""Mint one immutable post-swing teacher receipt from a source-bound inference capture.

MotionCommand owns an O_EXCL capture namespace and publishes raw state bytes from its reviewed
natural-wrap source path.  This consumer does not treat a producer-supplied callback label as
proof.  It verifies the exact producer source bytes and exclusive claim, then binds the actual
checkpoint through PyTorch's restricted weights-only loader, adjacent schema-3 hard contract and
launch claim before publishing the only receipt accepted by the trainer.
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
import subprocess
import sys
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WBT = REPO_ROOT / "hope_training/whole_body_tracking"
CONTRACT_PATH = (
    WBT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/post_swing_teacher.py"
)
SPEC = importlib.util.spec_from_file_location("post_swing_teacher_attestor_contract", CONTRACT_PATH)
teacher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = teacher
SPEC.loader.exec_module(teacher)


class AttestationError(RuntimeError):
    """The capture cannot produce an exact teacher receipt."""


RETRY_AUTHORIZATION_KIND = "hope_post_swing_teacher_attestor_retry_authorization"


def _read(path: Path, label: str) -> bytes:
    try:
        return teacher._read_regular_file_once(path, label)
    except teacher.PostSwingTeacherError as exc:
        raise AttestationError(str(exc)) from exc


def _json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = teacher._strict_json_bytes(raw, label)
    except teacher.PostSwingTeacherError as exc:
        raise AttestationError(str(exc)) from exc
    if not isinstance(value, dict):
        raise AttestationError(f"{label} must be a JSON object")
    return value


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise AttestationError(f"{label} keys differ from the frozen schema")
    return value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_content(value: Mapping[str, Any]) -> bytes:
    """Return the canonical JSON payload bytes used by embedded digests.

    Queue claim ``content_sha256`` values are computed over the compact JSON
    value itself.  The newline used when a complete JSON document is written
    to disk is framing, not part of that embedded content digest.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_document(value: Mapping[str, Any]) -> bytes:
    """Return one canonical JSON document with exactly one trailing newline."""

    return _canonical_content(value) + b"\n"


def _git_state(checkout: Path, expected_commit: str, label: str) -> dict[str, Any]:
    if not checkout.is_absolute() or checkout.is_symlink() or not checkout.is_dir():
        raise AttestationError(f"{label} must be an absolute non-symlink directory")
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != expected_commit or dirty:
        raise AttestationError(
            f"{label} is not clean exact {expected_commit}: head={head} dirty={bool(dirty)}"
        )
    return {"commit": head, "clean": True}


def _current_git_state(checkout: Path, label: str) -> dict[str, Any]:
    """Bind one clean checkout to the commit it contains at attestation time."""

    if not checkout.is_absolute() or checkout.is_symlink() or not checkout.is_dir():
        raise AttestationError(f"{label} must be an absolute non-symlink directory")
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if teacher._COMMIT.fullmatch(head) is None:
        raise AttestationError(f"{label} HEAD is not one exact commit")
    return _git_state(checkout, head, label)


def _plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise AttestationError(f"{label} must be a JSON integer (bool/coercion forbidden)")
    return value


def _positive_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise AttestationError(f"{label} must be a finite positive JSON float")
    return value


def _checkpoint(raw: bytes, hard_sha: str, claim_sha: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            names = [row.filename for row in archive.infolist()]
        if len(names) != len(set(names)):
            raise AttestationError("checkpoint archive contains duplicate ZIP members")
    except zipfile.BadZipFile:
        # Older torch checkpoints may be pickle streams.  Their immutable byte buffer is still
        # authoritative; torch's restricted consumer below remains the semantic parser.
        pass
    try:
        import torch

        # ``weights_only=False`` executes arbitrary pickle reducers and is forbidden for this
        # attestation boundary.  Older/non-tensor checkpoints that cannot pass the restricted
        # unpickler fail closed instead of being allow-listed dynamically.
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AttestationError(f"cannot load actual checkpoint bytes: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AttestationError("checkpoint root must be a mapping")
    infos = value.get("infos")
    if not isinstance(infos, Mapping):
        raise AttestationError("checkpoint infos missing")
    if _plain_int(infos.get("training_contract_schema_version"), "checkpoint schema") != 3:
        raise AttestationError("checkpoint is not schema 3")
    if infos.get("training_contract_sha256") != hard_sha:
        raise AttestationError("checkpoint hard-contract SHA differs from adjacent bytes")
    if _plain_int(infos.get("training_contract_lineage_exact"), "checkpoint lineage") != 1:
        raise AttestationError("checkpoint fresh/exact lineage must equal integer 1")
    if infos.get("training_launch_claim_sha256") != claim_sha:
        raise AttestationError("checkpoint launch-claim SHA differs from actual claim content")

    floating = 0
    nonfinite = 0

    def visit(item: Any) -> None:
        nonlocal floating, nonfinite
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, torch.Tensor) and item.is_floating_point():
            floating += 1
            nonfinite += int((~torch.isfinite(item)).sum().item())

    visit(value)
    if floating <= 0 or nonfinite != 0:
        raise AttestationError("checkpoint tensor payload is empty or non-finite")
    return {
        "sha256": _sha(raw),
        "training_contract_schema_version": 3,
        "training_contract_sha256": hard_sha,
        "training_contract_lineage_exact": True,
        "training_launch_claim_sha256": claim_sha,
    }


def _claim(raw: bytes) -> tuple[dict[str, Any], str, Path, str]:
    envelope = _json(raw, "training launch claim")
    if _plain_int(envelope.get("schema_version"), "claim schema") != 2:
        raise AttestationError("training launch claim schema must equal 2")
    content = envelope.get("content")
    if not isinstance(content, dict):
        raise AttestationError("training launch claim content missing")
    claim_sha = envelope.get("content_sha256")
    if type(claim_sha) is not str or claim_sha != _sha(_canonical_content(content)):
        raise AttestationError("training launch claim canonical digest mismatch")
    source = content.get("source")
    if not isinstance(source, dict):
        raise AttestationError("training launch claim source missing")
    commit = source.get("commit")
    checkout = source.get("checkout")
    if (
        type(commit) is not str
        or teacher._COMMIT.fullmatch(commit) is None
        or type(checkout) is not str
        or not Path(checkout).is_absolute()
    ):
        raise AttestationError("training launch claim source binding is malformed")
    return content, claim_sha, Path(checkout), commit


def _retry_authorization(
    raw: bytes,
    expected_file_sha256: str,
    *,
    capture_directory: Path,
    output_receipt: Path,
    capture_claim_sha256: str,
    states_sha256: str,
    capture_result_sha256: str,
    state_count: int,
    checkpoint_sha256: str,
    hard_contract_sha256: str,
    launch_claim_content_sha256: str,
    capture_source: Mapping[str, Any],
    producer_source_sha256: str,
    attestor_source: Mapping[str, Any],
    attestor_source_sha256: str,
) -> dict[str, Any]:
    if (
        type(expected_file_sha256) is not str
        or teacher._SHA256.fullmatch(expected_file_sha256) is None
        or _sha(raw) != expected_file_sha256
    ):
        raise AttestationError("retry authorization file SHA-256 mismatch")
    value = _keys(
        _json(raw, "attestor retry authorization"),
        {
            "schema_version", "artifact_kind", "authorization_id", "v3_plan",
            "capture", "teacher", "capture_source", "attestor_source", "decision",
        },
        "attestor retry authorization",
    )
    if (
        _plain_int(value["schema_version"], "retry authorization schema") != 1
        or value["artifact_kind"] != RETRY_AUTHORIZATION_KIND
        or type(value["authorization_id"]) is not str
        or not value["authorization_id"]
    ):
        raise AttestationError("attestor retry authorization header is malformed")
    plan = _keys(value["v3_plan"], {"plan_id", "file_sha256"}, "retry v3 plan")
    capture = _keys(
        value["capture"],
        {
            "output_directory", "output_receipt", "capture_claim_sha256",
            "states_sha256", "result_sha256", "state_count",
        },
        "retry capture",
    )
    teacher_row = _keys(
        value["teacher"],
        {"checkpoint_sha256", "hard_contract_sha256", "launch_claim_content_sha256"},
        "retry teacher",
    )
    capture_source_row = _keys(
        value["capture_source"],
        {"commit", "producer_source_sha256"},
        "retry capture source",
    )
    attestor_source_row = _keys(
        value["attestor_source"],
        {"commit", "attestor_source_sha256"},
        "retry attestor source",
    )
    decision = _keys(
        value["decision"],
        {
            "capture_retry_authorized", "attestor_attempt2_authorized",
            "first_reset_probe_authorized", "scientific_training_authorized",
        },
        "retry decision",
    )
    for label, digest in (
        ("retry plan file", plan["file_sha256"]),
        ("retry capture claim", capture["capture_claim_sha256"]),
        ("retry states", capture["states_sha256"]),
        ("retry result", capture["result_sha256"]),
        ("retry checkpoint", teacher_row["checkpoint_sha256"]),
        ("retry hard contract", teacher_row["hard_contract_sha256"]),
        ("retry launch claim", teacher_row["launch_claim_content_sha256"]),
        ("retry producer", capture_source_row["producer_source_sha256"]),
        ("retry attestor", attestor_source_row["attestor_source_sha256"]),
    ):
        if type(digest) is not str or teacher._SHA256.fullmatch(digest) is None:
            raise AttestationError(f"{label} SHA-256 is malformed")
    for label, commit in (
        ("retry capture source", capture_source_row["commit"]),
        ("retry attestor source", attestor_source_row["commit"]),
    ):
        if type(commit) is not str or teacher._COMMIT.fullmatch(commit) is None:
            raise AttestationError(f"{label} commit is malformed")
    expected_capture_source = {
        "commit": capture_source["commit"],
        "producer_source_sha256": producer_source_sha256,
    }
    expected_attestor_source = {
        "commit": attestor_source["commit"],
        "attestor_source_sha256": attestor_source_sha256,
    }
    if (
        plan["plan_id"] != capture_directory.name
        or capture
        != {
            "output_directory": str(capture_directory),
            "output_receipt": str(output_receipt),
            "capture_claim_sha256": capture_claim_sha256,
            "states_sha256": states_sha256,
            "result_sha256": capture_result_sha256,
            "state_count": state_count,
        }
        or teacher_row
        != {
            "checkpoint_sha256": checkpoint_sha256,
            "hard_contract_sha256": hard_contract_sha256,
            "launch_claim_content_sha256": launch_claim_content_sha256,
        }
        or capture_source_row != expected_capture_source
        or attestor_source_row != expected_attestor_source
        or decision
        != {
            "capture_retry_authorized": False,
            "attestor_attempt2_authorized": True,
            "first_reset_probe_authorized": False,
            "scientific_training_authorized": False,
        }
    ):
        raise AttestationError("retry authorization is rebound from the immutable v3 attempt")
    return {
        "authorization_id": value["authorization_id"],
        "file_sha256": expected_file_sha256,
        "v3_plan_file_sha256": plan["file_sha256"],
    }


def _capture_result(
    raw: bytes, result_path: Path
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    result = _keys(
        _json(raw, "natural-wrap capture result"),
        {
            "schema_version",
            "artifact_kind",
            "capture_contract",
            "evidence",
            "motion_clips",
            "states",
        },
        "natural-wrap capture result",
    )
    if _plain_int(result["schema_version"], "capture schema") != 2:
        raise AttestationError("unsupported capture result schema")
    if result["artifact_kind"] != teacher.CAPTURE_RESULT_KIND:
        raise AttestationError("capture result kind mismatch")
    if result["capture_contract"] != teacher.CAPTURE_CONTRACT:
        raise AttestationError("capture result declares different natural-wrap semantics")
    evidence = _keys(
        result["evidence"],
        {
            "producer_source_sha256",
            "runtime_hard_contract_sha256",
            "exclusive_claim_sha256",
            "exclusive_claim_relative_path",
            "no_clobber",
        },
        "capture evidence",
    )
    if (
        evidence["exclusive_claim_relative_path"] != teacher.CAPTURE_CLAIM_NAME
        or evidence["no_clobber"] is not True
    ):
        raise AttestationError("capture result lacks the fixed exclusive no-clobber claim")
    states = result.get("states")
    if not isinstance(states, dict) or states.get("relative_path") != teacher.CAPTURE_STATE_NAME:
        raise AttestationError("capture state path is not the fixed source-owned output")
    state_path = result_path.parent / teacher.CAPTURE_STATE_NAME
    state_raw = _read(state_path, "natural-wrap state payload")
    if states.get("sha256") != _sha(state_raw):
        raise AttestationError("capture state payload SHA mismatch")
    claim_path = result_path.parent / teacher.CAPTURE_CLAIM_NAME
    claim_raw = _read(claim_path, "exclusive natural-wrap capture claim")
    if evidence["exclusive_claim_sha256"] != _sha(claim_raw):
        raise AttestationError("capture result does not bind the actual exclusive claim bytes")
    claim = _keys(
        _json(claim_raw, "exclusive natural-wrap capture claim"),
        {
            "schema_version",
            "artifact_kind",
            "producer_source_sha256",
            "runtime_hard_contract_sha256",
            "target_count",
            "motion_clips",
            "joint_names",
            "exclusive_create",
        },
        "exclusive natural-wrap capture claim",
    )
    if (
        _plain_int(claim["schema_version"], "capture claim schema") != 1
        or claim["artifact_kind"] != teacher.CAPTURE_CLAIM_KIND
        or claim["exclusive_create"] is not True
    ):
        raise AttestationError("exclusive natural-wrap capture claim is malformed")
    return result, state_raw, claim, claim_raw


def _load_arrays(raw: bytes, result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            names = [row.filename for row in archive.infolist()]
        expected = {f"{key}.npy" for key in teacher._NPZ_KEYS}
        if len(names) != len(set(names)) or set(names) != expected:
            raise AttestationError("capture NPZ has duplicate or unexpected ZIP keys")
        with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in teacher._NPZ_KEYS}
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        if isinstance(exc, AttestationError):
            raise
        raise AttestationError(f"cannot parse capture NPZ: {exc}") from exc
    states = result["states"]
    count = _plain_int(states.get("count"), "capture state count")
    joints = states.get("joint_names")
    expected_shapes = {
        "root_state_origin_relative": (count, 13),
        "joint_pos": (count, len(joints) if isinstance(joints, list) else -1),
        "joint_vel": (count, len(joints) if isinstance(joints, list) else -1),
    }
    for key, value in arrays.items():
        if value.dtype != np.float32 or value.shape != expected_shapes[key] or not np.isfinite(value).all():
            raise AttestationError(f"capture array {key} has wrong dtype/shape/finiteness")
    return arrays


def attest(args: argparse.Namespace) -> dict[str, Any]:
    result_path = args.capture_result.resolve(strict=False)
    output = args.output_receipt.resolve(strict=False)
    if output.parent != result_path.parent:
        raise AttestationError("teacher receipt must stay beside its no-clobber capture payload")
    if os.path.lexists(output):
        raise AttestationError("teacher receipt already exists; overwrite is forbidden")
    result_raw = _read(result_path, "natural-wrap capture result")
    result, state_raw, capture_claim, capture_claim_raw = _capture_result(
        result_raw, result_path
    )
    arrays = _load_arrays(state_raw, result)

    claim_raw = _read(args.launch_claim, "training launch claim")
    _claim_content, claim_sha, checkpoint_source_checkout, checkpoint_source_commit = _claim(claim_raw)
    _git_state(checkpoint_source_checkout, checkpoint_source_commit, "checkpoint source checkout")

    checkpoint_path = args.checkpoint.resolve(strict=False)
    hard_path = checkpoint_path.parent / "params/training_contract.json"
    if args.hard_contract.resolve(strict=False) != hard_path:
        raise AttestationError("hard contract must be the checkpoint-adjacent params/training_contract.json")
    hard_raw = _read(hard_path, "adjacent hard contract")
    hard = _json(hard_raw, "adjacent hard contract")
    if _plain_int(hard.get("schema_version"), "hard contract schema") != 3:
        raise AttestationError("hard contract must be schema 3")
    hard_sha = _sha(hard_raw)
    evidence = result["evidence"]
    if (
        evidence["runtime_hard_contract_sha256"] != hard_sha
        or capture_claim["runtime_hard_contract_sha256"] != hard_sha
    ):
        raise AttestationError("capture runtime contract differs from the checkpoint-adjacent contract")
    checkpoint_raw = _read(checkpoint_path, "teacher checkpoint")
    checkpoint_attestation = _checkpoint(checkpoint_raw, hard_sha, claim_sha)

    capture_source = _current_git_state(
        args.capture_source_checkout, "capture producer source checkout"
    )
    attestor_source = _current_git_state(REPO_ROOT, "attestor source checkout")
    producer_path = (
        args.capture_source_checkout
        / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
    )
    producer_sha = _sha(_read(producer_path, "capture producer source"))
    attestor_path = REPO_ROOT / "scripts/attest_post_swing_teacher.py"
    running_attestor_path = Path(__file__).resolve()
    if running_attestor_path != attestor_path:
        raise AttestationError("running attestor is outside its own clean source checkout")
    attestor_sha = _sha(_read(attestor_path, "running teacher attestor source"))
    if (
        producer_sha != evidence["producer_source_sha256"]
        or producer_sha != capture_claim["producer_source_sha256"]
    ):
        raise AttestationError("capture source bytes differ from source-bound capture evidence")

    motion_rows = hard.get("motion_clips")
    capture_rows = result.get("motion_clips")
    if not isinstance(motion_rows, list) or not isinstance(capture_rows, list):
        raise AttestationError("hard contract/capture motion rows missing")
    if len(args.motion) != len(motion_rows) or len(args.motion) != len(capture_rows):
        raise AttestationError("ordered motion path count differs from hard contract/capture")
    if capture_claim["motion_clips"] != capture_rows:
        raise AttestationError("exclusive capture claim motion order differs from result")
    motions = []
    for index, path in enumerate(args.motion):
        motion_sha = _sha(_read(path, f"motion clip {index}"))
        if (
            motion_rows[index].get("index") != index
            or motion_rows[index].get("sha256") != motion_sha
            or capture_rows[index] != {"index": index, "sha256": motion_sha}
        ):
            raise AttestationError(f"motion clip {index} bytes/order differ")
        motions.append({"index": index, "sha256": motion_sha})

    joint_names = hard.get("articulation_joint_names")
    if (
        result["states"].get("joint_names") != joint_names
        or capture_claim["joint_names"] != joint_names
    ):
        raise AttestationError("capture joint names/order differ from hard contract runtime order")
    if _plain_int(capture_claim["target_count"], "capture claim target_count") != _plain_int(
        result["states"].get("count"), "capture state count"
    ):
        raise AttestationError("exclusive capture target count differs from emitted state count")
    joint_limits_raw = hard.get("joint_velocity_limits")
    if (
        not isinstance(joint_limits_raw, list)
        or len(joint_limits_raw) != len(joint_names)
        or any(type(value) not in (int, float) or isinstance(value, bool) for value in joint_limits_raw)
    ):
        raise AttestationError("hard contract joint velocity limits are malformed")
    joint_limits = [float(value) for value in joint_limits_raw]
    if any(not math.isfinite(value) or value <= 0.0 for value in joint_limits):
        raise AttestationError("hard contract joint velocity limits must be finite and positive")
    root_lin = _positive_float(args.root_linear_limit_mps, "root linear limit")
    root_ang = _positive_float(args.root_angular_limit_radps, "root angular limit")
    root = arrays["root_state_origin_relative"]
    joint_vel = arrays["joint_vel"]
    if np.any(np.linalg.norm(root[:, 7:10].astype(np.float64), axis=1) > root_lin):
        raise AttestationError("capture root linear velocity exceeds the preregistered limit")
    if np.any(np.linalg.norm(root[:, 10:13].astype(np.float64), axis=1) > root_ang):
        raise AttestationError("capture root angular velocity exceeds the preregistered limit")
    if np.any(np.abs(joint_vel.astype(np.float64)) > np.asarray(joint_limits)[None, :]):
        raise AttestationError("capture joint velocity exceeds the runtime plant limit")

    retry_authorization_raw = _read(
        args.retry_authorization, "attestor retry authorization"
    )
    retry_authorization = _retry_authorization(
        retry_authorization_raw,
        args.expected_retry_authorization_sha256,
        capture_directory=result_path.parent,
        output_receipt=output,
        capture_claim_sha256=_sha(capture_claim_raw),
        states_sha256=_sha(state_raw),
        capture_result_sha256=_sha(result_raw),
        state_count=_plain_int(result["states"]["count"], "teacher state count"),
        checkpoint_sha256=checkpoint_attestation["sha256"],
        hard_contract_sha256=hard_sha,
        launch_claim_content_sha256=claim_sha,
        capture_source=capture_source,
        producer_source_sha256=producer_sha,
        attestor_source=attestor_source,
        attestor_source_sha256=attestor_sha,
    )

    states = dict(result["states"])
    states["velocity_limits"] = {
        "root_linear_norm_max_mps": root_lin,
        "root_angular_norm_max_radps": root_ang,
        "joint_abs_max_radps": joint_limits,
    }
    receipt = {
        "schema_version": teacher.SCHEMA_VERSION,
        "artifact_kind": teacher.ARTIFACT_KIND,
        "capture_contract": dict(teacher.CAPTURE_CONTRACT),
        "teacher": {
            "source_commit": checkpoint_source_commit,
            "checkpoint_sha256": checkpoint_attestation["sha256"],
            "training_contract_sha256": hard_sha,
            "training_contract_schema_version": 3,
            "fresh_lineage": True,
        },
        "motion_clips": motions,
        "states": states,
        "attestation": {
            "schema_version": 2,
            "artifact_kind": teacher.ATTESTATION_KIND,
            "capture_result_sha256": _sha(result_raw),
            "capture_result_relative_path": result_path.name,
            "capture_claim_sha256": _sha(capture_claim_raw),
            "capture_claim_relative_path": teacher.CAPTURE_CLAIM_NAME,
            "checkpoint": checkpoint_attestation,
            "hard_contract": {"sha256": hard_sha, "schema_version": 3},
            "checkpoint_source": {
                "commit": checkpoint_source_commit,
                "launch_claim_content_sha256": claim_sha,
            },
            "capture_source": {
                **capture_source,
                "producer_source_sha256": producer_sha,
            },
            "attestor_source": {
                **attestor_source,
                "attestor_source_sha256": attestor_sha,
            },
            "retry_authorization": retry_authorization,
        },
    }
    receipt_raw = _json_document(receipt)

    # Validate exact trainer consumption before publishing any new final path.
    with tempfile.TemporaryDirectory(prefix="post-swing-attest-") as temp_dir:
        temp = Path(temp_dir)
        (temp / states["relative_path"]).write_bytes(state_raw)
        (temp / result_path.name).write_bytes(result_raw)
        (temp / teacher.CAPTURE_CLAIM_NAME).write_bytes(capture_claim_raw)
        temp_receipt = temp / output.name
        temp_receipt.write_bytes(receipt_raw)
        teacher.load_post_swing_teacher_states(
            temp_receipt,
            _sha(receipt_raw),
            retry_authorization_path=args.retry_authorization,
            expected_retry_authorization_sha256=(
                args.expected_retry_authorization_sha256
            ),
            expected_motion_sha256=[row["sha256"] for row in motions],
            expected_joint_names=joint_names,
            expected_joint_velocity_limits=joint_limits,
            expected_root_linear_velocity_limit_mps=root_lin,
            expected_root_angular_velocity_limit_radps=root_ang,
            min_fill=_plain_int(states["count"], "teacher state count"),
            buffer_size=_plain_int(states["count"], "teacher state count"),
        )
    teacher._publish_bytes_no_clobber(output, receipt_raw, "teacher receipt")
    return {"receipt": str(output), "sha256": _sha(receipt_raw), "count": states["count"]}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--capture-result", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--hard-contract", type=Path, required=True)
    value.add_argument("--launch-claim", type=Path, required=True)
    value.add_argument("--capture-source-checkout", type=Path, required=True)
    value.add_argument("--motion", type=Path, action="append", required=True)
    value.add_argument("--root-linear-limit-mps", type=float, required=True)
    value.add_argument("--root-angular-limit-radps", type=float, required=True)
    value.add_argument("--retry-authorization", type=Path, required=True)
    value.add_argument("--expected-retry-authorization-sha256", required=True)
    value.add_argument("--output-receipt", type=Path, required=True)
    return value


def main() -> int:
    try:
        result = attest(parser().parse_args())
    except (AttestationError, teacher.PostSwingTeacherError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
