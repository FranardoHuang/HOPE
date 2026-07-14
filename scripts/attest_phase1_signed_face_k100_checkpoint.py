#!/usr/bin/env python3
"""Attest one exact checkpoint against the immutable signed-face K100 inputs.

This is an attestation-only consumer.  It never launches a trainer, exporter,
judge, simulator, deployment process, signal, SSH command, or robot command.
``static-validate`` checks the tracked source contract.  ``plan`` validates an
exact per-checkpoint request without reading runtime-only files.  ``attest``
rechecks every runtime byte and writes one evidence document followed by one
claim in a checkpoint-SHA-derived, no-replace namespace.

The claim is not permission to run a judge and never contains a checkpoint
promotion or stop decision.  A future reviewed runner must consume it exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


MANIFEST_ID = "phase1-signed-face-k100-checkpoint-attestor-20260714-v1"
MANIFEST_RELATIVE_PATH = Path(
    "configs/phase1_signed_face_k100_checkpoint_attestor_20260714.json"
)
OUTPUT_ROOT = Path(
    "/workspace/codexschema/phase1_signed_face_rescue_20260713/"
    "executions/signed_face_k100_v1"
)
CORRECTION_ID = (
    "phase1-signed-face-exam-k100-runtime-receipt-correction-20260714-v1"
)
REQUEST_STATUS = "exact_checkpoint_attestation_requested_judge_not_started"
ACTIVATION_KIND = "phase1_signed_face_exam_k100_paper_activation"
EVIDENCE_KIND = "phase1_signed_face_k100_checkpoint_execution_evidence"
CLAIM_KIND = "phase1_signed_face_k100_checkpoint_execution_claim"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
REQUEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
CHECKPOINT_RE = re.compile(r"^model_([0-9]+)\.pt$")
PATH_META_RE = re.compile(r"[*?\[\]{}]")

EXPECTED_FACE_CONTRACT = {
    "schema_version": 1,
    "achieved_and_target_frame": "mount_plusY_A",
    "external_frame": "physical_striking_face_B",
    "physical_B_to_raw_A": "raw_A=mount_normal_sign_per_clip[clip]*physical_B",
    "clip_order": ["forehand", "backhand"],
    "mount_normal_sign_per_clip": [1.0, -1.0],
    "signed_face_required": True,
    "unsigned_or_oriented_plane_fallback_allowed": False,
    "identity_checked_before_orient_normal": True,
    "physical_B_min_x_strict": 1.0e-6,
    "unit_normal_atol": 2.0e-4,
    "identity_gate": (
        "dot(achieved_raw_A,target_raw_A)>0_and_achieved_physical_B.x>1e-6_and_"
        "target_physical_B.x>1e-6_before_orient_normal"
    ),
}

PAPER_AUTHORIZATION = {
    "paper_materialization_only": True,
    "auto_start": False,
    "trainer_started": False,
    "judge_started": False,
    "l2_training_authorized": False,
    "second_seed_authorized": False,
    "checkpoint_stop_or_promote_authorized": False,
    "formal_score_authorized": False,
    "gate3_authorized": False,
    "deployment_authorized": False,
    "real_robot_authorized": False,
}

ATTESTATION_AUTHORIZATION = {
    "attestation_only": True,
    "auto_start": False,
    "trainer_started": False,
    "judge_started": False,
    "checkpoint_execution_started": False,
    "l2_training_authorized": False,
    "second_seed_authorized": False,
    "checkpoint_stop_or_promote_authorized": False,
    "formal_score_authorized": False,
    "gate3_authorized": False,
    "deployment_authorized": False,
    "real_robot_authorized": False,
}

PLANT_KEYS = (
    "articulation_joint_names",
    "action_joint_ids",
    "joint_names",
    "default_joint_pos",
    "action_scale",
    "joint_stiffness",
    "joint_damping",
    "joint_effort_limits",
    "joint_actuator_types",
    "joint_armature",
    "joint_friction_coefficients",
    "joint_velocity_limits",
    "joint_friction_backend",
    "joint_friction_semantics",
    "joint_friction_units",
    "qdes_joint_pos_limits",
    "action_use_default_offset",
    "qdes_clamp",
    "physics_step_dt_s",
    "policy_step_dt_s",
    "control_decimation",
)

CHECKPOINT_AUDIT_CODE = r"""
import json,sys,torch
obj=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
stack=[obj]; seen=set(); tensors=elements=nonfinite=0
while stack:
 value=stack.pop()
 if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
  tensors+=1; elements+=value.numel(); nonfinite+=int((~torch.isfinite(value)).sum().item())
 elif isinstance(value,dict) and id(value) not in seen:
  seen.add(id(value)); stack.extend(value.values())
 elif isinstance(value,(list,tuple)) and id(value) not in seen:
  seen.add(id(value)); stack.extend(value)
infos=obj.get('infos') if isinstance(obj,dict) else {}; infos=infos if isinstance(infos,dict) else {}
print(json.dumps({'iter':obj.get('iter'),'training_contract_schema_version':infos.get('training_contract_schema_version'),'training_contract_sha256':infos.get('training_contract_sha256'),'training_contract_lineage_exact':infos.get('training_contract_lineage_exact'),'training_launch_claim_sha256':infos.get('training_launch_claim_sha256'),'floating_tensor_count':tensors,'floating_elements':elements,'nonfinite_floating_elements':nonfinite},sort_keys=True,allow_nan=False))
"""

RUNTIME_FINGERPRINT_CODE = r"""
import importlib.metadata,json,platform,sys
names=json.loads(sys.argv[1])
print(json.dumps({'implementation':platform.python_implementation(),'version':sys.version,'executable':sys.executable,'packages':{name:importlib.metadata.version(name) for name in names}},sort_keys=True,allow_nan=False))
"""


class ContractError(RuntimeError):
    """One exact source, request, paper, checkpoint, or runtime fact changed."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not finite canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ContractError(f"non-finite JSON constant: {value}")


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read strict JSON {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} root must be a JSON object")
    require_finite_json(value, label)
    return value


def require_finite_json(value: Any, label: str) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ContractError(f"{label} contains a non-finite JSON number")
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def require_exact(actual: Any, expected: Any, label: str) -> None:
    if not strict_equal(actual, expected):
        raise ContractError(f"{label} changed or has a bool/int/float type confusion")


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys changed: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def require_sha(value: Any, label: str) -> str:
    if type(value) is not str or not SHA_RE.fullmatch(value):
        raise ContractError(f"{label} must be one lowercase SHA-256")
    return value


def require_git(value: Any, label: str) -> str:
    if type(value) is not str or not GIT_RE.fullmatch(value):
        raise ContractError(f"{label} must be one 40-character lowercase git id")
    return value


def require_plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be a plain integer >= {minimum}")
    return value


def require_absolute(value: Any, label: str) -> Path:
    if type(value) is not str:
        raise ContractError(f"{label} must be an absolute path string")
    if not value or "\x00" in value or any(ord(char) < 32 for char in value):
        raise ContractError(f"{label} contains an empty/control path component")
    if PATH_META_RE.search(value):
        raise ContractError(f"{label} must not contain glob or wildcard syntax")
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or value.startswith("//")
        or str(path) != value
    ):
        raise ContractError(f"{label} must be one lexical-canonical absolute non-parent path")
    return path


def require_repo_relative(value: Any, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise ContractError(f"{label} must be one non-empty repository-relative path")
    if any(ord(char) < 32 for char in value) or PATH_META_RE.search(value):
        raise ContractError(f"{label} must not contain control/glob/wildcard syntax")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ContractError(f"{label} must be one lexical-canonical repository-relative path")
    return path


def require_regular(path: Path, label: str, *, symlink_forbidden: bool = True) -> os.stat_result:
    if not path.is_file() or (symlink_forbidden and path.is_symlink()):
        raise ContractError(f"{label} must be a regular non-symlink file: {path}")
    return path.stat()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _repo_bound_file(repo_root: Path, relative_value: Any, label: str) -> Path:
    relative = require_repo_relative(relative_value, label)
    unresolved = repo_root / relative
    if unresolved.is_symlink():
        raise ContractError(f"{label} must not be a symlink")
    path = unresolved.resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ContractError(f"{label} escapes repository root") from exc
    if path != unresolved:
        raise ContractError(f"{label} must have no symlink ancestry")
    require_regular(path, label)
    return path


def _source_file(repo_root: Path, spec: Mapping[str, Any], label: str) -> Path:
    exact_keys(spec, {"path", "sha256"}, f"source binding {label}")
    require_sha(spec["sha256"], f"source binding {label}")
    path = _repo_bound_file(repo_root, spec["path"], f"source binding {label}")
    if sha256_file(path) != spec["sha256"]:
        raise ContractError(f"source binding {label} bytes changed")
    return path


def load_manifest(path: Path, *, repo_root: Path) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != path:
        raise ContractError("attestor manifest must be a canonical non-symlink file")
    if repo_root.resolve() != repo_root:
        raise ContractError("repo root must be one canonical path")
    expected_path = (repo_root / MANIFEST_RELATIVE_PATH).resolve()
    if path.resolve() != expected_path:
        raise ContractError("attestor manifest must be the tracked canonical path")
    manifest = read_json(path, "attestor manifest")
    exact_keys(
        manifest,
        {
            "schema_version",
            "manifest_id",
            "status",
            "recorded_local_date",
            "human_owner",
            "executor",
            "simulation_only",
            "real_robot_commands_forbidden",
            "source_bindings",
            "paper",
            "receipt_correction",
            "execution_semantics",
            "output",
            "authorization",
        },
        "attestor manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["manifest_id"] != MANIFEST_ID
        or manifest["status"] != "source_reviewed_runtime_attestation_not_run"
        or manifest["recorded_local_date"] != "2026-07-14"
        or manifest["human_owner"] != "Franco"
        or manifest["executor"] != "Codex"
        or manifest["simulation_only"] is not True
        or manifest["real_robot_commands_forbidden"] is not True
    ):
        raise ContractError("attestor identity/status/ownership/safety changed")
    require_exact(manifest["authorization"], ATTESTATION_AUTHORIZATION, "authorization")
    sources = manifest["source_bindings"]
    exact_keys(
        sources,
        {
            "runner",
            "judge",
            "play_exporter",
            "std_sidecar",
            "training_contract",
            "policy_runner",
            "question_bank_loader",
            "mujoco_evaluator",
            "signed_face_scorer",
            "schedule_module",
        },
        "source_bindings",
    )
    resolved = {
        name: _source_file(repo_root, spec, name) for name, spec in sources.items()
    }
    if resolved["runner"] != Path(__file__).resolve():
        raise ContractError("manifest runner path does not resolve to this consumer")
    manifest["_resolved_sources"] = resolved
    validate_manifest_paper(manifest["paper"])
    correction_path = _repo_bound_file(
        repo_root,
        manifest["receipt_correction"]["path"],
        "receipt correction",
    )
    validate_correction(correction_path, manifest["receipt_correction"], repo_root=repo_root)
    semantics = manifest["execution_semantics"]
    exact_keys(
        semantics,
        {
            "evaluation_contract_exact",
            "signed_face_exact",
            "checkpoint_contract",
            "runtime_packages",
            "plant_contract_keys",
            "plant_execution",
            "paper_execution",
        },
        "execution_semantics",
    )
    require_exact(semantics["evaluation_contract_exact"], True, "evaluation exactness")
    require_exact(semantics["signed_face_exact"], True, "signed-face exactness")
    require_exact(
        semantics["checkpoint_contract"],
        {
            "schema_version": 3,
            "fresh_lineage_exact": 1,
            "actor_obs_contract": "deploy_parity_face179",
            "actor_obs_total_dim": 179,
            "face_command_pairing": "shared_plus_y",
            "mount_normal_sign_per_clip": [1.0, -1.0],
            "motion_kinematics_exact": True,
            "motion_allow_legacy_link_origin_velocity": False,
        },
        "checkpoint contract",
    )
    require_exact(
        semantics["runtime_packages"],
        {"checkpoint_python": ["torch"], "evaluator_python": ["mujoco", "numpy", "onnx", "onnxruntime"]},
        "runtime package closure",
    )
    require_exact(semantics["plant_contract_keys"], list(PLANT_KEYS), "plant keys")
    require_exact(
        semantics["plant_execution"],
        {
            "passive_damping": "auto",
            "frictionloss": "auto",
            "qdes_clamp": True,
            "require_bound_plant_match": True,
            "allow_inexact_contract": False,
        },
        "plant execution",
    )
    require_exact(
        semantics["paper_execution"],
        {
            "all_scheduled_attempts_in_denominator": True,
            "missing_invalid_or_reset_attempts_count_as_failures": True,
            "attempts": 100,
            "attempts_per_side": 50,
            "schedule_seed": 0,
            "hold_range": [0, 100],
            "no_wrap": True,
        },
        "paper execution",
    )
    output = manifest["output"]
    exact_keys(output, {"root", "evidence_basename", "claim_basename", "claim_written_last"}, "output")
    require_absolute(output["root"], "output.root")
    if Path(output["root"]) != OUTPUT_ROOT:
        raise ContractError("output root changed from the global checkpoint-SHA namespace")
    if output["evidence_basename"] != "execution_evidence.json" or output["claim_basename"] != "execution_claim.json":
        raise ContractError("output basenames changed")
    if output["claim_written_last"] is not True:
        raise ContractError("claim must be written last")
    return manifest


def validate_manifest_paper(paper: Any) -> None:
    exact_keys(
        paper,
        {"bank", "schedule", "activation", "signed_face_contract", "denominator"},
        "paper",
    )
    require_exact(
        paper["bank"],
        {
            "sha256": "60e1a7ade72eaf64e17a1b83795125551f08c6699c8a3cc3c269500d8e6cd1ca",
            "schema_version": 3,
            "split": "exam",
            "physics_contract_sha256": "09dfe8999c54e36b258fe54b5ec3da5d9816ff3be3675963b919371d7f4afb95",
            "source_family_sha256": "9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db",
        },
        "paper bank",
    )
    require_exact(
        paper["schedule"],
        {
            "path": "/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/signed_face_exam_k100_v1/signed_face_exam_k100.schedule.json",
            "bytes": 20237,
            "file_sha256": "f2777dcd02080ba68b839c76ea9d3f14c938457c9bc01b5692fe86ae59157ec7",
            "semantic_sha256": "3ca4bdba7f4acbe6f211d90e95305fc4a9459c118e4220c9683060c0a6723365",
            "question_id_order_sha256": "09f778f2afd7888069ac75aabee3bf19deda015acbbad843ccdb70ca39548bd0",
            "scheduled_attempts": 100,
            "unique_question_ids": 100,
            "selected_per_side": {"forehand": 50, "backhand": 50},
        },
        "paper schedule",
    )
    require_exact(
        paper["activation"],
        {
            "path": "/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/signed_face_exam_k100_v1/signed_face_exam_k100.activation.json",
            "bytes": 11620,
            "file_sha256": "e0125b0e937655672e68ac79578c075e4cf8e99fc1cad5655bcb7e3e4a977bb4",
            "content_sha256": "533beb03a13236eb93404b5c141f6459b08520060b0c2ed0b6dc4ee2e46db3d8",
            "artifact_kind": ACTIVATION_KIND,
        },
        "paper activation",
    )
    require_exact(paper["signed_face_contract"], EXPECTED_FACE_CONTRACT, "paper signed face")
    require_exact(paper["denominator"], {"aggregate": 100, "forehand": 50, "backhand": 50}, "paper denominator")


def validate_correction(path: Path, spec: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    exact_keys(spec, {"path", "sha256"}, "receipt correction binding")
    require_repo_relative(spec["path"], "receipt correction path")
    require_sha(spec["sha256"], "receipt correction")
    if path != _repo_bound_file(repo_root, spec["path"], "receipt correction"):
        raise ContractError("receipt correction path does not resolve to its tracked file")
    if sha256_file(path) != spec["sha256"]:
        raise ContractError("receipt correction bytes changed")
    value = read_json(path, "receipt correction")
    exact_keys(
        value,
        {
            "schema_version",
            "correction_id",
            "status",
            "recorded_local_date",
            "human_owner",
            "executor",
            "original_receipt",
            "affected_summary_field",
            "recorded_summary_value",
            "correction",
            "trust_rule",
            "authorization",
        },
        "receipt correction",
    )
    if (
        value["schema_version"] != 1
        or value["correction_id"] != CORRECTION_ID
        or value["recorded_local_date"] != "2026-07-14"
        or value["human_owner"] != "Franco"
        or value["executor"] != "Codex"
    ):
        raise ContractError("receipt correction identity changed")
    if value["status"] != "versioned_summary_correction_original_receipt_preserved":
        raise ContractError("receipt correction status changed")
    original = exact_keys(
        value["original_receipt"],
        {"path", "sha256", "receipt_id", "preserved_unchanged"},
        "original runtime receipt binding",
    )
    require_exact(original["preserved_unchanged"], True, "original receipt preservation")
    require_exact(
        original["receipt_id"],
        "phase1-signed-face-exam-k100-runtime-receipt-20260714-v1",
        "original receipt id",
    )
    require_sha(original["sha256"], "original runtime receipt SHA")
    original_path = _repo_bound_file(repo_root, original["path"], "original runtime receipt")
    if sha256_file(original_path) != original["sha256"]:
        raise ContractError("original runtime receipt was edited or replaced")
    old = read_json(original_path, "original runtime receipt")
    require_exact(
        old.get("signed_face_contract", {}).get("mount_normal_sign_per_clip"),
        [1, -1],
        "original receipt recorded summary",
    )
    require_exact(
        value["affected_summary_field"],
        "signed_face_contract.mount_normal_sign_per_clip",
        "affected receipt summary field",
    )
    require_exact(value["recorded_summary_value"], [1, -1], "correction recorded summary")
    correction = exact_keys(
        value["correction"],
        {
            "kind",
            "correct_exact_value",
            "integers_are_not_accepted_as_float_equivalents",
            "source_manifest",
            "materializer",
            "actual_activation",
        },
        "correction body",
    )
    require_exact(correction["kind"], "json_numeric_type_correction", "correction kind")
    require_exact(correction["correct_exact_value"], [1.0, -1.0], "correct signed-face value")
    require_exact(
        correction["integers_are_not_accepted_as_float_equivalents"],
        True,
        "strict correction type rule",
    )
    for name, expected in (
        (
            "source_manifest",
            {
                "path": "configs/phase1_signed_face_exam_k100_activation_prereg_20260714.json",
                "sha256": "e401305d4564def80677e6d881ef4afabde01d96ea7ea6aa08224d86835de556",
            },
        ),
        (
            "materializer",
            {
                "path": "scripts/materialize_phase1_signed_face_exam_k100.py",
                "sha256": "4e094bbebe525fb9cd756c3fa6eebe7436c72f94aba2a12ecd136f612761ac6e",
            },
        ),
    ):
        require_exact(correction[name], expected, f"correction {name}")
        _source_file(repo_root, correction[name], f"correction {name}")
    actual = exact_keys(
        correction["actual_activation"],
        {
            "path",
            "bytes",
            "file_sha256",
            "content_sha256",
            "authoritative_field",
            "required_exact_value",
        },
        "correction actual activation",
    )
    require_exact(
        actual,
        {
            "path": "/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/signed_face_exam_k100_v1/signed_face_exam_k100.activation.json",
            "bytes": 11620,
            "file_sha256": "e0125b0e937655672e68ac79578c075e4cf8e99fc1cad5655bcb7e3e4a977bb4",
            "content_sha256": "533beb03a13236eb93404b5c141f6459b08520060b0c2ed0b6dc4ee2e46db3d8",
            "authoritative_field": "content.signed_face_contract.mount_normal_sign_per_clip",
            "required_exact_value": [1.0, -1.0],
        },
        "actual activation correction authority",
    )
    trust = value["trust_rule"]
    expected_trust = {
        "old_receipt_summary_may_describe_file_hashes": True,
        "old_receipt_summary_must_not_supply_signed_face_numeric_types": True,
        "consumer_must_verify_actual_activation_bytes_and_content": True,
        "silent_edit_or_replacement_of_original_receipt_forbidden": True,
    }
    require_exact(trust, expected_trust, "receipt correction trust rule")
    require_exact(
        value["authorization"],
        {
            "paper_materialization_only": True,
            "trainer_started": False,
            "judge_started": False,
            "checkpoint_execution_started": False,
            "stop_or_promote_authorized": False,
            "deployment_authorized": False,
            "real_robot_authorized": False,
        },
        "receipt correction authorization",
    )
    return value


def load_request(path: Path, manifest: Mapping[str, Any], *, runtime: bool) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ContractError("checkpoint execution request must be an absolute canonical non-symlink file")
    request = read_json(path, "checkpoint execution request")
    exact_keys(
        request,
        {
            "schema_version",
            "request_id",
            "status",
            "human_owner",
            "executor",
            "simulation_only",
            "real_robot_commands_forbidden",
            "source_checkout",
            "checkpoint",
            "adjacent_hard_contract",
            "runtime",
            "mjcf",
            "output",
            "authorization",
        },
        "checkpoint execution request",
    )
    if request["schema_version"] != 1 or request["status"] != REQUEST_STATUS:
        raise ContractError("request schema/status changed")
    if type(request["request_id"]) is not str or not REQUEST_RE.fullmatch(request["request_id"]):
        raise ContractError("request_id must be one safe explicit identifier")
    if request["human_owner"] != "Franco" or request["executor"] != "Codex":
        raise ContractError("request human owner/executor changed")
    if request["simulation_only"] is not True or request["real_robot_commands_forbidden"] is not True:
        raise ContractError("request must stay simulation-only and forbid robot commands")
    require_exact(request["authorization"], ATTESTATION_AUTHORIZATION, "request authorization")
    source = request["source_checkout"]
    exact_keys(source, {"path", "commit", "tree", "clean_required"}, "source checkout")
    require_absolute(source["path"], "source checkout path")
    require_git(source["commit"], "source checkout commit")
    require_git(source["tree"], "source checkout tree")
    require_exact(source["clean_required"], True, "source checkout cleanliness")

    checkpoint = request["checkpoint"]
    exact_keys(
        checkpoint,
        {
            "path",
            "bytes",
            "sha256",
            "filename_iteration",
            "embedded_iteration",
            "training_contract_schema_version",
            "training_contract_sha256",
            "training_contract_lineage_exact",
            "producer_claim",
        },
        "checkpoint",
    )
    checkpoint_path = require_absolute(checkpoint["path"], "checkpoint.path")
    require_plain_int(checkpoint["bytes"], "checkpoint.bytes", minimum=1)
    require_sha(checkpoint["sha256"], "checkpoint.sha256")
    file_iteration = require_plain_int(checkpoint["filename_iteration"], "checkpoint filename iteration")
    embedded_iteration = require_plain_int(checkpoint["embedded_iteration"], "checkpoint embedded iteration")
    if file_iteration != embedded_iteration:
        raise ContractError("request filename/embed iterations differ")
    match = CHECKPOINT_RE.fullmatch(checkpoint_path.name)
    if match is None or int(match.group(1)) != file_iteration:
        raise ContractError("checkpoint filename does not encode the requested iteration")
    require_exact(checkpoint["training_contract_schema_version"], 3, "checkpoint contract schema")
    require_sha(checkpoint["training_contract_sha256"], "checkpoint training contract")
    require_exact(checkpoint["training_contract_lineage_exact"], 1, "checkpoint fresh lineage")
    claim = checkpoint["producer_claim"]
    exact_keys(claim, {"path", "file_sha256", "canonical_sha256"}, "producer claim")
    require_absolute(claim["path"], "producer claim path")
    require_sha(claim["file_sha256"], "producer claim file")
    require_sha(claim["canonical_sha256"], "producer claim canonical")

    hard = request["adjacent_hard_contract"]
    exact_keys(hard, {"path", "bytes", "sha256", "plant_contract_sha256"}, "adjacent hard contract")
    hard_path = require_absolute(hard["path"], "hard contract path")
    expected_hard = checkpoint_path.parent / "params" / "training_contract.json"
    if hard_path != expected_hard:
        raise ContractError("hard contract is not adjacent to the exact checkpoint")
    require_plain_int(hard["bytes"], "hard contract bytes", minimum=1)
    require_sha(hard["sha256"], "hard contract SHA")
    require_sha(hard["plant_contract_sha256"], "plant contract SHA")
    if hard["sha256"] != checkpoint["training_contract_sha256"]:
        raise ContractError("request checkpoint/hard-contract SHA values disagree")

    runtime_spec = request["runtime"]
    exact_keys(runtime_spec, {"checkpoint_python", "evaluator_python"}, "runtime")
    package_sets = manifest["execution_semantics"]["runtime_packages"]
    for name in ("checkpoint_python", "evaluator_python"):
        spec = runtime_spec[name]
        exact_keys(
            spec,
            {"path", "resolved_path", "resolved_sha256", "fingerprint"},
            f"runtime {name}",
        )
        require_absolute(spec["path"], f"runtime {name}.path")
        require_absolute(spec["resolved_path"], f"runtime {name}.resolved_path")
        require_sha(spec["resolved_sha256"], f"runtime {name}.resolved_sha256")
        fingerprint = spec["fingerprint"]
        exact_keys(fingerprint, {"implementation", "version", "executable", "packages"}, f"runtime {name}.fingerprint")
        if type(fingerprint["implementation"]) is not str or type(fingerprint["version"]) is not str:
            raise ContractError(f"runtime {name} implementation/version must be strings")
        require_absolute(fingerprint["executable"], f"runtime {name}.fingerprint.executable")
        expected_packages = package_sets[name]
        if not isinstance(fingerprint["packages"], dict) or list(fingerprint["packages"]) != expected_packages:
            raise ContractError(f"runtime {name} package closure/order changed")
        if any(type(value) is not str or not value for value in fingerprint["packages"].values()):
            raise ContractError(f"runtime {name} package versions must be non-empty strings")

    mjcf = request["mjcf"]
    exact_keys(mjcf, {"path", "bytes", "sha256"}, "MJCF")
    require_absolute(mjcf["path"], "MJCF path")
    require_plain_int(mjcf["bytes"], "MJCF bytes", minimum=1)
    require_sha(mjcf["sha256"], "MJCF SHA")

    output = request["output"]
    exact_keys(output, {"root"}, "request output")
    output_root = require_absolute(output["root"], "request output root")
    expected_output = Path(manifest["output"]["root"]) / checkpoint["sha256"]
    if output_root != expected_output:
        raise ContractError("request output is not the unique checkpoint-SHA namespace")
    if runtime:
        validate_runtime_request(request, manifest)
    return request


def git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"git source check failed: {exc.output.strip()}") from exc


def validate_source_checkout(request: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    spec = request["source_checkout"]
    root = Path(spec["path"])
    if root.resolve() != root or not (root / ".git").exists():
        raise ContractError("source checkout must be one real git worktree root")
    if git_output(root, "rev-parse", "HEAD") != spec["commit"]:
        raise ContractError("source checkout commit changed")
    if git_output(root, "rev-parse", "HEAD^{tree}") != spec["tree"]:
        raise ContractError("source checkout tree changed")
    if git_output(root, "status", "--porcelain"):
        raise ContractError("source checkout is not clean")
    resolved = {}
    for name, source_spec in manifest["source_bindings"].items():
        resolved[name] = _source_file(root, source_spec, name)
    if resolved["runner"] != Path(__file__).resolve():
        raise ContractError("runtime source checkout does not contain this exact runner")
    return {"path": str(root), "commit": spec["commit"], "tree": spec["tree"], "clean": True}


def validate_file_binding(path: Path, *, size: int, digest: str, label: str) -> os.stat_result:
    if path.resolve() != path:
        raise ContractError(f"{label} path must be canonical with no symlink ancestry")
    before = require_regular(path, label)
    actual_digest = sha256_file(path)
    after = require_regular(path, label)
    if _stat_identity(before) != _stat_identity(after):
        raise ContractError(f"{label} was replaced or changed while hashing")
    if after.st_size != size or actual_digest != digest:
        raise ContractError(f"{label} bytes/SHA changed")
    return after


def revalidate_file_binding(
    path: Path,
    *,
    initial: os.stat_result,
    size: int,
    digest: str,
    label: str,
) -> os.stat_result:
    current = validate_file_binding(path, size=size, digest=digest, label=label)
    if _stat_identity(current) != _stat_identity(initial):
        raise ContractError(f"{label} was replaced or changed after validation")
    return current


def validate_json_snapshot(
    path: Path,
    expected: Mapping[str, Any],
    label: str,
    *,
    initial: os.stat_result | None = None,
) -> tuple[os.stat_result, str]:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ContractError(f"{label} must be an absolute canonical non-symlink file")
    before = require_regular(path, label)
    value = read_json(path, label)
    digest = sha256_file(path)
    after = require_regular(path, label)
    if _stat_identity(before) != _stat_identity(after):
        raise ContractError(f"{label} was replaced or changed while reading")
    if initial is not None and _stat_identity(after) != _stat_identity(initial):
        raise ContractError(f"{label} was replaced or changed after validation")
    require_exact(value, dict(expected), f"{label} loaded content")
    return after, digest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import bound source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_actual_paper(manifest: Mapping[str, Any]) -> dict[str, Any]:
    paper = manifest["paper"]
    schedule_spec = paper["schedule"]
    activation_spec = paper["activation"]
    schedule_path = Path(schedule_spec["path"])
    activation_path = Path(activation_spec["path"])
    schedule_stat = validate_file_binding(
        schedule_path,
        size=schedule_spec["bytes"],
        digest=schedule_spec["file_sha256"],
        label="actual immutable schedule",
    )
    activation_stat = validate_file_binding(
        activation_path,
        size=activation_spec["bytes"],
        digest=activation_spec["file_sha256"],
        label="actual paper activation",
    )
    schedule_document = read_json(schedule_path, "actual immutable schedule")
    schedule_module = _load_module(
        manifest["_resolved_sources"]["schedule_module"],
        "signed_k100_execution_schedule_bound",
    )
    try:
        artifact = schedule_module._artifact_from_document(schedule_document)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"actual schedule schema/semantic validation failed: {exc}") from exc
    items = list(artifact.items)
    order = [item.question_id for item in items]
    counts = {
        "forehand": sum(item.clip == 0 for item in items),
        "backhand": sum(item.clip == 1 for item in items),
    }
    if (
        artifact.bank_sha256 != paper["bank"]["sha256"]
        or artifact.schema_version != 3
        or artifact.bank_schema_version != 3
        or artifact.artifact_type != "bank-exam-schedule"
        or list(artifact.clip_order) != ["forehand", "backhand"]
        or artifact.per_clip_quota != 50
        or artifact.schedule_seed != 0
        or list(artifact.hold_range) != [0, 100]
        or artifact.no_wrap is not True
        or len(items) != 100
        or counts != {"forehand": 50, "backhand": 50}
        or len(set(order)) != 100
        or artifact.schedule_sha256 != schedule_spec["semantic_sha256"]
        or canonical_sha256(order) != schedule_spec["question_id_order_sha256"]
    ):
        raise ContractError("actual schedule differs from immutable K100 semantics/order")

    activation = read_json(activation_path, "actual paper activation")
    exact_keys(activation, {"schema_version", "artifact_kind", "content_sha256", "content"}, "actual activation")
    content = activation["content"]
    if not isinstance(content, dict):
        raise ContractError("actual activation content must be an object")
    if (
        activation["schema_version"] != 1
        or activation["artifact_kind"] != ACTIVATION_KIND
        or activation["content_sha256"] != activation_spec["content_sha256"]
        or activation["content_sha256"] != canonical_sha256(content)
        or content.get("status") != "paper_materialized_not_started"
        or content.get("bank", {}).get("sha256") != paper["bank"]["sha256"]
    ):
        raise ContractError("actual activation identity/content binding changed")
    require_exact(content.get("signed_face_contract"), EXPECTED_FACE_CONTRACT, "actual activation signed face")
    require_exact(content.get("scoring_denominator"), paper["denominator"], "actual activation denominator")
    require_exact(content.get("all_scheduled_attempts_in_denominator"), True, "activation full denominator")
    require_exact(content.get("authorization"), PAPER_AUTHORIZATION, "actual activation authorization")
    receipt = content.get("schedule")
    if not isinstance(receipt, dict):
        raise ContractError("actual activation lacks schedule receipt")
    required_receipt = {
        "path": str(schedule_path.resolve()),
        "bytes": schedule_spec["bytes"],
        "file_sha256": schedule_spec["file_sha256"],
        "semantic_sha256": schedule_spec["semantic_sha256"],
        "question_id_order_sha256": schedule_spec["question_id_order_sha256"],
        "question_id_order": order,
        "schedule_k": 100,
        "selected_per_side": {"forehand": 50, "backhand": 50},
    }
    require_exact(receipt, required_receipt, "actual activation schedule receipt")
    revalidate_file_binding(
        schedule_path,
        initial=schedule_stat,
        size=schedule_spec["bytes"],
        digest=schedule_spec["file_sha256"],
        label="actual immutable schedule",
    )
    revalidate_file_binding(
        activation_path,
        initial=activation_stat,
        size=activation_spec["bytes"],
        digest=activation_spec["file_sha256"],
        label="actual paper activation",
    )
    return {
        "schedule": dict(schedule_spec),
        "activation": dict(activation_spec),
        "actual_signed_face_contract": content["signed_face_contract"],
        "question_id_order_sha256": canonical_sha256(order),
        "all_scheduled_attempts_in_denominator": True,
    }


def _runtime_fingerprint(spec: Mapping[str, Any], package_names: list[str], label: str) -> dict[str, Any]:
    path = Path(spec["path"])
    resolved = path.resolve()
    resolved_stat = require_regular(resolved, f"{label} resolved executable")
    executable_stat = validate_file_binding(
        resolved,
        size=resolved_stat.st_size,
        digest=spec["resolved_sha256"],
        label=f"{label} resolved executable",
    )
    if str(resolved) != spec["resolved_path"]:
        raise ContractError(f"{label} resolved executable changed")
    try:
        raw = subprocess.check_output(
            [str(path), "-c", RUNTIME_FINGERPRINT_CODE, json.dumps(package_names)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} fingerprint failed: {exc}") from exc
    require_exact(value, spec["fingerprint"], f"{label} fingerprint")
    if path.resolve() != resolved:
        raise ContractError(f"{label} launcher target changed during fingerprint")
    revalidate_file_binding(
        resolved,
        initial=executable_stat,
        size=executable_stat.st_size,
        digest=spec["resolved_sha256"],
        label=f"{label} resolved executable",
    )
    return value


def checkpoint_audit(python: Path, checkpoint: Path) -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            [str(python), "-c", CHECKPOINT_AUDIT_CODE, str(checkpoint)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ContractError(f"checkpoint tensor/lineage audit failed: {exc}") from exc
    exact_keys(
        value,
        {
            "iter",
            "training_contract_schema_version",
            "training_contract_sha256",
            "training_contract_lineage_exact",
            "training_launch_claim_sha256",
            "floating_tensor_count",
            "floating_elements",
            "nonfinite_floating_elements",
        },
        "checkpoint audit",
    )
    return value


def validate_hard_contract(value: Mapping[str, Any], request: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    expected = manifest["execution_semantics"]["checkpoint_contract"]
    for key, wanted in (
        ("schema_version", expected["schema_version"]),
        ("actor_obs_contract", expected["actor_obs_contract"]),
        ("actor_obs_total_dim", expected["actor_obs_total_dim"]),
        ("face_command_pairing", expected["face_command_pairing"]),
        ("mount_normal_sign_per_clip", expected["mount_normal_sign_per_clip"]),
        ("motion_kinematics_exact", expected["motion_kinematics_exact"]),
        ("motion_allow_legacy_link_origin_velocity", expected["motion_allow_legacy_link_origin_velocity"]),
    ):
        require_exact(value.get(key), wanted, f"hard contract {key}")
    if len(value.get("joint_names", [])) != 31 or len(value.get("action_joint_ids", [])) != 31:
        raise ContractError("hard contract must bind exactly 31 joints/actions")
    plant = {}
    for key in PLANT_KEYS:
        if key not in value:
            raise ContractError(f"hard contract lacks plant field {key}")
        plant[key] = value[key]
    plant_sha = canonical_sha256(plant)
    if plant_sha != request["adjacent_hard_contract"]["plant_contract_sha256"]:
        raise ContractError("hard-contract plant semantic SHA changed")
    return plant_sha


def validate_runtime_request(request: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = validate_source_checkout(request, manifest)
    checkpoint = request["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    checkpoint_stat = validate_file_binding(
        checkpoint_path,
        size=checkpoint["bytes"],
        digest=checkpoint["sha256"],
        label="exact checkpoint",
    )
    hard_spec = request["adjacent_hard_contract"]
    hard_path = Path(hard_spec["path"])
    hard_stat = validate_file_binding(
        hard_path,
        size=hard_spec["bytes"],
        digest=hard_spec["sha256"],
        label="adjacent hard contract",
    )
    hard = read_json(hard_path, "adjacent hard contract")
    plant_sha = validate_hard_contract(hard, request, manifest)

    claim_spec = checkpoint["producer_claim"]
    claim_path = Path(claim_spec["path"])
    if claim_path.resolve() != claim_path:
        raise ContractError("producer claim path must be canonical with no symlink ancestry")
    claim_stat = require_regular(claim_path, "producer claim")
    claim_stat = validate_file_binding(
        claim_path,
        size=claim_stat.st_size,
        digest=claim_spec["file_sha256"],
        label="producer claim",
    )
    producer_claim = read_json(claim_path, "producer claim")
    if canonical_sha256(producer_claim) != claim_spec["canonical_sha256"]:
        raise ContractError("producer claim canonical SHA changed")

    package_sets = manifest["execution_semantics"]["runtime_packages"]
    checkpoint_runtime = _runtime_fingerprint(
        request["runtime"]["checkpoint_python"], package_sets["checkpoint_python"], "checkpoint Python"
    )
    evaluator_runtime = _runtime_fingerprint(
        request["runtime"]["evaluator_python"], package_sets["evaluator_python"], "evaluator Python"
    )
    audit = checkpoint_audit(Path(request["runtime"]["checkpoint_python"]["path"]), checkpoint_path)
    expected_audit = {
        "iter": checkpoint["embedded_iteration"],
        "training_contract_schema_version": checkpoint["training_contract_schema_version"],
        "training_contract_sha256": checkpoint["training_contract_sha256"],
        "training_contract_lineage_exact": checkpoint["training_contract_lineage_exact"],
        "training_launch_claim_sha256": claim_spec["canonical_sha256"],
    }
    for key, wanted in expected_audit.items():
        require_exact(audit.get(key), wanted, f"checkpoint audit {key}")
    for key in ("floating_tensor_count", "floating_elements"):
        if type(audit[key]) is not int or audit[key] <= 0:
            raise ContractError(f"checkpoint audit {key} must be a positive integer")
    require_exact(audit["nonfinite_floating_elements"], 0, "checkpoint finite audit")

    mjcf = request["mjcf"]
    mjcf_stat = validate_file_binding(
        Path(mjcf["path"]), size=mjcf["bytes"], digest=mjcf["sha256"], label="exact MJCF"
    )
    paper = validate_actual_paper(manifest)
    revalidate_file_binding(
        Path(mjcf["path"]),
        initial=mjcf_stat,
        size=mjcf["bytes"],
        digest=mjcf["sha256"],
        label="exact MJCF",
    )
    revalidate_file_binding(
        checkpoint_path,
        initial=checkpoint_stat,
        size=checkpoint["bytes"],
        digest=checkpoint["sha256"],
        label="exact checkpoint",
    )
    revalidate_file_binding(
        hard_path,
        initial=hard_stat,
        size=hard_spec["bytes"],
        digest=hard_spec["sha256"],
        label="adjacent hard contract",
    )
    revalidate_file_binding(
        claim_path,
        initial=claim_stat,
        size=claim_stat.st_size,
        digest=claim_spec["file_sha256"],
        label="producer claim",
    )
    source_after = validate_source_checkout(request, manifest)
    require_exact(source_after, source, "source checkout post-runtime stability")
    return {
        "source_checkout": source,
        "checkpoint_audit": audit,
        "hard_contract_sha256": hard_spec["sha256"],
        "plant_contract_sha256": plant_sha,
        "producer_claim": dict(claim_spec),
        "checkpoint_python": checkpoint_runtime,
        "evaluator_python": evaluator_runtime,
        "mjcf": dict(mjcf),
        "paper": paper,
    }


def content_document(kind: str, content: Mapping[str, Any]) -> dict[str, Any]:
    material = dict(content)
    return {
        "schema_version": 1,
        "artifact_kind": kind,
        "content_sha256": canonical_sha256(material),
        "content": material,
    }


def write_exclusive(path: Path, document: Mapping[str, Any]) -> None:
    payload = canonical_bytes(document) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def require_safe_output_ancestry(path: Path) -> None:
    current = path
    while not _lexists(current):
        if current == current.parent:
            break
        current = current.parent
    if current.is_symlink() or current.resolve() != current:
        raise ContractError(f"output ancestry must be canonical and symlink-free: {current}")


def attest(
    request_path: Path,
    request: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_document = {
        key: value for key, value in manifest.items() if not key.startswith("_")
    }
    request_stat, request_sha = validate_json_snapshot(
        request_path, request, "checkpoint execution request snapshot"
    )
    manifest_stat, manifest_sha = validate_json_snapshot(
        manifest_path, manifest_document, "checkpoint attestor manifest snapshot"
    )
    output_root = Path(request["output"]["root"])
    if _lexists(output_root):
        raise ContractError(f"no-clobber: checkpoint execution namespace exists: {output_root}")
    require_safe_output_ancestry(output_root.parent)
    runtime = validate_runtime_request(request, manifest)
    _, request_sha_after = validate_json_snapshot(
        request_path,
        request,
        "checkpoint execution request snapshot",
        initial=request_stat,
    )
    _, manifest_sha_after = validate_json_snapshot(
        manifest_path,
        manifest_document,
        "checkpoint attestor manifest snapshot",
        initial=manifest_stat,
    )
    if request_sha_after != request_sha or manifest_sha_after != manifest_sha:
        raise ContractError("request or manifest bytes changed during runtime attestation")
    if _lexists(output_root):
        raise ContractError("checkpoint execution namespace appeared during pre-write validation")
    require_safe_output_ancestry(output_root.parent)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.parent.resolve() != output_root.parent:
        raise ContractError("output parent must be canonical with no symlink ancestry")
    output_root.mkdir(exist_ok=False)
    evidence_path = output_root / manifest["output"]["evidence_basename"]
    claim_path = output_root / manifest["output"]["claim_basename"]
    correction_spec = manifest["receipt_correction"]
    content = {
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "request_id": request["request_id"],
        "request_file_sha256": request_sha,
        "request_canonical_sha256": canonical_sha256(dict(request)),
        "status": "exact_checkpoint_inputs_attested_judge_not_started",
        "checkpoint": dict(request["checkpoint"]),
        "adjacent_hard_contract": dict(request["adjacent_hard_contract"]),
        "source_checkout": runtime["source_checkout"],
        "checkpoint_audit": runtime["checkpoint_audit"],
        "producer_claim": runtime["producer_claim"],
        "runtime": {
            "checkpoint_python": runtime["checkpoint_python"],
            "evaluator_python": runtime["evaluator_python"],
        },
        "mjcf": runtime["mjcf"],
        "plant_contract_sha256": runtime["plant_contract_sha256"],
        "plant_execution": manifest["execution_semantics"]["plant_execution"],
        "paper": runtime["paper"],
        "receipt_correction": dict(correction_spec),
        "signed_face_contract": EXPECTED_FACE_CONTRACT,
        "evaluation_contract_exact": True,
        "signed_face_exact": True,
        "authorization": ATTESTATION_AUTHORIZATION,
    }
    evidence = content_document(EVIDENCE_KIND, content)
    write_exclusive(evidence_path, evidence)
    evidence_file_sha = sha256_file(evidence_path)
    claim_content = {
        "manifest_id": MANIFEST_ID,
        "request_id": request["request_id"],
        "checkpoint_sha256": request["checkpoint"]["sha256"],
        "checkpoint_iteration": request["checkpoint"]["embedded_iteration"],
        "training_contract_sha256": request["checkpoint"]["training_contract_sha256"],
        "producer_claim_canonical_sha256": request["checkpoint"]["producer_claim"]["canonical_sha256"],
        "plant_contract_sha256": runtime["plant_contract_sha256"],
        "mjcf_sha256": request["mjcf"]["sha256"],
        "schedule_file_sha256": manifest["paper"]["schedule"]["file_sha256"],
        "schedule_semantic_sha256": manifest["paper"]["schedule"]["semantic_sha256"],
        "schedule_question_id_order_sha256": manifest["paper"]["schedule"]["question_id_order_sha256"],
        "activation_file_sha256": manifest["paper"]["activation"]["file_sha256"],
        "activation_content_sha256": manifest["paper"]["activation"]["content_sha256"],
        "evidence_path": str(evidence_path),
        "evidence_file_sha256": evidence_file_sha,
        "evidence_content_sha256": evidence["content_sha256"],
        "status": "attested_not_executed_no_decision",
        "judge_started": False,
        "stop_or_promote_authorized": False,
        "real_robot_authorized": False,
    }
    claim = content_document(CLAIM_KIND, claim_content)
    write_exclusive(claim_path, claim)
    return {
        "status": "checkpoint_execution_inputs_attested_judge_not_started",
        "checkpoint_sha256": request["checkpoint"]["sha256"],
        "output_root": str(output_root),
        "evidence_path": str(evidence_path),
        "evidence_file_sha256": evidence_file_sha,
        "claim_path": str(claim_path),
        "claim_file_sha256": sha256_file(claim_path),
        "claim_content_sha256": claim["content_sha256"],
        "judge_started": False,
        "stop_or_promote_authorized": False,
        "real_robot_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--manifest",
        type=Path,
        default=root / "configs/phase1_signed_face_k100_checkpoint_attestor_20260714.json",
    )
    value.add_argument("--repo-root", type=Path, default=root)
    value.add_argument("--request", type=Path)
    value.add_argument("command", choices=("static-validate", "plan", "attest"))
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    if args.command == "static-validate":
        print(
            json.dumps(
                {
                    "status": "source_reviewed_runtime_attestation_not_run",
                    "manifest_id": MANIFEST_ID,
                    "actual_activation_required": True,
                    "old_receipt_summary_authoritative_for_signed_numeric_types": False,
                    "checkpoint_bound": False,
                    "judge_started": False,
                    "stop_or_promote_authorized": False,
                    "real_robot_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.request is None:
        raise ContractError(f"{args.command} requires --request")
    request_path = args.request
    request = load_request(request_path, manifest, runtime=False)
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "status": "exact_request_valid_runtime_not_read_no_writes",
                    "request_id": request["request_id"],
                    "checkpoint_sha256": request["checkpoint"]["sha256"],
                    "unique_output_root": request["output"]["root"],
                    "actual_activation_will_be_read": True,
                    "judge_started": False,
                    "stop_or_promote_authorized": False,
                    "real_robot_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    result = attest(request_path, request, manifest_path, manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[signed-k100-checkpoint-attestor][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2)
