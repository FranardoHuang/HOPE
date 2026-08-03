#!/usr/bin/env python3
"""Plan or explicitly execute the fail-closed A225 four-arm diagnostic.

This planner is deliberately separate from the historical fixed-194 launcher.  It
defaults to read-only planning; execution requires an exact recomputed claim digest.
A canonical spec selects one of four code-owned A225 arms and one finite stage,
binds a tracked A225 lineage, requires a fresh namespace on one physical GPU, and
emits a digest-bound claim.  The GPU is empty by default; exact VendorV2 colocation
is available only through an explicit opt-in and is excluded from speed evidence.

The 4096-env stage is retained as an independent blocked scale plan.  It is not a
prerequisite of the 512-env long stage.  A long plan instead requires a canonical
PASS oracle32 receipt for the same arm, ABI, reward/policy contract, seed and A225
lineage.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_FILE = SCRIPT_DIR / "launch_n1_reward_screen_diagnostic.py"
ADMISSION_FILE = SCRIPT_DIR / "vendor_v2_gpu_admission.py"
EXACT_GROUP_FILE = SCRIPT_DIR / "exact_process_group.py"
OLD_VALIDATOR_FILE = SCRIPT_DIR / "launch_n1_measured_vendor_v2_diagnostic.py"


def _load_helper(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot import helper %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_B = _load_helper("_a225_four_arm_base", BASE_FILE)
_A = _load_helper("_a225_four_arm_gpu_admission", ADMISSION_FILE)
_G = _load_helper("_a225_four_arm_exact_process_group", EXACT_GROUP_FILE)
_OLD = _load_helper("_a225_four_arm_oracle32_validator", OLD_VALIDATOR_FILE)

LaunchRefused = _B.LaunchRefused

SCHEMA_VERSION = 2
SPEC_KIND = "action_ball_a225_four_arm_diagnostic_spec_v2"
CLAIM_KIND = "action_ball_a225_four_arm_diagnostic_claim_v2"
LINEAGE_KIND = "action_ball_a225_fixed_question_lineage_v1"
MATERIALIZATION_KIND = "action_ball_a225_arm_materialization_v1"
ORACLE32_KIND = "action_ball_a225_oracle32_receipt_v1"
RESULT_KIND = "action_ball_a225_four_arm_diagnostic_launch_result_v1"
EXPERIMENT_NAME = "agibot_a3_action_ball_a225_four_arm_diagnostic"

ACTOR_CONTRACT = "action_ball_a225"
ACTOR_WIDTH = 225
CRITIC_CONTRACT = "action_ball_a225_critic_v1"
CRITIC_WIDTH = 318
ACTOR_NORMALIZER_IDENTITY = "action_ball_a225_actor_norm_v1"
CRITIC_NORMALIZER_IDENTITY = "action_ball_a225_critic_norm_v1"
TASK_PROFILE_ID = "HOPEPingPongActionBallA225VendorV2N1Learnability"
GYM_TASK_ID = "HOPE-PingPong-ActionBall-A225Learnability-AgibotA3-v0"
TARGET_SEMANTICS = "a225_desired_contact_v1"
PHYSICAL_BALL_SEMANTICS = "analytic_virtual_ball_authoritative_physx_disabled"
COLOCATION_SPEC_KEY = "allow_vendor_v2_colocation"
HARD_TERMINATION_UNION = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)

LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_action_ball_a225_four_arm_diagnostic.py"
)
ADMISSION_SOURCE = (
    "hope_training/whole_body_tracking/scripts/vendor_v2_gpu_admission.py"
)
EXACT_GROUP_SOURCE = (
    "hope_training/whole_body_tracking/scripts/exact_process_group.py"
)
BASE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_reward_screen_diagnostic.py"
)
TRAIN_SOURCE = "hope_training/whole_body_tracking/scripts/train.py"
OLD_VALIDATOR_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_measured_vendor_v2_diagnostic.py"
)
KIT_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh"
)
TASK_PROFILE_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA225VendorV2N1Learnability.yaml"
)
A225_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/action_ball_225_trainability.py"
)
A225_ENV_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)
A225_REGISTRY_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/__init__.py"
)

RUNTIME_SOURCE_PATHS = (
    (LAUNCHER_SOURCE, "VendorV2 N1 launcher"),
    (ADMISSION_SOURCE, "VendorV2 GPU admission"),
    (EXACT_GROUP_SOURCE, "exact process-group helper"),
    (BASE_SOURCE, "no-clobber base helper"),
    (KIT_LAUNCHER_SOURCE, "locked Kit launcher"),
    (TRAIN_SOURCE, "training entrypoint"),
    (OLD_VALIDATOR_SOURCE, "oracle32 acceptance validator"),
    (TASK_PROFILE_SOURCE, "A225 task profile"),
    (A225_CONTRACT_SOURCE, "A225 trainability contract"),
    (A225_ENV_SOURCE, "A225 environment config"),
    (A225_REGISTRY_SOURCE, "A225 Gym registration"),
)

ARM_IDS = (
    "L0-corrected-metrics-fixedlr",
    "L1-legacy-penalty-fixedlr",
    "L2-corrected-phase-fixedlr",
    "L3-corrected-phase-adaptive",
)


def _arm(
    death: float,
    qdes: float,
    projection: float,
    joint: float,
    guard: str,
    schedule: str,
    learning_rate: float,
) -> dict[str, Any]:
    return {
        "soft_weights": {
            "death_penalty": death,
            "qdes_limit": qdes,
            "qdes_projection": projection,
            "joint_limit": joint,
        },
        "reference_guard_mode": guard,
        "ppo": {
            "schedule": schedule,
            "learning_rate": learning_rate,
            "desired_kl": 0.01,
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
        },
    }


ARMS: Mapping[str, dict[str, Any]] = {
    ARM_IDS[0]: _arm(-30.0, -0.5, -0.5, -0.5, "metrics_only", "fixed", 1.0e-4),
    ARM_IDS[1]: _arm(-300.0, -5.0, -5.0, -5.0, "metrics_only", "fixed", 1.0e-4),
    ARM_IDS[2]: _arm(-30.0, -0.5, -0.5, -0.5, "phase_gated", "fixed", 1.0e-4),
    ARM_IDS[3]: _arm(-30.0, -0.5, -0.5, -0.5, "phase_gated", "adaptive", 1.0e-3),
}

BUDGETS: Mapping[str, tuple[int, int, int]] = {
    "materialize": (1, 0, 1),
    "oracle32": (1, 0, 1),
    "smoke": (1, 2, 1),
    "probe512": (512, 5, 1),
    "long512": (512, 1000, 100),
    "scale4096": (4096, 5, 1),
}

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PIN_KEYS = ("path", "sha256")
FORBIDDEN_KEY_FRAGMENTS = (
    "target_recipe",
    "target_validity_mask",
    "resume",
    "checkpoint",
)
FORBIDDEN_VALUE_TOKENS = (
    "action_ball_c225",
    "c225",
    "l194",
)


def canonical_sha256(value: Any) -> str:
    return _B.canonical_sha256(value)


def _exact_dict(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    return _B._exact_dict(value, tuple(keys), name=name)


def _assert_no_retired_contract(value: Any, *, name: str) -> None:
    """Reject historical ABI/control-plane vocabulary at the plan boundary."""

    if isinstance(value, dict):
        for key, child in value.items():
            if type(key) is not str:
                raise LaunchRefused("%s contains a non-string key" % name)
            lowered = key.lower()
            if any(token in lowered for token in FORBIDDEN_KEY_FRAGMENTS):
                raise LaunchRefused("%s contains retired key %s" % (name, key))
            _assert_no_retired_contract(child, name=name)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_retired_contract(child, name=name)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in FORBIDDEN_VALUE_TOKENS):
            raise LaunchRefused("%s contains a retired ABI/arm token" % name)


def _pin(value: Any, *, name: str) -> dict[str, str]:
    row = _exact_dict(value, PIN_KEYS, name=name)
    path = _B._relative_path(row["path"], name="%s.path" % name)
    digest = _B._sha256(row["sha256"], name="%s.sha256" % name)
    return {"path": path, "sha256": digest}


def _external_pin(value: Any, *, name: str) -> tuple[dict[str, str], Path]:
    row = _exact_dict(value, PIN_KEYS, name=name)
    path = _B._absolute_path(row["path"], name="%s.path" % name, must_exist=True)
    _B._stable_regular_file(path, name=name)
    digest = _B._sha256(row["sha256"], name="%s.sha256" % name)
    if _B.sha256_file(path) != digest:
        raise LaunchRefused("%s file SHA differs" % name)
    return {"path": str(path), "sha256": digest}, path


def _canonical_external_json(value: Any, *, name: str) -> tuple[dict[str, str], dict]:
    pin, path = _external_pin(value, name=name)
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name=name)
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("%s must be canonical JSON plus newline" % name)
    return pin, document


def _tracked_json(
    checkout: Path,
    commit: str,
    value: Any,
    *,
    name: str,
) -> tuple[dict[str, str], dict]:
    pin = _pin(value, name=name)
    normalized, path = _B._verify_tracked_file(
        checkout, commit, pin, name=name
    )
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name=name)
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("%s must be canonical JSON plus newline" % name)
    return normalized, document


def _validate_lineage(
    checkout: Path, commit: str, value: Any
) -> dict[str, Any]:
    pin, row = _tracked_json(checkout, commit, value, name="A225 lineage")
    row = _exact_dict(
        row,
        (
            "schema_version",
            "kind",
            "actor_contract",
            "actor_width",
            "critic_contract",
            "critic_width",
            "task_profile",
            "gym_task",
            "target_semantics",
            "action_id",
            "teacher_id",
            "seed",
            "bundle",
            "motion",
            "immutable_tape",
            "action_manifest",
            "dynamic_ready_artifact",
            "dynamic_ready_nominal_receipt",
        ),
        name="A225 lineage",
    )
    _assert_no_retired_contract(row, name="A225 lineage")
    expected = {
        "schema_version": 1,
        "kind": LINEAGE_KIND,
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "task_profile": TASK_PROFILE_ID,
        "gym_task": GYM_TASK_ID,
        "target_semantics": TARGET_SEMANTICS,
    }
    for key, wanted in expected.items():
        if row[key] != wanted:
            raise LaunchRefused("A225 lineage %s differs" % key)
    action_id = row["action_id"]
    teacher_id = row["teacher_id"]
    if (
        type(action_id) is not str
        or not action_id
        or SAFE_COMPONENT.fullmatch(action_id) is None
        or type(teacher_id) is not str
        or not teacher_id
        or SAFE_COMPONENT.fullmatch(teacher_id) is None
    ):
        raise LaunchRefused("A225 action/teacher identity is unsafe")
    seed = _B._plain_int(row["seed"], name="A225 lineage seed", maximum=(1 << 31) - 1)
    if seed != 0:
        raise LaunchRefused("A225 first-wave lineage requires seed 0")
    pins = {}
    for key in (
        "bundle",
        "motion",
        "immutable_tape",
        "action_manifest",
        "dynamic_ready_artifact",
        "dynamic_ready_nominal_receipt",
    ):
        normalized, _path = _B._verify_tracked_file(
            checkout, commit, _pin(row[key], name="lineage.%s" % key),
            name="A225 %s" % key,
        )
        pins[key] = normalized
    return {
        **expected,
        "action_id": action_id,
        "teacher_id": teacher_id,
        "seed": seed,
        **pins,
        "artifact": pin,
        "lineage_sha256": pin["sha256"],
    }


def _arm_contract(arm_id: str) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise LaunchRefused("arm_id must select one of the four code-owned A225 arms")
    payload = {
        "schema_version": 1,
        "kind": "action_ball_a225_learnability_arm_v1",
        "arm_id": arm_id,
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "fresh_normalizers_required": True,
        "init_noise_std": 0.02,
        "noise_std_type": "log",
        "entropy_coef": 0.01,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        **json.loads(json.dumps(ARMS[arm_id])),
    }
    return {**payload, "arm_contract_sha256": canonical_sha256(payload)}


def _planned_materialization(
    *, arm: Mapping[str, Any], lineage: Mapping[str, Any]
) -> dict[str, Any]:
    """Materialize the code-owned reward/policy identities inside the plan."""

    reward = {
        "soft_weights": arm["soft_weights"],
        "reference_guard_mode": arm["reference_guard_mode"],
        "weight_independent_projection_exposure_required": True,
    }
    policy = {
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "actor_normalizer_identity": ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "init_noise_std": arm["init_noise_std"],
        "noise_std_type": arm["noise_std_type"],
        "entropy_coef": arm["entropy_coef"],
        "actor_hidden_dims": arm["actor_hidden_dims"],
        "critic_hidden_dims": arm["critic_hidden_dims"],
        "ppo": arm["ppo"],
    }
    unsigned = {
        "schema_version": 1,
        "kind": MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
        "reward_contract_sha256": canonical_sha256(reward),
        "policy_contract_sha256": canonical_sha256(policy),
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _validated_stage_result(
    value: Any, *, expected_stage: str, name: str
) -> tuple[dict[str, str], dict[str, Any]]:
    pin, row = _canonical_external_json(value, name=name)
    row = _exact_dict(
        row,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "accepted",
            "launch_claim_sha256",
            "stage",
            "namespace",
            "completion",
            "gpu_admission",
            "output_contract",
            "arm_materialization",
            "oracle32_receipt",
            "predecessor_result",
            "content_sha256",
        ),
        name=name,
    )
    unsigned = dict(row)
    seal = unsigned.pop("content_sha256")
    if (
        row["schema_version"] != 1
        or row["kind"] != RESULT_KIND
        or row["diagnostic_unauthorized"] is not True
        or row["accepted"] is not True
        or row["stage"] != expected_stage
        or _B._sha256(row["launch_claim_sha256"], name="%s claim SHA" % name)
        != row["launch_claim_sha256"]
        or type(row["namespace"]) is not str
        or not row["namespace"]
        or _B._sha256(seal, name="%s content SHA" % name)
        != canonical_sha256(unsigned)
    ):
        raise LaunchRefused("%s identity differs" % name)
    return pin, row


def _validate_materialization(value: Any, *, arm: Mapping[str, Any], lineage: Mapping[str, Any]) -> dict:
    pin, result = _validated_stage_result(
        value, expected_stage="materialize", name="A225 materialize result"
    )
    row = result["arm_materialization"]
    row = _exact_dict(
        row,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "arm_id",
            "lineage_sha256",
            "arm_contract_sha256",
            "reward_contract_sha256",
            "policy_contract_sha256",
            "actor_contract",
            "actor_width",
            "critic_contract",
            "critic_width",
            "content_sha256",
        ),
        name="A225 arm materialization",
    )
    _assert_no_retired_contract(row, name="A225 arm materialization")
    unsigned = dict(row)
    seal = unsigned.pop("content_sha256")
    if _B._sha256(seal, name="materialization content SHA") != canonical_sha256(unsigned):
        raise LaunchRefused("A225 arm materialization content seal differs")
    expected = {
        "schema_version": 1,
        "kind": MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
    }
    if any(row[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("A225 arm materialization binding differs")
    reward_sha = _B._sha256(row["reward_contract_sha256"], name="reward contract SHA")
    policy_sha = _B._sha256(row["policy_contract_sha256"], name="policy contract SHA")
    return {
        "materialize_result": pin,
        **expected,
        "reward_contract_sha256": reward_sha,
        "policy_contract_sha256": policy_sha,
        "content_sha256": seal,
    }


def _validate_oracle32(value: Any, *, arm: Mapping[str, Any], lineage: Mapping[str, Any], materialization: Mapping[str, Any]) -> dict:
    pin, result = _validated_stage_result(
        value, expected_stage="oracle32", name="A225 oracle32 result"
    )
    row = result["oracle32_receipt"]
    row = _exact_dict(
        row,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "verdict",
            "episodes",
            "arm_id",
            "lineage_sha256",
            "arm_contract_sha256",
            "reward_contract_sha256",
            "runtime_effective_reward_sha256",
            "policy_contract_sha256",
            "runtime_policy_recipe_sha256",
            "actor_contract",
            "actor_width",
            "critic_contract",
            "critic_width",
            "seed",
            "raw_oracle_sha256",
            "content_sha256",
        ),
        name="A225 oracle32 receipt",
    )
    _assert_no_retired_contract(row, name="A225 oracle32 receipt")
    unsigned = dict(row)
    seal = unsigned.pop("content_sha256")
    if _B._sha256(seal, name="oracle32 content SHA") != canonical_sha256(unsigned):
        raise LaunchRefused("A225 oracle32 content seal differs")
    expected = {
        "schema_version": 1,
        "kind": ORACLE32_KIND,
        "diagnostic_unauthorized": True,
        "verdict": "PASS",
        "episodes": 32,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
        "reward_contract_sha256": materialization["reward_contract_sha256"],
        "runtime_effective_reward_sha256": _B._sha256(
            row["runtime_effective_reward_sha256"],
            name="runtime effective reward SHA",
        ),
        "policy_contract_sha256": materialization["policy_contract_sha256"],
        "runtime_policy_recipe_sha256": _B._sha256(
            row["runtime_policy_recipe_sha256"],
            name="runtime policy recipe SHA",
        ),
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "seed": lineage["seed"],
        "raw_oracle_sha256": _B._sha256(
            row["raw_oracle_sha256"], name="raw oracle SHA"
        ),
    }
    if any(row[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("A225 oracle32 receipt binding differs")
    return {"oracle32_result": pin, **expected, "content_sha256": seal}


def _validate_predecessor_result(
    value: Any,
    *,
    expected_stage: str,
    materialization: Mapping[str, Any],
    oracle32: Mapping[str, Any],
) -> dict[str, Any]:
    pin, result = _validated_stage_result(
        value,
        expected_stage=expected_stage,
        name="A225 %s predecessor result" % expected_stage,
    )
    expected_materialization_sha = materialization["content_sha256"]
    expected_oracle_sha = oracle32["content_sha256"]
    if (
        not isinstance(result["arm_materialization"], dict)
        or result["arm_materialization"].get("content_sha256")
        != expected_materialization_sha
        or not isinstance(result["oracle32_receipt"], dict)
        or result["oracle32_receipt"].get("content_sha256") != expected_oracle_sha
    ):
        raise LaunchRefused("A225 predecessor arm/oracle lineage differs")
    return {
        "artifact": pin,
        "stage": expected_stage,
        "launch_claim_sha256": result["launch_claim_sha256"],
        "arm_materialization_content_sha256": expected_materialization_sha,
        "oracle32_content_sha256": expected_oracle_sha,
    }


def _validate_gpu(value: Any, *, allow_colocation: bool) -> dict[str, Any]:
    row = _exact_dict(
        value,
        ("index", "uuid", "owner", "lock_path", "require_empty"),
        name="spec.gpu",
    )
    index = _B._plain_int(row["index"], name="spec.gpu.index", maximum=31)
    uuid = row["uuid"]
    if (
        type(uuid) is not str
        or not uuid.startswith("GPU-")
        or len(uuid) < 8
        or "," in uuid
        or "\n" in uuid
    ):
        raise LaunchRefused("spec.gpu.uuid must be an explicit GPU UUID")
    owner = row["owner"]
    if (
        type(owner) is not str
        or owner != owner.strip()
        or not owner
        or owner.lower()
        in {"codex", "claude", "fable", "agent", "unassigned"}
    ):
        raise LaunchRefused("spec.gpu.owner must be an explicit human name")
    lock_path = _B._absolute_path(row["lock_path"], name="spec.gpu.lock_path")
    expected_lock = Path("/tmp/hope_lean_queue_gpu%d.lock" % index)
    if lock_path != expected_lock:
        raise LaunchRefused("spec.gpu.lock_path must be %s" % expected_lock)
    expected_empty = not allow_colocation
    if row["require_empty"] is not expected_empty:
        raise LaunchRefused(
            "spec.gpu.require_empty must be %s when allow_vendor_v2_colocation=%s"
            % (str(expected_empty).lower(), str(allow_colocation).lower())
        )
    return {
        "index": index,
        "uuid": uuid,
        "owner": owner,
        "lock_path": str(lock_path),
        "require_empty": expected_empty,
    }


def _isaac_python_entry(value: Any) -> Path:
    entry = _B._absolute_path(value, name="source.isaac_python", must_exist=True)
    try:
        real = entry.resolve(strict=True)
        info = real.stat()
    except OSError as exc:
        raise LaunchRefused("source.isaac_python cannot resolve to a real file") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(real, os.X_OK):
        raise LaunchRefused(
            "source.isaac_python must resolve to an executable regular file"
        )
    return entry


def _validate_spec(document: Any, *, claimed: bool = False) -> dict[str, Any]:
    keys = (
            "schema_version",
            "kind",
            "source",
            "arm_id",
            "lineage",
            "arm_materialization",
            "oracle32_receipt",
            "predecessor_result",
            "stage",
            "num_envs",
            "max_iterations",
            "save_interval",
            "gpu",
            "namespace",
            "log_path",
    )
    actual = frozenset(document) if type(document) is dict else frozenset()
    required = frozenset(keys)
    optional = frozenset((COLOCATION_SPEC_KEY,))
    if not required.issubset(actual) or not actual.issubset(required | optional):
        raise LaunchRefused(
            "A225 launch spec keys differ: missing=%s extra=%s"
            % (sorted(required - actual), sorted(actual - required - optional))
        )
    row = dict(document)
    allow_colocation = row.get(COLOCATION_SPEC_KEY, False)
    if type(allow_colocation) is not bool:
        raise LaunchRefused("allow_vendor_v2_colocation must be a boolean")
    _assert_no_retired_contract(row, name="A225 launch spec")
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != SPEC_KIND:
        raise LaunchRefused("A225 launch spec schema/kind differs")
    source = _exact_dict(
        row["source"], ("checkout", "commit_sha", "isaac_python"), name="spec.source"
    )
    checkout = _B._absolute_path(source["checkout"], name="source.checkout", must_exist=True)
    commit = source["commit_sha"]
    if type(commit) is not str or _B.COMMIT_RE.fullmatch(commit) is None:
        raise LaunchRefused("source.commit_sha must be exact lowercase 40-hex")
    python = _isaac_python_entry(source["isaac_python"])
    arm = _arm_contract(row["arm_id"])
    stage = row["stage"]
    if stage not in BUDGETS:
        raise LaunchRefused("stage must be materialize, oracle32, smoke, probe512, long512, or scale4096")
    actual_budget = (
        _B._plain_int(row["num_envs"], name="num_envs", minimum=1),
        _B._plain_int(row["max_iterations"], name="max_iterations", minimum=0),
        _B._plain_int(row["save_interval"], name="save_interval", minimum=1),
    )
    if actual_budget != BUDGETS[stage]:
        raise LaunchRefused("%s budget must be exactly %s" % (stage, BUDGETS[stage]))
    namespace = _B._absolute_path(row["namespace"], name="namespace")
    if SAFE_COMPONENT.fullmatch(namespace.name or "") is None:
        raise LaunchRefused("namespace basename is unsafe")
    if os.path.lexists(namespace):
        if not claimed:
            raise LaunchRefused("namespace already exists and is permanently spent")
        info = namespace.lstat()
        if not stat.S_ISDIR(info.st_mode) or namespace.resolve(strict=True) != namespace:
            raise LaunchRefused("claimed namespace is not a real directory")
    elif claimed:
        raise LaunchRefused("claimed namespace vanished")
    parent = namespace.parent
    if not parent.exists() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise LaunchRefused("namespace parent must be an existing real directory")
    if parent.name != EXPERIMENT_NAME:
        raise LaunchRefused("namespace parent must be the dedicated A225 experiment root")
    log_path = _B._absolute_path(row["log_path"], name="log_path")
    if log_path != namespace / "run.log":
        raise LaunchRefused("log_path must equal <namespace>/run.log")
    if stage == "materialize":
        if row["arm_materialization"] is not None or row["oracle32_receipt"] is not None:
            raise LaunchRefused("materialize stage must start without generated receipts")
    elif row["arm_materialization"] is None:
        raise LaunchRefused("stage requires its same-arm materialization receipt")
    if stage in ("smoke", "probe512", "long512"):
        if row["oracle32_receipt"] is None:
            raise LaunchRefused(
                "%s requires its same-arm oracle32 PASS receipt" % stage
            )
    elif row["oracle32_receipt"] is not None:
        raise LaunchRefused(
            "only smoke, probe512, and long512 consume an oracle32 receipt"
        )
    expected_predecessor = {
        "probe512": "smoke",
        "long512": "probe512",
    }.get(stage)
    if expected_predecessor is None:
        if row["predecessor_result"] is not None:
            raise LaunchRefused("stage must not consume a predecessor result")
    elif row["predecessor_result"] is None:
        raise LaunchRefused(
            "%s requires a completed %s result" % (stage, expected_predecessor)
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {"checkout": str(checkout), "commit_sha": commit, "isaac_python": str(python)},
        "arm_id": arm["arm_id"],
        "lineage": _pin(row["lineage"], name="spec.lineage"),
        "arm_materialization": row["arm_materialization"],
        "oracle32_receipt": row["oracle32_receipt"],
        "predecessor_result": row["predecessor_result"],
        "stage": stage,
        "num_envs": actual_budget[0],
        "max_iterations": actual_budget[1],
        "save_interval": actual_budget[2],
        "gpu": _validate_gpu(row["gpu"], allow_colocation=allow_colocation),
        "namespace": str(namespace),
        "log_path": str(log_path),
        COLOCATION_SPEC_KEY: allow_colocation,
    }


def _runtime_sources(checkout: Path, commit: str) -> dict[str, dict[str, str]]:
    output = {}
    for relative, label in RUNTIME_SOURCE_PATHS:
        normalized, _path = _B._verify_tracked_file(
            checkout,
            commit,
            {"path": relative, "sha256": _B.sha256_file(checkout / relative)},
            name=label,
        )
        output[label] = normalized
    return output


def _training_argv(spec: Mapping[str, Any], lineage: Mapping[str, Any], arm: Mapping[str, Any]) -> list[str]:
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / _B.WBT_RELATIVE
    motion = checkout / lineage["motion"]["path"]
    tape = checkout / lineage["immutable_tape"]["path"]
    manifest = checkout / lineage["action_manifest"]["path"]
    dynamic_ready = checkout / lineage["dynamic_ready_artifact"]["path"]
    dynamic_receipt = checkout / lineage["dynamic_ready_nominal_receipt"]["path"]
    action_list = json.dumps([lineage["action_id"]], separators=(",", ":"))
    motion_list = json.dumps([str(motion)], separators=(",", ":"))
    ppo = arm["ppo"]
    weights = arm["soft_weights"]
    materialization = _planned_materialization(arm=arm, lineage=lineage)
    argv = [
        spec["source"]["isaac_python"],
        str(wbt / "scripts/train.py"),
        "task=%s" % TASK_PROFILE_ID,
        "algo=ppo",
        "headless=true",
        "logger=tensorboard",
        "video=false",
        "device=cuda:0",
        "seed=%d" % lineage["seed"],
        "num_envs=%d" % spec["num_envs"],
        "max_iterations=%d" % spec["max_iterations"],
        "algo.runner.save_interval=%d" % spec["save_interval"],
        "algo.runner.empirical_normalization=true",
        "algo.policy.actor_hidden_dims=[512,256,128]",
        "algo.policy.critic_hidden_dims=[512,256,128]",
        "algo.policy.init_noise_std=0.02",
        "algo.policy.noise_std_type=log",
        "algo.algorithm.entropy_coef=0.01",
        "algo.algorithm.schedule=%s" % ppo["schedule"],
        "algo.algorithm.learning_rate=%s" % format(ppo["learning_rate"], ".12g"),
        "algo.algorithm.desired_kl=0.01",
        "algo.algorithm.clip_param=0.2",
        "algo.algorithm.num_learning_epochs=5",
        "algo.algorithm.num_mini_batches=4",
        "run_name=%s-DIAGNOSTIC_UNAUTHORIZED" % Path(spec["namespace"]).name,
        "task.experiment_name=%s" % EXPERIMENT_NAME,
        "task.gym_task=%s" % GYM_TASK_ID,
        "task.actor_obs_contract=%s" % ACTOR_CONTRACT,
        "action_ball_dynamic_ready_bootstrap=true",
        "action_ball_dynamic_ready_artifact_path=%s" % dynamic_ready,
        "action_ball_dynamic_ready_artifact_sha256=%s"
        % lineage["dynamic_ready_artifact"]["sha256"],
        "action_ball_dynamic_ready_nominal_receipt_path=%s" % dynamic_receipt,
        "action_ball_dynamic_ready_nominal_receipt_sha256=%s"
        % lineage["dynamic_ready_nominal_receipt"]["sha256"],
        "motion_file=%s" % motion_list,
        "task.racket.clip_names=%s" % action_list,
        "task.racket.action_ball_manifest_path=%s" % manifest,
        "task.racket.action_ball_manifest_sha256=%s"
        % lineage["action_manifest"]["sha256"],
        "task.racket.action_ball_policy_contract_sha256=%s"
        % materialization["policy_contract_sha256"],
        "task.racket.action_ball_seed=%d" % lineage["seed"],
        "task.racket.action_ball_target_source=immutable_tape",
        "task.racket.action_ball_immutable_tape_path=%s" % tape,
        "task.racket.action_ball_immutable_tape_sha256=%s" % lineage["immutable_tape"]["sha256"],
        "task.racket.action_ball_diagnostic_unauthorized=true",
        "+task.racket.reference_guard_mode=%s" % arm["reference_guard_mode"],
        "task.rewards.death_penalty_weight=%s" % weights["death_penalty"],
        "task.rewards.qdes_limit_barrier_weight=%s" % weights["qdes_limit"],
        "+task.rewards.qdes_projection_penalty_weight=%s" % weights["qdes_projection"],
        "task.rewards.joint_limit_weight=%s" % weights["joint_limit"],
        "task.actions.control_step_action_delay_min=0",
        "task.actions.control_step_action_delay_max=0",
        "task.push.enable=false",
        "task.physical_ball=false",
        "task.racket.virtual_ball=true",
        "task.racket.action_ball_target_observation_noise=false",
        "task.racket.question_bank=",
        "task.racket.cq_anchor_bank=",
        "task.racket.exam_bank=",
    ]
    if spec["stage"] == "oracle32":
        argv.extend(
            [
                "+action_ball_teacher_qdes_oracle_output_path=%s"
                % (Path(spec["namespace"]) / "teacher_qdes_oracle_32ep.json"),
                "+action_ball_teacher_qdes_oracle_episodes=32",
            ]
        )
    return argv


def _output_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    stage = spec["stage"]
    output = {
        "ppo_update_count": spec["max_iterations"],
        "finite_model_save_interval": spec["save_interval"],
        "arm_materialization_embedded_in_claim": stage == "materialize",
        "teacher_qdes_oracle32": None,
        "boot_marker": "Learning iteration",
        "iter500_quantitative_threshold_status": "UNSET",
        "iter500_action": "diagnostic_continue_only",
        "automatic_winner_selection_prohibited": True,
        "speed_benchmark_eligible": not spec[COLOCATION_SPEC_KEY],
        "colocation_result_scope": (
            "training_diagnostic_only"
            if spec[COLOCATION_SPEC_KEY]
            else "training_and_speed_benchmark_diagnostic"
        ),
    }
    if stage == "materialize":
        output["boot_marker"] = "ACTION_BALL_A225_TRAINABILITY_PREFLIGHT_JSON"
    elif stage == "oracle32":
        output["teacher_qdes_oracle32"] = str(
            Path(spec["namespace"]) / "teacher_qdes_oracle_32ep.json"
        )
        output["boot_marker"] = "ACTION_BALL_TEACHER_QDES_ORACLE_COMPLETE_JSON"
    return output


def _normalizer_contract() -> dict[str, Any]:
    return {
        "actor": {"identity": ACTOR_NORMALIZER_IDENTITY, "state": "fresh_empty"},
        "critic": {"identity": CRITIC_NORMALIZER_IDENTITY, "state": "fresh_empty"},
        "distinct_objects_required": True,
    }


def _termination_contract() -> dict[str, Any]:
    return {
        "hard_union": list(HARD_TERMINATION_UNION),
        "single_stroke_terminal": "action_ball_single_stroke_complete",
        "finite_horizon_terminal": "time_out",
    }


def _continuation_stop_gate() -> dict[str, Any]:
    return {
        "exact_stage_budget_required": True,
        "hard_termination_count_max": 0,
        "table_contact_count_max": 0,
        "nonfinite_count_max": 0,
        "finite_model_required_when_updates_positive": True,
        "oracle32_pass_required_for_training_stages": True,
        "scale4096_required_for_long": False,
        "iter500_quantitative_threshold_status": "UNSET",
        "iter500_action": "diagnostic_continue_only",
        "automatic_winner_selection_prohibited": True,
    }


def _admission_training_argv(
    spec: Mapping[str, Any], bundle: Mapping[str, Any]
) -> list[str]:
    row = _exact_dict(
        bundle,
        (
            "lineage",
            "arm",
            "normalizers",
            "termination_contract",
            "continuation_stop_gate",
        ),
        name="A225 claim bundle",
    )
    return _training_argv(spec, row["lineage"], row["arm"])


_ADMISSION = _A.VendorV2GPUAdmission(
    base=_B,
    schema_version=SCHEMA_VERSION,
    claim_kind=CLAIM_KIND,
    experiment_name=EXPERIMENT_NAME,
    colocation_spec_key=COLOCATION_SPEC_KEY,
    physical_ball_semantics=PHYSICAL_BALL_SEMANTICS,
    runtime_source_paths=RUNTIME_SOURCE_PATHS,
    launcher_source=LAUNCHER_SOURCE,
    admission_source=ADMISSION_SOURCE,
    exact_group_source=EXACT_GROUP_SOURCE,
    exact_group=_G,
    canonical_sha256=canonical_sha256,
    exact_dict=_exact_dict,
    validate_spec=_validate_spec,
    output_contract=_output_contract,
    training_argv=_admission_training_argv,
)
_open_gpu_shared_lock = _ADMISSION._open_gpu_shared_lock
_lock_gpu_admission = _ADMISSION._lock_gpu_admission
_unlock_gpu_admission = _ADMISSION._unlock_gpu_admission
_query_gpu_processes = _ADMISSION._query_gpu_processes
_validate_runtime_gpu_process = _ADMISSION._validate_runtime_gpu_process
_live_reservations = _ADMISSION._live_reservations
_reservation_document = _ADMISSION._reservation_document
_runtime_namespace_receipt = _ADMISSION._runtime_namespace_receipt
_cleanup_post_boot_admission_failure = (
    _ADMISSION._cleanup_post_boot_admission_failure
)


def _verify_gpu_admission(
    spec: Mapping[str, Any],
    *,
    phase: str,
    current_namespace: Path | None,
    require_current_compute: bool = False,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    return _ADMISSION._verify_gpu_admission(
        spec,
        phase=phase,
        current_namespace=current_namespace,
        require_current_compute=require_current_compute,
        proc_root=proc_root,
        query_gpu_processes=_query_gpu_processes,
        validate_runtime_gpu_process=_validate_runtime_gpu_process,
        live_reservations=_live_reservations,
    )


def _validate_raw_oracle32(
    path: Path,
    *,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the trainer's generic oracle and distill an A225-bound receipt."""

    _B._stable_regular_file(path, name="A225 raw oracle32")
    raw = path.read_bytes()
    row = _B._strict_json_bytes(raw, name="A225 raw oracle32")
    if raw != _B._canonical_bytes(row) + b"\n":
        raise LaunchRefused("A225 raw oracle32 must be canonical JSON plus newline")
    row = _exact_dict(
        row,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "bindings",
            "completion",
            "phase_by_termination",
            "exact_strike",
            "capture_rejection",
            "measurement_contract",
            "safety_exposure",
            "teacher_qdes",
            "episodes",
        ),
        name="A225 raw oracle32",
    )
    if (
        row["schema_version"] != 2
        or row["kind"] != "action_ball_teacher_qdes_dynamic_oracle_v2"
        or row["diagnostic_unauthorized"] is not True
    ):
        raise LaunchRefused("A225 raw oracle32 schema/kind/authorization differs")
    bindings = _exact_dict(
        row["bindings"],
        (
            "source_sha256",
            "task_sha256",
            "hard_contract_sha256",
            "reward_sha256",
            "policy_sha256",
            "policy_contract_sha256",
            "dynamic_ready_sha256",
            "dynamic_ready_artifact_sha256",
            "dynamic_ready_nominal_hold_sha256",
            "manifest_sha256",
            "motion_sha256",
            "tape_file_sha256",
            "tape_canonical_sha256",
            "tape_base_question_sha256",
            "tape_target_producer_sha256",
            "tape_target_column_sha256",
        ),
        name="A225 raw oracle32 bindings",
    )
    for name, digest in bindings.items():
        _B._sha256(digest, name="A225 raw oracle32 %s" % name)
    bundle = claim["bundle"]
    lineage = bundle["lineage"]
    arm = bundle["arm"]
    materialization = claim["materialization_inputs"]["arm_materialization"]
    sources = claim["runtime_sources"]
    expected_bindings = {
        "source_sha256": sources["training entrypoint"]["sha256"],
        "task_sha256": sources["A225 task profile"]["sha256"],
        "policy_contract_sha256": materialization["policy_contract_sha256"],
        "dynamic_ready_artifact_sha256": lineage["dynamic_ready_artifact"]["sha256"],
        "dynamic_ready_nominal_hold_sha256": lineage[
            "dynamic_ready_nominal_receipt"
        ]["sha256"],
        "manifest_sha256": lineage["action_manifest"]["sha256"],
        "motion_sha256": lineage["motion"]["sha256"],
        "tape_file_sha256": lineage["immutable_tape"]["sha256"],
    }
    if any(bindings[key] != value for key, value in expected_bindings.items()):
        raise LaunchRefused("A225 raw oracle32 lineage bindings differ")
    spec = claim["spec"]
    checkout = Path(spec["source"]["checkout"])
    root = checkout / _B.WBT_RELATIVE / "logs/rsl_rl" / EXPERIMENT_NAME
    suffix = "_%s-DIAGNOSTIC_UNAUTHORIZED" % Path(spec["namespace"]).name
    candidates = (
        []
        if not root.is_dir()
        else [candidate for candidate in root.iterdir() if candidate.name.endswith(suffix)]
    )
    if len(candidates) != 1:
        raise LaunchRefused("A225 raw oracle32 has no unique runtime directory")
    hard_contract = candidates[0] / "params/training_contract.json"
    _B._stable_regular_file(hard_contract, name="A225 oracle32 hard contract")
    if _B.sha256_file(hard_contract) != bindings["hard_contract_sha256"]:
        raise LaunchRefused("A225 oracle32 hard-contract SHA differs")
    try:
        hard_document = _B._strict_json_bytes(
            hard_contract.read_bytes(), name="A225 oracle32 hard contract"
        )
        training_contract = _OLD._load_training_contract_module(checkout)
        training_contract.validate_schema3_contract_structure(hard_document)
        diagnostic = training_contract.validate_action_ball_training_authorization(
            hard_document
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise LaunchRefused(
            "A225 oracle32 hard contract is not an authorized schema-3 diagnostic"
        ) from exc
    expected_hard_identity = {
        "schema_version": 3,
        "target_mode": "action_ball",
        "actor_obs_contract": ACTOR_CONTRACT,
        "actor_obs_total_dim": ACTOR_WIDTH,
        "critic_obs_contract": CRITIC_CONTRACT,
        "critic_obs_total_dim": CRITIC_WIDTH,
        "actor_obs_normalizer_identity": ACTOR_NORMALIZER_IDENTITY,
        "critic_obs_normalizer_identity": CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
    }
    if diagnostic is not True or any(
        hard_document.get(key) != value
        for key, value in expected_hard_identity.items()
    ):
        raise LaunchRefused("A225 oracle32 hard-contract ABI/authorization differs")
    runtime_reward = hard_document.get("effective_reward_recipe")
    runtime_policy = hard_document.get("action_ball_ppo_runner_recipe")
    if (
        not isinstance(runtime_reward, dict)
        or runtime_reward.get("sha256") != bindings["reward_sha256"]
        or not isinstance(runtime_reward.get("terms"), list)
        or not isinstance(runtime_policy, dict)
        or runtime_policy.get("sha256") != bindings["policy_sha256"]
        or not isinstance(runtime_policy.get("recipe"), dict)
    ):
        raise LaunchRefused(
            "A225 oracle32 runtime reward/policy receipt differs from hard contract"
        )
    reward_terms = runtime_reward["terms"]
    reward_weights = {}
    for term in reward_terms:
        if not isinstance(term, dict) or type(term.get("name")) is not str:
            raise LaunchRefused("A225 runtime reward term is malformed")
        name = term["name"]
        if name in reward_weights:
            raise LaunchRefused("A225 runtime reward term name is duplicated")
        reward_weights[name] = term.get("weight")
    expected_weights = {
        "death_penalty": arm["soft_weights"]["death_penalty"],
        "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
        "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
        "joint_limit": arm["soft_weights"]["joint_limit"],
    }
    if any(reward_weights.get(name) != weight for name, weight in expected_weights.items()):
        raise LaunchRefused("A225 runtime effective reward weights differ from arm")
    policy_recipe = runtime_policy["recipe"]
    algorithm = policy_recipe.get("algorithm")
    policy = policy_recipe.get("policy")
    ppo = arm["ppo"]
    expected_algorithm = {
        "schedule": ppo["schedule"],
        "learning_rate": ppo["learning_rate"],
        "desired_kl": 0.01,
        "clip_param": 0.2,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "entropy_coef": arm["entropy_coef"],
    }
    expected_policy = {
        "actor_hidden_dims": arm["actor_hidden_dims"],
        "critic_hidden_dims": arm["critic_hidden_dims"],
        "init_noise_std": arm["init_noise_std"],
        "noise_std_type": arm["noise_std_type"],
    }
    if (
        not isinstance(algorithm, dict)
        or not isinstance(policy, dict)
        or any(algorithm.get(name) != value for name, value in expected_algorithm.items())
        or any(policy.get(name) != value for name, value in expected_policy.items())
    ):
        raise LaunchRefused("A225 runtime PPO recipe differs from arm")
    completion = row["completion"]
    exact = row["exact_strike"]
    capture = row["capture_rejection"]
    safety = row["safety_exposure"]
    teacher_qdes = row["teacher_qdes"]
    try:
        failures = _OLD._oracle32_acceptance_failures(
            completion=completion,
            observed=completion["exact_strike_observed_nonterminal"],
            exact_summary=exact,
            capture=capture,
            unknown=completion["pre_strike_or_same_step_unknown"],
            termination=safety["termination"],
            projection=safety["projection"],
            qdes=teacher_qdes,
            soft_limit=safety["soft_limit"],
            reference=safety["reference_guard"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchRefused("A225 raw oracle32 ledger is malformed") from exc
    reference = safety["reference_guard"]
    if arm["reference_guard_mode"] == "phase_gated":
        failures = [item for item in failures if item != "reference_exposure_denominator"]
        if reference.get("mode") != "phase_gated":
            failures.append("reference_guard_mode")
        if reference.get("available") is True and reference.get("sample_count") != completion.get(
            "control_steps"
        ):
            failures.append("reference_exposure_denominator")
    if failures:
        raise LaunchRefused(
            "A225 oracle32 acceptance failed: %s" % ",".join(failures)
        )
    unsigned = {
        "schema_version": 1,
        "kind": ORACLE32_KIND,
        "diagnostic_unauthorized": True,
        "verdict": "PASS",
        "episodes": 32,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
        "reward_contract_sha256": materialization["reward_contract_sha256"],
        "runtime_effective_reward_sha256": bindings["reward_sha256"],
        "policy_contract_sha256": materialization["policy_contract_sha256"],
        "runtime_policy_recipe_sha256": bindings["policy_sha256"],
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "seed": lineage["seed"],
        "raw_oracle_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def build_plan(spec_path: Path) -> dict[str, Any]:
    path = _B._absolute_path(str(spec_path), name="--spec", must_exist=True)
    _B._stable_regular_file(path, name="A225 launch spec")
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name="A225 launch spec")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("A225 launch spec must be canonical JSON plus newline")
    spec = _validate_spec(document)
    checkout = Path(spec["source"]["checkout"])
    commit = spec["source"]["commit_sha"]
    source = _B._verify_clean_source(checkout, commit)
    runtime_sources = _runtime_sources(checkout, commit)
    runtime_assets = _B._validate_runtime_asset_environment()
    lineage = _validate_lineage(checkout, commit, spec["lineage"])
    arm = _arm_contract(spec["arm_id"])
    materialization = (
        _planned_materialization(arm=arm, lineage=lineage)
        if spec["stage"] == "materialize"
        else _validate_materialization(
            spec["arm_materialization"], arm=arm, lineage=lineage
        )
    )
    oracle32 = (
        _validate_oracle32(
            spec["oracle32_receipt"],
            arm=arm,
            lineage=lineage,
            materialization=materialization,
        )
        if spec["stage"] in ("smoke", "probe512", "long512")
        else None
    )
    expected_predecessor = {
        "probe512": "smoke",
        "long512": "probe512",
    }.get(spec["stage"])
    predecessor = (
        _validate_predecessor_result(
            spec["predecessor_result"],
            expected_stage=expected_predecessor,
            materialization=materialization,
            oracle32=oracle32,
        )
        if expected_predecessor is not None
        else None
    )
    output_contract = _output_contract(spec)
    bundle = {
        "lineage": lineage,
        "arm": arm,
        "normalizers": _normalizer_contract(),
        "termination_contract": _termination_contract(),
        "continuation_stop_gate": _continuation_stop_gate(),
    }
    materialization_inputs = {
        "arm_materialization": materialization,
        "oracle32_receipt": oracle32,
        "predecessor_result": predecessor,
    }
    training_argv = _training_argv(spec, lineage, arm)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "diagnostic_unauthorized": True,
        "formal_evidence_prohibited": True,
        "promotion_prohibited": True,
        "resume_prohibited": True,
        "export_prohibited": True,
        "deployment_prohibited": True,
        "hardware_prohibited": True,
        "single_gpu": True,
        "max_compute_pids_on_physical_gpu": _A.MAX_VENDOR_V2_COMPUTE_PIDS,
        "minimum_free_memory_mib": _A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
        "gpu_default_empty": not spec[COLOCATION_SPEC_KEY],
        "vendor_v2_colocation_opt_in": spec[COLOCATION_SPEC_KEY],
        "fresh_only": True,
        "reward_materialization_only": spec["stage"] == "materialize",
        "policy_recipe_materialization_only": False,
        "teacher_qdes_oracle_only": spec["stage"] == "oracle32",
        "ppo_updates_authorized": output_contract["ppo_update_count"],
        "control_step_action_delay": 0,
        "reset_inverse_solve": False,
        "physical_ball_semantics": PHYSICAL_BALL_SEMANTICS,
        "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
        "source": source,
        "runtime_sources": runtime_sources,
        "runtime_assets": runtime_assets,
        "bundle": bundle,
        "materialization_inputs": materialization_inputs,
        "output_contract": output_contract,
        "boot_marker": output_contract["boot_marker"],
        "training_argv": training_argv,
    }
    retired_check = dict(payload)
    retired_check.pop("resume_prohibited")
    _assert_no_retired_contract(retired_check, name="A225 launch claim")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": canonical_sha256(payload),
        "canonical_payload": payload,
    }


def _revalidate_claim_payload(payload: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    spec = _validate_spec(payload["spec"], claimed=True)
    checkout = Path(spec["source"]["checkout"])
    commit = spec["source"]["commit_sha"]
    if _B._verify_clean_source(checkout, commit) != payload["source"]:
        raise LaunchRefused("clean source claim drifted")
    if _runtime_sources(checkout, commit) != payload["runtime_sources"]:
        raise LaunchRefused("runtime source identity drifted")
    _B._validate_runtime_asset_claim(payload["runtime_assets"])
    lineage = _validate_lineage(checkout, commit, spec["lineage"])
    arm = _arm_contract(spec["arm_id"])
    materialization = (
        _planned_materialization(arm=arm, lineage=lineage)
        if spec["stage"] == "materialize"
        else _validate_materialization(
            spec["arm_materialization"], arm=arm, lineage=lineage
        )
    )
    oracle32 = (
        _validate_oracle32(
            spec["oracle32_receipt"],
            arm=arm,
            lineage=lineage,
            materialization=materialization,
        )
        if spec["stage"] in ("smoke", "probe512", "long512")
        else None
    )
    predecessor_stage = {
        "probe512": "smoke",
        "long512": "probe512",
    }.get(spec["stage"])
    predecessor = (
        _validate_predecessor_result(
            spec["predecessor_result"],
            expected_stage=predecessor_stage,
            materialization=materialization,
            oracle32=oracle32,
        )
        if predecessor_stage is not None
        else None
    )
    expected_bundle = {
        "lineage": lineage,
        "arm": arm,
        "normalizers": _normalizer_contract(),
        "termination_contract": _termination_contract(),
        "continuation_stop_gate": _continuation_stop_gate(),
    }
    expected_inputs = {
        "arm_materialization": materialization,
        "oracle32_receipt": oracle32,
        "predecessor_result": predecessor,
    }
    if (
        payload["spec"] != spec
        or payload["bundle"] != expected_bundle
        or payload["materialization_inputs"] != expected_inputs
        or payload["output_contract"] != _output_contract(spec)
        or payload["boot_marker"] != payload["output_contract"]["boot_marker"]
        or payload["training_argv"] != _training_argv(spec, lineage, arm)
    ):
        raise LaunchRefused("A225 claim lineage, output contract, or training argv drifted")
    return spec, lineage, arm


def _internal_exec(claim_path: Path, claim_sha: str, lock_fd: int) -> int:
    path = _B._absolute_path(str(claim_path), name="internal claim", must_exist=True)
    raw = path.read_bytes()
    outer = _B._strict_json_bytes(raw, name="internal claim")
    if raw != _B._canonical_bytes(outer) + b"\n":
        raise LaunchRefused("internal claim is not canonical")
    outer = _exact_dict(
        outer,
        ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
        name="internal claim",
    )
    if (
        outer["schema_version"] != SCHEMA_VERSION
        or outer["kind"] != CLAIM_KIND
        or outer["launch_claim_sha256"] != claim_sha
        or canonical_sha256(outer["canonical_payload"]) != claim_sha
    ):
        raise LaunchRefused("internal claim digest differs")
    payload = outer["canonical_payload"]
    spec, _lineage, _arm = _revalidate_claim_payload(payload)
    lock_path = Path(spec["gpu"]["lock_path"])
    descriptor_info = os.fstat(lock_fd)
    path_info = lock_path.lstat()
    if (
        not stat.S_ISREG(descriptor_info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or (descriptor_info.st_dev, descriptor_info.st_ino)
        != (path_info.st_dev, path_info.st_ino)
    ):
        raise LaunchRefused("inherited GPU lock identity differs")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise LaunchRefused("inherited GPU lock is not held") from exc
    _lock_gpu_admission(lock_fd)
    try:
        gpu = _verify_gpu_admission(
            spec, phase="pre_exec", current_namespace=Path(spec["namespace"])
        )
        _B._write_exclusive_json(
            Path(spec["namespace"]) / "pre_exec_gpu_admission.json",
            {
                "schema_version": 1,
                "kind": "action_ball_a225_pre_exec_gpu_admission_v1",
                "launch_claim_sha256": claim_sha,
                "gpu": gpu,
            },
        )
        namespace_receipt, namespace_receipt_sha = _runtime_namespace_receipt(
            spec, claim_sha
        )
    finally:
        _unlock_gpu_admission(lock_fd)
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / _B.WBT_RELATIVE
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(wbt / "source/whole_body_tracking"),
        "CUDA_VISIBLE_DEVICES": str(spec["gpu"]["index"]),
        "HYDRA_FULL_ERROR": "1",
        "WANDB_MODE": "offline",
        "HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256": claim_sha,
        _A.GPU_NAMESPACE_RECEIPT_ENV: str(namespace_receipt),
        _A.GPU_NAMESPACE_RECEIPT_SHA_ENV: namespace_receipt_sha,
        **_B._runtime_asset_exec_environment(payload["runtime_assets"]),
    }
    os.chdir(wbt)
    argv = payload["training_argv"]
    os.execve(argv[0], argv, environment)
    raise AssertionError("execve returned")


def _validate_completion_state(path: Path) -> dict[str, str]:
    _B._stable_regular_file(path, name="A225 completion state")
    observed: dict[str, str] = {}
    required = {"completion_exit_code", "terminal_kind", "terminal_exit_code"}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in required:
            if key in observed:
                raise LaunchRefused("A225 completion state has duplicate %s" % key)
            observed[key] = value
    if observed != {
        "completion_exit_code": "0",
        "terminal_kind": "clean_completion",
        "terminal_exit_code": "0",
    }:
        raise LaunchRefused("A225 workload did not exit cleanly and uniquely")
    return observed


def _completion_stage(stage: str) -> bool:
    return stage in ("materialize", "oracle32", "smoke", "probe512")


def execute(plan: dict[str, Any], *, confirm_claim: str) -> dict[str, Any]:
    expected = _B._sha256(confirm_claim, name="--confirm-claim")
    if expected != plan["launch_claim_sha256"]:
        raise LaunchRefused("--confirm-claim differs from freshly recomputed plan")
    spec = plan["canonical_payload"]["spec"]
    if spec["stage"] == "scale4096":
        raise LaunchRefused("scale4096 is independently BLOCKED and cannot execute")
    checkout = Path(spec["source"]["checkout"])
    _B._verify_clean_source(checkout, spec["source"]["commit_sha"])
    _B._validate_runtime_asset_claim(plan["canonical_payload"]["runtime_assets"])
    lock_fd = _open_gpu_shared_lock(Path(spec["gpu"]["lock_path"]))
    namespace = None
    try:
        _lock_gpu_admission(lock_fd)
        try:
            first = _verify_gpu_admission(
                spec, phase="pre_launch", current_namespace=None
            )
            namespace = _B._claim_namespace(plan)
            _B._write_exclusive_json(
                namespace / _A.GPU_RESERVATION_FILENAME,
                _reservation_document(spec, expected),
            )
            _B._write_exclusive_json(
                namespace / "pre_launch_gpu_admission.json",
                {
                    "schema_version": 1,
                    "kind": "action_ball_a225_pre_launch_gpu_admission_v1",
                    "launch_claim_sha256": expected,
                    "gpu": first,
                },
            )
        finally:
            _unlock_gpu_admission(lock_fd)
        state = Path(spec["log_path"] + ".launch")
        internal = [
            spec["source"]["isaac_python"],
            str(checkout / LAUNCHER_SOURCE),
            "_exec",
            "--claim",
            str(namespace / "launch_claim.json"),
            "--claim-sha256",
            expected,
            "--gpu-lock-fd",
            str(lock_fd),
        ]
        completion_stage = _completion_stage(spec["stage"])
        environment = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "KIT_BOOT_MARKER": plan["canonical_payload"]["boot_marker"],
            "KIT_BOOT_TIMEOUT_S": "2700",
            "KIT_BOOT_STALE_TIMEOUT_S": "1800",
            "KIT_BOOT_POLL_S": "5",
            "KIT_BOOT_STATE_FILE": str(state),
            **(
                {
                    "KIT_WAIT_FOR_COMPLETION": "1",
                    "KIT_COMPLETION_TIMEOUT_S": "7200",
                }
                if completion_stage
                else {}
            ),
        }
        result = subprocess.run(
            [str(checkout / KIT_LAUNCHER_SOURCE), spec["log_path"], *internal],
            cwd=checkout / _B.WBT_RELATIVE,
            env=environment,
            pass_fds=(lock_fd,),
            check=False,
        )
        if result.returncode != 0:
            raise LaunchRefused(
                "locked Kit launcher returned %d; namespace remains spent"
                % result.returncode
            )
        completion = _validate_completion_state(state) if completion_stage else None
        _lock_gpu_admission(lock_fd)
        try:
            try:
                final_gpu = _verify_gpu_admission(
                    spec,
                    phase="post_completion" if completion_stage else "post_boot",
                    current_namespace=namespace,
                    require_current_compute=not completion_stage,
                )
                _B._write_exclusive_json(
                    namespace
                    / (
                        "post_completion_gpu_admission.json"
                        if completion_stage
                        else "post_boot_gpu_admission.json"
                    ),
                    {
                        "schema_version": 1,
                        "kind": (
                            "action_ball_a225_post_completion_gpu_admission_v1"
                            if completion_stage
                            else "action_ball_a225_post_boot_gpu_admission_v1"
                        ),
                        "launch_claim_sha256": expected,
                        "gpu": final_gpu,
                    },
                )
            except (LaunchRefused, FileNotFoundError, ValueError, OSError) as exc:
                if completion_stage:
                    raise LaunchRefused(
                        "post-completion admission refused after exact clean exit"
                    ) from exc
                failure = _cleanup_post_boot_admission_failure(
                    namespace, state, expected, str(exc)
                )
                outcome = (
                    "completed"
                    if failure["cleanup"]["completed"] is True
                    else "incomplete"
                )
                raise LaunchRefused(
                    "post-boot admission refused; exact current-trainer cleanup %s; "
                    "failure receipt=%s" % (outcome, failure["path"])
                ) from exc
        finally:
            _unlock_gpu_admission(lock_fd)
        materialization = plan["canonical_payload"]["materialization_inputs"][
            "arm_materialization"
        ]
        oracle32 = plan["canonical_payload"]["materialization_inputs"][
            "oracle32_receipt"
        ]
        if spec["stage"] == "oracle32":
            oracle32 = _validate_raw_oracle32(
                Path(plan["canonical_payload"]["output_contract"]["teacher_qdes_oracle32"]),
                claim=plan["canonical_payload"],
            )
        unsigned_result = {
            "schema_version": 1,
            "kind": RESULT_KIND,
            "diagnostic_unauthorized": True,
            "accepted": True,
            "launch_claim_sha256": expected,
            "stage": spec["stage"],
            "namespace": str(namespace),
            "completion": completion,
            "gpu_admission": final_gpu,
            "output_contract": plan["canonical_payload"]["output_contract"],
            "arm_materialization": materialization,
            "oracle32_receipt": oracle32,
            "predecessor_result": plan["canonical_payload"][
                "materialization_inputs"
            ]["predecessor_result"],
        }
        launch_result = {
            **unsigned_result,
            "content_sha256": canonical_sha256(unsigned_result),
        }
        _B._write_exclusive_json(namespace / "launch_result.json", launch_result)
        return launch_result
    finally:
        os.close(lock_fd)


def _write_template(args: argparse.Namespace) -> dict[str, Any]:
    budget = BUDGETS[args.stage]
    materialization_pair = (
        args.arm_materialization_path,
        args.arm_materialization_sha256,
    )
    oracle_pair = (args.oracle32_receipt_path, args.oracle32_receipt_sha256)
    predecessor_pair = (
        args.predecessor_result_path,
        args.predecessor_result_sha256,
    )
    if (materialization_pair[0] is None) != (materialization_pair[1] is None):
        raise LaunchRefused("arm materialization path/SHA must be supplied together")
    if (oracle_pair[0] is None) != (oracle_pair[1] is None):
        raise LaunchRefused("oracle32 receipt path/SHA must be supplied together")
    if (predecessor_pair[0] is None) != (predecessor_pair[1] is None):
        raise LaunchRefused("predecessor result path/SHA must be supplied together")
    if args.stage == "materialize":
        if (
            materialization_pair[0] is not None
            or oracle_pair[0] is not None
            or predecessor_pair[0] is not None
        ):
            raise LaunchRefused("materialize template accepts no generated receipt")
    elif materialization_pair[0] is None:
        raise LaunchRefused("stage requires an A225 materialize result path/SHA")
    if args.stage in ("smoke", "probe512", "long512"):
        if oracle_pair[0] is None:
            raise LaunchRefused(
                "%s template requires an oracle32 result path/SHA" % args.stage
            )
    elif oracle_pair[0] is not None:
        raise LaunchRefused(
            "only smoke, probe512, and long512 templates accept an oracle32 result"
        )
    needs_predecessor = args.stage in ("probe512", "long512")
    if needs_predecessor is not (predecessor_pair[0] is not None):
        raise LaunchRefused(
            "%s template predecessor-result requirement differs" % args.stage
        )
    namespace = Path(args.namespace).resolve(strict=False)
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(Path(args.checkout).resolve(strict=True)),
            "commit_sha": args.commit_sha,
            "isaac_python": str(_isaac_python_entry(args.isaac_python)),
        },
        "arm_id": args.arm_id,
        "lineage": {"path": args.lineage_path, "sha256": args.lineage_sha256},
        "arm_materialization": (
            None
            if args.arm_materialization_path is None
            else {
                "path": args.arm_materialization_path,
                "sha256": args.arm_materialization_sha256,
            }
        ),
        "oracle32_receipt": (
            None
            if args.oracle32_receipt_path is None
            else {
                "path": args.oracle32_receipt_path,
                "sha256": args.oracle32_receipt_sha256,
            }
        ),
        "predecessor_result": (
            None
            if args.predecessor_result_path is None
            else {
                "path": args.predecessor_result_path,
                "sha256": args.predecessor_result_sha256,
            }
        ),
        "stage": args.stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "gpu": {
            "index": args.gpu_index,
            "uuid": args.gpu_uuid,
            "owner": args.owner,
            "lock_path": "/tmp/hope_lean_queue_gpu%d.lock" % args.gpu_index,
            "require_empty": not args.allow_colocation,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }
    if args.allow_colocation:
        document[COLOCATION_SPEC_KEY] = True
    _assert_no_retired_contract(document, name="A225 launch spec")
    document = _validate_spec(document)
    if not args.allow_colocation:
        document.pop(COLOCATION_SPEC_KEY)
    output = Path(args.output).resolve(strict=False)
    _B._write_exclusive_json(output, document)
    return {"status": "CREATED", "spec": str(output), "arm_id": args.arm_id, "stage": args.stage}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--output", required=True)
    template.add_argument("--checkout", required=True)
    template.add_argument("--commit-sha", required=True)
    template.add_argument("--isaac-python", required=True)
    template.add_argument("--arm-id", required=True, choices=ARM_IDS)
    template.add_argument("--lineage-path", required=True)
    template.add_argument("--lineage-sha256", required=True)
    template.add_argument("--arm-materialization-path")
    template.add_argument("--arm-materialization-sha256")
    template.add_argument("--oracle32-receipt-path")
    template.add_argument("--oracle32-receipt-sha256")
    template.add_argument("--predecessor-result-path")
    template.add_argument("--predecessor-result-sha256")
    template.add_argument("--stage", required=True, choices=tuple(BUDGETS))
    template.add_argument("--gpu-index", required=True, type=int)
    template.add_argument("--gpu-uuid", required=True)
    template.add_argument("--owner", required=True)
    template.add_argument("--namespace", required=True)
    template.add_argument("--allow-colocation", action="store_true")
    for command in ("plan", "execute"):
        child = sub.add_parser(command)
        child.add_argument("--spec", required=True)
        if command == "execute":
            child.add_argument("--confirm-claim", required=True)
    internal = sub.add_parser("_exec", help=argparse.SUPPRESS)
    internal.add_argument("--claim", required=True)
    internal.add_argument("--claim-sha256", required=True)
    internal.add_argument("--gpu-lock-fd", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "template":
            result = _write_template(args)
        elif args.command == "_exec":
            return _internal_exec(Path(args.claim), args.claim_sha256, args.gpu_lock_fd)
        else:
            plan = build_plan(Path(args.spec))
            result = (
                plan
                if args.command == "plan"
                else execute(plan, confirm_claim=args.confirm_claim)
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    except (LaunchRefused, FileNotFoundError, ValueError, OSError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
