#!/usr/bin/env python3
"""Plan or launch one isolated measured-racket VendorV2 N1 diagnostic.

This launcher deliberately does not modify or reuse the formal VendorV1 launcher,
experiment name or namespace.  It is a thin safety adapter over the reviewed N1
diagnostic primitives: exact clean commit, tracked scientific blobs, external runtime
asset pins, fresh/no-clobber namespace, explicit physical GPU UUID, empty-GPU checks
and a lifetime flock.

Five recipes are code-owned.  All use the existing fixed-194 actor / 318-D critic canary
ABI and an exact immutable tape; only target validity/content differs.  The mask is constant
within each independent arm, so it is not an observation column.  Invalid target columns are
zero-filled.  This cannot be promoted to the final varying-ball/N73 ABI.  Reset installs a
tape row and never solves an inverse problem::

    current_lm                    111
    analytic_full                111
    analytic_no_velocity         101
    teacher_pos_face_no_velocity 101
    outcome_dense_only           000

``physical_ball=false`` means the analytic virtual ball/scorer is authoritative.  It does
not claim PhysX paddle contact.  Every stage is fresh, delay-0, single-GPU and
``diagnostic_unauthorized``.  A zero-PPO ``materialize`` stage first publishes the exact
fully composed reward receipt.  A separate zero-PPO ``recipe`` stage must consume that
receipt and publishes the exact dynamic-ready policy recipe.  Smoke/probe512/long512/probe must
consume both artifacts, so neither SHA is guessed or inherited from an older lineage.
``plan`` is read-only; ``launch`` recomputes the plan and requires its exact claim digest.
No arbitrary Hydra override or resume input exists.
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


THIS_FILE = Path(__file__).resolve()
BASE_FILE = THIS_FILE.with_name("launch_n1_reward_screen_diagnostic.py")
BASE_SPEC = importlib.util.spec_from_file_location("_measured_vendor_v2_base", BASE_FILE)
if BASE_SPEC is None or BASE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import diagnostic launcher base")
_B = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(_B)

SCHEMA_VERSION = 2
SPEC_KIND = "n1_measured_vendor_v2_diagnostic_spec_v2"
CLAIM_KIND = "n1_measured_vendor_v2_diagnostic_claim_v2"
EXPERIMENT_NAME = "agibot_a3_action_ball_measured_vendor_v2_n1_diagnostic"
TASK_PROFILE_ID = "HOPEPingPongActionBallA3VendorV2N1Diagnostic"
TASK_PROFILE_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV2N1Diagnostic.yaml"
)
VENDOR_V2_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV2.yaml"
)
LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_measured_vendor_v2_diagnostic.py"
)
BASE_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_reward_screen_diagnostic.py"
)
MATERIALIZER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_measured_action_ball_n1_bundle.py"
)
TAPE_PRODUCER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_action_ball_n1_fixed_tape_variants.py"
)
HOPE_COMMANDS_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
)
ACTOR_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/actor_observation_contract.py"
)
FIXED_TAPE_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_fixed_question_tape.py"
)
TRAINING_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/training_contract.py"
)
EFFECTIVE_REWARD_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/effective_reward_recipe.py"
)
# Tonight's fixed-question canary deliberately retains the already exercised fixed-194 actor and
# 318-D critic.  Validity is constant per independent arm and does not need to identify an arm to
# the policy.  This is not the future varying-ball/N73 ABI.
ACTOR_CONTRACT = "action_ball_table_pose_twist_heading_task_teacher_start_v2"
BUNDLE_KIND = "measured_action_ball_n1_diagnostic_bundle_v1"
PHYSICAL_BALL_SEMANTICS = "analytic_virtual_ball_authoritative_physx_disabled"
ACTION_ID = "take_061_unit04_bh"
MEASURED_UID = "Take_061_unit04_BH"
ACTION_UID = 5527597793770800
MOTION_PATH = (
    "assets/motions/chingmu73_measured_v4_20260803/"
    "hope_Take_061_unit04_BH.npz"
)
MOTION_SHA256 = "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
RACKET_ALIGNMENT_GATES = frozenset(
    {
        "full_position_p95_le_0p05_m",
        "full_face_p95_le_10_deg",
        "full_long_axis_p95_le_10_deg",
        "full_so3_p95_le_10_deg",
        "hit_position_le_0p05_m",
        "hit_face_le_5_deg",
        "hit_long_axis_le_5_deg",
        "hit_so3_le_5_deg",
        "hit_velocity_direction_observable",
        "hit_velocity_direction_le_15_deg",
        "hit_velocity_relative_le_0p20",
    }
)
RECIPES: Mapping[str, tuple[bool, bool, bool]] = {
    "current_lm": (True, True, True),
    "analytic_full": (True, True, True),
    "analytic_no_velocity": (True, False, True),
    "teacher_pos_face_no_velocity": (True, False, True),
    "outcome_dense_only": (False, False, False),
}
TARGET_ORDER = ("position", "velocity", "face")
REWARD_MATERIALIZATION_PROFILE = "measured_vendor_v2_n1_static_v1"
REWARD_RECIPE_FILENAME = "measured_vendor_v2_effective_reward_recipe.json"
POLICY_RECIPE_FILENAME = "measured_vendor_v2_dynamic_ready_policy_recipe.json"
RECIPE_SENTINEL_POLICY_SHA256 = "0" * 64
# ``noise_std_type`` is owned by cfg/algo/ppo.yaml.  This must be a normal
# Hydra override; ``+`` is reserved for keys absent from the composed config.
POLICY_NOISE_STD_OVERRIDE = "algo.policy.noise_std_type=log"
DISABLED_PUSH_DORMANT_FIELDS = (
    "recipe",
    "interval_range_s",
    "combined_exclusive",
    "velocity_range",
)
BUNDLE_KEYS = (
    "schema_version",
    "artifact_type",
    "action_id",
    "action_uid",
    "measured_uid",
    "source_manifest",
    "fixed_n1_source_manifest",
    "motion",
    "racket_alignment_audit",
    "racket_alignment",
    "measured_bank_receipt",
    "measured_manifest_build_report",
    "measured_provenance",
    "core_contact_bundle",
    "prepared_core_bundle",
    "task_profile",
    "immutable_tape_build_report",
    "immutable_tape",
    "mechanical_audit",
    "mechanical_selection",
    "target_recipe",
    "target_validity",
    "runtime_contract",
    "claims",
)
DYNAMIC_READY_V2_KIND = "agibot_a3_action_dynamic_ready_candidate_v2"
_DYNAMIC_READY_V2_KEYS = (
    "schema_version",
    "kind",
    "action_id",
    "robot",
    "authorization",
    "ready_source",
    "sources",
    "teacher_reference",
    "physical_birth_composition",
    "physical_ready",
    "physical_birth_static_evidence",
    "runtime_plant",
    "hold_candidate",
    "required_next_gate",
    "non_claims",
    "producer",
    "content_sha256",
)
_MEASURED_DYNAMIC_READY_SOURCE_KEYS = (
    "stable_motion",
    "measured_bank_receipt",
    "measured_mechanical_audit",
    "physical_birth_seed",
    "mujoco_model",
    "runtime_training_contract",
)
_NOMINAL_HOLD_RECEIPT_KEYS = (
    "schema_version",
    "kind",
    "verdict",
    "action_id",
    "artifact",
    "motion_sha256",
    "teacher_reference_unchanged",
    "teacher_physical_birth_separated",
    "candidate_physical_birth_written",
    "candidate_hold_qdes_and_delay_history_installed",
    "plant_contract_match",
    "control_step_action_delay_runtime",
    "active_terminations",
    "requested_duration_s",
    "completed_duration_s",
    "completed_policy_steps",
    "completed_physics_steps",
    "terminal_reasons",
    "generic_terminated",
    "generic_truncated",
    "minimum_root_z_m",
    "maximum_root_tilt_rad",
    "both_feet_contact_fraction",
    "joint_safety_telemetry",
    "screenshots",
    "content_sha256",
)
BUDGETS = {
    "materialize": (1, 0, 1),
    "recipe": (1, 0, 1),
    "smoke": (1, 2, 1),
    # The full 4096-env A3 scene can exceed the 30-minute boot-staleness
    # watchdog before its first rollout on Pod1.  This bounded diagnostic
    # budget separates recipe learnability from that scene-scaling gate.
    "probe512": (512, 5, 1),
    # Promotion from the five-update recipe canary is still diagnostic-only,
    # fresh and bounded.  This is long enough to expose contact learning while
    # retaining periodic finite checkpoints for early causal review.
    "long512": (512, 1000, 100),
    "probe": (4096, 5, 1),
}
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CUDA_LAUNCH_BLOCKING_SPEC_KEY = "cuda_launch_blocking"

LaunchRefused = _B.LaunchRefused
canonical_sha256 = _B.canonical_sha256


def _exact_dict(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    return _B._exact_dict(value, keys, name=name)


def _validate_external_pin(value: Any, *, name: str) -> tuple[dict[str, str], Path]:
    row = _exact_dict(value, ("path", "sha256"), name=name)
    path = _B._absolute_path(row["path"], name=f"{name}.path", must_exist=True)
    _B._stable_regular_file(path, name=name)
    expected = _B._sha256(row["sha256"], name=f"{name}.sha256")
    observed = _B.sha256_file(path)
    if observed != expected:
        raise LaunchRefused(
            f"{name} file SHA differs: pin={expected} observed={observed}"
        )
    return {"path": str(path), "sha256": expected}, path


def _validate_reward_materialization(value: Any) -> dict[str, Any]:
    pin, path = _validate_external_pin(value, name="reward materialization")
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name="reward materialization")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("reward materialization must be canonical JSON plus newline")
    row = _exact_dict(
        document,
        ("schema_version", "terms", "sha256"),
        name="reward materialization",
    )
    if row["schema_version"] != 1 or type(row["terms"]) is not list:
        raise LaunchRefused("reward materialization schema differs")
    digest = canonical_sha256(
        {"schema_version": row["schema_version"], "terms": row["terms"]}
    )
    if row["sha256"] != digest:
        raise LaunchRefused("reward materialization semantic SHA differs")
    names = [
        term.get("name")
        for term in row["terms"]
        if type(term) is dict and type(term.get("name")) is str
    ]
    if (
        len(names) != len(row["terms"])
        or names != sorted(names)
        or len(names) != len(set(names))
    ):
        raise LaunchRefused("reward materialization terms are not canonical/unique")
    return {
        "artifact": pin,
        "effective_reward_recipe_sha256": digest,
        "term_count": len(names),
    }


def _validate_policy_materialization_header(value: Any) -> dict[str, Any]:
    pin, path = _validate_external_pin(value, name="policy materialization")
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name="policy materialization")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("policy materialization must be canonical JSON plus newline")
    row = _exact_dict(
        document,
        (
            "schema_version",
            "kind",
            "action_count",
            "action_order",
            "policy_contract_sha256",
            "action_ball_ppo_runner_recipe",
            "policy_bootstrap",
        ),
        name="policy materialization",
    )
    policy_sha = _B._sha256(
        row["policy_contract_sha256"], name="materialized policy contract SHA"
    )
    if policy_sha == RECIPE_SENTINEL_POLICY_SHA256:
        raise LaunchRefused("materialized policy contract cannot be the recipe sentinel")
    if (
        row["schema_version"] != 1
        or row["kind"]
        != "action_ball_shared_ready_policy_recipe_materialization_v1"
        or row["action_count"] != 1
        or row["action_order"] != [ACTION_ID]
    ):
        raise LaunchRefused("policy materialization header differs")
    return {
        "artifact": pin,
        "policy_contract_sha256": policy_sha,
        "document": row,
    }


def _isaac_python_entry(value: Any) -> Path:
    """Validate the real executable while preserving the venv entry pathname."""

    entry = _B._absolute_path(
        value, name="source.isaac_python", must_exist=True
    )
    try:
        real = entry.resolve(strict=True)
        info = real.stat()
    except OSError as exc:
        raise LaunchRefused("source.isaac_python cannot resolve to a real file") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(real, os.X_OK):
        raise LaunchRefused(
            "source.isaac_python must resolve to an executable regular file"
        )
    # Do not return ``real``.  Executing a resolved venv symlink bypasses the venv's
    # interpreter discovery and can silently drop its installed packages.
    return entry


def _validate_spec(document: dict[str, Any], *, claimed: bool = False) -> dict[str, Any]:
    keys = (
        "schema_version",
        "kind",
        "source",
        "action_id",
        "bundle",
        "target_recipe",
        "target_validity_mask",
        "reward_materialization",
        "policy_materialization",
        "policy_contract_sha256",
        "expected_effective_reward_recipe_sha256",
        "seed",
        "stage",
        "num_envs",
        "max_iterations",
        "save_interval",
        "gpu",
        "namespace",
        "log_path",
    )
    actual_keys = frozenset(document) if type(document) is dict else frozenset()
    if actual_keys == frozenset(keys):
        row = _exact_dict(document, keys, name="launch spec")
        cuda_launch_blocking = False
    else:
        row = _exact_dict(
            document,
            (*keys, CUDA_LAUNCH_BLOCKING_SPEC_KEY),
            name="launch spec",
        )
        cuda_launch_blocking = row[CUDA_LAUNCH_BLOCKING_SPEC_KEY]
        if type(cuda_launch_blocking) is not bool:
            raise LaunchRefused("cuda_launch_blocking must be a boolean")
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != SPEC_KIND:
        raise LaunchRefused("launch spec schema/kind differs")
    source = _exact_dict(
        row["source"], ("checkout", "commit_sha", "isaac_python"), name="spec.source"
    )
    checkout = _B._absolute_path(source["checkout"], name="source.checkout", must_exist=True)
    commit = source["commit_sha"]
    if type(commit) is not str or _B.COMMIT_RE.fullmatch(commit) is None:
        raise LaunchRefused("source.commit_sha must be exact lowercase 40-hex")
    isaac_python = _isaac_python_entry(source["isaac_python"])
    action_id = row["action_id"]
    if action_id != ACTION_ID:
        raise LaunchRefused("action_id must be the code-owned %s" % ACTION_ID)
    bundle = _exact_dict(row["bundle"], ("path", "sha256"), name="spec.bundle")
    recipe = row["target_recipe"]
    if type(recipe) is not str or recipe not in RECIPES:
        raise LaunchRefused("target_recipe must be one of the five code-owned recipes")
    if row["target_validity_mask"] != list(RECIPES[recipe]):
        raise LaunchRefused("target_validity_mask differs from target_recipe")
    seed = _B._plain_int(row["seed"], name="seed", maximum=(1 << 31) - 1)
    if seed != 0:
        raise LaunchRefused("this fixed first-wave launcher requires seed 0")
    stage = row["stage"]
    if stage not in BUDGETS:
        raise LaunchRefused(
            "stage must be materialize, recipe, smoke, probe512, long512, or probe"
        )
    if stage in ("materialize", "recipe") and recipe != "current_lm":
        raise LaunchRefused(
            "%s stage must use the code-owned current_lm identity arm" % stage
        )
    reward_materialization = None
    policy_materialization = None
    if stage == "materialize":
        if (
            row["reward_materialization"] is not None
            or row["policy_materialization"] is not None
            or row["expected_effective_reward_recipe_sha256"] is not None
            or row["policy_contract_sha256"] is not None
        ):
            raise LaunchRefused(
                "materialize stage must not predeclare reward/policy identities"
            )
        reward_sha = None
        policy_sha = None
    else:
        reward_materialization = _validate_reward_materialization(
            row["reward_materialization"]
        )
        reward_sha = _B._sha256(
            row["expected_effective_reward_recipe_sha256"],
            name="effective reward SHA",
        )
        if reward_sha != reward_materialization["effective_reward_recipe_sha256"]:
            raise LaunchRefused(
                "expected reward SHA differs from its materialized receipt"
            )
        if stage == "recipe":
            if (
                row["policy_materialization"] is not None
                or row["policy_contract_sha256"] is not None
            ):
                raise LaunchRefused(
                    "recipe stage must materialize rather than predeclare policy identity"
                )
            policy_sha = None
        else:
            policy_materialization = _validate_policy_materialization_header(
                row["policy_materialization"]
            )
            policy_sha = _B._sha256(
                row["policy_contract_sha256"], name="policy contract SHA"
            )
            if policy_sha != policy_materialization["policy_contract_sha256"]:
                raise LaunchRefused(
                    "policy contract SHA differs from its materialized recipe"
                )
    expected_budget = BUDGETS[stage]
    actual_budget = (
        _B._plain_int(row["num_envs"], name="num_envs", minimum=1),
        _B._plain_int(row["max_iterations"], name="max_iterations", minimum=0),
        _B._plain_int(row["save_interval"], name="save_interval", minimum=1),
    )
    if actual_budget != expected_budget:
        raise LaunchRefused(
            "%s budget must be exactly %s" % (stage, expected_budget)
        )
    gpu = _B._validate_gpu(row["gpu"])
    namespace = _B._absolute_path(row["namespace"], name="namespace")
    if SAFE_COMPONENT.fullmatch(namespace.name or "") is None:
        raise LaunchRefused("namespace basename is unsafe")
    log_path = _B._absolute_path(row["log_path"], name="log_path")
    if log_path != namespace / "run.log":
        raise LaunchRefused("log_path must equal <namespace>/run.log")
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
        raise LaunchRefused(
            "namespace parent must be the dedicated VendorV2 diagnostic root %s"
            % EXPERIMENT_NAME
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": commit,
            "isaac_python": str(isaac_python),
        },
        "action_id": action_id,
        "bundle": dict(bundle),
        "target_recipe": recipe,
        "target_validity_mask": list(RECIPES[recipe]),
        "reward_materialization": (
            None
            if reward_materialization is None
            else reward_materialization["artifact"]
        ),
        "policy_materialization": (
            None
            if policy_materialization is None
            else policy_materialization["artifact"]
        ),
        "policy_contract_sha256": policy_sha,
        "expected_effective_reward_recipe_sha256": reward_sha,
        "seed": seed,
        "stage": stage,
        "num_envs": actual_budget[0],
        "max_iterations": actual_budget[1],
        "save_interval": actual_budget[2],
        "gpu": gpu,
        "namespace": str(namespace),
        "log_path": str(log_path),
        CUDA_LAUNCH_BLOCKING_SPEC_KEY: cuda_launch_blocking,
    }


def _cuda_launch_blocking_environment(
    spec: Mapping[str, Any],
) -> dict[str, str]:
    """Return the one claim-owned synchronous-CUDA diagnostic setting."""

    if spec[CUDA_LAUNCH_BLOCKING_SPEC_KEY] is True:
        return {"CUDA_LAUNCH_BLOCKING": "1"}
    return {}


def _pin_tracked(checkout: Path, commit: str, relative: str, *, name: str) -> dict[str, str]:
    path = checkout / relative
    pin = {"path": relative, "sha256": _B.sha256_file(path)}
    normalized, _path = _B._verify_tracked_file(checkout, commit, pin, name=name)
    return normalized


def _runtime_sources(checkout: Path, commit: str) -> dict[str, dict[str, str]]:
    rows = {}
    for relative, name in (
        (LAUNCHER_SOURCE, "VendorV2 N1 launcher"),
        (BASE_LAUNCHER_SOURCE, "N1 safety base"),
        (MATERIALIZER_SOURCE, "measured N1 materializer"),
        (TAPE_PRODUCER_SOURCE, "offline fixed-tape producer"),
        (_B.TRAIN_SOURCE, "training entrypoint"),
        (TASK_PROFILE_SOURCE, "VendorV2 N1 task leaf"),
        (VENDOR_V2_SOURCE, "VendorV2 parent task"),
        (_B.KIT_LAUNCHER_SOURCE, "locked Kit launcher"),
        (HOPE_COMMANDS_SOURCE, "ActionBall runtime"),
        (ACTOR_CONTRACT_SOURCE, "actor observation contract"),
        (FIXED_TAPE_SOURCE, "immutable fixed-question tape runtime"),
        (TRAINING_CONTRACT_SOURCE, "dynamic-ready policy contract"),
        (EFFECTIVE_REWARD_SOURCE, "effective reward receipt contract"),
    ):
        rows[name] = _pin_tracked(checkout, commit, relative, name=name)
    if THIS_FILE != checkout / LAUNCHER_SOURCE:
        raise LaunchRefused("running launcher is not the selected checkout launcher")
    # These markers are the minimum source-level proof that the new ABI is wired.  Bundle/runtime
    # receipt validation still owns semantics; absence fails before GPU/namespace mutation.
    command_bytes = (checkout / HOPE_COMMANDS_SOURCE).read_bytes()
    for marker in (
        b"action_ball_target_source",
        b"action_ball_immutable_tape_path",
        b"action_ball_immutable_tape_sha256",
        b"action_ball_target_recipe",
        b"action_ball_target_validity_mask",
        b"immutable_tape",
    ):
        if marker not in command_bytes:
            raise LaunchRefused(
                "immutable-tape runtime dependency is not wired: missing %s"
                % marker.decode("ascii")
            )
    actor_bytes = (checkout / ACTOR_CONTRACT_SOURCE).read_bytes()
    if ACTOR_CONTRACT.encode("ascii") not in actor_bytes:
        raise LaunchRefused("existing fixed-194 ActionBall actor ABI is not wired")
    return rows


def _validate_measured_dynamic_ready_v2(
    checkout: Path,
    commit: str,
    value: Any,
    *,
    action_id: str,
    motion_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Validate the measured launcher's exact schema-v2 hold handoff."""

    row = _B._exact_dict(
        value, _B._DYNAMIC_READY_KEYS, name="measured N1 bundle.dynamic_ready"
    )
    artifact_pin, candidate = _B._load_tracked_json(
        checkout,
        commit,
        row["artifact"],
        name="measured N1 dynamic-ready artifact",
    )
    candidate = _B._exact_dict(
        candidate,
        _DYNAMIC_READY_V2_KEYS,
        name="measured N1 dynamic-ready artifact",
    )
    candidate_content_sha = _B._verify_content_seal(
        candidate,
        name="measured N1 dynamic-ready artifact",
        ensure_ascii=True,
    )
    robot = _B._exact_dict(
        candidate["robot"],
        ("family", "joint_names"),
        name="measured N1 dynamic-ready robot",
    )
    sources = _B._exact_dict(
        candidate["sources"],
        _MEASURED_DYNAMIC_READY_SOURCE_KEYS,
        name="measured N1 dynamic-ready sources",
    )
    stable_motion = _B._exact_dict(
        sources["stable_motion"],
        ("path", "sha256", "frame_index"),
        name="measured N1 dynamic-ready stable motion",
    )
    required_gate = _B._exact_dict(
        candidate["required_next_gate"],
        ("kind", "minimum_horizon_semantics", "zero_terminal_required"),
        name="measured N1 dynamic-ready required gate",
    )
    authorization = _B._exact_dict(
        candidate["authorization"],
        (
            "training_authorized",
            "deployment_authorized",
            "hardware_authorized",
            "isaac_nominal_hold_validated",
        ),
        name="measured N1 dynamic-ready authorization",
    )
    if (
        candidate["schema_version"] != 2
        or candidate["kind"] != DYNAMIC_READY_V2_KIND
        or candidate["action_id"] != action_id
        or robot["family"] != "AgiBot A3"
        or type(robot["joint_names"]) is not list
        or len(robot["joint_names"]) != 31
        or len(set(robot["joint_names"])) != 31
        or any(type(name) is not str or not name for name in robot["joint_names"])
        or stable_motion["frame_index"] != 0
        or stable_motion["sha256"] != motion_sha256
        or type(candidate["runtime_plant"]) is not dict
        or not candidate["runtime_plant"]
        or any(flag is not False for flag in authorization.values())
        or required_gate["kind"] != _B.NOMINAL_HOLD_RECEIPT_KIND
        or required_gate["minimum_horizon_semantics"]
        != "validated_t_hit_plus_reaction_margin"
    ):
        raise LaunchRefused(
            "measured launch requires the exact schema-v2 A3 action/motion plant"
        )

    receipt_pin, receipt = _B._load_tracked_json(
        checkout,
        commit,
        row["nominal_hold_receipt"],
        name="measured N1 nominal-hold receipt",
    )
    receipt = _B._exact_dict(
        receipt,
        _NOMINAL_HOLD_RECEIPT_KEYS,
        name="measured N1 nominal-hold receipt",
    )
    _B._verify_content_seal(
        receipt,
        name="measured N1 nominal-hold receipt",
        ensure_ascii=False,
    )
    receipt_artifact = _B._exact_dict(
        receipt["artifact"],
        ("path", "sha256", "content_sha256"),
        name="measured N1 nominal-hold receipt artifact",
    )
    required_terminations = required_gate["zero_terminal_required"]
    active_terminations = receipt["active_terminations"]
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != _B.NOMINAL_HOLD_RECEIPT_KIND
        or receipt["verdict"] != "PASS"
        or receipt["action_id"] != action_id
        or receipt["motion_sha256"] != motion_sha256
        or receipt["teacher_reference_unchanged"] is not True
        or receipt["teacher_physical_birth_separated"] is not True
        or receipt["candidate_physical_birth_written"] is not True
        or receipt["candidate_hold_qdes_and_delay_history_installed"] is not True
        or receipt["plant_contract_match"] is not True
        or receipt["terminal_reasons"] != []
        or receipt["generic_terminated"] is not False
        or receipt["generic_truncated"] is not False
        or receipt_artifact["sha256"] != artifact_pin["sha256"]
        or receipt_artifact["content_sha256"] != candidate_content_sha
        or type(required_terminations) is not list
        or type(active_terminations) is not list
        or not all(
            type(reason) is str and reason in active_terminations
            for reason in required_terminations
        )
    ):
        raise LaunchRefused(
            "measured nominal-hold receipt does not prove the exact schema-v2 "
            "action/motion plant with zero terminal"
        )
    try:
        binding = _load_training_contract_module(
            checkout
        ).load_action_ball_dynamic_ready_runtime_binding(
            artifact_path=str(checkout / artifact_pin["path"]),
            artifact_sha256=artifact_pin["sha256"],
            nominal_hold_receipt_path=str(checkout / receipt_pin["path"]),
            nominal_hold_receipt_sha256=receipt_pin["sha256"],
            action_order=[action_id],
            motion_paths=[str(checkout / MOTION_PATH)],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LaunchRefused(
            f"measured schema-v2 dynamic-ready runtime binding is invalid: {exc}"
        ) from exc
    if (
        binding.get("schema_version") != 2
        or binding.get("kind") != "action_ball_dynamic_ready_runtime_binding_v2"
        or binding.get("action_order") != [action_id]
        or binding.get("motion_sha256_per_action") != [motion_sha256]
    ):
        raise LaunchRefused(
            "measured dynamic-ready runtime binding identity differs"
        )
    return {
        "artifact": artifact_pin,
        "nominal_hold_receipt": receipt_pin,
    }


def _validate_bundle(
    checkout: Path,
    commit: str,
    pin: dict[str, Any],
    *,
    action_id: str,
    recipe: str,
    seed: int,
) -> dict[str, Any]:
    normalized, bundle = _B._load_tracked_json(checkout, commit, pin, name="measured N1 bundle")
    bundle = _exact_dict(bundle, BUNDLE_KEYS, name="measured N1 bundle")
    if bundle["schema_version"] != 1 or bundle["artifact_type"] != BUNDLE_KIND:
        raise LaunchRefused("measured bundle schema/kind differs")
    if bundle["action_id"] != action_id or bundle["target_recipe"] != recipe:
        raise LaunchRefused("measured bundle action/recipe differs")
    if bundle["action_uid"] != ACTION_UID or bundle["measured_uid"] != MEASURED_UID:
        raise LaunchRefused("measured bundle numeric/string action identity differs")
    prepared_pin, prepared = _B._load_tracked_json(
        checkout,
        commit,
        bundle["prepared_core_bundle"],
        name="prepared measured N1 core bundle",
    )
    prepared_keys = {
        "schema_version",
        "artifact_type",
        "action_id",
        "action_uid",
        "measured_uid",
        "source_manifest",
        "fixed_n1_source_manifest",
        "motion",
        "racket_alignment_audit",
        "racket_alignment",
        "measured_bank_receipt",
        "measured_manifest_build_report",
        "measured_provenance",
        "core_contact_bundle",
        "task_profile",
        "mechanical_audit",
        "mechanical_selection",
        "claims",
    }
    if (
        set(prepared) != prepared_keys
        or prepared.get("schema_version") != 1
        or prepared.get("artifact_type") != "measured_action_ball_n1_prepared_core_v1"
        or prepared.get("action_id") != ACTION_ID
        or prepared.get("action_uid") != ACTION_UID
        or prepared.get("measured_uid") != MEASURED_UID
        or any(
            bundle[key] != prepared[key]
            for key in prepared_keys - {"artifact_type"}
        )
    ):
        raise LaunchRefused("final bundle differs from its exact prepared core")
    if prepared.get("claims", {}).get("dynamic_ready_status") != "PASS":
        raise LaunchRefused("prepared core is not exact dynamic-ready plus nominal-hold PASS")
    validity = bundle["target_validity"]
    if validity != {"order": list(TARGET_ORDER), "mask": list(RECIPES[recipe])}:
        raise LaunchRefused("measured bundle validity mask differs")
    task_pin, _task_path = _B._verify_tracked_file(
        checkout,
        commit,
        bundle["task_profile"],
        name="VendorV2 N1 task profile",
    )
    if task_pin["path"] != TASK_PROFILE_SOURCE:
        raise LaunchRefused("bundle task profile is not the isolated VendorV2 N1 leaf")
    legacy_dynamic_ready_validator = _B._validate_dynamic_ready
    try:
        _B._validate_dynamic_ready = _validate_measured_dynamic_ready_v2
        core = _B._validate_bundle(
            checkout,
            commit,
            bundle["core_contact_bundle"],
            expected_action=action_id,
            expected_scope="full",
            require_dynamic_ready=True,
        )
    finally:
        _B._validate_dynamic_ready = legacy_dynamic_ready_validator
    motion_pin, _motion_path = _B._verify_tracked_file(
        checkout, commit, bundle["motion"], name="measured motion"
    )
    if motion_pin != {"path": MOTION_PATH, "sha256": MOTION_SHA256}:
        raise LaunchRefused("measured bundle motion is not the code-owned exact clip")
    if core["motion"] != motion_pin:
        raise LaunchRefused("core bundle and measured wrapper motion differ")
    alignment_pin, alignment_report = _B._load_tracked_json(
        checkout,
        commit,
        bundle["racket_alignment_audit"],
        name="independent racket FK alignment audit",
    )
    alignment = bundle["racket_alignment"]
    if (
        alignment_report.get("admitted") is not True
        or alignment_report.get("finite") is not True
        or alignment_report.get("motion_sha256") != motion_pin["sha256"]
        or type(alignment_report.get("gates")) is not dict
        or set(alignment_report["gates"]) != RACKET_ALIGNMENT_GATES
        or any(value is not True for value in alignment_report["gates"].values())
        or alignment_report.get("uid") != MEASURED_UID
        or type(alignment) is not dict
        or alignment.get("motion_sha256") != motion_pin["sha256"]
        or alignment.get("all_11_gates_pass") is not True
        or alignment.get("diagnostic_unauthorized") is not True
    ):
        raise LaunchRefused("independent racket FK alignment evidence differs")
    bank_pin, bank = _B._load_tracked_json(
        checkout,
        commit,
        bundle["measured_bank_receipt"],
        name="measured bank receipt",
    )
    report_pin, report = _B._load_tracked_json(
        checkout,
        commit,
        bundle["measured_manifest_build_report"],
        name="measured manifest build report",
    )
    if (
        report.get("measured_bank_receipt_sha256") != bank_pin["sha256"]
        or report.get("file_sha256") != bundle["source_manifest"].get("sha256")
        or report.get("racket_authority") != "measured_channel"
        or bank.get("authorization", {}).get("training") is not False
        or bank.get("authorization", {}).get("mechanical_admission") is not False
        or type(bundle["measured_provenance"]) is not dict
    ):
        raise LaunchRefused("measured bank/build provenance boundary differs")
    _B._load_tracked_json(
        checkout,
        commit,
        bundle["source_manifest"],
        name="measured source manifest",
    )
    _B._load_tracked_json(
        checkout,
        commit,
        bundle["fixed_n1_source_manifest"],
        name="fixed N1 source manifest",
    )
    _B._load_tracked_json(
        checkout,
        commit,
        bundle["mechanical_audit"],
        name="mechanical audit",
    )
    selection = bundle["mechanical_selection"]
    if (
        type(selection) is not dict
        or selection.get("motion_sha256") != motion_pin["sha256"]
        or selection.get("kinematic_limit_verdict") != "PASS"
        or selection.get("mechanical_verdict") not in ("PASS", "UNKNOWN")
        or selection.get("diagnostic_unauthorized") is not True
    ):
        raise LaunchRefused("mechanical diagnostic selection is not eligible")
    report_pin, _report = _B._load_tracked_json(
        checkout,
        commit,
        bundle["immutable_tape_build_report"],
        name="immutable tape build report",
    )
    materializer_spec = importlib.util.spec_from_file_location(
        "_measured_vendor_v2_materializer_validation",
        checkout / MATERIALIZER_SOURCE,
    )
    if materializer_spec is None or materializer_spec.loader is None:
        raise LaunchRefused("cannot import measured N1 materializer")
    materializer = importlib.util.module_from_spec(materializer_spec)
    sys.modules[materializer_spec.name] = materializer
    try:
        materializer_spec.loader.exec_module(materializer)
        if set(BUNDLE_KEYS) != set(materializer.FINAL_BUNDLE_KEYS):
            raise LaunchRefused("finalize/launcher bundle schema differs")
        validated_report_pin, tape = materializer._validate_tape_build_report(
            checkout,
            checkout / report_pin["path"],
            report_pin["sha256"],
            action_uid=ACTION_UID,
            motion_sha=MOTION_SHA256,
            recipe=recipe,
        )
    except Exception as exc:
        raise LaunchRefused(
            "immutable tape build report failed validation: %s" % exc
        ) from exc
    if validated_report_pin != report_pin:
        raise LaunchRefused("immutable tape build report pin differs")
    tape_pin, _tape_path = _B._verify_tracked_file(
        checkout, commit, tape["artifact"], name="immutable target tape"
    )
    if (
        bundle["immutable_tape"] != tape_pin
        or tape["target_validity"] != validity
        or tape["sampler_seed"] != seed
        or tape["source_identity"]["manifest_sha256"]
        != core["manifest"]["sha256"]
        or tape["source_identity"]["physics_sha256"]
        != core["profile_pins"]["physics_profile_sha256"]
        or tape["source_identity"]["solver_sha256"]
        != core["profile_pins"]["solver_profile_sha256"]
    ):
        raise LaunchRefused("immutable tape canonical identity/lineage differs")
    runtime = bundle["runtime_contract"]
    if runtime != {
        "target_source": "immutable_tape",
        "reset_inverse_solve": False,
        "control_step_action_delay": [0, 0],
        "physical_ball_semantics": PHYSICAL_BALL_SEMANTICS,
        "canary_contract": "fixed_question_ablation_canary_v1",
        "actor_obs_contract": ACTOR_CONTRACT,
        "actor_width": 194,
        "critic_width": 318,
        "final_varying_ball_abi": False,
        "target_validity_is_fixed_recipe_constant": True,
        "invalid_target_columns_zero_filled": True,
        "invalid_target_columns_masked_from_reward": True,
        "target_noise_disabled": True,
        "adaptive_sigma_disabled": True,
    }:
        raise LaunchRefused("measured bundle runtime contract differs")
    claims = bundle["claims"]
    required_false = (
        "training_authorized",
    )
    required_true = (
        "diagnostic_unauthorized",
        "formal_evidence_prohibited",
        "promotion_prohibited",
        "export_prohibited",
        "deployment_prohibited",
        "hardware_prohibited",
    )
    if (
        type(claims) is not dict
        or any(claims.get(key) is not False for key in required_false)
        or any(claims.get(key) is not True for key in required_true)
    ):
        raise LaunchRefused("measured bundle diagnostic authority boundary differs")
    return {
        "bundle": normalized,
        "prepared_core_bundle": prepared_pin,
        "action_id": action_id,
        "target_recipe": recipe,
        "target_validity": validity,
        "motion": motion_pin,
        "racket_alignment_audit": alignment_pin,
        "racket_alignment": alignment,
        "measured_bank_receipt": bank_pin,
        "measured_manifest_build_report": report_pin,
        "core": core,
        "task_profile": task_pin,
        "immutable_tape_build_report": report_pin,
        "immutable_tape": tape_pin,
        "tape_row_count": tape["row_count"],
        "tape_sampler_seed": tape["sampler_seed"],
        "selected_target_lineage": tape["selected_target_lineage"],
        "mechanical_selection": selection,
        "runtime_contract": runtime,
    }


def _load_training_contract_module(checkout: Path):
    module_spec = importlib.util.spec_from_file_location(
        "_measured_vendor_v2_training_contract",
        checkout / TRAINING_CONTRACT_SOURCE,
    )
    if module_spec is None or module_spec.loader is None:
        raise LaunchRefused("cannot import dynamic-ready training contract")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise LaunchRefused(
            f"cannot import dynamic-ready training contract: {exc}"
        ) from exc
    return module


def _validate_policy_materialization(
    value: Any, *, checkout: Path, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    header = _validate_policy_materialization_header(value)
    row = header["document"]
    policy_sha = header["policy_contract_sha256"]
    runner_recipe = row["action_ball_ppo_runner_recipe"]
    bootstrap = row["policy_bootstrap"]
    training_contract = _load_training_contract_module(checkout)
    dynamic = bundle["core"]["dynamic_ready"]
    try:
        expected_binding = (
            training_contract.load_action_ball_dynamic_ready_runtime_binding(
                artifact_path=str(checkout / dynamic["artifact"]["path"]),
                artifact_sha256=dynamic["artifact"]["sha256"],
                nominal_hold_receipt_path=str(
                    checkout / dynamic["nominal_hold_receipt"]["path"]
                ),
                nominal_hold_receipt_sha256=(
                    dynamic["nominal_hold_receipt"]["sha256"]
                ),
                action_order=[ACTION_ID],
                motion_paths=[str(checkout / bundle["motion"]["path"])],
            )
        )
        training_contract.validate_action_ball_policy_bootstrap(
            bootstrap, expected_action_count=1
        )
        portable_bootstrap = (
            training_contract.action_ball_policy_bootstrap_scientific_identity(
                bootstrap, repo_root=checkout
            )
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise LaunchRefused(
            f"materialized policy failed dynamic-ready validation: {exc}"
        ) from exc
    runner_payload = (
        runner_recipe.get("recipe") if type(runner_recipe) is dict else None
    )
    runner_sha = None
    if type(runner_payload) is dict:
        runner_sha = canonical_sha256(runner_payload)
    initialization = (
        bootstrap.get("initialization") if type(bootstrap) is dict else None
    )
    ready_source = (
        bootstrap.get("ready_source") if type(bootstrap) is dict else None
    )
    identity = (
        ready_source.get("identity") if type(ready_source) is dict else None
    )
    init_noise_std = (
        initialization.get("init_noise_std")
        if type(initialization) is dict
        else None
    )
    realized_noise_std = (
        initialization.get("required_realized_init_noise_std")
        if type(initialization) is dict
        else None
    )
    if (
        type(runner_recipe) is not dict
        or runner_recipe.get("sha256") != policy_sha
        or type(runner_payload) is not dict
        or runner_sha != policy_sha
        or runner_payload.get("policy_initialization") != portable_bootstrap
        or type(bootstrap) is not dict
        or bootstrap.get("schema_version") != 3
        or bootstrap.get("action_count") != 1
        or bootstrap.get("action_order") != [ACTION_ID]
        or type(identity) is not dict
        or identity.get("binding_sha256")
        != expected_binding["binding_sha256"]
        or type(initialization) is not dict
        or initialization.get("noise_std_type") != "log"
        or type(init_noise_std) not in (int, float)
        or float(init_noise_std) != 0.02
        or type(realized_noise_std) not in (int, float)
        or float(realized_noise_std) != 0.02
    ):
        raise LaunchRefused(
            "materialized policy is not the exact log-std dynamic-ready N1 contract"
        )
    return {
        "artifact": header["artifact"],
        "policy_contract_sha256": policy_sha,
        "dynamic_ready_binding_sha256": expected_binding["binding_sha256"],
        "noise_std_type": "log",
        "configured_and_realized_init_noise_std": 0.02,
    }


def _check_rsl_namespace(checkout: Path, namespace_name: str) -> None:
    root = checkout / _B.WBT_RELATIVE / "logs/rsl_rl" / EXPERIMENT_NAME
    if not root.exists():
        return
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise LaunchRefused("RSL experiment root is not a real directory")
    suffix = "_%s-DIAGNOSTIC_UNAUTHORIZED" % namespace_name
    spent = [child.name for child in root.iterdir() if child.name.endswith(suffix)]
    if spent:
        raise LaunchRefused("trainer run_name is already spent: %s" % sorted(spent)[0])


def _training_argv(spec: dict[str, Any], bundle: dict[str, Any]) -> list[str]:
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / _B.WBT_RELATIVE
    core = bundle["core"]
    dynamic = core["dynamic_ready"]
    motion = checkout / bundle["motion"]["path"]
    manifest = checkout / core["manifest"]["path"]
    tape = checkout / bundle["immutable_tape"]["path"]
    validity = json.dumps(spec["target_validity_mask"], separators=(",", ":"))
    list_one = json.dumps([str(motion)], separators=(",", ":"))
    action_one = json.dumps([spec["action_id"]], separators=(",", ":"))
    policy_sha = (
        RECIPE_SENTINEL_POLICY_SHA256
        if spec["stage"] in ("materialize", "recipe")
        else spec["policy_contract_sha256"]
    )
    argv = [
        spec["source"]["isaac_python"],
        str(wbt / "scripts/train.py"),
        "task=%s" % TASK_PROFILE_ID,
        "algo=ppo",
        "algo.policy.init_noise_std=0.02",
        POLICY_NOISE_STD_OVERRIDE,
        "headless=true",
        "logger=tensorboard",
        "video=false",
        "device=cuda:0",
        "seed=%d" % spec["seed"],
        "num_envs=%d" % spec["num_envs"],
        "max_iterations=%d" % spec["max_iterations"],
        "algo.runner.save_interval=%d" % spec["save_interval"],
        "run_name=%s-DIAGNOSTIC_UNAUTHORIZED" % Path(spec["namespace"]).name,
        "task.experiment_name=%s" % EXPERIMENT_NAME,
        "task.actor_obs_contract=%s" % ACTOR_CONTRACT,
        "action_ball_dynamic_ready_bootstrap=true",
        "action_ball_dynamic_ready_artifact_path=%s"
        % (checkout / dynamic["artifact"]["path"]),
        "action_ball_dynamic_ready_artifact_sha256=%s"
        % dynamic["artifact"]["sha256"],
        "action_ball_dynamic_ready_nominal_receipt_path=%s"
        % (checkout / dynamic["nominal_hold_receipt"]["path"]),
        "action_ball_dynamic_ready_nominal_receipt_sha256=%s"
        % dynamic["nominal_hold_receipt"]["sha256"],
        "motion_file=%s" % list_one,
        "task.racket.clip_names=%s" % action_one,
        "task.racket.action_ball_manifest_path=%s" % manifest,
        "task.racket.action_ball_manifest_sha256=%s" % core["manifest"]["sha256"],
        "task.racket.action_ball_policy_contract_sha256=%s"
        % policy_sha,
        "task.racket.action_ball_diagnostic_unauthorized=true",
        "task.motion.action_ball_diagnostic_split_ready_teacher=true",
        "task.racket.action_ball_seed=%d" % spec["seed"],
        "task.racket.action_ball_target_source=immutable_tape",
        "task.racket.action_ball_immutable_tape_path=%s" % tape,
        "task.racket.action_ball_immutable_tape_sha256=%s"
        % bundle["immutable_tape"]["sha256"],
        "task.racket.action_ball_target_recipe=%s" % spec["target_recipe"],
        "task.racket.action_ball_target_validity_mask=%s" % validity,
        "task.racket.action_ball_target_observation_noise=false",
        "task.racket.adaptive_sigma=false",
        "task.racket.adaptive_sigma_monotonic=false",
        "task.racket.adaptive_sigma_normal=false",
        "task.racket.target_noise_white=0.0",
        "task.racket.target_noise_ar1_sigma=0.0",
        "task.actions.control_step_action_delay_min=0",
        "task.actions.control_step_action_delay_max=0",
        "task.push.enable=false",
        *("~task.push.%s" % field for field in DISABLED_PUSH_DORMANT_FIELDS),
        "task.physical_ball=false",
        "task.racket.virtual_ball=true",
        "task.racket.action_ball_pool_refill_rows=1",
        "task.racket.question_bank=",
        "task.racket.cq_anchor_bank=",
        "task.racket.exam_bank=",
    ]
    if spec["stage"] == "materialize":
        argv.extend(
            [
                "+n1_vendor_sigma_profile=%s"
                % REWARD_MATERIALIZATION_PROFILE,
                "+action_ball_effective_reward_recipe_output_path=%s"
                % (Path(spec["namespace"]) / REWARD_RECIPE_FILENAME),
            ]
        )
    else:
        argv.append(
            "expected_effective_reward_recipe_sha256=%s"
            % spec["expected_effective_reward_recipe_sha256"]
        )
        if spec["stage"] == "recipe":
            argv.append(
                "action_ball_policy_recipe_output_path=%s"
                % (Path(spec["namespace"]) / POLICY_RECIPE_FILENAME)
            )
    return argv


def _output_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec["stage"] == "materialize":
        return {
            "ppo_update_count": 0,
            "effective_reward_recipe": str(
                Path(spec["namespace"]) / REWARD_RECIPE_FILENAME
            ),
            "policy_recipe": None,
            "boot_marker": "ACTION_BALL_EFFECTIVE_REWARD_RECIPE_MATERIALIZED_JSON",
        }
    if spec["stage"] == "recipe":
        return {
            "ppo_update_count": 0,
            "effective_reward_recipe": None,
            "policy_recipe": str(
                Path(spec["namespace"]) / POLICY_RECIPE_FILENAME
            ),
            "boot_marker": "ACTION_BALL_POLICY_RECIPE_MATERIALIZED",
        }
    return {
        "ppo_update_count": spec["max_iterations"],
        "effective_reward_recipe": None,
        "policy_recipe": None,
        "boot_marker": "Learning iteration",
    }


def build_plan(spec_path: Path) -> dict[str, Any]:
    spec_path = _B._absolute_path(str(spec_path), name="--spec", must_exist=True)
    _B._stable_regular_file(spec_path, name="launch spec")
    raw = spec_path.read_bytes()
    document = _B._strict_json_bytes(raw, name="launch spec")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("launch spec must be canonical JSON plus newline")
    spec = _validate_spec(document)
    checkout = Path(spec["source"]["checkout"])
    commit = spec["source"]["commit_sha"]
    source = _B._verify_clean_source(checkout, commit)
    sources = _runtime_sources(checkout, commit)
    assets = _B._validate_runtime_asset_environment()
    bundle = _validate_bundle(
        checkout,
        commit,
        spec["bundle"],
        action_id=spec["action_id"],
        recipe=spec["target_recipe"],
        seed=spec["seed"],
    )
    materialization_inputs = {
        "reward": (
            None
            if spec["stage"] == "materialize"
            else _validate_reward_materialization(
                spec["reward_materialization"]
            )
        ),
        "policy": (
            None
            if spec["stage"] in ("materialize", "recipe")
            else _validate_policy_materialization(
                spec["policy_materialization"],
                checkout=checkout,
                bundle=bundle,
            )
        ),
    }
    if (
        materialization_inputs["reward"] is not None
        and materialization_inputs["reward"][
            "effective_reward_recipe_sha256"
        ]
        != spec["expected_effective_reward_recipe_sha256"]
    ):
        raise LaunchRefused("reward materialization drifted after spec validation")
    if (
        materialization_inputs["policy"] is not None
        and materialization_inputs["policy"]["policy_contract_sha256"]
        != spec["policy_contract_sha256"]
    ):
        raise LaunchRefused("policy materialization drifted after spec validation")
    _check_rsl_namespace(checkout, Path(spec["namespace"]).name)
    argv = _training_argv(spec, bundle)
    output_contract = _output_contract(spec)
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
        "fresh_only": True,
        "reward_materialization_only": spec["stage"] == "materialize",
        "policy_recipe_materialization_only": spec["stage"] == "recipe",
        "ppo_updates_authorized": output_contract["ppo_update_count"],
        "control_step_action_delay": 0,
        "reset_inverse_solve": False,
        "physical_ball_semantics": PHYSICAL_BALL_SEMANTICS,
        "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
        "source": source,
        "runtime_sources": sources,
        "runtime_assets": assets,
        "bundle": bundle,
        "materialization_inputs": materialization_inputs,
        "output_contract": output_contract,
        "boot_marker": output_contract["boot_marker"],
        "training_argv": argv,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": canonical_sha256(payload),
        "canonical_payload": payload,
    }


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
    spec = _validate_spec(payload["spec"], claimed=True)
    checkout = Path(spec["source"]["checkout"])
    _B._verify_clean_source(checkout, spec["source"]["commit_sha"])
    sources = _runtime_sources(checkout, spec["source"]["commit_sha"])
    if sources != payload["runtime_sources"]:
        raise LaunchRefused("runtime source identity drifted")
    assets = _B._validate_runtime_asset_claim(payload["runtime_assets"])
    bundle = _validate_bundle(
        checkout,
        spec["source"]["commit_sha"],
        spec["bundle"],
        action_id=spec["action_id"],
        recipe=spec["target_recipe"],
        seed=spec["seed"],
    )
    materialization_inputs = {
        "reward": (
            None
            if spec["stage"] == "materialize"
            else _validate_reward_materialization(
                spec["reward_materialization"]
            )
        ),
        "policy": (
            None
            if spec["stage"] in ("materialize", "recipe")
            else _validate_policy_materialization(
                spec["policy_materialization"],
                checkout=checkout,
                bundle=bundle,
            )
        ),
    }
    if (
        bundle != payload["bundle"]
        or materialization_inputs != payload["materialization_inputs"]
        or _output_contract(spec) != payload["output_contract"]
        or payload["boot_marker"] != payload["output_contract"]["boot_marker"]
        or _training_argv(spec, bundle) != payload["training_argv"]
    ):
        raise LaunchRefused("bundle or training argv drifted")
    lock_path = Path(spec["gpu"]["lock_path"])
    info = os.fstat(lock_fd)
    path_info = lock_path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino)
    ):
        raise LaunchRefused("inherited GPU lock identity differs")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise LaunchRefused("inherited GPU lock is not held") from exc
    gpu = _B._verify_gpu_empty(spec["gpu"]["index"], spec["gpu"]["uuid"])
    _B._write_exclusive_json(
        Path(spec["namespace"]) / "pre_exec_gpu_admission.json",
        {
            "schema_version": 1,
            "kind": "measured_vendor_v2_pre_exec_gpu_admission_v1",
            "launch_claim_sha256": claim_sha,
            "gpu": gpu,
        },
    )
    wbt = checkout / _B.WBT_RELATIVE
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(wbt / "source/whole_body_tracking"),
        "CUDA_VISIBLE_DEVICES": str(spec["gpu"]["index"]),
        "HYDRA_FULL_ERROR": "1",
        "WANDB_MODE": "offline",
        "HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256": claim_sha,
        **_B._runtime_asset_exec_environment(assets),
        **_cuda_launch_blocking_environment(spec),
    }
    os.chdir(wbt)
    os.execve(payload["training_argv"][0], payload["training_argv"], environment)
    raise AssertionError("execve returned")


def launch(plan: dict[str, Any], *, confirm_claim: str) -> dict[str, Any]:
    expected = _B._sha256(confirm_claim, name="--confirm-claim")
    if expected != plan["launch_claim_sha256"]:
        raise LaunchRefused("--confirm-claim differs from freshly recomputed plan")
    spec = plan["canonical_payload"]["spec"]
    checkout = Path(spec["source"]["checkout"])
    _B._verify_clean_source(checkout, spec["source"]["commit_sha"])
    _B._validate_runtime_asset_claim(plan["canonical_payload"]["runtime_assets"])
    lock_fd = _B._open_gpu_lock(Path(spec["gpu"]["lock_path"]))
    namespace = None
    try:
        first = _B._verify_gpu_empty(spec["gpu"]["index"], spec["gpu"]["uuid"])
        namespace = _B._claim_namespace(plan)
        _B._write_exclusive_json(
            namespace / "pre_launch_gpu_admission.json",
            {
                "schema_version": 1,
                "kind": "measured_vendor_v2_pre_launch_gpu_admission_v1",
                "launch_claim_sha256": expected,
                "gpu": first,
            },
        )
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
        environment = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": os.environ.get("HOME", "/root"),
            "LANG": "C",
            "LC_ALL": "C",
            "KIT_BOOT_MARKER": plan["canonical_payload"]["boot_marker"],
            "KIT_BOOT_TIMEOUT_S": "2700",
            "KIT_BOOT_STALE_TIMEOUT_S": "1800",
            "KIT_BOOT_POLL_S": "5",
            "KIT_BOOT_STATE_FILE": str(state),
        }
        result = subprocess.run(
            [str(checkout / _B.KIT_LAUNCHER_SOURCE), spec["log_path"], *internal],
            cwd=checkout / _B.WBT_RELATIVE,
            env=environment,
            pass_fds=(lock_fd,),
            check=False,
        )
        if result.returncode != 0:
            raise LaunchRefused(
                "locked Kit launcher returned %d; namespace remains spent" % result.returncode
            )
        materialized_reward = None
        materialized_policy = None
        output_contract = plan["canonical_payload"]["output_contract"]
        if spec["stage"] == "materialize":
            output_path = Path(output_contract["effective_reward_recipe"])
            materialized_reward = _validate_reward_materialization(
                {
                    "path": str(output_path),
                    "sha256": _B.sha256_file(output_path),
                }
            )
        elif spec["stage"] == "recipe":
            output_path = Path(output_contract["policy_recipe"])
            materialized_policy = _validate_policy_materialization(
                {
                    "path": str(output_path),
                    "sha256": _B.sha256_file(output_path),
                },
                checkout=checkout,
                bundle=plan["canonical_payload"]["bundle"],
            )
        return {
            "schema_version": 1,
            "kind": "n1_measured_vendor_v2_diagnostic_launch_result_v2",
            "launch_claim_sha256": expected,
            "stage": spec["stage"],
            "namespace": str(namespace),
            "log_path": spec["log_path"],
            "state_path": str(state),
            "gpu": spec["gpu"],
            "output_contract": output_contract,
            "materialized_effective_reward_recipe": materialized_reward,
            "materialized_policy_recipe": materialized_policy,
            "ppo_update_count": output_contract["ppo_update_count"],
            "diagnostic_unauthorized": True,
            "accepted": True,
        }
    finally:
        os.close(lock_fd)


def _write_template(args: argparse.Namespace) -> dict[str, Any]:
    budget = BUDGETS[args.stage]
    namespace = Path(args.namespace).resolve(strict=False)
    isaac_python = _isaac_python_entry(args.isaac_python)
    reward_pin = None
    policy_pin = None
    if args.stage != "materialize":
        if (
            args.reward_materialization_path is None
            or args.reward_materialization_sha256 is None
        ):
            raise LaunchRefused(
                "%s template requires the exact reward materialization path/SHA"
                % args.stage
            )
        reward_materialization = _validate_reward_materialization(
            {
                "path": args.reward_materialization_path,
                "sha256": args.reward_materialization_sha256,
            }
        )
        reward_pin = reward_materialization["artifact"]
        reward_sha = reward_materialization[
            "effective_reward_recipe_sha256"
        ]
    else:
        if (
            args.reward_materialization_path is not None
            or args.reward_materialization_sha256 is not None
        ):
            raise LaunchRefused(
                "materialize template must not accept a reward materialization"
            )
        reward_sha = None
    if args.stage not in ("materialize", "recipe"):
        if (
            args.policy_materialization_path is None
            or args.policy_materialization_sha256 is None
        ):
            raise LaunchRefused(
                "%s template requires the exact policy materialization path/SHA"
                % args.stage
            )
        policy_materialization = _validate_policy_materialization_header(
            {
                "path": args.policy_materialization_path,
                "sha256": args.policy_materialization_sha256,
            }
        )
        policy_pin = policy_materialization["artifact"]
        policy_sha = policy_materialization["policy_contract_sha256"]
    else:
        if (
            args.policy_materialization_path is not None
            or args.policy_materialization_sha256 is not None
        ):
            raise LaunchRefused(
                "%s template must not accept a policy materialization"
                % args.stage
            )
        policy_sha = None
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(Path(args.checkout).resolve(strict=True)),
            "commit_sha": args.commit_sha,
            "isaac_python": str(isaac_python),
        },
        "action_id": args.action_id,
        "bundle": {"path": args.bundle_path, "sha256": args.bundle_sha256},
        "target_recipe": args.target_recipe,
        "target_validity_mask": list(RECIPES[args.target_recipe]),
        "reward_materialization": reward_pin,
        "policy_materialization": policy_pin,
        "policy_contract_sha256": policy_sha,
        "expected_effective_reward_recipe_sha256": reward_sha,
        "seed": args.seed,
        "stage": args.stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "gpu": {
            "index": args.gpu_index,
            "uuid": args.gpu_uuid,
            "owner": args.owner,
            "lock_path": "/tmp/hope_lean_queue_gpu%d.lock" % args.gpu_index,
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }
    if getattr(args, CUDA_LAUNCH_BLOCKING_SPEC_KEY, False):
        document[CUDA_LAUNCH_BLOCKING_SPEC_KEY] = True
    output = Path(args.output).resolve(strict=False)
    _B._write_exclusive_json(output, document)
    return {"status": "CREATED", "spec": str(output), "target_recipe": args.target_recipe}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--output", required=True)
    template.add_argument("--checkout", required=True)
    template.add_argument("--commit-sha", required=True)
    template.add_argument("--isaac-python", required=True)
    template.add_argument("--action-id", required=True)
    template.add_argument("--bundle-path", required=True)
    template.add_argument("--bundle-sha256", required=True)
    template.add_argument("--target-recipe", required=True, choices=tuple(RECIPES))
    template.add_argument("--reward-materialization-path")
    template.add_argument("--reward-materialization-sha256")
    template.add_argument("--policy-materialization-path")
    template.add_argument("--policy-materialization-sha256")
    template.add_argument("--seed", required=True, type=int, choices=(0,))
    template.add_argument("--stage", required=True, choices=tuple(BUDGETS))
    template.add_argument("--gpu-index", required=True, type=int)
    template.add_argument("--gpu-uuid", required=True)
    template.add_argument("--owner", required=True)
    template.add_argument("--namespace", required=True)
    template.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help="diagnostic-only: set CUDA_LAUNCH_BLOCKING=1 in the trainer",
    )
    for command in ("plan", "launch"):
        child = sub.add_parser(command)
        child.add_argument("--spec", required=True)
        if command == "launch":
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
            if args.command == "plan":
                result = plan
            else:
                result = launch(plan, confirm_claim=args.confirm_claim)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    except (LaunchRefused, FileNotFoundError, ValueError, OSError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
