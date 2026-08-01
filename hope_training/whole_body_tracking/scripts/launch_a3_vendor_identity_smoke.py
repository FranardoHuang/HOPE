#!/usr/bin/env python3
"""Materialize the first A3-vendor shared-ready identity, fail closed.

This launcher exists only to break the one-time bootstrap cycle in which the
new A3 plant needs a live ``training_contract.json`` before an action-specific
dynamic-ready bundle can be rematerialized.  It has exactly two stages:

* ``recipe``: one environment, no PPO update, and one no-clobber shared-ready
  policy recipe at ``<namespace>/vendor_shared_ready_policy_recipe.json``;
* ``smoke``: one environment and exactly two PPO updates, saving every update,
  so the run emits its live training contract and bounded smoke checkpoints.

The smoke spec must pin the policy-contract SHA emitted by the recipe stage.
Both stages use one code-reviewed action selected from the A3-vendor action
registry and its fixed, tracked stable-v2 motion plus identity-bootstrap-only
N=1 manifest, repinned to the exact current solver and physics profile through
a separately sealed receipt.  An action whose identity artifacts do not yet
have code-owned digests fails closed before any runtime or GPU work.  The
operator may select only the action id; artifact pins are never operator input.
The stages intentionally
consume no bundle at all, so no schema-v2/dynamic-ready artifact can enter the
cycle.  The selected task is
always ``HOPEPingPongActionBallA3VendorV1`` and all task-owned vendor PD, push
and control-step-delay settings remain untouched.

This is diagnostic identity materialization, not a training wave.  It cannot
resume, launch a long stage, mint formal evidence, promote a curriculum,
export a policy, judge a checkpoint, or authorize hardware.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


_THIS_FILE = Path(__file__).resolve()
_SAFETY_FILE = _THIS_FILE.with_name("launch_n1_reward_screen_diagnostic.py")
_SAFETY_SPEC = importlib.util.spec_from_file_location(
    "_hope_vendor_identity_safety", _SAFETY_FILE
)
if _SAFETY_SPEC is None or _SAFETY_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load launch safety implementation: {_SAFETY_FILE}")
_S = importlib.util.module_from_spec(_SAFETY_SPEC)
_SAFETY_SPEC.loader.exec_module(_S)

_REGISTRY_FILE = _THIS_FILE.with_name("a3_vendor_action_registry.py")
_REGISTRY_SPEC = importlib.util.spec_from_file_location(
    "_hope_vendor_identity_action_registry", _REGISTRY_FILE
)
if _REGISTRY_SPEC is None or _REGISTRY_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load vendor action registry: {_REGISTRY_FILE}")
_R = importlib.util.module_from_spec(_REGISTRY_SPEC)
sys.modules[_REGISTRY_SPEC.name] = _R
_REGISTRY_SPEC.loader.exec_module(_R)


SCHEMA_VERSION = 2
SPEC_KIND = "a3_vendor_identity_smoke_spec_v2"
CLAIM_KIND = "a3_vendor_identity_smoke_claim_v2"
RESULT_KIND = "a3_vendor_identity_smoke_launch_result_v2"
EXPERIMENT_NAME = "agibot_a3_action_ball_vendor_identity_smoke"
TASK_PROFILE_ID = "HOPEPingPongActionBallA3VendorV1"
ACTION_ID = _R.DEFAULT_ACTION_ID
SCOPE = _R.ACTION_CONFIGS[ACTION_ID].scope
SEED = 0
ACTOR_OBS_CONTRACT = (
    "action_ball_table_pose_twist_heading_task_teacher_start_v2"
)
RECIPE_SENTINEL_POLICY_SHA256 = "0" * 64
RECIPE_FILENAME = "vendor_shared_ready_policy_recipe.json"
DIAGNOSTIC_SUFFIX = "DIAGNOSTIC_UNAUTHORIZED"
# Exact receipt emitted by the adopted vendor ActionBall reward subtree.  It
# retains the shared current-low recipe and adds the independently paid
# coarse-position strike-window kernel (weight 1.0, std 0.30) selected by the
# 2026-07-31 entry-distance probe.  The live recipe stage independently
# recomputes this value before scene construction.
EXPECTED_REWARD_RECIPE_SHA256 = (
    "8220f3397cb07a143149353d13f21914a90ac7be874169d519ebf5b2b9154dc3"
)

LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_a3_vendor_identity_smoke.py"
)
ACTION_REGISTRY_SOURCE = (
    "hope_training/whole_body_tracking/scripts/a3_vendor_action_registry.py"
)
SAFETY_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_reward_screen_diagnostic.py"
)
TRAIN_SOURCE = "hope_training/whole_body_tracking/scripts/train.py"
TASK_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV1.yaml"
)
ROBOT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/agibot_a3.py"
)
TRAINING_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/training_contract.py"
)
ACTION_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_actions.py"
)
RUNNER_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/my_on_policy_runner.py"
)
KIT_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh"
)
IDENTITY_REPIN_PRODUCER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_a3_vendor_identity_manifest.py"
)
PROFILE_PINNER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "pin_action_ball_profile_contracts.py"
)
WBT_RELATIVE = Path("hope_training/whole_body_tracking")

_DEFAULT_CONFIG = _R.ACTION_CONFIGS[ACTION_ID]
BOOTSTRAP_SOURCE_COMMIT = _R.require_identity_source_commit(_DEFAULT_CONFIG)
PROFILE_PIN: Mapping[str, str] = {
    "path": (
        "configs/a3_vendor_profile_pins_20260731_r4/"
        "action_ball_profile_pins.v1.509f3812c933.json"
    ),
    "sha256": "509f3812c9336a14ceaf85fd94901f13a0471eb03c985ad0ebea45fa7e5f34c1",
}
PROTOTYPE_PIN: Mapping[str, str] = _R.require_materialized_pin(
    _DEFAULT_CONFIG.identity_prototype,
    action_id=ACTION_ID,
    layer="identity prototype",
)
RECEIPT_PIN: Mapping[str, str] = _R.require_materialized_pin(
    _DEFAULT_CONFIG.identity_repin_receipt,
    action_id=ACTION_ID,
    layer="identity repin receipt",
)
SOURCE_MANIFEST_PIN: Mapping[str, str] = _R.stable_pin(
    _DEFAULT_CONFIG.stable_source_manifest
)
SOURCE_PROTOTYPE_PIN: Mapping[str, str] = _R.stable_pin(
    _DEFAULT_CONFIG.stable_source_prototype
)
PRODUCER_PIN: Mapping[str, str] = {
    **_R.require_materialized_pin(
        _DEFAULT_CONFIG.identity_repin_producer,
        action_id=ACTION_ID,
        layer="identity repin producer",
    )
}
LEGACY_REGISTRY_FREE_PRODUCER_SHA256 = (
    "a1df3e9154ecd895e0f2f3de8f9ceaf80414bab3a0cf9abb43ed7052e58ba752"
)
PINNER_PIN: Mapping[str, str] = {
    "path": PROFILE_PINNER_SOURCE,
    "sha256": "69fc50c850d4dc1bdae6b2e138c63b2437e45cee14ad861f2bbb958f78fdcfc1",
}
SOLVER_PROFILE_SHA256 = (
    "f89587db587f6a418cde1d1fd41f16d60533f8748c1c66701075473eb0bd6971"
)
PHYSICS_PROFILE_SHA256 = (
    "aa5c9085f9b48ca65b3a0ee2cbb35588a5e85a08e84dc3f2ce552d3ef4af85b7"
)
GEOMETRY_PAYLOAD_SHA256 = (
    "3e91be97fcf9c23be1e34b8fd3d6916e8cd1d66ea48e13ea35cedb6d8c839e29"
)
SOLVER_SOURCE_NAMES = (
    "hope_commands.py",
    "continuous_questions.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "virtual_ball.py",
    "counter_rally.py",
    "counter_rally_torch.py",
)
MDP_SOURCE_DIRECTORY = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)

MOTION_PIN: Mapping[str, str] = _R.stable_pin(_DEFAULT_CONFIG.stable_motion)
MANIFEST_PIN: Mapping[str, str] = _R.require_materialized_pin(
    _DEFAULT_CONFIG.identity_manifest,
    action_id=ACTION_ID,
    layer="identity manifest",
)

_SPEC_KEYS = (
    "schema_version",
    "kind",
    "source",
    "action_id",
    "motion",
    "manifest",
    "expected_effective_reward_recipe_sha256",
    "policy_contract_sha256",
    "seed",
    "stage",
    "num_envs",
    "max_iterations",
    "save_interval",
    "gpu",
    "namespace",
    "log_path",
)
_SOURCE_KEYS = ("checkout", "commit_sha", "isaac_python")
_PIN_KEYS = ("path", "sha256")
_ACTION_REGISTRY_PIN_KEYS = (
    "path",
    "action_id",
    "source_identity_sha256",
)

LaunchRefused = _S.LaunchRefused
canonical_sha256 = _S.canonical_sha256


def _action_identity_pins(action_id: object) -> dict[str, Any]:
    """Resolve one reviewed identity chain without accepting arbitrary pins."""

    try:
        config = _R.get_action_config(action_id)
        return {
            "config": config,
            "motion": dict(_R.stable_pin(config.stable_motion)),
            "source_manifest": dict(
                _R.stable_pin(config.stable_source_manifest)
            ),
            "source_prototype": dict(
                _R.stable_pin(config.stable_source_prototype)
            ),
            "prototype": dict(
                _R.require_materialized_pin(
                    config.identity_prototype,
                    action_id=config.action_id,
                    layer="identity prototype",
                )
            ),
            "receipt": dict(
                _R.require_materialized_pin(
                    config.identity_repin_receipt,
                    action_id=config.action_id,
                    layer="identity repin receipt",
                )
            ),
            "manifest": dict(
                _R.require_materialized_pin(
                    config.identity_manifest,
                    action_id=config.action_id,
                    layer="identity manifest",
                )
            ),
            "identity_source_commit": _R.require_identity_source_commit(config),
            "producer": dict(
                _R.require_materialized_pin(
                    config.identity_repin_producer,
                    action_id=config.action_id,
                    layer="identity repin producer",
                )
            ),
        }
    except _R.VendorActionRegistryError as exc:
        raise LaunchRefused(str(exc)) from exc


def _validate_stage(
    stage: Any,
    num_envs: Any,
    max_iterations: Any,
    save_interval: Any,
    policy_contract_sha256: Any,
) -> dict[str, Any]:
    if stage not in ("recipe", "smoke") or type(stage) is not str:
        raise LaunchRefused("stage must be exactly 'recipe' or 'smoke'")
    envs = _S._plain_int(num_envs, name="num_envs", minimum=1)
    iterations = _S._plain_int(
        max_iterations, name="max_iterations", minimum=1
    )
    save = _S._plain_int(save_interval, name="save_interval", minimum=1)
    expected_budget = (1, 1, 1) if stage == "recipe" else (1, 2, 1)
    if (envs, iterations, save) != expected_budget:
        explanation = (
            "recipe is exactly 1 env / materialization-only / configured "
            "iteration budget 1 / save interval 1"
            if stage == "recipe"
            else "smoke is exactly 1 env / 2 PPO updates / save interval 1"
        )
        raise LaunchRefused(explanation)
    if stage == "recipe":
        if policy_contract_sha256 is not None:
            raise LaunchRefused(
                "recipe policy_contract_sha256 must be null; the stage "
                "materializes that identity"
            )
        policy_sha = None
    else:
        policy_sha = _S._sha256(
            policy_contract_sha256, name="policy_contract_sha256"
        )
        if policy_sha == RECIPE_SENTINEL_POLICY_SHA256:
            raise LaunchRefused(
                "smoke cannot use the recipe-stage policy SHA sentinel"
            )
    return {
        "stage": stage,
        "num_envs": envs,
        "max_iterations": iterations,
        "save_interval": save,
        "policy_contract_sha256": policy_sha,
    }


def _validate_spec_document(
    document: dict[str, Any], *, namespace_claimed: bool = False
) -> dict[str, Any]:
    row = _S._exact_dict(document, _SPEC_KEYS, name="identity-smoke spec")
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != SPEC_KIND:
        raise LaunchRefused(
            f"identity-smoke spec must be schema {SCHEMA_VERSION} / {SPEC_KIND!r}"
        )
    source = _S._exact_dict(row["source"], _SOURCE_KEYS, name="spec.source")
    checkout = _S._absolute_path(
        source["checkout"], name="spec.source.checkout", must_exist=True
    )
    commit_sha = source["commit_sha"]
    if type(commit_sha) is not str or _S.COMMIT_RE.fullmatch(commit_sha) is None:
        raise LaunchRefused("spec.source.commit_sha must be 40 lowercase hex")
    isaac_python = _S._absolute_path(
        source["isaac_python"], name="spec.source.isaac_python", must_exist=True
    )
    try:
        python_stat = isaac_python.stat()
    except OSError as exc:
        raise LaunchRefused(f"cannot stat Isaac Python: {exc}") from exc
    if not stat.S_ISREG(python_stat.st_mode) or not os.access(isaac_python, os.X_OK):
        raise LaunchRefused("spec.source.isaac_python must be executable")

    action_pins = _action_identity_pins(row["action_id"])
    action_id = action_pins["config"].action_id
    motion = _S._exact_dict(
        row["motion"], _PIN_KEYS, name="spec.motion"
    )
    manifest = _S._exact_dict(
        row["manifest"], _PIN_KEYS, name="spec.manifest"
    )
    if dict(motion) != action_pins["motion"]:
        raise LaunchRefused(
            f"motion must be the fixed tracked code-pinned {action_id} "
            "stable-v2 motion"
        )
    if dict(manifest) != action_pins["manifest"]:
        raise LaunchRefused(
            f"manifest must be the fixed tracked code-pinned {action_id} "
            "vendor-identity N=1 manifest"
        )
    if row["seed"] != SEED or type(row["seed"]) is not int:
        raise LaunchRefused("identity-smoke seed must be exactly 0")
    stage = _validate_stage(
        row["stage"],
        row["num_envs"],
        row["max_iterations"],
        row["save_interval"],
        row["policy_contract_sha256"],
    )
    reward_sha = _S._sha256(
        row["expected_effective_reward_recipe_sha256"],
        name="expected_effective_reward_recipe_sha256",
    )
    if reward_sha != EXPECTED_REWARD_RECIPE_SHA256:
        raise LaunchRefused(
            "expected_effective_reward_recipe_sha256 must equal the fixed "
            "vendor-task inherited ActionBall reward receipt"
        )
    gpu = _S._validate_gpu(row["gpu"])
    namespace = _S._absolute_path(row["namespace"], name="namespace")
    if (
        namespace.name in ("", ".", "..")
        or _S.SAFE_COMPONENT_RE.fullmatch(namespace.name) is None
        or not namespace.name.startswith("a3vendor-identity-")
    ):
        raise LaunchRefused(
            "namespace basename must start with 'a3vendor-identity-' and be safe"
        )
    log_path = _S._absolute_path(row["log_path"], name="log_path")
    if log_path != namespace / "run.log":
        raise LaunchRefused("log_path must be exactly <namespace>/run.log")
    if os.path.lexists(namespace):
        if not namespace_claimed:
            raise LaunchRefused(
                f"run namespace already exists and is permanently spent: {namespace}"
            )
        try:
            namespace_stat = namespace.lstat()
        except OSError as exc:
            raise LaunchRefused(f"claimed namespace cannot be inspected: {exc}") from exc
        if (
            not stat.S_ISDIR(namespace_stat.st_mode)
            or namespace.resolve(strict=True) != namespace
        ):
            raise LaunchRefused("claimed namespace must remain a real directory")
    elif namespace_claimed:
        raise LaunchRefused("claimed namespace vanished before trainer exec")
    parent = namespace.parent
    if not parent.exists() or parent.resolve(strict=True) != parent:
        raise LaunchRefused(
            "namespace parent must be an existing real absolute directory"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": commit_sha,
            "isaac_python": str(isaac_python),
        },
        "action_id": action_id,
        "motion": action_pins["motion"],
        "manifest": action_pins["manifest"],
        "expected_effective_reward_recipe_sha256": reward_sha,
        "policy_contract_sha256": stage["policy_contract_sha256"],
        "seed": SEED,
        "stage": stage["stage"],
        "num_envs": stage["num_envs"],
        "max_iterations": stage["max_iterations"],
        "save_interval": stage["save_interval"],
        "gpu": gpu,
        "namespace": str(namespace),
        "log_path": str(log_path),
    }


def _validate_runtime_sources(
    checkout: Path, commit_sha: str, action_id: str = ACTION_ID
) -> dict[str, dict[str, Any]]:
    action_pins = _action_identity_pins(action_id)
    result: dict[str, dict[str, Any]] = {}
    sources = (
        (LAUNCHER_SOURCE, "A3 vendor identity-smoke launcher", None),
        (ACTION_REGISTRY_SOURCE, "A3 vendor action registry", None),
        (SAFETY_SOURCE, "identity-smoke launch safety implementation", None),
        (TRAIN_SOURCE, "training entrypoint", None),
        (TASK_SOURCE, f"immutable task profile {TASK_PROFILE_ID}", None),
        (ROBOT_SOURCE, "A3 vendor actuator source", None),
        (TRAINING_CONTRACT_SOURCE, "training-contract implementation", None),
        (ACTION_SOURCE, "control-step action-delay implementation", None),
        (RUNNER_SOURCE, "runtime ABI/std receipt implementation", None),
        (KIT_LAUNCHER_SOURCE, "Kit locked launcher", None),
        (
            action_pins["producer"]["path"],
            "identity-bootstrap repin producer",
            action_pins["producer"],
        ),
        (PROFILE_PINNER_SOURCE, "formal profile-pins producer", PINNER_PIN),
    )
    for relative, label, fixed_pin in sources:
        pin = (
            dict(fixed_pin)
            if fixed_pin is not None
            else {"path": relative, "sha256": _S.sha256_file(checkout / relative)}
        )
        normalized, _path = _S._verify_tracked_file(
            checkout, commit_sha, pin, name=label
        )
        result[label] = normalized
    if _THIS_FILE != checkout / LAUNCHER_SOURCE:
        raise LaunchRefused(
            "running identity-smoke launcher is not the selected checkout path"
        )
    return result


def _canonical_ascii_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _load_canonical_tracked_json(
    checkout: Path,
    commit_sha: str,
    pin: Mapping[str, str],
    *,
    name: str,
    require_canonical: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    normalized, document = _S._load_tracked_json(
        checkout, commit_sha, dict(pin), name=name
    )
    raw = (checkout / normalized["path"]).read_bytes()
    if require_canonical and raw != _S._canonical_bytes(document) + b"\n":
        raise LaunchRefused(f"{name} must be canonical JSON plus newline")
    return normalized, document, raw


def _verify_pin_at_bootstrap_source(
    checkout: Path,
    pin: Mapping[str, str],
    *,
    name: str,
    source_commit: str = BOOTSTRAP_SOURCE_COMMIT,
) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "show",
            f"{source_commit}:{pin['path']}",
        ],
        capture_output=True,
        check=False,
    )
    observed = hashlib.sha256(result.stdout).hexdigest()
    if result.returncode != 0 or observed != pin["sha256"]:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise LaunchRefused(
            f"{name} is not exact at identity source commit {source_commit}: "
            f"expected={pin['sha256']}, observed={observed}, detail={detail}"
        )


def _without_prototype_repin_fields(
    document: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    result = deepcopy(dict(document))
    provenance = result.get("provenance")
    if type(provenance) is not dict:
        raise LaunchRefused(f"{name} provenance must be an object")
    for key in ("producer", "producer_source_sha256", "profile_pins"):
        provenance.pop(key, None)
    return result


def _without_manifest_repin_fields(document: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(document))
    for key in (
        "manifest_id",
        "solver_profile_sha256",
        "physics_profile_sha256",
        "prototype",
        "notes",
    ):
        result.pop(key, None)
    return result


def _profile_provenance_pin() -> dict[str, str]:
    return {
        **dict(PROFILE_PIN),
        "solver_profile_sha256": SOLVER_PROFILE_SHA256,
        "physics_profile_sha256": PHYSICS_PROFILE_SHA256,
        "geometry_payload_sha256": GEOMETRY_PAYLOAD_SHA256,
    }


def _identity_receipt_requires_registry(action_id: str) -> bool:
    producer_pin = _action_identity_pins(action_id)["producer"]
    return (
        action_id != ACTION_ID
        or producer_pin["sha256"] != LEGACY_REGISTRY_FREE_PRODUCER_SHA256
    )


def _expected_repin_receipt(
    action_id: str = ACTION_ID,
    *,
    action_registry_pin: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    action_pins = _action_identity_pins(action_id)
    inputs = {
        "source_manifest": action_pins["source_manifest"],
        "source_prototype": action_pins["source_prototype"],
        "profile_pins": _profile_provenance_pin(),
        "producer": dict(action_pins["producer"]),
    }
    registry_required = _identity_receipt_requires_registry(action_id)
    if registry_required:
        if action_registry_pin is None:
            raise LaunchRefused(
                "identity-bootstrap receipt requires its tracked action registry pin"
            )
        inputs["action_registry"] = dict(
            _S._exact_dict(
                action_registry_pin,
                _ACTION_REGISTRY_PIN_KEYS,
                name="identity-bootstrap action registry pin",
            )
        )
    return {
        "schema_version": 1,
        "kind": "agibot_a3_vendor_identity_manifest_repin_receipt_v1",
        "purpose": "identity_bootstrap_repin",
        "source_commit": action_pins["identity_source_commit"],
        "inputs": inputs,
        "outputs": {
            "prototype": action_pins["prototype"],
            "manifest": action_pins["manifest"],
        },
        "allowed_changes": {
            "prototype": [
                "provenance.producer",
                "provenance.producer_source_sha256",
                "provenance.profile_pins",
            ],
            "manifest": [
                "manifest_id",
                "solver_profile_sha256",
                "physics_profile_sha256",
                "prototype",
                "notes",
            ],
        },
        "invariants": {
            "action_order_unchanged": True,
            "action_uid_unchanged": True,
            "ball_profile_unchanged": True,
            "counter_rally_objective_unchanged": True,
            "motion_binding_unchanged": True,
            "only_allowlisted_fields_changed": True,
            "prototype_motion_provenance_unchanged": True,
            "prototype_scopes_geometry_unchanged": True,
            "prototype_source_manifest_provenance_unchanged": True,
        },
        "authorization": {
            "identity_bootstrap_repin": True,
            "formal_bundle": False,
            "contact_admission": False,
            "dynamic_ready": False,
            "training": False,
            "deployment": False,
            "hardware": False,
        },
    }


def _validate_bootstrap_repin_documents(
    *,
    profile: Mapping[str, Any],
    prototype: Mapping[str, Any],
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source_prototype: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_map: Mapping[str, str],
    action_id: str = ACTION_ID,
    action_registry_pin: Mapping[str, str] | None = None,
) -> None:
    action_pins = _action_identity_pins(action_id)
    config = action_pins["config"]
    producer_pin = action_pins["producer"]
    profile_keys = {
        "schema_version",
        "kind",
        "source_authority",
        "cfg",
        "geometry",
        "venue_yaml",
        "venue_yaml_sha256",
        "planes",
        "solver_implementation_source_sha256",
        "contact_geometry",
        "counter_rally",
        "physics_profile_sha256",
        "solver_profile_sha256",
        "physics_payload",
        "solver_payload",
    }
    physics_payload = profile.get("physics_payload")
    solver_payload = profile.get("solver_payload")
    geometry = profile.get("contact_geometry")
    expected_authority = {
        "schema_version": 1,
        "authority": "external_exact_commit_subset_blob_map_v1",
        "commit_binding": "external_preexec_immutable_launch_capsule_v1",
        "embedded_commit": False,
        "source_blob_map_sha256": _canonical_ascii_sha256(dict(source_map)),
    }
    if (
        set(profile) != profile_keys
        or profile.get("schema_version") != 1
        or profile.get("kind") != "whole_body_tracking.action_ball.profile_pins"
        or type(physics_payload) is not dict
        or type(solver_payload) is not dict
        or type(geometry) is not dict
        or set(geometry) != {"payload", "sha256"}
        or profile.get("solver_profile_sha256") != SOLVER_PROFILE_SHA256
        or profile.get("physics_profile_sha256") != PHYSICS_PROFILE_SHA256
        or geometry.get("sha256") != GEOMETRY_PAYLOAD_SHA256
        or _canonical_ascii_sha256(physics_payload) != PHYSICS_PROFILE_SHA256
        or _canonical_ascii_sha256(solver_payload) != SOLVER_PROFILE_SHA256
        or _canonical_ascii_sha256(geometry.get("payload"))
        != GEOMETRY_PAYLOAD_SHA256
        or profile.get("solver_implementation_source_sha256")
        != dict(source_map)
        or solver_payload.get("implementation_source_sha256")
        != dict(source_map)
        or solver_payload.get("physics_profile_sha256")
        != PHYSICS_PROFILE_SHA256
        or solver_payload.get("contact_geometry") != geometry
        or profile.get("source_authority") != expected_authority
    ):
        raise LaunchRefused(
            "identity-bootstrap profile pins fail payload/source-map closure"
        )

    if receipt != _expected_repin_receipt(
        action_id, action_registry_pin=action_registry_pin
    ):
        raise LaunchRefused(
            "identity-bootstrap repin receipt differs from exact authorization"
        )

    if (
        _without_prototype_repin_fields(
            prototype, name="identity-bootstrap prototype"
        )
        != _without_prototype_repin_fields(
            source_prototype, name="source prototype"
        )
    ):
        raise LaunchRefused(
            "identity-bootstrap prototype changed outside its allowlist"
        )
    provenance = prototype.get("provenance")
    source_provenance = source_prototype.get("provenance")
    velocity_contract = prototype.get("velocity_contract")
    if (
        prototype.get("schema_version") != 2
        or type(provenance) is not dict
        or type(source_provenance) is not dict
        or type(velocity_contract) is not dict
        or provenance.get("producer") != Path(producer_pin["path"]).name
        or provenance.get("producer_source_sha256") != producer_pin["sha256"]
        or provenance.get("profile_pins") != _profile_provenance_pin()
        or provenance.get("motion") != action_pins["motion"]
        or provenance.get("motion") != source_provenance.get("motion")
        or provenance.get("source_manifest")
        != source_provenance.get("source_manifest")
        or provenance.get("geometry_source_file_sha256")
        != source_map.get("racket_contact_geometry.py")
        or velocity_contract.get("geometry_source_sha256")
        != GEOMETRY_PAYLOAD_SHA256
        or prototype.get("scopes") != source_prototype.get("scopes")
        or prototype.get("derived_sha256")
        != source_prototype.get("derived_sha256")
    ):
        raise LaunchRefused(
            "identity-bootstrap prototype provenance/geometry is not exact"
        )

    if _without_manifest_repin_fields(manifest) != _without_manifest_repin_fields(
        source_manifest
    ):
        raise LaunchRefused(
            "identity-bootstrap manifest changed outside its allowlist"
        )
    actions = manifest.get("actions")
    source_actions = source_manifest.get("actions")
    prototype_binding = manifest.get("prototype")
    if (
        manifest.get("schema_version") != 3
        or manifest.get("action_order") != [action_id]
        or manifest.get("mobility_mode") != "no_move"
        or manifest.get("solver_profile_sha256") != SOLVER_PROFILE_SHA256
        or manifest.get("physics_profile_sha256") != PHYSICS_PROFILE_SHA256
        or prototype_binding
        != {**action_pins["prototype"], "scope": config.scope}
        or not isinstance(actions, list)
        or len(actions) != 1
        or not isinstance(source_actions, list)
        or len(source_actions) != 1
        or type(actions[0]) is not dict
        or type(source_actions[0]) is not dict
        or actions[0].get("action_id") != action_id
        or actions[0].get("action_uid") != source_actions[0].get("action_uid")
        or actions[0].get("motion_path") != action_pins["motion"]["path"]
        or actions[0].get("motion_sha256") != action_pins["motion"]["sha256"]
        or actions[0].get("ball_profile")
        != source_actions[0].get("ball_profile")
        or manifest.get("counter_rally_objective")
        != source_manifest.get("counter_rally_objective")
    ):
        raise LaunchRefused(
            "identity-bootstrap manifest action/motion/objective binding is not exact"
        )


def _validate_bootstrap_repin_artifacts(
    checkout: Path, commit_sha: str, action_id: str = ACTION_ID
) -> dict[str, Any]:
    action_pins = _action_identity_pins(action_id)
    identity_source_commit = action_pins["identity_source_commit"]
    action_registry_pin = None
    if _identity_receipt_requires_registry(action_id):
        action_registry_candidate = {
            "path": ACTION_REGISTRY_SOURCE,
            "sha256": _S.sha256_file(checkout / ACTION_REGISTRY_SOURCE),
        }
        _normalized_registry, _action_registry_path = _S._verify_tracked_file(
            checkout,
            commit_sha,
            action_registry_candidate,
            name="identity-bootstrap action registry",
        )
        action_registry_pin = dict(
            _R.action_source_registry_pin(action_pins["config"])
        )
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "merge-base",
            "--is-ancestor",
            identity_source_commit,
            commit_sha,
        ],
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise LaunchRefused(
            "identity-bootstrap source commit is not an ancestor of launch commit"
        )

    for pin, name in (
        (PROFILE_PIN, "identity-bootstrap profile pins"),
        (action_pins["source_manifest"], "identity-bootstrap source manifest"),
        (action_pins["source_prototype"], "identity-bootstrap source prototype"),
        (action_pins["producer"], "identity-bootstrap producer source"),
        (PINNER_PIN, "formal profile-pins producer source"),
    ):
        _verify_pin_at_bootstrap_source(
            checkout,
            pin,
            name=name,
            source_commit=identity_source_commit,
        )

    profile_pin, profile, profile_raw = _load_canonical_tracked_json(
        checkout,
        commit_sha,
        PROFILE_PIN,
        name="identity-bootstrap profile pins",
        require_canonical=False,
    )
    prototype_pin, prototype, _prototype_raw = _load_canonical_tracked_json(
        checkout,
        commit_sha,
        action_pins["prototype"],
        name="identity-bootstrap prototype",
    )
    manifest_pin, manifest, _manifest_raw = _load_canonical_tracked_json(
        checkout,
        commit_sha,
        action_pins["manifest"],
        name="identity-bootstrap manifest",
    )
    receipt_pin, receipt, _receipt_raw = _load_canonical_tracked_json(
        checkout,
        commit_sha,
        action_pins["receipt"],
        name="identity-bootstrap repin receipt",
    )
    _source_manifest_pin, source_manifest, _source_manifest_raw = (
        _load_canonical_tracked_json(
            checkout,
            commit_sha,
            action_pins["source_manifest"],
            name="identity-bootstrap source manifest",
            require_canonical=False,
        )
    )
    _source_prototype_pin, source_prototype, _source_prototype_raw = (
        _load_canonical_tracked_json(
            checkout,
            commit_sha,
            action_pins["source_prototype"],
            name="identity-bootstrap source prototype",
            require_canonical=False,
        )
    )

    _S._verify_tracked_file(
        checkout,
        commit_sha,
        dict(action_pins["producer"]),
        name="identity-bootstrap producer source",
    )
    _S._verify_tracked_file(
        checkout,
        commit_sha,
        dict(PINNER_PIN),
        name="formal profile-pins producer source",
    )
    source_map: dict[str, str] = {}
    for filename in SOLVER_SOURCE_NAMES:
        relative = f"{MDP_SOURCE_DIRECTORY}/{filename}"
        pin = {"path": relative, "sha256": _S.sha256_file(checkout / relative)}
        normalized, _source_path = _S._verify_tracked_file(
            checkout,
            commit_sha,
            pin,
            name=f"identity-bootstrap solver source {filename}",
        )
        source_map[filename] = normalized["sha256"]

    pinner_env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    reproduced = subprocess.run(
        [
            sys.executable,
            str(checkout / PINNER_PIN["path"]),
            "--repo-root",
            str(checkout),
            "--source-rev",
            commit_sha,
        ],
        cwd=checkout,
        env=pinner_env,
        capture_output=True,
        check=False,
    )
    if reproduced.returncode != 0 or reproduced.stdout != profile_raw:
        detail = reproduced.stderr.decode("utf-8", "replace").strip()
        raise LaunchRefused(
            "official profile pinner does not reproduce the pinned profile: "
            f"returncode={reproduced.returncode}, detail={detail}"
        )

    _validate_bootstrap_repin_documents(
        profile=profile,
        prototype=prototype,
        manifest=manifest,
        receipt=receipt,
        source_prototype=source_prototype,
        source_manifest=source_manifest,
        source_map=source_map,
        action_id=action_id,
        action_registry_pin=action_registry_pin,
    )
    return {
        "action_id": action_id,
        "source_commit": identity_source_commit,
        "profile_pins": profile_pin,
        "prototype": prototype_pin,
        "manifest": manifest_pin,
        "receipt": receipt_pin,
        "solver_profile_sha256": SOLVER_PROFILE_SHA256,
        "physics_profile_sha256": PHYSICS_PROFILE_SHA256,
        "geometry_payload_sha256": GEOMETRY_PAYLOAD_SHA256,
        "producer": dict(action_pins["producer"]),
        "profile_pinner": dict(PINNER_PIN),
        **(
            {"action_registry": action_registry_pin}
            if action_registry_pin is not None
            else {}
        ),
    }


def _validate_scientific_inputs(
    checkout: Path,
    commit_sha: str,
    motion_pin: Mapping[str, Any],
    manifest_pin: Mapping[str, Any],
    action_id: str = ACTION_ID,
) -> dict[str, Any]:
    action_pins = _action_identity_pins(action_id)
    normalized_motion, _motion_path = _S._verify_tracked_file(
        checkout,
        commit_sha,
        dict(motion_pin),
        name="identity-smoke stable-v2 motion",
    )
    normalized_manifest, manifest = _S._load_tracked_json(
        checkout,
        commit_sha,
        dict(manifest_pin),
        name="identity-smoke N=1 manifest",
    )
    actions = manifest.get("actions") if type(manifest) is dict else None
    if (
        normalized_motion != action_pins["motion"]
        or normalized_manifest != action_pins["manifest"]
        or manifest.get("schema_version") != 3
        or manifest.get("action_order") != [action_id]
        or manifest.get("mobility_mode") != "no_move"
        or not isinstance(actions, list)
        or len(actions) != 1
        or type(actions[0]) is not dict
        or actions[0].get("action_id") != action_id
        or actions[0].get("motion_path") != action_pins["motion"]["path"]
        or actions[0].get("motion_sha256") != action_pins["motion"]["sha256"]
    ):
        raise LaunchRefused(
            f"identity-smoke manifest is not exact N=1 {action_id}/no_move/stable-v2"
        )
    return {"motion": normalized_motion, "manifest": normalized_manifest}


def _rsl_output_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    namespace_name = Path(spec["namespace"]).name
    suffix = f"_{namespace_name}-{DIAGNOSTIC_SUFFIX}"
    root = (
        Path(spec["source"]["checkout"])
        / WBT_RELATIVE
        / "logs/rsl_rl"
        / EXPERIMENT_NAME
    )
    if spec["stage"] == "recipe":
        return {
            "namespace_recipe": str(Path(spec["namespace"]) / RECIPE_FILENAME),
            "ppo_update_count": 0,
            "rsl_experiment_root": str(root),
            "rsl_run_suffix": suffix,
            "training_contract": None,
            "checkpoints": [],
            "required_runtime_log_events": [],
        }
    return {
        "namespace_recipe": None,
        "ppo_update_count": 2,
        "rsl_experiment_root": str(root),
        "rsl_run_suffix": suffix,
        "training_contract": "params/training_contract.json",
        "checkpoints": ["model_0.pt", "model_1.pt"],
        "required_runtime_log_events": [
            "HOPE_CONTROL_STEP_ACTION_DELAY_RUNTIME_JSON=",
            "HOPE_RSL_RL_RUNTIME_ABI_JSON=",
            "HOPE_POLICY_STD_UPDATE_JSON=",
        ],
    }


def _check_rsl_namespace_available(spec: Mapping[str, Any]) -> None:
    output = _rsl_output_contract(spec)
    root = Path(output["rsl_experiment_root"])
    if not os.path.lexists(root):
        return
    if root.resolve(strict=True) != root or not root.is_dir():
        raise LaunchRefused("identity-smoke RSL experiment root is not real")
    spent = sorted(
        child.name
        for child in root.iterdir()
        if child.name.endswith(output["rsl_run_suffix"])
    )
    if spent:
        raise LaunchRefused(f"identity-smoke trainer run_name is spent: {spent[0]}")


def _build_training_argv(
    spec: Mapping[str, Any], scientific_inputs: Mapping[str, Any]
) -> list[str]:
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / WBT_RELATIVE
    motion = checkout / scientific_inputs["motion"]["path"]
    manifest = checkout / scientific_inputs["manifest"]["path"]
    json_list = lambda values: json.dumps(  # noqa: E731
        values, separators=(",", ":"), ensure_ascii=False
    )
    policy_sha = (
        RECIPE_SENTINEL_POLICY_SHA256
        if spec["stage"] == "recipe"
        else spec["policy_contract_sha256"]
    )
    argv = [
        spec["source"]["isaac_python"],
        str(wbt / "scripts/train.py"),
        f"task={TASK_PROFILE_ID}",
        "algo=ppo",
        "algo.policy.init_noise_std=0.02",
        "action_ball_shared_ready_bootstrap=true",
        "headless=true",
        "logger=tensorboard",
        "video=false",
        "device=cuda:0",
        f"seed={SEED}",
        f"num_envs={spec['num_envs']}",
        f"max_iterations={spec['max_iterations']}",
        f"algo.runner.save_interval={spec['save_interval']}",
        f"run_name={Path(spec['namespace']).name}",
        f"task.experiment_name={EXPERIMENT_NAME}",
        (
            "expected_effective_reward_recipe_sha256="
            f"{spec['expected_effective_reward_recipe_sha256']}"
        ),
        f"task.actor_obs_contract={ACTOR_OBS_CONTRACT}",
        "task.rewards.full_body_mimic=false",
        f"motion_file={json_list([str(motion)])}",
        f"task.racket.clip_names={json_list([spec['action_id']])}",
        "task.racket.target_mode=action_ball",
        f"task.racket.action_ball_manifest_path={manifest}",
        (
            "task.racket.action_ball_manifest_sha256="
            f"{scientific_inputs['manifest']['sha256']}"
        ),
        f"task.racket.action_ball_policy_contract_sha256={policy_sha}",
        "task.racket.action_ball_diagnostic_unauthorized=true",
        "+task.racket.reference_guard_mode=metrics_only",
        f"task.racket.action_ball_seed={SEED}",
        "task.racket.question_bank=",
        "task.racket.question_bank_allow_legacy=false",
        "task.racket.cq_anchor_bank=",
        "task.racket.exam_bank=",
    ]
    if spec["stage"] == "recipe":
        argv.append(
            "action_ball_policy_recipe_output_path="
            f"{Path(spec['namespace']) / RECIPE_FILENAME}"
        )
    forbidden = (
        "action_ball_dynamic_ready",
        "checkpoint_path=",
        "stable_ready_plant",
        "task.push",
        "push_robot",
        "randomize_pd_gains",
        "kp_gain_range",
        "kd_gain_range",
        "control_step_action_delay_",
    )
    if any(fragment in item for item in argv for fragment in forbidden):
        raise LaunchRefused(
            "identity-smoke argv overrides a forbidden resume/dynamic/task-owned axis"
        )
    return argv


def build_plan(spec_path: Path) -> dict[str, Any]:
    absolute = _S._absolute_path(str(spec_path), name="--spec", must_exist=True)
    _S._stable_regular_file(absolute, name="identity-smoke spec")
    raw = absolute.read_bytes()
    document = _S._strict_json_bytes(raw, name="identity-smoke spec")
    if raw != _S._canonical_bytes(document) + b"\n":
        raise LaunchRefused("identity-smoke spec must be canonical JSON plus newline")
    spec = _validate_spec_document(document)
    checkout = Path(spec["source"]["checkout"])
    commit_sha = spec["source"]["commit_sha"]
    source = _S._verify_clean_source(checkout, commit_sha)
    runtime_sources = _validate_runtime_sources(
        checkout, commit_sha, spec["action_id"]
    )
    bootstrap_repin_artifacts = _validate_bootstrap_repin_artifacts(
        checkout, commit_sha, spec["action_id"]
    )
    runtime_assets = _S._validate_runtime_asset_environment()
    scientific_inputs = _validate_scientific_inputs(
        checkout,
        commit_sha,
        spec["motion"],
        spec["manifest"],
        spec["action_id"],
    )
    _check_rsl_namespace_available(spec)
    argv = _build_training_argv(spec, scientific_inputs)
    output_contract = _rsl_output_contract(spec)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "diagnostic_unauthorized": True,
        "identity_materialization_only": True,
        "formal_evidence_prohibited": True,
        "curriculum_promotion_prohibited": True,
        "resume_prohibited": True,
        "export_prohibited": True,
        "judge_prohibited": True,
        "hardware_authority_prohibited": True,
        "long_stage_prohibited": True,
        "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
        "fixed_identity": {
            "task_profile": TASK_PROFILE_ID,
            "action_id": spec["action_id"],
            "scope": _action_identity_pins(spec["action_id"])["config"].scope,
            "actor_obs_contract": ACTOR_OBS_CONTRACT,
            "bootstrap": "shared_ready_fresh",
            "dynamic_ready": False,
        },
        "source": source,
        "runtime_sources": runtime_sources,
        "bootstrap_repin_artifacts": bootstrap_repin_artifacts,
        "runtime_assets": runtime_assets,
        "scientific_inputs": scientific_inputs,
        "output_contract": output_contract,
        "boot_marker": (
            "ACTION_BALL_POLICY_RECIPE_MATERIALIZED"
            if spec["stage"] == "recipe"
            else "Learning iteration"
        ),
        "training_argv": argv,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": canonical_sha256(payload),
        "canonical_payload": payload,
    }


def _load_internal_claim(
    claim_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute = _S._absolute_path(
        str(claim_path), name="internal claim", must_exist=True
    )
    _S._stable_regular_file(absolute, name="internal identity-smoke claim")
    raw = absolute.read_bytes()
    plan = _S._strict_json_bytes(raw, name="internal identity-smoke claim")
    if raw != _S._canonical_bytes(plan) + b"\n":
        raise LaunchRefused("internal identity-smoke claim is not canonical")
    outer = _S._exact_dict(
        plan,
        ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
        name="internal identity-smoke claim",
    )
    if (
        outer["schema_version"] != SCHEMA_VERSION
        or outer["kind"] != CLAIM_KIND
        or outer["launch_claim_sha256"] != expected_sha256
        or canonical_sha256(outer["canonical_payload"]) != expected_sha256
    ):
        raise LaunchRefused("internal identity-smoke claim digest differs")
    payload = outer["canonical_payload"]
    if type(payload) is not dict or payload.get("kind") != CLAIM_KIND:
        raise LaunchRefused("internal identity-smoke payload kind differs")
    return plan, payload


def _internal_exec(claim_path: Path, expected_sha256: str, lock_fd: int) -> int:
    _plan, payload = _load_internal_claim(claim_path, expected_sha256)
    spec = _validate_spec_document(payload["spec"], namespace_claimed=True)
    checkout = Path(spec["source"]["checkout"])
    commit_sha = spec["source"]["commit_sha"]
    source = _S._verify_clean_source(checkout, commit_sha)
    if source != payload["source"]:
        raise LaunchRefused("source identity drifted after namespace claim")
    runtime = _validate_runtime_sources(checkout, commit_sha, spec["action_id"])
    if runtime != payload["runtime_sources"]:
        raise LaunchRefused("runtime source identity drifted after namespace claim")
    bootstrap_repin_artifacts = _validate_bootstrap_repin_artifacts(
        checkout, commit_sha, spec["action_id"]
    )
    if bootstrap_repin_artifacts != payload.get("bootstrap_repin_artifacts"):
        raise LaunchRefused(
            "identity-bootstrap repin artifacts drifted after namespace claim"
        )
    runtime_assets = _S._validate_runtime_asset_claim(payload.get("runtime_assets"))
    scientific_inputs = _validate_scientific_inputs(
        checkout,
        commit_sha,
        spec["motion"],
        spec["manifest"],
        spec["action_id"],
    )
    if scientific_inputs != payload["scientific_inputs"]:
        raise LaunchRefused("scientific inputs drifted after namespace claim")
    argv = _build_training_argv(spec, scientific_inputs)
    if argv != payload["training_argv"]:
        raise LaunchRefused("training argv differs from immutable claim")
    if _rsl_output_contract(spec) != payload["output_contract"]:
        raise LaunchRefused("output contract differs from immutable claim")

    lock_path = Path(spec["gpu"]["lock_path"])
    try:
        lock_stat = os.fstat(lock_fd)
        path_stat = lock_path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"inherited GPU lock cannot be verified: {exc}") from exc
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or (lock_stat.st_dev, lock_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise LaunchRefused("inherited GPU lock differs from shared lock path")
    try:
        _S.fcntl.flock(lock_fd, _S.fcntl.LOCK_EX | _S.fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise LaunchRefused("inherited GPU lock is not owned by this launch") from exc
    second_gpu = _S._verify_gpu_empty(
        spec["gpu"]["index"], spec["gpu"]["uuid"]
    )
    _S._write_exclusive_json(
        Path(spec["namespace"]) / "pre_exec_gpu_admission.json",
        {
            "schema_version": 1,
            "kind": "a3_vendor_identity_smoke_pre_exec_gpu_admission_v1",
            "launch_claim_sha256": expected_sha256,
            "gpu": second_gpu,
        },
    )
    wbt = checkout / WBT_RELATIVE
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
        **_S._runtime_asset_exec_environment(runtime_assets),
    }
    os.chdir(wbt)
    os.execve(argv[0], argv, environment)
    raise AssertionError("os.execve returned")


def _build_internal_exec_command(
    spec: Mapping[str, Any],
    checkout: Path,
    namespace: Path,
    expected_claim_sha256: str,
    lock_fd: int,
) -> list[str]:
    """Keep the reviewed venv entry path instead of resolving its Python symlink.

    The Pod venv entry is intentionally a symlink to the system interpreter;
    resolving it drops the venv's site-packages and makes the formal profile
    pinner fail before Kit starts.  The spec path was already normalized,
    checked executable, and sealed into the launch claim.
    """

    return [
        spec["source"]["isaac_python"],
        str(checkout / LAUNCHER_SOURCE),
        "_exec",
        "--claim",
        str(namespace / "launch_claim.json"),
        "--claim-sha256",
        expected_claim_sha256,
        "--gpu-lock-fd",
        str(lock_fd),
    ]


def launch(plan: dict[str, Any], *, confirm_claim: str) -> dict[str, Any]:
    expected = _S._sha256(confirm_claim, name="--confirm-claim")
    if expected != plan["launch_claim_sha256"]:
        raise LaunchRefused(
            "--confirm-claim differs from the freshly recomputed plan"
        )
    payload = plan["canonical_payload"]
    spec = payload["spec"]
    checkout = Path(spec["source"]["checkout"])
    _S._verify_clean_source(checkout, spec["source"]["commit_sha"])
    _S._validate_runtime_asset_claim(payload.get("runtime_assets"))
    lock_fd = _S._open_gpu_lock(Path(spec["gpu"]["lock_path"]))
    namespace: Path | None = None
    try:
        first_gpu = _S._verify_gpu_empty(
            spec["gpu"]["index"], spec["gpu"]["uuid"]
        )
        namespace = _S._claim_namespace(plan)
        _S._write_exclusive_json(
            namespace / "pre_launch_gpu_admission.json",
            {
                "schema_version": 1,
                "kind": "a3_vendor_identity_smoke_pre_launch_gpu_admission_v1",
                "launch_claim_sha256": expected,
                "gpu": first_gpu,
            },
        )
        launcher = checkout / KIT_LAUNCHER_SOURCE
        state_path = Path(spec["log_path"] + ".launch")
        internal_command = _build_internal_exec_command(
            spec, checkout, namespace, expected, lock_fd
        )
        environment = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": os.environ.get("HOME", "/root"),
            "LANG": "C",
            "LC_ALL": "C",
            "KIT_BOOT_MARKER": payload["boot_marker"],
            "KIT_BOOT_TIMEOUT_S": "2700",
            "KIT_BOOT_STALE_TIMEOUT_S": "1800",
            "KIT_BOOT_POLL_S": "5",
            "KIT_BOOT_STATE_FILE": str(state_path),
        }
        result = subprocess.run(
            [str(launcher), spec["log_path"], *internal_command],
            cwd=checkout / WBT_RELATIVE,
            env=environment,
            pass_fds=(lock_fd,),
            check=False,
        )
        if result.returncode != 0:
            raise LaunchRefused(
                f"locked Kit launcher returned {result.returncode}; "
                f"namespace remains spent at {namespace}"
            )
        return {
            "schema_version": 1,
            "kind": RESULT_KIND,
            "launch_claim_sha256": expected,
            "stage": spec["stage"],
            "namespace": str(namespace),
            "log_path": spec["log_path"],
            "state_path": str(state_path),
            "gpu": spec["gpu"],
            "output_contract": payload["output_contract"],
            "diagnostic_unauthorized": True,
            "accepted": True,
        }
    finally:
        os.close(lock_fd)


def _template_document(args: argparse.Namespace) -> dict[str, Any]:
    action_id = getattr(args, "action_id", ACTION_ID)
    action_pins = _action_identity_pins(action_id)
    namespace = Path(args.namespace)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": args.checkout,
            "commit_sha": args.commit_sha,
            "isaac_python": args.isaac_python,
        },
        "action_id": action_id,
        "motion": action_pins["motion"],
        "manifest": action_pins["manifest"],
        "expected_effective_reward_recipe_sha256": (
            EXPECTED_REWARD_RECIPE_SHA256
        ),
        "policy_contract_sha256": (
            None if args.stage == "recipe" else args.policy_contract_sha256
        ),
        "seed": SEED,
        "stage": args.stage,
        "num_envs": 1,
        "max_iterations": 1 if args.stage == "recipe" else 2,
        "save_interval": 1,
        "gpu": {
            "index": args.gpu_index,
            "uuid": args.gpu_uuid,
            "owner": args.owner,
            "lock_path": f"/tmp/hope_lean_queue_gpu{args.gpu_index}.lock",
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser(
        "template", help="print one canonical concrete Pod spec"
    )
    template.add_argument("--stage", choices=("recipe", "smoke"), required=True)
    template.add_argument(
        "--action-id",
        choices=tuple(sorted(_R.ALLOWED_ACTION_IDS)),
        default=ACTION_ID,
    )
    template.add_argument("--checkout", required=True)
    template.add_argument("--commit-sha", required=True)
    template.add_argument("--isaac-python", required=True)
    template.add_argument("--policy-contract-sha256")
    template.add_argument("--gpu-index", required=True, type=int)
    template.add_argument("--gpu-uuid", required=True)
    template.add_argument("--owner", required=True)
    template.add_argument("--namespace", required=True)
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
            if args.stage == "recipe" and args.policy_contract_sha256 is not None:
                raise LaunchRefused(
                    "recipe template does not accept --policy-contract-sha256"
                )
            if args.stage == "smoke" and args.policy_contract_sha256 is None:
                raise LaunchRefused(
                    "smoke template requires --policy-contract-sha256"
                )
            print(
                json.dumps(
                    _template_document(args),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "_exec":
            return _internal_exec(
                Path(args.claim), args.claim_sha256, args.gpu_lock_fd
            )
        plan = build_plan(Path(args.spec))
        if args.command == "plan":
            print(
                json.dumps(
                    plan,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            return 0
        result = launch(plan, confirm_claim=args.confirm_claim)
        print(
            json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        return 0
    except LaunchRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
