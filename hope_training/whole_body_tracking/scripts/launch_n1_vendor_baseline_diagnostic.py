#!/usr/bin/env python3
"""Launch one fail-closed, single-GPU A3-vendor ActionBall diagnostic.

This is a deliberately thin policy adapter over
``launch_n1_reward_screen_diagnostic.py``.  It reuses that launcher's
canonical-spec, clean-commit, tracked-bundle, fresh-namespace, empty-GPU and
lifetime-lock implementation, while narrowing the scientific recipe to the
immutable ``HOPEPingPongActionBallA3VendorV1`` task profile.

Four exact stages are schema-known: ``smoke`` (1 env x 2 updates), ``probe``
(4096 envs x 5 updates), ``push_evidence`` (4096 envs x 32 updates), and the
finite ``long`` (4096 envs x 20001 updates).  Long fails closed unless its
canonical spec pins one tracked ``vendor_probe_gate_receipt`` whose probe and
push evidence are exact PASS and whose artifact-descendant diff remains in the
receipt's narrow allowlist.  Seeds are restricted to 0, 1, or 2, and the only
action is ``bh_loop_c``.  There is no arbitrary Hydra override input.
The result remains diagnostic-only: it cannot mint formal evaluator,
promotion, resume, export, or judge authority.

Unlike the historical reward-screen launcher, this adapter forces the
already-adopted ``stable_ready_plant=true`` safety baseline.  The tracked
vendor task remains the authoritative full-DR target identity, but this
diagnostic stage intentionally disables CoM, mass, and PD-gain randomization;
those axes return only through a later restore gate.  Reward weights, push,
and control-step action delay remain owned by the exact vendor task profile.
The stable-ready override is part of the immutable claim and conflicting
inherited values fail closed.  The spec must also pin
``vendor_runtime_training_contract_sha256``; the tracked dynamic-ready
artifact must carry the identical SHA under
``sources.runtime_training_contract.sha256``.  Both values must equal the SHA
authorized by the launcher's fixed, tracked vendor identity manifest; the spec
is never its own authority.  Legacy ready bundles therefore fail closed before
launch state is claimed.  Until that manifest records a genuinely materialized
runtime contract, every vendor launch also fails closed.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


_THIS_FILE = Path(__file__).resolve()
_BASE_FILE = _THIS_FILE.with_name("launch_n1_reward_screen_diagnostic.py")
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_hope_n1_reward_screen_diagnostic_base", _BASE_FILE
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load diagnostic base launcher: {_BASE_FILE}")
_B = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_B)


SCHEMA_VERSION = 1
SPEC_KIND = "n1_vendor_baseline_diagnostic_spec_v1"
CLAIM_KIND = "n1_vendor_baseline_diagnostic_claim_v1"
EXPERIMENT_NAME = "agibot_a3_hope_action_ball_vendor_v1_diagnostic"
TASK_PROFILE_ID = "HOPEPingPongActionBallA3VendorV1"
TASK_PROFILE_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV1.yaml"
)
ROBOT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/agibot_a3.py"
)
VENDOR_IDENTITY_MANIFEST_SOURCE = (
    "configs/a3_vendor_runtime_contract_20260731/required_identity.v1.json"
)
VENDOR_IDENTITY_MANIFEST_SHA256 = (
    "1147cbce8277c95cebf0e5657293c39a0e95ee17319b38f5312d935bcc8bd865"
)
VENDOR_AUTHORITY_MODULE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_a3_vendor_runtime_authority.py"
)
# Two-commit lifecycle: the bootstrap commit can emit the runtime contract,
# The authority receipt was produced from the first clean live-runtime commit;
# this later artifact commit tracks it and fixes the digest in code.  ``None``
# remains the fail-closed pre-materialization state, never operator input.
VENDOR_AUTHORITY_RECEIPT_SHA256: str | None = (
    "891676149cd4f2d6c2246d2f95bc957903e1fced66a4f7dbf1bcdacd113d4a11"
)
LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_vendor_baseline_diagnostic.py"
)
BASE_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_reward_screen_diagnostic.py"
)
REWARD_PROFILE = "vendor_task_defaults"
VENDOR_CONTRACT_FIELD = "vendor_runtime_training_contract_sha256"
STABLE_READY_PLANT_OVERRIDE = "+task.domain_rand.stable_ready_plant=true"
PUSH_EVIDENCE_STAGE = "push_evidence"
PUSH_EVIDENCE_ARGV_MARKER = "+n1_vendor_diagnostic_stage=push_evidence"
VENDOR_DIAGNOSTIC_STAGE_ARG_PREFIX = "+n1_vendor_diagnostic_stage="
VENDOR_CONTRACT_ARG_PREFIX = "+vendor_runtime_training_contract_sha256="
PUSH_EVIDENCE_CLAIM_FIELD = "push_evidence_runtime_sources"
VENDOR_PROBE_GATE_FIELD = "vendor_probe_gate_receipt"
VENDOR_PROBE_GATE_KIND = "n1_vendor_probe_gate_receipt_v1"
VENDOR_PROBE_GATE_PRODUCER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_n1_vendor_probe_gate_receipt.py"
)
PUSH_EVIDENCE_RUNTIME_SOURCE_PINS = {
    "IsaacLab interval event manager": {
        "path": (
            "/workspace/IsaacLab/source/isaaclab/isaaclab/managers/"
            "event_manager.py"
        ),
        "sha256": (
            "89c037c9f051605d400ebe723b4505e5a20093c8f3339dac77973a87ef8c35da"
        ),
    },
    "IsaacLab push-by-velocity event": {
        "path": (
            "/workspace/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py"
        ),
        "sha256": (
            "387abba19606f8c93a7cfc0d60fbd55d0b877342d347d64146553cfbd97e87d4"
        ),
    },
}
ALLOWED_SEEDS = frozenset((0, 1, 2))
ALLOWED_STAGES = frozenset(("smoke", "probe", PUSH_EVIDENCE_STAGE, "long"))


LaunchRefused = _B.LaunchRefused
canonical_sha256 = _B.canonical_sha256

_base_validate_spec_document = _B._validate_spec_document
_base_validate_budget = _B._validate_budget
_base_build_training_argv = _B._build_training_argv
_base_launch = _B.launch


def _configure_base() -> None:
    """Install the narrow vendor policy into the shared safety mechanism."""

    _B.SCHEMA_VERSION = SCHEMA_VERSION
    _B.SPEC_KIND = SPEC_KIND
    _B.CLAIM_KIND = CLAIM_KIND
    _B.EXPERIMENT_NAME = EXPERIMENT_NAME
    _B.LAUNCHER_SOURCE = LAUNCHER_SOURCE
    _B.TASK_SOURCE = TASK_PROFILE_SOURCE
    _B.ALLOWED_ACTIONS = frozenset(("bh_loop_c",))
    _B.ALLOWED_STAGES = ALLOWED_STAGES
    # Keep the inherited values visible in the immutable claim, but never
    # transmit them as Hydra overrides.  The tracked task profile is the only
    # owner of these defaults.
    _B.REWARD_PROFILES = {
        REWARD_PROFILE: {
            "racket_position_weight": 4.0,
            "racket_velocity_weight": 0.5,
            "racket_normal_weight": 0.5,
            "motion_scale": 1.0,
        }
    }
    _B._validate_budget = _validate_budget
    _B._validate_spec_document = _validate_spec_document
    _B._build_training_argv = _build_training_argv
    _B._validate_dynamic_ready = _validate_vendor_dynamic_ready
    _B._validate_runtime_sources = _validate_runtime_sources
    _B.launch = launch


def _validate_budget(
    stage: Any, num_envs: Any, max_iterations: Any, save_interval: Any
) -> dict[str, Any]:
    """Extend the shared exact budgets with one vendor push-evidence stage."""

    if stage != PUSH_EVIDENCE_STAGE:
        return _base_validate_budget(
            stage, num_envs, max_iterations, save_interval
        )
    envs = _B._plain_int(num_envs, name="num_envs", minimum=1)
    iterations = _B._plain_int(
        max_iterations, name="max_iterations", minimum=1
    )
    save = _B._plain_int(
        save_interval, name="save_interval", minimum=1
    )
    if (envs, iterations, save) != (4096, 32, 8):
        raise LaunchRefused(
            "push_evidence is exactly 4096 envs / 32 updates / "
            "save interval 8"
        )
    return {
        "stage": PUSH_EVIDENCE_STAGE,
        "num_envs": envs,
        "max_iterations": iterations,
        "save_interval": save,
    }


def _validate_spec_document(
    document: dict[str, Any], *, namespace_claimed: bool = False
) -> dict[str, Any]:
    if type(document) is not dict or VENDOR_CONTRACT_FIELD not in document:
        raise LaunchRefused(
            f"launch spec requires {VENDOR_CONTRACT_FIELD!r}"
        )
    contract_sha = _B._sha256(
        document[VENDOR_CONTRACT_FIELD], name=VENDOR_CONTRACT_FIELD
    )
    if document.get("action_id") != "bh_loop_c":
        raise LaunchRefused(
            "vendor diagnostic action_id must be exactly bh_loop_c"
        )
    stage = document.get("stage")
    has_gate = VENDOR_PROBE_GATE_FIELD in document
    if stage == "long" and not has_gate:
        raise LaunchRefused(
            "vendor long requires one exact vendor_probe_gate_receipt pin"
        )
    if stage != "long" and has_gate:
        raise LaunchRefused(
            "vendor_probe_gate_receipt is permitted only for exact long"
        )
    gate_pin = None
    if has_gate:
        gate_pin = dict(
            _B._exact_dict(
                document[VENDOR_PROBE_GATE_FIELD],
                _B._PIN_KEYS,
                name="spec.vendor_probe_gate_receipt",
            )
        )
    base_document = dict(document)
    del base_document[VENDOR_CONTRACT_FIELD]
    base_document.pop(VENDOR_PROBE_GATE_FIELD, None)
    spec = _base_validate_spec_document(
        base_document, namespace_claimed=namespace_claimed
    )
    if spec["reward_profile"] != REWARD_PROFILE:
        raise LaunchRefused(
            f"reward_profile must be exactly {REWARD_PROFILE!r}"
        )
    if spec["seed"] not in ALLOWED_SEEDS:
        raise LaunchRefused("vendor diagnostic seed must be exactly 0, 1, or 2")
    if spec["stage"] not in ALLOWED_STAGES:
        raise LaunchRefused(
            "vendor diagnostic stage must be smoke, probe, push_evidence, "
            "or long"
        )
    spec[VENDOR_CONTRACT_FIELD] = contract_sha
    if gate_pin is not None:
        spec[VENDOR_PROBE_GATE_FIELD] = gate_pin
    return spec


def _validate_vendor_dynamic_ready(
    checkout: Path,
    commit_sha: str,
    value: Any,
    *,
    action_id: str,
    motion_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Vendor-only counterpart of the base v1 dynamic-ready validator."""

    row = _B._exact_dict(
        value, _B._DYNAMIC_READY_KEYS, name="N1 bundle.dynamic_ready"
    )
    artifact_pin, candidate = _B._load_tracked_json(
        checkout,
        commit_sha,
        row["artifact"],
        name="N1 vendor dynamic-ready artifact",
    )
    candidate_content_sha = _B._verify_content_seal(
        candidate,
        name="N1 vendor dynamic-ready artifact",
        ensure_ascii=True,
    )
    robot = candidate.get("robot")
    sources = candidate.get("sources")
    stable_motion = (
        sources.get("stable_motion") if type(sources) is dict else None
    )
    if (
        candidate.get("schema_version") != 2
        or candidate.get("kind")
        != "agibot_a3_action_dynamic_ready_candidate_v2"
        or candidate.get("action_id") != action_id
        or type(robot) is not dict
        or robot.get("family") != "AgiBot A3"
        or type(stable_motion) is not dict
        or stable_motion.get("frame_index") != 0
        or stable_motion.get("sha256") != motion_sha256
        or type(candidate.get("runtime_plant")) is not dict
    ):
        raise LaunchRefused(
            "vendor launch requires the exact schema-v2 A3 action/motion plant"
        )

    receipt_pin, receipt = _B._load_tracked_json(
        checkout,
        commit_sha,
        row["nominal_hold_receipt"],
        name="N1 vendor nominal-hold receipt",
    )
    _B._verify_content_seal(
        receipt,
        name="N1 vendor nominal-hold receipt",
        ensure_ascii=False,
    )
    receipt_artifact = receipt.get("artifact")
    required_gate = candidate.get("required_next_gate")
    required_terminations = (
        required_gate.get("zero_terminal_required")
        if type(required_gate) is dict
        else None
    )
    active_terminations = receipt.get("active_terminations")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != _B.NOMINAL_HOLD_RECEIPT_KIND
        or receipt.get("verdict") != "PASS"
        or receipt.get("action_id") != action_id
        or receipt.get("motion_sha256") != motion_sha256
        or receipt.get("plant_contract_match") is not True
        or receipt.get("terminal_reasons") != []
        or receipt.get("generic_terminated") is not False
        or receipt.get("generic_truncated") is not False
        or type(receipt_artifact) is not dict
        or receipt_artifact.get("sha256") != artifact_pin["sha256"]
        or receipt_artifact.get("content_sha256") != candidate_content_sha
        or type(required_gate) is not dict
        or required_gate.get("kind") != _B.NOMINAL_HOLD_RECEIPT_KIND
        or type(required_terminations) is not list
        or type(active_terminations) is not list
        or not all(
            type(reason) is str and reason in active_terminations
            for reason in required_terminations
        )
    ):
        raise LaunchRefused(
            "vendor nominal-hold receipt does not prove the exact schema-v2 "
            "action/motion plant with zero terminal"
        )
    return {
        "artifact": artifact_pin,
        "nominal_hold_receipt": receipt_pin,
    }


def _validate_vendor_runtime_binding(
    checkout: Path,
    commit_sha: str,
    validated_bundle: Mapping[str, Any],
    expected_contract_sha256: str,
) -> dict[str, str]:
    """Bind dynamic-ready to the exact vendor runtime training contract."""

    dynamic_ready = validated_bundle.get("dynamic_ready")
    artifact_pin = (
        dynamic_ready.get("artifact")
        if type(dynamic_ready) is dict
        else None
    )
    if type(artifact_pin) is not dict:
        raise LaunchRefused(
            "vendor diagnostic requires a validated dynamic-ready artifact"
        )
    normalized_pin, artifact = _B._load_tracked_json(
        checkout,
        commit_sha,
        artifact_pin,
        name="vendor-bound N1 dynamic-ready artifact",
    )
    sources = artifact.get("sources") if type(artifact) is dict else None
    runtime_contract = (
        sources.get("runtime_training_contract")
        if type(sources) is dict
        else None
    )
    actual_sha = (
        runtime_contract.get("sha256")
        if type(runtime_contract) is dict
        else None
    )
    if type(actual_sha) is not str:
        raise LaunchRefused(
            "dynamic-ready artifact lacks "
            "sources.runtime_training_contract.sha256; legacy bundle refused"
        )
    actual_sha = _B._sha256(
        actual_sha,
        name="dynamic-ready sources.runtime_training_contract.sha256",
    )
    if actual_sha != expected_contract_sha256:
        raise LaunchRefused(
            "dynamic-ready runtime training contract SHA differs from spec"
        )
    return {
        "artifact_path": normalized_pin["path"],
        "artifact_sha256": normalized_pin["sha256"],
        "runtime_training_contract_sha256": actual_sha,
    }


def _validate_vendor_identity_manifest(
    checkout: Path, commit_sha: str
) -> dict[str, Any]:
    """Resolve the trusted contract SHA from one fixed tracked authority."""

    pin = {
        "path": VENDOR_IDENTITY_MANIFEST_SOURCE,
        "sha256": VENDOR_IDENTITY_MANIFEST_SHA256,
    }
    normalized_pin, manifest = _B._load_tracked_json(
        checkout,
        commit_sha,
        pin,
        name="vendor runtime training-contract identity manifest",
    )
    row = _B._exact_dict(
        manifest,
        (
            "schema_version",
            "kind",
            "status",
            "authority",
            "source_commit_binding",
            "sources",
            "robot_action_contract",
            "runtime_materialization",
        ),
        name="vendor runtime identity manifest",
    )
    if (
        row["schema_version"] != 1
        or row["kind"]
        != "a3_vendor_runtime_training_contract_required_identity_v1"
        or row["source_commit_binding"] != "launcher_selected_clean_commit"
    ):
        raise LaunchRefused("vendor runtime identity manifest kind differs")
    sources = _B._exact_dict(
        row["sources"],
        (
            "robot_config",
            "task_profile",
            "training_contract_builder",
            "training_entrypoint",
        ),
        name="vendor runtime identity manifest.sources",
    )
    for name, source_pin in sources.items():
        _B._verify_tracked_file(
            checkout,
            commit_sha,
            source_pin,
            name=f"vendor identity source {name}",
        )
    robot_action = row["robot_action_contract"]
    if (
        type(robot_action) is not dict
        or robot_action.get("runtime_dof_count") != 31
        or robot_action.get("vendor_body_dof_count") != 29
        or robot_action.get("legacy_head_dof_count") != 2
        or robot_action.get("action_scale_rule")
        != "0.25 * base_effort_limit / base_stiffness"
        or not isinstance(robot_action.get("groups"), list)
        or len(robot_action["groups"]) != 11
    ):
        raise LaunchRefused("vendor robot/action projection differs")
    runtime = _B._exact_dict(
        row["runtime_materialization"],
        (
            "required_training_contract_schema_version",
            "training_contract_sha256",
            "required_dynamic_ready_actions",
            "required_nominal_hold_verdict",
            "note",
        ),
        name="vendor runtime identity manifest.runtime_materialization",
    )
    if (
        row["status"] != "materialized"
        or runtime["required_training_contract_schema_version"] != 3
        or runtime["required_dynamic_ready_actions"]
        != ["bh_loop_c"]
        or runtime["required_nominal_hold_verdict"] != "PASS"
        or type(runtime["training_contract_sha256"]) is not str
    ):
        raise LaunchRefused(
            "vendor runtime identity is awaiting exact runtime materialization"
        )
    contract_sha = _B._sha256(
        runtime["training_contract_sha256"],
        name="vendor identity runtime training contract SHA",
    )
    return {
        "manifest": normalized_pin,
        "runtime_training_contract_sha256": contract_sha,
    }


def _load_vendor_authority_module(checkout: Path):
    module_path = checkout / VENDOR_AUTHORITY_MODULE_SOURCE
    spec = importlib.util.spec_from_file_location(
        "_hope_a3_vendor_runtime_authority", module_path
    )
    if spec is None or spec.loader is None:
        raise LaunchRefused("cannot load vendor runtime authority validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise LaunchRefused(
            f"cannot import vendor runtime authority validator: {exc}"
        ) from exc
    return module


def _validate_actual_vendor_authority(
    checkout: Path,
    commit_sha: str,
    validated_bundle: Mapping[str, Any],
    authoritative_contract_sha256: str,
) -> dict[str, Any]:
    """Require the fixed actual receipt and compare the full candidate plant."""

    if VENDOR_AUTHORITY_RECEIPT_SHA256 is None:
        raise LaunchRefused(
            "vendor runtime authority receipt is awaiting the second tracked "
            "materialization commit"
        )
    authority_module = _load_vendor_authority_module(checkout)
    receipt_path = checkout / authority_module.RECEIPT_REPO_PATH
    try:
        authority = authority_module.load_and_validate_vendor_runtime_authority(
            receipt_path,
            repo_root=checkout,
            expected_receipt_sha256=VENDOR_AUTHORITY_RECEIPT_SHA256,
            expected_runtime_training_contract_sha256=(
                authoritative_contract_sha256
            ),
            launch_commit=commit_sha,
            require_fixed_path=True,
        )
        dynamic_ready = validated_bundle.get("dynamic_ready")
        artifact_pin = (
            dynamic_ready.get("artifact")
            if type(dynamic_ready) is dict
            else None
        )
        if type(artifact_pin) is not dict:
            raise LaunchRefused(
                "vendor authority requires one validated dynamic-ready artifact"
            )
        _normalized, candidate = _B._load_tracked_json(
            checkout,
            commit_sha,
            artifact_pin,
            name="vendor-authority dynamic-ready candidate",
        )
        verified_runtime = authority.get("verified_vendor_runtime")
        candidate_sources = (
            candidate.get("sources") if type(candidate) is dict else None
        )
        candidate_motion = (
            candidate_sources.get("stable_motion")
            if type(candidate_sources) is dict
            else None
        )
        if (
            type(verified_runtime) is not dict
            or verified_runtime.get("action_id") != "bh_loop_c"
            or candidate.get("action_id") != verified_runtime.get("action_id")
            or type(candidate_motion) is not dict
            or candidate_motion.get("sha256")
            != verified_runtime.get("motion_sha256")
        ):
            raise LaunchRefused(
                "dynamic-ready action/motion differs from bh_loop_c authority"
            )
        plant = (
            authority_module
            .validate_candidate_runtime_plant_against_vendor_authority(
                candidate, authority
            )
        )
    except LaunchRefused:
        raise
    except authority_module.VendorRuntimeAuthorityError as exc:
        raise LaunchRefused(f"vendor runtime authority refused: {exc}") from exc
    return {
        "receipt_path": authority["receipt_path"],
        "receipt_sha256": authority["receipt_sha256"],
        "runtime_training_contract": authority[
            "runtime_training_contract"
        ],
        "verified_vendor_runtime": authority["verified_vendor_runtime"],
        "runtime_plant_identity": plant,
    }


def _build_training_argv(
    spec: dict[str, Any], bundle: dict[str, Any]
) -> list[str]:
    """Build the inherited argv and seal the adopted stable-ready plant."""

    argv = _base_build_training_argv(spec, bundle)
    remove_prefixes = (
        "+task.rewards.motion_scale=",
        "task.rewards.racket_position_weight=",
        "task.rewards.racket_velocity_weight=",
        "task.rewards.racket_normal_weight=",
    )
    result: list[str] = []
    task_replaced = False
    for item in argv:
        if item.startswith(
            (VENDOR_DIAGNOSTIC_STAGE_ARG_PREFIX, VENDOR_CONTRACT_ARG_PREFIX)
        ):
            raise LaunchRefused(
                "diagnostic base argv conflicts with vendor completion identity"
            )
        if item == "task=HOPEPingPongActionBall":
            result.append(f"task={TASK_PROFILE_ID}")
            task_replaced = True
            continue
        if "task.domain_rand.stable_ready_plant" in item:
            if item != STABLE_READY_PLANT_OVERRIDE:
                raise LaunchRefused(
                    "diagnostic base argv conflicts with stable-ready plant"
                )
            continue
        if item.startswith(remove_prefixes):
            continue
        result.append(item)
    if not task_replaced:
        raise LaunchRefused(
            "diagnostic base argv contract changed; vendor adapter refuses drift"
        )
    # Append exactly one canonical override in the adapter itself.  This keeps
    # the safety setting mechanically present even if the shared base later
    # stops supplying it, while deduplicating the current inherited value.
    result.append(STABLE_READY_PLANT_OVERRIDE)
    # The zero-PPO dynamic-ready recipe reuses this builder with its narrower
    # internal spec, which intentionally has no diagnostic stage field.
    diagnostic_stage = spec.get("stage")
    if diagnostic_stage is not None:
        if diagnostic_stage not in ALLOWED_STAGES:
            raise LaunchRefused("vendor diagnostic argv stage differs")
        result.extend(
            (
                VENDOR_DIAGNOSTIC_STAGE_ARG_PREFIX + diagnostic_stage,
                VENDOR_CONTRACT_ARG_PREFIX + spec[VENDOR_CONTRACT_FIELD],
            )
        )
        if (
            sum(
                item.startswith(VENDOR_DIAGNOSTIC_STAGE_ARG_PREFIX)
                for item in result
            )
            != 1
            or sum(
                item.startswith(VENDOR_CONTRACT_ARG_PREFIX) for item in result
            )
            != 1
        ):
            raise LaunchRefused("vendor completion argv identity is not exact-once")
    forbidden_fragments = (
        "push.enable",
        "push_robot",
        "randomize_pd_gains",
        "kp_gain_range",
        "kd_gain_range",
        "control_step_action_delay_",
    )
    if any(
        fragment in item
        for item in result
        for fragment in forbidden_fragments
    ):
        raise LaunchRefused(
            "vendor diagnostic argv unexpectedly overrides task-owned DR"
        )
    return result


def _push_evidence_runtime_source_origins() -> Mapping[str, Path]:
    """Resolve the reviewed current-Pod IsaacLab source origins.

    Kept as a zero-argument loader so host tests can inject isolated origins
    without requiring a local IsaacLab installation.
    """

    return {
        label: Path(pin["path"])
        for label, pin in PUSH_EVIDENCE_RUNTIME_SOURCE_PINS.items()
    }


def _runtime_file_identity(info: Any) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _validate_push_evidence_runtime_sources() -> dict[str, dict[str, str]]:
    """Pin exact interval scheduling and velocity-push implementations."""

    origins = _push_evidence_runtime_source_origins()
    expected_labels = frozenset(PUSH_EVIDENCE_RUNTIME_SOURCE_PINS)
    if type(origins) is not dict or frozenset(origins) != expected_labels:
        raise LaunchRefused(
            "push-evidence runtime source loader returned unexpected labels"
        )
    result: dict[str, dict[str, str]] = {}
    for label, expected in PUSH_EVIDENCE_RUNTIME_SOURCE_PINS.items():
        path = origins[label]
        if not isinstance(path, Path) or not path.is_absolute():
            raise LaunchRefused(
                f"push-evidence runtime source {label!r} is not absolute"
            )
        before = _B._stable_regular_file(path, name=label)
        try:
            observed_sha = _B.sha256_file(path)
        except OSError as exc:
            raise LaunchRefused(
                f"push-evidence runtime source cannot be hashed: {label}: {exc}"
            ) from exc
        after = _B._stable_regular_file(path, name=label)
        if _runtime_file_identity(before) != _runtime_file_identity(after):
            raise LaunchRefused(
                f"push-evidence runtime source changed while hashing: {label}"
            )
        if observed_sha != expected["sha256"]:
            raise LaunchRefused(
                f"push-evidence runtime source SHA differs: {label}"
            )
        result[label] = {"path": str(path), "sha256": observed_sha}
    return result


def _revalidate_push_evidence_claim_sources(payload: Mapping[str, Any]) -> None:
    """Bind a push-evidence claim to the unchanged external runtime files."""

    spec = payload.get("spec")
    if type(spec) is not dict:
        raise LaunchRefused("vendor claim spec is missing")
    if spec.get("stage") != PUSH_EVIDENCE_STAGE:
        if PUSH_EVIDENCE_CLAIM_FIELD in payload:
            raise LaunchRefused(
                "non-push vendor claim carries push-evidence runtime sources"
            )
        return
    claimed = payload.get(PUSH_EVIDENCE_CLAIM_FIELD)
    observed = _validate_push_evidence_runtime_sources()
    if claimed != observed:
        raise LaunchRefused(
            "push-evidence runtime source identity drifted after plan"
        )


def _load_probe_gate_module(checkout: Path):
    path = checkout / VENDOR_PROBE_GATE_PRODUCER_SOURCE
    spec = importlib.util.spec_from_file_location(
        "_hope_vendor_probe_gate_receipt_validator", path
    )
    if spec is None or spec.loader is None:
        raise LaunchRefused("cannot load vendor probe-gate validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise LaunchRefused(
            f"cannot import vendor probe-gate validator: {exc}"
        ) from exc
    return module


def _git_is_ancestor(checkout: Path, ancestor: str, descendant: str) -> bool:
    result = _B._run_git(
        checkout,
        ("merge-base", "--is-ancestor", ancestor, descendant),
    )
    return result.returncode == 0


def _validate_probe_gate_descendant_policy(
    checkout: Path,
    commit_sha: str,
    *,
    receipt_pin: Mapping[str, str],
    spec: Mapping[str, Any],
    producer: Mapping[str, Any],
    successor_policy: Mapping[str, Any],
) -> None:
    gate_source = producer.get("gate_source_commit")
    receipt_path = receipt_pin["path"]
    if (
        type(gate_source) is not str
        or _B.COMMIT_RE.fullmatch(gate_source) is None
        or not _git_is_ancestor(checkout, gate_source, commit_sha)
    ):
        raise LaunchRefused(
            "probe-gate code commit is not an ancestor of launch commit"
        )
    if (
        successor_policy.get("required_gate_source_ancestor_commit")
        != gate_source
    ):
        raise LaunchRefused("probe-gate successor ancestor differs")
    allow = successor_policy.get("allowed_artifact_descendant_diff")
    if type(allow) is not dict or set(allow) != {"exact_paths", "prefixes"}:
        raise LaunchRefused("probe-gate successor diff policy is incomplete")
    exact = allow["exact_paths"]
    prefixes = allow["prefixes"]
    if (
        type(exact) is not list
        or len(exact) != 2
        or len(set(exact)) != 2
        or receipt_path not in exact
        or type(prefixes) is not list
        or prefixes != ["docs/"]
        or any(
            type(path) is not str
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in exact
        )
    ):
        raise LaunchRefused("probe-gate exact artifact allowlist differs")
    if (
        not receipt_path.startswith("configs/n1_vendor_probe_gate_20260731/")
        or not receipt_path.endswith(".json")
    ):
        raise LaunchRefused("probe-gate receipt path is not the fixed config class")
    long_spec_paths = [path for path in exact if path != receipt_path]
    long_spec_path = long_spec_paths[0] if len(long_spec_paths) == 1 else None
    if (
        type(long_spec_path) is not str
        or not long_spec_path.startswith("configs/n1_vendor_launch_20260731/")
        or ".long." not in Path(long_spec_path).name
        or not long_spec_path.endswith(".json")
    ):
        raise LaunchRefused(
            "probe-gate long-spec allowlist path is not the fixed config class"
        )
    diff = _B._run_git(
        checkout,
        (
            "diff",
            "--name-status",
            "--diff-filter=ACDMRTUXB",
            f"{gate_source}..{commit_sha}",
            "--",
        ),
    )
    if diff.returncode != 0:
        raise LaunchRefused(
            f"cannot inspect probe-gate artifact descendant: {diff.stderr.strip()}"
        )
    statuses = {}
    for line in diff.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise LaunchRefused("probe-gate artifact descendant diff is malformed")
        statuses[parts[1]] = parts[0]
    changed = list(statuses)
    forbidden = [
        path
        for path in changed
        if path not in exact and not any(path.startswith(prefix) for prefix in prefixes)
    ]
    if forbidden:
        raise LaunchRefused(
            "probe-gate artifact descendant changes non-allowlisted source: "
            + forbidden[0]
        )
    if statuses.get(receipt_path) != "A" or statuses.get(long_spec_path) != "A":
        raise LaunchRefused(
            "probe-gate receipt and long-spec template must both be added "
            "after the gate-code commit"
        )
    long_path = checkout / long_spec_path
    long_pin = {
        "path": long_spec_path,
        "sha256": _B.sha256_file(long_path),
    }
    _normalized, template = _B._load_tracked_json(
        checkout,
        commit_sha,
        long_pin,
        name="vendor long spec template",
    )
    ignored = {"source", "gpu", "namespace", "log_path"}
    template_projection = {
        key: value for key, value in template.items() if key not in ignored
    }
    spec_projection = {
        key: value for key, value in spec.items() if key not in ignored
    }
    if (
        template_projection != spec_projection
        or template.get(VENDOR_PROBE_GATE_FIELD) != dict(receipt_pin)
    ):
        raise LaunchRefused(
            "tracked long-spec template scientific fields differ from launch spec"
        )


def _validate_probe_gate_stage(
    stage: Any,
    *,
    expected_stage: str,
    evidence_source_commit: str,
    expected_contract_sha256: str,
    gate_module: Any,
) -> None:
    if type(stage) is not dict or stage.get("stage") != expected_stage:
        raise LaunchRefused(f"probe-gate {expected_stage} stage differs")
    if stage.get("source_commit") != evidence_source_commit:
        raise LaunchRefused(
            f"probe-gate {expected_stage} source commit differs"
        )
    expected_budget = gate_module.EXPECTED_STAGES[expected_stage]
    if stage.get("budget") != expected_budget:
        raise LaunchRefused(
            f"probe-gate {expected_stage} exact budget differs"
        )
    namespace = stage.get("namespace")
    run_dir = stage.get("run_directory")
    if (
        type(namespace) is not str
        or not Path(namespace).is_absolute()
        or type(run_dir) is not str
        or not Path(run_dir).is_absolute()
    ):
        raise LaunchRefused("probe-gate evidence paths must be absolute")
    claim = stage.get("launch_claim")
    log = stage.get("run_log")
    if (
        type(claim) is not dict
        or set(claim)
        != {"path", "file_sha256", "launch_claim_sha256"}
        or type(log) is not dict
        or set(log) != {"path", "sha256"}
        or any(
            _B.SHA256_RE.fullmatch(value or "") is None
            for value in (
                claim.get("file_sha256"),
                claim.get("launch_claim_sha256"),
                log.get("sha256"),
            )
        )
    ):
        raise LaunchRefused("probe-gate claim/log pins are incomplete")
    delay = stage.get("control_step_action_delay")
    hard_contract_sha256 = (
        delay.get("training_contract_sha256") if type(delay) is dict else None
    )
    # The hard task contract and vendor plant-authority contract are separate
    # layers and legitimately have different SHAs.  Bind each independently;
    # do not collapse this into a false equality check.
    if (
        type(hard_contract_sha256) is not str
        or _B.SHA256_RE.fullmatch(hard_contract_sha256) is None
    ):
        raise LaunchRefused("probe-gate hard training contract is incomplete")
    checkpoints = stage.get("checkpoints")
    expected_indices = gate_module.EXPECTED_CHECKPOINT_INDICES[expected_stage]
    if (
        type(checkpoints) is not list
        or tuple(item.get("index") for item in checkpoints) != expected_indices
    ):
        raise LaunchRefused("probe-gate checkpoint sequence differs")
    previous_normalizer_counts: dict[str, float] | None = None
    for item in checkpoints:
        if (
            type(item) is not dict
            or set(item)
            != {
                "index",
                "path",
                "sha256",
                "embedded_iteration",
                "training_launch_claim_sha256",
                "training_contract_sha256",
                "tensor_count",
                "element_count",
                "all_finite",
                "actor_normalizer",
                "critic_normalizer",
            }
            or type(item["path"]) is not str
            or not Path(item["path"]).is_absolute()
            or _B.SHA256_RE.fullmatch(item["sha256"] or "") is None
            or item["embedded_iteration"] != item["index"]
            or item["training_launch_claim_sha256"]
            != claim["launch_claim_sha256"]
            or item["training_contract_sha256"]
            != hard_contract_sha256
            or type(item["tensor_count"]) is not int
            or item["tensor_count"] <= 0
            or type(item["element_count"]) is not int
            or item["element_count"] <= 0
            or item["all_finite"] is not True
        ):
            raise LaunchRefused("probe-gate checkpoint finite summary differs")
        normalizer_counts = {}
        for role, expected_features in (("actor", 194), ("critic", 318)):
            summary = item[f"{role}_normalizer"]
            if (
                type(summary) is not dict
                or set(summary)
                != {
                    "state_keys",
                    "mean_key",
                    "scale_key",
                    "count_key",
                    "feature_count",
                    "count",
                    "tensor_count",
                    "element_count",
                    "all_finite",
                }
                or type(summary["state_keys"]) is not list
                or summary["state_keys"] != sorted(summary["state_keys"])
                or any(
                    type(key) is not str or key not in summary["state_keys"]
                    for key in (
                        summary["mean_key"],
                        summary["scale_key"],
                        summary["count_key"],
                    )
                )
                or summary["feature_count"] != expected_features
                or type(summary["count"]) not in (int, float)
                or not _B.math.isfinite(float(summary["count"]))
                or float(summary["count"]) <= 0.0
                or type(summary["tensor_count"]) is not int
                or summary["tensor_count"] < 3
                or type(summary["element_count"]) is not int
                or summary["element_count"]
                < expected_features * 2 + 1
                or summary["all_finite"] is not True
            ):
                raise LaunchRefused(
                    f"probe-gate {role} normalizer checkpoint summary differs"
                )
            normalizer_counts[role] = float(summary["count"])
        if previous_normalizer_counts is not None and any(
            normalizer_counts[role] < previous_normalizer_counts[role]
            for role in normalizer_counts
        ):
            raise LaunchRefused(
                "probe-gate checkpoint normalizer count regressed"
            )
        previous_normalizer_counts = normalizer_counts
    abi = stage.get("runtime_abi")
    if (
        type(abi) is not dict
        or abi.get("event") != "hope_rsl_rl_runtime_abi"
        or abi.get("schema_version") != 1
        or abi.get("capabilities", {}).get(
            "empirical_normalization_preflight"
        )
        is not True
        or abi.get("capabilities", {}).get(
            "positive_realized_policy_std_guard"
        )
        is not True
    ):
        raise LaunchRefused("probe-gate runtime ABI evidence differs")
    delay_terms = delay.get("delay_terms") if type(delay) is dict else None
    if (
        type(delay_terms) is not list
        or len(delay_terms) != 1
    ):
        raise LaunchRefused("probe-gate delay evidence is incomplete")
    delay_term = delay_terms[0]
    histogram = delay_term.get("lag_histogram")
    if (
        delay_term.get("num_envs") != expected_budget["num_envs"]
        or delay_term.get("initialized_env_count") != expected_budget["num_envs"]
        or type(histogram) is not dict
        or set(histogram) != {"0", "1", "2"}
        or any(type(value) is not int or value <= 0 for value in histogram.values())
        or sum(histogram.values()) != expected_budget["num_envs"]
    ):
        raise LaunchRefused("probe-gate delay histogram differs")
    std = stage.get("policy_std_lr_updates")
    if type(std) is not list or len(std) != expected_budget["max_iterations"]:
        raise LaunchRefused("probe-gate std/LR update count differs")
    for update, item in enumerate(std):
        values = (
            item.get("policy_std_min"),
            item.get("policy_std_mean"),
            item.get("policy_std_max"),
            item.get("learning_rate"),
        )
        if (
            item.get("ppo_update") != update
            or any(
                type(value) not in (int, float)
                or not _B.math.isfinite(float(value))
                or float(value) <= 0
                for value in values
            )
            or not values[0] <= values[1] <= values[2]
        ):
            raise LaunchRefused("probe-gate std/LR values differ")
    completion = stage.get("training_completion")
    if completion != {
        "cleanup_complete": True,
        "completed_ppo_updates": expected_budget["max_iterations"],
        "event": "hope_training_complete",
        "num_envs": expected_budget["num_envs"],
        "schema_version": 1,
        "stage": expected_stage,
        "training_contract_sha256": hard_contract_sha256,
        "training_launch_claim_sha256": claim["launch_claim_sha256"],
        "vendor_runtime_training_contract_sha256": expected_contract_sha256,
    }:
        raise LaunchRefused("probe-gate natural-completion marker differs")
    joint = stage.get("joint_safety")
    totals = joint.get("aggregate_counter_totals") if type(joint) is dict else None
    joint_updates = joint.get("updates") if type(joint) is dict else None
    recomputed_joint_totals: dict[str, int] = {}
    recomputed_minimum_gap: float | None = None
    if type(joint_updates) is list:
        for update, joint_row in enumerate(joint_updates):
            counters = (
                joint_row.get("counter_totals")
                if type(joint_row) is dict
                else None
            )
            gap = (
                joint_row.get("minimum_hard_gap_rad")
                if type(joint_row) is dict
                else None
            )
            if (
                type(joint_row) is not dict
                or joint_row.get("event")
                != "hope_joint_safety_diagnostic_compact_update"
                or joint_row.get("schema_version") != 1
                or joint_row.get("status")
                != "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
                or joint_row.get("ppo_update") != update
                or joint_row.get("num_envs") != expected_budget["num_envs"]
                or joint_row.get("policy_step_count")
                != gate_module.ROLLOUT_STEPS_PER_UPDATE
                or type(counters) is not dict
                or counters.get("policy_steps")
                != expected_budget["num_envs"]
                * gate_module.ROLLOUT_STEPS_PER_UPDATE
                or counters.get("complete_policy_steps")
                != expected_budget["num_envs"]
                * gate_module.ROLLOUT_STEPS_PER_UPDATE
                or type(gap) not in (int, float)
                or not _B.math.isfinite(float(gap))
                or float(gap) <= 0
            ):
                raise LaunchRefused("probe-gate joint-safety raw update differs")
            recomputed_minimum_gap = (
                float(gap)
                if recomputed_minimum_gap is None
                else min(recomputed_minimum_gap, float(gap))
            )
            for key, value in counters.items():
                if type(key) is not str or type(value) is not int or value < 0:
                    raise LaunchRefused(
                        "probe-gate joint-safety raw counter differs"
                    )
                recomputed_joint_totals[key] = (
                    recomputed_joint_totals.get(key, 0) + value
                )
    if (
        type(totals) is not dict
        or any(
            totals.get(key, 0) != 0
            for key in (
                "actual_hard_edge_events",
                "qdes_events",
            )
        )
        or joint.get("fatal_marker_count") != 0
        or type(joint_updates) is not list
        or len(joint_updates) != expected_budget["max_iterations"]
        or totals != dict(sorted(recomputed_joint_totals.items()))
        or type(joint.get("minimum_hard_gap_rad")) not in (int, float)
        or joint["minimum_hard_gap_rad"] <= 0
        or joint["minimum_hard_gap_rad"] != recomputed_minimum_gap
    ):
        raise LaunchRefused("probe-gate joint-hard/qdes zero gate differs")
    behavior = stage.get("behavior")
    reach = (
        behavior.get("reachability_and_failure_rates")
        if type(behavior) is dict
        else None
    )
    entry = (
        behavior.get("strike_window_entry_conservation")
        if type(behavior) is dict
        else None
    )
    terminal = behavior.get("terminal_conservation") if type(behavior) is dict else None
    reference = (
        behavior.get("reference_guard_conservation")
        if type(behavior) is dict
        else None
    )
    if (
        type(reach) is not dict
        or reach.get("pass") is not True
        or reach.get("environment_policy_step_denominator")
        != expected_budget["num_envs"]
        * gate_module.ROLLOUT_STEPS_PER_UPDATE
        * expected_budget["max_iterations"]
        or reach.get("table_contact_per_env_step_limit")
        != gate_module.BEHAVIOR_RATE_LIMITS[expected_stage][
            "table_contact_per_env_step"
        ]
        or reach.get("physical_fall_per_env_step_limit")
        != gate_module.BEHAVIOR_RATE_LIMITS[expected_stage]["fall_per_env_step"]
        or reach.get("table_contact_per_env_step")
        > reach.get("table_contact_per_env_step_limit")
        or reach.get("physical_fall_per_env_step")
        > reach.get("physical_fall_per_env_step_limit")
        or reach.get("conservative_mean_episode_age_steps")
        < gate_module.MIN_CONSERVATIVE_EPISODE_AGE_STEPS
        or reach.get("strike_opportunity_count", 0) <= 0
        or reach.get("swing_start_count", 0) <= 0
        or reach.get("swing_outcome_count", 0) <= 0
        or type(entry) is not dict
        or entry.get("matches") is not True
        or entry.get("nonfinite_count") != 0
        or type(terminal) is not dict
        or terminal.get("physical_partition_matches") is not True
        or terminal.get("terminal_partition_matches") is not True
        or type(reference) is not dict
        or reference.get("matches") is not True
        or reference.get("hard_without_snapshot_count") != 0
    ):
        raise LaunchRefused(
            "probe-gate behavior reachability/rate/conservation differs"
        )
    aggregate = behavior.get("aggregate_counters")
    behavior_updates = behavior.get("updates") if type(behavior) is dict else None
    recomputed_behavior: dict[str, int | float] = {}
    if type(behavior_updates) is list:
        for update, behavior_row in enumerate(behavior_updates):
            counters = (
                behavior_row.get("counters")
                if type(behavior_row) is dict
                else None
            )
            if (
                type(behavior_row) is not dict
                or behavior_row.get("event") != "hope_exact_behavior_update"
                or behavior_row.get("schema_version") != 1
                or behavior_row.get("ppo_update") != update
                or type(counters) is not dict
            ):
                raise LaunchRefused("probe-gate behavior raw update differs")
            for key, value in counters.items():
                if (
                    type(key) is not str
                    or type(value) not in (int, float)
                    or not _B.math.isfinite(float(value))
                    or value < 0
                ):
                    raise LaunchRefused("probe-gate behavior raw counter differs")
                recomputed_behavior[key] = recomputed_behavior.get(key, 0) + value
    if (
        type(aggregate) is not dict
        or type(behavior_updates) is not list
        or len(behavior_updates) != expected_budget["max_iterations"]
        or aggregate != dict(sorted(recomputed_behavior.items()))
        or aggregate.get("ready_nonfinite_value_count") != 0
        or any(
            value != 0
            for key, value in aggregate.items()
            if key.startswith("termination_reason_")
            and (
                "joint_actual_forbidden" in key
                or "joint_qdes_forbidden" in key
            )
        )
    ):
        raise LaunchRefused("probe-gate behavior hard/nonfinite counters differ")
    push_diagnostic = stage.get("push_velocity_diagnostic")
    push_updates = (
        push_diagnostic.get("updates")
        if type(push_diagnostic) is dict
        else None
    )
    push_aggregate = (
        push_diagnostic.get("aggregate")
        if type(push_diagnostic) is dict
        else None
    )
    if (
        type(push_updates) is not list
        or len(push_updates) != expected_budget["max_iterations"]
        or type(push_aggregate) is not dict
        or push_aggregate.get("delta_nonfinite_element_count") != 0
        or push_aggregate.get("below_range_count") != 0
        or push_aggregate.get("above_range_count") != 0
    ):
        raise LaunchRefused("probe-gate runtime velocity-push evidence differs")
    recomputed_push = {
        "event_call_count": 0,
        "env_application_count": 0,
        "delta_nonfinite_element_count": 0,
        "below_range_count": 0,
        "above_range_count": 0,
    }
    for update, push_row in enumerate(push_updates):
        counters = push_row.get("counters") if type(push_row) is dict else None
        if (
            type(push_row) is not dict
            or push_row.get("event")
            != "hope_push_velocity_diagnostic_update"
            or push_row.get("schema_version") != 1
            or push_row.get("ppo_update") != update
            or type(counters) is not dict
            or type(counters.get("event_call_count")) is not int
            or counters["event_call_count"] < 0
            or type(counters.get("env_application_count")) is not int
            or counters["env_application_count"] < 0
            or type(counters.get("axes")) is not dict
            or set(counters["axes"])
            != {"x", "y", "z", "roll", "pitch", "yaw"}
        ):
            raise LaunchRefused(
                "probe-gate velocity-push update sequence differs"
            )
        for key in (
            "event_call_count",
            "env_application_count",
            "delta_nonfinite_element_count",
        ):
            value = counters.get(key)
            if type(value) is not int or value < 0:
                raise LaunchRefused(
                    "probe-gate velocity-push raw counter differs"
                )
            recomputed_push[key] += value
        for values in counters["axes"].values():
            if type(values) is not dict:
                raise LaunchRefused(
                    "probe-gate velocity-push axis evidence differs"
                )
            for field, aggregate_field in (
                ("below_range_count", "below_range_count"),
                ("above_range_count", "above_range_count"),
            ):
                value = values.get(field)
                if type(value) is not int or value < 0:
                    raise LaunchRefused(
                        "probe-gate velocity-push range counter differs"
                    )
                recomputed_push[aggregate_field] += value
    if push_aggregate != recomputed_push:
        raise LaunchRefused(
            "probe-gate velocity-push aggregate differs from raw updates"
        )
    if expected_stage == "probe" and (
        push_aggregate.get("event_call_count") != 0
        or push_aggregate.get("env_application_count") != 0
    ):
        raise LaunchRefused("probe-gate short probe unexpectedly contains push")
    try:
        recomputed_abi = gate_module._validate_abi([abi])
        recomputed_delay = gate_module._validate_delay(
            [delay], num_envs=expected_budget["num_envs"]
        )
        recomputed_std = gate_module._validate_std_lr(
            std, updates=expected_budget["max_iterations"]
        )
        recomputed_joint = gate_module._validate_joint_safety(
            joint_updates,
            [],
            updates=expected_budget["max_iterations"],
            num_envs=expected_budget["num_envs"],
        )
        recomputed_behavior_summary = gate_module._validate_behavior(
            behavior_updates,
            stage=expected_stage,
            updates=expected_budget["max_iterations"],
            num_envs=expected_budget["num_envs"],
        )
        recomputed_push_summary = gate_module._validate_push_velocity(
            push_updates,
            stage=expected_stage,
            updates=expected_budget["max_iterations"],
            num_envs=expected_budget["num_envs"],
        )
    except gate_module.ReceiptRefused as exc:
        raise LaunchRefused(
            f"probe-gate raw stage evidence failed producer replay: {exc}"
        ) from exc
    if (
        recomputed_abi != abi
        or recomputed_delay != delay
        or recomputed_std != std
        or recomputed_joint != joint
        or recomputed_behavior_summary != behavior
        or recomputed_push_summary != push_diagnostic
    ):
        raise LaunchRefused(
            "probe-gate stored summaries differ from producer replay"
        )
    if expected_stage == PUSH_EVIDENCE_STAGE:
        timer = stage.get("push_timer_control_flow")
        expected_sources = {
            label: {"path": pin["path"], "sha256": pin["sha256"]}
            for label, pin in PUSH_EVIDENCE_RUNTIME_SOURCE_PINS.items()
        }
        counter = timer.get("push_counter") if type(timer) is dict else None
        duration = (
            expected_budget["max_iterations"]
            * gate_module.ROLLOUT_STEPS_PER_UPDATE
            * gate_module.POLICY_DT_S
        )
        if (
            type(timer) is not dict
            or timer.get("runtime_sources") != expected_sources
            or timer.get("interval_range_s")
            != list(gate_module.PUSH_INTERVAL_RANGE_S)
            or timer.get("duration_s") != duration
            or timer.get("strict_upper_bound_crossed") is not True
            or type(counter) is not dict
            or counter.get("kind")
            != "runtime_observed_population_equivalent_v1"
            or counter.get("event_call_count")
            != push_aggregate.get("event_call_count")
            or counter.get("environment_application_count")
            != push_aggregate.get("env_application_count")
            or counter.get("minimum_environment_application_count")
            != expected_budget["num_envs"]
            or counter.get("event_call_count", 0) <= 0
            or counter.get("environment_application_count", 0)
            < expected_budget["num_envs"]
        ):
            raise LaunchRefused("probe-gate push counter/timer/source proof differs")


def _validate_vendor_probe_gate_receipt(
    checkout: Path,
    commit_sha: str,
    value: Any,
    *,
    spec: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    pin, receipt = _B._load_tracked_json(
        checkout,
        commit_sha,
        value,
        name="vendor probe gate receipt",
    )
    _B._verify_content_seal(
        receipt, name="vendor probe gate receipt", ensure_ascii=False
    )
    row = _B._exact_dict(
        receipt,
        (
            "schema_version",
            "kind",
            "verdict",
            "producer",
            "evidence_source_commit",
            "scientific_identity",
            "stages",
            "acceptance",
            "successor_policy",
            "authorization",
            "content_sha256",
        ),
        name="vendor probe gate receipt",
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != VENDOR_PROBE_GATE_KIND
        or row["verdict"] != "PASS"
    ):
        raise LaunchRefused("vendor probe gate receipt is not schema-1 PASS")
    producer = _B._exact_dict(
        row["producer"],
        ("source", "gate_source_commit", "algorithm", "self_reference_free"),
        name="vendor probe gate producer",
    )
    producer_pin, _producer_path = _B._verify_tracked_file(
        checkout,
        commit_sha,
        producer["source"],
        name="vendor probe gate producer source",
    )
    if (
        producer_pin["path"] != VENDOR_PROBE_GATE_PRODUCER_SOURCE
        or producer["algorithm"] != "exact_probe_push_evidence_v1"
        or producer["self_reference_free"] is not True
    ):
        raise LaunchRefused("vendor probe gate producer identity differs")
    evidence_source = row["evidence_source_commit"]
    if (
        type(evidence_source) is not str
        or _B.COMMIT_RE.fullmatch(evidence_source) is None
        or evidence_source != producer["gate_source_commit"]
    ):
        raise LaunchRefused(
            "probe-gate evidence must use the exact gate-code source commit"
        )
    gate_module = _load_probe_gate_module(checkout)
    expected_identity = gate_module._scientific_identity(payload)
    if row["scientific_identity"] != expected_identity:
        raise LaunchRefused(
            "probe-gate scientific identity differs from long claim"
        )
    expected_contract_sha256 = row["scientific_identity"].get(
        VENDOR_CONTRACT_FIELD
    )
    if (
        type(expected_contract_sha256) is not str
        or _B.SHA256_RE.fullmatch(expected_contract_sha256) is None
    ):
        raise LaunchRefused(
            "probe-gate scientific identity omits the vendor training contract"
        )
    stages = _B._exact_dict(
        row["stages"],
        ("probe", "push_evidence"),
        name="vendor probe gate stages",
    )
    for stage_name in ("probe", "push_evidence"):
        _validate_probe_gate_stage(
            stages[stage_name],
            expected_stage=stage_name,
            evidence_source_commit=evidence_source,
            expected_contract_sha256=expected_contract_sha256,
            gate_module=gate_module,
        )
        try:
            observed_stage, observed_identity = gate_module._stage_evidence(
                Path(stages[stage_name]["namespace"]),
                Path(stages[stage_name]["run_directory"]),
                expected_stage=stage_name,
            )
        except gate_module.ReceiptRefused as exc:
            raise LaunchRefused(
                f"probe-gate {stage_name} raw evidence replay refused: {exc}"
            ) from exc
        if (
            observed_stage != stages[stage_name]
            or observed_identity != row["scientific_identity"]
        ):
            raise LaunchRefused(
                f"probe-gate {stage_name} raw replay differs from receipt"
            )
    if (
        stages["probe"]["control_step_action_delay"][
            "training_contract_sha256"
        ]
        != stages["push_evidence"]["control_step_action_delay"][
            "training_contract_sha256"
        ]
    ):
        raise LaunchRefused("probe/push hard training-contract SHA differs")
    if stages["probe"]["runtime_abi"] != stages["push_evidence"]["runtime_abi"]:
        raise LaunchRefused("probe/push runtime ABI markers differ")
    acceptance = row["acceptance"]
    expected_acceptance = {
        "probe_exact_pass": True,
        "push_evidence_exact_pass": True,
        "finite_checkpoints": True,
        "normalizer_checkpoint_persistence": True,
        "runtime_abi_exact": True,
        "control_step_delay_exact": True,
        "positive_policy_std_and_finite_lr": True,
        "zero_actual_hard_edge": True,
        "bounded_table_contact_rate": True,
        "bounded_physical_fall_rate": True,
        "minimum_episode_age_and_strike_swing_reachability": True,
        "zero_qdes_edge": True,
        "zero_nonfinite": True,
        "terminal_aggregation_conserved": True,
        "strike_entry_histogram_conserved": True,
        "push_timer_control_flow_proved": True,
        "natural_training_completion": True,
    }
    if acceptance != expected_acceptance:
        raise LaunchRefused("probe-gate acceptance vector differs")
    authorization = row["authorization"]
    expected_authorization = {
        "vendor_n1_long_launch": True,
        "formal_evidence": False,
        "curriculum_promotion": False,
        "resume": False,
        "export": False,
        "judge": False,
        "deployment": False,
        "hardware": False,
    }
    if authorization != expected_authorization:
        raise LaunchRefused("probe-gate authorization boundary differs")
    _validate_probe_gate_descendant_policy(
        checkout,
        commit_sha,
        receipt_pin=pin,
        spec=spec,
        producer=producer,
        successor_policy=row["successor_policy"],
    )
    return {
        "pin": pin,
        "content_sha256": row["content_sha256"],
        "evidence_source_commit": evidence_source,
        "gate_source_commit": producer["gate_source_commit"],
        "scientific_argv_canonical_sha256": row["scientific_identity"][
            "scientific_argv_canonical_sha256"
        ],
        "authorization": dict(authorization),
    }


def _validate_runtime_sources(
    checkout: Path, commit_sha: str
) -> dict[str, dict[str, Any]]:
    """Pin both the thin adapter and the safety implementation it imports."""

    result: dict[str, dict[str, Any]] = {}
    for relative, label in (
        (LAUNCHER_SOURCE, "N1 vendor diagnostic launcher"),
        (BASE_LAUNCHER_SOURCE, "N1 diagnostic safety base"),
        (_B.TRAIN_SOURCE, "training entrypoint"),
        (TASK_PROFILE_SOURCE, f"immutable task profile {TASK_PROFILE_ID}"),
        (ROBOT_SOURCE, "vendor A3 robot source"),
        (
            VENDOR_IDENTITY_MANIFEST_SOURCE,
            "vendor runtime training-contract identity manifest",
        ),
        (
            VENDOR_AUTHORITY_MODULE_SOURCE,
            "vendor runtime authority validator",
        ),
        (
            VENDOR_PROBE_GATE_PRODUCER_SOURCE,
            "vendor probe-gate receipt producer",
        ),
        (_B.KIT_LAUNCHER_SOURCE, "Kit locked launcher"),
    ):
        pin = {
            "path": relative,
            "sha256": _B.sha256_file(checkout / relative),
        }
        normalized, _path = _B._verify_tracked_file(
            checkout, commit_sha, pin, name=label
        )
        result[label] = normalized
    if _THIS_FILE != checkout / LAUNCHER_SOURCE:
        raise LaunchRefused(
            "running vendor launcher is not the exact selected checkout path"
        )
    return result


def launch(plan: dict[str, Any], *, confirm_claim: str) -> dict[str, Any]:
    payload = plan.get("canonical_payload")
    if type(payload) is not dict:
        raise LaunchRefused("vendor launch plan payload is missing")
    # Re-read the external IsaacLab implementations immediately before the
    # shared launcher claims mutable namespace/GPU state.
    _revalidate_push_evidence_claim_sources(payload)
    spec = payload.get("spec")
    if type(spec) is dict and spec.get("stage") == "long":
        observed = _validate_vendor_probe_gate_receipt(
            Path(spec["source"]["checkout"]),
            spec["source"]["commit_sha"],
            spec[VENDOR_PROBE_GATE_FIELD],
            spec=spec,
            payload=payload,
        )
        if payload.get(VENDOR_PROBE_GATE_FIELD) != observed:
            raise LaunchRefused(
                "vendor probe gate receipt differs immediately before claim"
            )
    result = _base_launch(plan, confirm_claim=confirm_claim)
    result["kind"] = "n1_vendor_baseline_diagnostic_launch_result_v1"
    result["task_profile"] = TASK_PROFILE_ID
    return result


def build_plan(spec_path: Path) -> dict[str, Any]:
    plan = _B.build_plan(spec_path)
    payload = plan["canonical_payload"]
    spec = payload["spec"]
    vendor_identity = _validate_vendor_identity_manifest(
        Path(spec["source"]["checkout"]),
        spec["source"]["commit_sha"],
    )
    authoritative_sha = vendor_identity[
        "runtime_training_contract_sha256"
    ]
    if spec[VENDOR_CONTRACT_FIELD] != authoritative_sha:
        raise LaunchRefused(
            "spec vendor runtime contract SHA differs from tracked authority"
        )
    actual_authority = _validate_actual_vendor_authority(
        Path(spec["source"]["checkout"]),
        spec["source"]["commit_sha"],
        payload["bundle"],
        authoritative_sha,
    )
    _validate_vendor_runtime_binding(
        Path(spec["source"]["checkout"]),
        spec["source"]["commit_sha"],
        payload["bundle"],
        authoritative_sha,
    )
    if spec["stage"] == PUSH_EVIDENCE_STAGE:
        payload[PUSH_EVIDENCE_CLAIM_FIELD] = (
            _validate_push_evidence_runtime_sources()
        )
    payload["vendor_runtime_authority"] = actual_authority
    if spec["stage"] == "long":
        payload[VENDOR_PROBE_GATE_FIELD] = (
            _validate_vendor_probe_gate_receipt(
                Path(spec["source"]["checkout"]),
                spec["source"]["commit_sha"],
                spec[VENDOR_PROBE_GATE_FIELD],
                spec=spec,
                payload=payload,
            )
        )
    plan["launch_claim_sha256"] = canonical_sha256(payload)
    return plan


def _load_internal_plan_for_vendor_binding(
    claim_path: Path, expected_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the immutable claim envelope before vendor-specific reads."""

    absolute = _B._absolute_path(
        str(claim_path), name="internal claim", must_exist=True
    )
    _B._stable_regular_file(absolute, name="internal launch claim")
    raw = absolute.read_bytes()
    plan = _B._strict_json_bytes(raw, name="internal launch claim")
    if raw != _B._canonical_bytes(plan) + b"\n":
        raise LaunchRefused("internal launch claim is not canonical")
    outer = _B._exact_dict(
        plan,
        ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
        name="internal launch claim",
    )
    if (
        outer["schema_version"] != SCHEMA_VERSION
        or outer["kind"] != CLAIM_KIND
        or outer["launch_claim_sha256"] != expected_sha
        or canonical_sha256(outer["canonical_payload"]) != expected_sha
    ):
        raise LaunchRefused("internal launch claim digest differs")
    payload = outer["canonical_payload"]
    if type(payload) is not dict or payload.get("kind") != CLAIM_KIND:
        raise LaunchRefused("internal launch payload kind differs")
    return plan, payload


def _internal_exec(claim_path: Path, expected_sha: str, lock_fd: int) -> int:
    _plan, payload = _load_internal_plan_for_vendor_binding(
        claim_path, expected_sha
    )
    spec = _validate_spec_document(
        payload["spec"], namespace_claimed=True
    )
    _revalidate_push_evidence_claim_sources(payload)
    checkout = Path(spec["source"]["checkout"])
    commit_sha = spec["source"]["commit_sha"]
    _B._verify_clean_source(checkout, commit_sha)
    vendor_identity = _validate_vendor_identity_manifest(
        checkout, commit_sha
    )
    authoritative_sha = vendor_identity[
        "runtime_training_contract_sha256"
    ]
    if spec[VENDOR_CONTRACT_FIELD] != authoritative_sha:
        raise LaunchRefused(
            "spec vendor runtime contract SHA differs from tracked authority"
        )
    bundle = _B._validate_bundle(
        checkout,
        commit_sha,
        spec["bundle"],
        expected_action=spec["action_id"],
        expected_scope=spec["scope"],
        require_dynamic_ready=True,
    )
    actual_authority = _validate_actual_vendor_authority(
        checkout,
        commit_sha,
        bundle,
        authoritative_sha,
    )
    if payload.get("vendor_runtime_authority") != actual_authority:
        raise LaunchRefused(
            "vendor runtime authority summary differs from immutable claim"
        )
    _validate_vendor_runtime_binding(
        checkout,
        commit_sha,
        bundle,
        authoritative_sha,
    )
    if spec["stage"] == "long":
        observed_gate = _validate_vendor_probe_gate_receipt(
            checkout,
            commit_sha,
            spec[VENDOR_PROBE_GATE_FIELD],
            spec=spec,
            payload=payload,
        )
        if payload.get(VENDOR_PROBE_GATE_FIELD) != observed_gate:
            raise LaunchRefused(
                "vendor probe gate receipt differs from immutable claim"
            )
    # The safety base repeats the complete claim/source/bundle/GPU validation
    # before exec.  The extra pass above exists only to add the vendor binding.
    return _B._internal_exec(claim_path, expected_sha, lock_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
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
        if args.command == "_exec":
            return _internal_exec(
                Path(args.claim), args.claim_sha256, args.gpu_lock_fd
            )
        plan = build_plan(Path(args.spec))
        if args.command == "plan":
            print(
                _B.json.dumps(
                    plan,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            return 0
        result = launch(plan, confirm_claim=args.confirm_claim)
        print(
            _B.json.dumps(
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


_configure_base()


if __name__ == "__main__":
    raise SystemExit(main())
