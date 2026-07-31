#!/usr/bin/env python3
"""Launch one fail-closed, single-GPU A3-vendor ActionBall diagnostic.

This is a deliberately thin policy adapter over
``launch_n1_reward_screen_diagnostic.py``.  It reuses that launcher's
canonical-spec, clean-commit, tracked-bundle, fresh-namespace, empty-GPU and
lifetime-lock implementation, while narrowing the scientific recipe to the
immutable ``HOPEPingPongActionBallA3VendorV1`` task profile.

Two exact stages are currently launchable: ``smoke`` (1 env x 2 updates) and
``probe`` (4096 envs x 5 updates).  ``long`` is schema-reserved but fails closed
until a later contract consumes a named ``vendor_probe_gate_receipt``.  Seeds
are restricted to 0, 1, or 2, and the only action is ``bh_loop_c``.  There is
no arbitrary Hydra override input.
The result remains diagnostic-only: it cannot mint formal evaluator,
promotion, resume, export, or judge authority.

Unlike the historical reward-screen launcher, this adapter does not force
``stable_ready_plant`` and does not override reward weights, push, PD-gain
randomization, or control-step action delay.  Those settings therefore come
from the exact tracked vendor task profile named above.  The spec must also
pin ``vendor_runtime_training_contract_sha256``; the tracked dynamic-ready
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
    "240f3757e45006de9dc5f4ecabcfc40071058009751fd1f0b8eb92656e1801ff"
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
    "f66a9e59f441c22c465d3236d717c95354393d04c5975f58ece3e7612a65461a"
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
ALLOWED_SEEDS = frozenset((0, 1, 2))
ALLOWED_STAGES = frozenset(("smoke", "probe", "long"))


LaunchRefused = _B.LaunchRefused
canonical_sha256 = _B.canonical_sha256

_base_validate_spec_document = _B._validate_spec_document
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
    _B._validate_spec_document = _validate_spec_document
    _B._build_training_argv = _build_training_argv
    _B._validate_dynamic_ready = _validate_vendor_dynamic_ready
    _B._validate_runtime_sources = _validate_runtime_sources
    _B.launch = launch


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
    base_document = dict(document)
    del base_document[VENDOR_CONTRACT_FIELD]
    spec = _base_validate_spec_document(
        base_document, namespace_claimed=namespace_claimed
    )
    if spec["reward_profile"] != REWARD_PROFILE:
        raise LaunchRefused(
            f"reward_profile must be exactly {REWARD_PROFILE!r}"
        )
    if spec["seed"] not in ALLOWED_SEEDS:
        raise LaunchRefused("vendor diagnostic seed must be exactly 0, 1, or 2")
    if spec["stage"] == "long":
        raise LaunchRefused(
            "vendor long requires a named vendor_probe_gate_receipt; "
            "this launcher revision authorizes smoke/probe only"
        )
    if spec["stage"] not in ALLOWED_STAGES:
        raise LaunchRefused("vendor diagnostic stage must be smoke, probe, or long")
    spec[VENDOR_CONTRACT_FIELD] = contract_sha
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
    """Build the inherited argv, then remove reward-screen-only mutations."""

    argv = _base_build_training_argv(spec, bundle)
    remove_prefixes = (
        "+task.rewards.motion_scale=",
        "task.rewards.racket_position_weight=",
        "task.rewards.racket_velocity_weight=",
        "task.rewards.racket_normal_weight=",
    )
    result: list[str] = []
    task_replaced = False
    stable_ready_seen = False
    for item in argv:
        if item == "task=HOPEPingPongActionBall":
            result.append(f"task={TASK_PROFILE_ID}")
            task_replaced = True
            continue
        if item == "+task.domain_rand.stable_ready_plant=true":
            stable_ready_seen = True
            continue
        if item.startswith(remove_prefixes):
            continue
        result.append(item)
    if not task_replaced or not stable_ready_seen:
        raise LaunchRefused(
            "diagnostic base argv contract changed; vendor adapter refuses drift"
        )
    forbidden_fragments = (
        "stable_ready_plant",
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
    payload["vendor_runtime_authority"] = actual_authority
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
