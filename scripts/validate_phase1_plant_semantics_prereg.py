#!/usr/bin/env python3
"""Validate the fail-closed Phase-1 plant-semantics repair preregistration.

This tool is deliberately dependency-free and launch-free.  It validates the
experiment/evidence contract; it does not tune friction, touch a Pod, launch
training, or authorize a hardware probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "phase1_plant_semantics_repair_prereg_20260711.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_REL_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")

EXPECTED_CURRENT_CELLS = {
    "SZ": ("shared_plus_y", "all_zero_joint_friction", True),
    "SP": ("shared_plus_y", "legacy_direct_number_physx_proxy", False),
    "LZ": ("legacy_signed_vs_A", "all_zero_joint_friction", False),
    "LP": ("legacy_signed_vs_A", "legacy_direct_number_physx_proxy", False),
}
EXPECTED_LATENT_FAMILIES = {
    "constant_breakaway_plus_coulomb_plus_viscous",
    "load_affine_breakaway_plus_coulomb_plus_viscous",
    "monotone_load_table_breakaway_plus_coulomb_plus_viscous",
}
EXPECTED_READY_BINDINGS = {
    "measurement_protocol_sha256",
    "raw_calibration_dataset_manifest_sha256",
    "repeatability_report_sha256",
    "session_split_sha256",
    "numeric_threshold_contract_sha256",
    "latent_model_selection_report_sha256",
    "latent_model_sha256",
    "physx_adapter_source_sha256",
    "mujoco_adapter_source_sha256",
    "vendor_gate3_mjcf_sha256",
    "vendor_gate3_runtime_source_sha256",
    "vendor_gate3_plant_instantiation_report_sha256",
    "vendor_gate3b_plant_eval_profile_sha256",
    "physx_runtime_probe_report_sha256",
    "mujoco_runtime_probe_report_sha256",
    "cross_engine_equivalence_report_sha256",
    "a3_asset_sha256",
    "joint_order_sha256",
    "engine_versions_and_solver_contract_sha256",
    "fresh_axis_runtime_manifest_sha256",
}
EXPECTED_CLAIM_LABELS = {
    "training_lineage_exact",
    "evaluation_protocol_exact",
    "plant_adapter_replay_exact",
    "cross_engine_plant_equivalence_passed",
    "deployment_plant_calibrated",
    "deployment_candidate",
}


class PlantPreregError(ValueError):
    """A plant-semantics preregistration contract violation."""


def _required(mapping: dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise PlantPreregError(f"{context} missing required keys: {missing}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlantPreregError(message)


def _sha(value: Any, context: str) -> str:
    _require(isinstance(value, str) and bool(SHA256_RE.fullmatch(value)),
             f"{context} must be a lowercase SHA-256")
    return value


def _official_urls(values: Any, *, host_suffixes: tuple[str, ...], context: str) -> None:
    _require(isinstance(values, list) and len(values) >= 1,
             f"{context} must be a non-empty URL list")
    for index, raw in enumerate(values):
        _require(isinstance(raw, str), f"{context}[{index}] must be a URL string")
        parsed = urlparse(raw)
        _require(parsed.scheme == "https" and parsed.hostname is not None,
                 f"{context}[{index}] must be an https URL")
        _require(any(parsed.hostname == suffix or parsed.hostname.endswith("." + suffix)
                     for suffix in host_suffixes),
                 f"{context}[{index}] must use an official primary-source host")


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_manifest(data: dict[str, Any]) -> None:
    _require(isinstance(data, dict), "manifest root must be an object")
    _required(
        data,
        {
            "schema_version", "preregistration_id", "created_date", "status", "scope",
            "repository_baseline", "source_semantics", "current_phase1_cells",
            "legacy_frozen_probe", "calibration_protocol", "latent_model_selection",
            "adapter_contract", "minimum_training_axis", "checkpoint_decision_contract",
            "claim_labels", "required_before_ready", "evidence_bindings",
            "fail_closed_conditions",
        },
        "manifest",
    )
    _require(data["schema_version"] == 1, "schema_version must be 1")
    _require(data["preregistration_id"] == "phase1-plant-semantics-repair-v1",
             "unexpected preregistration_id")
    _require(data["created_date"] == "2026-07-11", "created_date must remain frozen")

    scope = data["scope"]
    _require(isinstance(scope, dict), "scope must be an object")
    for key in (
        "changes_current_phase1_training",
        "changes_current_phase1_evaluation",
        "permits_real_robot_commands",
    ):
        _require(scope.get(key) is False, f"scope.{key} must be false")

    baseline = data["repository_baseline"]
    _require(isinstance(baseline, dict), "repository_baseline must be an object")
    _require(
        isinstance(baseline.get("git_commit"), str)
        and bool(GIT_OBJECT_RE.fullmatch(baseline["git_commit"])),
        "repository_baseline.git_commit must be a full lowercase 40-character Git object id",
    )
    sources = baseline.get("audited_sources")
    _require(isinstance(sources, list) and len(sources) >= 6,
             "repository_baseline.audited_sources must bind all implementation seams")
    seen_paths: set[str] = set()
    for index, item in enumerate(sources):
        context = f"repository_baseline.audited_sources[{index}]"
        _require(isinstance(item, dict), f"{context} must be an object")
        path = item.get("path")
        _require(
            isinstance(path, str)
            and bool(SAFE_REL_PATH_RE.fullmatch(path))
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts,
            f"{context}.path must be a safe repository-relative path",
        )
        _require(path not in seen_paths, f"duplicate audited source path {path!r}")
        seen_paths.add(path)
        _sha(item.get("sha256"), f"{context}.sha256")
    required_source_suffixes = {
        "robots/agibot_a3.py",
        "utils/training_contract.py",
        "utils/plant_contract.py",
        "scripts/compile_semantics_correct_plant_contract.py",
        "scripts/train.py",
        "scripts/mujoco_eval_onnx.py",
        "scripts/judge.sh",
        "configs/phase1_scaleout_matrix_20260711.json",
    }
    _require(
        all(any(path.endswith(suffix) for path in seen_paths) for suffix in required_source_suffixes),
        "audited sources do not cover the training, plant compiler, evaluator, judge and matrix seams",
    )

    semantics = data["source_semantics"]
    _require(isinstance(semantics, dict), "source_semantics must be an object")
    physx = semantics.get("physx")
    mujoco = semantics.get("mujoco")
    mapping = semantics.get("mapping_rule")
    _require(isinstance(physx, dict) and isinstance(mujoco, dict) and isinstance(mapping, dict),
             "source_semantics must contain physx, mujoco and mapping_rule objects")
    _require(physx.get("units") == "dimensionless",
             "PhysX legacy joint-friction coefficient must remain dimensionless")
    _require(physx.get("load_dependence") == "transmitted_spatial_force_dependent",
             "PhysX load-dependent semantics were weakened")
    _official_urls(
        physx.get("primary_sources"),
        host_suffixes=("isaac-sim.github.io", "nvidia-omniverse.github.io"),
        context="source_semantics.physx.primary_sources",
    )
    _require(mujoco.get("units") == "generalized_force; N*m for the A3 scalar hinge joints",
             "MuJoCo frictionloss units must be expressed as generalized force")
    _require(mujoco.get("load_dependence") == "load_independent",
             "MuJoCo frictionloss must remain load-independent")
    _official_urls(
        mujoco.get("primary_sources"),
        host_suffixes=("mujoco.readthedocs.io",),
        context="source_semantics.mujoco.primary_sources",
    )
    _require(mapping.get("kind") == "no_direct_numeric_mapping",
             "mapping_rule.kind must forbid direct numeric mapping")
    _require(mapping.get("same_number_allowed") is False,
             "same numeric PhysX/MuJoCo values must remain forbidden")
    _require("engine-specific adapters" in str(mapping.get("required_method", "")),
             "mapping must require separately fitted engine adapters")
    _require("all-zero" in str(mapping.get("zero_special_case", "")),
             "mapping must preserve the current all-zero exact special case")

    cells = data["current_phase1_cells"]
    _require(isinstance(cells, dict) and set(cells) == set(EXPECTED_CURRENT_CELLS),
             "current_phase1_cells must contain exactly SZ/SP/LZ/LP")
    for name, (pairing, plant, exact_eligible) in EXPECTED_CURRENT_CELLS.items():
        cell = cells[name]
        _require(isinstance(cell, dict), f"current_phase1_cells.{name} must be an object")
        _require(cell.get("face_command_pairing") == pairing,
                 f"{name} face pairing changed")
        _require(cell.get("plant") == plant, f"{name} plant semantics changed")
        _require(cell.get("current_bankexam_exact_eligible") is exact_eligible,
                 f"{name} current exact-eligibility changed")
        _require(cell.get("plant_semantics_calibrated") is False,
                 f"{name} cannot be labelled calibrated")
        _require(cell.get("deployment_candidate") is False,
                 f"{name} cannot be labelled a deployment candidate")

    frozen = data["legacy_frozen_probe"]
    _require(isinstance(frozen, dict), "legacy_frozen_probe must be an object")
    _require(frozen.get("role") == "directional_deployment_blocker_not_calibration_data",
             "legacy frozen probe may only be a directional blocker")
    _require(frozen.get("raw_artifact_sha256") is None,
             "the untracked legacy raw probe must not acquire an invented SHA")
    _require(float(frozen.get("zero_plant_virtual_hit")) == 0.9997,
             "legacy zero-plant virtual-hit record changed")
    _require(float(frozen.get("nonzero_legacy_plant_virtual_hit")) == 0.63,
             "legacy non-zero-plant virtual-hit record changed")
    _require(float(frozen.get("zero_plant_fall")) == 0.27,
             "legacy zero-plant fall record changed")
    _require(float(frozen.get("nonzero_legacy_plant_fall")) == 0.87,
             "legacy non-zero-plant fall record changed")

    protocol = data["calibration_protocol"]
    _require(isinstance(protocol, dict), "calibration_protocol must be an object")
    axes = protocol.get("required_axes")
    _require(isinstance(axes, dict), "calibration_protocol.required_axes must be an object")
    _require(set(axes.get("directions", [])) == {"negative", "positive"},
             "calibration must measure both directions")
    _require(set(axes.get("velocity_regimes", [])) == {
        "breakaway_from_rest", "low_speed_sliding", "nominal_speed_sliding"
    }, "calibration must cover breakaway, low-speed and nominal-speed regimes")
    _require(isinstance(axes.get("minimum_distinct_load_conditions"), int)
             and axes["minimum_distinct_load_conditions"] >= 3,
             "calibration requires at least three load conditions")
    _require(isinstance(axes.get("minimum_repetitions_per_cell"), int)
             and axes["minimum_repetitions_per_cell"] >= 5,
             "calibration requires at least five repetitions per cell")
    fields = protocol.get("required_sample_fields")
    _require(isinstance(fields, list) and len(fields) == len(set(fields)),
             "required_sample_fields must be a unique list")
    for field in (
        "joint_name", "joint_velocity_rad_s", "commanded_or_measured_joint_torque_Nm",
        "estimated_transmitted_load_Nm", "load_estimator_uncertainty_Nm", "temperature_C",
        "firmware_version", "source_session_id",
    ):
        _require(field in fields, f"calibration sample field {field!r} is required")
    _require("complete source session" in str(protocol.get("split_rule", "")),
             "calibration split must isolate complete sessions")
    _require("before fitting adapters" in str(protocol.get("threshold_rule", "")),
             "numeric tolerances must be frozen before adapter fitting")
    _require("authorizes no hardware command" in str(protocol.get("real_robot_rule", "")),
             "preregistration must not authorize hardware commands")

    latent = data["latent_model_selection"]
    _require(isinstance(latent, dict), "latent_model_selection must be an object")
    _require(set(latent.get("candidate_families", [])) == EXPECTED_LATENT_FAMILIES,
             "latent-model candidate family set changed")
    constraints = latent.get("constraints")
    _require(isinstance(constraints, list) and len(constraints) >= 5,
             "latent-model constraints are incomplete")
    _require("simplest family" in str(latent.get("selection_rule", "")),
             "latent-model selection must preserve the simplest-passing-family rule")
    _require("same selected latent-model SHA" in str(latent.get("shared_model_rule", "")),
             "both adapters must bind one shared latent model")

    adapters = data["adapter_contract"]
    _require(isinstance(adapters, dict), "adapter_contract must be an object")
    _require(set(adapters.get("physx", {}).get("allowed_backends", [])) == {
        "native_transmitted_force_coefficient",
        "versioned_explicit_generalized_friction_adapter",
    }, "unexpected PhysX adapter backends")
    _require(set(adapters.get("mujoco", {}).get("allowed_backends", [])) == {
        "native_frictionloss_plus_damping",
        "versioned_explicit_generalized_friction_adapter",
    }, "unexpected MuJoCo adapter backends")
    mujoco_adapter = adapters.get("mujoco", {})
    _require(
        mujoco_adapter.get("final_runtime_target")
        == "agibot_vendor_mujoco_gate3_gate3b",
        "the final MuJoCo target must remain the Agibot vendor Gate3/Gate3B runtime",
    )
    _require(
        mujoco_adapter.get("vendor_mjcf_path")
        == "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml",
        "the Agibot vendor Gate3/Gate3B MJCF path changed",
    )
    _require(
        "standalone generic MuJoCo wrapper is development evidence only"
        in str(mujoco_adapter.get("final_evidence_rule", "")),
        "generic MuJoCo evidence cannot replace the vendor Gate3/Gate3B runtime",
    )
    runtime_binding = adapters.get("runtime_binding")
    _require(isinstance(runtime_binding, list) and len(runtime_binding) >= 7,
             "adapter runtime binding is incomplete")
    cross_accept = str(adapters.get("cross_engine_acceptance", ""))
    _require("parameter equality is neither required nor sufficient" in cross_accept,
             "cross-engine acceptance must be behavioral, not numeric equality")

    axis = data["minimum_training_axis"]
    _require(isinstance(axis, dict), "minimum_training_axis must be an object")
    _require(axis.get("axis_id") == "fresh_shared_face_zero_vs_calibrated_plant_v1",
             "unexpected plant axis id")
    _require(axis.get("fixed", {}).get("face_command_pairing") == "shared_plus_y",
             "the minimum plant axis must freeze shared_plus_y")
    train_levels = axis.get("training_factor", {}).get("levels")
    eval_plants = axis.get("evaluation_factors", {}).get("eval_plant_levels")
    engines = axis.get("evaluation_factors", {}).get("engine_levels")
    seeds = axis.get("paired_seed_blocks")
    _require(train_levels == ["Z_zero", "C_calibrated"],
             "training factor must be ordered Z_zero/C_calibrated")
    _require(eval_plants == ["Z_zero", "C_calibrated"],
             "evaluation plant levels must be ordered Z_zero/C_calibrated")
    _require(engines == ["isaac", "agibot_vendor_mujoco_gate3_gate3b"],
             "both Isaac and Agibot vendor Gate3/Gate3B MuJoCo evaluation legs are mandatory")
    _require(isinstance(seeds, list) and len(seeds) >= 2 and len(set(seeds)) == len(seeds)
             and all(isinstance(seed, int) and seed > 0 for seed in seeds),
             "at least two unique positive paired seed blocks are required")
    expected_arms = len(train_levels) * len(seeds)
    expected_evals = expected_arms * len(eval_plants) * len(engines)
    _require(axis.get("minimum_from_scratch_training_arms") == expected_arms,
             f"minimum_from_scratch_training_arms must be {expected_arms}")
    _require(axis.get("evaluations_per_milestone") == expected_evals,
             f"evaluations_per_milestone must be {expected_evals}")
    _require(set(axis.get("excluded_cells", [])) == {"SP", "LP"},
             "SP and LP must be excluded from the calibrated axis")
    _require("not the matched control" in str(axis.get("existing_SZ_reuse_rule", "")),
             "existing SZ reuse must fail closed on non-plant contract drift")
    contrasts = axis.get("primary_contrasts")
    _require(isinstance(contrasts, list) and len(contrasts) == 4,
             "the four paired plant/cross-engine contrasts are required")

    checkpoint = data["checkpoint_decision_contract"]
    _require(isinstance(checkpoint, dict), "checkpoint_decision_contract must be an object")
    _require(checkpoint.get("screen_schedule_k") == 10,
             "screen schedule must remain q10")
    _require(checkpoint.get("decision_schedule_k") == 100
             and checkpoint.get("decision_quota_per_side") == 50,
             "decision schedule must remain 100 questions / 50 per side")
    _require(checkpoint.get("screen_only") is True
             and checkpoint.get("q10_may_stop_or_promote") is False,
             "q10 must remain screen-only")
    _require(checkpoint.get("retain_best_finite_checkpoint") is True,
             "best finite checkpoint retention is mandatory")
    _require(checkpoint.get("same_immutable_schedule_across_train_plant_and_engine") is True,
             "all plant/engine cells must use the same immutable schedule")
    _require(checkpoint.get("terminal_checkpoint_required_for_selection") is False,
             "selection must not wait for terminal checkpoint")

    labels = data["claim_labels"]
    _require(isinstance(labels, dict) and set(labels) == EXPECTED_CLAIM_LABELS,
             "claim_labels must keep provenance, protocol, adapter, equivalence, calibration and deployment separate")
    _require("alone says nothing about physical calibration" in labels["training_lineage_exact"],
             "training lineage must not imply calibration")
    _require("independent G07 safety gates" in labels["deployment_candidate"],
             "deployment candidate must remain gated by G07")

    required = data["required_before_ready"]
    _require(isinstance(required, list) and len(required) == len(set(required)),
             "required_before_ready must be a unique list")
    _require(set(required) == EXPECTED_READY_BINDINGS,
             "required_before_ready binding set changed")
    evidence = data["evidence_bindings"]
    _require(isinstance(evidence, dict), "evidence_bindings must be an object")
    unknown = sorted(set(evidence) - EXPECTED_READY_BINDINGS)
    _require(not unknown, f"unknown evidence bindings: {unknown}")
    for key, value in evidence.items():
        _sha(value, f"evidence_bindings.{key}")

    status = data["status"]
    _require(status in {"blocked_on_calibration_evidence", "ready_for_semantics_correct_launch"},
             "unsupported status")
    if status == "ready_for_semantics_correct_launch":
        missing = sorted(EXPECTED_READY_BINDINGS - set(evidence))
        _require(not missing, f"ready status is missing evidence bindings: {missing}")
    else:
        _require(set(evidence) != EXPECTED_READY_BINDINGS,
                 "all readiness evidence is present; status must be reviewed and advanced explicitly")

    fail_closed = data["fail_closed_conditions"]
    _require(isinstance(fail_closed, list) and len(fail_closed) >= 10,
             "fail_closed_conditions are incomplete")
    joined_failures = "\n".join(str(value) for value in fail_closed)
    for token in (
        "SP or LP", "copied numerically", "q10", "Isaac-only", "out-of-support",
        "NaN/Inf", "standalone generic MuJoCo adapter",
    ):
        _require(token in joined_failures, f"fail_closed_conditions missing {token!r} guard")

    forbidden_parameter_keys = {
        "calibrated_physx_coefficients",
        "calibrated_mujoco_frictionloss",
        "shared_numeric_friction_vector",
    }
    present_forbidden = sorted(forbidden_parameter_keys & set(_walk_keys(data)))
    _require(not present_forbidden,
             f"preregistration must not guess calibrated adapter parameters: {present_forbidden}")


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlantPreregError(f"cannot read manifest {path}: {exc}") from None
    validate_manifest(data)
    return data


def verify_repository_baseline(data: dict[str, Any], repo_root: Path = REPO_ROOT) -> None:
    for item in data["repository_baseline"]["audited_sources"]:
        path = repo_root / item["path"]
        if not path.is_file():
            raise PlantPreregError(f"audited repository source is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise PlantPreregError(
                f"audited source SHA mismatch for {item['path']}: "
                f"expected {item['sha256']}, got {actual}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--verify-repository-baseline",
        action="store_true",
        help="also compare the audited source paths against their preregistered bytes",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        data = load_manifest(args.manifest)
        if args.verify_repository_baseline:
            verify_repository_baseline(data)
    except PlantPreregError as exc:
        print(f"PLANT_SEMANTICS_PREREG_FAIL: {exc}")
        return 2
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    axis = data["minimum_training_axis"]
    print("PLANT_SEMANTICS_PREREG_OK")
    print(f"manifest_sha256={digest}")
    print(f"status={data['status']}")
    print(f"minimum_training_arms={axis['minimum_from_scratch_training_arms']}")
    print(f"evaluations_per_milestone={axis['evaluations_per_milestone']}")
    print("hardware_commands_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
