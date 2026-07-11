"""Fail-closed contract compiler for a semantics-correct A3 joint-friction plant.

This module is deliberately dependency-light and is not wired into the current
schema-3 training or BankExam paths.  It defines the future plant-contract v1
boundary: one calibrated physical model, two independently fitted engine
adapters, explicit units, content-addressed evidence, and a support-envelope
check before an adapter can be prepared for runtime replay.

The MuJoCo adapter's final target is the Agibot vendor Gate3/Gate3B runtime;
standalone generic MuJoCo evaluation cannot satisfy this schema.

It never converts a non-zero PhysX coefficient into MuJoCo ``frictionloss``.
The only cross-unit value with engine-independent meaning is exact zero.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


PLANT_CONTRACT_SCHEMA_VERSION = 1
PLANT_CONTRACT_STATUS = "ready_for_semantics_correct_runtime"
PLANT_CONTRACT_ROLE = "fresh_semantics_correct_calibrated_plant"
A3_ACTUATED_JOINT_COUNT = 31

PHYSX_ENGINE = "physx"
MUJOCO_ENGINE = "mujoco"
ENGINES = (PHYSX_ENGINE, MUJOCO_ENGINE)

PHYSX_BACKEND = "native_transmitted_force_coefficient"
MUJOCO_BACKEND = "native_frictionloss_plus_damping"
PHYSX_SEMANTICS = "load_dependent_spatial_force_coefficient"
MUJOCO_SEMANTICS = "load_independent_coulomb_bound_plus_viscous"

DIMENSIONLESS = "dimensionless"
TORQUE_NM = "N*m"
VISCOUS_NM_S_PER_RAD = "N*m*s/rad"
LOAD_NM = "N*m"
SPEED_RAD_S = "rad/s"
TEMPERATURE_C = "degC"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class PlantContractError(ValueError):
    """Raised when a plant contract cannot support an exact replay claim."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise PlantContractError(message)


def _exact_keys(value: Mapping[str, Any], keys: set[str], where: str) -> None:
    actual = set(value)
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        raise PlantContractError(
            f"{where} keys changed: missing={missing}, unknown={unknown}"
        )


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    _fail(isinstance(value, Mapping), f"{where} must be an object")
    return value


def _sha(value: Any, where: str) -> str:
    _fail(
        isinstance(value, str) and bool(SHA256_RE.fullmatch(value)),
        f"{where} must be a lowercase SHA-256",
    )
    return value


def _safe_id(value: Any, where: str) -> str:
    _fail(
        isinstance(value, str) and bool(SAFE_ID_RE.fullmatch(value)),
        f"{where} must be a safe non-empty identifier",
    )
    return value


def _positive_number(value: Any, where: str) -> float:
    _fail(not isinstance(value, bool), f"{where} must be a finite positive number")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise PlantContractError(f"{where} must be a finite positive number") from None
    _fail(math.isfinite(out) and out > 0.0, f"{where} must be a finite positive number")
    return out


def _finite_vector(
    value: Any,
    *,
    length: int,
    where: str,
    non_negative: bool,
) -> list[float]:
    _fail(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{where} must be a numeric array",
    )
    _fail(len(value) == length, f"{where} must contain exactly {length} values")
    out: list[float] = []
    for index, raw in enumerate(value):
        _fail(not isinstance(raw, bool), f"{where}[{index}] must be numeric")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise PlantContractError(f"{where}[{index}] must be numeric") from None
        _fail(math.isfinite(number), f"{where}[{index}] contains NaN/Inf")
        if non_negative:
            _fail(number >= 0.0, f"{where}[{index}] must be non-negative")
        out.append(number)
    return out


def _range(value: Any, *, where: str, non_negative: bool) -> tuple[float, float]:
    _fail(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2,
        f"{where} must be [min, max]",
    )
    parsed = _finite_vector(
        value, length=2, where=where, non_negative=non_negative
    )
    _fail(parsed[0] <= parsed[1], f"{where} min must be <= max")
    return parsed[0], parsed[1]


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON bytes (sorted keys, compact separators, UTF-8)."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PlantContractError(
            f"canonical payload is not finite JSON (NaN/Inf or unsupported type): {exc}"
        ) from None
    return hashlib.sha256(payload).hexdigest()


def contract_payload_sha256(contract: Mapping[str, Any]) -> str:
    """Hash a contract while excluding its non-self-referential digest field."""

    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return canonical_sha256(payload)


def bind_contract_sha256(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied contract with its canonical payload digest bound."""

    out = copy.deepcopy(dict(draft))
    out.pop("contract_sha256", None)
    out["contract_sha256"] = contract_payload_sha256(out)
    return out


def zero_only_unit_conversion(
    values: Sequence[Any], *, source_units: str, target_units: str
) -> list[float]:
    """Convert only identical-unit vectors or exact zero across different units.

    A non-zero dimensionless PhysX coefficient and a non-zero MuJoCo torque are
    incommensurate.  This helper exists specifically so callers cannot hide a
    direct-number copy behind a generic "converter" API.
    """

    parsed = _finite_vector(
        values,
        length=len(values),
        where="quantity vector",
        non_negative=False,
    )
    _fail(isinstance(source_units, str) and source_units, "source_units is required")
    _fail(isinstance(target_units, str) and target_units, "target_units is required")
    if source_units == target_units:
        return parsed
    if all(value == 0.0 for value in parsed):
        return [0.0] * len(parsed)
    raise PlantContractError(
        "no non-zero numeric conversion exists between "
        f"{source_units!r} and {target_units!r}; fit an engine-specific adapter"
    )


def _validate_support(value: Any, *, where: str) -> dict[str, Any]:
    support = _mapping(value, where)
    _exact_keys(
        support,
        {
            "load_abs_Nm",
            "speed_abs_rad_s",
            "temperature_C",
            "pose_ids",
            "pose_ids_sha256",
        },
        where,
    )
    load = _range(support["load_abs_Nm"], where=f"{where}.load_abs_Nm", non_negative=True)
    speed = _range(
        support["speed_abs_rad_s"],
        where=f"{where}.speed_abs_rad_s",
        non_negative=True,
    )
    temperature = _range(
        support["temperature_C"],
        where=f"{where}.temperature_C",
        non_negative=False,
    )
    pose_ids_raw = support["pose_ids"]
    _fail(
        isinstance(pose_ids_raw, Sequence)
        and not isinstance(pose_ids_raw, (str, bytes))
        and bool(pose_ids_raw),
        f"{where}.pose_ids must be a non-empty array",
    )
    pose_ids = [_safe_id(value, f"{where}.pose_ids") for value in pose_ids_raw]
    _fail(len(set(pose_ids)) == len(pose_ids), f"{where}.pose_ids must be unique")
    expected_pose_sha = canonical_sha256(pose_ids)
    _fail(
        _sha(support["pose_ids_sha256"], f"{where}.pose_ids_sha256")
        == expected_pose_sha,
        f"{where}.pose_ids_sha256 does not bind pose_ids",
    )
    return {
        "load_abs_Nm": load,
        "speed_abs_rad_s": speed,
        "temperature_C": temperature,
        "pose_ids": pose_ids,
    }


def _validate_adapter(
    value: Any,
    *,
    engine: str,
    joint_count: int,
    latent_model_sha256: str,
    threshold_contract_sha256: str,
    probe_schedule_sha256: str,
) -> dict[str, Any]:
    where = f"adapters.{engine}"
    adapter = _mapping(value, where)
    common = {
        "engine",
        "engine_version",
        "runtime_target",
        "backend",
        "parameter_semantics",
        "source_parameter_origin",
        "latent_model_sha256",
        "threshold_contract_sha256",
        "adapter_source_sha256",
        "fit_report_sha256",
        "runtime_probe_report_sha256",
        "runtime_probe_passed",
        "runtime_source_sha256",
        "runtime_instantiation_report_sha256",
        "probe_schedule_sha256",
        "asset_sha256",
        "solver_contract_sha256",
        "physics_step_dt_s",
        "policy_step_dt_s",
        "control_decimation",
        "integrator",
        "parameters",
    }
    if engine == MUJOCO_ENGINE:
        common.add("vendor_mjcf_path")
    _exact_keys(adapter, common, where)
    _fail(adapter["engine"] == engine, f"{where}.engine must be {engine!r}")
    _fail(
        isinstance(adapter["engine_version"], str) and adapter["engine_version"].strip(),
        f"{where}.engine_version must be explicit",
    )
    expected_runtime_target = (
        "isaac_training_and_companion_eval"
        if engine == PHYSX_ENGINE
        else "agibot_vendor_mujoco_gate3_gate3b"
    )
    _fail(
        adapter["runtime_target"] == expected_runtime_target,
        f"{where}.runtime_target must be {expected_runtime_target!r}",
    )
    if engine == MUJOCO_ENGINE:
        _fail(
            adapter["vendor_mjcf_path"]
            == "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
            "a3_pingpong/a3_pingpong.xml",
            f"{where}.vendor_mjcf_path must bind the Agibot Gate3/Gate3B asset",
        )
    _fail(
        adapter["source_parameter_origin"]
        == "engine_specific_fit_to_shared_latent_model",
        f"{where} must be independently fit to the shared latent model",
    )
    _fail(
        _sha(adapter["latent_model_sha256"], f"{where}.latent_model_sha256")
        == latent_model_sha256,
        f"{where}.latent_model_sha256 drifted",
    )
    _fail(
        _sha(
            adapter["threshold_contract_sha256"],
            f"{where}.threshold_contract_sha256",
        )
        == threshold_contract_sha256,
        f"{where}.threshold_contract_sha256 drifted",
    )
    for key in (
        "adapter_source_sha256",
        "fit_report_sha256",
        "runtime_probe_report_sha256",
        "runtime_source_sha256",
        "runtime_instantiation_report_sha256",
        "asset_sha256",
        "solver_contract_sha256",
    ):
        _sha(adapter[key], f"{where}.{key}")
    _fail(
        adapter["runtime_probe_passed"] is True,
        f"{where}.runtime_probe_passed must be true",
    )
    _fail(
        _sha(adapter["probe_schedule_sha256"], f"{where}.probe_schedule_sha256")
        == probe_schedule_sha256,
        f"{where}.probe_schedule_sha256 drifted",
    )
    physics_dt = _positive_number(adapter["physics_step_dt_s"], f"{where}.physics_step_dt_s")
    policy_dt = _positive_number(adapter["policy_step_dt_s"], f"{where}.policy_step_dt_s")
    decimation = adapter["control_decimation"]
    _fail(
        isinstance(decimation, int) and not isinstance(decimation, bool) and decimation > 0,
        f"{where}.control_decimation must be a positive integer",
    )
    _fail(
        math.isclose(policy_dt, physics_dt * decimation, rel_tol=0.0, abs_tol=1e-12),
        f"{where} policy dt must equal physics dt * control_decimation",
    )
    _fail(
        isinstance(adapter["integrator"], str) and adapter["integrator"].strip(),
        f"{where}.integrator must be explicit",
    )

    parameters = _mapping(adapter["parameters"], f"{where}.parameters")
    if engine == PHYSX_ENGINE:
        _fail(adapter["backend"] == PHYSX_BACKEND, f"{where}.backend is not implemented")
        _fail(
            adapter["parameter_semantics"] == PHYSX_SEMANTICS,
            f"{where}.parameter_semantics must describe PhysX load dependence",
        )
        _exact_keys(parameters, {"friction_coefficient"}, f"{where}.parameters")
        quantity = _mapping(
            parameters["friction_coefficient"],
            f"{where}.parameters.friction_coefficient",
        )
        _exact_keys(
            quantity,
            {"units", "values"},
            f"{where}.parameters.friction_coefficient",
        )
        _fail(
            quantity["units"] == DIMENSIONLESS,
            f"{where} PhysX friction coefficients must be dimensionless",
        )
        compiled_parameters = {
            "friction_coefficient": {
                "units": DIMENSIONLESS,
                "values": _finite_vector(
                    quantity["values"],
                    length=joint_count,
                    where=f"{where}.parameters.friction_coefficient.values",
                    non_negative=True,
                ),
            }
        }
    else:
        _fail(adapter["backend"] == MUJOCO_BACKEND, f"{where}.backend is not implemented")
        _fail(
            adapter["parameter_semantics"] == MUJOCO_SEMANTICS,
            f"{where}.parameter_semantics must describe constant-Nm plus viscous friction",
        )
        _exact_keys(parameters, {"frictionloss", "damping"}, f"{where}.parameters")
        compiled_parameters = {}
        for name, units in (
            ("frictionloss", TORQUE_NM),
            ("damping", VISCOUS_NM_S_PER_RAD),
        ):
            quantity = _mapping(parameters[name], f"{where}.parameters.{name}")
            _exact_keys(quantity, {"units", "values"}, f"{where}.parameters.{name}")
            _fail(
                quantity["units"] == units,
                f"{where}.parameters.{name}.units must be {units!r}",
            )
            compiled_parameters[name] = {
                "units": units,
                "values": _finite_vector(
                    quantity["values"],
                    length=joint_count,
                    where=f"{where}.parameters.{name}.values",
                    non_negative=True,
                ),
            }

    normalized = {
        "engine_version": adapter["engine_version"],
        "runtime_target": adapter["runtime_target"],
        "backend": adapter["backend"],
        "parameter_semantics": adapter["parameter_semantics"],
        "parameters": compiled_parameters,
        "asset_sha256": adapter["asset_sha256"],
        "solver_contract_sha256": adapter["solver_contract_sha256"],
        "physics_step_dt_s": physics_dt,
        "policy_step_dt_s": policy_dt,
        "control_decimation": decimation,
        "integrator": adapter["integrator"],
        "fit_report_sha256": adapter["fit_report_sha256"],
        "runtime_probe_report_sha256": adapter["runtime_probe_report_sha256"],
        "runtime_source_sha256": adapter["runtime_source_sha256"],
        "runtime_instantiation_report_sha256": adapter[
            "runtime_instantiation_report_sha256"
        ],
    }
    if engine == MUJOCO_ENGINE:
        normalized["vendor_mjcf_path"] = adapter["vendor_mjcf_path"]
    return normalized


def validate_plant_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a ready plant-contract v1.

    A passing result proves replay provenance inside the declared support
    envelope.  It does not authorize a real-robot command or policy promotion.
    """

    root = _mapping(contract, "plant contract")
    _exact_keys(
        root,
        {
            "schema_version",
            "contract_id",
            "contract_sha256",
            "status",
            "lineage_role",
            "hardware_commands_authorized",
            "legacy_direct_number_proxy",
            "joint_order",
            "physical_model",
            "cross_engine",
            "adapters",
        },
        "plant contract",
    )
    _fail(
        root["schema_version"] == PLANT_CONTRACT_SCHEMA_VERSION,
        f"schema_version must be {PLANT_CONTRACT_SCHEMA_VERSION}",
    )
    _safe_id(root["contract_id"], "contract_id")
    _fail(root["status"] == PLANT_CONTRACT_STATUS, "plant contract is not runtime-ready")
    _fail(root["lineage_role"] == PLANT_CONTRACT_ROLE, "lineage_role is not semantics-correct")
    _fail(
        root["hardware_commands_authorized"] is False,
        "plant contract cannot authorize hardware commands",
    )
    _fail(
        root["legacy_direct_number_proxy"] is False,
        "legacy direct-number proxies cannot pass the semantics-correct contract",
    )
    claimed_digest = _sha(root["contract_sha256"], "contract_sha256")
    actual_digest = contract_payload_sha256(root)
    _fail(claimed_digest == actual_digest, "contract_sha256 does not bind canonical payload")

    joint_order = _mapping(root["joint_order"], "joint_order")
    _exact_keys(joint_order, {"names", "sha256"}, "joint_order")
    names_raw = joint_order["names"]
    _fail(
        isinstance(names_raw, Sequence) and not isinstance(names_raw, (str, bytes)),
        "joint_order.names must be an array",
    )
    names = [_safe_id(value, "joint_order.names") for value in names_raw]
    _fail(
        len(names) == A3_ACTUATED_JOINT_COUNT,
        f"joint_order.names must contain exactly {A3_ACTUATED_JOINT_COUNT} A3 joints",
    )
    _fail(len(set(names)) == len(names), "joint_order.names must be unique")
    joint_order_sha = _sha(joint_order["sha256"], "joint_order.sha256")
    _fail(joint_order_sha == canonical_sha256(names), "joint_order.sha256 does not bind names")

    model = _mapping(root["physical_model"], "physical_model")
    _exact_keys(
        model,
        {
            "family",
            "units",
            "latent_model_sha256",
            "source_dataset_manifest_sha256",
            "session_split_sha256",
            "repeatability_report_sha256",
            "threshold_contract_sha256",
            "selection_report_sha256",
            "support_envelope_sha256",
            "support",
        },
        "physical_model",
    )
    _fail(
        model["family"]
        in {
            "constant_breakaway_plus_coulomb_plus_viscous",
            "load_affine_breakaway_plus_coulomb_plus_viscous",
            "monotone_load_table_breakaway_plus_coulomb_plus_viscous",
        },
        "physical_model.family is not preregistered",
    )
    units = _mapping(model["units"], "physical_model.units")
    _exact_keys(
        units,
        {"load", "speed", "generalized_torque", "viscous", "temperature"},
        "physical_model.units",
    )
    _fail(
        units
        == {
            "load": LOAD_NM,
            "speed": SPEED_RAD_S,
            "generalized_torque": TORQUE_NM,
            "viscous": VISCOUS_NM_S_PER_RAD,
            "temperature": TEMPERATURE_C,
        },
        "physical_model.units must use the canonical SI-derived unit strings",
    )
    for key in (
        "latent_model_sha256",
        "source_dataset_manifest_sha256",
        "session_split_sha256",
        "repeatability_report_sha256",
        "threshold_contract_sha256",
        "selection_report_sha256",
        "support_envelope_sha256",
    ):
        _sha(model[key], f"physical_model.{key}")
    support = _validate_support(model["support"], where="physical_model.support")
    _fail(
        model["support_envelope_sha256"] == canonical_sha256(model["support"]),
        "physical_model.support_envelope_sha256 does not bind support",
    )

    cross = _mapping(root["cross_engine"], "cross_engine")
    _exact_keys(
        cross,
        {
            "probe_schedule_sha256",
            "threshold_contract_sha256",
            "equivalence_report_sha256",
            "equivalence_passed",
            "same_latent_model_required",
            "parameter_equality_is_acceptance",
        },
        "cross_engine",
    )
    probe_schedule_sha = _sha(
        cross["probe_schedule_sha256"], "cross_engine.probe_schedule_sha256"
    )
    _fail(
        _sha(
            cross["threshold_contract_sha256"],
            "cross_engine.threshold_contract_sha256",
        )
        == model["threshold_contract_sha256"],
        "cross_engine threshold contract drifted from the physical model",
    )
    _sha(cross["equivalence_report_sha256"], "cross_engine.equivalence_report_sha256")
    _fail(cross["equivalence_passed"] is True, "cross-engine equivalence has not passed")
    _fail(cross["same_latent_model_required"] is True, "adapters must share one latent model")
    _fail(
        cross["parameter_equality_is_acceptance"] is False,
        "numeric parameter equality cannot be a cross-engine acceptance rule",
    )

    adapters = _mapping(root["adapters"], "adapters")
    _exact_keys(adapters, set(ENGINES), "adapters")
    normalized_adapters = {
        engine: _validate_adapter(
            adapters[engine],
            engine=engine,
            joint_count=len(names),
            latent_model_sha256=model["latent_model_sha256"],
            threshold_contract_sha256=model["threshold_contract_sha256"],
            probe_schedule_sha256=probe_schedule_sha,
        )
        for engine in ENGINES
    }
    _fail(
        adapters[PHYSX_ENGINE]["fit_report_sha256"]
        != adapters[MUJOCO_ENGINE]["fit_report_sha256"],
        "PhysX and MuJoCo require distinct engine-specific fit reports",
    )

    return {
        "contract_id": root["contract_id"],
        "contract_sha256": claimed_digest,
        "joint_names": names,
        "joint_order_sha256": joint_order_sha,
        "latent_model_sha256": model["latent_model_sha256"],
        "threshold_contract_sha256": model["threshold_contract_sha256"],
        "probe_schedule_sha256": probe_schedule_sha,
        "support": support,
        "adapters": normalized_adapters,
    }


def prepare_runtime_adapter(
    contract: Mapping[str, Any],
    *,
    engine: str,
    requested_support: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one engine adapter after proving the requested envelope is in support."""

    _fail(engine in ENGINES, f"unsupported engine {engine!r}")
    normalized = validate_plant_contract(contract)
    requested = _validate_support(requested_support, where="requested_support")
    calibrated = normalized["support"]
    for key in ("load_abs_Nm", "speed_abs_rad_s", "temperature_C"):
        cal_lo, cal_hi = calibrated[key]
        req_lo, req_hi = requested[key]
        _fail(
            cal_lo <= req_lo <= req_hi <= cal_hi,
            f"requested_support.{key} is outside calibrated support",
        )
    _fail(
        set(requested["pose_ids"]).issubset(calibrated["pose_ids"]),
        "requested_support.pose_ids contains an out-of-support pose",
    )

    adapter = normalized["adapters"][engine]
    runtime = {
        "schema_version": PLANT_CONTRACT_SCHEMA_VERSION,
        "contract_id": normalized["contract_id"],
        "plant_contract_sha256": normalized["contract_sha256"],
        "engine": engine,
        "engine_version": adapter["engine_version"],
        "runtime_target": adapter["runtime_target"],
        "backend": adapter["backend"],
        "parameter_semantics": adapter["parameter_semantics"],
        "joint_names": normalized["joint_names"],
        "joint_order_sha256": normalized["joint_order_sha256"],
        "latent_model_sha256": normalized["latent_model_sha256"],
        "threshold_contract_sha256": normalized["threshold_contract_sha256"],
        "probe_schedule_sha256": normalized["probe_schedule_sha256"],
        "fit_report_sha256": adapter["fit_report_sha256"],
        "runtime_probe_report_sha256": adapter["runtime_probe_report_sha256"],
        "runtime_source_sha256": adapter["runtime_source_sha256"],
        "runtime_instantiation_report_sha256": adapter[
            "runtime_instantiation_report_sha256"
        ],
        "asset_sha256": adapter["asset_sha256"],
        "solver_contract_sha256": adapter["solver_contract_sha256"],
        "physics_step_dt_s": adapter["physics_step_dt_s"],
        "policy_step_dt_s": adapter["policy_step_dt_s"],
        "control_decimation": adapter["control_decimation"],
        "integrator": adapter["integrator"],
        "requested_support": copy.deepcopy(dict(requested_support)),
        "parameters": copy.deepcopy(adapter["parameters"]),
        "hardware_commands_authorized": False,
    }
    if engine == MUJOCO_ENGINE:
        runtime["vendor_mjcf_path"] = adapter["vendor_mjcf_path"]
    runtime["runtime_adapter_sha256"] = canonical_sha256(runtime)
    return runtime
