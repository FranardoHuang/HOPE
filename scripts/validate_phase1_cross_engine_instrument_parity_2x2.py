#!/usr/bin/env python3
"""Validate the frozen 2-engine x 2-instrument parity prerequisite and evidence.

The validator is deliberately unable to close the gate from virtual-only evidence.  A complete
paper requires four independently content-addressed cells on one immutable question order:
Isaac/MuJoCo x physical truth/analytic counterfactual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PREREG_SCHEMA = "hope.cross-engine-instrument-parity-prereg.v1"
CELL_SCHEMA = "hope.cross-engine-instrument-cell.v1"
EVIDENCE_SCHEMA = "hope.cross-engine-instrument-parity-evidence.v1"
INSTRUMENTATION_SCHEMA = "hope.cross-engine-state-instrumentation.v1"
REQUIRED_CELLS = {
    "isaac:physical_truth",
    "isaac:analytic_counterfactual",
    "mujoco:physical_truth",
    "mujoco:analytic_counterfactual",
}
PHYSICAL_CAPABILITY = "physical_paddle_contact_and_post_contact_flight_v1"
ANALYTIC_CAPABILITY = "analytic_counterfactual_contact_and_flight_v1"


class ParityContractError(RuntimeError):
    """Fail-closed contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ParityContractError(f"JSON root must be an object: {path}")
    return value


def require_hex(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ParityContractError(f"{field} is not a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ParityContractError(f"{field} is not hexadecimal") from exc
    return value


def require_local_binding(repo: Path, spec: Mapping[str, Any], *, field: str) -> None:
    rel = spec.get("path")
    expected = require_hex(spec.get("sha256"), field=f"{field}.sha256")
    if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ParityContractError(f"{field}.path must be a safe repo-relative path")
    path = repo / rel
    if not path.is_file() or sha256_file(path) != expected:
        raise ParityContractError(f"{field} local source binding mismatch: {path}")


def finite_values(value: Any, *, size: int, field: str) -> list[float]:
    if not isinstance(value, Mapping) or value.get("shape") is None or not isinstance(
        value.get("values"), list
    ):
        raise ParityContractError(f"{field} is not a numeric array document")
    values = value["values"]
    if len(values) != size:
        raise ParityContractError(f"{field} has {len(values)} values, expected {size}")
    numbers = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ParityContractError(f"{field} contains a non-finite/non-numeric value")
        numbers.append(float(item))
    return numbers


def validate_prereg(config_path: Path, repo: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("schema") != PREREG_SCHEMA or config.get("schema_version") != 1:
        raise ParityContractError("unsupported instrument-parity preregistration schema")
    if config.get("status") != "preregistered_runtime_blocked":
        raise ParityContractError("preregistration must remain runtime-blocked until evidence exists")
    if config.get("auto_start") is not False or config.get("threshold_changes_allowed") is not False:
        raise ParityContractError("2x2 paper must be manual and threshold-frozen")

    forensic = config.get("forensic_input")
    if not isinstance(forensic, Mapping):
        raise ParityContractError("forensic input binding is missing")
    require_local_binding(repo, forensic, field="forensic_input")
    if forensic.get("sha256") != "aff8f4e665d20bb76a56e079735f32b6766388ee05f61c51e93adeb568be45c9":
        raise ParityContractError("preregistration is not bound to the accepted saturation forensic")

    tools = config.get("tools")
    if not isinstance(tools, Mapping):
        raise ParityContractError("tool bindings are missing")
    for name in ("validator", "isaac_evaluator", "isaac_adapter"):
        spec = tools.get(name)
        if not isinstance(spec, Mapping):
            raise ParityContractError(f"tool binding {name} is missing")
        require_local_binding(repo, spec, field=f"tools.{name}")

    target = config.get("target")
    schedule = config.get("schedule")
    if not isinstance(target, Mapping) or not isinstance(schedule, Mapping):
        raise ParityContractError("target/schedule contract is missing")
    if (
        target.get("fresh_lineage") is not True
        or target.get("evaluation_contract_exact") is not True
        or target.get("plant_cell") != "SZ_zero_friction_protocol_exact"
        or target.get("checkpoint_iteration") != 2000
    ):
        raise ParityContractError("2x2 target must remain the selected exact fresh SZ model_2000")
    for field in ("checkpoint_sha256", "training_contract_sha256", "exam_bank_sha256"):
        require_hex(target.get(field), field=f"target.{field}")
    if (
        schedule.get("schedule_k") != 100
        or schedule.get("attempts_per_side") != 50
        or schedule.get("one_question_reset") is not True
        or schedule.get("no_wrap") is not True
        or schedule.get("censored_attempts") != 0
    ):
        raise ParityContractError("immutable q50 schedule contract changed")
    for field in ("file_sha256", "semantic_sha256", "question_id_order_sha256"):
        require_hex(schedule.get(field), field=f"schedule.{field}")

    threshold = config.get("analytic_threshold_contract")
    if not isinstance(threshold, Mapping):
        raise ParityContractError("analytic threshold contract is missing")
    threshold_payload = {key: value for key, value in threshold.items() if key != "sha256"}
    if canonical_sha256(threshold_payload) != threshold.get("sha256"):
        raise ParityContractError("analytic threshold contract SHA mismatch")
    if (
        threshold.get("capture_radius_m") != 0.095
        or threshold.get("min_approach_speed_mps") != 0.3
        or threshold.get("thresholds_changed_from_forensic") is not False
    ):
        raise ParityContractError("forensic analytic thresholds were changed")

    cells = config.get("cells")
    if not isinstance(cells, list):
        raise ParityContractError("required cell list is missing")
    cell_keys = {f"{cell.get('engine')}:{cell.get('instrument')}" for cell in cells if isinstance(cell, Mapping)}
    if cell_keys != REQUIRED_CELLS or len(cells) != 4:
        raise ParityContractError(f"required 2x2 cells are not exact: {sorted(cell_keys)}")
    for cell in cells:
        key = f"{cell['engine']}:{cell['instrument']}"
        expected = PHYSICAL_CAPABILITY if cell["instrument"] == "physical_truth" else ANALYTIC_CAPABILITY
        if (
            cell.get("required") is not True
            or cell.get("required_capability") != expected
            or cell.get("same_schedule_required") is not True
        ):
            raise ParityContractError(f"cell {key} weakened its required capability")

    blockers = config.get("runtime_blockers")
    if not isinstance(blockers, list) or not any(
        isinstance(item, str) and "no racket impulse" in item for item in blockers
    ):
        raise ParityContractError("Isaac Phase-B physical-contact blocker must remain explicit")
    return {
        "status": "valid_preregistered_runtime_blocked",
        "preregistration_sha256": sha256_file(config_path),
        "required_cells": sorted(REQUIRED_CELLS),
        "instrument_parity_gate_closed": False,
        "runtime_blockers": blockers,
    }


def require_external_artifact(root: Path, spec: Mapping[str, Any], *, field: str) -> Path:
    rel = spec.get("path")
    expected = require_hex(spec.get("sha256"), field=f"{field}.sha256")
    if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ParityContractError(f"{field}.path must be safe and relative to artifact root")
    path = root / rel
    if not path.is_file() or sha256_file(path) != expected:
        raise ParityContractError(f"{field} artifact missing or SHA-mismatched: {path}")
    return path


def validate_state(state: Any, *, field: str) -> None:
    if not isinstance(state, Mapping) or state.get("schema") != INSTRUMENTATION_SCHEMA:
        raise ParityContractError(f"{field} has no cross-engine instrumentation schema")
    base = state.get("base", {})
    racket = state.get("racket", {})
    incoming = state.get("incoming_ball", {})
    finite_values(base.get("root_state"), size=13, field=f"{field}.base.root_state")
    for key in (
        "position_env_m",
        "linear_velocity_world_mps",
        "face_normal_signed_pre_orient_world",
        "face_normal_raw_plus_y_world",
        "analytic_face_normal_oriented_world",
    ):
        finite_values(racket.get(key), size=3, field=f"{field}.racket.{key}")
    finite_values(
        incoming.get("linear_velocity_world_mps"),
        size=3,
        field=f"{field}.incoming_ball.linear_velocity_world_mps",
    )
    finite_values(
        incoming.get("spin_world_radps"),
        size=3,
        field=f"{field}.incoming_ball.spin_world_radps",
    )


def validate_cell(
    *,
    prereg: Mapping[str, Any],
    engine: str,
    instrument: str,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    key = f"{engine}:{instrument}"
    schedule = prereg["schedule"]
    target = prereg["target"]
    if (
        document.get("schema") != CELL_SCHEMA
        or document.get("status") != "complete"
        or document.get("engine") != engine
        or document.get("instrument") != instrument
        or document.get("schedule_file_sha256") != schedule["file_sha256"]
        or document.get("schedule_semantic_sha256") != schedule["semantic_sha256"]
        or document.get("question_id_order_sha256") != schedule["question_id_order_sha256"]
        or document.get("checkpoint_sha256") != target["checkpoint_sha256"]
        or document.get("training_contract_sha256") != target["training_contract_sha256"]
        or document.get("exam_bank_sha256") != target["exam_bank_sha256"]
        or document.get("evaluation_contract_exact") is not True
        or document.get("fresh_lineage") is not True
        or document.get("censored_attempts") != 0
    ):
        raise ParityContractError(f"cell {key} header/provenance contract mismatch")
    ready = document.get("numeric_ready_state")
    if not isinstance(ready, Mapping) or ready.get("schema") != INSTRUMENTATION_SCHEMA:
        raise ParityContractError(f"cell {key} lacks a numeric ready state")
    require_hex(ready.get("sha256"), field=f"cell.{key}.numeric_ready_state.sha256")
    finite_values(ready.get("root_state"), size=13, field=f"cell.{key}.ready.root_state")

    capability = document.get("instrument_capability")
    analytic_only = document.get("analytic_only")
    expected_capability = PHYSICAL_CAPABILITY if instrument == "physical_truth" else ANALYTIC_CAPABILITY
    if capability != expected_capability:
        raise ParityContractError(f"cell {key} capability {capability!r} is not {expected_capability!r}")
    if instrument == "physical_truth" and analytic_only is not False:
        raise ParityContractError(f"cell {key} is virtual-only and cannot fill a physical truth cell")

    attempts = document.get("attempts")
    question_order = document.get("question_id_order")
    if not isinstance(attempts, list) or len(attempts) != 100:
        raise ParityContractError(f"cell {key} must contain 100 attempts")
    if not isinstance(question_order, list) or len(question_order) != 100 or len(set(question_order)) != 100:
        raise ParityContractError(f"cell {key} question order is invalid")
    if canonical_sha256(question_order) != schedule["question_id_order_sha256"]:
        raise ParityContractError(f"cell {key} question order bytes changed")
    if [row.get("question_id") for row in attempts if isinstance(row, Mapping)] != question_order:
        raise ParityContractError(f"cell {key} attempts are reordered or malformed")
    for index, row in enumerate(attempts):
        if not isinstance(row, Mapping) or row.get("schedule_index") != index or row.get("censored") is not False:
            raise ParityContractError(f"cell {key} attempt {index} is censored/reordered")
        validate_state(row.get("instrumentation"), field=f"cell.{key}.attempt[{index}]")
        outcome = row.get("outcome")
        if not isinstance(outcome, Mapping):
            raise ParityContractError(f"cell {key} attempt {index} has no outcome")
        if instrument == "physical_truth":
            if outcome.get("available") is not True or outcome.get("capability") != PHYSICAL_CAPABILITY:
                raise ParityContractError(f"cell {key} attempt {index} lacks physical truth")
            if any(not isinstance(outcome.get(name), bool) for name in ("contacted", "net_clear", "landed_ok", "returned")):
                raise ParityContractError(f"cell {key} attempt {index} physical outcome is incomplete")
        else:
            if outcome.get("available") is not True or outcome.get("capability") != ANALYTIC_CAPABILITY:
                raise ParityContractError(f"cell {key} attempt {index} lacks analytic counterfactual")
            if any(not isinstance(outcome.get(name), bool) for name in ("capture_gate", "net_clear", "on_opponent", "returned")):
                raise ParityContractError(f"cell {key} attempt {index} analytic outcome is incomplete")
    return {
        "engine": engine,
        "instrument": instrument,
        "capability": capability,
        "numeric_ready_state_sha256": ready["sha256"],
        "question_id_order_sha256": schedule["question_id_order_sha256"],
        "n_attempts": 100,
    }


def validate_evidence(
    config_path: Path, evidence_path: Path, artifact_root: Path, repo: Path
) -> dict[str, Any]:
    prereg_status = validate_prereg(config_path, repo)
    prereg = load_json(config_path)
    evidence = load_json(evidence_path)
    if evidence.get("schema") != EVIDENCE_SCHEMA or evidence.get("schema_version") != 1:
        raise ParityContractError("unsupported 2x2 evidence schema")
    if evidence.get("preregistration_sha256") != prereg_status["preregistration_sha256"]:
        raise ParityContractError("evidence is not bound to this preregistration")
    cells = evidence.get("cells")
    if not isinstance(cells, list):
        raise ParityContractError("evidence has no cell list")
    specs: dict[str, Mapping[str, Any]] = {}
    for spec in cells:
        if not isinstance(spec, Mapping):
            raise ParityContractError("cell artifact spec is malformed")
        key = f"{spec.get('engine')}:{spec.get('instrument')}"
        if key in specs:
            raise ParityContractError(f"duplicate evidence cell {key}")
        specs[key] = spec
    if set(specs) != REQUIRED_CELLS:
        missing = sorted(REQUIRED_CELLS.difference(specs))
        extra = sorted(set(specs).difference(REQUIRED_CELLS))
        raise ParityContractError(f"2x2 evidence incomplete: missing={missing}, extra={extra}")

    accepted = []
    question_orders = set()
    ready_by_engine: dict[str, set[str]] = {"isaac": set(), "mujoco": set()}
    for key in sorted(REQUIRED_CELLS):
        spec = specs[key]
        path = require_external_artifact(
            artifact_root, spec.get("artifact", {}), field=f"cells.{key}.artifact"
        )
        engine, instrument = key.split(":", 1)
        cell = validate_cell(
            prereg=prereg, engine=engine, instrument=instrument, document=load_json(path)
        )
        cell["artifact_sha256"] = sha256_file(path)
        accepted.append(cell)
        question_orders.add(cell["question_id_order_sha256"])
        ready_by_engine[engine].add(cell["numeric_ready_state_sha256"])
    if len(question_orders) != 1:
        raise ParityContractError("accepted cells do not share one question order")
    if any(len(values) != 1 for values in ready_by_engine.values()):
        raise ParityContractError("physical/analytic cells within one engine used different ready states")
    return {
        "status": "complete_four_cell_instrument_parity_evidence",
        "preregistration_sha256": prereg_status["preregistration_sha256"],
        "evidence_sha256": sha256_file(evidence_path),
        "accepted_cells": accepted,
        "instrument_parity_gate_closed": True,
        "scope": "instrument parity prerequisite only; not a training, calibrated-plant, continuity, deployment, or real-robot gate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.evidence is None:
        result = validate_prereg(args.config.resolve(), repo)
    else:
        if args.artifact_root is None:
            raise ParityContractError("--artifact-root is required with --evidence")
        result = validate_evidence(
            args.config.resolve(), args.evidence.resolve(), args.artifact_root.resolve(), repo
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
