#!/usr/bin/env python3
"""Materialize one measured-racket, fixed-question N=1 diagnostic bundle.

The output is deliberately diagnostic-only.  ``prepare`` projects one action from an
exact schema-v3 ActionBall manifest, preserves its legal sampling envelope, and delegates
contact/prototype construction to the existing N1 bundle materializer.  The immutable
seed row, rather than an invalid zero-width profile, freezes the actual question.  A
runnable bundle requires a fresh dynamic-ready artifact and nominal-hold receipt that name
this exact action and motion SHA; receipts from r9/bh_loop_c or any other action are
rejected.  Offline producers then create the shared question receipt,
five same-question target receipts and one canonical tape.  ``finalize`` binds that tape
back to the exact prepared manifest/solver/physics.  This ordering avoids a circular
dependency and makes all five arms share one physical question and one core.

The environment must install tape rows at reset and must never run an inverse solver in
its hot path.

The five code-owned recipes use the same existing fixed-194 actor / 318-D critic canary
ABI.  Their only target-content difference is the position/velocity/face validity mask::

    current_lm                  111
    analytic_full              111
    analytic_no_velocity       101
    teacher_pos_face_no_velocity 101
    outcome_dense_only         000

The validity mask is a fixed receipt/config constant within each independent arm, so it is
not added to tonight's observation.  Invalid target columns are zero-filled.  This narrow
canary must not be presented as the final varying-ball/N73 ABI.  The recipe remains explicit
even when two arms share a mask: the producer differs.  The tape receipt therefore binds
both its exact bytes and a per-column producer SHA map.

This script accepts a mechanically UNKNOWN action only with the explicit
``--allow-mechanical-unknown-diagnostic`` flag.  Observed position/velocity hard failures
are always rejected.  UNKNOWN never becomes training, promotion, export, deployment or
hardware authority.

All outputs use content-addressed names and exclusive creation.  A partially written
namespace is retained for forensics and is never reused.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPT_DIR.parents[2]
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RECIPES: Mapping[str, tuple[bool, bool, bool]] = {
    "current_lm": (True, True, True),
    "analytic_full": (True, True, True),
    "analytic_no_velocity": (True, False, True),
    "teacher_pos_face_no_velocity": (True, False, True),
    "outcome_dense_only": (False, False, False),
}
TARGET_ORDER = ("position", "velocity", "face")
PHYSICAL_BALL_SEMANTICS = "analytic_virtual_ball_authoritative_physx_disabled"
ACTION_ID = "take_061_unit04_bh"
MEASURED_UID = "Take_061_unit04_BH"
ACTION_FACTS = {
    "action_uid": 5527597793770800,
    "motion_path": (
        "assets/motions/chingmu73_measured_v4_20260803/"
        "hope_Take_061_unit04_BH.npz"
    ),
    "motion_sha256": "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e",
    "reference_t_hit_s": 0.96,
    "reference_t_cycle_s": 1.12,
    "reference_racket_site_speed_mps": 1.8901338577270508,
    "strike_phase": 0.8571,
    "family": "backhand",
}
TAPE_RECEIPT_KIND = "measured_action_ball_immutable_tape_receipt_v1"
TAPE_BUILD_REPORT_KIND = "measured_action_ball_n1_fixed_tape_build_report_v1"
PREPARED_KIND = "measured_action_ball_n1_prepared_core_v1"
BUNDLE_KIND = "measured_action_ball_n1_diagnostic_bundle_v1"
FIXED_TAPE_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_fixed_question_tape.py"
)
TAPE_PRODUCER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_action_ball_n1_fixed_tape_variants.py"
)
TASK_PROFILE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV2N1Diagnostic.yaml"
)
RACKET_MESH_SHA256 = "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd"
FIXED_LANDING_DEPTH_X_M = 2.30
FIXED_TEACHER_RATE_MAX = 1.01
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


class BundleError(ValueError):
    """The requested bundle is incomplete, inconsistent or unauthorized."""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BundleError("cannot import %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _canonical_payload_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _materialize_live_profile_pins(
    *,
    root: Path,
    template_path: Path,
    template_expected_sha: str,
    destination: Path,
    output_relative: str,
) -> tuple[Path, str]:
    """Repin the unchanged solver recipe to the exact live implementation bytes."""

    template, _relative, _actual = _exact_path(
        root,
        template_path,
        template_expected_sha,
        label="profile-pins template",
    )
    document = deepcopy(_strict_json(template, label="profile-pins template"))
    geometry_path = root / (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/racket_contact_geometry.py"
    )
    geometry = _load_module("_measured_n1_live_geometry", geometry_path)
    geometry_contract = {
        "payload": geometry.GEOMETRY_SOURCE_PAYLOAD,
        "sha256": geometry.GEOMETRY_SOURCE_SHA256,
    }
    source_names = (
        "hope_commands.py",
        "continuous_questions.py",
        "racket_contact_geometry.py",
        "stroke_adapt_torch.py",
        "virtual_ball.py",
        "counter_rally.py",
        "counter_rally_torch.py",
    )
    mdp_dir = geometry_path.parent
    source_sha = {name: _sha256(mdp_dir / name) for name in source_names}
    solver = document.get("solver_payload")
    physics = document.get("physics_payload")
    if type(solver) is not dict or type(physics) is not dict:
        raise BundleError("profile-pins template lacks solver/physics payload")
    if (
        _canonical_payload_sha(solver)
        != _require_sha(
            document.get("solver_profile_sha256"),
            label="template solver_profile_sha256",
        )
        or _canonical_payload_sha(physics)
        != _require_sha(
            document.get("physics_profile_sha256"),
            label="template physics_profile_sha256",
        )
    ):
        raise BundleError("profile-pins template payload seal differs")
    document["contact_geometry"] = geometry_contract
    document["solver_implementation_source_sha256"] = source_sha
    solver["contact_geometry"] = geometry_contract
    solver["implementation_source_sha256"] = source_sha
    document["solver_profile_sha256"] = _canonical_payload_sha(solver)
    raw = _canonical_bytes(document)
    digest = hashlib.sha256(raw).hexdigest()
    name = "action_ball_profile_pins.live.v1.%s.json" % digest[:12]
    _exclusive_write(destination / name, document)
    return destination / name, digest


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    seen: set[str] = set()

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise BundleError("%s contains duplicate key %r" % (label, key))
            result[key] = value
            seen.add(key)
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                BundleError("%s contains non-finite %s" % (label, value))
            ),
        )
    except BundleError:
        raise
    except Exception as exc:
        raise BundleError("cannot read %s as strict JSON: %s" % (label, exc)) from exc
    if type(value) is not dict:
        raise BundleError("%s root must be an object" % label)
    return value


def _require_sha(value: object, *, label: str) -> str:
    if type(value) is not str or SHA_RE.fullmatch(value) is None:
        raise BundleError("%s must be 64 lowercase hexadecimal characters" % label)
    return value


def _exact_path(
    root: Path, value: str | Path, expected_sha: str, *, label: str
) -> tuple[Path, str, str]:
    expected = _require_sha(expected_sha, label=label + " SHA")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path = path.resolve(strict=True)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise BundleError("%s must remain inside repo root" % label) from exc
    actual = _sha256(path)
    if actual != expected:
        raise BundleError(
            "%s SHA differs: expected=%s actual=%s" % (label, expected, actual)
        )
    return path, relative, actual


def _finite(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise BundleError("%s must be a finite number" % label)
    return float(value)


# [已删除 2026-08-06 过期结构清理] _freeze_pair(4 行):零调用点。现役冻结走下面
# _freeze_ball_profile 的后缀规则(std_lower_/std_upper_/_neg_/_pos_),不是逐三元组搬 center。
def _freeze_ball_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(profile))
    for key in tuple(row):
        if "std_lower_" in key or "std_upper_" in key:
            value = row[key]
            row[key] = [0.0 for _ in value] if isinstance(value, list) else 0.0
        elif key.endswith("_neg_initial_deg") or key.endswith("_neg_max_deg"):
            row[key] = 0.0
        elif key.endswith("_pos_initial_deg") or key.endswith("_pos_max_deg"):
            row[key] = 0.0
    # Bounds stay strict and schema-valid; zero proposal widths make every
    # sampled value equal its centre without claiming a degenerate envelope.
    return row


def _freeze_landing(landing: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(landing))
    for key in tuple(row):
        if "std_lower_" in key or "std_upper_" in key:
            row[key] = [0.0, 0.0]
    if not (
        float(row["min_w_xy_m"][0])
        <= FIXED_LANDING_DEPTH_X_M
        <= float(row["max_w_xy_m"][0])
    ):
        raise BundleError("fixed landing depth is outside source support")
    row["center_w_xy_m"][0] = FIXED_LANDING_DEPTH_X_M
    return row


def _exclusive_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
    except FileExistsError as exc:
        raise BundleError("no-clobber output already exists: %s" % path) from exc


def _validate_tape_receipt(
    root: Path,
    receipt_path: Path,
    receipt_sha: str,
    *,
    action_id: str,
    action_uid: int,
    motion_sha: str,
    recipe: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    path, relative, actual = _exact_path(
        root, receipt_path, receipt_sha, label="immutable tape receipt"
    )
    row = _strict_json(path, label="immutable tape receipt")
    expected_keys = {
        "schema_version",
        "kind",
        "action_id",
        "target_recipe",
        "target_validity",
        "artifact",
        "row_count",
        "per_column_producer_sha256",
        "physical_ball_semantics",
        "reset_inverse_solve",
        "diagnostic_unauthorized",
    }
    if set(row) != expected_keys:
        raise BundleError("immutable tape receipt keys differ")
    if row["schema_version"] != 1 or row["kind"] != TAPE_RECEIPT_KIND:
        raise BundleError("immutable tape receipt schema/kind differs")
    if row["action_id"] != action_id or row["target_recipe"] != recipe:
        raise BundleError("immutable tape action/recipe differs")
    validity = row["target_validity"]
    if (
        type(validity) is not dict
        or set(validity) != {"order", "mask"}
        or validity["order"] != list(TARGET_ORDER)
        or validity["mask"] != list(RECIPES[recipe])
    ):
        raise BundleError("immutable tape target validity differs from recipe")
    if row["row_count"] != 1:
        raise BundleError("immutable tape row_count must be exactly one")
    producers = row["per_column_producer_sha256"]
    required_producers = {
        "incoming_ball",
        "teacher_contact",
        "desired_contact",
        "landing_spin_task",
    }
    if type(producers) is not dict or set(producers) != required_producers:
        raise BundleError("immutable tape per-column producer set differs")
    for name, digest in producers.items():
        _require_sha(digest, label="producer %s" % name)
    if (
        row["physical_ball_semantics"] != PHYSICAL_BALL_SEMANTICS
        or row["reset_inverse_solve"] is not False
        or row["diagnostic_unauthorized"] is not True
    ):
        raise BundleError("immutable tape authority/physics semantics differ")
    artifact = row["artifact"]
    if type(artifact) is not dict or set(artifact) != {"path", "sha256"}:
        raise BundleError("immutable tape artifact pin is malformed")
    artifact_path, artifact_relative, artifact_sha = _exact_path(
        root, artifact["path"], artifact["sha256"], label="immutable tape"
    )
    tape_module = _load_module(
        "_measured_action_ball_fixed_tape_validation",
        root / FIXED_TAPE_SOURCE,
    )
    try:
        tape = tape_module.load_immutable_n1_tape(
            artifact_path,
            expected_file_sha256=artifact_sha,
        )
    except Exception as exc:
        raise BundleError("immutable tape failed canonical runtime validation: %s" % exc) from exc
    if tuple(tape_module.TARGET_RECIPES) != tuple(RECIPES):
        raise BundleError("immutable tape runtime recipe order differs")
    if {
        name: tuple(mask)
        for name, mask in tape_module.TARGET_VALIDITY_BY_RECIPE.items()
    } != dict(RECIPES):
        raise BundleError("immutable tape runtime validity authority differs")
    source_receipt = tape.source_receipt
    if (
        source_receipt.action_uid != action_uid
        or source_receipt.motion_sha256 != motion_sha
        or source_receipt.mobility_mode != "no_move"
    ):
        raise BundleError("immutable tape action/motion/mobility identity differs")
    all_lineage = {
        name: tape.target_lineage(name)
        for name in tape_module.TARGET_RECIPES
    }
    base_question_shas = {
        lineage["base_question_sha256"] for lineage in all_lineage.values()
    }
    if len(base_question_shas) != 1:
        raise BundleError("immutable tape recipes do not share one base question")
    lineage = all_lineage[recipe]
    if (
        lineage["target_validity_mask"] != list(RECIPES[recipe])
        or lineage["target_producer_sha256"] != producers["desired_contact"]
    ):
        raise BundleError("immutable tape selected target lineage differs from receipt")
    return {"path": relative, "sha256": actual}, {
        "artifact": {"path": artifact_relative, "sha256": artifact_sha},
        "row_count": 1,
        "canonical_sha256": tape.canonical_sha256,
        "base_question_sha256": tape.question_sha256,
        "target_validity": validity,
        "per_column_producer_sha256": producers,
        "selected_target_lineage": lineage,
        "all_target_lineage": all_lineage,
        "source_identity": {
            "action_uid": source_receipt.action_uid,
            "action_slot": source_receipt.action_slot,
            "profile_sha256": source_receipt.profile_sha256,
            "motion_sha256": source_receipt.motion_sha256,
            "manifest_sha256": source_receipt.manifest_sha256,
            "sampler_sha256": source_receipt.sampler_sha256,
            "physics_sha256": source_receipt.physics_sha256,
            "solver_sha256": source_receipt.solver_sha256,
            "mobility_mode": source_receipt.mobility_mode,
        },
    }


def _validate_tape_build_report(
    root: Path,
    report_path: Path,
    report_sha: str,
    *,
    action_uid: int,
    motion_sha: str,
    recipe: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    path, relative, actual = _exact_path(
        root, report_path, report_sha, label="immutable tape build report"
    )
    report = _strict_json(path, label="immutable tape build report")
    expected_keys = {
        "schema_version", "kind", "diagnostic_unauthorized", "sampler_seed",
        "prepared_core", "source_identity", "base_question", "target_receipts",
        "producer_contracts", "tape", "reset_semantics",
        "diagnostic_admissibility", "claims",
    }
    if set(report) != expected_keys:
        raise BundleError("immutable tape build report keys differ")
    if (
        report["schema_version"] != 1
        or report["kind"] != TAPE_BUILD_REPORT_KIND
        or report["diagnostic_unauthorized"] is not True
        or report["sampler_seed"] not in (0, 1, 2)
    ):
        raise BundleError("immutable tape build report schema/authority differs")
    reset = report["reset_semantics"]
    if reset != {
        "selection": "constant_row_zero",
        "online_sampler_calls": 0,
        "online_lm_calls": 0,
        "physical_rng_draws": 0,
    }:
        raise BundleError("immutable tape reset semantics differ")
    claims = report["claims"]
    if (
        type(claims) is not dict
        or claims.get("diagnostic_unauthorized") is not True
        or any(
            claims.get(key) is not True
            for key in (
                "formal_evidence_prohibited", "promotion_prohibited",
                "export_prohibited", "deployment_prohibited", "hardware_prohibited",
            )
        )
    ):
        raise BundleError("immutable tape build report claims differ")
    source_identity = report["source_identity"]
    required_identity = {
        "action_uid", "action_slot", "profile_sha256", "motion_sha256",
        "manifest_sha256", "sampler_sha256", "physics_sha256", "solver_sha256",
        "mobility_mode", "counter_rally_objective_profile_sha256",
    }
    if (
        type(source_identity) is not dict
        or set(source_identity) != required_identity
        or source_identity["action_uid"] != action_uid
        or source_identity["action_slot"] != 0
        or source_identity["motion_sha256"] != motion_sha
        or source_identity["mobility_mode"] != "no_move"
    ):
        raise BundleError("immutable tape source identity differs")
    for key in required_identity - {"action_uid", "action_slot", "mobility_mode"}:
        _require_sha(source_identity[key], label="source identity %s" % key)
    artifact = report["tape"].get("artifact")
    if type(artifact) is not dict or set(artifact) != {"path", "sha256"}:
        raise BundleError("immutable tape artifact pin is malformed")
    artifact_path, artifact_relative, artifact_sha = _exact_path(
        root, artifact["path"], artifact["sha256"], label="immutable tape"
    )
    tape_module = _load_module(
        "_measured_action_ball_fixed_tape_build_validation",
        root / FIXED_TAPE_SOURCE,
    )
    try:
        tape = tape_module.load_immutable_n1_tape(
            artifact_path, expected_file_sha256=artifact_sha
        )
    except Exception as exc:
        raise BundleError(
            "immutable tape failed canonical runtime validation: %s" % exc
        ) from exc
    source = tape.source_receipt
    counter = source.counter_rally_task
    actual_identity = {
        "action_uid": source.action_uid,
        "action_slot": source.action_slot,
        "profile_sha256": source.profile_sha256,
        "motion_sha256": source.motion_sha256,
        "manifest_sha256": source.manifest_sha256,
        "sampler_sha256": source.sampler_sha256,
        "physics_sha256": source.physics_sha256,
        "solver_sha256": source.solver_sha256,
        "mobility_mode": source.mobility_mode,
        "counter_rally_objective_profile_sha256": (
            None if counter is None else counter.objective_profile_sha256
        ),
    }
    if actual_identity != source_identity:
        raise BundleError("immutable tape receipt/source identity differs")
    tape_row = report["tape"]
    if (
        tape_row.get("row_count") != 1
        or tape_row.get("canonical_sha256") != tape.canonical_sha256
        or tape_row.get("base_question_sha256") != tape.question_sha256
        or report["base_question"].get("base_question_sha256")
        != tape.question_sha256
    ):
        raise BundleError("immutable tape report summary differs")
    targets = report["target_receipts"]
    if type(targets) is not dict or set(targets) != set(RECIPES):
        raise BundleError("immutable tape target receipt set differs")
    all_lineage = {name: tape.target_lineage(name) for name in RECIPES}
    contracts = report["producer_contracts"]
    if (
        type(contracts) is not dict
        or set(contracts) != {"common", "desired_contact"}
        or type(contracts["desired_contact"]) is not dict
        or set(contracts["desired_contact"]) != set(RECIPES)
    ):
        raise BundleError("immutable tape producer contract set differs")
    producer_source_sha = _sha256(root / TAPE_PRODUCER_SOURCE)
    for name, expected_mask in RECIPES.items():
        target = targets[name]
        contract = contracts["desired_contact"][name]
        if (
            type(target) is not dict
            or target.get("validity_mask") != list(expected_mask)
            or target.get("base_question_sha256") != tape.question_sha256
            or target.get("target_producer_sha256")
            != all_lineage[name]["target_producer_sha256"]
            or target.get("target_column_sha256")
            != all_lineage[name]["target_column_sha256"]
            or contract.get("sha256")
            != all_lineage[name]["target_producer_sha256"]
            or contract.get("payload", {})
            .get("implementation_source_sha256", {})
            .get("fixed_tape_variant_producer")
            != producer_source_sha
        ):
            raise BundleError("immutable tape target lineage differs for %s" % name)
        pin = target.get("artifact")
        if type(pin) is not dict or set(pin) != {"path", "sha256"}:
            raise BundleError("target artifact pin differs for %s" % name)
        _exact_path(root, pin["path"], pin["sha256"], label=name + " target receipt")
    selected = all_lineage[recipe]
    return {"path": relative, "sha256": actual}, {
        "artifact": {"path": artifact_relative, "sha256": artifact_sha},
        "row_count": 1,
        "canonical_sha256": tape.canonical_sha256,
        "base_question_sha256": tape.question_sha256,
        "target_validity": {"order": list(TARGET_ORDER), "mask": list(RECIPES[recipe])},
        "selected_target_lineage": selected,
        "all_target_lineage": all_lineage,
        "source_identity": source_identity,
        "sampler_seed": report["sampler_seed"],
    }


def _mechanical_selection(
    root: Path,
    report_path: Path,
    report_sha: str,
    *,
    motion_sha: str,
    action_uid: str,
    allow_unknown: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    path, relative, actual = _exact_path(
        root, report_path, report_sha, label="mechanical audit"
    )
    report = _strict_json(path, label="mechanical audit")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "measured_racket_mechanical_admission_audit_v1"
        or report.get("diagnostic_unauthorized") is not True
    ):
        raise BundleError("mechanical audit schema/authority differs")
    matches = [row for row in report.get("actions", []) if row.get("uid") == action_uid]
    if len(matches) != 1:
        raise BundleError("mechanical audit must contain exactly one selected action row")
    row = matches[0]
    if row.get("sha256") != motion_sha:
        raise BundleError("mechanical audit motion SHA differs")
    if row.get("kinematic_limit_verdict") != "PASS":
        raise BundleError("selected action has an observed position/velocity hard failure")
    verdict = row.get("mechanical_verdict")
    if verdict == "FAIL" or verdict not in ("PASS", "UNKNOWN"):
        raise BundleError("selected action mechanical verdict is not diagnostic-eligible")
    if verdict == "UNKNOWN" and not allow_unknown:
        raise BundleError(
            "mechanical verdict is UNKNOWN; rerun only with "
            "--allow-mechanical-unknown-diagnostic"
        )
    return {"path": relative, "sha256": actual}, {
        "uid": action_uid,
        "motion_sha256": motion_sha,
        "kinematic_limit_verdict": "PASS",
        "mechanical_verdict": verdict,
        "mechanical_admitted": row.get("mechanical_admitted") is True,
        "diagnostic_unauthorized": True,
        "unknown_explicitly_accepted_for_sim_diagnostic": verdict == "UNKNOWN",
    }


def _validate_racket_alignment(
    root: Path,
    report_path: Path,
    report_sha: str,
    *,
    motion_sha: str,
    action_uid: str,
    frame_count: int,
    strike_frame: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    path, relative, actual = _exact_path(
        root, report_path, report_sha, label="independent racket FK alignment audit"
    )
    report = _strict_json(path, label="independent racket FK alignment audit")
    if (
        report.get("schema_version") != 3
        or report.get("kind") != "materialized_measured_racket_fk_audit_v3"
        or report.get("uid") != action_uid
        or report.get("motion_sha256") != motion_sha
        or report.get("frames") != frame_count
        or report.get("finite") is not True
        or report.get("admitted") is not True
    ):
        raise BundleError("independent racket FK alignment audit identity/admission differs")
    gates = report.get("gates")
    if (
        type(gates) is not dict
        or set(gates) != RACKET_ALIGNMENT_GATES
        or any(value is not True for value in gates.values())
    ):
        raise BundleError("independent racket FK alignment audit has a failed gate")
    hit = report.get("hit")
    if type(hit) is not dict or hit.get("frame") != strike_frame:
        raise BundleError("independent racket FK alignment hit frame differs")
    axis = report.get("robot_butt_to_blade_axis_local")
    expected_axis = [math.sqrt(0.5), 0.0, math.sqrt(0.5)]
    if (
        type(axis) is not list
        or len(axis) != 3
        or any(abs(float(left) - right) > 1.0e-12 for left, right in zip(axis, expected_axis))
        or report.get("robot_rigid_visual_mesh_sha256") != RACKET_MESH_SHA256
    ):
        raise BundleError("independent racket FK audit long-axis/mesh authority differs")
    authority = report.get("authorization")
    if (
        type(authority) is not dict
        or authority.get("diagnostic_unauthorized") is not True
        or authority.get("training") is not False
        or authority.get("promotion") is not False
        or authority.get("deployment") is not False
    ):
        raise BundleError("independent racket FK audit authority differs")
    return {"path": relative, "sha256": actual}, {
        "uid": action_uid,
        "motion_sha256": motion_sha,
        "frames": frame_count,
        "hit": hit,
        "position_error_m": report.get("position_error_m"),
        "face_error_deg": report.get("face_error_deg"),
        "long_axis_error_deg": report.get("long_axis_error_deg"),
        "so3_error_deg": report.get("so3_error_deg"),
        "all_11_gates_pass": True,
        "diagnostic_unauthorized": True,
    }


def _validate_measured_provenance(
    root: Path,
    *,
    bank_receipt_path: Path,
    bank_receipt_sha: str,
    build_report_path: Path,
    build_report_sha: str,
    source_manifest_sha: str,
    action_id: str,
    measured_uid: str,
    motion_sha: str,
    frame_count: int,
    strike_frame: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    bank_path, bank_relative, bank_actual = _exact_path(
        root, bank_receipt_path, bank_receipt_sha, label="measured bank receipt"
    )
    bank = _strict_json(bank_path, label="measured bank receipt")
    if (
        bank.get("schema_version") != 1
        or bank.get("kind") != "chingmu73_measured_racket_schema_v4_repo_import"
    ):
        raise BundleError("measured bank receipt schema/kind differs")
    # The v4 receipt expresses its boundary under authorization rather than a top-level flag.
    authorization = bank.get("authorization")
    if (
        type(authorization) is not dict
        or authorization.get("diagnostic_unauthorized") is not True
        or authorization.get("training") is not False
        or authorization.get("promotion") is not False
        or authorization.get("deployment") is not False
        or authorization.get("mechanical_admission") is not False
    ):
        raise BundleError("measured bank receipt authority differs")
    bank_rows = [row for row in bank.get("actions", []) if row.get("uid") == measured_uid]
    if len(bank_rows) != 1:
        raise BundleError("measured bank receipt must contain one selected action")
    bank_row = bank_rows[0]
    if (
        bank_row.get("sha256") != motion_sha
        or bank_row.get("frames") != frame_count
        or bank_row.get("hit_frame_50") != strike_frame
    ):
        raise BundleError("measured bank selected row differs from motion/strike")

    report_path, report_relative, report_actual = _exact_path(
        root, build_report_path, build_report_sha, label="measured manifest build report"
    )
    report = _strict_json(report_path, label="measured manifest build report")
    if (
        report.get("file_sha256") != source_manifest_sha
        or report.get("measured_bank_receipt_sha256") != bank_actual
        or report.get("n_actions") != 73
        or report.get("racket_authority") != "measured_channel"
    ):
        raise BundleError("measured build report does not bind source manifest/bank")
    action_rows = [row for row in report.get("per_action", []) if row.get("action_id") == action_id]
    if len(action_rows) != 1:
        raise BundleError("measured build report must contain one selected action")
    report_row = action_rows[0]
    if (
        report_row.get("uid") != measured_uid
        or report_row.get("racket_authority") != "measured_channel"
        or report_row.get("measured_racket_schema_version") != 4
        or report_row.get("t_hit_s") != ACTION_FACTS["reference_t_hit_s"]
        or report_row.get("t_cycle_s") != ACTION_FACTS["reference_t_cycle_s"]
    ):
        raise BundleError("measured build-report action authority differs")
    return (
        {"path": bank_relative, "sha256": bank_actual},
        {"path": report_relative, "sha256": report_actual},
        {
            "bank_action_row": bank_row,
            "build_report_action": {
                key: report_row[key]
                for key in (
                    "uid",
                    "action_id",
                    "racket_authority",
                    "measured_racket_schema_version",
                    "measured_racket_source_sha256",
                    "measured_racket_retarget_receipt_sha256",
                    "measured_racket_joint_order_contract_sha256",
                )
            },
        },
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve(strict=True)
    if args.action_id != ACTION_ID:
        raise BundleError(
            "tonight's measured N1 diagnostic action is code-owned as %s" % ACTION_ID
        )
    source_path, source_relative, source_sha = _exact_path(
        root, args.source_manifest, args.expected_source_manifest_sha256,
        label="measured source manifest",
    )
    source = _strict_json(source_path, label="measured source manifest")
    if source.get("schema_version") != 3:
        raise BundleError("measured source manifest must be schema 3")
    action_rows = [row for row in source.get("actions", []) if row.get("action_id") == args.action_id]
    if len(action_rows) != 1:
        raise BundleError("source manifest must contain exactly one selected action")
    source_action = deepcopy(action_rows[0])
    for key, expected in ACTION_FACTS.items():
        if source_action.get(key) != expected:
            raise BundleError("selected action %s differs from code-owned fact" % key)
    action_uid = str(args.action_uid or "")
    if not action_uid:
        motion_stem = Path(source_action["motion_path"]).stem
        action_uid = motion_stem[5:] if motion_stem.startswith("hope_") else motion_stem
    if action_uid != MEASURED_UID:
        raise BundleError("measured action UID must be %s" % MEASURED_UID)
    motion_path, motion_relative, motion_sha = _exact_path(
        root,
        source_action["motion_path"],
        source_action["motion_sha256"],
        label="measured motion",
    )
    import numpy as np
    with np.load(str(motion_path), allow_pickle=False) as loaded:
        required = {
            "joint_pos",
            "measured_racket_site_pos_w",
            "measured_racket_normal_w",
            "measured_racket_long_axis_w",
            "measured_racket_schema_version",
            "measured_racket_retarget_admitted",
            "measured_racket_uid",
        }
        missing = required - set(loaded.files)
        if missing:
            raise BundleError("measured motion lacks fields: %s" % sorted(missing))
        schema = int(np.asarray(loaded["measured_racket_schema_version"]).reshape(-1)[0])
        admitted = int(np.asarray(loaded["measured_racket_retarget_admitted"]).reshape(-1)[0])
        stored_uid = str(np.asarray(loaded["measured_racket_uid"]).reshape(-1)[0])
        frame_count = int(np.asarray(loaded["joint_pos"]).shape[0])
    if schema != 4 or admitted != 1 or stored_uid != action_uid:
        raise BundleError("motion is not the selected admitted measured-racket schema-v4 clip")
    strike_frame = round(_finite(source_action["strike_phase"], label="strike_phase") * (frame_count - 1))
    if strike_frame <= 0 or strike_frame >= frame_count - 1:
        raise BundleError("selected strike frame must be interior")
    alignment_pin, alignment = _validate_racket_alignment(
        root,
        Path(args.racket_alignment_audit),
        args.expected_racket_alignment_audit_sha256,
        motion_sha=motion_sha,
        action_uid=action_uid,
        frame_count=frame_count,
        strike_frame=strike_frame,
    )
    bank_receipt_pin, build_report_pin, measured_provenance = (
        _validate_measured_provenance(
            root,
            bank_receipt_path=Path(args.measured_bank_receipt),
            bank_receipt_sha=args.expected_measured_bank_receipt_sha256,
            build_report_path=Path(args.measured_manifest_build_report),
            build_report_sha=args.expected_measured_manifest_build_report_sha256,
            source_manifest_sha=source_sha,
            action_id=args.action_id,
            measured_uid=action_uid,
            motion_sha=motion_sha,
            frame_count=frame_count,
            strike_frame=strike_frame,
        )
    )

    mechanical_pin, mechanical = _mechanical_selection(
        root,
        Path(args.mechanical_audit),
        args.expected_mechanical_audit_sha256,
        motion_sha=motion_sha,
        action_uid=action_uid,
        allow_unknown=args.allow_mechanical_unknown_diagnostic,
    )
    task_path, task_relative, task_sha = _exact_path(
        root, TASK_PROFILE, args.expected_task_profile_sha256, label="VendorV2 N1 task profile"
    )
    del task_path

    fixed_source = deepcopy(source)
    fixed_source["manifest_id"] = "measured_n1_%s_fixed_question_v1" % args.action_id
    fixed_source["action_order"] = [args.action_id]
    fixed_action = deepcopy(source_action)
    fixed_action["ball_profile"] = _freeze_ball_profile(
        source_action["ball_profile"]
    )
    if float(fixed_action["teacher_rate_max"]) > FIXED_TEACHER_RATE_MAX:
        raise BundleError("source teacher-rate maximum exceeds diagnostic cap")
    fixed_action["teacher_rate_max"] = FIXED_TEACHER_RATE_MAX
    fixed_source["actions"] = [fixed_action]
    fixed_source["landing_aim"] = _freeze_landing(source["landing_aim"])
    fixed_source["holdout"] = dict(source["holdout"])
    fixed_source["holdout"]["split_id"] = "diagnostic_fixed_%s_v1" % args.action_id
    fixed_source["notes"] = (
        "Diagnostic-only N=1 single-action projection. One exact immutable seed row "
        "owns the fixed incoming ball/task and arm-specific answer; reset inverse "
        "solving is prohibited."
    )
    fixed_bytes = _canonical_bytes(fixed_source)
    fixed_sha = hashlib.sha256(fixed_bytes).hexdigest()
    destination = Path(args.output_dir)
    destination = destination if destination.is_absolute() else root / destination
    destination = destination.resolve(strict=False)
    try:
        output_relative = destination.relative_to(root).as_posix()
    except ValueError as exc:
        raise BundleError("output directory must remain inside repo root") from exc
    source_name = "%s.fixed_source.v3.%s.json" % (args.action_id, fixed_sha[:12])
    fixed_path = destination / source_name
    _exclusive_write(fixed_path, fixed_source)
    live_profile_path, live_profile_sha = _materialize_live_profile_pins(
        root=root,
        template_path=Path(args.profile_pins),
        template_expected_sha=args.expected_profile_pins_sha256,
        destination=destination,
        output_relative=output_relative,
    )

    base = _load_module(
        "_measured_n1_contact_bundle_base",
        SCRIPT_DIR / "materialize_n1_contact_training_bundle.py",
    )
    base.SUPPORTED_ACTIONS = {
        args.action_id: {
            "action_uid": int(source_action["action_uid"]),
            "family": source_action["family"],
            "motion_path": motion_relative,
            "motion_sha256": motion_sha,
            "reference_t_hit_s": source_action["reference_t_hit_s"],
            "reference_t_cycle_s": source_action["reference_t_cycle_s"],
            "reference_racket_site_speed_mps": source_action["reference_racket_site_speed_mps"],
            "priority": 0,
        }
    }
    base.SOURCE_SUPPORTED_ACTIONS = {args.action_id: deepcopy(fixed_action)}
    base.FULL_SUPPORTED_ACTIONS = {
        args.action_id: {
            "motion_path": motion_relative,
            "motion_sha256": motion_sha,
            "reference_t_hit_s": source_action["reference_t_hit_s"],
            "reference_t_cycle_s": source_action["reference_t_cycle_s"],
        }
    }
    def rooted(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    offline_core_only = bool(args.offline_core_only_without_dynamic_ready)
    dynamic_ready_artifact = (
        None if offline_core_only else rooted(args.dynamic_ready_artifact)
    )
    nominal_hold_receipt = (
        None if offline_core_only else rooted(args.nominal_hold_receipt)
    )
    core = base.materialize_n1_contact_bundle(
        repo_root=root,
        action_id=args.action_id,
        source_manifest=fixed_path,
        expected_source_manifest_sha256=fixed_sha,
        profile_pins=live_profile_path,
        expected_profile_pins_sha256=live_profile_sha,
        dynamic_ready_artifact=dynamic_ready_artifact,
        expected_dynamic_ready_artifact_sha256=args.expected_dynamic_ready_artifact_sha256,
        nominal_hold_receipt=nominal_hold_receipt,
        expected_nominal_hold_receipt_sha256=args.expected_nominal_hold_receipt_sha256,
        output_dir=destination,
        require_git_tracked_motion=False,
        scope="full",
        strike_frame=strike_frame,
        full_episode_length_s=args.episode_length_s,
        full_attempt_close_margin_s=0.02,
        offline_core_only_without_dynamic_ready=offline_core_only,
        full_solver_preflight_support_source={
            "path": source_relative,
            "sha256": source_sha,
        },
    )
    core_pin = {"path": core["bundle_path"], "sha256": core["bundle_sha256"]}
    prepared = {
        "schema_version": 1,
        "artifact_type": PREPARED_KIND,
        "action_id": args.action_id,
        "action_uid": int(source_action["action_uid"]),
        "measured_uid": action_uid,
        "source_manifest": {"path": source_relative, "sha256": source_sha},
        "fixed_n1_source_manifest": {
            "path": (PurePosixPath(output_relative) / source_name).as_posix(),
            "sha256": fixed_sha,
        },
        "motion": {"path": motion_relative, "sha256": motion_sha},
        "racket_alignment_audit": alignment_pin,
        "racket_alignment": alignment,
        "measured_bank_receipt": bank_receipt_pin,
        "measured_manifest_build_report": build_report_pin,
        "measured_provenance": measured_provenance,
        "core_contact_bundle": core_pin,
        "task_profile": {"path": task_relative, "sha256": task_sha},
        "mechanical_audit": mechanical_pin,
        "mechanical_selection": mechanical,
        "claims": {
            "diagnostic_unauthorized": True,
            "training_authorized": False,
            "formal_evidence_prohibited": True,
            "promotion_prohibited": True,
            "export_prohibited": True,
            "deployment_prohibited": True,
            "hardware_prohibited": True,
            "dynamic_ready_status": (
                "BLOCKED_EXTERNAL_EVIDENCE"
                if offline_core_only
                else "PASS"
            ),
        },
    }
    prepared_bytes = _canonical_bytes(prepared)
    prepared_sha = hashlib.sha256(prepared_bytes).hexdigest()
    prepared_name = "%s.measured_prepared_core.v1.%s.json" % (
        args.action_id,
        prepared_sha[:12],
    )
    _exclusive_write(destination / prepared_name, prepared)
    return {
        "status": "PREPARED_DIAGNOSTIC_ONLY",
        "diagnostic_unauthorized": True,
        "prepared_core_bundle_path": (
            PurePosixPath(output_relative) / prepared_name
        ).as_posix(),
        "prepared_core_bundle_sha256": prepared_sha,
        "core_contact_bundle": core_pin,
        "action_id": args.action_id,
        "measured_uid": action_uid,
        "fixed_question": True,
        "next_required_artifacts": [
            "one offline ActionBallTaskReceipt for the fixed base question",
            "five same-question target receipts with distinct producer SHAs",
            "one canonical immutable N1 tape plus tracked receipt",
        ],
        "mechanical_verdict": mechanical["mechanical_verdict"],
    }


_PREPARED_KEYS = {
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
FINAL_BUNDLE_KEYS = frozenset(
    _PREPARED_KEYS
    | {
        "prepared_core_bundle",
        "immutable_tape_build_report",
        "immutable_tape",
        "target_recipe",
        "target_validity",
        "runtime_contract",
    }
)


def _validate_prepared_core(
    root: Path, path: Path, expected_sha: str
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    prepared_path, relative, actual = _exact_path(
        root, path, expected_sha, label="prepared measured N1 core bundle"
    )
    prepared = _strict_json(prepared_path, label="prepared measured N1 core bundle")
    if set(prepared) != _PREPARED_KEYS:
        raise BundleError("prepared measured N1 core bundle keys differ")
    if (
        prepared["schema_version"] != 1
        or prepared["artifact_type"] != PREPARED_KIND
        or prepared["action_id"] != ACTION_ID
        or prepared["action_uid"] != ACTION_FACTS["action_uid"]
        or prepared["measured_uid"] != MEASURED_UID
        or prepared["motion"]
        != {"path": ACTION_FACTS["motion_path"], "sha256": ACTION_FACTS["motion_sha256"]}
    ):
        raise BundleError("prepared measured N1 core identity differs")
    for key, label in (
        ("source_manifest", "measured source manifest"),
        ("fixed_n1_source_manifest", "fixed N1 source manifest"),
        ("motion", "measured motion"),
        ("racket_alignment_audit", "independent racket FK audit"),
        ("measured_bank_receipt", "measured bank receipt"),
        ("measured_manifest_build_report", "measured manifest build report"),
        ("core_contact_bundle", "core contact bundle"),
        ("task_profile", "VendorV2 N1 task profile"),
        ("mechanical_audit", "mechanical audit"),
    ):
        pin = prepared[key]
        if type(pin) is not dict or set(pin) != {"path", "sha256"}:
            raise BundleError("%s pin is malformed" % label)
        _exact_path(root, pin["path"], pin["sha256"], label=label)
    core_path = (root / prepared["core_contact_bundle"]["path"]).resolve(strict=True)
    core = _strict_json(core_path, label="core contact bundle")
    if (
        core.get("schema_version") != 2
        or core.get("artifact_type") != "n1_contact_training_bundle_v2"
        or core.get("action_id") != ACTION_ID
        or core.get("action_uid") != ACTION_FACTS["action_uid"]
        or core.get("scope") != "full"
        or core.get("motion") != prepared["motion"]
    ):
        raise BundleError("prepared core contact bundle identity differs")
    claims = prepared["claims"]
    if (
        type(claims) is not dict
        or claims.get("diagnostic_unauthorized") is not True
        or claims.get("training_authorized") is not False
        or claims.get("formal_evidence_prohibited") is not True
        or claims.get("promotion_prohibited") is not True
        or claims.get("export_prohibited") is not True
        or claims.get("deployment_prohibited") is not True
        or claims.get("hardware_prohibited") is not True
        or claims.get("dynamic_ready_status")
        not in {"PASS", "BLOCKED_EXTERNAL_EVIDENCE"}
    ):
        raise BundleError("prepared core authority boundary differs")
    return {"path": relative, "sha256": actual}, prepared, core


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve(strict=True)
    prepared_pin, prepared, core = _validate_prepared_core(
        root,
        Path(args.prepared_core_bundle),
        args.expected_prepared_core_bundle_sha256,
    )
    if prepared["claims"]["dynamic_ready_status"] != "PASS":
        raise BundleError(
            "finalize requires exact dynamic-ready plus Isaac nominal-hold PASS"
        )
    recipe = args.target_recipe
    tape_report_pin, tape = _validate_tape_build_report(
        root,
        Path(args.immutable_tape_build_report),
        args.expected_immutable_tape_build_report_sha256,
        action_uid=ACTION_FACTS["action_uid"],
        motion_sha=ACTION_FACTS["motion_sha256"],
        recipe=recipe,
    )
    identity = tape["source_identity"]
    profile_path = (root / core["profile_pins"]["path"]).resolve(strict=True)
    profile = _strict_json(profile_path, label="core profile pins")
    if (
        identity["action_slot"] != 0
        or identity["manifest_sha256"] != core["manifest"]["sha256"]
        or identity["physics_sha256"] != profile.get("physics_profile_sha256")
        or identity["solver_sha256"] != profile.get("solver_profile_sha256")
        or identity["counter_rally_objective_profile_sha256"]
        != profile.get("solver_payload", {}).get("counter_rally", {}).get(
            "objective_profile_sha256"
        )
    ):
        raise BundleError(
            "immutable tape was not produced from the exact prepared manifest/profile"
        )
    destination = Path(args.output_dir)
    destination = destination if destination.is_absolute() else root / destination
    destination = destination.resolve(strict=False)
    try:
        output_relative = destination.relative_to(root).as_posix()
    except ValueError as exc:
        raise BundleError("output directory must remain inside repo root") from exc
    bundle = {
        **prepared,
        "artifact_type": BUNDLE_KIND,
        "prepared_core_bundle": prepared_pin,
        "immutable_tape_build_report": tape_report_pin,
        "immutable_tape": tape["artifact"],
        "target_recipe": recipe,
        "target_validity": {
            "order": list(TARGET_ORDER),
            "mask": list(RECIPES[recipe]),
        },
        "runtime_contract": {
            "target_source": "immutable_tape",
            "reset_inverse_solve": False,
            "control_step_action_delay": [0, 0],
            "physical_ball_semantics": PHYSICAL_BALL_SEMANTICS,
            "canary_contract": "fixed_question_ablation_canary_v1",
            "actor_obs_contract": "action_ball_table_pose_twist_heading_task_teacher_start_v2",
            "actor_width": 194,
            "critic_width": 318,
            "final_varying_ball_abi": False,
            "target_validity_is_fixed_recipe_constant": True,
            "invalid_target_columns_zero_filled": True,
            "invalid_target_columns_masked_from_reward": True,
            "target_noise_disabled": True,
            "adaptive_sigma_disabled": True,
        },
    }
    if set(bundle) != FINAL_BUNDLE_KEYS:
        raise AssertionError("final measured bundle schema drifted")
    bundle_bytes = _canonical_bytes(bundle)
    bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
    bundle_name = "%s.%s.measured_bundle.v1.%s.json" % (
        ACTION_ID,
        recipe,
        bundle_sha[:12],
    )
    _exclusive_write(destination / bundle_name, bundle)
    return {
        "status": "PASS_DIAGNOSTIC_ONLY",
        "diagnostic_unauthorized": True,
        "bundle_path": (PurePosixPath(output_relative) / bundle_name).as_posix(),
        "bundle_sha256": bundle_sha,
        "prepared_core_bundle": prepared_pin,
        "core_contact_bundle": prepared["core_contact_bundle"],
        "action_id": ACTION_ID,
        "measured_uid": MEASURED_UID,
        "target_recipe": recipe,
        "target_validity_mask": list(RECIPES[recipe]),
        "fixed_question": True,
        "reset_inverse_solve": False,
        "physical_ball_semantics": PHYSICAL_BALL_SEMANTICS,
        "mechanical_verdict": prepared["mechanical_selection"]["mechanical_verdict"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser(
        "prepare",
        help="create the shared fixed source/core before offline receipt production",
    )
    prepare_parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    prepare_parser.add_argument("--action-id", required=True)
    prepare_parser.add_argument(
        "--action-uid", default=None, help="measured UID; defaults from motion filename"
    )
    prepare_parser.add_argument("--source-manifest", required=True)
    prepare_parser.add_argument("--expected-source-manifest-sha256", required=True)
    prepare_parser.add_argument("--measured-bank-receipt", required=True)
    prepare_parser.add_argument("--expected-measured-bank-receipt-sha256", required=True)
    prepare_parser.add_argument("--measured-manifest-build-report", required=True)
    prepare_parser.add_argument(
        "--expected-measured-manifest-build-report-sha256", required=True
    )
    prepare_parser.add_argument("--mechanical-audit", required=True)
    prepare_parser.add_argument("--expected-mechanical-audit-sha256", required=True)
    prepare_parser.add_argument("--racket-alignment-audit", required=True)
    prepare_parser.add_argument(
        "--expected-racket-alignment-audit-sha256", required=True
    )
    prepare_parser.add_argument(
        "--allow-mechanical-unknown-diagnostic", action="store_true"
    )
    prepare_parser.add_argument("--profile-pins", required=True)
    prepare_parser.add_argument("--expected-profile-pins-sha256", required=True)
    prepare_parser.add_argument("--dynamic-ready-artifact")
    prepare_parser.add_argument(
        "--expected-dynamic-ready-artifact-sha256"
    )
    prepare_parser.add_argument("--nominal-hold-receipt")
    prepare_parser.add_argument(
        "--expected-nominal-hold-receipt-sha256"
    )
    prepare_parser.add_argument(
        "--offline-core-only-without-dynamic-ready",
        action="store_true",
        help=(
            "materialize only the diagnostic question/core identity while exact "
            "dynamic-ready and Isaac nominal-hold evidence remains blocked"
        ),
    )
    prepare_parser.add_argument("--expected-task-profile-sha256", required=True)
    prepare_parser.add_argument("--episode-length-s", type=float, default=10.0)
    prepare_parser.add_argument("--output-dir", required=True)

    finalize_parser = sub.add_parser(
        "finalize",
        help="bind one canonical five-recipe tape to the already prepared exact core",
    )
    finalize_parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    finalize_parser.add_argument("--prepared-core-bundle", required=True)
    finalize_parser.add_argument(
        "--expected-prepared-core-bundle-sha256", required=True
    )
    finalize_parser.add_argument("--target-recipe", required=True, choices=tuple(RECIPES))
    finalize_parser.add_argument("--immutable-tape-build-report", required=True)
    finalize_parser.add_argument(
        "--expected-immutable-tape-build-report-sha256", required=True
    )
    finalize_parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = prepare(args) if args.command == "prepare" else finalize(args)
    except (BundleError, FileNotFoundError, ValueError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
