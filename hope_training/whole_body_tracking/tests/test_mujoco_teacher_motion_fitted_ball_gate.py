from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import mujoco
except ImportError:  # pragma: no cover - host without MuJoCo
    mujoco = None


REPO = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO
    / "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate.py"
)
SPEC = importlib.util.spec_from_file_location("teacher_fitted_ball_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)

requires_mujoco = pytest.mark.skipif(
    mujoco is None, reason="MuJoCo is not installed"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trusted_action_set(
    action_ids=GATE.FRESH_N5_ORDER,
    action_uids=None,
    *,
    scope="upper",
    mobility_mode="no_move",
    manifest_sha256="9" * 64,
):
    ids = list(action_ids)
    uids = (
        list(range(100, 100 + len(ids)))
        if action_uids is None
        else list(action_uids)
    )
    row = {
        "profile_id": f"fixture_n{len(ids)}",
        "expected_n": len(ids),
        "scope": scope,
        "mobility_mode": mobility_mode,
        "ordered_action_ids": ids,
        "ordered_action_uids": uids,
        "order_uid_digest_sha256": (
            GATE.action_set_contract.order_uid_digest(ids, uids)
        ),
        "manifest_path": "configs/fixture.json",
        "manifest_sha256": manifest_sha256,
        "experiment_name": f"fixture_n{len(ids)}",
    }
    return GATE.action_set_contract.validate_contract(
        row, profile_id=row["profile_id"], profile_policies={}
    )


def _venue() -> object:
    path = REPO / "configs/ball_physics_venue.yaml"
    return GATE.load_venue_yaml(path, _sha(path))


def _face_state(
    sign: int,
    *,
    site=(0.0, 0.0, 0.0),
    site_velocity=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
) -> object:
    rotation = np.eye(3)
    center = np.asarray(site) + rotation @ (
        GATE.racket_geometry.face_center_from_site_local(sign)
    )
    normal = rotation @ GATE.racket_geometry.face_normal_local(sign)
    return GATE.FaceState(
        site_position_m=np.asarray(site, float),
        rotation_w_from_local=rotation,
        site_linear_velocity_mps=np.asarray(site_velocity, float),
        angular_velocity_radps=np.asarray(angular, float),
        center_position_m=center,
        normal_w=normal,
    )


def _result(
    *,
    contact_time=1.0,
    contact_position=(0.2, 0.0, 1.2),
    velocity=(4.0, 0.0, 2.0),
    net_z=1.05,
    landing=(2.5, 0.1),
    landing_time=1.5,
) -> dict:
    return {
        "paddle_contact": {
            "time_s": contact_time,
            "ball_center_m": list(contact_position),
            "velocity_plus_mps": list(velocity),
        },
        "net_crossing": {"ball_center_z_m": net_z},
        "first_landing": {
            "ball_center_xy_m": list(landing),
            "time_s": landing_time,
        },
    }


def _motion_clip(
    *,
    root_start=(0.0, 0.0, 1.0),
    root_end=(0.0, 0.0, 1.0),
    joint_end_offset=0.0,
    endpoint_velocity=0.0,
) -> object:
    body_pos = np.zeros((2, 32, 3), dtype=float)
    body_pos[0, 0] = root_start
    body_pos[1, 0] = root_end
    body_quat = np.zeros((2, 32, 4), dtype=float)
    body_quat[..., 0] = 1.0
    joint_pos = np.zeros((2, 31), dtype=float)
    joint_pos[1, 0] = joint_end_offset
    joint_vel = np.zeros((2, 31), dtype=float)
    joint_vel[[0, -1], 0] = endpoint_velocity
    body_lin = np.zeros((2, 32, 3), dtype=float)
    body_lin[[0, -1], 0, 0] = endpoint_velocity
    body_ang = np.zeros((2, 32, 3), dtype=float)
    body_ang[[0, -1], 0, 2] = endpoint_velocity
    return GATE.motion_player.MotionClip(
        path=Path("fixture.npz"),
        fps=1.0,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        has_migration_provenance=False,
        body_lin_vel_point="center_of_mass",
    )


def _physical_task_binding_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, dict]:
    monkeypatch.setattr(GATE, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(GATE.native_diag, "REPO_ROOT", tmp_path)
    producer_relative = (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    )
    producer = tmp_path / producer_relative
    producer.parent.mkdir(parents=True)
    producer.write_text("EXACT_RUNTIME_RECEIPT_PRODUCER = True\n")
    source_map = {
        "continuous_questions.py": "1" * 64,
        "hope_commands.py": _sha(producer),
        "racket_contact_geometry.py": "2" * 64,
        "stroke_adapt_torch.py": "3" * 64,
        "virtual_ball.py": "4" * 64,
    }
    profile = {
        "time_to_contact_center_s": 1.2,
        "time_to_contact_min_s": 1.1,
        "time_to_contact_max_s": 1.3,
        "incoming_speed_min_mps": 1.0,
        "incoming_speed_max_mps": 3.0,
        "spin_magnitude_min_radps": 0.0,
        "spin_magnitude_max_radps": 20.0,
    }
    action = GATE.native_diag.ActionSpec(
        action_id="bh_loop_c",
        action_uid=123,
        motion_path=tmp_path / "motion.npz",
        motion_sha256="5" * 64,
        strike_phase=0.5,
        t_hit_s=1.0,
        t_cycle_s=2.0,
        racket_speed_mps=1.0,
        reaction_margin_s=0.1,
        mount_normal_sign=1,
        ball_profile=profile,
    )
    solver_sha = "6" * 64
    physics_sha = "7" * 64
    execution_identity = {
        "artifact_type": "frozen_ball_to_task_solver_execution_v1",
        "execution_id": "fixture-execution",
        "executed_before_gate": True,
        "solver_replayed_exact": True,
        "selector_executed": False,
        "action_identity_frozen": True,
        "action_switching_allowed": False,
        "hardware_authorized": False,
    }
    execution_sha = GATE._canonical_payload_sha256(
        execution_identity
    )

    def make_case(
        role: str,
        seed: int,
        *,
        support: bool = False,
    ) -> dict:
        contact = [0.51, 0.0, 1.2] if support else [0.5, 0.0, 1.2]
        ttc = 1.25 if support else 1.2
        incoming = [-2.1, 0.0, 0.0] if support else [-2.0, 0.0, 0.0]
        launch_payload = {
            "activation_time_s": 0.0,
            "position_w_m": [3.0, 0.0, 1.3],
            "velocity_w_mps": [-2.0, 0.0, 0.0],
            "spin_w_radps": [0.0, 0.0, 0.0],
            "required_incoming_table_bounces": 1,
        }
        launch = {
            **launch_payload,
            "state_sha256": GATE._canonical_payload_sha256(
                launch_payload
            ),
        }
        proposal = {
            "action_id": action.action_id,
            "action_uid": action.action_uid,
            "motion_sha256": action.motion_sha256,
            "sample_seed": seed,
            "sample_index": seed,
            "ball_contact_w_m": contact,
            "time_to_contact_s": ttc,
            "incoming_velocity_w_mps": incoming,
            "incoming_spin_w_radps": [0.0, 0.0, 0.0],
            "base_spawn_w_m": [0.0, 0.0, 1.0],
            "base_goal_w_m": [0.0, 0.0, 1.0],
            "landing_aim_w_xy_m": [2.5, 0.0],
            "launch": launch,
        }
        proposal_sha = GATE._canonical_payload_sha256(proposal)
        geometry = (
            GATE.racket_geometry._production.solve_exact_face_contact(
                ball_contact_w_m=contact,
                racket_face_center_velocity_w_mps=[1.0, 0.0, 0.0],
                solved_raw_a_normal_w=[1.0, 0.0, 0.0],
                mount_normal_sign=1,
                reference_racket_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
                reference_racket_angular_velocity_w_radps=[
                    0.0,
                    0.0,
                    0.0,
                ],
                reference_racket_site_speed_mps=1.0,
                teacher_rate_min=0.5,
                teacher_rate_max=1.5,
            )
        )
        task = {
            "action_id": action.action_id,
            "action_uid": action.action_uid,
            "motion_sha256": action.motion_sha256,
            "ball_proposal_sha256": proposal_sha,
            "mount_normal_sign": 1,
            "ball_contact_w_m": contact,
            "racket_site_target_w_m": list(
                geometry.racket_site_target_w_m
            ),
            "racket_normal_w": [1.0, 0.0, 0.0],
            "reference_racket_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "reference_racket_angular_velocity_w_radps": [
                0.0,
                0.0,
                0.0,
            ],
            "racket_command_quat_wxyz": list(
                geometry.racket_command_quat_wxyz
            ),
            "racket_face_center_velocity_w_mps": list(
                geometry.racket_face_center_velocity_w_mps
            ),
            "racket_site_velocity_w_mps": list(
                geometry.racket_site_velocity_w_mps
            ),
            "racket_command_angular_velocity_w_radps": list(
                geometry.racket_command_angular_velocity_w_radps
            ),
            "geometry_source_sha256": (
                GATE.racket_geometry.GEOMETRY_SOURCE_SHA256
            ),
            "reference_t_hit_s": 1.0,
            "reference_t_cycle_s": 2.0,
            "reference_racket_site_speed_mps": 1.0,
            "required_racket_site_speed_mps": 1.0,
            "reaction_margin_s": 0.1,
            "teacher_rate_min": 0.5,
            "teacher_rate_max": 1.5,
            "teacher_rate": 1.0,
            "scaled_t_hit_s": 1.0,
            "scaled_t_cycle_s": 2.0,
            "pre_swing_wait_s": ttc - 1.0,
            "solver_residual_m": 0.01,
            "landing_aim_w_xy_m": [2.5, 0.0],
            "solver_profile_sha256": solver_sha,
            "physics_profile_sha256": physics_sha,
        }
        task_sha = GATE._canonical_payload_sha256(task)
        if role in GATE.PHYSICAL_TASK_POSITIVE_ROLES:
            fault = {"kind": "none"}
            expected = "PASS"
            reason = None
        elif role == "negative_t_hit_offset":
            fault = {
                "kind": "teacher_t_hit_offset",
                "offset_s": 0.05,
            }
            expected = "FAIL"
            reason = "teacher_task_contact_time_mismatch"
        elif role == "negative_face_sign":
            fault = {"kind": "selected_face_sign_flip"}
            expected = "FAIL"
            reason = "teacher_task_face_sign_mismatch"
        else:
            fault = {
                "kind": "launch_velocity_delta",
                "launch_velocity_delta_w_mps": [0.0, 0.3, 0.0],
            }
            expected = "FAIL"
            reason = "teacher_task_ball_state_mismatch"
        case_id = f"{action.action_id}:{role}"
        binding_payload = {
            "action_id": action.action_id,
            "action_uid": action.action_uid,
            "motion_sha256": action.motion_sha256,
            "case_id": case_id,
            "case_role": role,
            "sample_seed": seed,
            "ball_proposal_sha256": proposal_sha,
            "task_payload_sha256": task_sha,
            "solver_execution_identity_sha256": execution_sha,
            "fault_injection": fault,
            "expected_physical_verdict": expected,
            "expected_failure_reason": reason,
        }
        return {
            "case_id": case_id,
            "case_role": role,
            "sample_seed": seed,
            "expected_physical_verdict": expected,
            "expected_failure_reason": reason,
            "ball_proposal": proposal,
            "ball_proposal_sha256": proposal_sha,
            "task_payload": task,
            "task_payload_sha256": task_sha,
            "fault_injection": fault,
            "case_binding_sha256": GATE._canonical_payload_sha256(
                binding_payload
            ),
        }

    cases = [
        make_case(role, index + 1, support=role == "support_positive")
        for index, role in enumerate(GATE.PHYSICAL_TASK_CASE_ROLES)
    ]
    external = {
        "schema_version": 1,
        "artifact_type": (
            "frozen_action_ball_solver_execution_receipt_v1"
        ),
        "producer": {
            "source_path": producer_relative,
            "source_sha256": _sha(producer),
            "runtime_receipt_type": "ActionBallTaskReceipt",
            "exact_solver_replay_required": True,
            "selector_executed": False,
            "hardware_authorized": False,
        },
        "action_identity": {
            "action_id": action.action_id,
            "action_uid": action.action_uid,
            "motion_sha256": action.motion_sha256,
        },
        "profile_identity": {
            "ball_profile_sha256": GATE._canonical_payload_sha256(
                profile
            ),
            "solver_profile_sha256": solver_sha,
            "physics_profile_sha256": physics_sha,
            "solver_implementation_source_sha256": source_map,
            "geometry_source_sha256": (
                GATE.racket_geometry.GEOMETRY_SOURCE_SHA256
            ),
        },
        "solver_execution_identity": execution_identity,
        "cases": cases,
    }
    external["receipt_payload_sha256"] = (
        GATE._canonical_payload_sha256(external)
    )
    external_path = tmp_path / "solver_receipt.json"
    external_path.write_text(json.dumps(external))
    binding = {
        "schema_version": 1,
        "authority": GATE.PHYSICAL_TASK_BINDING_AUTHORITY,
        "action_id": action.action_id,
        "action_uid": action.action_uid,
        "motion_sha256": action.motion_sha256,
        "ball_profile_sha256": GATE._canonical_payload_sha256(profile),
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "solver_implementation_source_sha256": source_map,
        "solver_execution_receipt_path": external_path.name,
        "solver_execution_receipt_sha256": _sha(external_path),
        "solver_execution_identity": execution_identity,
        "solver_execution_identity_sha256": execution_sha,
        "selector_executed": False,
        "action_identity_frozen": True,
        "cases": cases,
        "cases_sha256": GATE.native_diag.sha256_bytes(
            GATE.native_diag.canonical_json_bytes(cases)
        ),
    }
    return action, binding


def test_checked_in_n5_named_manifest_is_rejected_as_n4_before_runtime():
    path = REPO / "configs/action_ball_n5_nomove_f20_20260728.json"
    raw, _receipt = GATE.native_diag.read_json_exact(
        path, "manifest", expected_sha256=_sha(path)
    )
    with pytest.raises(GATE.FittedGateError, match=r"action_count_mismatch.*N=5.*has 4"):
        GATE.validate_physical_manifest(
            raw, trusted_action_set=_trusted_action_set()
        )


def test_fresh_n5_order_is_frozen_and_cli_has_no_expected_actions_escape_hatch():
    assert GATE.FRESH_N5_ORDER == (
        "bh_loop_c",
        "v12_forehand_block",
        "bh_block",
        "s0_highpress",
        "fh_loop_high",
    )
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--expected-actions" not in source
    assert "contains_retired_old_forehand_fh_loop" in source


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


@pytest.mark.parametrize("action_count", (1, 5, 73))
def test_physical_overlay_closure_is_exact_for_n1_n5_n73(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action_count: int,
):
    monkeypatch.setattr(GATE, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(GATE.native_diag, "REPO_ROOT", tmp_path)
    ids = [f"a{index:03d}" for index in range(action_count)]
    uids = list(range(1000, 1000 + action_count))
    strict = {
        "schema_version": 3,
        "manifest_id": f"n{action_count}",
        "mobility_mode": "no_move",
        "action_order": ids,
        "prototype": {"scope": "full" if action_count == 73 else "upper"},
        "actions": [
            {
                "action_id": action_id,
                "action_uid": uid,
                "family": "backhand" if index % 2 == 0 else "forehand",
                "motion_path": f"motions/{action_id}.npz",
                "motion_sha256": f"{index + 1:064x}",
                "ball_profile": {
                    "contact_offset_center_b_yaw_m": [
                        0.6,
                        0.01 * index,
                        1.0,
                    ],
                    "time_to_contact_center_s": 1.2 + 0.001 * index,
                    "incoming_direction_center_b_yaw": [
                        -1.0,
                        0.0,
                        0.0,
                    ],
                    "incoming_speed_center_mps": 2.0 + 0.01 * index,
                    "spin_direction_center_b_yaw": [0.0, 1.0, 0.0],
                    "spin_magnitude_center_radps": 0.0,
                    "base_spawn_center_w_xy_m": [0.0, 0.0],
                    "base_travel_center_b_yaw_xy_m": [0.0, 0.0],
                },
            }
            for index, (action_id, uid) in enumerate(zip(ids, uids))
        ],
    }
    strict_path = tmp_path / f"strict_n{action_count}.json"
    _write_json(strict_path, strict)
    trusted = _trusted_action_set(
        ids,
        uids,
        scope="full" if action_count == 73 else "upper",
        manifest_sha256=_sha(strict_path),
    )
    trusted = dict(trusted)
    trusted["manifest_path"] = strict_path.name
    # Recompute the derived contract after changing its fixture path.
    trusted = GATE.action_set_contract.validate_contract(
        {key: trusted[key] for key in GATE.action_set_contract.CONTRACT_KEYS},
        profile_id=trusted["profile_id"],
        profile_policies={},
    )
    gate_fields = {
        "racket_geometry_contract": {"schema_version": 2},
        "physical_contact_contract": {"schema_version": 2},
    }
    physical = json.loads(json.dumps(strict))
    physical.update(gate_fields)
    bundle_actions = []
    candidates = []
    compiler_manifests = []
    bank_gate_reports = []
    for index, row in enumerate(physical["actions"]):
        candidate_path = tmp_path / f"candidate_{index:03d}.json"
        _write_json(candidate_path, {"action_id": ids[index]})
        candidate = {
            "action_id": ids[index],
            "path": candidate_path.name,
            "sha256": _sha(candidate_path),
        }
        candidates.append(candidate)
        compiler_path = tmp_path / f"compiler_{index:03d}.json"
        bank_path = tmp_path / f"bank_{index:03d}.json"
        _write_json(
            compiler_path,
            {"action_id": ids[index], "kind": "compiler"},
        )
        _write_json(
            bank_path,
            {"action_id": ids[index], "kind": "bank"},
        )
        compiler_row = {
            "action_id": ids[index],
            "path": compiler_path.name,
            "sha256": _sha(compiler_path),
        }
        bank_row = {
            "action_id": ids[index],
            "path": bank_path.name,
            "sha256": _sha(bank_path),
        }
        compiler_manifests.append(compiler_row)
        bank_gate_reports.append(bank_row)
        launch = {"state": index}
        task = {"cases": index}
        row["physical_ball_launch"] = launch
        row["physical_task_binding"] = task
        row["admission"] = {
            "registry_entry_path": candidate["path"],
            "registry_entry_sha256": candidate["sha256"],
            "compiler_manifest_path": compiler_row["path"],
            "compiler_manifest_sha256": compiler_row["sha256"],
            "bank_gate_report_path": bank_row["path"],
            "bank_gate_report_sha256": bank_row["sha256"],
        }
        bundle_actions.append(
            {
                "action_id": ids[index],
                "action_uid": uids[index],
                "physical_ball_launch": launch,
                "physical_task_binding": task,
            }
        )
    physical_path = tmp_path / f"physical_n{action_count}.json"
    _write_json(physical_path, physical)
    identity_matrix = GATE.materialization_action_identity_matrix(
        strict, trusted
    )
    bundle = {
        "base_manifest": {
            "path": strict_path.name,
            "raw_sha256": _sha(strict_path),
            "schema_version": 3,
            "strict_training_input": True,
        },
        "action_order": ids,
        "mobility_mode": "no_move",
        "selector_executed": False,
        "action_identity_frozen": True,
        "action_switching_allowed": False,
        "gate_materialization_fields": gate_fields,
        "action_identity_matrix": identity_matrix,
        "actions": bundle_actions,
    }
    bundle_path = tmp_path / f"bundle_n{action_count}.json"
    _write_json(bundle_path, bundle)
    receipt = {
        "schema_version": 2,
        "kind": (
            GATE.GENERIC_PHYSICAL_GATE_MATERIALIZATION_RECEIPT_KIND
        ),
        "action_set_contract": trusted,
        "action_identity_matrix": identity_matrix,
        "strict_training_manifest": {
            "path": strict_path.name,
            "sha256": _sha(strict_path),
        },
        "physical_task_bundle": {
            "path": bundle_path.name,
            "sha256": _sha(bundle_path),
        },
        "physical_gate_manifest": {
            "path": physical_path.name,
            "sha256": _sha(physical_path),
        },
        "candidate_entries": candidates,
        "compiler_manifests": compiler_manifests,
        "bank_gate_reports": bank_gate_reports,
        "action_order": ids,
        "strict_training_manifest_preserved": True,
        "inline_manifest_gate_only": True,
        "selector_executed": False,
        "authorization_granted": False,
    }
    receipt_path = tmp_path / f"receipt_n{action_count}.json"
    _write_json(receipt_path, receipt)
    result = GATE.validate_physical_materialization_closure(
        strict_manifest=strict,
        strict_manifest_path=strict_path,
        strict_manifest_sha256=_sha(strict_path),
        physical_manifest=physical,
        physical_manifest_path=physical_path,
        physical_manifest_sha256=_sha(physical_path),
        receipt_path=receipt_path,
        receipt_sha256=_sha(receipt_path),
        trusted_action_set=trusted,
    )
    assert result["action_set_contract_sha256"] == trusted[
        "contract_sha256"
    ]
    assert len(result["action_identity_matrix_sha256"]) == 64
    if action_count > 1:
        bundle["actions"] = list(reversed(bundle["actions"]))
        _write_json(bundle_path, bundle)
        receipt["physical_task_bundle"]["sha256"] = _sha(bundle_path)
        _write_json(receipt_path, receipt)
        with pytest.raises(
            GATE.FittedGateError,
            match="bundle action rows are not in exact trusted order",
        ):
            GATE.validate_physical_materialization_closure(
                strict_manifest=strict,
                strict_manifest_path=strict_path,
                strict_manifest_sha256=_sha(strict_path),
                physical_manifest=physical,
                physical_manifest_path=physical_path,
                physical_manifest_sha256=_sha(physical_path),
                receipt_path=receipt_path,
                receipt_sha256=_sha(receipt_path),
                trusted_action_set=trusted,
            )


def test_schema1_materialization_receipt_is_n5_profile_only():
    trusted = _trusted_action_set(["only"], [1000])
    with pytest.raises(
        GATE.FittedGateError,
        match="schema-1 fresh-N5.*non-N5",
    ):
        GATE.validate_physical_materialization_receipt(
            {
                "schema_version": 1,
                "kind": GATE.PHYSICAL_GATE_MATERIALIZATION_RECEIPT_KIND,
                "strict_training_manifest": {
                    "path": "strict.json",
                    "sha256": "1" * 64,
                },
                "physical_task_bundle": {
                    "path": "bundle.json",
                    "sha256": "2" * 64,
                },
                "physical_gate_manifest": {
                    "path": "physical.json",
                    "sha256": "3" * 64,
                },
                "candidate_entries": [],
                "compiler_manifests": {},
                "bank_gate_reports": {},
                "action_order": ["only"],
                "strict_training_manifest_preserved": True,
                "inline_manifest_gate_only": True,
                "selector_executed": False,
                "authorization_granted": False,
            },
            strict_manifest_pin={
                "path": "strict.json",
                "sha256": "1" * 64,
            },
            physical_manifest_pin={
                "path": "physical.json",
                "sha256": "3" * 64,
            },
            trusted_action_set=trusted,
        )


def test_formal_video_sampling_is_complete_for_small_n_and_bounded_for_n73():
    assert GATE.formal_video_action_slots(1) == (0,)
    assert GATE.formal_video_action_slots(5) == (0, 1, 2, 3, 4)
    n73 = GATE.formal_video_action_slots(73)
    assert len(n73) == 8
    assert n73[0] == 0
    assert n73[-1] == 72
    assert tuple(sorted(set(n73))) == n73


def test_return_safety_rows_preserve_every_action_case_dt_without_average():
    def dt_result(*, returned: bool) -> dict:
        return {
            "verdict": "PASS" if returned else "FAIL",
            "mandatory_gates": {
                "physical_ball_selected_face_return_and_first_landing": (
                    returned
                    ),
                    "teacher_robot_and_racket_five_solid_clearance": True,
                    "teacher_ground_contact_safety": True,
                },
                "five_solid_robot_safety": {
                    "contact_count": 0,
                    "swept_hit_count": 0,
                },
                "ground_contact_safety": {
                    "contact_count": 4,
                    "legal_foot_support_contact_count": 4,
                    "foot_floor_penetration_violation_count": 0,
                    "nonfoot_ground_contact_violation_count": 0,
                },
            "fall": None,
            "joint_limit_violation": None,
            "failure_reasons": [] if returned else ["did_not_land"],
        }

    actions = [
        {
            "scope": "upper",
            "action_id": "a",
            "action_uid": 1,
            "family": "backhand",
            "motion_sha256": "1" * 64,
            "profile_center_sha256": "2" * 64,
            "physical_task_binding": {
                "cases": [
                    {
                        "case_id": "a:center",
                        "case_role": "center_positive_seed_0",
                        "expected_physical_verdict": "PASS",
                        "dt_results": {
                            "0.0010": dt_result(returned=True),
                            "0.0005": dt_result(returned=False),
                        },
                    }
                ]
            },
        }
    ]
    rows = GATE.build_teacher_return_safety_rows(actions)
    assert len(rows) == 2
    assert [row["timestep_s"] for row in rows] == [0.001, 0.0005]
    assert [row["teacher_return_pass"] for row in rows] == [True, False]
    assert all(row["teacher_ground_safety_pass"] for row in rows)
    assert all(row["ground_contact_count"] == 4 for row in rows)
    assert rows[1]["failure_reasons"] == ["did_not_land"]


@pytest.mark.parametrize(
    "disconnect",
    ("strict_manifest", "physical_gate_manifest", "candidate_entry"),
)
def test_physical_overlay_receipt_rejects_swapped_or_disconnected_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disconnect: str,
):
    # Reuse the complete N1 fixture, then corrupt one receipt edge while
    # keeping the receipt file itself freshly hashed.
    test_physical_overlay_closure_is_exact_for_n1_n5_n73(
        tmp_path, monkeypatch, 1
    )
    receipt_path = tmp_path / "receipt_n1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if disconnect == "candidate_entry":
        receipt["candidate_entries"][0]["sha256"] = "0" * 64
    else:
        receipt[
            "strict_training_manifest"
            if disconnect == "strict_manifest"
            else disconnect
        ]["sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    strict_path = tmp_path / "strict_n1.json"
    physical_path = tmp_path / "physical_n1.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    ids = ["a000"]
    uids = [1000]
    trusted = _trusted_action_set(
        ids, uids, manifest_sha256=_sha(strict_path)
    )
    trusted = dict(trusted)
    trusted["manifest_path"] = strict_path.name
    trusted = GATE.action_set_contract.validate_contract(
        {key: trusted[key] for key in GATE.action_set_contract.CONTRACT_KEYS},
        profile_id=trusted["profile_id"],
        profile_policies={},
    )
    with pytest.raises(GATE.FittedGateError):
        GATE.validate_physical_materialization_closure(
            strict_manifest=strict,
            strict_manifest_path=strict_path,
            strict_manifest_sha256=_sha(strict_path),
            physical_manifest=physical,
            physical_manifest_path=physical_path,
            physical_manifest_sha256=_sha(physical_path),
            receipt_path=receipt_path,
            receipt_sha256=_sha(receipt_path),
            trusted_action_set=trusted,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "strict_action_field",
        "bundle_task_binding",
        "action_set_contract",
        "compiler_pin",
    ),
)
def test_physical_overlay_rejects_cross_layer_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
):
    test_physical_overlay_closure_is_exact_for_n1_n5_n73(
        tmp_path, monkeypatch, 1
    )
    strict_path = tmp_path / "strict_n1.json"
    physical_path = tmp_path / "physical_n1.json"
    bundle_path = tmp_path / "bundle_n1.json"
    receipt_path = tmp_path / "receipt_n1.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if tamper == "strict_action_field":
        physical["actions"][0]["action_uid"] += 1
        _write_json(physical_path, physical)
        receipt["physical_gate_manifest"]["sha256"] = _sha(
            physical_path
        )
    elif tamper == "bundle_task_binding":
        bundle["actions"][0]["physical_task_binding"] = {
            "disconnected": True
        }
        _write_json(bundle_path, bundle)
        receipt["physical_task_bundle"]["sha256"] = _sha(bundle_path)
    elif tamper == "action_set_contract":
        receipt["action_set_contract"]["scope"] = "full"
    else:
        receipt["compiler_manifests"][0]["sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    trusted = _trusted_action_set(
        ["a000"], [1000], manifest_sha256=_sha(strict_path)
    )
    trusted = dict(trusted)
    trusted["manifest_path"] = strict_path.name
    trusted = GATE.action_set_contract.validate_contract(
        {key: trusted[key] for key in GATE.action_set_contract.CONTRACT_KEYS},
        profile_id=trusted["profile_id"],
        profile_policies={},
    )
    with pytest.raises(GATE.FittedGateError):
        GATE.validate_physical_materialization_closure(
            strict_manifest=strict,
            strict_manifest_path=strict_path,
            strict_manifest_sha256=_sha(strict_path),
            physical_manifest=physical,
            physical_manifest_path=physical_path,
            physical_manifest_sha256=_sha(physical_path),
            receipt_path=receipt_path,
            receipt_sha256=_sha(receipt_path),
            trusted_action_set=trusted,
        )


def test_clean_checkout_pin_requires_git_object_id_not_sha256():
    with pytest.raises(GATE.FittedGateError, match="40-digit Git SHA"):
        GATE.validate_clean_checkout("a" * 64)


def test_fitted_gate_consumes_non_authorizing_candidate_evidence_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(GATE.native_diag, "REPO_ROOT", tmp_path)
    action = SimpleNamespace(
        action_id="bh_loop_c",
        motion_sha256=hashlib.sha256(b"motion").hexdigest(),
    )
    artifacts = {
        "registry_entry": {
            "action_id": action.action_id,
            "motion_sha256": action.motion_sha256,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "compiler_manifest": {
            "publication_class": "compiler_candidate",
            "training_authorized": False,
            "outputs": [
                {
                    "motion_id": action.action_id,
                    "npz_sha256": action.motion_sha256,
                }
            ],
        },
        "bank_gate_report": {
            "schema_version": 1,
            "verdict": "PASS",
            "bank_gate_pass": True,
            "candidate_integrity_pass": True,
            "grounded_trace_status": (
                "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
            ),
            "publication_class": "post_build_diagnostic_only",
            "training_authorized": False,
            "hardware_authorized": False,
            "contracts": {
                "shared_ready": True,
                "six_endpoint_velocity_classes_exact_zero": True,
                "grounded_inverse_dynamics": (
                    "content_addressed_actual_time_law_trace_reopened_"
                    "then_double_support_lp_at_left_midpoint_right_of_"
                    "every_cell"
                ),
                "grounded_trace_status": (
                    "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
                ),
            },
            "aggregate": {
                "clip_count": 1,
                "joint_limit_pass_count": 1,
                "geometry_pass_count": 1,
                "complete_dynamics_pass_count": 1,
                "grounded_lmr_pass_count": 1,
                "time_law_artifact_count": 1,
                "failed_count": 0,
                "incomplete_fail_closed_count": 0,
                "grounded_lmr_incomplete_count": 0,
                "self_collision_violation_count": 0,
                "foot_floor_penetration_violation_count": 0,
                "nonfoot_floor_penetration_violation_count": 0,
                "other_world_penetration_violation_count": 0,
            },
            "clips": [
                {
                    "motion_id": action.action_id,
                    "scope": "upper",
                    "sha256": action.motion_sha256,
                    "canonical_time_law": {
                        "schema_version": 2,
                        "artifact_type": (
                            "canonical_time_law_collocation_v2"
                        ),
                        "artifact_npz_sha256": "1" * 64,
                        "artifact_manifest_sha256": "2" * 64,
                        "artifact_bundle_sha256": "3" * 64,
                        "schema2_joint_tick_q_exact_after_published_"
                        "dtype_cast": True,
                        "schema2_joint_tick_qdot_exact_after_"
                        "published_dtype_cast": True,
                        "solver_input_output_array_binding_recomputed": (
                            True
                        ),
                        "finite_difference_reconstruction_used": False,
                        "soft_safety_envelope_pass": True,
                    },
                    "grounded_left_midpoint_right": {
                        "status": (
                            "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
                        ),
                        "sample_count": 3,
                        "cell_count": 1,
                        "roles": ["left", "midpoint", "right"],
                        "all_feasible": True,
                        "finite_difference_qacc_used": False,
                        "qacc_contract": (
                            "q_s*u+q_ss*x_from_persisted_compiler_trace"
                        ),
                    },
                }
            ],
        },
    }
    admission = {
        "evidence_stage": "compiler_candidate_pre_admission_v1",
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
        "scope": "upper",
    }
    for name, payload in artifacts.items():
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        admission[f"{name}_path"] = path.name
        admission[f"{name}_sha256"] = _sha(path)
    evidence = GATE.validate_candidate_pre_admission(action, admission)
    assert evidence["training_authorized"] is False
    assert evidence["evidence_stage"] == (
        "compiler_candidate_pre_admission_v1"
    )
    assert "promotion_certificate_path" not in admission
    assert "trust_set_path" not in admission

    registry_path = tmp_path / "registry_entry.json"
    staged_registry = tmp_path / "staging" / "registry_entry.json"
    staged_registry.parent.mkdir()
    staged_registry.write_bytes(registry_path.read_bytes())
    registry_path.unlink()
    staged_evidence = GATE.validate_candidate_pre_admission(
        action,
        admission,
        repo_file_overrides={
            admission["registry_entry_path"]: staged_registry
        },
    )
    assert staged_evidence["artifacts"]["registry_entry"]["path"] == str(
        staged_registry.resolve()
    )
    with pytest.raises(
        GATE.FittedGateError,
        match="only this action's registry entry|cannot be staged",
    ):
        GATE.validate_candidate_pre_admission(
            action,
            admission,
            repo_file_overrides={
                admission["compiler_manifest_path"]: (
                    tmp_path / "compiler_manifest.json"
                )
            },
        )
    registry_path.write_bytes(staged_registry.read_bytes())

    bank_path = tmp_path / "bank_gate_report.json"
    bad_bank = json.loads(bank_path.read_text())
    bad_bank["grounded_trace_status"] = "INCOMPLETE_FAIL_CLOSED"
    bank_path.write_text(json.dumps(bad_bank))
    forged_ground = dict(admission)
    forged_ground["bank_gate_report_sha256"] = _sha(bank_path)
    with pytest.raises(
        GATE.FittedGateError, match="exact modern non-authorizing grounded PASS"
    ):
        GATE.validate_candidate_pre_admission(
            action, forged_ground
        )

    forged = dict(admission)
    forged["training_authorized"] = True
    with pytest.raises(
        GATE.FittedGateError, match="non-authorizing compiler-candidate"
    ):
        GATE.validate_candidate_pre_admission(action, forged)


def test_receipt_path_is_atomically_reserved_and_never_clobbered(
    tmp_path: Path,
):
    path, descriptor = GATE._reserve_receipt_path(
        tmp_path / "receipt.json"
    )
    with pytest.raises(GATE.FittedGateError, match="existing/case-colliding"):
        GATE._reserve_receipt_path(path)
    identity = GATE._write_reserved_receipt(
        path, descriptor, {"status": "fixture"}
    )
    assert json.loads(path.read_text()) == {"status": "fixture"}
    assert identity["descriptor_readback_verified"] is True
    assert identity["pathname_identity_verified_after_write"] is True
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_receipt_write_rejects_path_swap_after_initial_inode_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path, descriptor = GATE._reserve_receipt_path(
        tmp_path / "receipt.json"
    )
    real_ftruncate = GATE.os.ftruncate
    replaced = False

    def replace_path_then_truncate(fd: int, length: int) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            path.unlink()
            path.write_text('{"forged":true}\n')
        real_ftruncate(fd, length)

    monkeypatch.setattr(GATE.os, "ftruncate", replace_path_then_truncate)
    with pytest.raises(
        GATE.FittedGateError,
        match="pathname changed during/after durable write",
    ):
        GATE._write_reserved_receipt(
            path, descriptor, {"status": "PASS"}
        )
    os.close(descriptor)
    assert json.loads(path.read_text()) == {"forged": True}


def test_pinned_regular_file_rejects_symlink_and_hashes_consumed_bytes(
    tmp_path: Path,
):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"exact")
    digest = hashlib.sha256(b"exact").hexdigest()
    resolved, raw = GATE.read_pinned_regular_file(
        payload, digest, "fixture"
    )
    assert resolved == payload.resolve()
    assert raw == b"exact"
    alias = tmp_path / "alias.bin"
    alias.symlink_to(payload)
    with pytest.raises(GATE.FittedGateError, match="must not be a symlink"):
        GATE.read_pinned_regular_file(alias, digest, "fixture alias")


def test_compiler_assets_must_match_verified_source_closure():
    payload = b"mesh"
    digest = hashlib.sha256(payload).hexdigest()
    closure = {
        "members": [
            {
                "path": "meshes/example.stl",
                "sha256": digest,
                "size_bytes": len(payload),
            }
        ]
    }
    receipt = GATE.verify_compiler_assets_against_source_closure(
        {"meshes/example.stl": payload}, closure
    )
    assert receipt[0]["sha256"] == digest
    with pytest.raises(GATE.FittedGateError, match="changed"):
        GATE.verify_compiler_assets_against_source_closure(
            {"meshes/example.stl": b"drift"}, closure
        )


def test_launch_source_artifact_must_parse_and_bind_action_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(GATE.native_diag, "REPO_ROOT", tmp_path)
    upstream = tmp_path / "recording.bin"
    action = GATE.native_diag.ActionSpec(
        action_id="bh_loop_c",
        action_uid=123,
        motion_path=tmp_path / "motion.npz",
        motion_sha256="1" * 64,
        strike_phase=0.5,
        t_hit_s=1.0,
        t_cycle_s=2.0,
        racket_speed_mps=3.0,
        reaction_margin_s=0.1,
        mount_normal_sign=-1,
        ball_profile={},
    )
    launch = {
        "source": "recorded_pre_hit_state_v1",
        "activation_time_s": 0.2,
        "position_w_m": [3.0, 0.0, 1.2],
        "velocity_w_mps": [-3.0, 0.0, -0.2],
        "spin_w_radps": [0.0, 0.0, 0.0],
        "required_incoming_table_bounces": 1,
    }
    upstream_payload = {
        "schema_version": 1,
        "artifact_type": "recorded_ball_state_series_v1",
        "action_id": action.action_id,
        "action_uid": action.action_uid,
        "motion_sha256": action.motion_sha256,
        "coordinate_frame": "mujoco_world",
        "units": {
            "position": "m",
            "velocity": "m/s",
            "spin": "rad/s",
            "time": "s",
        },
        "samples": [
            {
                "sample_time_s": 0.2,
                "position_w_m": launch["position_w_m"],
                "velocity_w_mps": launch["velocity_w_mps"],
                "spin_w_radps": launch["spin_w_radps"],
            }
        ]
        * 8,
    }
    upstream_payload["receipt_payload_sha256"] = hashlib.sha256(
        GATE.native_diag.canonical_json_bytes(upstream_payload)
    ).hexdigest()
    upstream.write_text(json.dumps(upstream_payload))
    upstream_sha = _sha(upstream)
    artifact = {
        "schema_version": 1,
        "artifact_type": launch["source"],
        "action_id": action.action_id,
        "action_uid": action.action_uid,
        "motion_sha256": action.motion_sha256,
        "coordinate_frame": "mujoco_world",
        "units": {
            "position": "m",
            "velocity": "m/s",
            "spin": "rad/s",
            "time": "s",
        },
        "authorization": {
            "physical_gate_input_authorized": True,
            "hardware_authorized": False,
        },
        "launch_state": json.loads(json.dumps(launch)),
        "upstream_evidence_path": upstream.name,
        "upstream_evidence_sha256": upstream_sha,
        "recording_sample_index": 7,
        "recording_sample_time_s": 0.2,
    }
    artifact_path = tmp_path / "launch.json"
    artifact_path.write_text(json.dumps(artifact))
    receipt = GATE.validate_launch_source_artifact(
        path=artifact_path,
        expected_sha256=_sha(artifact_path),
        source=launch["source"],
        action=action,
        launch=launch,
    )
    assert receipt["motion_sha256"] == action.motion_sha256
    artifact["launch_state"]["position_w_m"][0] = 2.9
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(GATE.FittedGateError, match="does not match manifest"):
        GATE.validate_launch_source_artifact(
            path=artifact_path,
            expected_sha256=_sha(artifact_path),
            source=launch["source"],
            action=action,
            launch=launch,
        )
    artifact["launch_state"] = json.loads(json.dumps(launch))
    artifact["recording_sample_time_s"] = 0.21
    upstream_payload = json.loads(upstream.read_text())
    upstream_payload.pop("receipt_payload_sha256")
    upstream_payload["samples"][7]["sample_time_s"] = 0.21
    upstream_payload["receipt_payload_sha256"] = hashlib.sha256(
        GATE.native_diag.canonical_json_bytes(upstream_payload)
    ).hexdigest()
    upstream.write_text(json.dumps(upstream_payload))
    artifact["upstream_evidence_sha256"] = _sha(upstream)
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(GATE.FittedGateError, match="bind the launch state"):
        GATE.validate_launch_source_artifact(
            path=artifact_path,
            expected_sha256=_sha(artifact_path),
            source=launch["source"],
            action=action,
            launch=launch,
        )


def test_recorded_position_venue_fit_source_separates_contact_and_birth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(GATE.native_diag, "REPO_ROOT", tmp_path)
    action = GATE.native_diag.ActionSpec(
        action_id="chingmu_fixture",
        action_uid=321,
        motion_path=tmp_path / "motion.npz",
        motion_sha256="2" * 64,
        strike_phase=0.5,
        t_hit_s=1.0,
        t_cycle_s=2.0,
        racket_speed_mps=3.0,
        reaction_margin_s=0.1,
        mount_normal_sign=-1,
        ball_profile={},
    )
    source = GATE.RECORDED_POSITION_VENUE_FIT_ZERO_SPIN_SOURCE
    launch = {
        "source": source,
        "activation_time_s": 0.2,
        "position_w_m": [3.2, 0.1, 1.4],
        "velocity_w_mps": [-2.4, -0.1, -0.4],
        "spin_w_radps": [0.0, 0.0, 0.0],
        "required_incoming_table_bounces": 1,
    }
    contact_position = [0.6, -0.2, 1.1]
    contact_velocity = [-3.1, 0.2, -0.6]
    venue = tmp_path / "venue.yaml"
    fit_solver = tmp_path / "fit_solver.py"
    birth_solver = tmp_path / "birth_solver.py"
    venue.write_text("venue: fixture\n")
    fit_solver.write_text("FIT = 1\n")
    birth_solver.write_text("BIRTH = 1\n")
    fit_input = {
        "recorded_contact_position_w_m": contact_position,
        "contact_time_s": 1.2,
    }
    birth_input = {
        "contact_position_w_m": contact_position,
        "contact_velocity_w_mps": contact_velocity,
        "activation_time_s": launch["activation_time_s"],
    }
    provenance = {
        "position": "recorded",
        "velocity": "venue_fit_not_measured",
        "spin": "assumed_zero_not_measured",
        "measured_velocity_used": False,
        "measured_spin_used": False,
    }
    units = {
        "position": "m",
        "velocity": "m/s",
        "spin": "rad/s",
        "time": "s",
    }
    recorded_sample = {
        "sample_index": 12,
        "sample_time_s": action.t_hit_s,
        "position_w_m": contact_position,
    }
    venue_fit = {
        "status": "PASS",
        "contact_velocity_w_mps": contact_velocity,
        "target_contact_position_w_m": contact_position,
        "target_contact_time_s": 1.2,
        "venue_yaml_path": venue.name,
        "venue_yaml_sha256": _sha(venue),
        "solver_source_path": fit_solver.name,
        "solver_source_sha256": _sha(fit_solver),
        "fit_input": fit_input,
        "fit_input_sha256": hashlib.sha256(
            GATE.native_diag.canonical_json_bytes(fit_input)
        ).hexdigest(),
    }
    birth_solution = {
        "status": "PASS",
        "activation_time_s": launch["activation_time_s"],
        "position_w_m": launch["position_w_m"],
        "velocity_w_mps": launch["velocity_w_mps"],
        "required_incoming_table_bounces": 1,
        "solver_source_path": birth_solver.name,
        "solver_source_sha256": _sha(birth_solver),
        "solver_input": birth_input,
        "solver_input_sha256": hashlib.sha256(
            GATE.native_diag.canonical_json_bytes(birth_input)
        ).hexdigest(),
    }
    spin_assumption = {
        "source": "assumed_zero_not_measured",
        "spin_w_radps": [0.0, 0.0, 0.0],
    }
    upstream_payload = {
        "schema_version": 1,
        "artifact_type": "recorded_position_venue_fit_ball_state_v1",
        "action_id": action.action_id,
        "action_uid": action.action_uid,
        "motion_sha256": action.motion_sha256,
        "coordinate_frame": "mujoco_world",
        "units": units,
        "recorded_sample": recorded_sample,
        "venue_fit": venue_fit,
        "birth_solution": birth_solution,
        "spin_assumption": spin_assumption,
        "provenance": provenance,
    }
    upstream_payload["receipt_payload_sha256"] = hashlib.sha256(
        GATE.native_diag.canonical_json_bytes(upstream_payload)
    ).hexdigest()
    upstream = tmp_path / "upstream.json"
    upstream.write_text(json.dumps(upstream_payload))
    artifact = {
        "schema_version": 1,
        "artifact_type": source,
        "action_id": action.action_id,
        "action_uid": action.action_uid,
        "motion_sha256": action.motion_sha256,
        "coordinate_frame": "mujoco_world",
        "units": units,
        "authorization": {
            "physical_gate_input_authorized": True,
            "hardware_authorized": False,
        },
        "launch_state": launch,
        "upstream_evidence_path": upstream.name,
        "upstream_evidence_sha256": _sha(upstream),
        "recording_sample_index": 12,
        "recording_sample_time_s": action.t_hit_s,
    }
    artifact_path = tmp_path / "launch.json"
    artifact_path.write_text(json.dumps(artifact))
    receipt = GATE.validate_launch_source_artifact(
        path=artifact_path,
        expected_sha256=_sha(artifact_path),
        source=source,
        action=action,
        launch=launch,
        expected_venue_sha256=_sha(venue),
        expected_recorded_contact_position_w_m=contact_position,
        expected_recording_sample_time_s=action.t_hit_s,
        expected_target_contact_time_s=1.2,
        expected_contact_velocity_w_mps=contact_velocity,
        expected_contact_spin_w_radps=[0.0, 0.0, 0.0],
    )
    assert receipt["recorded_sample"]["position_w_m"] == contact_position
    assert receipt["birth_solution"]["position_w_m"] == launch[
        "position_w_m"
    ]
    assert receipt["provenance"]["measured_velocity_used"] is False
    runtime_launch = GATE.LaunchState(
        source=source,
        activation_time_s=float(launch["activation_time_s"]),
        position_w_m=np.asarray(launch["position_w_m"], np.float64),
        velocity_w_mps=np.asarray(
            launch["velocity_w_mps"], np.float64
        ),
        spin_w_radps=np.asarray(launch["spin_w_radps"], np.float64),
        state_sha256="9" * 64,
        source_artifact_path=artifact_path,
        source_artifact_sha256=_sha(artifact_path),
    )
    runtime_manifest = GATE.PhysicalManifest(
        base=SimpleNamespace(actions=(action,)),
        raw={},
        contract={},
        launches={action.action_id: runtime_launch},
        launch_source_receipts={action.action_id: receipt},
    )
    runtime_pins = GATE.venue_fit_launch_runtime_pins(
        runtime_manifest
    )
    assert [row["role"] for row in runtime_pins] == [
        f"launch_fit_venue:{action.action_id}",
        f"launch_fit_solver_source:{action.action_id}",
        f"launch_birth_solver_source:{action.action_id}",
    ]
    assert {Path(row["path"]) for row in runtime_pins} == {
        venue.resolve(),
        fit_solver.resolve(),
        birth_solver.resolve(),
    }
    raw_input = {
        "schema_version": 1,
        "artifact_type": "recorded_position_venue_fit_input_v1",
        "action_id": action.action_id,
        "action_uid": action.action_uid,
        "motion_sha256": action.motion_sha256,
        "coordinate_frame": "mujoco_world",
        "units": units,
        "recorded_sample": recorded_sample,
        "venue_fit": venue_fit,
        "birth_solution": birth_solution,
        "spin_assumption": spin_assumption,
        "provenance": provenance,
    }
    GATE.validate_recorded_position_venue_fit_raw_input(
        raw_input,
        action=action,
        source_receipt=receipt,
        expected_launch_state=launch,
    )
    for _label, mutation in (
        (
            "recorded contact position",
            lambda row: row["recorded_sample"]["position_w_m"].__setitem__(
                0, row["recorded_sample"]["position_w_m"][0] + 0.01
            ),
        ),
        (
            "fitted contact velocity",
            lambda row: row["venue_fit"][
                "contact_velocity_w_mps"
            ].__setitem__(
                0,
                row["venue_fit"]["contact_velocity_w_mps"][0] + 0.01,
            ),
        ),
        (
            "target contact time",
            lambda row: row["venue_fit"].__setitem__(
                "target_contact_time_s",
                row["venue_fit"]["target_contact_time_s"] + 0.01,
            ),
        ),
        (
            "fit solver SHA",
            lambda row: row["venue_fit"].__setitem__(
                "solver_source_sha256", "f" * 64
            ),
        ),
        (
            "birth solver SHA",
            lambda row: row["birth_solution"].__setitem__(
                "solver_source_sha256", "e" * 64
            ),
        ),
        (
            "nonzero assumed spin",
            lambda row: row["spin_assumption"][
                "spin_w_radps"
            ].__setitem__(2, 1.0),
        ),
    ):
        drifted = json.loads(json.dumps(raw_input))
        mutation(drifted)
        with pytest.raises(
            GATE.FittedGateError,
            match="schema/provenance is not exact",
        ):
            GATE.validate_recorded_position_venue_fit_raw_input(
                drifted,
                action=action,
                source_receipt=receipt,
                expected_launch_state=launch,
            )
    forged_raw = dict(raw_input)
    forged_raw["measured_spin_w_radps"] = [0.0, 0.0, 0.0]
    with pytest.raises(
        GATE.FittedGateError, match="schema/provenance is not exact"
    ):
        GATE.validate_recorded_position_venue_fit_raw_input(
            forged_raw,
            action=action,
            source_receipt=receipt,
            expected_launch_state=launch,
        )

    artifact["measured_velocity_w_mps"] = contact_velocity
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(
        GATE.FittedGateError, match="artifact key set is not exact"
    ):
        GATE.validate_launch_source_artifact(
            path=artifact_path,
            expected_sha256=_sha(artifact_path),
            source=source,
            action=action,
            launch=launch,
            expected_venue_sha256=_sha(venue),
            expected_recorded_contact_position_w_m=contact_position,
            expected_recording_sample_time_s=action.t_hit_s,
            expected_target_contact_time_s=1.2,
            expected_contact_velocity_w_mps=contact_velocity,
            expected_contact_spin_w_radps=[0.0, 0.0, 0.0],
        )


def test_independent_launch_trust_root_binds_raw_capture_and_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(GATE, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(GATE.native_diag, "REPO_ROOT", tmp_path)
    action = GATE.native_diag.ActionSpec(
        action_id="bh_loop_c",
        action_uid=123,
        motion_path=tmp_path / "motion.npz",
        motion_sha256="1" * 64,
        strike_phase=0.5,
        t_hit_s=1.0,
        t_cycle_s=2.0,
        racket_speed_mps=3.0,
        reaction_margin_s=0.1,
        mount_normal_sign=-1,
        ball_profile={},
    )
    launch_row = {
        "source": "recorded_pre_hit_state_v1",
        "activation_time_s": 0.2,
        "position_w_m": [3.0, 0.0, 1.2],
        "velocity_w_mps": [-3.0, 0.0, -0.2],
        "spin_w_radps": [0.0, 0.0, 0.0],
        "required_incoming_table_bounces": 1,
    }
    upstream = tmp_path / "upstream.json"
    upstream.write_text("{}")
    raw_capture = tmp_path / "capture.json"
    raw_capture.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "recorded_ball_capture_v1",
                "action_id": action.action_id,
                "action_uid": action.action_uid,
                "motion_sha256": action.motion_sha256,
                "coordinate_frame": "mujoco_world",
                "units": {
                    "position": "m",
                    "velocity": "m/s",
                    "spin": "rad/s",
                    "time": "s",
                },
                "samples": [
                    {
                        "sample_time_s": launch_row["activation_time_s"],
                        "position_w_m": launch_row["position_w_m"],
                        "velocity_w_mps": launch_row["velocity_w_mps"],
                        "spin_w_radps": launch_row["spin_w_radps"],
                    }
                ],
            }
        )
    )
    validator = tmp_path / "validator.py"
    validator.write_text("VALIDATOR = 1\n")
    validator_bytes = validator.read_bytes()
    monkeypatch.setattr(
        GATE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=validator_bytes),
    )
    launch = GATE.LaunchState(
        source=launch_row["source"],
        activation_time_s=launch_row["activation_time_s"],
        position_w_m=np.asarray(launch_row["position_w_m"]),
        velocity_w_mps=np.asarray(launch_row["velocity_w_mps"]),
        spin_w_radps=np.asarray(launch_row["spin_w_radps"]),
        state_sha256="2" * 64,
        source_artifact_path=tmp_path / "outer.json",
        source_artifact_sha256="3" * 64,
    )
    base = SimpleNamespace(
        action_order=(action.action_id,),
        actions=(action,),
    )
    manifest = GATE.PhysicalManifest(
        base=base,
        raw={
            "actions": [
                {
                    "action_id": action.action_id,
                    "physical_ball_launch": launch_row,
                }
            ]
        },
        contract={},
        launches={action.action_id: launch},
        launch_source_receipts={
            action.action_id: {
                "recording_sample_index": 0,
                "upstream_evidence": {
                    "path": str(upstream.resolve()),
                    "sha256": _sha(upstream),
                },
            }
        },
    )
    validator_subset_sha = hashlib.sha256(
        GATE.native_diag.canonical_json_bytes(
            [
                {
                    "action_id": action.action_id,
                    "repo_path": validator.name,
                    "sha256": _sha(validator),
                }
            ]
        )
    ).hexdigest()
    root_payload = {
        "schema_version": 1,
        "artifact_type": (
            "pre_registered_launch_evidence_trust_root_v1"
        ),
        "manifest_sha256": "9" * 64,
        "commit_binding": {
            "schema_version": 1,
            "authority": (
                "external_preexec_exact_commit_subset_blob_map_v1"
            ),
            "embedded_commit": False,
            "validator_subset_blob_map_sha256": (
                validator_subset_sha
            ),
        },
        "action_order": [action.action_id],
        "authorization": {
            "physical_gate_input_authorized": True,
            "hardware_authorized": False,
        },
        "pre_registration": {
            "registered_before_gate_run": True,
            "decision_id": "fixture-decision",
            "human_dri": "Fixture Human",
        },
        "entries": [
            {
                "action_id": action.action_id,
                "action_uid": action.action_uid,
                "motion_sha256": action.motion_sha256,
                "source": launch.source,
                "source_artifact_sha256": (
                    launch.source_artifact_sha256
                ),
                "upstream_evidence_path": upstream.name,
                "upstream_evidence_sha256": _sha(upstream),
                "raw_input_path": raw_capture.name,
                "raw_input_sha256": _sha(raw_capture),
                "validator_source_path": validator.name,
                "validator_source_sha256": _sha(validator),
            }
        ],
    }
    root_payload["receipt_payload_sha256"] = hashlib.sha256(
        GATE.native_diag.canonical_json_bytes(root_payload)
    ).hexdigest()
    root = tmp_path / "trust.json"
    root.write_text(json.dumps(root_payload))
    receipt, files = GATE.validate_launch_evidence_trust_root(
        path=root,
        expected_sha256=_sha(root),
        manifest_sha256="9" * 64,
        expected_commit="a" * 40,
        manifest=manifest,
    )
    assert receipt["entries"][0]["raw_input_sha256"] == _sha(
        raw_capture
    )
    assert receipt["commit_binding"]["external_code_commit"] == "a" * 40
    assert {row["role"] for row in files} == {
        "launch_trust_root",
        f"launch_raw_input:{action.action_id}",
        f"launch_validator:{action.action_id}",
    }


def test_action_uid_derivation_matches_checked_in_canonical_rows():
    raw = json.loads(
        (
            REPO / "configs/action_ball_n5_nomove_f20_20260728.json"
        ).read_text()
    )
    for row in raw["actions"]:
        assert GATE.derive_action_uid(
            row["action_id"], row["family"], row["motion_sha256"]
        ) == row["action_uid"]


@pytest.mark.parametrize("sign", (1, -1))
def test_official_selected_face_mesh_has_finite_outer_triangles(sign: int):
    mesh = GATE.load_binary_stl_face(sign)
    assert mesh.triangles_xz_m.shape == (31, 3, 2)
    assert mesh.sha256 == _sha(mesh.path)
    center = GATE.racket_geometry.face_center_from_site_local(sign)[[0, 2]]
    assert GATE.point_in_triangles(center, mesh.triangles_xz_m) >= 0
    assert GATE.point_in_triangles([0.5, 0.5], mesh.triangles_xz_m) == -1


def test_swept_selected_red_face_hit_uses_ball_radius_and_finite_footprint():
    mesh = GATE.load_binary_stl_face(1)
    face = _face_state(1)
    radius = 0.02
    hit = GATE.swept_selected_face_intersection(
        ball_start_m=[0.0, 0.04, 0.0],
        ball_end_m=[0.0, 0.01, 0.0],
        ball_velocity_start_mps=[0.0, -3.0, 0.0],
        ball_velocity_end_mps=[0.0, -3.0, 0.0],
        face_start=face,
        face_end=face,
        mesh=mesh,
        ball_radius_m=radius,
    )
    assert hit is not None
    assert hit.ball_center_m[1] == pytest.approx(radius, abs=1.0e-10)
    assert hit.relative_normal_speed_mps == pytest.approx(-3.0)
    assert hit.triangle_index >= 0

    miss = GATE.swept_selected_face_intersection(
        ball_start_m=[0.3, 0.04, 0.3],
        ball_end_m=[0.3, 0.01, 0.3],
        ball_velocity_start_mps=[0.0, -3.0, 0.0],
        ball_velocity_end_mps=[0.0, -3.0, 0.0],
        face_start=face,
        face_end=face,
        mesh=mesh,
        ball_radius_m=radius,
    )
    assert miss is None


def test_swept_selected_face_rejects_away_motion_and_wrong_side():
    mesh = GATE.load_binary_stl_face(1)
    face = _face_state(1)
    assert (
        GATE.swept_selected_face_intersection(
            ball_start_m=[0.0, 0.01, 0.0],
            ball_end_m=[0.0, 0.04, 0.0],
            ball_velocity_start_mps=[0.0, 3.0, 0.0],
            ball_velocity_end_mps=[0.0, 3.0, 0.0],
            face_start=face,
            face_end=face,
            mesh=mesh,
            ball_radius_m=0.02,
        )
        is None
    )


def test_face_contact_point_velocity_includes_omega_cross_r():
    mesh = GATE.load_binary_stl_face(1)
    face = _face_state(1, angular=(0.0, 0.0, 10.0))
    hit = GATE.swept_selected_face_intersection(
        ball_start_m=[0.05, 0.04, 0.0],
        ball_end_m=[0.05, 0.01, 0.0],
        ball_velocity_start_mps=[0.0, -3.0, 0.0],
        ball_velocity_end_mps=[0.0, -3.0, 0.0],
        face_start=face,
        face_end=face,
        mesh=mesh,
        ball_radius_m=0.02,
    )
    assert hit is not None
    expected = np.cross(
        np.array([0.0, 0.0, 10.0]), hit.face_point_m - face.site_position_m
    )
    assert np.allclose(hit.face_point_velocity_mps, expected)


def test_table_face_toi_arbitration_is_earliest_and_ties_fail_closed():
    paddle = SimpleNamespace(alpha=0.7)
    table = (0.3, np.asarray((2.0, 0.0, 0.78)))
    choice, ambiguous = GATE.arbitrate_table_face_toi(
        paddle_hit=paddle,
        table_hit=table,
        segment_duration_s=0.001,
    )
    assert ambiguous is False
    assert choice is not None and choice[1] == "table"
    near_tie = (
        paddle.alpha
        + 0.5 * GATE.FORMAL_EVENT_TIME_GUARD_S / 0.001,
        table[1],
    )
    choice, ambiguous = GATE.arbitrate_table_face_toi(
        paddle_hit=paddle,
        table_hit=near_tie,
        segment_duration_s=0.001,
    )
    assert choice is None
    assert ambiguous is True


def test_fitted_contact_delegates_to_pinned_venue_contact_model():
    assert _sha(GATE.CONTACT_MODEL_PATH) == GATE.CONTACT_MODEL_SHA256
    result = GATE.fitted_contact(
        [-3.0, 0.2, -0.5],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
        e_eff=0.7,
        a_t=0.52,
        b_t=0.0,
        mu=0.5,
    )
    expected = GATE.contact_model.predict_contact(
        np.array([[-3.0, 0.2, -0.5]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 5.0]]),
        0.7,
        0.52,
        0.0,
        0.5,
    )
    assert np.array_equal(result["velocity_plus_mps"], expected["v_plus"][0])
    assert np.array_equal(result["spin_plus_radps"], expected["omega_plus"][0])


def test_table_sweep_requires_descending_crossing_inside_real_footprint():
    hit = GATE.swept_table_crossing(
        [2.0, 0.1, 0.80],
        [2.0, 0.1, 0.77],
        [0.0, 0.0, -3.0],
        center_surface_z_m=0.78,
        near_x_m=0.5,
        far_x_m=3.24,
        half_width_m=0.7625,
    )
    assert hit is not None
    assert hit[1][2] == pytest.approx(0.78)
    assert (
        GATE.swept_table_crossing(
            [2.0, 1.0, 0.80],
            [2.0, 1.0, 0.77],
            [0.0, 0.0, -3.0],
            center_surface_z_m=0.78,
            near_x_m=0.5,
            far_x_m=3.24,
            half_width_m=0.7625,
        )
        is None
    )


def test_chronological_table_impulse_erodes_footprint_by_ball_radius():
    venue = _venue()
    pins = json.loads(
        (
            REPO / "configs/action_ball_profile_pins_20260728.json"
        ).read_text()
    )
    profile = pins["physics_payload"]["geometry_and_grading"]
    aabbs = GATE._obstacle_aabbs(
        GATE.table_scene.obstacle_geometry()
    )
    action = GATE.native_diag.ActionSpec(
        action_id="fixture",
        action_uid=1,
        motion_path=Path("fixture.npz"),
        motion_sha256="1" * 64,
        strike_phase=0.5,
        t_hit_s=1.0,
        t_cycle_s=2.0,
        racket_speed_mps=3.0,
        reaction_margin_s=0.1,
        mount_normal_sign=1,
        ball_profile={},
    )
    face = _face_state(1)
    mesh = GATE.load_binary_stl_face(1)

    near_edge_events = GATE.FittedEvents()
    result = GATE.process_surface_events_chronologically(
        p0=np.asarray((0.51, 0.0, 0.79)),
        p1=np.asarray((0.51, 0.0, 0.77)),
        v0=np.asarray((0.0, 0.0, -2.0)),
        v1=np.asarray((0.0, 0.0, -2.0)),
        w0=np.zeros(3),
        w1=np.zeros(3),
        time_s=1.0,
        dt=0.01,
        face_before=face,
        face_after=face,
        face_mesh=mesh,
        action=action,
        venue=venue,
        profile=profile,
        aabbs=aabbs,
        events=near_edge_events,
        returned=True,
    )
    assert result[1][2] < 0.0
    assert near_edge_events.table_contacts == []

    interior_events = GATE.FittedEvents()
    GATE.process_surface_events_chronologically(
        p0=np.asarray((0.60, 0.0, 0.79)),
        p1=np.asarray((0.60, 0.0, 0.77)),
        v0=np.asarray((0.0, 0.0, -2.0)),
        v1=np.asarray((0.0, 0.0, -2.0)),
        w0=np.zeros(3),
        w1=np.zeros(3),
        time_s=1.0,
        dt=0.01,
        face_before=face,
        face_after=face,
        face_mesh=mesh,
        action=action,
        venue=venue,
        profile=profile,
        aabbs=aabbs,
        events=interior_events,
        returned=True,
    )
    assert len(interior_events.table_contacts) == 1
    assert interior_events.table_contacts[0][
        "eroded_footprint_margin_m"
    ] == pytest.approx(
        venue.ball_radius + GATE.FORMAL_SHADOW_CLEARANCE_GUARD_M
    )


def test_analytic_ball_referee_does_not_see_robot_only_keepout():
    venue = _venue()
    pins = json.loads(
        (
            REPO / "configs/action_ball_profile_pins_20260728.json"
        ).read_text()
    )
    profile = pins["physics_payload"]["geometry_and_grading"]
    rows = GATE.table_scene.action_ball_policy_obstacle_geometry()
    aabbs = GATE._obstacle_aabbs(rows)
    assert tuple(aabbs) == GATE.table_scene.OBSTACLE_NAMES
    assert (
        GATE.table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME not in aabbs
    )
    action = GATE.native_diag.ActionSpec(
        action_id="keepout_referee_fixture",
        action_uid=1,
        motion_path=Path("fixture.npz"),
        motion_sha256="1" * 64,
        strike_phase=0.5,
        t_hit_s=1.0,
        t_cycle_s=2.0,
        racket_speed_mps=3.0,
        reaction_margin_s=0.1,
        mount_normal_sign=1,
        ball_profile={},
    )
    face = _face_state(1, site=(0.0, 2.0, 2.0))
    events = GATE.FittedEvents()
    _p1, _v1, _w1, returned, segments = (
        GATE.process_surface_events_chronologically(
            p0=np.asarray((0.3, 0.0, 0.35)),
            p1=np.asarray((3.4, 0.0, 0.35)),
            v0=np.asarray((3.1, 0.0, 0.0)),
            v1=np.asarray((3.1, 0.0, 0.0)),
            w0=np.zeros(3),
            w1=np.zeros(3),
            time_s=0.0,
            dt=1.0,
            face_before=face,
            face_after=face,
            face_mesh=GATE.load_binary_stl_face(1),
            action=action,
            venue=venue,
            profile=profile,
            aabbs=aabbs,
            events=events,
            returned=False,
        )
    )
    assert returned is False
    assert segments
    assert events.table_contacts == []
    assert events.ball_net_collision is None
    assert events.net_crossing is None
    assert events.ball_forbidden_contacts == []


def test_shadow_contact_whitelist_binds_time_position_and_finite_surface():
    paddle_events = GATE.FittedEvents(
        paddle_contact={
            "time_s": 1.0,
            "ball_center_m": [0.2, 0.0, 1.0],
            "selected_face_return_normal_w": [1.0, 0.0, 0.0],
            "face_edge_clearance_m": 0.021,
            "required_face_edge_clearance_m": 0.0205,
        }
    )
    assert GATE._matches_authorized_paddle_contact(
        sample_time_s=1.0,
        ball_position_m=np.asarray((0.2, 0.0, 1.0)),
        events=paddle_events,
    )
    assert not GATE._matches_authorized_paddle_contact(
        sample_time_s=1.0,
        ball_position_m=np.asarray((0.21, 0.0, 1.0)),
        events=paddle_events,
    )

    radius = 0.02
    margin = radius + GATE.FORMAL_SHADOW_CLEARANCE_GUARD_M
    table_aabb = (
        np.asarray((0.5, -0.7625, 0.73)),
        np.asarray((3.24, 0.7625, 0.76)),
    )
    table_events = GATE.FittedEvents(
        table_contacts=[
            {
                "time_s": 1.2,
                "ball_center_m": [2.0, 0.0, 0.78],
                "normal_w": [0.0, 0.0, 1.0],
                "returned_before_event": True,
                "eroded_footprint_margin_m": margin,
            }
        ]
    )
    assert GATE._matches_authorized_table_contact(
        sample_time_s=1.2,
        ball_position_m=np.asarray((2.0, 0.0, 0.78)),
        events=table_events,
        table_aabb=table_aabb,
        ball_radius_m=radius,
    )
    table_events.table_contacts[0]["ball_center_m"] = [
        0.5 + 0.5 * margin,
        0.0,
        0.78,
    ]
    assert not GATE._matches_authorized_table_contact(
        sample_time_s=1.2,
        ball_position_m=np.asarray(
            table_events.table_contacts[0]["ball_center_m"]
        ),
        events=table_events,
        table_aabb=table_aabb,
        ball_radius_m=radius,
    )


def test_shadow_sampling_adapts_to_certified_robot_surface_path():
    clip = GATE.motion_player.MotionClip(
        path=Path("fixture.npz"),
        fps=1000.0,
        joint_pos=np.asarray([[0.0], [0.01]], dtype=float),
        joint_vel=np.zeros((2, 1), dtype=float),
        body_pos_w=np.zeros((2, 1, 3), dtype=float),
        body_quat_w=np.asarray(
            [[[1.0, 0.0, 0.0, 0.0]]] * 2, dtype=float
        ),
        body_lin_vel_w=np.zeros((2, 1, 3), dtype=float),
        body_ang_vel_w=np.zeros((2, 1, 3), dtype=float),
        has_migration_provenance=False,
        body_lin_vel_point="body_origin",
    )
    geom_bounds = (
        GATE.ShadowGeomMotionBound(
            geom_id=7,
            root_rotation_radius_m=1.0,
            hinge_terms=((0, 1.0),),
            slide_indices=(),
        ),
    )
    alphas, ball_bound, robot_bound = (
        GATE.adaptive_shadow_sample_alphas(
            clip=clip,
            wait_s=0.0,
            start_time_s=0.0,
            duration_s=0.001,
            start_ball_position_m=np.zeros(3),
            end_ball_position_m=np.asarray((0.001, 0.0, 0.0)),
            geom_bounds=geom_bounds,
        )
    )
    assert len(alphas) > 1 + math.ceil(
        0.001 / GATE.FORMAL_SHADOW_MAX_DT_S
    )
    assert ball_bound <= GATE.FORMAL_SHADOW_MAX_BALL_STEP_M
    assert (
        robot_bound
        <= GATE.FORMAL_SHADOW_MAX_ROBOT_SURFACE_STEP_M
    )
    assert (
        ball_bound + robot_bound
        < GATE.FORMAL_SHADOW_CLEARANCE_GUARD_M
    )


def test_shadow_geom_pair_filter_matches_masks_weld_parent_and_exclude():
    model = SimpleNamespace(
        geom_contype=np.asarray([1, 1, 0]),
        geom_conaffinity=np.asarray([1, 1, 0]),
        geom_bodyid=np.asarray([1, 2, 3]),
        body_weldid=np.asarray([0, 1, 2, 3]),
        body_parentid=np.asarray([0, 0, 0, 0]),
        opt=SimpleNamespace(disableflags=0),
        exclude_signature=np.asarray([(1 << 16) + 2]),
        nexclude=1,
    )
    mujoco = SimpleNamespace(
        mjtDisableBit=SimpleNamespace(mjDSBL_FILTERPARENT=1)
    )
    assert not GATE._geom_pair_enabled(mujoco, model, 0, 1)
    assert not GATE._geom_pair_enabled(mujoco, model, 0, 2)
    model.exclude_signature = np.asarray([], dtype=np.int64)
    model.nexclude = 0
    assert GATE._geom_pair_enabled(mujoco, model, 0, 1)


def test_segment_expanded_aabb_detects_net_tunnelling():
    alpha = GATE.segment_expanded_aabb_hit(
        [1.7, 0.0, 0.85],
        [2.0, 0.0, 0.85],
        [1.86, -0.01, 0.76],
        [1.88, 0.01, 0.92],
        0.02,
    )
    assert alpha is not None and 0.0 < alpha < 1.0


def test_scene_geometry_matches_same_profile_used_for_grading():
    venue = _venue()
    pins = json.loads(
        (
            REPO / "configs/action_ball_profile_pins_20260728.json"
        ).read_text()
    )
    profile = {
        "opponent_near_x_m": pins["physics_payload"][
            "geometry_and_grading"
        ]["opponent_near_x_m"],
        "opponent_far_x_m": pins["physics_payload"][
            "geometry_and_grading"
        ]["opponent_far_x_m"],
        "table_surface_z_m": pins["physics_payload"][
            "geometry_and_grading"
        ]["table_surface_z_m"],
        "table_half_width_m": pins["physics_payload"][
            "geometry_and_grading"
        ]["table_half_width_m"],
        "net_x_m": pins["physics_payload"]["geometry_and_grading"][
            "net_x_m"
        ],
        "ball_center_net_top_z_m": pins["physics_payload"][
            "geometry_and_grading"
        ]["ball_center_net_top_z_m"],
    }
    checks = GATE.validate_scene_against_profile(
        GATE.table_scene.obstacle_geometry(), profile, venue
    )
    assert all(
        row["scene"] == pytest.approx(row["profile"], abs=2.0e-9)
        for row in checks.values()
    )


def test_fitted_scene_disables_all_native_ball_contact_and_binds_thin_shell():
    venue = _venue()
    canonical = GATE.native_diag.DEFAULT_MJCF.read_bytes()
    xml, receipt = GATE.assemble_fitted_scene_xml(
        canonical,
        GATE.table_scene.obstacle_geometry(),
        venue,
        0.001,
    )
    text = xml.decode()
    assert f'name="{GATE.BALL_GEOM_NAME}"' in text
    assert 'contype="0"' in text
    assert 'conaffinity="0"' in text
    assert 'gravity="0 0 -9.8100000000000005"' in text
    expected_inertia = (
        venue.inertia_coeff * venue.ball_mass * venue.ball_radius**2
    )
    assert receipt["ball_diagonal_inertia_kg_m2"] == pytest.approx(
        expected_inertia, abs=1.0e-18
    )
    assert receipt["ball_native_contact_disabled"] is True
    assert receipt["gravity_mps2"] == [0.0, 0.0, -9.81]


def test_five_solid_robot_subject_excludes_collision_disabled_visual_geom():
    model = SimpleNamespace(
        ngeom=3,
        geom_contype=np.asarray([0, 1, 0]),
        geom_conaffinity=np.asarray([7, 7, 0]),
        geom_bodyid=np.asarray([0, 1, 2]),
        body_weldid=np.asarray([0, 1, 2]),
        body_parentid=np.asarray([0, 0, 0]),
        opt=SimpleNamespace(disableflags=0),
        exclude_signature=np.asarray([], dtype=np.int64),
        nexclude=0,
    )
    fake_mujoco = SimpleNamespace(
        mjtDisableBit=SimpleNamespace(mjDSBL_FILTERPARENT=1)
    )
    selected = GATE._five_solid_robot_geom_ids(
        fake_mujoco,
        model,
        ball_geom_id=99,
        obstacle_geom_ids=(0,),
    )
    assert selected == (1,)


def _minimal_five_solid_model():
    assert mujoco is not None
    base = b"""<mujoco>
      <option timestep="0.001" gravity="0 0 0"/>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1"
              contype="0" conaffinity="15"/>
        <body name="left_ankle_roll_Link" pos="-2 -0.2 0.049">
          <freejoint name="left_foot_joint"/>
          <geom name="left_ankle_roll_collision" type="sphere" size="0.05"
                mass="1" contype="8" conaffinity="8"/>
        </body>
        <body name="right_ankle_roll_Link" pos="-2 0.2 0.049">
          <freejoint name="right_foot_joint"/>
          <geom name="right_ankle_roll_collision" type="sphere" size="0.05"
                mass="1" contype="8" conaffinity="8"/>
        </body>
        <body name="left_hand_Link" pos="-2 0 1">
          <freejoint name="left_hand_joint"/>
          <geom name="left_hand_collision" type="sphere" size="0.05"
                mass="1" contype="8" conaffinity="8"/>
        </body>
        <body name="physics_robot" pos="-1 0 0.35">
          <freejoint name="physics_robot_joint"/>
          <geom name="physics_robot_geom" type="sphere" size="0.05"
                mass="1" contype="1" conaffinity="7"/>
        </body>
        <body name="visual_robot" pos="1 0 0.35">
          <freejoint name="visual_robot_joint"/>
          <geom name="visual_robot_geom" type="sphere" size="0.05"
                mass="1" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>"""
    venue = _venue()
    rows = GATE.table_scene.action_ball_policy_obstacle_geometry()
    four, _scene = GATE.assemble_fitted_scene_xml(
        base, rows, venue, 0.001
    )
    five = GATE.table_scene.append_action_ball_policy_keepout_xml(
        four, rows, collidable=True
    )
    model = mujoco.MjModel.from_xml_string(five.decode("utf-8"))
    contract = GATE.table_scene.action_ball_policy_geometry_contract(
        rows
    )
    return model, rows, contract, hashlib.sha256(five).hexdigest()


@requires_mujoco
def test_compiled_teacher_scene_binds_exact_five_solids_and_ball_filter():
    model, rows, contract, xml_sha = _minimal_five_solid_model()
    receipt = GATE.validate_compiled_obstacles(
        mujoco,
        model,
        rows,
        contract,
        assembled_xml_sha256=xml_sha,
    )
    assert tuple(
        row["name"] for row in receipt["compiled_obstacles"]
    ) == GATE.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    assert receipt["five_solid_geometry_sha256"] == contract["sha256"]
    assert receipt["assembled_xml_sha256"] == xml_sha
    assert receipt["physics_enabled_robot_geom_count"] == 1
    assert receipt["ball_keepout_native_pair_enabled"] is False
    assert receipt["ball_keepout_analytic_surface_enabled"] is False

    ball = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, GATE.BALL_GEOM_NAME
    )
    model.geom_contype[ball] = 1
    with pytest.raises(
        GATE.FittedGateError,
        match="robot-only keepout can affect the fitted ball",
    ):
        GATE.validate_compiled_obstacles(
            mujoco,
            model,
            rows,
            contract,
            assembled_xml_sha256=xml_sha,
        )


@requires_mujoco
def test_teacher_ground_guard_allows_only_exact_feet_with_small_penetration():
    model, _rows, _contract, _xml_sha = _minimal_five_solid_model()
    data = mujoco.MjData(model)
    ball = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, GATE.BALL_GEOM_NAME
    )
    obstacle_ids = {
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        ): name
        for name in GATE.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    }
    ground = GATE.build_ground_contact_contract(
        mujoco, model, ball_geom_id=ball
    )
    receipt = GATE.ground_contact_contract_receipt(
        mujoco, model, ground
    )
    assert receipt["legal_foot_body_names"] == list(
        GATE.LEGAL_FOOT_BODY_NAMES
    )
    assert receipt["legal_foot_geom_names"] == [
        "left_ankle_roll_collision",
        "right_ankle_roll_collision",
    ]
    mujoco.mj_forward(model, data)
    events = GATE.FittedEvents()
    GATE._scan_robot_contacts(
        mujoco,
        model,
        data,
        ball,
        obstacle_ids,
        events,
        0.0,
        ground,
    )
    assert events.legal_foot_support_contact_count > 0
    assert events.foot_floor_penetration_violation_count == 0
    assert events.nonfoot_ground_contact_violation_count == 0
    assert (
        events.ground_max_foot_penetration_m
        < GATE.FOOT_FLOOR_PENETRATION_TOLERANCE_M
    )


@requires_mujoco
def test_teacher_ground_guard_catches_nonfoot_and_excessive_foot_contact():
    model, _rows, _contract, _xml_sha = _minimal_five_solid_model()
    data = mujoco.MjData(model)
    ball = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, GATE.BALL_GEOM_NAME
    )
    obstacle_ids = {
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        ): name
        for name in GATE.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    }
    ground = GATE.build_ground_contact_contract(
        mujoco, model, ball_geom_id=ball
    )
    hand_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left_hand_joint"
    )
    hand_qadr = int(model.jnt_qposadr[hand_joint])
    data.qpos[hand_qadr + 2] = 0.02
    mujoco.mj_forward(model, data)
    nonfoot = GATE.FittedEvents()
    GATE._scan_robot_contacts(
        mujoco,
        model,
        data,
        ball,
        obstacle_ids,
        nonfoot,
        0.0,
        ground,
    )
    assert nonfoot.nonfoot_ground_contact_violation_count > 0
    assert {
        row["robot_geom"]
        for row in nonfoot.ground_contact_violations
        if not row["robot_body_is_legal_foot"]
    } == {"left_hand_collision"}

    data = mujoco.MjData(model)
    foot_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left_foot_joint"
    )
    foot_qadr = int(model.jnt_qposadr[foot_joint])
    data.qpos[foot_qadr + 2] = 0.04
    mujoco.mj_forward(model, data)
    excessive_foot = GATE.FittedEvents()
    GATE._scan_robot_contacts(
        mujoco,
        model,
        data,
        ball,
        obstacle_ids,
        excessive_foot,
        0.0,
        ground,
    )
    assert (
        excessive_foot.foot_floor_penetration_violation_count > 0
    )


@requires_mujoco
def test_teacher_ground_contract_rejects_collision_geom_filtered_from_floor():
    model, _rows, _contract, _xml_sha = _minimal_five_solid_model()
    ball = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, GATE.BALL_GEOM_NAME
    )
    hand = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_collision"
    )
    model.geom_contype[hand] = 16
    model.geom_conaffinity[hand] = 16
    with pytest.raises(
        GATE.FittedGateError,
        match="collision-enabled robot geom is filtered from the floor",
    ):
        GATE.build_ground_contact_contract(
            mujoco, model, ball_geom_id=ball
        )


def test_continuous_floor_interval_lower_bound_is_finite_and_conservative():
    certified = GATE.continuous_floor_interval_lower_bound_m(
        distance_lower_m=0.004,
        distance_midpoint_m=-0.003,
        distance_upper_m=0.005,
        surface_bound_lower_to_mid_m=0.008,
        surface_bound_mid_to_upper_m=0.009,
    )
    assert certified == pytest.approx(-0.004, abs=1.0e-15)
    with pytest.raises(GATE.FittedGateError, match="NaN/Inf"):
        GATE.continuous_floor_interval_lower_bound_m(
            distance_lower_m=math.nan,
            distance_midpoint_m=0.0,
            distance_upper_m=0.0,
            surface_bound_lower_to_mid_m=0.0,
            surface_bound_mid_to_upper_m=0.0,
        )
    with pytest.raises(GATE.FittedGateError, match="nonnegative"):
        GATE.continuous_floor_interval_lower_bound_m(
            distance_lower_m=0.0,
            distance_midpoint_m=0.0,
            distance_upper_m=0.0,
            surface_bound_lower_to_mid_m=-1.0e-6,
            surface_bound_mid_to_upper_m=0.0,
        )


def _continuous_ground_fixture():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.0005" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="10 10 0.1"
                  contype="8" conaffinity="8"/>
            <body name="pelvis_link" pos="0 0 1">
              <freejoint name="root"/>
              <geom name="pelvis_collision" type="sphere" size="0.03"
                    mass="5" contype="8" conaffinity="8"/>
              <body name="left_ankle_roll_Link" pos="0 0.1 -0.975">
                <geom name="left_ankle_roll_collision" type="box"
                      size="0.05 0.03 0.025" mass="0.1"
                      contype="8" conaffinity="8"/>
              </body>
              <body name="right_ankle_roll_Link" pos="0 -0.1 -0.975">
                <geom name="right_ankle_roll_collision" type="box"
                      size="0.05 0.03 0.025" mass="0.1"
                      contype="8" conaffinity="8"/>
              </body>
              <body name="left_hand_Link" pos="0 0 -0.995">
                <joint name="hand_hinge" type="hinge" axis="0 1 0"/>
                <geom name="left_hand_collision" type="capsule"
                      fromto="0 0 0 0.01 0 0" size="0.002"
                      mass="0.1" contype="8" conaffinity="8"/>
              </body>
            </body>
            <body name="fitted_ball" pos="0 0 3">
              <freejoint name="fitted_ball_joint"/>
              <geom name="physical_ball" type="sphere" size="0.02"
                    mass="0.0027" contype="0" conaffinity="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    root_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root"
    )
    hand_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "hand_hinge"
    )
    pelvis = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link"
    )
    binding = GATE.motion_player.ModelBinding(
        root_qpos_adr=int(model.jnt_qposadr[root_joint]),
        root_dof_adr=int(model.jnt_dofadr[root_joint]),
        joint_qpos_adrs=np.asarray(
            [model.jnt_qposadr[hand_joint]], np.int64
        ),
        joint_dof_adrs=np.asarray(
            [model.jnt_dofadr[hand_joint]], np.int64
        ),
        body_ids=np.asarray([pelvis], np.int64),
        racket_site_id=-1,
        racket_site_body_id=-1,
        racket_site_body_column=-1,
    )
    clip = GATE.motion_player.MotionClip(
        path=Path("continuous_ground_fixture.npz"),
        fps=2000.0,
        joint_pos=np.asarray(
            [[0.0], [math.pi / 2.0], [0.0]], np.float64
        ),
        joint_vel=np.zeros((3, 1), np.float64),
        body_pos_w=np.asarray(
            [[[0.0, 0.0, 1.0]]] * 3, np.float64
        ),
        body_quat_w=np.asarray(
            [[[1.0, 0.0, 0.0, 0.0]]] * 3, np.float64
        ),
        body_lin_vel_w=np.zeros((3, 1, 3), np.float64),
        body_ang_vel_w=np.zeros((3, 1, 3), np.float64),
        has_migration_provenance=False,
        body_lin_vel_point="link_origin",
    )
    ball = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "physical_ball"
    )
    ground = GATE.build_ground_contact_contract(
        mujoco, model, ball_geom_id=ball
    )
    bounds = GATE.build_shadow_kinematic_bounds(
        mujoco=mujoco,
        model=model,
        binding=binding,
        robot_geom_ids=(
            tuple(ground.foot_geom_ids)
            + tuple(ground.nonfoot_robot_geom_ids)
        ),
    )
    return model, binding, clip, ground, bounds


@requires_mujoco
def test_continuous_ground_probe_catches_safe_endpoints_bad_motion_knot():
    model, binding, clip, ground, bounds = _continuous_ground_fixture()
    probe_data = mujoco.MjData(model)
    hand_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_collision"
    )
    floor_geom = int(ground.floor_geom_id)

    direct_distances = []
    for time_s in (0.0, 0.0005, 0.001):
        state = GATE.retimed_teacher_state(
            clip,
            world_time_s=time_s,
            pre_swing_wait_s=0.0,
            teacher_rate=1.0,
        )
        GATE.native_diag._set_teacher_state(
            mujoco, model, probe_data, binding, state
        )
        direct_distances.append(
            float(
                mujoco.mj_geomDistance(
                    model,
                    probe_data,
                    floor_geom,
                    hand_geom,
                    GATE.FORMAL_GROUND_DISTANCE_QUERY_CAP_M,
                    None,
                )
            )
        )
    assert direct_distances[0] > 0.0
    assert direct_distances[1] < 0.0
    assert direct_distances[2] > 0.0

    events = GATE.FittedEvents()
    GATE.run_continuous_ground_probe(
        mujoco=mujoco,
        model=model,
        probe_data=probe_data,
        binding=binding,
        clip=clip,
        wait_s=0.0,
        teacher_rate=1.0,
        start_time_s=0.0,
        duration_s=0.001,
        ground_contract=ground,
        ground_geom_motion_bounds=bounds,
        events=events,
    )
    assert events.ground_shadow_certificate_failure is None
    assert events.ground_shadow_certificate_intervals > 0
    assert events.ground_shadow_probe_samples > 3
    assert events.ground_shadow_covered_duration_s == pytest.approx(
        0.001, abs=1.0e-15
    )
    assert {
        row["robot_geom"]
        for row in events.shadow_nonfoot_ground_near_contacts
    } == {"left_hand_collision"}
    assert not events.shadow_foot_floor_penetration_violations
    assert (
        events.ground_shadow_min_nonfoot_lower_bound_m
        < GATE.FORMAL_NONFOOT_GROUND_CLEARANCE_GUARD_M
    )


@requires_mujoco
def test_teacher_safety_ignores_visual_but_catches_physics_keepout_contact():
    model, rows, _contract, _xml_sha = _minimal_five_solid_model()
    data = mujoco.MjData(model)
    ball = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, GATE.BALL_GEOM_NAME
    )
    obstacle_ids = {
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        ): name
        for name in GATE.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    }
    physical = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "physics_robot_geom"
    )
    visual = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "visual_robot_geom"
    )
    selected = GATE._five_solid_robot_geom_ids(
        mujoco,
        model,
        ball_geom_id=ball,
        obstacle_geom_ids=tuple(obstacle_ids),
    )
    assert selected == (physical,)
    assert visual not in selected

    mujoco.mj_forward(model, data)
    events = GATE.FittedEvents()
    GATE._scan_robot_contacts(
        mujoco, model, data, ball, obstacle_ids, events, 0.0
    )
    assert events.robot_obstacle_contact_count == 0

    joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "physics_robot_joint",
    )
    qadr = int(model.jnt_qposadr[joint])
    data.qpos[qadr : qadr + 3] = (1.0, 0.0, 0.35)
    mujoco.mj_forward(model, data)
    GATE._scan_robot_contacts(
        mujoco, model, data, ball, obstacle_ids, events, 0.001
    )
    assert events.robot_obstacle_contact_count > 0
    assert (
        events.robot_obstacle_contact_per_obstacle[
            GATE.table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME
        ]
        > 0
    )


@requires_mujoco
def test_teacher_sweep_catches_keepout_tunnel_with_clean_endpoints():
    model, rows, _contract, _xml_sha = _minimal_five_solid_model()
    data = mujoco.MjData(model)
    ball = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, GATE.BALL_GEOM_NAME
    )
    physical = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "physics_robot_geom"
    )
    obstacle_ids = {
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        ): name
        for name in GATE.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    }
    selected = GATE._five_solid_robot_geom_ids(
        mujoco,
        model,
        ball_geom_id=ball,
        obstacle_geom_ids=tuple(obstacle_ids),
    )
    assert selected == (physical,)
    joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "physics_robot_joint",
    )
    qadr = int(model.jnt_qposadr[joint])
    data.qpos[qadr : qadr + 3] = (0.0, 0.0, 0.35)
    mujoco.mj_forward(model, data)
    centers_before = np.asarray(data.geom_xpos[list(selected)]).copy()
    before_events = GATE.FittedEvents()
    GATE._scan_robot_contacts(
        mujoco, model, data, ball, obstacle_ids, before_events, 0.0
    )
    assert before_events.robot_obstacle_contact_count == 0

    data.qpos[qadr : qadr + 3] = (4.0, 0.0, 0.35)
    mujoco.mj_forward(model, data)
    after_events = GATE.FittedEvents()
    GATE._scan_robot_contacts(
        mujoco, model, data, ball, obstacle_ids, after_events, 0.001
    )
    assert after_events.robot_obstacle_contact_count == 0
    swept = GATE.scan_five_solid_robot_sweep(
        mujoco=mujoco,
        model=model,
        data=data,
        robot_geom_ids=selected,
        centers_before=centers_before,
        obstacle_aabbs=(
            GATE.table_scene.action_ball_policy_obstacle_aabbs(rows)
        ),
    )
    assert swept["hit_count"] > 0
    assert (
        swept["per_obstacle"][
            GATE.table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME
        ]
        > 0
    )
    assert swept["per_obstacle"]["motion_table_top"] == 0


@requires_mujoco
def test_teacher_fitted_ball_passes_keepout_without_native_contact():
    model, _rows, _contract, _xml_sha = _minimal_five_solid_model()
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    for joint_name, position in (
        ("physics_robot_joint", (-2.0, 0.0, 2.0)),
        ("visual_robot_joint", (-2.0, 1.0, 2.0)),
    ):
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        qadr = int(model.jnt_qposadr[joint])
        data.qpos[qadr : qadr + 3] = position
    ball_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, GATE.BALL_JOINT_NAME
    )
    ball_qadr = int(model.jnt_qposadr[ball_joint])
    ball_dadr = int(model.jnt_dofadr[ball_joint])
    data.qpos[ball_qadr : ball_qadr + 3] = (0.30, 0.0, 0.35)
    data.qvel[ball_dadr : ball_dadr + 3] = (4.0, 0.0, 0.0)
    ball_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, GATE.BALL_GEOM_NAME
    )
    for _ in range(500):
        mujoco.mj_step(model, data)
        assert all(
            ball_geom not in (
                int(data.contact[index].geom1),
                int(data.contact[index].geom2),
            )
            for index in range(int(data.ncon))
        )
    assert data.qpos[ball_qadr] == pytest.approx(2.30, abs=1.0e-9)
    assert data.qvel[ball_dadr] == pytest.approx(4.0, abs=1.0e-12)


@requires_mujoco
def test_vendor_stand_pose_is_clear_for_teacher_five_solid_guard():
    canonical = GATE.native_diag.DEFAULT_MJCF.read_bytes()
    rows = GATE.table_scene.action_ball_policy_obstacle_geometry()
    four, _scene = GATE.assemble_fitted_scene_xml(
        canonical, rows, _venue(), 0.001
    )
    five = GATE.table_scene.append_action_ball_policy_keepout_xml(
        four, rows, collidable=True
    )
    assets = GATE.table_scene._mesh_assets(
        canonical, GATE.native_diag.DEFAULT_MJCF.parent
    )
    model = mujoco.MjModel.from_xml_string(
        five.decode("utf-8"), assets=assets
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ball = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, GATE.BALL_GEOM_NAME
    )
    obstacle_ids = {
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        ): name
        for name in GATE.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    }
    robot_ids = GATE._five_solid_robot_geom_ids(
        mujoco,
        model,
        ball_geom_id=ball,
        obstacle_geom_ids=tuple(obstacle_ids),
    )
    ground = GATE.build_ground_contact_contract(
        mujoco, model, ball_geom_id=ball
    )
    events = GATE.FittedEvents()
    GATE._scan_robot_contacts(
        mujoco,
        model,
        data,
        ball,
        obstacle_ids,
        events,
        0.0,
        ground,
    )
    assert events.robot_obstacle_contact_count == 0
    assert events.legal_foot_support_contact_count > 0
    assert events.foot_floor_penetration_violation_count == 0
    assert events.nonfoot_ground_contact_violation_count == 0
    centers = np.asarray(data.geom_xpos[list(robot_ids)]).copy()
    swept = GATE.scan_five_solid_robot_sweep(
        mujoco=mujoco,
        model=model,
        data=data,
        robot_geom_ids=robot_ids,
        centers_before=centers,
        obstacle_aabbs=(
            GATE.table_scene.action_ball_policy_obstacle_aabbs(rows)
        ),
    )
    assert swept["hit_count"] == 0


def test_half_millisecond_and_one_millisecond_convergence_gate():
    coarse = _result()
    fine = _result(
        contact_time=1.001,
        contact_position=(0.203, 0.0, 1.2),
        velocity=(4.05, 0.0, 2.0),
        net_z=1.055,
        landing=(2.51, 0.1),
        landing_time=1.51,
    )
    assert GATE.compare_convergence(coarse, fine)["pass"] is True
    fine["first_landing"]["ball_center_xy_m"] = [2.55, 0.1]
    failed = GATE.compare_convergence(coarse, fine)
    assert failed["pass"] is False
    assert "nonconverged_landing_xy_m" in failed["failure_reasons"]


def test_ready_recovery_includes_root_pose_and_endpoint_velocity():
    clips = {
        "a": _motion_clip(),
        "b": _motion_clip(
            root_start=(0.02, 0.0, 1.0),
            root_end=(0.03, 0.0, 1.0),
            joint_end_offset=0.01,
            endpoint_velocity=0.02,
        ),
    }
    receipt = GATE.ready_recovery_metrics(clips, ("a", "b"))
    assert receipt["shared_ready"]["joint_linf_rad"] == 0.0
    assert receipt["shared_ready"]["root_position_l2_m"] == pytest.approx(
        0.02
    )
    assert (
        receipt["shared_ready"][
            "endpoint_root_linear_velocity_peak_mps"
        ]
        == pytest.approx(0.02)
    )
    assert receipt["recovery_by_action"]["b"][
        "joint_linf_rad"
    ] == pytest.approx(0.01)
    assert receipt["recovery_by_action"]["b"][
        "root_position_l2_m"
    ] == pytest.approx(0.01)


def test_retimed_teacher_scales_phase_and_all_velocity_classes():
    clip = _motion_clip(endpoint_velocity=2.0)
    state = GATE.retimed_teacher_state(
        clip,
        world_time_s=0.25,
        pre_swing_wait_s=0.05,
        teacher_rate=2.0,
    )
    assert state["source_motion_time_s"] == pytest.approx(0.4)
    assert state["joint_pos"][0] == pytest.approx(0.0)
    assert state["joint_vel"][0] == pytest.approx(4.0)
    assert state["root_lin_vel"][0] == pytest.approx(4.0)
    assert state["root_ang_vel"][2] == pytest.approx(4.0)


def test_physical_task_binding_replays_exact_geometry_and_external_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    action, raw = _physical_task_binding_fixture(
        tmp_path, monkeypatch
    )
    binding = GATE.validate_physical_task_binding(
        raw,
        action=action,
        solver_profile_sha256="6" * 64,
        physics_profile_sha256="7" * 64,
        geometry_source_sha256=(
            GATE.racket_geometry.GEOMETRY_SOURCE_SHA256
        ),
    )
    assert [case.case_role for case in binding.cases] == list(
        GATE.PHYSICAL_TASK_CASE_ROLES
    )
    assert binding.cases[0].task_payload_sha256 != (
        binding.cases[1].task_payload_sha256
    )
    assert binding.solver_execution_receipt_path.name == (
        "solver_receipt.json"
    )


def test_physical_task_binding_rejects_arbitrary_task_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    action, raw = _physical_task_binding_fixture(
        tmp_path, monkeypatch
    )
    raw["cases"][0]["task_payload"]["racket_site_target_w_m"][0] += (
        0.01
    )
    raw["cases"][0]["task_payload_sha256"] = (
        GATE._canonical_payload_sha256(
            raw["cases"][0]["task_payload"]
        )
    )
    case = raw["cases"][0]
    case["case_binding_sha256"] = GATE._canonical_payload_sha256(
        {
            "action_id": action.action_id,
            "action_uid": action.action_uid,
            "motion_sha256": action.motion_sha256,
            "case_id": case["case_id"],
            "case_role": case["case_role"],
            "sample_seed": case["sample_seed"],
            "ball_proposal_sha256": case[
                "ball_proposal_sha256"
            ],
            "task_payload_sha256": case["task_payload_sha256"],
            "solver_execution_identity_sha256": raw[
                "solver_execution_identity_sha256"
            ],
            "fault_injection": case["fault_injection"],
            "expected_physical_verdict": case[
                "expected_physical_verdict"
            ],
            "expected_failure_reason": case[
                "expected_failure_reason"
            ],
        }
    )
    raw["cases_sha256"] = GATE.native_diag.sha256_bytes(
        GATE.native_diag.canonical_json_bytes(raw["cases"])
    )
    external_path = tmp_path / raw["solver_execution_receipt_path"]
    external = json.loads(external_path.read_text())
    external["cases"] = raw["cases"]
    external.pop("receipt_payload_sha256")
    external["receipt_payload_sha256"] = (
        GATE._canonical_payload_sha256(external)
    )
    external_path.write_text(json.dumps(external))
    raw["solver_execution_receipt_sha256"] = _sha(external_path)
    with pytest.raises(
        GATE.FittedGateError, match="racket site target"
    ):
        GATE.validate_physical_task_binding(
            raw,
            action=action,
            solver_profile_sha256="6" * 64,
            physics_profile_sha256="7" * 64,
            geometry_source_sha256=(
                GATE.racket_geometry.GEOMETRY_SOURCE_SHA256
            ),
        )


def test_control_matrix_requires_positive_pass_and_negative_physical_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    action, raw = _physical_task_binding_fixture(
        tmp_path, monkeypatch
    )
    binding = GATE.validate_physical_task_binding(
        raw,
        action=action,
        solver_profile_sha256="6" * 64,
        physics_profile_sha256="7" * 64,
        geometry_source_sha256=(
            GATE.racket_geometry.GEOMETRY_SOURCE_SHA256
        ),
    )
    positive_result = {
        "0.0010": {
            "verdict": "PASS",
            "failure_reasons": [],
            "mandatory_gates": {
                "physical_ball_selected_face_return_and_first_landing": (
                    True
                )
            },
        },
        "0.0005": {
            "verdict": "PASS",
            "failure_reasons": [],
            "mandatory_gates": {
                "physical_ball_selected_face_return_and_first_landing": (
                    True
                )
            },
        },
    }
    positive = GATE.evaluate_physical_task_control(
        binding.cases[0],
        positive_result,
        {"pass": True},
        {"kind": "none", "applied": True},
    )
    assert positive["control_verdict"] == "PASS"
    negative_result = {
        key: {
            "verdict": "FAIL",
            "failure_reasons": [
                "teacher_task_face_normal_mismatch"
            ],
            "mandatory_gates": {
                "physical_ball_selected_face_return_and_first_landing": (
                    False
                )
            },
        }
        for key in ("0.0010", "0.0005")
    }
    negative = GATE.evaluate_physical_task_control(
        binding.cases[4],
        negative_result,
        {"pass": False},
        {"kind": "selected_face_sign_flip", "applied": True},
    )
    assert negative["control_verdict"] == "PASS"
    assert (
        negative["observed_failure_reason"]
        == "teacher_task_face_sign_mismatch"
    )


def test_formal_core_requires_one_video_directory_before_mujoco_import(
    tmp_path: Path,
):
    manifest = REPO / "configs/action_ball_n5_nomove_f20_20260728.json"
    pins = REPO / "configs/action_ball_profile_pins_20260728.json"
    out = tmp_path / "receipt.json"
    result = GATE.main(
        [
            "--training-manifest",
            str(manifest),
            "--training-manifest-sha256",
            _sha(manifest),
            "--physical-gate-manifest",
            str(manifest),
            "--physical-gate-manifest-sha256",
            _sha(manifest),
            "--physical-gate-materialization-receipt",
            str(manifest),
            "--physical-gate-materialization-receipt-sha256",
            _sha(manifest),
            "--profile-pins",
            str(pins),
            "--profile-pins-sha256",
            _sha(pins),
            "--out",
            str(out),
        ]
    )
    assert result == 3
    receipt = json.loads(out.read_text())
    assert (
        "missing_required_per_action_physical_video_render_dir"
        in receipt["preflight"]["blockers"]
    )
    assert receipt["formal_gate_executed"] is False


def test_preflight_current_tree_is_fail_closed_without_importing_mujoco(
    tmp_path: Path,
):
    manifest = REPO / "configs/action_ball_n5_nomove_f20_20260728.json"
    pins = REPO / "configs/action_ball_profile_pins_20260728.json"
    out = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--training-manifest",
            str(manifest),
            "--training-manifest-sha256",
            _sha(manifest),
            "--physical-gate-manifest",
            str(manifest),
            "--physical-gate-manifest-sha256",
            _sha(manifest),
            "--physical-gate-materialization-receipt",
            str(manifest),
            "--physical-gate-materialization-receipt-sha256",
            _sha(manifest),
            "--profile-pins",
            str(pins),
            "--profile-pins-sha256",
            _sha(pins),
            "--preflight-only",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    receipt = json.loads(out.read_text())
    blockers = receipt["preflight"]["blockers"]
    assert "missing_expected_clean_code_commit" in blockers
    assert any("action_set_contract:" in reason for reason in blockers)
    assert (
        "manifests_cannot_bind_missing_trusted_action_set_contract"
        in blockers
    )
    assert "missing_independent_launch_evidence_trust_root" in blockers
    assert receipt["native_ball_contact_enabled"] is False
    assert receipt["verdict"] == "BLOCKED"
    assert len(receipt["receipt_payload_sha256"]) == 64


def test_source_has_unique_fitted_authority_and_no_virtual_return_grader():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "mujoco.mj_step(model, data)" in source
    assert '\"contype\": \"0\"' in source
    assert '\"conaffinity\": \"0\"' in source
    assert "contact_model.predict_contact" in source
    assert "swept_selected_face_intersection" in source
    assert "virtual_return_scorer" not in source
    assert source.count("mujoco.mj_contactForce(") == 1
    contact_scan = source[
        source.index("def _scan_robot_contacts(") :
        source.index("def _record_net_events_on_segment(")
    ]
    assert "mujoco.mj_contactForce(" in contact_scan
    assert "native_ball_contact_count" in contact_scan
    assert "--contact-position-tolerance-m" not in source
    assert "--contact-time-tolerance-s" not in source
    assert "--post-contact-s" not in source
    assert "--mjcf" not in source
    assert "--identity-manifest" not in source
    assert '"PREFLIGHT_PASS"' in source
    assert '"PASS_PREFLIGHT_ONLY"' not in source
    assert '"selected_face_mesh_sha256"' in source
