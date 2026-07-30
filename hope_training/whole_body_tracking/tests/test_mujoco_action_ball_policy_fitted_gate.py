from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "mujoco_action_ball_policy_fitted_gate.py"
)
SPEC = importlib.util.spec_from_file_location("policy_fitted_gate_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _action_set(action_count: int = 5) -> dict:
    ids = [f"a{index}" for index in range(action_count)]
    uids = list(range(1000, 1000 + action_count))
    base = {
        "profile_id": f"fixture_n{action_count}",
        "expected_n": action_count,
        "scope": "upper" if action_count != 73 else "full",
        "mobility_mode": "no_move",
        "ordered_action_ids": ids,
        "ordered_action_uids": uids,
        "order_uid_digest_sha256": (
            gate.fitted.action_set_contract.order_uid_digest(ids, uids)
        ),
        "manifest_path": f"configs/fixture_n{action_count}.json",
        "manifest_sha256": "9" * 64,
        "experiment_name": f"fixture_n{action_count}",
    }
    return gate.fitted.action_set_contract.validate_contract(
        base, profile_id=base["profile_id"], profile_policies={}
    )


def _metadata(
    checkpoint_sha: str = "a" * 64,
    *,
    action_set: dict | None = None,
) -> dict[str, str]:
    action_set = _action_set() if action_set is None else action_set
    count = action_set["expected_n"]
    names, dims = gate._actor_term_contract(count)
    return {
        "actor_obs_contract": action_set["actor_obs_contract"],
        "actor_obs_mode": "hitter_footwork",
        "actor_obs_total_dim": str(action_set["actor_obs_width"]),
        "actor_obs_term_dims": ",".join(str(item) for item in dims),
        "observation_names": ",".join(names),
        "training_contract_exact": "1",
        "training_contract_schema_version": "3",
        "action_ball_profile_id": action_set["profile_id"],
        "action_ball_expected_n": str(count),
        "action_ball_scope": action_set["scope"],
        "action_ball_mobility_mode": action_set["mobility_mode"],
        "action_ball_action_set_contract_sha256": action_set[
            "contract_sha256"
        ],
        "action_ball_manifest_sha256": action_set["manifest_sha256"],
        "action_ball_action_order": json.dumps(
            action_set["ordered_action_ids"], separators=(",", ":")
        ),
        "action_ball_ordered_action_uids": json.dumps(
            action_set["ordered_action_uids"], separators=(",", ":")
        ),
        "action_ball_order_uid_digest_sha256": action_set[
            "order_uid_digest_sha256"
        ],
        "action_ball_action_set_contract_source_sha256": (
            gate.fitted.native_diag.sha256_file(
                gate.fitted.ACTION_SET_CONTRACT_SOURCE_PATH
            )
        ),
        "hope_metadata_schema_version": "2",
        "source_checkpoint_sha256": checkpoint_sha,
        "motion_clip_sha256": ",".join(
            f"{index + 1:064x}" for index in range(count)
        ),
        "clip_seg_lengths": ",".join(
            str(101 + index) for index in range(count)
        ),
        "physics_step_dt_s": "0.005",
        "policy_step_dt_s": "0.02",
        "control_decimation": "4",
        "action_use_default_offset": "1",
        "qdes_clamp": "1",
        "qdes_joint_pos_limits": ",".join(["-1", "1"] * 31),
        "joint_friction_backend": "physx",
        "joint_friction_semantics": "load_dependent_spatial_force_coefficient",
        "joint_friction_units": "dimensionless",
        "motion_hold_reference": "stand",
    }


def _validate_actor_metadata(
    metadata: dict[str, str],
    *,
    checkpoint_sha256: str,
    onnx_obs_dim: int,
    action_set: dict | None = None,
):
    return gate.validate_actor_metadata(
        metadata,
        checkpoint_sha256=checkpoint_sha256,
        onnx_obs_dim=onnx_obs_dim,
        trusted_action_set=(
            _action_set() if action_set is None else action_set
        ),
    )


def test_action_ball_n5_observation_tail_is_exact_and_one_hot() -> None:
    prefix = np.arange(177, dtype=np.float64)
    normal = np.asarray((0.2, -0.3, 0.9))
    obs = gate.build_action_ball_obs(
        prefix, target_normal_w=normal, action_slot=3
    )
    assert obs.shape == (186,)
    np.testing.assert_array_equal(obs[:177], prefix)
    np.testing.assert_array_equal(obs[177:180], normal)
    assert obs[180] == 0.0
    np.testing.assert_array_equal(obs[181:], (0.0, 0.0, 0.0, 1.0, 0.0))


def test_task_command_uses_yaw_heading_frame_not_full_pelvis_tilt() -> None:
    case = SimpleNamespace(
        racket_site_target_w_m=np.asarray((1.4, -0.2, 1.1)),
        racket_site_velocity_w_mps=np.asarray((3.0, 0.1, 0.4)),
        racket_normal_w=np.asarray((-0.9, 0.0, 0.435889894)),
        base_goal_w_m=np.asarray((0.3, -0.4, 0.0)),
        time_to_contact_s=0.8,
    )
    command = gate.TaskCommand(case, SimpleNamespace(mount_normal_sign=1))
    base_pos = np.asarray((0.1, 0.2, 0.9))
    # Non-zero yaw, roll and pitch makes a full-quaternion projection differ
    # materially from the training contract's yaw-only heading projection.
    roll, pitch, yaw = 0.31, -0.27, 0.63
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    base_quat = np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )
    racket_pos = np.asarray((0.9, -0.1, 1.0))
    yaw_rotation = gate.legacy_mj.mat_from_quat(
        gate.legacy_mj.yaw_quat(base_quat)
    )
    np.testing.assert_allclose(
        command.racket_target_pos_b_rel_fk(base_pos, base_quat, racket_pos),
        yaw_rotation.T @ (case.racket_site_target_w_m - racket_pos),
        atol=1.0e-12,
    )
    base_delta = np.asarray(
        (
            case.base_goal_w_m[0] - base_pos[0],
            case.base_goal_w_m[1] - base_pos[1],
            0.0,
        )
    )
    np.testing.assert_allclose(
        command.base_target_pos_b(base_pos, base_quat),
        (yaw_rotation.T @ base_delta)[:2],
        atol=1.0e-12,
    )
    full_rotation = gate.legacy_mj.mat_from_quat(base_quat)
    assert not np.allclose(
        command.racket_target_pos_b_rel_fk(base_pos, base_quat, racket_pos),
        full_rotation.T @ (case.racket_site_target_w_m - racket_pos),
    )


@pytest.mark.parametrize("slot", (-1, 5))
def test_action_ball_n5_observation_rejects_out_of_bank_slot(slot: int) -> None:
    with pytest.raises(gate.PolicyGateError):
        gate.build_action_ball_obs(
            np.zeros(177), target_normal_w=np.asarray((1.0, 0.0, 0.0)), action_slot=slot
        )


def test_actor_contract_accepts_only_exact_186d_n5_metadata() -> None:
    evidence = _validate_actor_metadata(
        _metadata(), checkpoint_sha256="a" * 64, onnx_obs_dim=186
    )
    assert evidence["contract"] == "action_ball_n5"
    assert evidence["obs_dim"] == 186
    assert evidence["clip_seg_lengths"] == [101, 102, 103, 104, 105]
    np.testing.assert_array_equal(
        evidence["qdes_joint_pos_limits"], np.tile((-1.0, 1.0), (31, 1))
    )


@pytest.mark.parametrize("action_count", (1, 5, 73))
def test_actor_and_one_hot_contract_are_exact_for_n1_n5_n73(
    action_count: int,
) -> None:
    trusted = _action_set(action_count)
    obs_dim = 181 + action_count
    evidence = _validate_actor_metadata(
        _metadata(action_set=trusted),
        checkpoint_sha256="a" * 64,
        onnx_obs_dim=obs_dim,
        action_set=trusted,
    )
    assert evidence["obs_dim"] == obs_dim
    assert len(evidence["clip_seg_lengths"]) == action_count
    obs = gate.build_action_ball_obs(
        np.zeros(177),
        target_normal_w=np.asarray((1.0, 0.0, 0.0)),
        action_slot=action_count - 1,
        action_count=action_count,
    )
    assert obs.shape == (obs_dim,)
    assert obs[-1] == 1.0
    assert np.count_nonzero(obs[-action_count:]) == 1


@pytest.mark.parametrize("action_count", (1, 5, 73))
@pytest.mark.parametrize(
    "field",
    (
        "action_ball_profile_id",
        "action_ball_expected_n",
        "action_ball_scope",
        "action_ball_mobility_mode",
        "action_ball_action_set_contract_sha256",
        "action_ball_manifest_sha256",
        "action_ball_action_order",
        "action_ball_ordered_action_uids",
        "action_ball_order_uid_digest_sha256",
        "action_ball_action_set_contract_source_sha256",
    ),
)
def test_actor_rejects_any_action_set_metadata_tamper(
    action_count: int, field: str
) -> None:
    trusted = _action_set(action_count)
    metadata = _metadata(action_set=trusted)
    metadata[field] = "tampered"
    with pytest.raises(gate.PolicyGateError):
        _validate_actor_metadata(
            metadata,
            checkpoint_sha256="a" * 64,
            onnx_obs_dim=181 + action_count,
            action_set=trusted,
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    (
        ("actor_obs_contract", "task_first_n5"),
        ("actor_obs_total_dim", "181"),
        ("training_contract_exact", "0"),
    ),
)
def test_actor_contract_fails_closed_on_semantic_drift(field: str, bad: str) -> None:
    metadata = _metadata()
    metadata[field] = bad
    with pytest.raises(gate.PolicyGateError):
        _validate_actor_metadata(
            metadata, checkpoint_sha256="a" * 64, onnx_obs_dim=186
        )


def test_actor_contract_rejects_checkpoint_or_motion_width_drift() -> None:
    with pytest.raises(gate.PolicyGateError):
        _validate_actor_metadata(
            _metadata(), checkpoint_sha256="f" * 64, onnx_obs_dim=186
        )
    metadata = _metadata()
    metadata["motion_clip_sha256"] = ",".join(["b" * 64] * 4)
    with pytest.raises(gate.PolicyGateError):
        _validate_actor_metadata(
            metadata, checkpoint_sha256="a" * 64, onnx_obs_dim=186
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    (
        ("action_use_default_offset", "0"),
        ("qdes_clamp", "0"),
        ("joint_friction_backend", "mujoco"),
        ("joint_friction_semantics", "constant_coulomb_torque"),
        ("joint_friction_units", "N*m"),
    ),
)
def test_actor_contract_rejects_control_plant_semantic_drift(
    field: str, bad: str
) -> None:
    metadata = _metadata()
    metadata[field] = bad
    with pytest.raises(gate.PolicyGateError):
        _validate_actor_metadata(
            metadata, checkpoint_sha256="a" * 64, onnx_obs_dim=186
        )


def test_actor_contract_rejects_invalid_training_qdes_limits() -> None:
    metadata = _metadata()
    limits = [-1.0, 1.0] * 31
    limits[12:14] = [0.2, -0.2]
    metadata["qdes_joint_pos_limits"] = ",".join(str(value) for value in limits)
    with pytest.raises(gate.PolicyGateError, match="lo > hi"):
        _validate_actor_metadata(
            metadata, checkpoint_sha256="a" * 64, onnx_obs_dim=186
        )


def _manifest_and_receipt() -> tuple[SimpleNamespace, dict]:
    trusted = _action_set()
    action_rows = []
    actions = []
    bindings = {}
    order = tuple(f"a{index}" for index in range(5))
    for index, action_id in enumerate(order):
        action = SimpleNamespace(
            action_id=action_id,
            action_uid=trusted["ordered_action_uids"][index],
            motion_sha256=f"{index + 1:064x}",
        )
        actions.append(action)
        cases = []
        receipt_cases = []
        for role in (
            *gate.POSITIVE_CASE_ROLES,
            "negative_t_hit_offset",
            "negative_face_sign",
            "negative_ball_state_mismatch",
        ):
            case = SimpleNamespace(
                case_role=role, case_binding_sha256=(role[0] * 64)
            )
            cases.append(case)
            receipt_cases.append(
                {
                    "case_role": role,
                    "case_binding_sha256": case.case_binding_sha256,
                    "control_verdict": "PASS",
                }
            )
        bindings[action_id] = SimpleNamespace(cases=cases)
        action_rows.append(
            {
                "action_id": action_id,
                "action_uid": action.action_uid,
                "motion_sha256": action.motion_sha256,
                "verdict": "PASS",
                "physical_task_binding": {"cases": receipt_cases},
            }
        )
    manifest = SimpleNamespace(
        base=SimpleNamespace(
            action_order=order,
            actions=tuple(actions),
        ),
        task_bindings=bindings,
        action_set_contract=trusted,
    )
    receipt = {
        "gate": "mujoco_teacher_motion_fitted_ball_gate",
        "status": "PASS",
        "verdict": "PASS",
        "analytic_return_scorer_executed": False,
        "selector_executed": False,
        "action_order": list(order),
        "action_set_contract": trusted,
        "actions": action_rows,
    }
    receipt["receipt_payload_sha256"] = gate.fitted.native_diag.sha256_bytes(
        gate.fitted.native_diag.canonical_json_bytes(receipt)
    )
    return manifest, receipt


def test_teacher_receipt_requires_exact_action_and_case_binding() -> None:
    manifest, receipt = _manifest_and_receipt()
    result = gate.validate_teacher_receipt(receipt, manifest)
    assert result["status"] == "PASS"
    assert all(
        len(rows) == 6
        for rows in result["controls_by_action"].values()
    )
    assert all(
        len(
            [
                row
                for row in rows
                if row["case_role"] not in gate.POSITIVE_CASE_ROLES
            ]
        )
        == 3
        for rows in result["controls_by_action"].values()
    )
    receipt["actions"][2]["physical_task_binding"]["cases"][0][
        "case_binding_sha256"
    ] = "0" * 64
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = gate.fitted.native_diag.sha256_bytes(
        gate.fitted.native_diag.canonical_json_bytes(payload)
    )
    with pytest.raises(gate.PolicyGateError):
        gate.validate_teacher_receipt(receipt, manifest)


def test_teacher_receipt_rejects_duplicate_or_extra_control_rows() -> None:
    manifest, receipt = _manifest_and_receipt()
    rows = receipt["actions"][0]["physical_task_binding"]["cases"]
    rows.append(dict(rows[-1]))
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = (
        gate.fitted.native_diag.sha256_bytes(
            gate.fitted.native_diag.canonical_json_bytes(payload)
        )
    )
    with pytest.raises(
        gate.PolicyGateError,
        match="missing, duplicated, extra, or reordered",
    ):
        gate.validate_teacher_receipt(receipt, manifest)


def test_physical_case_grade_never_confuses_safety_with_return_failure() -> None:
    events = gate.fitted.FittedEvents(
        paddle_impulse_count=1,
        paddle_contact={"time_s": 0.5},
        return_table_bounces=1,
        net_crossing={"time_s": 0.55},
        first_landing={"ball_center_xy_m": np.asarray((1.0, 0.1))},
    )
    case = SimpleNamespace(
        time_to_contact_s=0.5,
        landing_aim_w_xy_m=np.asarray((1.0, 0.1)),
    )
    safe = {
        "table_contact_steps": 0,
        "self_contact_steps": 0,
        "hard_joint_limit_steps": 0,
        "velocity_limit_steps": 0,
        "fall_steps": 0,
    }
    imitation = {
        "finite": True,
        "strike": {
            "racket_site_target_position_error_m": 0.01,
            "racket_site_target_velocity_error_mps": 0.1,
            "racket_face_normal_angle_error_rad": 0.05,
        },
    }
    verdict, reasons = gate._grade_case(
        events=events, case=case, safety=safe, imitation=imitation
    )
    assert verdict == "PASS"
    assert reasons == []
    unsafe = dict(safe, hard_joint_limit_steps=1)
    verdict, reasons = gate._grade_case(
        events=events, case=case, safety=unsafe, imitation=imitation
    )
    assert verdict == "FAIL"
    assert "hard_joint_limit_steps" in reasons
    assert "no_first_table_landing" not in reasons


def test_physical_case_grade_rejects_continuous_five_solid_sweep() -> None:
    events = gate.fitted.FittedEvents(
        paddle_impulse_count=1,
        paddle_contact={"time_s": 0.5},
        return_table_bounces=1,
        net_crossing={"time_s": 0.55},
        first_landing={"ball_center_xy_m": np.asarray((1.0, 0.1))},
    )
    case = SimpleNamespace(
        time_to_contact_s=0.5,
        landing_aim_w_xy_m=np.asarray((1.0, 0.1)),
    )
    safety = {
        "table_contact_steps": 0,
        "table_swept_guard_steps": 1,
        "self_contact_steps": 0,
        "hard_joint_limit_steps": 0,
        "velocity_limit_steps": 0,
        "fall_steps": 0,
        "qdes_clamp_joint_commands": 0,
    }
    imitation = {
        "finite": True,
        "strike": {
            "racket_site_target_position_error_m": 0.0,
            "racket_site_target_velocity_error_mps": 0.0,
            "racket_face_normal_angle_error_rad": 0.0,
        },
    }
    verdict, reasons = gate._grade_case(
        events=events, case=case, safety=safety, imitation=imitation
    )
    assert verdict == "FAIL"
    assert "table_swept_guard_steps" in reasons
    assert "table_contact_steps" not in reasons


def test_swept_guard_catches_under_table_tunnel_missed_by_top_slab() -> None:
    rows = gate.fitted.table_scene.action_ball_policy_obstacle_geometry()
    aabbs = (
        gate.fitted.table_scene.action_ball_policy_obstacle_aabbs(rows)
    )
    start = np.asarray((0.25, 0.0, 0.35))
    end = np.asarray((3.50, 0.0, 0.35))
    radius = 0.02
    keepout_lo, keepout_hi = aabbs[
        gate.fitted.table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME
    ]
    top_lo, top_hi = aabbs["motion_table_top"]
    assert gate._segment_intersects_inflated_aabb(
        start, end, keepout_lo, keepout_hi, radius
    )
    assert not gate._segment_intersects_inflated_aabb(
        start, end, top_lo, top_hi, radius
    )


def test_five_solid_geometry_sha_binds_every_axis_and_keepout() -> None:
    rows = gate.fitted.table_scene.action_ball_policy_obstacle_geometry()
    baseline = gate.fitted.table_scene.action_ball_policy_geometry_contract(
        rows
    )
    assert baseline["payload"]["obstacle_order"] == list(
        gate.fitted.table_scene.ACTION_BALL_POLICY_OBSTACLE_NAMES
    )
    changed = json.loads(json.dumps(rows))
    changed["robot_keepout"]["full_extents_m"][2] -= 0.001
    mutated = gate.fitted.table_scene.action_ball_policy_geometry_contract(
        changed
    )
    assert mutated["sha256"] != baseline["sha256"]


def test_compiled_five_solid_gate_is_robot_only_and_detects_keepout() -> None:
    """Exercise the actual MuJoCo filter/contact state, not just XML fields."""

    mj = pytest.importorskip("mujoco")
    rows = gate.fitted.table_scene.action_ball_policy_obstacle_geometry()
    contract = gate.fitted.table_scene.action_ball_policy_geometry_contract(
        rows
    )
    base = f"""<mujoco>
      <option timestep="0.001" gravity="0 0 0"/>
      <worldbody>
        <body name="guarded_robot" pos="0.25 0 0.35">
          <freejoint name="guarded_robot_joint"/>
          <geom name="guarded_robot_geom" type="sphere" size="0.05"
                mass="1" contype="1" conaffinity="7"/>
        </body>
        <body name="{gate.fitted.BALL_BODY_NAME}" pos="0.25 0 0.35">
          <freejoint name="{gate.fitted.BALL_JOINT_NAME}"/>
          <geom name="{gate.fitted.BALL_GEOM_NAME}" type="sphere" size="0.02"
                mass="0.0034" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>""".encode()
    four = gate.fitted.table_scene.augment_mjcf_xml(
        base, rows, collidable=True
    )
    five = gate.fitted.table_scene.append_action_ball_policy_keepout_xml(
        four, rows, collidable=True
    )
    model = mj.MjModel.from_xml_string(five.decode("utf-8"))
    data = mj.MjData(model)
    robot_geom = int(
        mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_GEOM, "guarded_robot_geom"
        )
    )
    wrapper = SimpleNamespace(
        mj=mj,
        model=model,
        data=data,
        action_ball_safety_geom_ids=np.asarray(
            (robot_geom,), dtype=np.int64
        ),
    )
    compiled = gate._validate_compiled_five_solid_scene(
        wrapper,
        rows,
        contract,
        assembled_xml_sha256=gate.fitted.native_diag.sha256_bytes(five),
    )
    assert compiled["five_solid_geometry_sha256"] == contract["sha256"]
    assert compiled["robot_collision_geom_count"] == 1
    assert compiled["ball_keepout_native_pair_enabled"] is False
    assert compiled["ball_keepout_analytic_surface_enabled"] is False

    robot_joint = int(
        mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_JOINT, "guarded_robot_joint"
        )
    )
    robot_qpos = int(model.jnt_qposadr[robot_joint])
    data.qpos[robot_qpos : robot_qpos + 3] = (1.0, 0.0, 0.35)
    mj.mj_forward(model, data)
    contact = gate._five_solid_contact_scan(wrapper)
    assert contact["contact_count"] > 0
    assert (
        contact["per_obstacle"][
            gate.fitted.table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME
        ]
        > 0
    )

    keepout = int(
        mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_GEOM,
            gate.fitted.table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME,
        )
    )
    ball = int(
        mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_GEOM, gate.fitted.BALL_GEOM_NAME
        )
    )
    assert not gate._collision_pair_enabled(model, ball, keepout)
    model.geom_contype[ball] = 1
    with pytest.raises(
        gate.PolicyGateError, match="keepout can affect the fitted ball"
    ):
        gate._validate_compiled_five_solid_scene(
            wrapper,
            rows,
            contract,
            assembled_xml_sha256=gate.fitted.native_diag.sha256_bytes(
                five
            ),
        )


def test_vendor_robot_swept_guard_catches_contact_free_table_tunnel() -> None:
    """A legal stand is clear; teleporting past the table cannot tunnel."""

    mj = pytest.importorskip("mujoco")
    rows = gate.fitted.table_scene.action_ball_policy_obstacle_geometry()
    aabbs = gate.fitted.table_scene.action_ball_policy_obstacle_aabbs(
        rows
    )
    scene = gate.fitted.table_scene.load_table_scene(
        mj,
        gate.fitted.CANONICAL_MJCF,
        collidable=True,
        action_ball_policy=True,
    )
    data = mj.MjData(scene.model)
    mj.mj_resetDataKeyframe(scene.model, data, 0)
    mj.mj_forward(scene.model, data)
    obstacle_ids = tuple(scene.obstacle_geom_ids.values())
    robot_ids = np.asarray(
        [
            geom_id
            for geom_id in range(int(scene.model.ngeom))
            if int(scene.model.geom_bodyid[geom_id]) != 0
            and any(
                gate._collision_pair_enabled(
                    scene.model, geom_id, obstacle_id
                )
                for obstacle_id in obstacle_ids
            )
        ],
        dtype=np.int64,
    )
    assert robot_ids.size > 0
    wrapper = SimpleNamespace(
        mj=mj,
        model=scene.model,
        data=data,
        action_ball_safety_geom_ids=robot_ids,
    )
    before = np.asarray(data.geom_xpos[robot_ids], np.float64).copy()
    legal = gate._five_solid_swept_scan(wrapper, before, aabbs)
    assert legal["hit_count"] == 0

    # Both endpoints are contact-free, but the path traverses the complete
    # table assembly.  The continuous guard must still catch the under-table
    # volume and physical top/net solids.
    data.qpos[0] = 4.0
    mj.mj_forward(scene.model, data)
    assert gate._five_solid_contact_scan(wrapper)["contact_count"] == 0
    swept = gate._five_solid_swept_scan(wrapper, before, aabbs)
    assert swept["hit_count"] > 0
    assert (
        swept["per_obstacle"][
            gate.fitted.table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME
        ]
        > 0
    )


@pytest.mark.parametrize(
    ("field", "limit", "reason"),
    (
        (
            "racket_site_target_position_error_m",
            gate.STRIKE_POSITION_ERROR_TOLERANCE_M,
            "bound_strike_position_error_exceeds_gate",
        ),
        (
            "racket_site_target_velocity_error_mps",
            gate.STRIKE_VELOCITY_ERROR_TOLERANCE_MPS,
            "bound_strike_velocity_error_exceeds_gate",
        ),
        (
            "racket_face_normal_angle_error_rad",
            gate.STRIKE_NORMAL_ERROR_TOLERANCE_RAD,
            "bound_strike_face_error_exceeds_gate",
        ),
    ),
)
def test_physical_case_grade_requires_the_bound_strike_frame(
    field: str, limit: float, reason: str
) -> None:
    events = gate.fitted.FittedEvents(
        paddle_impulse_count=1,
        paddle_contact={"time_s": 0.5},
        return_table_bounces=1,
        net_crossing={"time_s": 0.55},
        first_landing={"ball_center_xy_m": np.asarray((1.0, 0.1))},
    )
    case = SimpleNamespace(
        time_to_contact_s=0.5,
        landing_aim_w_xy_m=np.asarray((1.0, 0.1)),
    )
    strike = {
        "racket_site_target_position_error_m": 0.0,
        "racket_site_target_velocity_error_mps": 0.0,
        "racket_face_normal_angle_error_rad": 0.0,
    }
    strike[field] = limit + 1.0e-6
    verdict, reasons = gate._grade_case(
        events=events,
        case=case,
        safety={
            "table_contact_steps": 0,
            "self_contact_steps": 0,
            "hard_joint_limit_steps": 0,
            "velocity_limit_steps": 0,
            "fall_steps": 0,
        },
        imitation={"finite": True, "strike": strike},
    )
    assert verdict == "FAIL"
    assert reason in reasons


def test_no_clobber_writer_converts_numpy_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    gate._write_no_clobber(
        output, {"vector": np.asarray((1.0, 2.0)), "scalar": np.float64(3.0)}
    )
    assert '"vector": [' in output.read_text()
    with pytest.raises(FileExistsError):
        gate._write_no_clobber(output, {"other": True})
