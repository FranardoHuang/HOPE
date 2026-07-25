"""Tests for the canonical MuJoCo dynamics screen.

The pure contract tests run without MuJoCo.  The final two tests use the exact
vendor MJCF when ``mujoco`` is installed; otherwise they are skipped.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "hope_training/whole_body_tracking/scripts"
sys.path.insert(0, str(SCRIPTS))

import canonical_mujoco_dynamics_gate as gate  # noqa: E402


def _trajectory(frames: int = 7) -> gate.MotionTrajectory:
    q = np.zeros((frames, len(gate.RUNTIME_JOINT_NAMES)), np.float64)
    zeros3 = np.zeros((frames, 3), np.float64)
    quat = np.zeros((frames, 4), np.float64)
    quat[:, 0] = 1.0
    return gate.MotionTrajectory(
        joint_pos=q,
        joint_vel=q.copy(),
        root_pos_w=zeros3.copy(),
        root_quat_wxyz=quat,
        root_lin_vel_w=zeros3.copy(),
        root_ang_vel_w=zeros3.copy(),
        fps=50.0,
    )


def test_motion_trajectory_rejects_shape_nonfinite_and_bad_quaternion():
    base = _trajectory()
    with pytest.raises(gate.DynamicsGateError, match=r"joint_pos shape"):
        gate.MotionTrajectory(
            joint_pos=np.zeros((7, 30)),
            joint_vel=base.joint_vel,
            root_pos_w=base.root_pos_w,
            root_quat_wxyz=base.root_quat_wxyz,
            root_lin_vel_w=base.root_lin_vel_w,
            root_ang_vel_w=base.root_ang_vel_w,
            fps=50,
        )
    bad = np.array(base.joint_pos, copy=True)
    bad[2, 5] = np.nan
    with pytest.raises(gate.DynamicsGateError, match="finite"):
        gate.MotionTrajectory(
            joint_pos=bad,
            joint_vel=base.joint_vel,
            root_pos_w=base.root_pos_w,
            root_quat_wxyz=base.root_quat_wxyz,
            root_lin_vel_w=base.root_lin_vel_w,
            root_ang_vel_w=base.root_ang_vel_w,
            fps=50,
        )
    zero_quat = np.array(base.root_quat_wxyz, copy=True)
    zero_quat[3] = 0
    with pytest.raises(gate.DynamicsGateError, match="zero-norm"):
        gate.MotionTrajectory(
            joint_pos=base.joint_pos,
            joint_vel=base.joint_vel,
            root_pos_w=base.root_pos_w,
            root_quat_wxyz=zero_quat,
            root_lin_vel_w=base.root_lin_vel_w,
            root_ang_vel_w=base.root_ang_vel_w,
            fps=50,
        )


def test_com_velocity_to_origin_velocity_uses_world_cross_product():
    # Body is yawed +90deg.  Its body-frame inertial offset +X is therefore
    # world +Y.  omega=+Z gives omega x r = world -X.
    frames = 2
    yaw90 = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    quat = np.repeat(yaw90[None, :], frames, axis=0)
    omega = np.repeat(np.array([[0.0, 0.0, 2.0]]), frames, axis=0)
    ipos = np.array([0.25, 0.0, 0.0])
    v_origin = np.repeat(np.array([[1.0, 0.3, -0.2]]), frames, axis=0)
    offset_w = np.array([0.0, 0.25, 0.0])
    v_com = v_origin + np.cross(omega, offset_w)
    recovered = gate.com_velocity_to_origin_velocity(v_com, omega, quat, ipos)
    assert np.allclose(recovered, v_origin, atol=1e-12)


def test_limit_screen_reports_position_and_qdot_per_frame():
    frames, joints = 4, len(gate.RUNTIME_JOINT_NAMES)
    q = np.zeros((frames, joints))
    qd = np.zeros_like(q)
    lo = -np.ones(joints)
    hi = np.ones(joints)
    velocity = np.full(joints, 2.0)
    q[2, 3] = 1.1
    qd[1, 5] = -3.0
    result = gate.evaluate_limits(
        q, qd, lo, hi, velocity, gate.GateThresholds()
    )
    assert result["pass"] is False
    assert result["position"]["violation_frames"] == [2]
    assert result["position"]["peak_joint"] == gate.RUNTIME_JOINT_NAMES[3]
    assert result["position"]["peak_excess_rad"] == pytest.approx(0.1)
    assert result["velocity"]["violation_frames"] == [1]
    assert result["velocity"]["peak_joint"] == gate.RUNTIME_JOINT_NAMES[5]
    assert result["velocity"]["peak_utilization"] == pytest.approx(1.5)
    assert result["velocity"]["per_frame_peak_utilization"] == [0.0, 1.5, 0.0, 0.0]


def test_torque_interpretation_only_certifies_contact_free_zero_root_residual():
    thresholds = gate.GateThresholds(
        root_force_zero_tolerance_n=1e-3,
        root_torque_zero_tolerance_nm=1e-3,
    )
    force = np.zeros((5, 3))
    torque = np.zeros((5, 3))
    contact = np.zeros(5, dtype=int)
    mask = np.array([False, True, True, True, False])
    ok = gate.assess_torque_interpretation(
        direct_actuation_verified=True,
        root_force_w=force,
        root_torque_local=torque,
        contact_count=contact,
        thresholds=thresholds,
        evaluated_mask=mask,
    )
    assert ok["valid"] is True
    assert ok["label"] == "certified_contact_free_direct_motor_torque"

    contact[2] = 1
    force[3, 0] = 0.01
    torque[1, 2] = 0.02
    bad = gate.assess_torque_interpretation(
        direct_actuation_verified=True,
        root_force_w=force,
        root_torque_local=torque,
        contact_count=contact,
        thresholds=thresholds,
        evaluated_mask=mask,
    )
    assert bad["valid"] is False
    assert bad["label"] == "uncertified_contact_disabled_joint_effort_proxy"
    assert len(bad["reasons"]) == 3
    assert bad["contact_frames"] == [2]

    # The default certification mask covers endpoints too.
    endpoint_contact = np.zeros(5, dtype=int)
    endpoint_contact[0] = 1
    endpoint = gate.assess_torque_interpretation(
        direct_actuation_verified=True,
        root_force_w=np.zeros((5, 3)),
        root_torque_local=np.zeros((5, 3)),
        contact_count=endpoint_contact,
        thresholds=thresholds,
    )
    assert endpoint["valid"] is False
    assert endpoint["contact_frames"] == [0]

    unbound = gate.assess_torque_interpretation(
        direct_actuation_verified=True,
        plant_identity_bound=False,
        root_force_w=np.zeros((5, 3)),
        root_torque_local=np.zeros((5, 3)),
        contact_count=np.zeros(5, dtype=int),
        thresholds=thresholds,
    )
    assert unbound["valid"] is False
    assert any("not both bound" in reason for reason in unbound["reasons"])


def test_thresholds_and_limit_shapes_fail_closed():
    with pytest.raises(gate.DynamicsGateError, match="non-negative"):
        gate.GateThresholds(self_penetration_tolerance_m=-1)
    with pytest.raises(gate.DynamicsGateError, match="31 finite"):
        gate.evaluate_limits(
            np.zeros((3, 31)),
            np.zeros((3, 31)),
            np.zeros(30),
            np.ones(31),
            np.ones(31),
            gate.GateThresholds(),
        )


def test_cli_failure_is_json_and_exit_two(tmp_path, capsys):
    code = gate.main(
        ["--motion", str(tmp_path / "missing.npz"), "--mjcf", str(tmp_path / "missing.xml")]
    )
    report = __import__("json").loads(capsys.readouterr().out)
    assert code == 2
    assert report["verdict"] == "FAIL"
    assert report["screen_pass"] is False
    assert (
        "does not exist" in report["error"]
        or "mujoco is not installed" in report["error"]
    )


MJCF = REPO / gate.DEFAULT_MJCF_REL
needs_vendor = pytest.mark.skipif(
    gate.mujoco is None or not MJCF.is_file(),
    reason=f"mujoco installed={gate.mujoco is not None}, vendor MJCF={MJCF.is_file()}",
)


@pytest.fixture(scope="module")
def plant():
    diagnostic = gate.load_plant(MJCF)
    digest = hashlib.sha256(MJCF.read_bytes()).hexdigest()
    return gate.load_plant(
        MJCF,
        expected_mjcf_sha256=digest,
        expected_compiled_signature=diagnostic.compiled_signature_sha256,
    )


def _trajectory_from_qpos(
    plant: gate.PlantModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    fps: float,
) -> gate.MotionTrajectory:
    model = plant.contact_model
    root_quat = np.asarray(qpos[:, 3:7], np.float64)
    root_ang_w = np.zeros((len(qpos), 3))
    for frame in range(len(qpos)):
        rotation = gate._quat_to_mat_wxyz(root_quat[frame])
        root_ang_w[frame] = rotation @ qvel[frame, 3:6]
    return gate.MotionTrajectory(
        joint_pos=qpos[:, plant.joint_qposadr],
        joint_vel=qvel[:, plant.joint_dofadr],
        root_pos_w=qpos[:, :3],
        root_quat_wxyz=root_quat,
        root_lin_vel_w=qvel[:, :3],
        root_ang_vel_w=root_ang_w,
        fps=fps,
        source=f"synthetic:{model.nq}/{model.nv}",
    )


@needs_vendor
def test_vendor_model_hash_binding_and_grounded_stand_fail_closed(plant):
    digest = hashlib.sha256(MJCF.read_bytes()).hexdigest()
    rebound = gate.load_plant(
        MJCF,
        expected_mjcf_sha256=digest,
        expected_compiled_signature=plant.compiled_signature_sha256,
    )
    assert rebound.mjcf_sha256 == digest
    with pytest.raises(gate.DynamicsGateError, match="SHA mismatch"):
        gate.load_plant(MJCF, expected_mjcf_sha256="0" * 64)

    model = plant.contact_model
    frames = 7
    qpos0 = np.asarray(model.key_qpos[0], np.float64)
    qpos = np.repeat(qpos0[None, :], frames, axis=0)
    qvel = np.zeros((frames, model.nv))
    report = gate.analyze_trajectory(
        plant, _trajectory_from_qpos(plant, qpos, qvel, 50.0)
    )
    interpretation = report["checks"]["inverse_dynamics"]["torque_interpretation"]
    assert interpretation["valid"] is False
    assert report["verdict"] in {"INCOMPLETE_FAIL_CLOSED", "FAIL"}
    assert report["screen_pass"] is False
    assert interpretation["root_force_peak_n"] > 0.5 * plant.weight_n
    assert "not a balance certificate" in report["non_claims"]


@needs_vendor
def test_schema2_adapter_checks_all_32_body_channels(plant, tmp_path):
    model = plant.contact_model
    frames = 5
    qpos = np.repeat(np.asarray(model.key_qpos[0], np.float64)[None, :], frames, axis=0)
    data = gate.mujoco.MjData(model)
    body_pos = np.zeros((frames, 32, 3))
    body_quat = np.zeros((frames, 32, 4))
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = 0.0
        gate.mujoco.mj_forward(model, data)
        body_pos[frame] = np.asarray(data.xpos)[plant.runtime_body_ids]
        body_quat[frame] = np.asarray(data.xquat)[plant.runtime_body_ids]
    payload = {
        "fps": np.array([50], np.int64),
        "joint_pos": qpos[:, plant.joint_qposadr].astype(np.float32),
        "joint_vel": np.zeros((frames, 31), np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": np.zeros((frames, 32, 3), np.float32),
        "body_ang_vel_w": np.zeros((frames, 32, 3), np.float32),
        "body_names": np.asarray(plant.runtime_body_names),
        "kinematics_schema_version": np.array(2, np.int64),
        "body_pos_point": np.array("link_origin"),
        "body_lin_vel_point": np.array("center_of_mass"),
    }
    good = tmp_path / "good.npz"
    np.savez(good, **payload)
    trajectory = gate.trajectory_from_schema2_npz(good, plant)
    assert trajectory.schema2_body_validation is not None
    assert trajectory.schema2_body_validation["position_peak_error_m"] < 1e-6

    corrupted = dict(payload)
    corrupted["body_pos_w"] = np.array(payload["body_pos_w"], copy=True)
    corrupted["body_pos_w"][2, 17, 0] += 0.1
    bad = tmp_path / "bad.npz"
    np.savez(bad, **corrupted)
    with pytest.raises(gate.DynamicsGateError, match="body channels disagree"):
        gate.trajectory_from_schema2_npz(bad, plant)

    nonfinite = dict(payload)
    nonfinite["body_lin_vel_w"] = np.array(payload["body_lin_vel_w"], copy=True)
    nonfinite["body_lin_vel_w"][1, 9, 2] = np.nan
    bad_nan = tmp_path / "bad_nan.npz"
    np.savez(bad_nan, **nonfinite)
    with pytest.raises(gate.DynamicsGateError, match="NaN/Inf"):
        gate.trajectory_from_schema2_npz(bad_nan, plant)


def _moving_schema2_payload(
    plant: gate.PlantModel,
    *,
    frames: int = 9,
    fps: float = 50.0,
) -> dict[str, np.ndarray]:
    """Build exact analytic schema channels with intentionally coarse pose samples."""

    model = plant.contact_model
    time = np.arange(frames, dtype=np.float64) / fps
    qpos = np.repeat(
        np.asarray(model.key_qpos[0], np.float64)[None, :], frames, axis=0
    )
    qvel = np.zeros((frames, int(model.nv)), dtype=np.float64)

    # At omega*dt=0.7, a centred 50 Hz pose difference differs materially from
    # the exact instantaneous derivative even though both come from the same
    # smooth sinusoid.  This is the regression that the old verifier rejected.
    amplitude = 0.05
    omega = 35.0
    qpos[:, plant.joint_qposadr[0]] += amplitude * np.sin(omega * time)
    qvel[:, plant.joint_dofadr[0]] = amplitude * omega * np.cos(omega * time)

    body_pos = np.empty((frames, 32, 3), dtype=np.float64)
    body_quat = np.empty((frames, 32, 4), dtype=np.float64)
    body_lin = np.empty((frames, 32, 3), dtype=np.float64)
    body_ang = np.empty((frames, 32, 3), dtype=np.float64)
    data = gate.mujoco.MjData(model)
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        gate.mujoco.mj_forward(model, data)
        body_pos[frame] = np.asarray(data.xpos)[plant.runtime_body_ids]
        body_quat[frame] = np.asarray(data.xquat)[plant.runtime_body_ids]
        for column, body_id in enumerate(plant.runtime_body_ids.tolist()):
            velocity = np.zeros(6, dtype=np.float64)
            gate.mujoco.mj_objectVelocity(
                model,
                data,
                gate.mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                velocity,
                0,
            )
            body_ang[frame, column] = velocity[:3]
            body_lin[frame, column] = velocity[3:]
    return {
        "fps": np.array([fps], np.float32),
        "joint_pos": qpos[:, plant.joint_qposadr].astype(np.float32),
        "joint_vel": qvel[:, plant.joint_dofadr].astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
        "body_names": np.asarray(plant.runtime_body_names),
        "kinematics_schema_version": np.array(2, np.int64),
        "body_pos_point": np.array("link_origin"),
        "body_lin_vel_point": np.array("center_of_mass"),
    }


@needs_vendor
def test_schema2_adapter_uses_instantaneous_velocity_not_pose_difference(
    plant, tmp_path
):
    payload = _moving_schema2_payload(plant)
    motion = tmp_path / "moving_analytic_schema2.npz"
    np.savez(motion, **payload)

    trajectory = gate.trajectory_from_schema2_npz(motion, plant)
    receipt = trajectory.schema2_body_validation
    assert receipt is not None
    assert receipt["velocity_jacobian_min_rank"] == plant.nv
    assert receipt["linear_velocity_peak_abs_error_m_s"] < 1.0e-5
    assert receipt["angular_velocity_peak_abs_error_rad_s"] < 1.0e-5
    assert receipt["joint_velocity_peak_abs_mismatch_rad_s"] < 1.0e-5

    qpos, qvel, _ = gate.build_generalized_state(plant, trajectory)
    consistency = gate.velocity_consistency_report(
        plant, trajectory, qpos, qvel
    )
    assert consistency["pass"] is True
    assert consistency["method"].startswith("schema2_all_32_body")
    assert (
        consistency["sampled_pose_finite_difference_diagnostic"][
            "joint_velocity_peak_abs_mismatch_rad_s"
        ]
        > 0.05
    )


@needs_vendor
@pytest.mark.parametrize("corruption", ["body_velocity", "joint_velocity"])
def test_schema2_adapter_rejects_velocity_cross_channel_corruption(
    plant, tmp_path, corruption
):
    payload = _moving_schema2_payload(plant)
    corrupted = {key: np.array(value, copy=True) for key, value in payload.items()}
    if corruption == "body_velocity":
        corrupted["body_lin_vel_w"][3, 17, 0] += 0.01
    else:
        corrupted["joint_vel"][3, 0] += 0.01
    motion = tmp_path / f"bad_{corruption}.npz"
    np.savez(motion, **corrupted)
    with pytest.raises(gate.DynamicsGateError, match="body channels disagree"):
        gate.trajectory_from_schema2_npz(motion, plant)


@needs_vendor
def test_vendor_free_fall_can_certify_contact_free_torque(plant):
    model = plant.contact_model
    fps, frames = 50.0, 9
    time = np.arange(frames) / fps
    qpos0 = np.asarray(model.key_qpos[0], np.float64).copy()
    qpos0[2] += 5.0
    qpos = np.repeat(qpos0[None, :], frames, axis=0)
    gravity = np.asarray(model.opt.gravity, np.float64)
    qpos[:, :3] = qpos0[:3] + 0.5 * gravity[None, :] * time[:, None] ** 2
    qvel = np.zeros((frames, model.nv))
    qvel[:, :3] = gravity[None, :] * time[:, None]
    report = gate.analyze_trajectory(
        plant, _trajectory_from_qpos(plant, qpos, qvel, fps)
    )
    interpretation = report["checks"]["inverse_dynamics"]["torque_interpretation"]
    assert interpretation["valid"] is True, interpretation
    assert interpretation["root_force_peak_n"] < 1e-4
    assert report["checks"]["inverse_dynamics"][
        "joint_effort_proxy_peak_utilization"
    ] < 1e-5
    assert report["verdict"] == "PASS"
