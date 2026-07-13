"""Regressions for the implicit total-torque effort guard and the self-contact diagnostic.

Needs the ``mujoco`` python bindings (tiny synthetic models, CPU, milliseconds); no Isaac,
Torch, ONNX, or onnxruntime installation is required.

Background (Isaac<->MuJoCo parity audit 2026-07-13): Isaac's ImplicitActuator clamps the TOTAL
drive force kp*(q_des-q)-kd*qd to the PhysX drive max force.  Bound MuJoCo execution must send that
same clipped total through the motor; kd-as-passive-damping cannot share the limit and is explicitly
inexact. Self-collision is OFF in Isaac training and ON in the vendor MJCF; the robot-only scan
excludes dynamic balls and other non-robot bodies, and formal BankExam fails closed on a hit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "mujoco_eval_onnx.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mj_effort_guard_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_module()


# One hinge joint with vendor-wrist-like numbers and +/-6 Nm motor ctrlrange.  The exact bound
# profile keeps passive damping zero and sends clip(P-D) as the motor control.
_HINGE_XML = """
<mujoco>
  <option timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <body name="b0">
      <joint name="j0" type="hinge" axis="0 0 1" damping="0.0"/>
      <geom type="sphere" size="0.5" mass="10.0"/>
    </body>
  </worldbody>
  <actuator><motor name="j0_motor" joint="j0" ctrlrange="-6 6"/></actuator>
</mujoco>
"""


def _hinge_robot(**overrides):
    model = mujoco.MjModel.from_xml_string(_HINGE_XML)
    data = mujoco.MjData(model)
    robot = SimpleNamespace(
        mj=mujoco, model=model, data=data,
        qadr=np.array([0]), vadr=np.array([0]), act_id=np.array([0]),
        implicit_mask=np.array([True]), explicit_mask=np.array([False]),
        ctrl_lo=np.array([-6.0]), ctrl_hi=np.array([6.0]),
        # (implicit dof indices, bound effort limits) — the precomputed tuple for bound schema-3.
        _effort_guard=(np.array([0]), np.array([6.0])),
        effort_limit_hit_count=0, effort_limit_peak_ratio=0.0,
        allow_effort_limit_proxy=True,
        implicit_effort_execution_mode="isaac_total_pd_clip_exact",
        implicit_effort_proxy_nonexact=False,
        fail_on_self_contact=False,
        self_contact_scan=lambda: (0, 0.0, ""),
        joint_velocity_limits=None, velocity_limit_hit_count=0,
        velocity_limit_peak_ratio=0.0, allow_velocity_limit_proxy=True,
    )
    for key, value in overrides.items():
        setattr(robot, key, value)
    return robot


def _swing(robot, decimation=1):
    return M.MujocoRobot.apply_pd_and_step(
        robot, np.zeros(1), kp=np.array([0.0]), kd=np.array([2.0]), decimation=decimation
    )


def test_implicit_kd_braking_over_effort_limit_is_counted_with_wrist_ratio():
    robot = _hinge_robot()
    robot.data.qvel[0] = 12.7  # Isaac wrist velocity cap; legal in training
    _swing(robot)
    assert robot.effort_limit_hit_count == 1
    assert robot.data.ctrl[0] == pytest.approx(-6.0)
    # Raw kd*qd ~25.4 Nm is measured, while the applied total is exactly clipped to -6 Nm.
    assert robot.effort_limit_peak_ratio == pytest.approx(25.4 / 6.0, rel=0.05)


def test_formal_path_executes_the_same_clipped_total_instead_of_failing_on_saturation():
    robot = _hinge_robot(allow_effort_limit_proxy=False)
    robot.data.qvel[0] = 12.7
    _swing(robot)
    assert robot.data.ctrl[0] == pytest.approx(-6.0)
    assert robot.effort_limit_hit_count == 1


@pytest.mark.parametrize(
    ("target", "kp", "qd", "kd", "expected_ctrl", "expected_saturations"),
    [
        (8.0, 1.0, 1.0, 2.0, 6.0, 0),    # cancellation reaches +L exactly
        (5.0, 1.0, -2.0, 2.0, 6.0, 1),   # P and -D add; total must clip
        (-2.0, 1.0, 2.0, 2.0, -6.0, 0),  # negative exact boundary
    ],
)
def test_runtime_motor_command_uses_total_pd_clip_in_all_sign_quadrants(
    target, kp, qd, kd, expected_ctrl, expected_saturations
):
    robot = _hinge_robot()
    robot.data.qvel[0] = qd
    M.MujocoRobot.apply_pd_and_step(
        robot,
        np.asarray([target]),
        kp=np.asarray([kp]),
        kd=np.asarray([kd]),
        decimation=1,
    )
    assert robot.data.ctrl[0] == pytest.approx(expected_ctrl)
    assert robot.effort_limit_hit_count == expected_saturations


def test_slow_joint_never_trips_and_diagnostic_never_mutates_state():
    robot = _hinge_robot()
    robot.data.qvel[0] = 1.0  # kd*qd = 2 Nm < 6 Nm
    control = _hinge_robot(_effort_guard=None)
    control.data.qvel[0] = 1.0
    _swing(robot)
    _swing(control)
    assert robot.effort_limit_hit_count == 0
    assert 0.0 < robot.effort_limit_peak_ratio < 1.0
    # Guard is read-only: with and without it the physics trajectory is byte-identical.
    np.testing.assert_array_equal(robot.data.qpos, control.data.qpos)
    np.testing.assert_array_equal(robot.data.qvel, control.data.qvel)


def test_unarmed_guard_is_skipped():
    # Explicit-only or non-bound plants never arm the guard (MujocoRobot.__init__ leaves
    # _effort_guard None); the fast wrist spin must book nothing.
    robot = _hinge_robot(
        implicit_mask=np.array([False]), explicit_mask=np.array([True]), _effort_guard=None
    )
    robot.data.qvel[0] = 12.7
    _swing(robot)
    assert robot.effort_limit_hit_count == 0
    assert robot.effort_limit_peak_ratio == 0.0


# Two sibling geoms in a pelvis subtree overlap, while a separate dynamic ball overlaps them too.
# Only the robot-robot pair is self-contact; floor and robot-ball contacts must not count.
_SELFCON_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis_link" pos="0 0 0.05">
      <freejoint/>
      <body name="robot_1"><geom name="g1" type="sphere" size="0.06" mass="1"/></body>
      <body name="robot_2" pos="0.05 0 0"><geom name="g2" type="sphere" size="0.06" mass="1"/></body>
    </body>
    <body name="ball" pos="0.025 0 0.05">
      <freejoint/><geom name="ball_geom" type="sphere" size="0.02" mass="0.0027"/>
    </body>
  </worldbody>
</mujoco>
"""


def _selfcon_robot(model, data):
    robot_body_names = {"pelvis_link", "robot_1", "robot_2"}
    robot_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in robot_body_names
    }
    robot_geom_mask = np.asarray(
        [int(body_id) in robot_body_ids for body_id in model.geom_bodyid], dtype=bool
    )
    return SimpleNamespace(
        mj=mujoco, model=model, data=data, robot_geom_mask=robot_geom_mask
    )


def test_self_contact_scan_counts_robot_pairs_and_ignores_floor():
    model = mujoco.MjModel.from_xml_string(_SELFCON_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = _selfcon_robot(model, data)
    count, max_pen, worst = M.MujocoRobot.self_contact_scan(robot)
    floor_contacts = sum(
        1 for i in range(data.ncon)
        if model.geom_bodyid[data.contact[i].geom1] == 0
        or model.geom_bodyid[data.contact[i].geom2] == 0
    )
    assert floor_contacts >= 1
    assert any(
        "ball_geom" in {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2),
        }
        for i in range(data.ncon)
    )
    assert count == 1                   # robot-ball contacts are deliberately excluded
    assert worst == "g1~g2"
    assert max_pen == pytest.approx(0.07, abs=0.02)  # 0.12 combined radius - 0.05 separation


def test_self_contact_scan_is_empty_without_robot_pairs():
    model = mujoco.MjModel.from_xml_string(_SELFCON_XML)
    data = mujoco.MjData(model)
    # Move the entire robot freejoint away from the dynamic ball; then separate robot_2 by editing
    # the model body offset so no robot pair remains.
    robot2_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_2")
    model.body_pos[robot2_bid, 0] = 2.0
    mujoco.mj_forward(model, data)
    robot = _selfcon_robot(model, data)
    count, max_pen, worst = M.MujocoRobot.self_contact_scan(robot)
    assert (count, max_pen, worst) == (0, 0.0, "")
    # The remaining robot-ball contact is legal under the robot-only formal classifier.
    M.enforce_self_contact_policy(
        fail_closed=True, count=count, penetration_m=max_pen, pair=worst
    )
