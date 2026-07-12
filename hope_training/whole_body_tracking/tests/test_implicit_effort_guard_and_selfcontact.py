"""Regressions for the implicit total-torque effort guard and the self-contact diagnostic.

Needs the ``mujoco`` python bindings (tiny synthetic models, CPU, milliseconds); no Isaac,
Torch, ONNX, or onnxruntime installation is required.

Background (Isaac<->MuJoCo parity audit 2026-07-13): Isaac's ImplicitActuator clamps the TOTAL
drive force kp*(q_des-q)-kd*qd to the PhysX drive max force, while the evaluator's implicit mode
integrates kd as passive dof_damping that no effort limit touches — wrist braking torque alone
can reach kd*vel_limit = 2.0*12.7 ~ 25 Nm against a 6 Nm training cap. The guard makes that
saturation visible (diagnostic) and fatal on the formal path, mirroring the joint-velocity-limit
pattern. Self-collision is OFF in Isaac training and ON in the vendor MJCF; the scan counts where
the two plants actually diverge without touching physics or scoring.
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


# One hinge joint with vendor-wrist-like numbers: kd=2.0 in passive damping (implicit profile),
# +/-6 Nm motor ctrlrange, and enough inertia (I=1.0) that a 12.7 rad/s spin survives one 5 ms
# implicitfast substep, so the unclamped braking torque kd*qd ~ 25 Nm is observable post-step.
_HINGE_XML = """
<mujoco>
  <option timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <body name="b0">
      <joint name="j0" type="hinge" axis="0 0 1" damping="2.0"/>
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
        # (implicit dof indices, effective passive damping, bound effort limits) — the
        # precomputed guard tuple MujocoRobot.__init__ arms for bound schema-3 implicit plants.
        _effort_guard=(np.array([0]), np.array([2.0]), np.array([6.0])),
        effort_limit_hit_count=0, effort_limit_peak_ratio=0.0,
        allow_effort_limit_proxy=True,
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
    # kd*qd ~ 2.0*12.7 = 25.4 Nm vs the 6 Nm total-torque cap Isaac enforces -> ratio ~ 4.2.
    assert robot.effort_limit_peak_ratio == pytest.approx(25.4 / 6.0, rel=0.05)


def test_formal_path_fail_louds_instead_of_booking_a_non_exact_trajectory():
    robot = _hinge_robot(allow_effort_limit_proxy=False)
    robot.data.qvel[0] = 12.7
    with pytest.raises(SystemExit, match="exceeded the bound"):
        _swing(robot)


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


# Two free spheres overlapping each other above a floor plane: exactly one robot-robot contact;
# floor contacts (worldbody geom) must not count as self-collision.
_SELFCON_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="b1" pos="0 0 0.05">
      <freejoint/><geom name="g1" type="sphere" size="0.06" mass="1"/>
    </body>
    <body name="b2" pos="0.05 0 0.05">
      <freejoint/><geom name="g2" type="sphere" size="0.06" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_self_contact_scan_counts_robot_pairs_and_ignores_floor():
    model = mujoco.MjModel.from_xml_string(_SELFCON_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = SimpleNamespace(mj=mujoco, model=model, data=data)
    count, max_pen, worst = M.MujocoRobot.self_contact_scan(robot)
    floor_contacts = sum(
        1 for i in range(data.ncon)
        if model.geom_bodyid[data.contact[i].geom1] == 0
        or model.geom_bodyid[data.contact[i].geom2] == 0
    )
    assert floor_contacts >= 1          # the spheres do rest on the floor...
    assert count == 1                   # ...but only the sphere-sphere pair is a self-contact
    assert worst == "g1~g2"
    assert max_pen == pytest.approx(0.07, abs=0.02)  # 0.12 combined radius - 0.05 separation


def test_self_contact_scan_is_empty_without_robot_pairs():
    model = mujoco.MjModel.from_xml_string(_SELFCON_XML)
    data = mujoco.MjData(model)
    data.qpos[7 + 0] = 2.0  # move b2 away along x (second freejoint qpos block)
    mujoco.mj_forward(model, data)
    robot = SimpleNamespace(mj=mujoco, model=model, data=data)
    count, max_pen, worst = M.MujocoRobot.self_contact_scan(robot)
    assert (count, max_pen, worst) == (0, 0.0, "")
