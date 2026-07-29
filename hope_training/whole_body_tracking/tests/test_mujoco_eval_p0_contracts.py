"""Pure/static P0 regression tests for ``scripts/mujoco_eval_onnx.py``.

No MuJoCo, Isaac, Torch, ONNX, or onnxruntime installation is required.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "mujoco_eval_onnx.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mj_eval_p0_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load_module()


@pytest.mark.parametrize(
    ("metadata", "clip_count", "expected"),
    [
        ({"motion_body_lin_vel_points": "center_of_mass"}, 1, ("center_of_mass",)),
        ({"motion_body_lin_vel_points": "link_origin"}, 1, ("link_origin",)),
        (
            {"motion_kinematics_exact": "0",
             "motion_body_lin_vel_points": "center_of_mass,link_origin"},
            2,
            ("center_of_mass", "link_origin"),
        ),
        ({"motion_kinematics_exact": "1"}, 2, ("center_of_mass", "center_of_mass")),
        ({"motion_kinematics_exact": "0"}, 2, None),
        ({"motion_allow_legacy_link_origin_velocity": "1"}, 2, None),
        ({}, None, None),
    ],
)
def test_motion_metadata_resolves_explicit_per_clip_points_without_aggregate_guessing(
    metadata, clip_count, expected
):
    assert M.motion_body_lin_vel_points_from_metadata(
        metadata, clip_count=clip_count
    ) == expected


@pytest.mark.parametrize(
    "metadata",
    [
        {"motion_kinematics_exact": "true"},
        {"motion_allow_legacy_link_origin_velocity": "false"},
        {"motion_body_lin_vel_points": "center_of_mass,unknown"},
    ],
)
def test_motion_metadata_rejects_noncanonical_velocity_point_flags(metadata):
    with pytest.raises(ValueError, match="must be 0\\|1|unknown values"):
        M.motion_body_lin_vel_points_from_metadata(metadata, clip_count=2)


def test_teacher_reference_velocity_points_reject_ambiguous_and_wrong_clip_count():
    with pytest.raises(ValueError, match="ambiguous"):
        M.validated_motion_body_lin_vel_points(None, 2)
    with pytest.raises(ValueError, match="count 1 != clip count 2"):
        M.validated_motion_body_lin_vel_points(("center_of_mass",), 2)
    assert M.validated_motion_body_lin_vel_points(
        ("center_of_mass", "link_origin"), 2
    ) == ("center_of_mass", "link_origin")


def test_freejoint_origin_velocity_conversion_is_point_explicit_and_numpy_only():
    quat = np.asarray([0.91, 0.17, -0.26, 0.28], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    declared = np.asarray([0.72, -0.41, 0.19], dtype=np.float64)
    omega = np.asarray([1.13, -0.67, 0.84], dtype=np.float64)
    body_ipos = np.asarray([-0.003, 0.012, -0.127], dtype=np.float64)
    expected_com = declared - np.cross(omega, M.mat_from_quat(quat) @ body_ipos)
    np.testing.assert_allclose(
        M.freejoint_origin_lin_vel_w(
            declared, omega, quat, body_ipos, "center_of_mass"
        ),
        expected_com,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        M.freejoint_origin_lin_vel_w(
            declared, omega, quat, body_ipos, "link_origin"
        ),
        declared,
    )
    with pytest.raises(ValueError, match="invalid root linear-velocity point"):
        M.freejoint_origin_lin_vel_w(declared, omega, quat, body_ipos, "ambiguous")


@pytest.mark.parametrize(
    ("proportional", "damping", "limit", "expected"),
    [
        # Opposite-sign cancellation before clipping: old clip(P)-D incorrectly produced 4.
        (8.0, 2.0, 6.0, 6.0),
        # Same-sign contributions share one limit: old clip(P)-D incorrectly produced 9.
        (5.0, -4.0, 6.0, 6.0),
        # Pure damping, including the negative saturation quadrant.
        (0.0, 8.0, 6.0, -6.0),
        # Exact boundary is not altered.
        (-2.0, 4.0, 6.0, -6.0),
    ],
)
def test_isaac_total_pd_effort_clips_after_combining_terms(
    proportional, damping, limit, expected
):
    actual = M.isaac_total_pd_effort(
        np.asarray([proportional]), np.asarray([damping]), np.asarray([limit])
    )
    np.testing.assert_array_equal(actual, np.asarray([expected]))


def test_formal_implicit_effort_plan_rejects_passive_or_unbound_proxy():
    assert M.resolve_implicit_effort_execution(
        has_implicit=True,
        effort_limits_bound=True,
        passive_damping=np.zeros(2),
        allow_inexact_proxy=False,
    ) == "isaac_total_pd_clip_exact"
    with pytest.raises(ValueError, match="passive damping cannot realize"):
        M.resolve_implicit_effort_execution(
            has_implicit=True,
            effort_limits_bound=True,
            passive_damping=np.asarray([0.0, 0.1]),
            allow_inexact_proxy=False,
        )
    with pytest.raises(ValueError, match="no bound effort limits"):
        M.resolve_implicit_effort_execution(
            has_implicit=True,
            effort_limits_bound=False,
            passive_damping=np.zeros(2),
            allow_inexact_proxy=False,
        )
    assert M.resolve_implicit_effort_execution(
        has_implicit=True,
        effort_limits_bound=True,
        passive_damping=np.asarray([0.0, 0.1]),
        allow_inexact_proxy=True,
    ) == "total_pd_clip_plus_passive_damping_inexact"


def test_formal_self_contact_policy_fails_and_diagnostic_policy_only_records():
    M.enforce_self_contact_policy(
        fail_closed=False, count=1, penetration_m=0.002, pair="arm~torso"
    )
    with pytest.raises(SystemExit, match="enabled_self_collisions=False"):
        M.enforce_self_contact_policy(
            fail_closed=True, count=1, penetration_m=0.002, pair="arm~torso"
        )


def _fake_pd_robot_with_substep_contacts(samples, *, fail_closed):
    samples = iter(samples)
    step_count = [0]

    def mj_step(_model, _data):
        step_count[0] += 1

    robot = SimpleNamespace(
        data=SimpleNamespace(
            qpos=np.zeros(1, dtype=np.float64),
            qvel=np.zeros(1, dtype=np.float64),
            ctrl=np.zeros(1, dtype=np.float64),
        ),
        model=object(),
        mj=SimpleNamespace(mj_step=mj_step),
        qadr=np.asarray([0]),
        vadr=np.asarray([0]),
        act_id=np.asarray([0]),
        explicit_mask=np.asarray([True]),
        ctrl_lo=np.asarray([-10.0]),
        ctrl_hi=np.asarray([10.0]),
        _effort_guard=None,
        joint_velocity_limits=None,
        fail_on_self_contact=bool(fail_closed),
        self_contact_scan=lambda: next(samples),
    )
    return robot, step_count


def test_formal_self_contact_refuses_the_first_transient_physics_substep():
    robot, step_count = _fake_pd_robot_with_substep_contacts(
        [(1, 0.002, "arm~torso"), (0, 0.0, "")], fail_closed=True
    )
    with pytest.raises(SystemExit, match="enabled_self_collisions=False"):
        M.MujocoRobot.apply_pd_and_step(
            robot,
            np.zeros(1),
            kp=np.zeros(1),
            kd=np.zeros(1),
            decimation=2,
        )
    assert step_count == [1]
    assert robot.last_control_self_contact == {
        "physics_substeps": 1,
        "substeps_with_contact": 1,
        "contact_count": 1,
        "max_penetration_m": pytest.approx(0.002),
        "worst_pair": "arm~torso",
    }


def test_diagnostic_self_contact_retains_transient_substep_after_final_state_is_clean():
    robot, step_count = _fake_pd_robot_with_substep_contacts(
        [(2, 0.003, "hand~hip"), (0, 0.0, ""), (0, 0.0, "")],
        fail_closed=False,
    )
    M.MujocoRobot.apply_pd_and_step(
        robot,
        np.zeros(1),
        kp=np.zeros(1),
        kd=np.zeros(1),
        decimation=3,
    )
    assert step_count == [3]
    assert robot.last_control_self_contact == {
        "physics_substeps": 3,
        "substeps_with_contact": 1,
        "contact_count": 2,
        "max_penetration_m": pytest.approx(0.003),
        "worst_pair": "hand~hip",
    }


def test_exact_strike_clock_is_post_step_and_holds_stay_pinned():
    dt = 0.02
    # An action selected one frame before contact produces the contact state after physics.
    assert M.post_step_time_to_strike(dt, dt, clock_advances=True) == pytest.approx(0.0)
    # Reusing the actor-input tts would instead fire on the following action/state (the old bug).
    assert M.post_step_time_to_strike(0.0, dt, clock_advances=True) == pytest.approx(-dt)
    # Pre-swing holds do simulate physics but MotionCommand intentionally pins its clock.
    assert M.post_step_time_to_strike(0.64, dt, clock_advances=False) == pytest.approx(0.64)
    with pytest.raises(ValueError):
        M.post_step_time_to_strike(0.1, 0.0, clock_advances=True)


def test_hold_protocol_scope_is_shared_by_main_and_rollout_and_bank_always_holds():
    sampler = SimpleNamespace(schedule=())
    assert M.training_hold_protocol_active(
        reset_mode="multiswing", deploy_faithful_cfg=None, venue_sampler=None
    )
    assert M.training_hold_protocol_active(
        reset_mode="teleport", deploy_faithful_cfg=None, venue_sampler=sampler
    )
    assert not M.training_hold_protocol_active(
        reset_mode="multiswing", deploy_faithful_cfg={}, venue_sampler=None
    )
    assert not M.training_hold_protocol_active(
        reset_mode="teleport", deploy_faithful_cfg=None, venue_sampler=None
    )


def test_explicit_inexact_escape_can_never_produce_an_exact_bank_score():
    assert M.bank_evaluation_contract_exact(
        artifact_exact=True, schedule_exact=True, diagnostic_escape=False
    )
    assert not M.bank_evaluation_contract_exact(
        artifact_exact=True, schedule_exact=True, diagnostic_escape=True
    )
    assert not M.bank_evaluation_contract_exact(
        artifact_exact=False, schedule_exact=True, diagnostic_escape=False
    )


def test_bank_loader_resolves_standalone_module_without_isaac_package_import():
    path = Path(M.stage1_question_bank_module_path(ROOT))
    assert path.name == "stage1_question_bank.py"
    assert path.is_file()


def test_rollout_wires_post_step_clock_before_metrics_and_uses_separate_noise_rng():
    source = inspect.getsource(M.run_rollout)
    physics = source.index("robot.apply_pd_and_step")
    aligned_clock = source.index("post_step_time_to_strike")
    metrics = source.index("# --- metrics (post-step state) ---")
    assert physics < aligned_clock < metrics
    assert "attempt_action_noise_rng[0].standard_normal(31)" in source


def test_episode_resets_clear_mujoco_hidden_state_before_installing_qpos_qvel():
    """One-question/one-reset must clear solver/actuator history, not only pose."""

    for method in (M.MujocoRobot.reset_to_reference, M.MujocoRobot.reset_to_stand):
        source = inspect.getsource(method)
        reset = source.index("mj_resetData")
        qpos = source.index("self.data.qpos")
        forward = source.index("mj_forward")
        assert reset < qpos < forward


def test_onnx_normalization_is_inferred_from_obs_dataflow_not_metadata(monkeypatch):
    def node(op, inputs, outputs):
        return SimpleNamespace(op_type=op, input=inputs, output=outputs)

    graph = SimpleNamespace(node=[
        node("Sub", ["obs", "mean"], ["centered"]),
        node("Div", ["centered", "denom"], ["normalized"]),
        node("Gemm", ["normalized", "weight", "bias"], ["hidden"]),
    ])
    monkeypatch.setitem(
        sys.modules, "onnx",
        SimpleNamespace(load=lambda _path: SimpleNamespace(graph=graph)),
    )
    baked, _ = M.inspect_onnx_obs_normalization("unused.onnx")
    assert baked is True

    graph.node = [node("Gemm", ["obs", "weight", "bias"], ["hidden"])]
    baked, _ = M.inspect_onnx_obs_normalization("unused.onnx")
    assert baked is False


def test_bank_sampler_reset_restores_question_cursor_for_paired_noise_columns():
    calls = []
    sampler = SimpleNamespace(
        reset_counters=lambda: setattr(sampler, "planner_counter", 0),
        rewind_schedule=lambda: calls.append("rewind"),
        planner_counter=99,
    )
    M.reset_sampler_for_paired_rollout(sampler)
    assert sampler.planner_counter == 0
    assert calls == ["rewind"]


def test_action_noise_rng_cannot_perturb_paired_target_sequence():
    targets_a, noise_a = M.paired_rollout_rngs(17)
    targets_b, noise_b = M.paired_rollout_rngs(17)
    seq_a = []
    seq_b = []
    for _ in range(20):
        seq_a.append(int(targets_a.integers(0, 10000)))
        noise_a.standard_normal(31)  # a dithered column consumes action noise every step
        seq_b.append(int(targets_b.integers(0, 10000)))  # ns=0 consumes no action-noise draws
    assert seq_a == seq_b
    assert np.array_equal(
        M.paired_rollout_rngs(17)[1].standard_normal(31),
        M.paired_rollout_rngs(17)[1].standard_normal(31),
    )


def test_attempt_summary_keeps_hold_deaths_in_unconditional_denominator():
    records = [
        dict(clip=0, exact=False, exact_composite=False, reason="fall_hold"),
        dict(clip=0, exact=True, exact_composite=True, reason="completed"),
        dict(clip=1, exact=True, exact_composite=False, reason="fall_post_strike"),
        dict(clip=1, exact=False, exact_composite=False, reason="switch_pre_strike"),
    ]
    out = M.summarize_attempt_records(records, 2)
    assert out["n_attempts"] == 4
    assert out["n_reached_exact"] == 2
    assert out["n_composite"] == 1
    assert out["exact_reach_rate"] == pytest.approx(0.5)
    assert out["composite_rate_per_attempt"] == pytest.approx(0.25)
    assert out["composite_rate_given_exact"] == pytest.approx(0.5)
    assert out["finalize_reason_counts"]["fall_hold"] == 1
    assert out["per_clip"]["forehand"]["n_attempts"] == 2
    assert out["per_clip"]["backhand"]["n_attempts"] == 2


def test_continuity_product_metric_keeps_return_but_fails_post_strike_recovery():
    records = [
        dict(returned=True, reason="completed"),
        dict(returned=True, reason="fall_post_strike"),
        dict(returned=False, reason="completed"),
        # Terminal row has no scheduled next opportunity and is excluded from this denominator.
        dict(returned=True, reason="completed"),
    ]
    out = M.summarize_continuity_records(records)
    assert out == {
        "n_opportunities_with_scheduled_next": 3,
        "n_returned": 2,
        "n_recovered_to_next": 2,
        "n_returned_and_recovered_to_next": 1,
        "return_and_recover_rate": pytest.approx(1 / 3),
        "recover_rate_given_return": pytest.approx(1 / 2),
    }


def test_attempt_flags_never_mark_a_truncation_eligible():
    flags = M.attempt_ledger_flags(
        "truncated_pre_strike", ("rollout_step_budget",), scheduled_exam=True
    )
    assert flags == {
        "censored": True,
        "eligible": False,
        "physical_fall": False,
        "table_hit": False,
        "guard_reset": False,
    }
    fall = M.attempt_ledger_flags("fall_pre_strike", ("fall_tilt",), scheduled_exam=True)
    assert fall["eligible"] is True and fall["physical_fall"] is True
    guard = M.attempt_ledger_flags("fall_pre_strike", ("anchor_pos",), scheduled_exam=True)
    assert guard["eligible"] is True and guard["guard_reset"] is True
    table = M.attempt_ledger_flags(
        "table_hit_pre_strike",
        ("robot_hit_table", "fall_tilt", "anchor_pos"),
        scheduled_exam=True,
    )
    assert table["table_hit"] is True
    assert table["physical_fall"] is True
    assert table["guard_reset"] is False


def test_formal_bank_cap_is_proven_from_k_timeout_and_clip_lengths():
    schedule = [
        SimpleNamespace(clip=0, hold_steps=10),
        SimpleNamespace(clip=1, hold_steps=30),
    ]
    # max(timeout=500, longest nominal=30+270+2) * K
    assert M.formal_bank_step_cap(schedule, [200, 270], 500) == 1000
    # A very long hold/clip raises the bound above the episode fallback.
    assert M.formal_bank_step_cap(schedule, [200, 600], 100) == 1264


def test_formal_bank_execution_contract_requires_schema3_dt_limits_and_clamp():
    limits = np.column_stack((np.full(31, -1.0), np.full(31, 1.0)))
    policy = SimpleNamespace(
        training_contract_exact="1",
        training_contract_schema_version="3",
        training_contract_sha256="b" * 64,
        source_checkpoint_sha256="a" * 64,
        qdes_joint_pos_limits=limits,
        physics_step_dt_s=0.005,
        policy_step_dt_s=0.02,
        control_decimation=4,
        qdes_clamp_meta=True,
        joint_actuator_types=("implicit",) * 31,
        joint_effort_limits=np.full(31, 24.0),
        joint_armature=np.full(31, 0.01),
        joint_friction_coefficients=np.zeros(31),
        joint_velocity_limits=np.full(31, 12.0),
        joint_friction_backend="physx",
        joint_friction_semantics="load_dependent_spatial_force_coefficient",
        joint_friction_units="dimensionless",
    )
    lo, hi = M.validate_formal_bank_execution_contract(
        policy, physics_step_dt_s=0.005, policy_step_dt_s=0.02,
        control_decimation=4, qdes_clamp=True,
    )
    assert np.array_equal(lo, limits[:, 0]) and np.array_equal(hi, limits[:, 1])
    policy.training_contract_schema_version = "2"
    with pytest.raises(SystemExit, match="exactly 3"):
        M.validate_formal_bank_execution_contract(
            policy, physics_step_dt_s=0.005, policy_step_dt_s=0.02,
            control_decimation=4, qdes_clamp=True,
        )


def test_formal_bank_rejects_nonzero_physx_friction_as_mujoco_frictionloss():
    limits = np.column_stack((np.full(31, -1.0), np.full(31, 1.0)))
    policy = SimpleNamespace(
        training_contract_exact="1",
        training_contract_schema_version="3",
        training_contract_sha256="b" * 64,
        source_checkpoint_sha256="a" * 64,
        qdes_joint_pos_limits=limits,
        physics_step_dt_s=0.005,
        policy_step_dt_s=0.02,
        control_decimation=4,
        qdes_clamp_meta=True,
        joint_actuator_types=("implicit",) * 31,
        joint_effort_limits=np.full(31, 24.0),
        joint_armature=np.full(31, 0.01),
        joint_friction_coefficients=np.full(31, 0.1),
        joint_velocity_limits=np.full(31, 12.0),
        joint_friction_backend="physx",
        joint_friction_semantics="load_dependent_spatial_force_coefficient",
        joint_friction_units="dimensionless",
    )
    with pytest.raises(SystemExit, match="cannot reproduce non-zero PhysX"):
        M.validate_formal_bank_execution_contract(
            policy, physics_step_dt_s=0.005, policy_step_dt_s=0.02,
            control_decimation=4, qdes_clamp=True,
        )


def _install_fake_mujoco(monkeypatch, *, armature=0.01, effort=24.0,
                         step_velocity=None):
    joint_names = [f"joint_{index}" for index in range(31)]
    body_names = list(dict.fromkeys([*M.TRACKED_BODIES, *M.FEET_BODIES]))
    joint_ids = {name: index + 1 for index, name in enumerate(joint_names)}
    # MuJoCo body 0 is world; all fake robot bodies live in the pelvis (id 1) subtree.
    body_ids = {name: index + 1 for index, name in enumerate(body_names)}
    actuator_ids = {name + "_motor": index for index, name in enumerate(joint_names)}
    qadr = np.arange(7, 38, dtype=int)
    vadr = np.arange(6, 37, dtype=int)
    model = SimpleNamespace(
        opt=SimpleNamespace(timestep=0.0, integrator=0),
        njnt=32,
        jnt_qposadr=np.concatenate(([0], qadr)),
        jnt_dofadr=np.concatenate(([0], vadr)),
        jnt_type=np.concatenate(([0], np.full(31, 3, dtype=int))),
        jnt_bodyid=np.concatenate((
            ([body_ids["pelvis_link"]], np.full(31, body_ids["pelvis_link"], dtype=int))
        )),
        actuator_ctrlrange=np.column_stack((
            np.full(31, -effort), np.full(31, effort),
        )),
        jnt_range=np.vstack((np.zeros((1, 2)),
                             np.column_stack((np.full(31, -2.0), np.full(31, 2.0))))),
        dof_armature=np.zeros(37),
        dof_damping=np.full(37, 0.5),
        dof_frictionloss=np.full(37, 0.2),
        nbody=len(body_names) + 1,
        body_parentid=np.asarray(
            [0, 0, *([body_ids["pelvis_link"]] * (len(body_names) - 1))], dtype=int
        ),
        ngeom=1,
        geom_bodyid=np.asarray([body_ids["pelvis_link"]], dtype=int),
        site_bodyid=np.asarray([body_ids["pelvis_link"]], dtype=int),
    )
    model.dof_armature[vadr] = armature

    class _Data:
        def __init__(self, _model):
            self.qpos = np.zeros(38)
            self.qvel = np.zeros(37)
            self.ctrl = np.zeros(31)
            self.ncon = 0

    obj = SimpleNamespace(
        mjOBJ_JOINT=1, mjOBJ_BODY=2, mjOBJ_ACTUATOR=3, mjOBJ_SITE=4,
    )

    def name2id(_model, kind, name):
        if kind == obj.mjOBJ_JOINT:
            return joint_ids.get(name, -1)
        if kind == obj.mjOBJ_BODY:
            return body_ids.get(name, -1)
        if kind == obj.mjOBJ_ACTUATOR:
            return actuator_ids.get(name, -1)
        if kind == obj.mjOBJ_SITE:
            return 0 if name == "right_racket" else -1
        return -1

    def step(_model, data):
        if step_velocity is not None:
            data.qvel[vadr[0]] = float(step_velocity)

    def object_velocity(_model, data, _kind, _object_id, out, _local):
        out[:] = 0.0
        # Deterministic site-velocity witness: the first articulated qvel is sampled before any
        # diagnostic post-step clip, so tests can prove the direction receipt did not read back
        # the safer continuation state.
        out[3] = data.qvel[vadr[0]]

    fake = SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=lambda _path: model),
        MjData=_Data,
        mjtObj=obj,
        mjtJoint=SimpleNamespace(mjJNT_FREE=0),
        mjtIntegrator=SimpleNamespace(mjINT_IMPLICITFAST=7),
        mj_name2id=name2id,
        mj_step=step,
        mj_forward=lambda *_args: None,
        mj_objectVelocity=object_velocity,
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    return joint_names, body_names, model


def _append_fake_freejoint(model, *, body_id):
    model.jnt_qposadr = np.append(model.jnt_qposadr, 38)
    model.jnt_dofadr = np.append(model.jnt_dofadr, 37)
    model.jnt_type = np.append(model.jnt_type, 0)
    model.jnt_bodyid = np.append(model.jnt_bodyid, int(body_id))
    model.jnt_range = np.vstack((model.jnt_range, np.zeros((1, 2))))
    model.njnt += 1


def test_mujoco_robot_rejects_nonzero_pelvis_freejoint_addresses(monkeypatch):
    joint_names, body_names, model = _install_fake_mujoco(monkeypatch)
    model.jnt_dofadr[0] = 1
    with pytest.raises(SystemExit, match="freejoint must start at qpos address 0"):
        M.MujocoRobot(
            "unused.xml", joint_names, body_names, 0.005,
            keep_native_damping=False, keep_frictionloss=False,
            pd_mode="explicit", actuator_types=("explicit",) * 31,
        )


def test_mujoco_robot_allows_other_free_bodies_but_requires_exactly_one_on_pelvis(monkeypatch):
    joint_names, body_names, model = _install_fake_mujoco(monkeypatch)
    _append_fake_freejoint(model, body_id=2)
    robot = M.MujocoRobot(
        "unused.xml", joint_names, body_names, 0.005,
        keep_native_damping=False, keep_frictionloss=False,
        pd_mode="explicit", actuator_types=("explicit",) * 31,
    )
    assert robot.root_free_jid == 0

    joint_names, body_names, model = _install_fake_mujoco(monkeypatch)
    _append_fake_freejoint(model, body_id=1)
    with pytest.raises(SystemExit, match="pelvis_link must own exactly one freejoint, found 2"):
        M.MujocoRobot(
            "unused.xml", joint_names, body_names, 0.005,
            keep_native_damping=False, keep_frictionloss=False,
            pd_mode="explicit", actuator_types=("explicit",) * 31,
        )

    joint_names, body_names, model = _install_fake_mujoco(monkeypatch)
    model.jnt_type[0] = 3
    with pytest.raises(SystemExit, match="pelvis_link must own exactly one freejoint, found 0"):
        M.MujocoRobot(
            "unused.xml", joint_names, body_names, 0.005,
            keep_native_damping=False, keep_frictionloss=False,
            pd_mode="explicit", actuator_types=("explicit",) * 31,
        )


def test_mujoco_robot_applies_bound_implicit_armature_effort_and_zero_friction(monkeypatch):
    joint_names, body_names, model = _install_fake_mujoco(monkeypatch)
    kd = np.linspace(1.0, 2.0, 31)
    robot = M.MujocoRobot(
        "unused.xml", joint_names, body_names, 0.005,
        keep_native_damping=False, keep_frictionloss=False,
        pd_mode="implicit", kd_for_implicit=kd,
        actuator_types=("implicit",) * 31,
        joint_armature=np.full(31, 0.01),
        joint_velocity_limits=np.full(31, 12.0),
        joint_effort_limits=np.full(31, 24.0),
        require_bound_plant_match=True,
        allow_velocity_limit_proxy=False,
    )
    assert np.array_equal(model.dof_armature[robot.vadr], np.full(31, 0.01))
    assert np.array_equal(model.dof_damping[robot.vadr], np.zeros(31))
    assert np.array_equal(model.dof_frictionloss[robot.vadr], np.zeros(31))
    assert np.array_equal(robot.ctrl_lo, np.full(31, -24.0))
    assert model.opt.integrator == 7
    assert robot.implicit_effort_execution_mode == "isaac_total_pd_clip_exact"


def test_mujoco_robot_accepts_float32_armature_roundtrip(monkeypatch):
    # Training metadata originates in Isaac float32 tensors, while the same decimal values
    # in MJCF are parsed as float64.  The exact gate must tolerate only that serialization
    # residue (A3 max observed 2.71e-9), not a physically meaningful plant change.
    source = np.full(31, 0.06646569891, dtype=np.float64)
    bound = source.astype(np.float32).astype(np.float64)
    joint_names, body_names, model = _install_fake_mujoco(
        monkeypatch, armature=source
    )
    robot = M.MujocoRobot(
        "unused.xml", joint_names, body_names, 0.005,
        keep_native_damping=False, keep_frictionloss=False,
        pd_mode="implicit", kd_for_implicit=np.ones(31),
        actuator_types=("implicit",) * 31,
        joint_armature=bound,
        joint_velocity_limits=np.full(31, 12.0),
        joint_effort_limits=np.full(31, 24.0),
        require_bound_plant_match=True,
        allow_velocity_limit_proxy=False,
    )
    assert np.array_equal(model.dof_armature[robot.vadr], bound)


def test_mujoco_robot_accepts_float32_effort_roundtrip(monkeypatch):
    # A3 ankle effort is 118.2 in MJCF and 118.199996948... after Isaac float32 storage.
    bound = np.full(31, np.float32(118.2), dtype=np.float32).astype(np.float64)
    joint_names, body_names, model = _install_fake_mujoco(
        monkeypatch, armature=0.01, effort=118.2
    )
    robot = M.MujocoRobot(
        "unused.xml", joint_names, body_names, 0.005,
        keep_native_damping=False, keep_frictionloss=False,
        pd_mode="implicit", kd_for_implicit=np.ones(31),
        actuator_types=("implicit",) * 31,
        joint_armature=np.full(31, 0.01),
        joint_velocity_limits=np.full(31, 12.0),
        joint_effort_limits=bound,
        require_bound_plant_match=True,
        allow_velocity_limit_proxy=False,
    )
    assert np.array_equal(model.actuator_ctrlrange[robot.act_id, 1], bound)


def test_float32_plant_match_requires_canonical_bound_and_rejects_next_grid_value():
    source = np.array([118.2, 0.06646569891], dtype=np.float64)
    canonical = source.astype(np.float32).astype(np.float64)
    assert M.matches_training_float32_plant(source, canonical)

    noncanonical = canonical.copy()
    noncanonical[0] += 1e-12
    assert not M.matches_training_float32_plant(source, noncanonical)

    assert M.matches_training_float32_plant(
        np.array([0.01], dtype=np.float64), np.array([0.01], dtype=np.float64)
    )

    next_grid = canonical.copy()
    next_grid[0] = np.nextafter(np.float32(canonical[0]), np.float32(np.inf)).item()
    assert not M.matches_training_float32_plant(source, next_grid)

    bound_one = np.array([1.0], dtype=np.float64)
    ulp = float(np.nextafter(np.float32(1.0), np.float32(np.inf)) - np.float32(1.0))
    assert M.matches_training_float32_plant(
        np.array([1.0 + 0.49 * ulp]), bound_one
    )
    assert not M.matches_training_float32_plant(
        np.array([1.0 + 0.51 * ulp]), bound_one
    )


def test_final_denominator_report_downgrades_bank_leg_for_inexact_artifact():
    sampler = SimpleNamespace(
        denominator_report=lambda: [
            "  evaluation_contract_exact=true",
            "  immutable_schedule: K=20",
        ]
    )
    assert M.final_denominator_report(
        sampler, evaluation_contract_exact=False
    ) == [
        "  evaluation_contract_exact=false",
        "  immutable_schedule: K=20",
    ]


@pytest.mark.parametrize(
    ("target_source", "allow_inexact_contract"),
    [("bank", False), ("boxes", True), ("venue-balls", True)],
)
def test_velocity_limit_proxy_requires_bank_and_inexact_consent(
    target_source, allow_inexact_contract
):
    with pytest.raises(SystemExit, match="requires both"):
        M.resolve_velocity_limit_proxy_contract(
            requested=True,
            target_source=target_source,
            allow_inexact_contract=allow_inexact_contract,
            formal_execution_contract_ok=True,
            evaluation_contract_exact=True,
        )


def test_velocity_limit_proxy_only_downgrades_authority_not_other_plant_guards():
    formal, exact, strict_other_plant = M.resolve_velocity_limit_proxy_contract(
        requested=True,
        target_source="bank",
        allow_inexact_contract=True,
        formal_execution_contract_ok=True,
        evaluation_contract_exact=True,
    )
    assert formal is False
    assert exact is False
    assert strict_other_plant is True


def test_mujoco_robot_fails_mismatched_bound_armature_and_active_velocity_limit(monkeypatch):
    joint_names, body_names, _ = _install_fake_mujoco(
        monkeypatch, armature=0.02, step_velocity=13.0
    )
    with pytest.raises(SystemExit, match="armature disagrees"):
        M.MujocoRobot(
            "unused.xml", joint_names, body_names, 0.005,
            keep_native_damping=False, keep_frictionloss=False,
            pd_mode="implicit", kd_for_implicit=np.ones(31),
            actuator_types=("implicit",) * 31,
            joint_armature=np.full(31, 0.01),
            joint_velocity_limits=np.full(31, 12.0),
            joint_effort_limits=np.full(31, 24.0),
            require_bound_plant_match=True,
            allow_velocity_limit_proxy=False,
        )

    # The diagnostic path may clamp a velocity-limit hit, but the exact path must stop because
    # PhysX's braking constraint is not reproduced by MuJoCo free integration.
    joint_names, body_names, _ = _install_fake_mujoco(
        monkeypatch, armature=0.01, step_velocity=13.0
    )
    robot = M.MujocoRobot(
        "unused.xml", joint_names, body_names, 0.005,
        keep_native_damping=False, keep_frictionloss=False,
        pd_mode="implicit", kd_for_implicit=np.ones(31),
        actuator_types=("implicit",) * 31,
        joint_armature=np.full(31, 0.01),
        joint_velocity_limits=np.full(31, 12.0),
        joint_effort_limits=np.full(31, 24.0),
        require_bound_plant_match=True,
        allow_velocity_limit_proxy=False,
    )
    with pytest.raises(SystemExit, match="reached bound PhysX joint-velocity limit"):
        robot.apply_pd_and_step(np.zeros(31), np.ones(31), np.ones(31), 1)
    assert robot.last_control_velocity_limit[
        "post_step_raw_racket_lin_vel_w"
    ] == pytest.approx([13.0, 0.0, 0.0])


def test_explicit_velocity_proxy_continues_and_keeps_other_plant_guards(monkeypatch):
    joint_names, body_names, _ = _install_fake_mujoco(
        monkeypatch, armature=0.01, step_velocity=13.0
    )
    robot = M.MujocoRobot(
        "unused.xml", joint_names, body_names, 0.005,
        keep_native_damping=False, keep_frictionloss=False,
        pd_mode="implicit", kd_for_implicit=np.ones(31),
        actuator_types=("implicit",) * 31,
        joint_armature=np.full(31, 0.01),
        joint_velocity_limits=np.full(31, 12.0),
        joint_effort_limits=np.full(31, 24.0),
        require_bound_plant_match=True,
        allow_velocity_limit_proxy=True,
        allow_effort_limit_proxy=False,
        fail_on_self_contact=True,
    )
    robot.apply_pd_and_step(np.zeros(31), np.ones(31), np.ones(31), 1)
    assert robot.velocity_limit_hit_count == 1
    assert robot.data.qvel[robot.vadr[0]] == pytest.approx(12.0)
    assert robot.last_control_velocity_limit[
        "post_step_raw_racket_lin_vel_w"
    ] == pytest.approx([13.0, 0.0, 0.0])
    assert robot.racket_lin_vel_w() == pytest.approx([12.0, 0.0, 0.0])
    assert robot.allow_effort_limit_proxy is False
    assert robot.fail_on_self_contact is True


def test_json_ready_replaces_nonfinite_and_numpy_scalars():
    out = M.json_ready({"x": np.array([1.0, np.nan]), "n": np.int64(3)})
    assert out == {"x": [1.0, None], "n": 3}


def _metadata(*, schema="2", baked=None, empirical=None, sidecar_sha=None,
              training_exact=False):
    joints = [f"joint_{i}" for i in range(31)]
    values = ["0"] * 31
    md = {
        "hope_metadata_schema_version": schema,
        "joint_names": ",".join(joints),
        "default_joint_pos": ",".join(values),
        "action_scale": ",".join(["0.25"] * 31),
        "joint_stiffness": ",".join(["10"] * 31),
        "joint_damping": ",".join(["1"] * 31),
        "body_names": ",".join(M.TRACKED_BODIES),
        "anchor_body_name": M.ANCHOR_BODY,
        "observation_names": "legacy_175",
        "clip_seg_lengths": "10,20",
        "actor_leg_ref_mask_provenance_epoch": "1",
        "training_contract_sha256": "b" * 64,
        "source_checkpoint_sha256": "a" * 64,
    }
    mask_payload = (
        "actor_leg_ref_mask_provenance_epoch=1\n"
        "actor_leg_ref_mask=0\n"
        f"training_contract_sha256={'b' * 64}\n"
        f"source_checkpoint_sha256={'a' * 64}\n"
    )
    md["actor_leg_ref_mask_provenance_sha256"] = hashlib.sha256(
        mask_payload.encode("ascii")
    ).hexdigest()
    if baked is not None:
        md["obs_norm_baked"] = str(int(baked))
    if empirical is not None:
        md["empirical_normalization"] = str(int(empirical))
        md["trained_with_obs_norm"] = str(int(empirical))
    if sidecar_sha is not None:
        md["obs_norm_sidecar_sha256"] = sidecar_sha
    if training_exact:
        md.update({
            "training_contract_exact": "1",
            "training_contract_schema_version": "2",
            "training_contract_sha256": "b" * 64,
            "source_checkpoint_sha256": "a" * 64,
            "motion_kinematics_exact": "1",
            "motion_body_lin_vel_points": "center_of_mass,center_of_mass",
            "actor_obs_contract": "deploy_parity",
            "actor_obs_mode": "deploy_parity",
            "actor_obs_total_dim": "175",
            "actor_obs_term_dims": ",".join(map(str, M.DEPLOY_PARITY_OBS_DIMS)),
            "observation_names": ",".join(M.DEPLOY_PARITY_OBS_NAMES),
        })
    return md


class _FakeSession:
    def __init__(self, md):
        self._md = md

    def get_inputs(self):
        return [SimpleNamespace(name="obs", shape=[1, 175], type="tensor(float)"),
                SimpleNamespace(name="time_step", shape=[1, 1], type="tensor(float)")]

    def get_outputs(self):
        shapes = {
            "actions": [1, 31],
            "joint_pos": [1, 31],
            "joint_vel": [1, 31],
            "body_pos_w": [1, 14, 3],
            "body_quat_w": [1, 14, 4],
            "body_lin_vel_w": [1, 14, 3],
            "body_ang_vel_w": [1, 14, 3],
        }
        return [
            SimpleNamespace(name=name, shape=shape, type="tensor(float)")
            for name, shape in shapes.items()
        ]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self._md)


def _policy(
    monkeypatch, tmp_path, md, obs_norm="auto", graph_baked=False,
    allow_inexact_contract=False,
):
    fake_ort = SimpleNamespace(InferenceSession=lambda *_args, **_kwargs: _FakeSession(md))
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        M,
        "inspect_onnx_obs_normalization",
        lambda _path: (graph_baked, "test graph inspection"),
    )
    return M.OnnxPolicy(
        str(tmp_path / "policy.onnx"),
        obs_norm=obs_norm,
        allow_inexact_contract=allow_inexact_contract,
    )


def test_actor_leg_ref_mask_epoch_is_required_for_command_models_and_mask_is_refused(
    monkeypatch, tmp_path,
):
    epoch_only_backfill = _metadata(schema="", baked=True, empirical=True)
    epoch_only_backfill.pop("actor_leg_ref_mask_provenance_sha256")
    with pytest.raises(SystemExit, match="not content-bound"):
        _policy(monkeypatch, tmp_path, epoch_only_backfill, graph_baked=True)

    missing_bound_hashes = _metadata(schema="", baked=True, empirical=True)
    missing_bound_hashes.pop("training_contract_sha256")
    missing_bound_hashes.pop("source_checkpoint_sha256")
    with pytest.raises(SystemExit, match="requires valid training-contract"):
        _policy(monkeypatch, tmp_path, missing_bound_hashes, graph_baked=True)

    md = _metadata(schema="", baked=True, empirical=True)
    md.pop("actor_leg_ref_mask_provenance_epoch")
    md.pop("actor_leg_ref_mask_provenance_sha256")
    with pytest.raises(SystemExit, match="masked and unmasked checkpoints are indistinguishable"):
        _policy(monkeypatch, tmp_path, md, graph_baked=True)

    diagnostic = _policy(
        monkeypatch,
        tmp_path,
        md,
        graph_baked=True,
        allow_inexact_contract=True,
    )
    assert diagnostic.actor_leg_ref_mask_provenance_exact is False
    assert diagnostic.evaluation_contract_exact is False

    md = _metadata(schema="", baked=True, empirical=True)
    md["actor_leg_ref_mask"] = "1"
    mask_payload = (
        "actor_leg_ref_mask_provenance_epoch=1\n"
        "actor_leg_ref_mask=1\n"
        f"training_contract_sha256={'b' * 64}\n"
        f"source_checkpoint_sha256={'a' * 64}\n"
    )
    md["actor_leg_ref_mask_provenance_sha256"] = hashlib.sha256(
        mask_payload.encode("ascii")
    ).hexdigest()
    with pytest.raises(SystemExit, match="no counterpart"):
        _policy(
            monkeypatch,
            tmp_path,
            md,
            graph_baked=True,
            allow_inexact_contract=True,
        )


def _write_sidecar(path, *, std=1.0, eps_value=1e-2):
    mean = np.zeros(175, np.float32)
    std_values = np.full(175, std, np.float32)
    eps = np.float32(eps_value)
    count = np.int64(100)
    np.savez(
        path,
        mean=mean,
        std=std_values,
        eps=eps,
        count=count,
        source_checkpoint_sha256=np.asarray("a" * 64),
        normalizer_state_sha256=np.asarray(
            M.normalizer_state_sha256(mean, std_values, eps, count)
        ),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_schema2_normalization_truth_is_fail_closed_but_unbound_sidecar_is_marked(
        monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="must explicitly declare"):
        _policy(monkeypatch, tmp_path, _metadata(baked=None, empirical=None))

    md = _metadata(baked=False, empirical=True)
    with pytest.raises(SystemExit, match="no sidecar exists"):
        _policy(monkeypatch, tmp_path, md)
    with pytest.raises(SystemExit, match="Formal scores fail closed"):
        _policy(monkeypatch, tmp_path, md, obs_norm="off")

    sidecar = tmp_path / "obs_norm.npz"
    digest = _write_sidecar(sidecar)
    policy = _policy(monkeypatch, tmp_path, md)
    assert policy.obs_norm_path == str(sidecar)
    assert policy.evaluation_contract_exact is False  # schema-2 legacy-unbound-sidecar

    bound = _policy(
        monkeypatch, tmp_path,
        _metadata(baked=False, empirical=True, sidecar_sha=digest),
    )
    assert bound.evaluation_contract_exact is False  # normalization bound, training contract absent

    fully_bound = _policy(
        monkeypatch, tmp_path,
        _metadata(
            baked=False, empirical=True, sidecar_sha=digest, training_exact=True
        ),
    )
    assert fully_bound.evaluation_contract_exact is True
    assert fully_bound.motion_body_lin_vel_points == (
        "center_of_mass", "center_of_mass",
    )

    legacy_exact_motion_md = _metadata(
        baked=False, empirical=True, sidecar_sha=digest, training_exact=True
    )
    legacy_exact_motion_md.pop("motion_body_lin_vel_points")
    legacy_exact_motion = _policy(monkeypatch, tmp_path, legacy_exact_motion_md)
    assert legacy_exact_motion.motion_body_lin_vel_points == (
        "center_of_mass", "center_of_mass",
    )
    assert legacy_exact_motion.evaluation_contract_exact is True

    mixed_exact_claim_md = _metadata(
        baked=False, empirical=True, sidecar_sha=digest, training_exact=True
    )
    mixed_exact_claim_md["motion_body_lin_vel_points"] = "center_of_mass,link_origin"
    mixed_exact_claim = _policy(monkeypatch, tmp_path, mixed_exact_claim_md)
    assert mixed_exact_claim.motion_body_lin_vel_points == (
        "center_of_mass", "link_origin",
    )
    assert mixed_exact_claim.evaluation_contract_exact is False

    wrong_point_count_md = _metadata(
        baked=False, empirical=True, sidecar_sha=digest, training_exact=True
    )
    wrong_point_count_md["motion_body_lin_vel_points"] = "center_of_mass"
    wrong_point_count = _policy(monkeypatch, tmp_path, wrong_point_count_md)
    assert wrong_point_count.evaluation_contract_exact is False

    ambiguous_motion = _policy(
        monkeypatch, tmp_path,
        _metadata(baked=False, empirical=True, sidecar_sha=digest),
    )
    assert ambiguous_motion.motion_body_lin_vel_points is None
    assert ambiguous_motion.evaluation_contract_exact is False

    with pytest.raises(SystemExit, match="contradicts graph dataflow"):
        _policy(
            monkeypatch, tmp_path,
            _metadata(baked=True, empirical=True),
            graph_baked=False,
        )


def test_schema3_requires_sidecar_hash_and_rejects_invalid_stats(monkeypatch, tmp_path):
    sidecar = tmp_path / "obs_norm.npz"
    _write_sidecar(sidecar)
    with pytest.raises(SystemExit, match=r"schema-v3\+ normalized raw ONNX requires"):
        _policy(monkeypatch, tmp_path, _metadata(schema="3", baked=False, empirical=True))

    _write_sidecar(sidecar, std=-1.0)
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="finite non-negative std"):
        _policy(
            monkeypatch, tmp_path,
            _metadata(schema="3", baked=False, empirical=True, sidecar_sha=digest),
        )


def test_obs_normalizer_accepts_epsilon_protected_zero_std_and_rejects_zero_sum(
        monkeypatch, tmp_path):
    sidecar = tmp_path / "obs_norm.npz"
    digest = _write_sidecar(sidecar, std=0.0, eps_value=1e-2)
    policy = _policy(
        monkeypatch, tmp_path,
        _metadata(baked=False, empirical=True, sidecar_sha=digest),
    )
    assert np.all(policy.obs_std == 0.0)
    assert policy.obs_eps == pytest.approx(1e-2)

    digest = _write_sidecar(sidecar, std=0.0, eps_value=0.0)
    with pytest.raises(SystemExit, match=r"std\+eps>0"):
        _policy(
            monkeypatch, tmp_path,
            _metadata(baked=False, empirical=True, sidecar_sha=digest),
        )


def test_schema3_execution_metadata_is_parsed_for_formal_bank(monkeypatch, tmp_path):
    md = _metadata(schema="3", baked=False, empirical=False, training_exact=True)
    md.update({
        "training_contract_schema_version": "3",
        "qdes_joint_pos_limits": ",".join(["-1", "1"] * 31),
        "physics_step_dt_s": "0.005",
        "policy_step_dt_s": "0.02",
        "control_decimation": "4",
        "qdes_clamp": "1",
        "joint_actuator_types": ",".join(["implicit"] * 31),
        "joint_effort_limits": ",".join(["24"] * 31),
        "joint_armature": ",".join(["0.01"] * 31),
        "joint_friction_coefficients": ",".join(["0"] * 31),
        "joint_velocity_limits": ",".join(["12"] * 31),
        "joint_friction_backend": "physx",
        "joint_friction_semantics": "load_dependent_spatial_force_coefficient",
        "joint_friction_units": "dimensionless",
    })
    policy = _policy(monkeypatch, tmp_path, md, graph_baked=False)
    assert policy.qdes_joint_pos_limits.shape == (31, 2)
    assert np.array_equal(policy.qdes_joint_pos_limits[0], [-1.0, 1.0])
    assert policy.physics_step_dt_s == pytest.approx(0.005)
    assert policy.policy_step_dt_s == pytest.approx(0.02)
    assert policy.control_decimation == 4
    assert policy.qdes_clamp_meta is True
    assert policy.joint_actuator_types == ("implicit",) * 31
    assert np.array_equal(policy.joint_effort_limits, np.full(31, 24.0))
    assert np.array_equal(policy.joint_friction_coefficients, np.zeros(31))
    assert policy.joint_friction_units == "dimensionless"


def test_exact_schema3_missing_actuator_plant_metadata_fails_closed(monkeypatch, tmp_path):
    md = _metadata(schema="3", baked=False, empirical=False, training_exact=True)
    md["training_contract_schema_version"] = "3"
    with pytest.raises(SystemExit, match="lacks actuator-plant metadata"):
        _policy(monkeypatch, tmp_path, md, graph_baked=False)

    md.update({
        "joint_actuator_types": ",".join(["implicit"] * 31),
        "joint_effort_limits": ",".join(["24"] * 31),
        "joint_armature": ",".join(["0.01"] * 31),
        "joint_friction_coefficients": ",".join(["0"] * 31),
        "joint_velocity_limits": ",".join(["12"] * 31),
        "joint_friction_backend": "physx",
        "joint_friction_semantics": "constant_coulomb_torque",
        "joint_friction_units": "N*m",
    })
    with pytest.raises(SystemExit, match="unsupported joint-friction semantics"):
        _policy(monkeypatch, tmp_path, md, graph_baked=False)


def test_legacy_model_keeps_explicit_nonexact_fallback(monkeypatch, tmp_path):
    policy = _policy(
        monkeypatch, tmp_path,
        _metadata(schema="", baked=None, empirical=None),
    )
    assert policy.obs_mean is None
    assert policy.evaluation_contract_exact is False


def test_cli_entrypoint_converts_unexpected_eval_exception_to_nonzero(monkeypatch, capsys):
    def explode():
        raise RuntimeError("rollout exploded")

    monkeypatch.setattr(M, "main", explode)
    assert M.cli_entrypoint() == 1
    assert "RuntimeError: rollout exploded" in capsys.readouterr().err
