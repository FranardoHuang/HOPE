"""Real-source and live-MjData contract tests for the native C211 producer."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch  # noqa: F401 - load Torch/OpenMP before NumPy on the macOS host
import numpy as np
import pytest

from hope_training.whole_body_tracking.mujoco_native import action_ball_211_abi as abi
from hope_training.whole_body_tracking.mujoco_native import action_ball_c211_env as c211
from hope_training.whole_body_tracking.mujoco_native import checkpoint
from hope_training.whole_body_tracking.mujoco_native import n1_reward_event_kernel
from hope_training.whole_body_tracking.mujoco_native import single_env
from hope_training.whole_body_tracking.mujoco_native import trainer


IMMUTABLE_TAPE = (
    "configs/action_ball_n1_measured_20260803/"
    "fresh_tape_seed0_20260803_take061_robust20n_r4_splitready/"
    "immutable_n1_tape.v1.22052606032f.json"
)
IMMUTABLE_TAPE_SHA256 = (
    "22052606032f74257ce98b5b6be8e8a4f8175848655ce604f50adf4751409e66"
)
MEASURED_MOTION = (
    "assets/motions/chingmu73_measured_v4_20260803/" "hope_Take_061_unit04_BH.npz"
)
MEASURED_MOTION_SHA256 = (
    "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
)
PLANT_CONTRACT = (
    "configs/a3_vendor_runtime_authority_20260802_r8/"
    "bh_loop_c.shared_ready.training_contract.json"
)
SPLIT_READY_SEED = (
    "configs/action_ball_n1_measured_20260803/"
    "evidence_holdpass_robust20n_20260803/"
    "take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json"
)
ROOT_MJCF = (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
LEG_INDICES = (0, 1, 3, 4, 6, 7, 9, 10, 14, 15, 19, 20)


@pytest.fixture(scope="module")
def authorities():
    task = c211.C211TaskAuthority.load(
        IMMUTABLE_TAPE,
        expected_file_sha256=IMMUTABLE_TAPE_SHA256,
    )
    mimic = c211.MeasuredC211MimicAuthority.load(
        MEASURED_MOTION,
        expected_file_sha256=MEASURED_MOTION_SHA256,
        task=task,
    )
    return task, mimic


def test_real_v4_task_and_measured_mimic_authorities_close(authorities):
    task, mimic = authorities
    assert task.target_recipe == "outcome_dense_only"
    assert task.motion_sha256 == MEASURED_MOTION_SHA256
    assert task.reference_t_hit_s == 0.96
    assert mimic.uid == "Take_061_unit04_BH"
    assert mimic.frame_count == 57
    assert mimic.reference_hit_frame == 48
    assert tuple(mimic.joint_pos.shape) == (57, 31)
    assert tuple(mimic.body_pos_w.shape[2:]) == (3,)
    assert tuple(mimic.body_lin_vel_w.shape) == (57, len(mimic.body_names), 3)
    assert tuple(mimic.body_ang_vel_w.shape) == (57, len(mimic.body_names), 3)
    assert tuple(mimic.measured_racket_long_axis_w.shape) == (57, 3)
    assert mimic.racket_long_axis_local.tolist() == pytest.approx(
        c211.C211_RACKET_LONG_AXIS_LOCAL.tolist(), abs=1.0e-15
    )
    assert mimic.receipt["body_velocity_channels_available"] is True
    assert mimic.receipt["measured_racket_long_axis_available"] is True
    assert mimic.receipt["diagnostic_unauthorized"] is True


class _FakeMujoco:
    __version__ = "fake-contract-test"

    class mjtObj:
        mjOBJ_BODY = 1
        mjOBJ_SITE = 2

    @staticmethod
    def mj_name2id(model, _object_type, name):
        return model.body_name_to_id.get(name, -1)

    @staticmethod
    def mj_jacSite(model, _data, jacp, jacr, _site_id):
        jacp[:] = 0.0
        jacr[:] = 0.0
        jacp[:, :3] = np.eye(3)

    @staticmethod
    def mj_jacBodyCom(model, _data, jacp, jacr, _body_id):
        jacp[:] = 0.0
        jacr[:] = 0.0
        jacp[:, :3] = np.eye(3)
        jacr[:, 3:6] = np.eye(3)

    @staticmethod
    def mj_objectVelocity(_model, data, object_type, _object_id, result, _local):
        if object_type == _FakeMujoco.mjtObj.mjOBJ_BODY:
            result[:] = np.asarray(data.ball_object_velocity, dtype=np.float64)
        elif object_type == _FakeMujoco.mjtObj.mjOBJ_SITE:
            result[:] = np.asarray(data.racket_object_velocity, dtype=np.float64)
        else:
            raise ValueError("unknown fake object type")


class _FakeDelay:
    delay_steps = 0

    @staticmethod
    def state():
        return np.zeros((1, 31), dtype=np.float64)


def _geometry_contract():
    payload = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.five_solid_robot_safety_geometry_v1",
        "primitive": "axis_aligned_box_full_extents_m",
        "obstacle_order": ["motion_table_top"],
        "obstacles": [
            {
                "role": "top",
                "name": "motion_table_top",
                "center_mjcf_world_m": [1.87, 0.0, 0.735],
                "full_extents_m": [2.74, 1.525, 0.05],
            }
        ],
    }
    return {"payload": payload, "sha256": c211._sha256_json(payload)}


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _install_exact_split_ready_candidate(tmp_path, binding, mimic):
    pelvis = mimic.body_names.index(c211.ROOT_BODY_NAME)
    physical_q = mimic.joint_pos[0].copy()
    physical_q[list(LEG_INDICES)] += 0.01
    root_pos = mimic.body_pos_w[0, pelvis].copy()
    root_pos[2] += 0.005
    hold_qdes = physical_q.copy()
    hold_action = (hold_qdes - binding.default_joint_pos) / binding.action_scale
    candidate = {
        "schema_version": single_env.ACTION_SPECIFIC_HOLD_SCHEMA_VERSION,
        "kind": single_env.ACTION_SPECIFIC_HOLD_KIND,
        "diagnostic_unauthorized": True,
        "authorization": {
            "training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
        "joint_names": list(binding.joint_names),
        "sources": {
            "training_contract": {
                "path": PLANT_CONTRACT,
                "sha256": _file_sha256(PLANT_CONTRACT),
            },
            "teacher_motion": {
                "path": MEASURED_MOTION,
                "sha256": mimic.file_sha256,
                "uid": mimic.uid,
                "frame": 0,
                "joint_order_contract_id": single_env.JOINT_ORDER_CONTRACT_ID,
                "joint_order_contract_sha256": mimic.joint_order_contract_sha256,
            },
            "shared_lower_root_seed": {
                "path": SPLIT_READY_SEED,
                "sha256": _file_sha256(SPLIT_READY_SEED),
            },
            "root_mjcf": {
                "path": ROOT_MJCF,
                "sha256": _file_sha256(ROOT_MJCF),
            },
        },
        "semantics": {
            "teacher_reference_unchanged": True,
            "physical_reset": "shared_grounded_lower_root_plus_teacher_nonleg",
            "controller_birth_target": "static_lp_hold_qdes",
            "history_fill": "same_static_lp_hold_action",
            "teacher_and_physical_reset_may_differ": True,
        },
        "physical_ready": {
            "joint_pos": physical_q.tolist(),
            "joint_vel": [0.0] * 31,
            "root_pos": root_pos.tolist(),
            "root_quat_wxyz": mimic.body_quat_wxyz[0, pelvis].tolist(),
            "root_lin_vel_w": [0.0, 0.0, 0.0],
            "root_ang_vel_w": [0.0, 0.0, 0.0],
            "root_lin_vel_point": "link_origin",
            "leg_joint_indices": list(LEG_INDICES),
            "leg_joint_names": [binding.joint_names[index] for index in LEG_INDICES],
            "nonleg_exact_teacher_q0": True,
        },
        "hold": {
            "joint_qdes": hold_qdes.tolist(),
            "normalized_action": hold_action.tolist(),
            "actuator_force_runtime_nm": [0.0] * 31,
            "maximum_abs_actuator_force_nm": 0.0,
            "maximum_abs_normalized_action": float(np.max(np.abs(hold_action))),
        },
        "static_evidence": {
            "gates": {
                "collision": "PASS",
                "double_support": "PASS",
                "exact_model_identity": "PASS",
                "foot_pose": "PASS",
                "joint_limits": "PASS",
                "leg_to_foot_jacobian": "PASS",
                "sole_floor": "PASS",
                "static_ground_dynamics": "PASS",
                "support_margin": "PASS",
            },
            "grounded_ready_receipt_sha256": "e" * 64,
        },
        "non_claims": ["test fixture is diagnostic only"],
    }
    unsigned = single_env._canonical_json_bytes(candidate)
    candidate["content_sha256"] = hashlib.sha256(unsigned).hexdigest()
    raw = single_env._canonical_json_bytes(candidate)
    path = tmp_path / "split_ready_hold_candidate.v2.json"
    path.write_bytes(raw)
    reset = SimpleNamespace(
        mode="action_specific_hold",
        hold_candidate_kind=single_env.ACTION_SPECIFIC_HOLD_KIND,
        hold_candidate_schema_version=single_env.ACTION_SPECIFIC_HOLD_SCHEMA_VERSION,
        hold_candidate_path=str(path.resolve()),
        hold_candidate_sha256=hashlib.sha256(raw).hexdigest(),
        hold_candidate_content_sha256=candidate["content_sha256"],
        source_motion_path=mimic.source_path,
        source_motion_sha256=mimic.file_sha256,
        source_motion_uid=mimic.uid,
        source_frame_index=0,
        source_joint_order_contract_id=single_env.JOINT_ORDER_CONTRACT_ID,
        source_joint_order_contract_sha256=mimic.joint_order_contract_sha256,
        joint_pos=physical_q,
        joint_vel=np.zeros(31, dtype=np.float64),
        root_pos=root_pos,
        root_quat_wxyz=mimic.body_quat_wxyz[0, pelvis].copy(),
        root_lin_vel_w=np.zeros(3, dtype=np.float64),
        root_ang_vel_w=np.zeros(3, dtype=np.float64),
        root_lin_vel_point="link_origin",
    )
    return reset, hold_action


def _fake_sources(tmp_path, task, mimic):
    names = tuple(f"joint_{index}" for index in range(31))
    contract = Path(PLANT_CONTRACT).resolve()
    body_name_to_id = {
        name: index for index, name in enumerate(c211.TRACKED_BODY_NAMES)
    }
    nbody = len(body_name_to_id)
    model = SimpleNamespace(
        nv=37,
        body_name_to_id=body_name_to_id,
        body_ipos=np.zeros((nbody, 3), dtype=np.float64),
        geom_size=np.asarray(((0.02, 0.0, 0.0),), dtype=np.float64),
    )
    root_id = body_name_to_id[c211.ROOT_BODY_NAME]
    model.body_ipos[root_id] = (0.1, 0.0, 0.0)
    qpos = np.zeros(7 + 31, dtype=np.float64)
    qvel = np.zeros(6 + 31, dtype=np.float64)
    default_q = np.linspace(-0.2, 0.2, 31, dtype=np.float64)
    qpos[7:] = default_q + 0.1
    qvel[:3] = (1.0, 2.0, 3.0)
    qvel[3:6] = (0.0, 0.0, 2.0)
    qvel[6:] = np.linspace(-0.3, 0.3, 31, dtype=np.float64)
    xpos = np.zeros((nbody, 3), dtype=np.float64)
    for row in range(nbody):
        xpos[row] = (0.1 * row, -0.02 * row, 1.0 + 0.01 * row)
    xpos[root_id] = (0.1, 0.0, 1.0)
    rotations = np.tile(np.eye(3).reshape(1, 9), (nbody, 1))
    data = SimpleNamespace(
        qpos=qpos,
        qvel=qvel,
        xpos=xpos,
        xmat=rotations,
        site_xpos=np.asarray(((0.5, -0.1, 1.2),), dtype=np.float64),
        site_xmat=np.eye(3, dtype=np.float64).reshape(1, 9),
        ctrl=np.zeros(31, dtype=np.float64),
        act=np.zeros(0, dtype=np.float64),
        qfrc_applied=np.zeros(37, dtype=np.float64),
        xfrc_applied=np.zeros((nbody, 6), dtype=np.float64),
        qacc_warmstart=np.zeros(37, dtype=np.float64),
        time=0.0,
        ball_object_velocity=np.zeros(6, dtype=np.float64),
        racket_object_velocity=np.zeros(6, dtype=np.float64),
    )
    binding = SimpleNamespace(
        default_joint_pos=default_q,
        action_scale=np.full(31, 0.1, dtype=np.float64),
        executed_qdes_limits=np.stack(
            (
                np.full(31, -2.0, dtype=np.float64),
                np.full(31, 2.0, dtype=np.float64),
            ),
            axis=1,
        ),
        joint_names=names,
        binding_sha256="1" * 64,
        source_path=str(contract.resolve()),
        source_sha256=c211._sha256_file(contract.resolve()),
        control_decimation=4,
    )
    def decode_action(action):
        raw = binding.default_joint_pos + binding.action_scale * np.asarray(
            action, dtype=np.float64
        )
        applied = np.clip(
            raw,
            binding.executed_qdes_limits[:, 0],
            binding.executed_qdes_limits[:, 1],
        )
        return raw, applied, int(np.count_nonzero(raw != applied))

    binding.decode_action = decode_action
    plant = SimpleNamespace(
        qpos_addr=np.arange(7, 38, dtype=np.int64),
        dof_addr=np.arange(6, 37, dtype=np.int64),
        geometry_contract=_geometry_contract(),
        delay=_FakeDelay(),
    )
    core = SimpleNamespace(
        mujoco=_FakeMujoco,
        model=model,
        data=data,
        plant=plant,
        binding=binding,
        scene=SimpleNamespace(
            canonical_xml_sha256="2" * 64,
            near_x=0.5,
            surface_z=0.76,
            ball_geom_id=0,
            ball_body_id=0,
            ball_dof_adr=0,
        ),
        scene_binding_sha256="3" * 64,
        observed_outcome_resolver_binding={"content_sha256": "a" * 64},
        observed_outcome_question_binding_sha256="b" * 64,
        selected_rubber_classifier_binding={"fake": True},
        _racket_site_id=0,
        _selected_rubber_action_lineage={"mount_normal_sign": 1},
        _observe_substep=lambda _model, _data, _substep: None,
        policy_tick=0,
    )
    reset, hold_action = _install_exact_split_ready_candidate(
        tmp_path, binding, mimic
    )
    tape = SimpleNamespace(
        reset_state=reset,
        history_fill_action=hold_action,
        joint_names=names,
        source_sha256="8" * 64,
    )
    question = SimpleNamespace(
        nominal_time_to_contact_s=task.time_to_contact_s + 0.02,
        landing_aim_xy_w_m=np.asarray(
            task.landing_aim_w_xy_m, dtype=np.float64
        ),
        source_sha256="c" * 64,
        authority={
            "immutable_n1_tape_bound": True,
            "immutable_tape_file_sha256": task.file_sha256,
            "immutable_tape_canonical_sha256": task.canonical_sha256,
            "base_question_sha256": task.question_sha256,
            "target_recipe": task.target_recipe,
            "target_producer_sha256": task.target_producer_sha256,
            "target_column_sha256": task.target_column_sha256,
            "motion_sha256": task.motion_sha256,
            "physics_sha256": task.physics_sha256,
            "profile_sha256": task.profile_sha256,
            "action_uid": task.action_uid,
        },
    )
    return core, tape, question


def _install_split_ready_frame0(tape, mimic):
    assert tape.reset_state.hold_candidate_kind == single_env.ACTION_SPECIFIC_HOLD_KIND
    return tape


def _patch_selected_rubber(monkeypatch):
    monkeypatch.setattr(
        c211.selected_rubber_classifier,
        "validate_classifier_binding",
        lambda _value: {"content_sha256": "4" * 64},
    )
    monkeypatch.setattr(
        c211.selected_rubber_classifier,
        "validate_action_lineage",
        lambda _value, *, classifier_binding: {
            "mount_normal_sign": 1,
            "content_sha256": "9" * 64,
            "classifier_binding_sha256": classifier_binding["content_sha256"],
        },
    )


def test_live_mjdata_c211_groups_wait_mask_and_active_teacher(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    _patch_selected_rubber(monkeypatch)
    producer = c211.C211ObservationProducer(
        cores=(core,),
        questions=(question,),
        robot_tape=tape,
        task=task,
        mimic=mimic,
        reset_wait_steps=1,
        policy_dt_s=0.02,
    )

    actor_wait, critic_wait = producer.tensors((False,))
    assert tuple(actor_wait.shape) == (1, 211)
    assert tuple(critic_wait.shape) == (1, 319)
    assert np.array_equal(
        actor_wait.numpy()[0, list(abi.C211_PROFILE.actor.task_mask_indices)],
        np.zeros(13, dtype=np.float32),
    )
    assert np.array_equal(
        critic_wait.numpy()[0, list(abi.C211_PROFILE.critic.task_mask_indices)],
        np.zeros(13, dtype=np.float32),
    )
    assert actor_wait[0, abi.C211_PROFILE.actor.task_valid_index].item() == 0.0
    command_span = abi.C211_PROFILE.critic.offsets["command"]
    assert np.allclose(
        critic_wait.numpy()[0, command_span][:31],
        core.data.qpos[core.plant.qpos_addr],
        rtol=0.0,
        atol=1.0e-7,
    )
    base_span = abi.C211_PROFILE.actor.offsets["actual_base_pose_lin_vel_world"]
    base_row = actor_wait.numpy()[0, base_span]
    assert np.allclose(base_row[:3], (-0.4, -0.7625, 0.24), atol=1.0e-7)
    assert np.allclose(base_row[9:12], (1.0, 2.2, 3.0), atol=1.0e-7)

    core.data.time = 0.02
    actor_active, critic_active = producer.tensors((True,))
    assert actor_active[0, abi.C211_PROFILE.actor.task_valid_index].item() == 1.0
    assert np.allclose(
        critic_active.numpy()[0, command_span][:31],
        mimic.joint_pos[0],
        rtol=0.0,
        atol=1.0e-7,
    )
    ttc_span = abi.C211_PROFILE.actor.offsets["time_to_contact"]
    assert actor_active.numpy()[0, ttc_span].item() == pytest.approx(
        task.time_to_contact_s
    )


def test_robot_tape_motion_mismatch_fails_before_core_binding(tmp_path, authorities):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    tape.reset_state.source_motion_sha256 = "f" * 64
    with pytest.raises(c211.C211EnvError, match="lineage differ"):
        c211.C211ObservationProducer(
            cores=(core,),
            questions=(question,),
            robot_tape=tape,
            task=task,
            mimic=mimic,
            reset_wait_steps=1,
            policy_dt_s=0.02,
        )


def test_split_ready_physical_birth_is_separate_and_task_hidden(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    _install_split_ready_frame0(tape, mimic)
    _patch_selected_rubber(monkeypatch)
    producer = c211.C211ObservationProducer(
        cores=(core,),
        questions=(question,),
        robot_tape=tape,
        task=task,
        mimic=mimic,
        reset_wait_steps=1,
        policy_dt_s=0.02,
    )
    assert producer.reset_birth_semantics == (
        "split_ready_physical_safe_birth_separate_from_measured_"
        "teacher_frame0_stationary_hidden_wait"
    )
    actor, critic = producer.tensors((False,))
    assert np.array_equal(
        actor.numpy()[0, list(abi.C211_PROFILE.actor.task_mask_indices)],
        np.zeros(13, dtype=np.float32),
    )
    assert np.array_equal(
        critic.numpy()[0, list(abi.C211_PROFILE.critic.task_mask_indices)],
        np.zeros(13, dtype=np.float32),
    )
    command_span = abi.C211_PROFILE.critic.offsets["command"]
    assert np.allclose(
        critic.numpy()[0, command_span][:31],
        core.data.qpos[core.plant.qpos_addr],
        rtol=0.0,
        atol=1.0e-7,
    )
    core.data.time = 0.02
    _actor_active, critic_active = producer.tensors((True,))
    assert np.allclose(
        critic_active.numpy()[0, command_span][:31],
        mimic.joint_pos[0],
        rtol=0.0,
        atol=1.0e-7,
    )


def test_split_ready_nonstationary_birth_fails_closed(tmp_path, authorities):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    _install_split_ready_frame0(tape, mimic)
    tape.reset_state.joint_vel[0] = 1.0e-6
    with pytest.raises(c211.C211EnvError, match="cannot be reopened"):
        c211.C211ObservationProducer(
            cores=(core,),
            questions=(question,),
            robot_tape=tape,
            task=task,
            mimic=mimic,
            reset_wait_steps=1,
            policy_dt_s=0.02,
        )


def test_unknown_action_specific_hold_fails_before_core_binding(tmp_path, authorities):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    tape.reset_state.hold_candidate_kind = "unknown_hold"
    tape.reset_state.hold_candidate_schema_version = 99
    with pytest.raises(c211.C211EnvError, match="only the split-ready"):
        c211.C211ObservationProducer(
            cores=(core,),
            questions=(question,),
            robot_tape=tape,
            task=task,
            mimic=mimic,
            reset_wait_steps=1,
            policy_dt_s=0.02,
        )


def test_teacher_frame_birth_is_not_active_c211_split_ready(tmp_path, authorities):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    tape.reset_state.mode = "teacher_frame"
    with pytest.raises(c211.C211EnvError, match="requires the split-ready"):
        c211.C211ObservationProducer(
            cores=(core,),
            questions=(question,),
            robot_tape=tape,
            task=task,
            mimic=mimic,
            reset_wait_steps=1,
            policy_dt_s=0.02,
        )


def test_legacy_exact_frame0_hold_is_not_active_c211_split_ready(
    tmp_path, authorities
):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    tape.reset_state.hold_candidate_kind = (
        single_env.EXACT_FRAME0_ACTION_SPECIFIC_HOLD_KIND
    )
    tape.reset_state.hold_candidate_schema_version = (
        single_env.EXACT_FRAME0_ACTION_SPECIFIC_HOLD_SCHEMA_VERSION
    )
    with pytest.raises(c211.C211EnvError, match="legacy exact-frame0"):
        c211.C211ObservationProducer(
            cores=(core,),
            questions=(question,),
            robot_tape=tape,
            task=task,
            mimic=mimic,
            reset_wait_steps=1,
            policy_dt_s=0.02,
        )


def test_reopened_candidate_receipt_cannot_be_relabelled_split_ready(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    original = single_env._revalidate_action_specific_reset_state

    def relabelled_candidate(binding, reset, history_fill_action):
        candidate = copy.deepcopy(original(binding, reset, history_fill_action))
        candidate["kind"] = single_env.EXACT_FRAME0_ACTION_SPECIFIC_HOLD_KIND
        candidate["schema_version"] = (
            single_env.EXACT_FRAME0_ACTION_SPECIFIC_HOLD_SCHEMA_VERSION
        )
        return candidate

    monkeypatch.setattr(
        single_env,
        "_revalidate_action_specific_reset_state",
        relabelled_candidate,
    )
    with pytest.raises(c211.C211EnvError, match="cannot be relabeled"):
        c211.C211ObservationProducer(
            cores=(core,),
            questions=(question,),
            robot_tape=tape,
            task=task,
            mimic=mimic,
            reset_wait_steps=1,
            policy_dt_s=0.02,
        )


@pytest.mark.parametrize(
    ("ball_velocity_y", "expected_eligible", "expected_reason"),
    (
        (-2.0, True, "eligible_exact_selected_positive_closing_contact"),
        (2.0, False, "nonpositive_selected_face_closing_speed"),
        (float("nan"), False, "nonfinite_contact_kinematics"),
    ),
)
def test_exact_contact_eligibility_uses_preimpact_positive_closing_speed(
    tmp_path,
    monkeypatch,
    authorities,
    ball_velocity_y,
    expected_eligible,
    expected_reason,
):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    _patch_selected_rubber(monkeypatch)
    producer = c211.C211ObservationProducer(
        cores=(core,),
        questions=(question,),
        robot_tape=tape,
        task=task,
        mimic=mimic,
        reset_wait_steps=1,
        policy_dt_s=0.02,
    )
    core.data.xpos[core.scene.ball_body_id] = np.asarray(
        (0.5, 0.0, 1.0), dtype=np.float64
    )
    core.data.site_xpos[core._racket_site_id] = np.asarray(
        (0.5, 0.0, 1.0), dtype=np.float64
    )
    core.data.ball_object_velocity = np.asarray(
        (0.0, 0.0, 0.0, 0.0, ball_velocity_y, 0.0), dtype=np.float64
    )
    core.data.racket_object_velocity = np.zeros(6, dtype=np.float64)
    core.policy_tick = 92
    core._first_racket_contact_stamp = {
        "policy_tick": 92,
        "physics_substep": 1,
    }
    producer._capture_first_contact_kinematics(0, 1)
    contact = n1_reward_event_kernel.ContactEvidence(
        True, n1_reward_event_kernel.EventStamp(92, 1), True
    )
    eligibility = producer.contact_eligibility(
        0, contact, nominal_strike_tick_1based=93
    )
    assert eligibility["eligible"] is expected_eligible
    assert eligibility["reason"] == expected_reason
    assert eligibility["contact_transition_tick_1based"] == 93


def test_contact_one_transition_early_is_not_nominal(tmp_path, monkeypatch, authorities):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    _patch_selected_rubber(monkeypatch)
    producer = c211.C211ObservationProducer(
        cores=(core,),
        questions=(question,),
        robot_tape=tape,
        task=task,
        mimic=mimic,
        reset_wait_steps=1,
        policy_dt_s=0.02,
    )
    binding = producer._bindings[0]
    producer._first_contact_kinematics[0] = c211._FirstContactKinematics(
        policy_tick_zero_based=91,
        physics_substep=2,
        selected_face_normal_w=(0.0, 1.0, 0.0),
        ball_contact_point_velocity_w_mps=(0.0, -2.0, 0.0),
        racket_site_velocity_w_mps=(0.0, 0.0, 0.0),
        selected_face_closing_speed_mps=2.0,
        finite=True,
        classifier_binding_sha256=binding.selected_rubber_classifier_sha256,
        selected_rubber_lineage_sha256=binding.selected_rubber_lineage_sha256,
    )
    result = producer.contact_eligibility(
        0,
        n1_reward_event_kernel.ContactEvidence(
            True, n1_reward_event_kernel.EventStamp(91, 2), True
        ),
        nominal_strike_tick_1based=93,
    )
    assert result["eligible"] is False
    assert result["reason"] == "contact_not_on_nominal_transition"


def test_contact_closing_uses_racket_angular_velocity_at_contact_point(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    _patch_selected_rubber(monkeypatch)
    producer = c211.C211ObservationProducer(
        cores=(core,),
        questions=(question,),
        robot_tape=tape,
        task=task,
        mimic=mimic,
        reset_wait_steps=1,
        policy_dt_s=0.02,
    )
    core.data.xpos[core.scene.ball_body_id] = (0.6, 0.0, 1.0)
    core.data.site_xpos[core._racket_site_id] = (0.5, 0.0, 1.0)
    core.data.ball_object_velocity = np.zeros(6, dtype=np.float64)
    core.data.racket_object_velocity = np.asarray(
        (0.0, 0.0, 10.0, 0.0, 0.0, 0.0), dtype=np.float64
    )
    core.policy_tick = 92
    core._first_racket_contact_stamp = {
        "policy_tick": 92,
        "physics_substep": 1,
    }
    producer._capture_first_contact_kinematics(0, 1)
    result = producer.contact_eligibility(
        0,
        n1_reward_event_kernel.ContactEvidence(
            True, n1_reward_event_kernel.EventStamp(92, 1), True
        ),
        nominal_strike_tick_1based=93,
    )
    assert result["eligible"] is True
    assert result["selected_face_closing_speed_mps"] == pytest.approx(1.0)


def test_plant_contract_source_sha_mismatch_fails_closed(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    core.binding.source_sha256 = "f" * 64
    _patch_selected_rubber(monkeypatch)
    with pytest.raises(c211.C211EnvError, match="cannot be reopened"):
        c211.C211ObservationProducer(
            cores=(core,),
            questions=(question,),
            robot_tape=tape,
            task=task,
            mimic=mimic,
            reset_wait_steps=1,
            policy_dt_s=0.02,
        )


class _FakeFixedCenterBase:
    """Minimal deterministic fixed-centre protocol around one live core double."""

    def __init__(self, core, tape, question):
        self.allow_action_ball_legacy_fixed_wait_test_double = True
        event_source = n1_reward_event_kernel.SourceBinding(
            source_id="fake_c211_native_event_source",
            source_sha256="d" * 64,
            event_contract_sha256=(
                n1_reward_event_kernel.native_physical_event_facts_contract()[
                    "content_sha256"
                ]
            ),
        )
        self.base_env = SimpleNamespace(
            cores=(core,),
            robot_tape=tape,
            questions=(question,),
            step_dt=0.02,
            max_episode_length=500,
            native_physical_event_runtime_available=True,
            native_physical_event_source_bindings=(event_source,),
        )
        self.base_env._single_stroke_timeout_authority_sha256 = None
        self.base_env._single_stroke_timeout_steps = [None]

        def install_single_stroke(
            *, env_ids, timeout_steps, authority_sha256
        ):
            if self.base_env._single_stroke_timeout_authority_sha256 not in (
                None,
                authority_sha256,
            ):
                raise RuntimeError("test timeout authority changed")
            self.base_env._single_stroke_timeout_authority_sha256 = (
                authority_sha256
            )
            for index, step in zip(env_ids, timeout_steps):
                self.base_env._single_stroke_timeout_steps[index] = step

        self.base_env.install_diagnostic_single_stroke_timeout_steps = (
            install_single_stroke
        )
        self.spec = SimpleNamespace(reset_wait_steps=1)
        self.num_envs = 1
        self.num_actions = 31
        self.num_observations = 76
        self.device = "cpu"
        self.cfg = {"kind": "fake_fixed_center_contract_test"}
        self._task_valid = (False,)
        self._boundary = True
        self._step = 0
        self.episode_length = 2
        self.forced_policy_tick = None
        self.official_racket_site_w_m = np.asarray(
            (0.5, -0.1, 1.2), dtype=np.float64
        )
        self._initial = {
            name: np.asarray(getattr(core.data, name)).copy()
            for name in (
                "qpos",
                "qvel",
                "ctrl",
                "act",
                "qfrc_applied",
                "xfrc_applied",
                "qacc_warmstart",
            )
        }
        self._identity = {
            "contract_sha256": "5" * 64,
            "observation_contract_sha256": "6" * 64,
            "action_contract_sha256": "7" * 64,
            "reward_contract_sha256": "8" * 64,
        }

    @property
    def task_valid(self):
        return self._task_valid

    def _reset_core(self):
        for name, value in self._initial.items():
            np.copyto(getattr(self.base_env.cores[0].data, name), value)
        self.base_env.cores[0].data.time = 0.0
        self.base_env.cores[0].policy_tick = 0

    def reset(self, *, seed=None):
        assert seed is None or type(seed) is int
        self._reset_core()
        self._task_valid = (False,)
        self._boundary = True
        self._step = 0
        self.forced_policy_tick = None
        return torch.zeros((1, 76)), {"observations": {"critic": torch.zeros((1, 76))}}

    def step(self, actions):
        assert tuple(actions.shape) == (1, 31)
        self._step += 1
        policy_tick = (
            self._step
            if self.forced_policy_tick is None
            else int(self.forced_policy_tick)
        )
        self.forced_policy_tick = None
        sample_time_s = policy_tick * 0.02
        core = self.base_env.cores[0]
        core.data.time = sample_time_s
        core.policy_tick = policy_tick
        core._observe_substep(
            core.model,
            core.data,
            core.binding.control_decimation - 1,
        )
        transition_valid = bool(self._task_valid[0])
        strike_now = bool(transition_valid and policy_tick == 93)
        native_facts = {"policy_tick": policy_tick}
        physical_sample = {
            "official_racket_site_w_m": tuple(
                float(value) for value in self.official_racket_site_w_m
            ),
            "selected_rubber_center_w_m": (0.5, -0.1, 1.2),
            "ball_center_w_m": (0.5, -0.1, 1.0),
            "sample_time_s": sample_time_s,
            "classifier_binding_sha256": "4" * 64,
            "selected_rubber_lineage_sha256": "9" * 64,
        }
        reward_row = {
            "task_valid": transition_valid,
            "sample_policy_tick_1based": policy_tick,
            "nominal_strike_tick": 93,
            "nominal_strike_sampled_now": strike_now,
        }
        done = self._step == self.episode_length
        if done:
            self._reset_core()
            self._task_valid = (False,)
            self._boundary = True
            self._step = 0
        else:
            self.base_env.cores[0].data.time = sample_time_s
            self.base_env.cores[0].policy_tick = policy_tick
            self._task_valid = (True,)
            self._boundary = False
        return (
            torch.zeros((1, 76)),
            torch.as_tensor((0.125,), dtype=torch.float32),
            torch.as_tensor((done,), dtype=torch.bool),
            {
                "observations": {"critic": torch.zeros((1, 76))},
                "time_outs": torch.as_tensor((done,), dtype=torch.bool),
                "episode_done_reasons": [
                    "time_out" if done else None
                ],
                "diagnostic_native_physical_event_facts": (native_facts,),
                "diagnostic_c_lite_physical_samples": (physical_sample,),
                "diagnostic_event_ledgers": (
                    {
                        "policy_ticks": policy_tick,
                        "termination": {
                            "exact_time_out_latched": done,
                            "exact_hard_terminated": False,
                            "exact_hard_reason": None,
                        },
                        "first_exact_hard_termination": None,
                        "joint_actual_forbidden_observed_ticks": 0,
                        "promotion_blocking_evidence": {
                            "promotion_blocked": False,
                            "reasons": [],
                        },
                    },
                ),
                "diagnostic_exact_hard_terminations": torch.as_tensor(
                    (False,), dtype=torch.bool
                ),
                "diagnostic_exact_hard_termination_reasons": [None],
                "task_valid_transition": [transition_valid],
                "task_valid_next": list(self._task_valid),
                "wait_assignment_transition": [
                    {"env_id": 0, "reset_generation": 1, "wait_ticks": 1}
                ],
                "reward_terms": [reward_row],
            },
        )

    def is_reset_boundary(self):
        return self._boundary

    def diagnostic_training_identity(self):
        return dict(self._identity)

    def diagnostic_training_receipt(self):
        return {
            "kind": trainer.DIAGNOSTIC_TRAINER_RECEIPT_KIND,
            "ppo_ready": True,
            "reward_available": True,
            "normal_step_available": True,
            "reset_boundary_checkpoint_available": True,
            "blockers": [],
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
            "mid_episode_resume": False,
            "terminal_row_telemetry_available": True,
            "terminal_row_telemetry_contract": (
                trainer.terminal_row_telemetry_contract()
            ),
            **self._identity,
        }


def _wrapped_fake_env(tmp_path, task, mimic, monkeypatch, *, split_ready=False):
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    if split_ready:
        _install_split_ready_frame0(tape, mimic)
    _patch_selected_rubber(monkeypatch)
    env = c211.MujocoC211DiagnosticVecEnv(
        base_env=_FakeFixedCenterBase(core, tape, question),
        task_authority=task,
        mimic_authority=mimic,
    )
    monkeypatch.setattr(
        env,
        "_event_evidence",
        lambda _index, _facts: (
            n1_reward_event_kernel.ContactEvidence(False, None, False),
            n1_reward_event_kernel.OutgoingFlightEvidence(
                False, None, None, None, None
            ),
        ),
    )
    def contact_eligibility(_index, contact, *, nominal_strike_tick_1based):
        if not contact.occurred or contact.stamp is None:
            return {
                "eligible": False,
                "reason": "no_actual_contact",
                "exact_nominal_contact_transition": False,
                "selected_rubber": False,
                "finite_contact_kinematics": False,
                "positive_closing_speed": False,
                "selected_face_closing_speed_mps": None,
            }
        exact = contact.stamp.policy_tick + 1 == nominal_strike_tick_1based
        eligible = bool(contact.selected_rubber and exact)
        return {
            "eligible": eligible,
            "reason": (
                "eligible_exact_selected_positive_closing_contact"
                if eligible
                else (
                    "first_contact_not_selected_rubber"
                    if not contact.selected_rubber
                    else "contact_not_on_nominal_transition"
                )
            ),
            "exact_nominal_contact_transition": exact,
            "selected_rubber": contact.selected_rubber,
            "finite_contact_kinematics": True,
            "positive_closing_speed": True,
            "selected_face_closing_speed_mps": 1.0,
        }
    monkeypatch.setattr(env.producer, "contact_eligibility", contact_eligibility)
    return env


def test_c211_rejects_missing_authoritative_continuous_wait_schedule(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    _patch_selected_rubber(monkeypatch)
    base = _FakeFixedCenterBase(core, tape, question)
    base.allow_action_ball_legacy_fixed_wait_test_double = False
    with pytest.raises(c211.C211EnvError, match="authoritative continuous WAIT"):
        c211.MujocoC211DiagnosticVecEnv(
            base_env=base,
            task_authority=task,
            mimic_authority=mimic,
        )


def test_c211_installs_one_stroke_close_tick_and_exposes_timeout_contract(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    env = _wrapped_fake_env(tmp_path, task, mimic, monkeypatch)
    wait_steps = env.producer._wait_steps_by_env[0]
    expected = int(
        math.ceil(
            (
                wait_steps * c211.C211_POLICY_DT_S
                + task.pre_swing_wait_s
                + task.scaled_t_cycle_s
            )
            / c211.C211_POLICY_DT_S
            - 1.0e-12
        )
    ) + 1
    assert env._native._single_stroke_timeout_steps == [expected]
    receipt = env.diagnostic_training_receipt()
    authority = receipt["single_stroke_timeout_authority"]
    assert authority["termination_reason"] == "action_ball_single_stroke_complete"
    assert authority["time_out"] is True
    assert authority["bootstrap_rule"] == trainer.TIMEOUT_BOOTSTRAP_RULE
    assert receipt["time_to_contact_observation_semantics"] == (
        "signed_unclamped_deadline_matching_Isaac"
    )


def test_c211_time_to_contact_stays_signed_after_nominal_strike(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    core, tape, question = _fake_sources(tmp_path, task, mimic)
    _patch_selected_rubber(monkeypatch)
    producer = c211.C211ObservationProducer(
        cores=(core,),
        questions=(question,),
        robot_tape=tape,
        task=task,
        mimic=mimic,
        reset_wait_steps=1,
        policy_dt_s=0.02,
    )
    core.data.time = question.nominal_time_to_contact_s + 0.04
    actor, critic = producer.tensors((True,))
    actor_span = abi.C211_PROFILE.actor.offsets["time_to_contact"]
    critic_span = abi.C211_PROFILE.critic.offsets["time_to_contact"]
    assert actor.numpy()[0, actor_span].item() == pytest.approx(-0.04)
    assert critic.numpy()[0, critic_span].item() == pytest.approx(-0.04)


def _wrapped_trainer(env):
    bootstrap = env.fresh_actor_bootstrap_contract()
    config = trainer.DiagnosticPPOConfig(
        action_dim=31,
        rollout_steps=2,
        hidden_dims=(8,),
        seed=67,
        learning_rate=1.0e-3,
        initial_action_std=0.02,
        fresh_actor_output_bias=tuple(
            float(value) for value in env.producer.robot_tape.history_fill_action
        ),
        fresh_actor_bootstrap_authority_sha256=bootstrap["content_sha256"],
        **abi.C211_PROFILE.trainer_config_kwargs(),
    )
    return trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=trainer.TrainerIdentity(**env.diagnostic_training_identity()),
        config=config,
    )


def test_isaac_synonymous_prior_raw_terms_weights_and_nonwrist_body_mask():
    body_count = len(c211.TRACKED_BODY_NAMES)
    rotations = np.tile(np.eye(3), (body_count, 1, 1))
    teacher_body_pos = np.zeros((body_count, 3), dtype=np.float64)
    live_body_pos = teacher_body_pos.copy()
    teacher_body_lin = np.zeros((body_count, 3), dtype=np.float64)
    live_body_lin = teacher_body_lin.copy()
    teacher_body_ang = np.zeros((body_count, 3), dtype=np.float64)
    live_body_ang = teacher_body_ang.copy()
    wrist = c211.TRACKED_BODY_NAMES.index(c211.RIGHT_WRIST_BODY_NAME)
    live_body_pos[wrist] = (100.0, -100.0, 50.0)
    live_body_lin[wrist] = (100.0, 100.0, 100.0)
    live_body_ang[wrist] = (-100.0, 100.0, -100.0)
    long_axis = c211.C211_RACKET_LONG_AXIS_LOCAL.copy()
    live = {
        "root_rotation": np.eye(3),
        "root_ang_vel_w": np.asarray((3.0, 4.0, 5.0)),
        "root_lin_vel_w": np.asarray((0.0, 0.0, 2.0)),
        "qd": np.ones(31),
        "body_pos": live_body_pos,
        "body_rotation": rotations.copy(),
        "body_lin_vel_w": live_body_lin,
        "body_ang_vel_w": live_body_ang,
        "anchor_rotation": np.eye(3),
        "racket_pos": np.zeros(3),
        "racket_velocity": np.zeros(3),
        "racket_normal": np.asarray((0.0, 1.0, 0.0)),
        "racket_long_axis": long_axis,
    }
    teacher = {
        "body_pos": teacher_body_pos,
        "body_rotation": rotations.copy(),
        "body_lin_vel_w": teacher_body_lin,
        "body_ang_vel_w": teacher_body_ang,
        "anchor_rotation": np.eye(3),
        "global_anchor_rotation": np.eye(3),
        "racket_pos": np.zeros(3),
        "racket_velocity": np.zeros(3),
        "racket_normal": np.asarray((0.0, 1.0, 0.0)),
        "racket_long_axis": long_axis,
    }
    row = c211._c211_isaac_synonymous_prior_terms(
        live=live,
        teacher=teacher,
        current_action=np.ones(31),
        previous_action=np.zeros(31),
    )
    terms = row["terms"]
    assert terms["upright_exp"]["raw_reward"] == pytest.approx(1.0)
    assert terms["base_ang_vel_xy"]["raw_reward"] == pytest.approx(25.0)
    assert terms["base_lin_vel_z"]["raw_reward"] == pytest.approx(4.0)
    assert terms["joint_vel"]["raw_reward"] == pytest.approx(31.0)
    # 2026-08-08:一阶平滑照开源改形状 —— 上游 isaaclab action_rate_l2 无封顶。
    # 31 维、每维差 1.0 ⇒ raw = 31.0(旧封顶版会把它削成 9.0,该断言当时必红)。
    assert terms["action_rate_l2"]["raw_reward"] == pytest.approx(31.0)
    assert terms["action_rate_l2"]["value_clamp"] is None
    assert terms["action_rate_l2"]["manager_weight"] == pytest.approx(-0.1)
    assert terms["action_rate_l2"]["post_policy_dt_reward"] == pytest.approx(
        -0.1 * 31.0 * 0.02
    )
    for name in (
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_racket_position",
        "motion_racket_velocity",
        "motion_racket_normal",
        "motion_racket_long_axis",
    ):
        assert terms[name]["raw_reward"] == pytest.approx(1.0)
    assert c211.RIGHT_WRIST_BODY_NAME not in terms["motion_body_pos"]["body_names"]
    expected = (
        1.0 * 1.0
        + 25.0 * -0.05
        + 4.0 * -0.5
        + 31.0 * -1.0e-4
        + 31.0 * -0.1          # 2026-08-08:无封顶 action_rate_l2(旧封顶版是 9.0 * -0.2)
        + 1.0 * 0.075
        + 4.0 * 0.15
        + 3.0 * 0.20
        + 1.0 * 0.10
    ) * c211.C211_POLICY_DT_S
    assert row["total_post_policy_dt_reward"] == pytest.approx(expected)


# --------------------------------------------------------------------------------------------- #
# 一阶平滑罚:C 族镜像的变异测试 + A/C 同形对拍(2026-08-08 Franco 裁定二第二条)
#
# 现役形状 = 上游 isaaclab ``action_rate_l2`` = ``sum((a_t − a_{t−1})²)``,无封顶。
# 这里测的是 C 族真在跑的那份实现(MuJoCo lane),并且逐位对拍上游公式 —— 两族"除了
# obs 和 reward 之外都一样"这条纪律,在 reward 这一项上要求的正是**同形同价**。
# --------------------------------------------------------------------------------------------- #
def _c211_action_rate(current, previous):
    """只取 action_rate_l2 那一项的 raw,其余输入固定成不影响该项的哑元。"""

    n_bodies = len(c211.TRACKED_BODY_NAMES)
    rotations = np.repeat(np.eye(3)[None, :, :], n_bodies, axis=0)
    zeros3 = np.zeros((n_bodies, 3))
    long_axis = c211.C211_RACKET_LONG_AXIS_LOCAL.copy()
    live = {
        "root_rotation": np.eye(3),
        "root_ang_vel_w": np.zeros(3),
        "root_lin_vel_w": np.zeros(3),
        "qd": np.zeros(31),
        "body_pos": zeros3.copy(),
        "body_rotation": rotations.copy(),
        "body_lin_vel_w": zeros3.copy(),
        "body_ang_vel_w": zeros3.copy(),
        "anchor_rotation": np.eye(3),
        "racket_pos": np.zeros(3),
        "racket_velocity": np.zeros(3),
        "racket_normal": np.asarray((0.0, 1.0, 0.0)),
        "racket_long_axis": long_axis,
    }
    teacher = {
        "body_pos": zeros3.copy(),
        "body_rotation": rotations.copy(),
        "body_lin_vel_w": zeros3.copy(),
        "body_ang_vel_w": zeros3.copy(),
        "anchor_rotation": np.eye(3),
        "global_anchor_rotation": np.eye(3),
        "racket_pos": np.zeros(3),
        "racket_velocity": np.zeros(3),
        "racket_normal": np.asarray((0.0, 1.0, 0.0)),
        "racket_long_axis": long_axis,
    }
    row = c211._c211_isaac_synonymous_prior_terms(
        live=live,
        teacher=teacher,
        current_action=np.asarray(current, dtype=np.float64),
        previous_action=np.asarray(previous, dtype=np.float64),
    )
    return row["terms"]["action_rate_l2"]


def test_c211_action_rate_responds_to_two_different_action_sequences():
    """该动的要动:两个不同的动作序列 -> 两个不同的数。"""

    small = _c211_action_rate(np.full(31, 0.1), np.zeros(31))
    large = _c211_action_rate(np.full(31, 1.0), np.zeros(31))
    assert small["raw_reward"] != large["raw_reward"]
    assert large["raw_reward"] > small["raw_reward"]
    assert small["post_policy_dt_reward"] > large["post_policy_dt_reward"]  # 都是负数


def test_c211_action_rate_is_bitwise_identical_on_the_same_sequence():
    """对照组:同一个序列喂两次 -> 逐位相同。"""

    first = _c211_action_rate(np.full(31, 0.37), np.full(31, -0.11))
    second = _c211_action_rate(np.full(31, 0.37), np.full(31, -0.11))
    assert first["raw_reward"] == second["raw_reward"]
    assert first["post_policy_dt_reward"] == second["post_policy_dt_reward"]


def test_c211_action_rate_matches_the_upstream_formula_and_has_no_ceiling():
    """A/C 同形:逐位等于上游 ``sum((a−a_prev)²)``,并且没有任何天花板。

    粗一档就过不了:重新引入 9.0(旧封顶档位)会让第二组塌到 9.0,该断言立刻红。
    """

    for current, previous in (
        (np.full(31, 1.0), np.zeros(31)),          # 31.0,旧档位下会被削成 9.0
        (np.full(31, 0.2), np.full(31, -0.3)),     # 31*0.25 = 7.75,旧档位下不变
        (np.full(31, 10.0), np.zeros(31)),         # 3100.0,离谱一帧照实付
    ):
        upstream = float(np.sum(np.square(current - previous)))
        assert _c211_action_rate(current, previous)["raw_reward"] == pytest.approx(
            upstream, rel=1e-12
        )
    # 线性无上界:输入的平方范数翻 100 倍,输出就翻 100 倍。
    base = _c211_action_rate(np.full(31, 1.0), np.zeros(31))["raw_reward"]
    scaled = _c211_action_rate(np.full(31, 10.0), np.zeros(31))["raw_reward"]
    assert scaled / base == pytest.approx(100.0, rel=1e-12)


@pytest.mark.parametrize(
    "distance_m,expected_reward",
    (
        (0.0, 4.8),
        (0.075, 3.84),
        (0.15, 2.4),
        (0.30, 0.96),
        (0.45, 0.48),
        (0.90, 4.8 / 37.0),
    ),
)
def test_exact_c211_strike_bridge_distance_vectors(distance_m, expected_reward):
    row = c211._c211_strike_reward_terms(
        official_racket_site_w_m=(distance_m, 0.0, 0.0),
        immutable_ball_contact_w_m=(0.0, 0.0, 0.0),
    )
    assert row["distance_m"] == pytest.approx(distance_m, abs=1.0e-15)
    assert row["post_policy_dt_reward"] == pytest.approx(
        expected_reward, abs=1.0e-14
    )


def test_exact_c211_landing_legal_offtable_and_zero_vectors():
    geometry = {
        "net_x_w_m": 1.87,
        "far_x_w_m": 3.24,
        "half_width_m": 0.7625,
    }
    legal = c211._c211_landing_reward_terms(
        landing_xy_w_m=(2.3, 0.0),
        landing_valid=True,
        net_crossed=True,
        net_clear=True,
        landing_aim_w_xy_m=(2.3, 0.0),
        **geometry,
    )
    assert legal["raw_reward"] == pytest.approx(1.0)
    assert legal["post_policy_dt_reward"] == pytest.approx(14.0)
    assert legal["classification"] == "legal_opponent_table"

    legal_far = c211._c211_landing_reward_terms(
        landing_xy_w_m=(3.24, 0.7625),
        landing_valid=True,
        net_crossed=True,
        net_clear=True,
        landing_aim_w_xy_m=(2.3, 0.0),
        **geometry,
    )
    assert 0.6 <= legal_far["raw_reward"] <= 1.0

    off = c211._c211_landing_reward_terms(
        landing_xy_w_m=(3.5, 0.0),
        landing_valid=True,
        net_crossed=True,
        net_clear=True,
        landing_aim_w_xy_m=(3.5, 0.0),
        **geometry,
    )
    assert off["classification"] == "opponent_side_off_table"
    assert off["raw_reward"] == pytest.approx(0.5)
    assert off["post_policy_dt_reward"] == pytest.approx(7.0)

    for landing_xy, net_crossed, net_clear in (
        ((1.5, 0.0), True, True),
        ((2.3, 0.0), False, False),
        ((2.3, 0.0), True, False),
    ):
        zero = c211._c211_landing_reward_terms(
            landing_xy_w_m=landing_xy,
            landing_valid=True,
            net_crossed=net_crossed,
            net_clear=net_clear,
            landing_aim_w_xy_m=(2.3, 0.0),
            **geometry,
        )
        assert zero["raw_reward"] == 0.0
        assert zero["post_policy_dt_reward"] == 0.0


def test_wait_keeps_isaac_prior_but_masks_task_and_ignores_fixed_center_scalar(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    env = _wrapped_fake_env(tmp_path, task, mimic, monkeypatch)
    _actor, rewards, dones, extras = env.step(
        torch.zeros((1, 31), dtype=torch.float32)
    )
    row = extras["reward_terms"][0]
    assert rewards.item() == pytest.approx(
        row["isaac_synonymous_prior_reward"], abs=1.0e-7
    )
    assert rewards.item() != 0.0
    assert dones.tolist() == [False]
    assert extras["reward_scope"] == c211.C211_REWARD_SCOPE
    assert extras["c211_achieved_outcome_reward_available"] is True
    assert extras["true_c211_training_lane_ready"] is False
    assert extras["fixed_center_rewards_not_consumed"].tolist() == pytest.approx(
        [0.125]
    )
    assert row["task_valid"] is False
    assert row["isaac_synonymous_prior_always_on"] is True
    assert row["isaac_synonymous_prior_task_mask_applied"] is False
    assert row["strike_reward"] == 0.0
    assert row["landing_reward"] == 0.0
    assert row["total_reward"] == pytest.approx(
        row["isaac_synonymous_prior_reward"]
    )
    assert set(row["isaac_synonymous_prior_terms"]) >= {
        "upright_exp",
        "base_ang_vel_xy",
        "base_lin_vel_z",
        "joint_vel",
        "action_rate_l2",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_racket_position",
        "motion_racket_velocity",
        "motion_racket_normal",
        "motion_racket_long_axis",
    }


def _selected_flight(contact_tick=92):
    return (
        n1_reward_event_kernel.ContactEvidence(
            True,
            n1_reward_event_kernel.EventStamp(contact_tick, 1),
            True,
        ),
        n1_reward_event_kernel.OutgoingFlightEvidence(
            True,
            n1_reward_event_kernel.EventStamp(contact_tick, 2),
            (0.7, 0.0, 1.1),
            (4.0, 0.0, 2.0),
            (0.0, 0.0, 0.0),
        ),
    )


def _prediction(landing_xy, *, legal=True):
    return {
        "landing_xy_w_m": list(landing_xy),
        "landing_valid": True,
        "net_crossed": True,
        "net_z_w_m": 1.1 if legal else 0.8,
        "net_clear": legal,
    }


def test_shared_fitted_rollout_consumes_achieved_outgoing_flight(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    env = _wrapped_fake_env(tmp_path, task, mimic, monkeypatch)
    _contact, flight = _selected_flight()
    predicted = env._predict_achieved_landing(flight)
    assert predicted["landing_valid"] is True
    assert predicted["net_crossed"] is True
    assert predicted["net_clear"] is True
    assert predicted["landing_xy_w_m"] == pytest.approx(
        (2.5515339374542236, 0.0), abs=1.0e-7
    )
    graded = c211._c211_landing_reward_terms(
        landing_xy_w_m=predicted["landing_xy_w_m"],
        landing_valid=predicted["landing_valid"],
        net_crossed=predicted["net_crossed"],
        net_clear=predicted["net_clear"],
        landing_aim_w_xy_m=task.landing_aim_w_xy_m,
        net_x_w_m=env._table_net_x,
        far_x_w_m=env._table_far_x,
        half_width_m=env._table_half_width,
    )
    assert graded["classification"] == "legal_opponent_table"
    assert 8.4 <= graded["post_policy_dt_reward"] <= 14.0


def test_exact_nominal_selected_flight_pays_once_and_only_task_reward(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    env = _wrapped_fake_env(tmp_path, task, mimic, monkeypatch)
    env.base.episode_length = 4
    env.step(torch.zeros((1, 31), dtype=torch.float32))  # RESET_WAIT
    env.base.official_racket_site_w_m = np.asarray(
        task.ball_contact_w_m, dtype=np.float64
    ) + np.asarray((0.15, 0.0, 0.0))
    monkeypatch.setattr(env, "_event_evidence", lambda _index, _facts: _selected_flight())
    monkeypatch.setattr(
        env,
        "_predict_achieved_landing",
        lambda _flight: _prediction(task.landing_aim_w_xy_m),
    )
    env.base.forced_policy_tick = 93
    _actor, rewards, dones, extras = env.step(
        torch.zeros((1, 31), dtype=torch.float32)
    )
    assert dones.tolist() == [False]
    row = extras["reward_terms"][0]
    assert rewards.item() == pytest.approx(
        16.4 + row["isaac_synonymous_prior_reward"], abs=1.0e-6
    )
    assert row["nominal_strike_sampled_now"] is True
    assert row["strike_terms"]["distance_m"] == pytest.approx(0.15)
    assert row["landing_terms"]["classification"] == "legal_opponent_table"
    assert row["outcome_evaluated_now"] is True

    env.base.forced_policy_tick = 94
    _actor, duplicate_rewards, _dones, duplicate_extras = env.step(
        torch.zeros((1, 31), dtype=torch.float32)
    )
    duplicate_row = duplicate_extras["reward_terms"][0]
    assert duplicate_rewards.item() == pytest.approx(
        duplicate_row["isaac_synonymous_prior_reward"], abs=1.0e-7
    )
    assert duplicate_row["landing_terms"] is None
    assert duplicate_row["outcome_evaluated_now"] is False


def test_terminal_pre_reset_selected_offtable_facts_are_paid_then_latch_resets(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    env = _wrapped_fake_env(tmp_path, task, mimic, monkeypatch)
    env.step(torch.zeros((1, 31), dtype=torch.float32))  # RESET_WAIT
    env.base.official_racket_site_w_m = np.asarray(
        task.ball_contact_w_m, dtype=np.float64
    )
    monkeypatch.setattr(env, "_event_evidence", lambda _index, _facts: _selected_flight())
    off_xy = (env._table_far_x + 0.2, task.landing_aim_w_xy_m[1])
    monkeypatch.setattr(
        env,
        "_predict_achieved_landing",
        lambda _flight: _prediction(off_xy),
    )
    env.base.forced_policy_tick = 93
    _actor, rewards, dones, extras = env.step(
        torch.zeros((1, 31), dtype=torch.float32)
    )
    assert dones.tolist() == [True]
    row = extras["reward_terms"][0]
    assert row["landing_terms"]["classification"] == "opponent_side_off_table"
    assert 0.0 < row["landing_terms"]["raw_reward"] <= 0.5
    assert rewards.item() == pytest.approx(
        row["isaac_synonymous_prior_reward"]
        + 4.8
        + row["landing_terms"]["post_policy_dt_reward"],
        abs=1.0e-6,
    )
    assert env.is_reset_boundary() is True
    assert env._reward_states[0].strike_sampled is False
    assert env._reward_states[0].outcome_evaluated is False


def test_task_invalid_masks_landing_but_early_contact_and_nonselected_are_distinct(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    env = _wrapped_fake_env(tmp_path, task, mimic, monkeypatch)
    monkeypatch.setattr(
        env,
        "_event_evidence",
        lambda _index, _facts: pytest.fail("WAIT must not consume task event facts"),
    )
    env.reset_reward_audit()
    env.base.forced_policy_tick = 93
    _actor, wait_reward, _done, _extras = env.step(
        torch.zeros((1, 31), dtype=torch.float32)
    )
    wait_row = _extras["reward_terms"][0]
    assert wait_reward.item() == pytest.approx(
        wait_row["isaac_synonymous_prior_reward"], abs=1.0e-7
    )
    assert wait_row["strike_reward"] == wait_row["landing_reward"] == 0.0
    audit = env.reward_audit_receipt()
    assert audit["transition_step_count"] == audit["row_count"] == 1
    assert audit["wait_row_count"] == 1
    assert audit["task_valid_row_count"] == 0
    assert audit["closed_attempt_count"] == 0
    assert audit["closed_attempt_without_selected_contact_count"] == 0
    assert audit["strike_sample_count"] == audit["landing_evaluation_count"] == 0
    assert audit["total_reward_sum"] == pytest.approx(wait_reward.item(), abs=1.0e-7)
    assert tuple(audit["prior_terms"]) == c211.C211_IMPLEMENTED_ISAAC_PRIOR_TERM_NAMES
    for name, term in wait_row["isaac_synonymous_prior_terms"].items():
        term_audit = audit["prior_terms"][name]
        assert term_audit["sample_count"] == 1
        assert term_audit["raw_reward_sum"] == pytest.approx(term["raw_reward"])
        assert term_audit["post_policy_dt_reward_sum"] == pytest.approx(
            term["post_policy_dt_reward"]
        )

    monkeypatch.setattr(
        env,
        "_event_evidence",
        lambda _index, _facts: _selected_flight(contact_tick=91),
    )
    monkeypatch.setattr(
        env,
        "_predict_achieved_landing",
        lambda _flight: {
            "landing_xy_w_m": task.landing_aim_w_xy_m,
            "landing_valid": True,
            "net_crossed": True,
            "net_clear": True,
            "net_z_w_m": 1.2,
        },
    )
    env.base.forced_policy_tick = 94
    _actor, wrong_tick_reward, _done, wrong_tick_extras = env.step(
        torch.zeros((1, 31), dtype=torch.float32)
    )
    wrong_tick_row = wrong_tick_extras["reward_terms"][0]
    assert wrong_tick_reward.item() == pytest.approx(
        wrong_tick_row["isaac_synonymous_prior_reward"], abs=1.0e-7
    )
    assert wrong_tick_extras["reward_terms"][0]["outcome_reason"] == (
        "contact_not_on_nominal_transition"
    )
    assert wrong_tick_extras["reward_terms"][0]["landing_terms"] is None

    env.reset()
    env.step(torch.zeros((1, 31), dtype=torch.float32))
    monkeypatch.setattr(
        env,
        "_event_evidence",
        lambda _index, _facts: (
            n1_reward_event_kernel.ContactEvidence(
                True, n1_reward_event_kernel.EventStamp(94, 1), False
            ),
            n1_reward_event_kernel.OutgoingFlightEvidence(
                False, None, None, None, None
            ),
        ),
    )
    env.base.forced_policy_tick = 94
    _actor, nonselected_reward, _done, nonselected_extras = env.step(
        torch.zeros((1, 31), dtype=torch.float32)
    )
    nonselected_row = nonselected_extras["reward_terms"][0]
    assert nonselected_reward.item() == pytest.approx(
        nonselected_row["isaac_synonymous_prior_reward"], abs=1.0e-7
    )
    assert (
        nonselected_extras["reward_terms"][0]["outcome_reason"]
        == "first_contact_not_selected_rubber"
    )


def test_wait_termination_does_not_enter_closed_attempt_denominator(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    env = _wrapped_fake_env(tmp_path, task, mimic, monkeypatch)
    env.base.episode_length = 1
    env.reset_reward_audit()
    monkeypatch.setattr(
        env,
        "_event_evidence",
        lambda _index, _facts: pytest.fail("WAIT must not consume task event facts"),
    )

    _actor, _reward, dones, extras = env.step(
        torch.zeros((1, 31), dtype=torch.float32)
    )

    assert dones.tolist() == [True]
    assert extras["reward_terms"][0]["task_valid"] is False
    assert extras["reward_terms"][0]["attempt_closed_now"] is False
    audit = env.reward_audit_receipt()
    assert audit["wait_row_count"] == 1
    assert audit["task_valid_row_count"] == 0
    assert audit["closed_attempt_count"] == 0
    assert audit["closed_attempt_without_selected_contact_count"] == 0
    assert audit["actual_contact_count"] == 0
    assert audit["selected_contact_count"] == 0
    assert audit["valid_achieved_flight_count"] == 0


def test_wrapped_c211_two_updates_compact_reset_and_cold_load_exact(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    env = _wrapped_fake_env(
        tmp_path, task, mimic, monkeypatch, split_ready=True
    )
    actor, extras = env.get_observations()
    assert tuple(actor.shape) == (1, 211)
    assert tuple(extras["observations"]["critic"].shape) == (1, 319)
    assert env.is_reset_boundary() is True
    readiness = env.diagnostic_training_receipt()
    assert readiness["kind"] == trainer.DIAGNOSTIC_TRAINER_RECEIPT_KIND
    assert readiness["actor_width"] == 211
    assert readiness["critic_width"] == 319
    assert readiness["safe_ready_authority_status"] == (
        "split_ready_physical_birth_diagnostic_only_cross_engine_unmeasured"
    )
    assert readiness["physical_birth_reset_semantics"] == (
        "split_ready_physical_safe_birth_separate_from_measured_"
        "teacher_frame0_stationary_hidden_wait"
    )
    assert readiness["safe_ready_formal_pass_claimed"] is False
    assert readiness["reward_available"] is True
    assert readiness["reward_scope"] == c211.C211_REWARD_SCOPE
    assert readiness["c211_achieved_outcome_reward_available"] is True
    assert readiness["isaac_synonymous_prior_subset_available"] is True
    assert readiness["full_body_measured_mimic_reward_available"] is True
    assert readiness["measured_paddle_prior_reward_available"] is True
    assert readiness["complete_isaac_reward_parity_claimed"] is False
    assert readiness["reward_parity_status"] == "partial_fail_closed"
    unavailable = {
        row["term"]: row["reason"]
        for row in readiness["unavailable_isaac_reward_terms"]
    }
    assert "foot_soft_landing" in unavailable
    assert "undesired_contacts" in unavailable
    assert "joint_torques" in unavailable
    assert readiness["true_c211_training_lane_ready"] is False
    assert any("split_ready" in value for value in readiness["formal_blockers"])
    assert any("full_body_mimic" in value for value in readiness["formal_blockers"])
    assert readiness["diagnostic_unauthorized"] is True
    bootstrap = readiness["fresh_actor_bootstrap"]
    gate = bootstrap["safety_gate"]
    forecast = gate["four_sigma_projection_forecast"]
    assert forecast["all_joints_strictly_inside"] is True
    assert forecast["sigma_envelope"] == 4.0
    assert forecast["role"] == (
        "analytic_projection_risk_forecast_not_launch_blocker"
    )
    assert gate["sealed_mean_gate"]["passed"] is True
    assert gate["plant_binding_sha256"] == "1" * 64
    assert gate["robot_tape_file_sha256"] == "8" * 64

    source = _wrapped_trainer(env)
    first = source.run_update()
    second = source.run_update()
    assert first["rollout_steps"] == second["rollout_steps"] == 2
    assert second["update_counter"] == 2
    assert second["at_reset_boundary"] is True
    path = tmp_path / "c211_reset_boundary.pt"
    checkpoint.ResetBoundaryCheckpoint().save(path, source)
    expected = source.run_update()
    expected_state = copy.deepcopy(source.model.state_dict())

    cold_env = _wrapped_fake_env(
        tmp_path, task, mimic, monkeypatch, split_ready=True
    )
    cold = _wrapped_trainer(cold_env)
    checkpoint.ResetBoundaryCheckpoint().load(path, cold)
    actual = cold.run_update()
    assert actual == expected
    for name, value in cold.model.state_dict().items():
        assert torch.equal(value, expected_state[name])


def test_fresh_actor_four_sigma_qdes_forecast_is_visible_but_not_a_hard_gate(
    tmp_path, monkeypatch, authorities
):
    task, mimic = authorities
    env = _wrapped_fake_env(
        tmp_path, task, mimic, monkeypatch, split_ready=True
    )
    binding = env._native.cores[0].binding
    bias = env.producer.robot_tape.history_fill_action
    mean = binding.default_joint_pos + binding.action_scale * bias
    excursion = 4.0 * 0.02 * abs(binding.action_scale[0])
    binding.executed_qdes_limits[0, 0] = mean[0] - excursion + 1.0e-9
    bootstrap = env.fresh_actor_bootstrap_contract()
    forecast = bootstrap["safety_gate"]["four_sigma_projection_forecast"]
    assert forecast["all_joints_strictly_inside"] is False
    assert forecast["joints_not_strictly_inside"] == ["joint_0"]
    assert forecast["role"] == (
        "analytic_projection_risk_forecast_not_launch_blocker"
    )
