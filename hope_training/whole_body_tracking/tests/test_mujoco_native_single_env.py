"""Contracts for the isolated diagnostic MuJoCo single-env core."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import single_env as core  # noqa: E402


CONTRACT = (
    REPO_ROOT
    / "configs/a3_vendor_runtime_authority_20260802_r8"
    / "bh_loop_c.shared_ready.training_contract.json"
)
TEACHER_MOTION = (
    REPO_ROOT
    / "assets/motions/chingmu73_measured_v4_20260803"
    / "hope_Take_061_unit04_BH.npz"
)
JOINT_ORDER_CONTRACT = REPO_ROOT / "configs/a3_joint_order_bijection_v1.json"


@pytest.fixture(scope="module")
def binding():
    return core.load_plant_binding(CONTRACT)


def test_real_schema3_contract_closes_exact_31d_plant(binding):
    assert len(binding.joint_names) == core.ACTION_DIM == 31
    assert len(set(binding.joint_names)) == 31
    assert binding.control_decimation == 4
    assert binding.physics_step_dt_s == pytest.approx(0.005)
    assert binding.policy_step_dt_s == pytest.approx(0.02)
    assert (binding.delay_min_steps, binding.delay_max_steps) == (0, 2)
    assert len(binding.source_sha256) == len(binding.binding_sha256) == 64
    assert binding.source_sha256 == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()


def test_delay_is_one_episode_fixed_whole_31d_row(binding):
    delay = core.ActionDelayLine(binding.delay_max_steps)
    fill = np.full(31, -7.0)
    a0 = np.arange(31, dtype=np.float64)
    a1 = a0 + 100.0
    a2 = a0 + 200.0
    delay.reset(2, fill)
    np.testing.assert_array_equal(delay.push(a0), fill)
    np.testing.assert_array_equal(delay.push(a1), fill)
    np.testing.assert_array_equal(delay.push(a2), a0)
    assert delay.state().shape == (3, 31)
    # A lag is not re-sampled by push; all 31 columns leave together.
    assert delay.delay_steps == 2


def test_delay_zero_has_no_queue_semantic_shift(binding):
    delay = core.ActionDelayLine(binding.delay_max_steps)
    delay.reset(0, np.zeros(31))
    for value in (1.0, 2.0, -3.0):
        action = np.full(31, value)
        np.testing.assert_array_equal(delay.push(action), action)


def test_action_decode_is_delay_then_affine_then_contract_clamp(binding):
    action = np.full(31, 1.0e6)
    raw, applied, count = binding.decode_action(action)
    np.testing.assert_allclose(raw, binding.default_joint_pos + binding.action_scale * action)
    np.testing.assert_array_less(binding.executed_qdes_limits[:, 0] - 1e-15, applied)
    np.testing.assert_array_less(applied, binding.executed_qdes_limits[:, 1] + 1e-15)
    np.testing.assert_array_equal(applied, binding.executed_qdes_limits[:, 1])
    assert count == 31


def test_schema3_execution_envelope_is_soft_five_percent_intersect_hard(binding):
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    soft = np.asarray(payload["qdes_joint_pos_limits"], dtype=np.float64)
    hard = payload["action_ball_ppo_runner_recipe"]["recipe"]["policy_initialization"][
        "hard_inner_guard"
    ]
    fraction = payload["finite_projection_soft_envelope_inset_fraction"]
    expected_lower = np.maximum(
        soft[:, 0] + fraction * (soft[:, 1] - soft[:, 0]),
        np.asarray(hard["hard_inner_lower"]),
    )
    expected_upper = np.minimum(
        soft[:, 1] - fraction * (soft[:, 1] - soft[:, 0]),
        np.asarray(hard["hard_inner_upper"]),
    )
    np.testing.assert_array_equal(
        binding.executed_qdes_limits, np.stack((expected_lower, expected_upper), axis=1)
    )
    assert binding.executed_qdes_limits[0, 1] == pytest.approx(2.414837598800659)


def test_nonimplicit_actuator_contract_fails_closed():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["joint_actuator_types"][0] = "explicit"
    with pytest.raises(core.ContractError, match="requires all 31.*implicit"):
        core.PlantBinding.from_mapping(
            payload, source_path=str(CONTRACT), source_sha256="0" * 64
        )


def test_total_pd_clips_the_combined_p_minus_d_not_each_term(binding):
    qdes = binding.default_joint_pos + 10.0
    q = binding.default_joint_pos.copy()
    qd = np.full(31, 1.0)
    raw, applied, count = core.total_pd_effort(binding, qdes, q, qd)
    expected_raw = binding.stiffness * 10.0 - binding.damping
    np.testing.assert_array_equal(raw, expected_raw)
    np.testing.assert_array_equal(
        applied, np.minimum(expected_raw, binding.effort_limits)
    )
    assert count == int(np.count_nonzero(expected_raw > binding.effort_limits))


def test_probe_tape_is_100x31_and_sha_bound(tmp_path, binding):
    payload = core.build_probe_tape(binding, delay_steps=2)
    assert np.asarray(payload["actions"]).shape == (100, 31)
    assert np.max(np.abs(np.asarray(payload["actions"]))) <= 0.020000000000001
    path = tmp_path / "fixed_tape.json"
    written_sha = core.write_fixed_tape(path, payload)
    assert written_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    tape = core.load_fixed_tape(path, binding)
    assert tape.source_sha256 == written_sha
    assert tape.plant_binding_sha256 == binding.binding_sha256
    np.testing.assert_array_equal(tape.actions, np.asarray(payload["actions"]))
    assert tape.reset_state.mode == (
        "named_stand_root_executed_zero_action_q_zero_velocity"
    )


def test_teacher_frame_tape_binds_selected_motion_q0_root_and_center(tmp_path, binding):
    payload = core.build_probe_tape(
        binding,
        delay_steps=2,
        teacher_motion=TEACHER_MOTION,
        teacher_frame_index=0,
    )
    path = tmp_path / "teacher_tape.json"
    core.write_fixed_tape(path, payload)
    tape = core.load_fixed_tape(path, binding)
    assert tape.reset_state.mode == "teacher_frame"
    assert tape.reset_state.source_motion_uid == "Take_061_unit04_BH"
    assert tape.reset_state.source_motion_sha256 == hashlib.sha256(
        TEACHER_MOTION.read_bytes()
    ).hexdigest()
    assert tape.reset_state.source_joint_order_contract_id == (
        "a3-gmr-dof-pos-to-runtime-articulation-v1"
    )
    assert tape.reset_state.source_joint_order_contract_sha256 == hashlib.sha256(
        JOINT_ORDER_CONTRACT.read_bytes()
    ).hexdigest()
    with np.load(TEACHER_MOTION, allow_pickle=False) as motion:
        np.testing.assert_array_equal(tape.reset_state.joint_pos, motion["joint_pos"][0])
        pelvis = list(motion["body_names"]).index("pelvis_link")
        np.testing.assert_array_equal(tape.reset_state.root_pos, motion["body_pos_w"][0, pelvis])
    _raw, applied, clamps = binding.decode_action(tape.history_fill_action)
    np.testing.assert_allclose(applied, tape.reset_state.joint_pos, rtol=0.0, atol=2e-6)
    assert clamps == 0


def test_teacher_frame_rejects_wrong_joint_order_contract_sha(tmp_path, binding):
    with np.load(TEACHER_MOTION, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["measured_racket_joint_order_contract_sha256"] = np.asarray("0" * 64)
    wrong = tmp_path / "wrong_joint_order_sha.npz"
    np.savez_compressed(wrong, **payload)
    with pytest.raises(core.ContractError, match="joint-order contract SHA-256 disagrees"):
        core.build_probe_tape(binding, delay_steps=0, teacher_motion=wrong)


def test_tape_consumer_revalidates_teacher_source_and_history_fill(tmp_path, binding):
    payload = core.build_probe_tape(
        binding, delay_steps=2, teacher_motion=TEACHER_MOTION
    )
    bad_fill = copy.deepcopy(payload)
    bad_fill["history_fill_action"] = [999.0] * core.ACTION_DIM
    bad_fill_path = tmp_path / "bad_teacher_fill.json"
    bad_fill_path.write_bytes(core._canonical_json_bytes(bad_fill))
    with pytest.raises(core.ContractError, match="history_fill_action"):
        core.load_fixed_tape(bad_fill_path, binding)

    missing_source = copy.deepcopy(payload)
    missing_source["reset_state"]["source_motion_path"] = str(
        tmp_path / "missing_teacher.npz"
    )
    missing_source_path = tmp_path / "missing_teacher_source.json"
    missing_source_path.write_bytes(core._canonical_json_bytes(missing_source))
    with pytest.raises(core.ContractError, match="cannot read teacher motion"):
        core.load_fixed_tape(missing_source_path, binding)


def test_named_stand_tape_rejects_nonzero_history_fill(tmp_path, binding):
    payload = core.build_probe_tape(binding, delay_steps=1)
    payload["history_fill_action"][0] = 1.0e-9
    path = tmp_path / "bad_stand_fill.json"
    path.write_bytes(core._canonical_json_bytes(payload))
    with pytest.raises(core.ContractError, match="exact zero/executed-qdes"):
        core.load_fixed_tape(path, binding)


def test_tape_rejects_wrong_binding_and_wrong_width(tmp_path, binding):
    payload = core.build_probe_tape(binding, delay_steps=0)
    wrong_binding = copy.deepcopy(payload)
    wrong_binding["plant_binding_sha256"] = "0" * 64
    path = tmp_path / "wrong_binding.json"
    path.write_bytes(core._canonical_json_bytes(wrong_binding))
    with pytest.raises(core.ContractError, match="different plant binding"):
        core.load_fixed_tape(path, binding)

    wrong_width = copy.deepcopy(payload)
    wrong_width["actions"][0].pop()
    path2 = tmp_path / "wrong_width.json"
    path2.write_bytes(core._canonical_json_bytes(wrong_width))
    with pytest.raises(core.ContractError, match="shape"):
        core.load_fixed_tape(path2, binding)


def test_strict_json_rejects_duplicate_key_and_nan(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(core.ContractError, match="duplicate JSON key"):
        core._load_strict_json(duplicate)
    nan = tmp_path / "nan.json"
    nan.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(core.ContractError, match="non-finite JSON"):
        core._load_strict_json(nan)


def test_output_is_no_clobber(tmp_path, binding):
    path = tmp_path / "tape.json"
    payload = core.build_probe_tape(binding, delay_steps=0)
    core.write_fixed_tape(path, payload)
    with pytest.raises(core.ContractError, match="refusing to overwrite"):
        core.write_fixed_tape(path, payload)


def test_trace_content_sha_binds_names_shapes_and_values():
    arrays = {"q": np.zeros((100, 31)), "qd": np.ones((100, 31))}
    metadata = {"ticks": 100, "action_dim": 31}
    digest = core._trace_content_sha256(arrays, metadata)
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["q"][99, 30] = 1.0e-12
    assert core._trace_content_sha256(changed, metadata) != digest
    assert core._trace_content_sha256(arrays, {**metadata, "ticks": 99}) != digest


def test_real_mujoco_a3_five_solid_100_tick_receipt(tmp_path, binding):
    pytest.importorskip("mujoco")
    payload = core.build_probe_tape(
        binding, delay_steps=0, teacher_motion=TEACHER_MOTION
    )
    tape_path = tmp_path / "fixed_tape.json"
    core.write_fixed_tape(tape_path, payload)
    tape = core.load_fixed_tape(tape_path, binding)
    runner = core.MujocoSingleEnv(binding)
    arrays, receipt = runner.run_tape(tape)
    assert all(value.shape == (100, 31) for value in arrays.values())
    assert receipt["status"] == "DIAGNOSTIC_FIXED_TAPE_COMPLETE"
    assert receipt["diagnostic_unauthorized"] is True
    assert not any(receipt["authorization"].values())
    assert receipt["counters"]["policy_ticks"] == 100
    assert receipt["counters"]["physics_substeps"] == 400
    assert receipt["reasons"]["fixed_tape_complete"] == 1
    assert receipt["runtime"]["delay_histogram_episode_count"] == {"0": 1}
    assert receipt["runtime"]["reset_mode"] == "teacher_frame"
    assert receipt["lineage"]["reset_state"]["source_motion_uid"] == (
        "Take_061_unit04_BH"
    )
    assert receipt["lineage"]["reset_state"][
        "source_joint_order_contract_sha256"
    ] == hashlib.sha256(JOINT_ORDER_CONTRACT.read_bytes()).hexdigest()
    assert receipt["lineage"]["fixed_tape_sha256"] == hashlib.sha256(
        tape_path.read_bytes()
    ).hexdigest()
    assert len(receipt["lineage"]["trace_content_sha256"]) == 64
    assert "max_self_penetration_m" in receipt["safety"]
    assert "worst_self_contact_pair" in receipt["safety"]
    assert receipt["safety"]["diagnostic_no_contact_gate_passed"] is (
        not receipt["safety"]["self_contact_observed"]
        and not receipt["safety"]["table_contact_observed"]
        and receipt["reasons"]["joint_velocity_limit_observed"] == 0
    )
    assert receipt["safety"]["safe_for_hardware_claim"] is False


def test_cli_make_tape_receipt_is_explicitly_diagnostic(tmp_path, capsys):
    tape = tmp_path / "probe.json"
    rc = core.main(
        [
            "make-tape",
            "--contract",
            str(CONTRACT),
            "--tape",
            str(tape),
            "--delay",
            "1",
        ]
    )
    assert rc == 0
    row = json.loads(capsys.readouterr().out)
    assert row["diagnostic_unauthorized"] is True
    assert row["ticks"] == 100 and row["action_dim"] == 31
    assert row["sha256"] == hashlib.sha256(tape.read_bytes()).hexdigest()
