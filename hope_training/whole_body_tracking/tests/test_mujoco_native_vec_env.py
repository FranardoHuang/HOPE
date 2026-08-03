"""Contracts for the no-reward native MuJoCo diagnostic VecEnv adapter."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import n1_ball_core as n1  # noqa: E402
from mujoco_native import single_env  # noqa: E402
from mujoco_native import vec_env  # noqa: E402


CONTRACT = (
    REPO_ROOT
    / "configs/a3_vendor_runtime_authority_20260802_r8"
    / "bh_loop_c.shared_ready.training_contract.json"
)


def _groups(offset: float = 0.0):
    return {
        name: np.arange(width, dtype=np.float64) + offset
        for name, width in vec_env.OBSERVATION_LAYOUT
    }


def _phase_sample(**overrides):
    sample = {
        "schema_version": 1,
        "kind": "a3_mujoco_phase_fidelity_sample_v1",
        "motion_phase_context": "non_hold_swing_or_follow_through",
        "in_hold": False,
        "reference_terminations_enabled": True,
        "anchor_pos_z_error_m": 0.0,
        "anchor_projected_gravity_z_error_abs": 0.0,
        "ee_body_pos_z_error_m": [0.0, 0.0, 0.0, 0.0],
    }
    sample.update(overrides)
    return sample


def test_flattened_layout_is_explicit_76d_and_fail_closed():
    flat = vec_env.flatten_observation_groups(_groups())
    assert vec_env.OBSERVATION_WIDTH == 76
    assert flat.shape == (76,)
    cursor = 0
    for name, width in vec_env.OBSERVATION_LAYOUT:
        np.testing.assert_array_equal(flat[cursor : cursor + width], _groups()[name])
        cursor += width

    missing = _groups()
    missing.pop("validity")
    with pytest.raises(vec_env.VecEnvContractError, match="groups differ"):
        vec_env.flatten_observation_groups(missing)
    wrong = _groups()
    wrong["landing_aim_xy_w_m"] = np.zeros(3)
    with pytest.raises(vec_env.VecEnvContractError, match="must be 2"):
        vec_env.flatten_observation_groups(wrong)


def test_reward_blocker_prohibits_fake_ppo_checkpoint_and_resume():
    receipt = vec_env.reward_blocker_receipt()
    assert receipt["status"] == "PPO_BLOCKED_MISSING_REAL_REWARD_CONTRACT"
    assert receipt["reward_available"] is False
    assert receipt["zero_reward_allowed"] is False
    assert receipt["improvised_proxy_reward_allowed"] is False
    assert set(receipt["blockers"]) == set(vec_env.REWARD_BLOCKERS)
    assert "optimizer_update" in receipt["prohibited_scope"]
    assert "cold_load_resume" in receipt["prohibited_scope"]
    assert receipt["enforcement_scope"]["vecenv_step_raises_before_physics"] is True
    assert receipt["enforcement_scope"]["upstream_runner_save_load_intercepted"] is False
    assert not any(receipt["authorization"].values())

    termination = vec_env.termination_blocker_receipt()
    assert termination["formal_termination_available"] is False
    assert termination["terminated_tensor_available"] is False
    assert termination["exact_base_subset_available"] is True
    assert termination["exact_base_subset_terminated_tensor_available"] is True
    assert termination["exact_robot_table_termination_available"] is True
    assert termination["exact_hard_subset_terminated_tensor_available"] is True
    assert termination["exact_time_out_latch_available"] is True
    assert termination["exact_episode_done_tensor_available"] is True
    assert termination["exact_phase_fidelity_predicate_available"] is True
    assert termination["exact_phase_fidelity_runtime_sample_available"] is False
    assert set(termination["blockers"]) == set(
        vec_env.FORMAL_TERMINATION_BLOCKERS
    )
    assert termination["reward_paid"] is False
    assert not any(termination["authorization"].values())
    exact = termination["exact_base_subset"]
    assert exact["reason_order"] == [
        "base_fell_tilt",
        "base_too_low",
        "joint_qdes_forbidden",
        "joint_actual_forbidden",
    ]
    assert exact["base_fell_tilt"]["limit_angle_rad"] == 0.7
    assert exact["base_too_low"]["minimum_height_m"] == 0.5
    assert len(exact["source_config_semantic_ast_sha256"]) == 64
    assert (
        exact["source_config_semantic_ast_sha256"]
        == vec_env.EXPECTED_PHASE_CONFIG_SEMANTIC_AST_SHA256
    )
    assert Path(exact["source_config_path"]) == vec_env.TERMINATION_SOURCE_CONFIG
    assert (
        exact["source_callables_semantic_ast_sha256"]
        == vec_env.EXPECTED_PHASE_RAW_CALLABLES_SEMANTIC_AST_SHA256
    )
    assert Path(exact["source_callables_path"]) == vec_env.TERMINATION_SOURCE_CALLABLES
    assert exact["joint_qdes_forbidden"] == {
        "source_callable": "pre_clamp_qdes_forbidden_zone",
        "source_config": "HOPEActionBallTerminationsCfg.joint_qdes_forbidden",
        "limit_source": "joint_pos_limits",
        "margin_rad": 0.0,
        "margin_fraction": 0.02,
        "finite_preclamp_qdes_projection_enabled": True,
        "mujoco_predicate": "any(nonfinite(qdes_raw))",
        "finite_request_semantics": (
            "project and retain transition; the projection penalty owns the event"
        ),
        "sample_timing": "post_control_step",
    }
    assert exact["joint_actual_forbidden"]["bounds_tolerance_rad"] == 0.0
    assert termination["exact_hard_reason_order"] == [
        "anchor_pos",
        "anchor_ori",
        "ee_body_pos",
        "base_fell_tilt",
        "base_too_low",
        "robot_hit_table",
        "joint_qdes_forbidden",
        "joint_actual_forbidden",
    ]
    assert termination["exact_active_reason_order"] == [
        "time_out",
        "anchor_pos",
        "anchor_ori",
        "ee_body_pos",
        "base_fell_tilt",
        "base_too_low",
        "robot_hit_table",
        "joint_qdes_forbidden",
        "joint_actual_forbidden",
    ]
    phase = termination["exact_phase_fidelity_subset"]
    assert phase["reason_order"] == ["anchor_pos", "anchor_ori", "ee_body_pos"]
    assert phase["sample_contract"]["comparison"] == "strict_greater_than"
    assert phase["sample_contract"]["ee_body_order"] == list(
        vec_env.PHASE_EE_BODY_NAMES
    )
    assert phase["source_wrappers_semantic_ast_sha256"] == (
        vec_env.EXPECTED_PHASE_WRAPPERS_SEMANTIC_AST_SHA256
    )
    assert phase["source_gate_semantic_ast_sha256"] == (
        vec_env.EXPECTED_PHASE_GATE_SEMANTIC_AST_SHA256
    )
    table = termination["exact_robot_table"]
    assert table["sticky_within_control_step"] is True
    assert table["required_control_decimation"] == 4
    assert table["episode_sticky_owner"] == "DiagnosticEventLedger"
    assert table["diagnostic_step_after_latch_requires_explicit_reset"] is False
    assert table["immediate_compact_reset_implemented"] is True
    assert table["source_action_latch_semantic_ast_sha256"] == (
        vec_env.table_termination.EXPECTED_ISAAC_ACTION_LATCH_SEMANTIC_AST_SHA256
    )
    assert table["required_portable_mujoco_identity_sha256"] == (
        vec_env.table_termination.EXPECTED_PORTABLE_MUJOCO_IDENTITY_SHA256
    )
    assert table["resolved_contact_required"] is False
    assert table["collision_proxy_sha256"] == (
        vec_env.table_termination.EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256
    )
    compact = termination["per_env_compact_reset"]
    assert compact["available"] is True
    assert compact["nonterminated_rows_advance_without_reset"] is True
    assert compact["returned_observations"] == (
        "post_compact_reset_next_observations"
    )
    termination["blockers"].append("caller_mutation")
    assert "caller_mutation" not in vec_env.termination_blocker_receipt()["blockers"]
    runtime = vec_env.termination_blocker_receipt(
        phase_fidelity_runtime_available=True
    )
    assert runtime["formal_termination_available"] is True
    assert runtime["terminated_tensor_available"] is True
    assert runtime["exact_phase_fidelity_runtime_sample_available"] is True
    assert runtime["blockers"] == []
    assert not any(runtime["authorization"].values())


def test_termination_receipt_fails_closed_if_pinned_source_drifts(
    tmp_path, monkeypatch
):
    drifted = tmp_path / "hope_env_cfg.py"
    source = vec_env.TERMINATION_SOURCE_CONFIG.read_text(encoding="utf-8")
    old = 'params={"command_name": "motion", "threshold": 0.25, "ignore_hold": True}'
    assert old in source
    drifted.write_text(
        source.replace(old, old.replace("0.25", "0.26"), 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_CONFIG", drifted)
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()
    with pytest.raises(
        vec_env.VecEnvContractError,
        match="action_ball_config semantic AST SHA-256 drifted",
    ):
        vec_env.termination_blocker_receipt()
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()


def test_termination_receipt_ignores_unrelated_config_class_assignment(
    tmp_path, monkeypatch
):
    expected_phase = vec_env.phase_fidelity_sample_contract()
    expected_table = vec_env.table_termination.verify_isaac_source_authority()
    source = vec_env.TERMINATION_SOURCE_CONFIG.read_text(encoding="utf-8")
    marker = (
        'class HOPEDeployParityTerminationsCfg(TerminationsCfg):\n'
        '    """Swing-only reference envelopes plus always-on absolute fall/sink guards."""\n'
    )
    assert marker in source
    unrelated = tmp_path / "hope_env_cfg.py"
    unrelated.write_text(
        source.replace(
            marker,
            marker + "\n    unrelated_a225_leaf_marker = 1\n",
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_CONFIG", unrelated)
    monkeypatch.setattr(
        vec_env.table_termination, "ISAAC_TERMINATION_CONFIG", unrelated
    )
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()
    receipt = vec_env.termination_blocker_receipt()
    assert vec_env.phase_fidelity_sample_contract() == expected_phase
    assert vec_env.table_termination.verify_isaac_source_authority() == expected_table
    assert receipt["formal_termination_available"] is False
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()


def test_termination_receipt_fails_closed_if_pinned_callable_source_drifts(
    tmp_path, monkeypatch
):
    drifted = tmp_path / "terminations.py"
    source = vec_env.TERMINATION_SOURCE_CALLABLES.read_text(encoding="utf-8")
    old = (
        "return torch.abs(command.anchor_pos_w[:, -1] - "
        "command.robot_anchor_pos_w[:, -1]) > threshold"
    )
    assert old in source
    drifted.write_text(
        source.replace(old, old.replace("> threshold", ">= threshold"), 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_CALLABLES", drifted)
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()
    with pytest.raises(
        vec_env.VecEnvContractError,
        match="raw_callables semantic AST SHA-256 drifted",
    ):
        vec_env.termination_blocker_receipt()
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()


def test_termination_receipt_fails_closed_if_pinned_action_latch_drifts(
    tmp_path, monkeypatch
):
    drifted = tmp_path / "hope_actions.py"
    source = vec_env.TERMINATION_SOURCE_ACTION_LATCH.read_text(encoding="utf-8")
    old = "return latch.finalize(self._sample_table_contact_current())"
    assert old in source
    drifted.write_text(
        source.replace(old, "return latch.hit", 1), encoding="utf-8"
    )
    monkeypatch.setattr(vec_env.table_termination, "ISAAC_ACTION_LATCH", drifted)
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()
    with pytest.raises(
        vec_env.VecEnvContractError,
        match="action-latch semantic AST SHA-256 drifted",
    ):
        vec_env.termination_blocker_receipt()
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()


def test_phase_fidelity_contract_fails_closed_if_wrapper_source_drifts(
    tmp_path, monkeypatch
):
    drifted = tmp_path / "hope_rewards.py"
    source = vec_env.TERMINATION_SOURCE_PHASE_WRAPPERS.read_text(encoding="utf-8")
    old = "return value & ~command.in_hold.bool()"
    assert old in source
    drifted.write_text(
        source.replace(old, "return value", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_PHASE_WRAPPERS", drifted)
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    with pytest.raises(
        vec_env.VecEnvContractError,
        match="hold_aware_wrappers semantic AST SHA-256 drifted",
    ):
        vec_env.phase_fidelity_sample_contract()
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()


def test_phase_fidelity_semantic_pin_ignores_unrelated_wrapper_append(
    tmp_path, monkeypatch
):
    expected = vec_env.phase_fidelity_sample_contract()
    source = vec_env.TERMINATION_SOURCE_PHASE_WRAPPERS.read_text(encoding="utf-8")
    unrelated = tmp_path / "hope_rewards.py"
    unrelated.write_text(
        source + "\n\ndef unrelated_phase_pin_probe():\n    return 17\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_PHASE_WRAPPERS", unrelated)
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    assert vec_env.phase_fidelity_sample_contract() == expected
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()


def test_phase_fidelity_contract_and_predicate_are_strict_hold_aware_and_gated():
    contract = vec_env.phase_fidelity_sample_contract()
    assert contract["kind"] == "a3_mujoco_phase_fidelity_sample_contract_v1"
    assert len(contract["content_sha256"]) == 64
    assert set(contract["authority_sources"]) == {
        "action_ball_config",
        "raw_callables",
        "hold_aware_wrappers",
        "frozen_phase_gate",
        "base_config",
        "a3_body_names",
    }
    at_boundary = _phase_sample(
        anchor_pos_z_error_m=vec_env.PHASE_ANCHOR_POS_Z_THRESHOLD_M,
        anchor_projected_gravity_z_error_abs=(
            vec_env.PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD
        ),
        ee_body_pos_z_error_m=[
            vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M
        ] * 4,
    )
    assert vec_env.exact_phase_fidelity_reasons(at_boundary) == ()

    above = _phase_sample(
        anchor_pos_z_error_m=np.nextafter(
            vec_env.PHASE_ANCHOR_POS_Z_THRESHOLD_M, np.inf
        ),
        anchor_projected_gravity_z_error_abs=np.nextafter(
            vec_env.PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD, np.inf
        ),
        ee_body_pos_z_error_m=[
            0.0,
            np.nextafter(vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M, np.inf),
            0.0,
            0.0,
        ],
    )
    assert vec_env.exact_phase_fidelity_reasons(above) == (
        "anchor_pos",
        "anchor_ori",
        "ee_body_pos",
    )
    assert vec_env.exact_phase_fidelity_reasons(
        {
            **above,
            "motion_phase_context": "recovery_hold",
            "in_hold": True,
        }
    ) == ()
    assert vec_env.exact_phase_fidelity_reasons(
        {**above, "reference_terminations_enabled": False}
    ) == ()


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (
            _phase_sample(
                motion_phase_context="recovery_hold",
                in_hold=False,
            ),
            "hold gate disagrees",
        ),
        (
            {
                key: value
                for key, value in _phase_sample().items()
                if key != "reference_terminations_enabled"
            },
            "keys differ",
        ),
        (_phase_sample(anchor_pos_z_error_m=np.nan), "finite and >=0"),
        (_phase_sample(ee_body_pos_z_error_m=[0.0] * 3), "four finite"),
    ],
)
def test_phase_fidelity_sample_schema_fails_without_ledger_commit(sample, message):
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    with pytest.raises(vec_env.VecEnvContractError, match=message):
        ledger.record_step(
            plant=_plant_row(),
            events=(),
            time_out=False,
            phase_fidelity_sample=sample,
        )
    assert ledger.policy_ticks == 0
    assert ledger.phase_fidelity_samples == 0


def test_rsl_step_raises_before_touching_physics():
    instance = object.__new__(vec_env.MujocoN1DiagnosticVecEnv)
    with pytest.raises(vec_env.RewardContractMissing, match="before physics"):
        instance.step(object())


class _FakeCore:
    def __init__(self, index: int, *, control_decimation: int = 4):
        self.index = index
        self.binding = SimpleNamespace(
            binding_sha256="a" * 64,
            policy_step_dt_s=0.02,
            control_decimation=control_decimation,
        )
        self.scene_binding_sha256 = "b" * 64
        self.ticks = 0

    def reset(self, *, robot_tape, question):
        assert robot_tape.plant_binding_sha256 == "a" * 64
        assert question.scene_binding_sha256 == "b" * 64
        self.ticks = 0
        return _groups(float(self.index))

    def step(self, action):
        assert np.asarray(action).shape == (31,)
        self.ticks += 1
        return {
            "plant": _plant_row(),
            "observation_groups": _groups(float(self.index + self.ticks)),
            "new_events": [],
        }


class _NativeEventCore(_FakeCore):
    native_physical_event_contract_sha256 = (
        vec_env.n1_reward_event_kernel.native_physical_event_facts_contract()[
            "content_sha256"
        ]
    )

    @property
    def native_physical_event_source_binding(self):
        return vec_env.n1_reward_event_kernel.SourceBinding(
            source_id=f"fake-native-event-core-{self.index}",
            source_sha256=f"{self.index + 1:064x}",
            event_contract_sha256=(
                self.native_physical_event_contract_sha256
            ),
        )

    def step(self, action):
        result = super().step(action)
        source = self.native_physical_event_source_binding
        result["native_physical_event_facts"] = {
            "schema_version": 1,
            "kind": (
                vec_env.n1_reward_event_kernel.NATIVE_PHYSICAL_EVENT_FACTS_KIND
            ),
            "source": {
                "source_id": source.source_id,
                "source_sha256": source.source_sha256,
                "event_contract_sha256": source.event_contract_sha256,
            },
            "policy_tick": self.ticks - 1,
            "racket_contact_edge_count_total": 0,
            "first_racket_contact_stamp": None,
            "outgoing_flight": None,
            "invalid_reasons": [],
            "selected_rubber_authority_available": False,
        }
        return result


def _plant_row(**overrides):
    row = {
        "qdes_clamp_joint_events": 0,
        "effort_clip_joint_events": 0,
        "velocity_limit_joint_events": 0,
        "table_contact_pairs": 0,
        "self_contact_pairs": 0,
        "table_contact_substeps": 0,
        "self_contact_substeps": 0,
        "max_table_penetration_m": 0.0,
        "max_self_penetration_m": 0.0,
        "max_joint_velocity_ratio": 0.25,
        "pelvis_height_m": 1.0,
        "pelvis_up_world_z": 1.0,
        "qdes_raw": np.zeros(31, dtype=np.float64),
        "q": np.zeros(31, dtype=np.float64),
        "joint_position_limits": np.tile(
            np.asarray([-1.0, 1.0], dtype=np.float64), (31, 1)
        ),
        "joint_actual_forbidden_substep": False,
        "robot_hit_table_substep": False,
        "robot_hit_table_first_substep": None,
        "first_table_contact_pair": None,
        "first_self_contact_pair": None,
    }
    row.update(overrides)
    return row


def test_vecenv_rejects_nonfour_control_decimation_before_reset():
    core = _FakeCore(0, control_decimation=3)
    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((1, 31), dtype=np.float64),
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    with pytest.raises(vec_env.VecEnvContractError, match="control_decimation=4"):
        vec_env.MujocoN1DiagnosticVecEnv(
            cores=(core,), robot_tape=tape, questions=(question,)
        )


def test_vecenv_rejects_malformed_per_env_question_source_sha():
    core = _FakeCore(0)
    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((1, 31), dtype=np.float64),
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="NOT_A_SHA",
    )
    with pytest.raises(vec_env.VecEnvContractError, match="lowercase SHA-256"):
        vec_env.MujocoN1DiagnosticVecEnv(
            cores=(core,), robot_tape=tape, questions=(question,)
        )


def test_vecenv_rejects_mixed_phase_fidelity_abi_advertisement_before_torch():
    cores = (_FakeCore(0), _FakeCore(1))
    cores[0].phase_fidelity_sample_contract_sha256 = (
        vec_env.phase_fidelity_sample_contract()["content_sha256"]
    )
    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((2, 31), dtype=np.float64),
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    with pytest.raises(vec_env.VecEnvContractError, match="every core or none"):
        vec_env.MujocoN1DiagnosticVecEnv(
            cores=cores, robot_tape=tape, questions=(question, question)
        )


def test_vecenv_rejects_mixed_native_event_abi_advertisement_before_torch():
    cores = (_NativeEventCore(0), _FakeCore(1))
    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((2, 31), dtype=np.float64),
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    with pytest.raises(vec_env.VecEnvContractError, match="every core or none"):
        vec_env.MujocoN1DiagnosticVecEnv(
            cores=cores, robot_tape=tape, questions=(question, question)
        )


def test_vecenv_rejects_wrong_phase_fidelity_abi_sha_before_torch():
    core = _FakeCore(0)
    core.phase_fidelity_sample_contract_sha256 = "0" * 64
    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((2, 31), dtype=np.float64),
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    with pytest.raises(vec_env.VecEnvContractError, match="different phase-fidelity"):
        vec_env.MujocoN1DiagnosticVecEnv(
            cores=(core,), robot_tape=tape, questions=(question,)
        )


def test_event_ledger_validates_substeps_latches_facts_and_refuses_termination():
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    first = ledger.record_step(
        plant=_plant_row(
            table_contact_pairs=2,
            table_contact_substeps=1,
            max_table_penetration_m=0.001,
            first_table_contact_pair="robot~table",
        ),
        events=(
            {
                "policy_tick": 0,
                "physics_substep": 2,
                "time_s": 0.015,
                "event": "racket",
            },
        ),
        time_out=False,
    )
    assert first["policy_ticks"] == 1
    assert first["physics_substeps"] == 4
    assert first["contact_edge_counts"]["racket"] == 1
    assert first["latches"]["ball_racket_contact_seen"] is True
    assert first["latches"]["robot_obstacle_contact_seen"] is True
    assert first["first_robot_obstacle_contact"] == {
        "policy_tick": 0,
        "pair": "robot~table",
    }
    assert first["termination"] == {
        "exact_time_out_latched": False,
        "exact_base_subset_available": True,
        "exact_robot_table_termination_available": True,
        "exact_hard_subset_available": True,
        "exact_phase_fidelity_predicate_available": True,
        "exact_phase_fidelity_runtime_sample_seen": False,
        "exact_hard_terminated": False,
        "exact_hard_reason": None,
        "formal_hard_termination_available": False,
        "formal_hard_terminated": None,
        "blocker_sha256": vec_env.termination_blocker_receipt()[
            "content_sha256"
        ],
    }
    assert first["reward_paid"] is False

    second = ledger.record_step(
        plant=_plant_row(velocity_limit_joint_events=1),
        events=(),
        time_out=True,
    )
    assert second["policy_ticks"] == 2
    assert second["physics_substeps"] == 8
    assert second["termination"]["exact_time_out_latched"] is True
    assert second["latches"]["joint_velocity_limit_seen"] is True
    assert second["termination"]["formal_hard_terminated"] is None


def test_event_ledger_exact_base_thresholds_order_and_latch_are_strict():
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    at_boundary = ledger.record_step(
        plant=_plant_row(
            pelvis_height_m=vec_env.BASE_TOO_LOW_MINIMUM_HEIGHT_M,
            pelvis_up_world_z=vec_env.BASE_FELL_TILT_MIN_UP_WORLD_Z,
        ),
        events=(),
        time_out=False,
    )
    assert at_boundary["termination"]["exact_hard_terminated"] is False

    simultaneous = ledger.record_step(
        plant=_plant_row(
            pelvis_height_m=np.nextafter(
                vec_env.BASE_TOO_LOW_MINIMUM_HEIGHT_M, -np.inf
            ),
            pelvis_up_world_z=np.nextafter(
                vec_env.BASE_FELL_TILT_MIN_UP_WORLD_Z, -np.inf
            ),
        ),
        events=(),
        time_out=False,
    )
    assert simultaneous["termination"]["exact_hard_terminated"] is True
    assert simultaneous["termination"]["exact_hard_reason"] == "base_fell_tilt"
    assert simultaneous["first_exact_hard_termination"] == {
        "policy_tick": 1,
        "sample_timing": "post_control_step",
        "physics_substep": None,
        "robot_hit_table_first_substep": None,
        "reason": "base_fell_tilt",
        "all_reasons": ["base_fell_tilt", "base_too_low"],
    }
    assert simultaneous["exact_hard_reason_counts"] == {
        "anchor_pos": 0,
        "anchor_ori": 0,
        "ee_body_pos": 0,
        "base_fell_tilt": 1,
        "base_too_low": 1,
        "joint_qdes_forbidden": 0,
        "joint_actual_forbidden": 0,
        "robot_hit_table": 0,
    }
    simultaneous["first_exact_hard_termination"]["all_reasons"].append(
        "caller_corruption"
    )

    recovered_sample = ledger.record_step(
        plant=_plant_row(), events=(), time_out=False
    )
    assert recovered_sample["termination"]["exact_hard_terminated"] is True
    assert recovered_sample["termination"]["exact_hard_reason"] == "base_fell_tilt"
    assert recovered_sample["first_exact_hard_termination"]["all_reasons"] == [
        "base_fell_tilt",
        "base_too_low",
    ]


def test_event_ledger_phase_fidelity_precedes_base_and_table_reason_order():
    above = np.nextafter(vec_env.PHASE_ANCHOR_POS_Z_THRESHOLD_M, np.inf)
    sample = _phase_sample(
        anchor_pos_z_error_m=above,
        anchor_projected_gravity_z_error_abs=np.nextafter(
            vec_env.PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD, np.inf
        ),
        ee_body_pos_z_error_m=[
            np.nextafter(vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M, np.inf),
            0.0,
            0.0,
            0.0,
        ],
    )
    result = vec_env.DiagnosticEventLedger(control_decimation=4).record_step(
        plant=_plant_row(
            pelvis_height_m=0.49,
            robot_hit_table_substep=True,
            robot_hit_table_first_substep=3,
            joint_actual_forbidden_substep=True,
        ),
        events=(),
        time_out=False,
        phase_fidelity_sample=sample,
    )
    assert result["first_exact_hard_termination"] == {
        "policy_tick": 0,
        "sample_timing": "post_control_step",
        "physics_substep": None,
        "robot_hit_table_first_substep": 3,
        "reason": "anchor_pos",
        "all_reasons": [
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
            "base_too_low",
            "robot_hit_table",
            "joint_actual_forbidden",
        ],
    }
    assert result["phase_fidelity"]["exact_sample_count"] == 1
    assert result["phase_fidelity"]["exact_runtime_sample_seen"] is True
    assert result["exact_hard_reason_counts"]["anchor_pos"] == 1
    assert result["exact_hard_reason_counts"]["anchor_ori"] == 1
    assert result["exact_hard_reason_counts"]["ee_body_pos"] == 1


def test_event_ledger_robot_table_substep_is_sticky_and_isaac_ordered():
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    table_only = ledger.record_step(
        plant=_plant_row(
            robot_hit_table_substep=True,
            robot_hit_table_first_substep=1,
        ),
        events=(),
        time_out=False,
    )
    assert table_only["termination"]["exact_hard_terminated"] is True
    assert table_only["termination"]["exact_hard_reason"] == "robot_hit_table"
    assert table_only["first_exact_hard_termination"] == {
        "policy_tick": 0,
        "sample_timing": "physics_substep",
        "physics_substep": 1,
        "robot_hit_table_first_substep": 1,
        "reason": "robot_hit_table",
        "all_reasons": ["robot_hit_table"],
    }
    recovered = ledger.record_step(plant=_plant_row(), events=(), time_out=False)
    assert recovered["termination"]["exact_hard_terminated"] is True
    assert recovered["termination"]["exact_hard_reason"] == "robot_hit_table"
    assert recovered["exact_hard_reason_counts"]["robot_hit_table"] == 1

    simultaneous = vec_env.DiagnosticEventLedger(control_decimation=4).record_step(
        plant=_plant_row(
            pelvis_height_m=0.49,
            joint_actual_forbidden_substep=True,
            robot_hit_table_substep=True,
            robot_hit_table_first_substep=0,
        ),
        events=(),
        time_out=False,
    )
    assert simultaneous["first_exact_hard_termination"]["reason"] == "base_too_low"
    assert simultaneous["first_exact_hard_termination"]["all_reasons"] == [
        "base_too_low",
        "robot_hit_table",
        "joint_actual_forbidden",
    ]
    assert simultaneous["first_exact_hard_termination"][
        "robot_hit_table_first_substep"
    ] == 0


@pytest.mark.parametrize(
    ("hit", "substep"),
    [(True, None), (True, 4), (False, 0), (False, False)],
)
def test_event_ledger_robot_table_substep_schema_fails_without_commit(hit, substep):
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    with pytest.raises(vec_env.VecEnvContractError, match="robot/table"):
        ledger.record_step(
            plant=_plant_row(
                robot_hit_table_substep=hit,
                robot_hit_table_first_substep=substep,
            ),
            events=(),
            time_out=False,
        )
    assert ledger.policy_ticks == 0
    assert ledger.exact_hard_termination_latched is False


def test_event_ledger_joint_actual_bounds_tolerance_order_and_latch_are_strict():
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    tolerance = vec_env.JOINT_ACTUAL_FORBIDDEN_BOUNDS_TOLERANCE_RAD
    safe_q = np.zeros(31, dtype=np.float64)
    safe_q[7] = np.nextafter(-1.0, np.inf)
    safe = ledger.record_step(
        plant=_plant_row(q=safe_q), events=(), time_out=False
    )
    assert safe["termination"]["exact_hard_terminated"] is False

    boundary_q = np.zeros(31, dtype=np.float64)
    boundary_q[7] = -1.0 + tolerance
    simultaneous = ledger.record_step(
        plant=_plant_row(
            q=boundary_q,
            pelvis_height_m=np.nextafter(
                vec_env.BASE_TOO_LOW_MINIMUM_HEIGHT_M, -np.inf
            ),
            pelvis_up_world_z=np.nextafter(
                vec_env.BASE_FELL_TILT_MIN_UP_WORLD_Z, -np.inf
            ),
        ),
        events=(),
        time_out=False,
    )
    assert simultaneous["first_exact_hard_termination"] == {
        "policy_tick": 1,
        "sample_timing": "post_control_step",
        "physics_substep": None,
        "robot_hit_table_first_substep": None,
        "reason": "base_fell_tilt",
        "all_reasons": [
            "base_fell_tilt",
            "base_too_low",
            "joint_actual_forbidden",
        ],
    }
    assert simultaneous["exact_hard_reason_counts"] == {
        "anchor_pos": 0,
        "anchor_ori": 0,
        "ee_body_pos": 0,
        "base_fell_tilt": 1,
        "base_too_low": 1,
        "joint_qdes_forbidden": 0,
        "joint_actual_forbidden": 1,
        "robot_hit_table": 0,
    }

    recovered = ledger.record_step(plant=_plant_row(), events=(), time_out=False)
    assert recovered["termination"]["exact_hard_terminated"] is True
    assert recovered["termination"]["exact_hard_reason"] == "base_fell_tilt"

    substep = vec_env.DiagnosticEventLedger(control_decimation=4).record_step(
        plant=_plant_row(joint_actual_forbidden_substep=True),
        events=(),
        time_out=False,
    )
    assert substep["termination"]["exact_hard_reason"] == (
        "joint_actual_forbidden"
    )


def test_event_ledger_joint_qdes_projection_semantics_order_and_latch_are_exact():
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    finite_projected = np.zeros(31, dtype=np.float64)
    finite_projected[4] = 1.0e100
    safe = ledger.record_step(
        plant=_plant_row(qdes_raw=finite_projected), events=(), time_out=False
    )
    assert safe["termination"]["exact_hard_terminated"] is False

    nonfinite = np.zeros(31, dtype=np.float64)
    nonfinite[4] = np.nan
    actual_boundary = np.zeros(31, dtype=np.float64)
    actual_boundary[4] = 1.0
    simultaneous = ledger.record_step(
        plant=_plant_row(qdes_raw=nonfinite, q=actual_boundary),
        events=(),
        time_out=False,
    )
    assert simultaneous["first_exact_hard_termination"] == {
        "policy_tick": 1,
        "sample_timing": "post_control_step",
        "physics_substep": None,
        "robot_hit_table_first_substep": None,
        "reason": "joint_qdes_forbidden",
        "all_reasons": [
            "joint_qdes_forbidden",
            "joint_actual_forbidden",
        ],
    }
    assert simultaneous["exact_hard_reason_counts"] == {
        "anchor_pos": 0,
        "anchor_ori": 0,
        "ee_body_pos": 0,
        "base_fell_tilt": 0,
        "base_too_low": 0,
        "joint_qdes_forbidden": 1,
        "joint_actual_forbidden": 1,
        "robot_hit_table": 0,
    }

    recovered = ledger.record_step(plant=_plant_row(), events=(), time_out=False)
    assert recovered["termination"]["exact_hard_terminated"] is True
    assert recovered["termination"]["exact_hard_reason"] == (
        "joint_qdes_forbidden"
    )


def test_event_ledger_joint_qdes_wrong_shape_fails_without_commit():
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    with pytest.raises(vec_env.VecEnvContractError, match="qdes_raw"):
        ledger.record_step(
            plant=_plant_row(qdes_raw=np.zeros(30, dtype=np.float64)),
            events=(),
            time_out=False,
        )
    assert ledger.policy_ticks == 0
    assert ledger.exact_hard_termination_latched is False


@pytest.mark.parametrize(
    "bad_limits",
    [
        np.zeros((30, 2), dtype=np.float64),
        np.tile(np.asarray([1.0, -1.0]), (31, 1)),
        np.tile(np.asarray([np.nan, 1.0]), (31, 1)),
    ],
)
def test_event_ledger_joint_actual_invalid_bounds_fail_closed_without_commit(
    bad_limits,
):
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    if bad_limits.shape == (31, 2):
        result = ledger.record_step(
            plant=_plant_row(joint_position_limits=bad_limits),
            events=(),
            time_out=False,
        )
        assert result["termination"]["exact_hard_reason"] == (
            "joint_actual_forbidden"
        )
        return
    with pytest.raises(vec_env.VecEnvContractError, match="shape"):
        ledger.record_step(
            plant=_plant_row(joint_position_limits=bad_limits),
            events=(),
            time_out=False,
        )
    assert ledger.policy_ticks == 0


@pytest.mark.parametrize("pelvis_up_world_z", [True, 1.0000001, -1.0000001])
def test_event_ledger_rejects_malformed_pelvis_up_sample_without_commit(
    pelvis_up_world_z,
):
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    with pytest.raises(vec_env.VecEnvContractError, match="pelvis"):
        ledger.record_step(
            plant=_plant_row(pelvis_up_world_z=pelvis_up_world_z),
            events=(),
            time_out=False,
        )
    assert ledger.policy_ticks == 0
    assert ledger.exact_hard_termination_latched is False


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            (
                {
                    "policy_tick": 1,
                    "physics_substep": 0,
                    "time_s": 0.005,
                    "event": "racket",
                },
            ),
            "policy tick differs",
        ),
        (
            (
                {
                    "policy_tick": 0,
                    "physics_substep": 4,
                    "time_s": 0.005,
                    "event": "racket",
                },
            ),
            "index is out of range",
        ),
        (
            (
                {
                    "policy_tick": 0,
                    "physics_substep": 0,
                    "time_s": 0.005,
                    "event": "wall",
                },
            ),
            "label is unsupported",
        ),
    ],
)
def test_event_ledger_rejects_malformed_substep_evidence_without_commit(
    events, message
):
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    with pytest.raises(vec_env.VecEnvContractError, match=message):
        ledger.record_step(plant=_plant_row(), events=events, time_out=False)
    assert ledger.policy_ticks == 0
    assert ledger.physics_substeps == 0
    assert not any(ledger.contact_edge_counts.values())


def test_torch_vecenv_n8_reset_rollout_is_deterministic_and_no_reward():
    torch = pytest.importorskip("torch")
    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((5, 31), dtype=np.float64),
        source_sha256="c" * 64,
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    cores = tuple(_FakeCore(index) for index in range(8))
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=cores,
        robot_tape=tape,
        questions=(question,) * 8,
    )
    # The rsl_rl runner fetches observations from the fresh instance without
    # calling reset first.
    initial, _ = env.get_observations()
    assert initial.shape == (8, 76)
    obs, extras = env.reset()
    assert obs.shape == (8, 76)
    assert extras["observations"]["critic"].shape == (8, 76)
    step = env.diagnostic_step(torch.zeros((8, 31)))
    assert step.observations.shape == (8, 76)
    assert step.exact_phase_fidelity_runtime_available is False
    assert step.per_env_phase_fidelity_samples == (None,) * 8
    assert not step.time_outs.any()
    assert env.episode_length_buf.tolist() == [1] * 8
    assert len(step.per_env_ledgers) == 8
    assert all(row["policy_ticks"] == 1 for row in step.per_env_ledgers)
    assert all(
        row["termination"]["formal_hard_terminated"] is None
        for row in step.per_env_ledgers
    )
    assert not step.exact_hard_terminations.any()
    assert step.exact_hard_termination_reasons == (None,) * 8
    assert not step.episode_dones.any()
    assert step.episode_done_reasons == (None,) * 8
    assert not step.terminal_observation_mask.any()
    assert step.reset_env_ids == ()
    np.testing.assert_array_equal(
        step.terminal_observations.numpy(), step.observations.numpy()
    )

    actions = torch.zeros((3, 8, 31))
    trace_a, receipt_a = env.run_diagnostic_rollout(actions)
    trace_b, receipt_b = env.run_diagnostic_rollout(actions)
    np.testing.assert_array_equal(trace_a, trace_b)
    assert receipt_a["trace_and_event_sha256"] == receipt_b[
        "trace_and_event_sha256"
    ]
    assert receipt_a["status"] == "DIAGNOSTIC_NO_REWARD_ROLLOUT_COMPLETE"
    assert receipt_a["reward_blocker"]["reward_available"] is False
    assert receipt_a["termination_blocker"]["formal_termination_available"] is False
    assert len(receipt_a["event_ledger_transcript"]) == 3
    assert all(
        row["termination"]["formal_hard_terminated"] is None
        for row in receipt_a["final_event_ledgers"]
    )
    before = [core.ticks for core in cores]
    with pytest.raises(vec_env.RewardContractMissing):
        env.step(torch.zeros((8, 31)))
    assert [core.ticks for core in cores] == before


def test_native_physical_facts_reach_transition_and_rollout_but_not_reward():
    torch = pytest.importorskip("torch")
    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((3, 31), dtype=np.float64),
        source_sha256="c" * 64,
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    cores = (_NativeEventCore(0), _NativeEventCore(1))
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=cores,
        robot_tape=tape,
        questions=(question, question),
    )
    _observations, extras = env.get_observations()
    contract = extras["native_physical_event_contract"]
    assert contract["runtime_available"] is True
    assert contract["reward_authorized"] is False
    transition = env.diagnostic_step(torch.zeros((2, 31)))
    assert transition.native_physical_event_runtime_available is True
    assert [
        row["policy_tick"]
        for row in transition.per_env_native_physical_event_facts
    ] == [0, 0]
    before = [core.ticks for core in cores]
    with pytest.raises(vec_env.RewardContractMissing, match="before physics"):
        env.step(torch.zeros((2, 31)))
    assert [core.ticks for core in cores] == before

    _trace, receipt = env.run_diagnostic_rollout(
        torch.zeros((2, 2, 31))
    )
    assert receipt["kind"] == "a3_mujoco_n1_diagnostic_vecenv_rollout_v4"
    assert receipt["semantic"][
        "native_physical_event_runtime_available"
    ] is True
    assert len(receipt["native_physical_event_transcript"]) == 2
    assert receipt["reward_blocker"]["reward_available"] is False


def test_native_physical_fact_omission_invalidates_whole_vecenv_batch():
    torch = pytest.importorskip("torch")

    class _OmittingNativeEventCore(_NativeEventCore):
        def step(self, action):
            result = super().step(action)
            result.pop("native_physical_event_facts")
            return result

    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((3, 31), dtype=np.float64),
        source_sha256="c" * 64,
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=(_NativeEventCore(0), _OmittingNativeEventCore(1)),
        robot_tape=tape,
        questions=(question, question),
    )
    with pytest.raises(vec_env.VecEnvContractError, match="omitted its facts"):
        env.diagnostic_step(torch.zeros((2, 31)))
    assert env._has_reset is False
    assert all(ledger.policy_ticks == 0 for ledger in env._event_ledgers)


def test_phase_fidelity_runtime_sample_drives_only_matching_env_compact_reset():
    torch = pytest.importorskip("torch")

    class _PhaseCore(_FakeCore):
        phase_fidelity_sample_contract_sha256 = (
            vec_env.phase_fidelity_sample_contract()["content_sha256"]
        )

        def __init__(self, index):
            super().__init__(index)
            self.emitted_failure = False

        def step(self, action):
            row = super().step(action)
            above = np.nextafter(
                vec_env.PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD,
                np.inf,
            )
            if self.index == 0 and not self.emitted_failure:
                self.emitted_failure = True
                row["phase_fidelity_sample"] = _phase_sample(
                    anchor_projected_gravity_z_error_abs=above
                )
            elif self.index == 1:
                row["phase_fidelity_sample"] = _phase_sample(
                    motion_phase_context="recovery_hold",
                    in_hold=True,
                    anchor_pos_z_error_m=10.0,
                    anchor_projected_gravity_z_error_abs=10.0,
                    ee_body_pos_z_error_m=[10.0] * 4,
                )
            else:
                row["phase_fidelity_sample"] = _phase_sample()
            return row

    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((5, 31), dtype=np.float64),
        source_sha256="c" * 64,
    )
    questions = tuple(
        SimpleNamespace(
            scene_binding_sha256="b" * 64,
            source_sha256=character * 64,
        )
        for character in ("d", "e")
    )
    cores = (_PhaseCore(0), _PhaseCore(1))
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=cores, robot_tape=tape, questions=questions
    )
    assert env.exact_phase_fidelity_runtime_available is True

    first = env.diagnostic_step(torch.zeros((2, 31)))
    assert first.exact_phase_fidelity_runtime_available is True
    assert first.episode_dones.tolist() == [True, False]
    assert first.episode_done_reasons == ("anchor_ori", None)
    assert first.reset_env_ids == (0,)
    assert first.per_env_ledgers[0]["phase_fidelity"]["exact_sample_count"] == 1
    assert first.per_env_ledgers[1]["phase_fidelity"]["exact_sample_count"] == 1
    assert first.per_env_ledgers[0]["termination"][
        "formal_hard_termination_available"
    ] is True
    assert first.per_env_ledgers[0]["termination"][
        "formal_hard_terminated"
    ] is True
    assert first.per_env_phase_fidelity_samples[0][
        "anchor_projected_gravity_z_error_abs"
    ] > vec_env.PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD
    assert first.per_env_phase_fidelity_samples[1]["in_hold"] is True
    assert env.episode_length_buf.tolist() == [0, 1]

    second = env.diagnostic_step(torch.zeros((2, 31)))
    assert second.episode_dones.tolist() == [False, False]
    assert second.exact_hard_termination_reasons == (None, None)
    assert second.per_env_ledgers[0]["phase_fidelity"]["exact_sample_count"] == 1
    assert second.per_env_ledgers[1]["phase_fidelity"]["exact_sample_count"] == 2
    assert env.episode_length_buf.tolist() == [1, 2]


def test_phase_fidelity_advertised_core_missing_sample_invalidates_vecenv():
    torch = pytest.importorskip("torch")

    class _MissingPhaseSampleCore(_FakeCore):
        phase_fidelity_sample_contract_sha256 = (
            vec_env.phase_fidelity_sample_contract()["content_sha256"]
        )

    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((2, 31), dtype=np.float64),
        source_sha256="c" * 64,
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=(_MissingPhaseSampleCore(0),),
        robot_tape=tape,
        questions=(question,),
    )
    with pytest.raises(vec_env.VecEnvContractError, match="omitted its sample"):
        env.diagnostic_step(torch.zeros((1, 31)))
    assert env._has_reset is False


def test_real_mujoco_n1_vecenv_finite_rollout_and_reset(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    binding = single_env.load_plant_binding(CONTRACT)
    core = n1.MujocoN1BallCore(binding)
    robot_payload = single_env.build_probe_tape(binding, delay_steps=0)
    robot_path = tmp_path / "robot.json"
    single_env.write_fixed_tape(robot_path, robot_payload)
    robot_tape = single_env.load_fixed_tape(robot_path, binding)
    payload = n1.build_question_payload(
        question_id="vecenv_flight_probe",
        scene_binding_sha256=core.scene_binding_sha256,
        birth_position_w_m=(2.3, 0.0, 1.5),
        birth_linear_velocity_w_mps=(-1.0, 0.0, 0.0),
        landing_aim_xy_w_m=(2.3, 0.0),
        nominal_time_to_contact_s=0.5,
    )
    question_path = tmp_path / "question.json"
    n1.write_question(question_path, payload)
    question = n1.load_question(
        question_path,
        expected_file_sha256=hashlib.sha256(question_path.read_bytes()).hexdigest(),
        scene_binding_sha256=core.scene_binding_sha256,
    )
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=(core,), robot_tape=robot_tape, questions=(question,)
    )
    actions = torch.zeros((3, 1, 31))
    trace_a, receipt_a = env.run_diagnostic_rollout(actions)
    trace_b, receipt_b = env.run_diagnostic_rollout(actions)
    np.testing.assert_array_equal(trace_a, trace_b)
    assert trace_a.shape == (4, 1, 76)
    assert receipt_a["trace_and_event_sha256"] == receipt_b[
        "trace_and_event_sha256"
    ]
    assert receipt_a["reward_blocker"]["reward_available"] is False
    assert receipt_a["termination_blocker"]["formal_termination_available"] is False
    assert receipt_a["final_event_ledgers"][0]["physics_substeps"] == 12
    assert receipt_a["final_event_ledgers"][0]["reward_paid"] is False
    sim_time = float(core.data.time)
    with pytest.raises(vec_env.RewardContractMissing):
        env.step(torch.zeros((1, 31)))
    assert float(core.data.time) == sim_time


def test_vecenv_per_env_compact_reset_preserves_terminal_observation_and_batch():
    torch = pytest.importorskip("torch")

    class _SelectiveFallenCore(_FakeCore):
        def step(self, action):
            row = super().step(action)
            if self.index == 0:
                row["plant"] = _plant_row(pelvis_height_m=0.49)
            return row

    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((2, 31), dtype=np.float64),
        source_sha256="c" * 64,
    )
    question_a = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    question_b = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="e" * 64,
    )
    cores = (_SelectiveFallenCore(0), _SelectiveFallenCore(1))
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=cores, robot_tape=tape, questions=(question_a, question_b)
    )
    first = env.diagnostic_step(torch.zeros((2, 31)))
    assert first.exact_hard_terminations.tolist() == [True, False]
    assert first.exact_hard_termination_reasons == ("base_too_low", None)
    assert first.time_outs.tolist() == [False, False]
    assert first.episode_dones.tolist() == [True, False]
    assert first.episode_done_reasons == ("base_too_low", None)
    assert first.terminal_observation_mask.tolist() == [True, False]
    assert first.reset_env_ids == (0,)
    np.testing.assert_array_equal(
        first.terminal_observations[0].numpy(),
        vec_env.flatten_observation_groups(_groups(1.0)).astype(np.float32),
    )
    np.testing.assert_array_equal(
        first.observations[0].numpy(),
        vec_env.flatten_observation_groups(_groups(0.0)).astype(np.float32),
    )
    np.testing.assert_array_equal(
        first.observations[1].numpy(),
        vec_env.flatten_observation_groups(_groups(2.0)).astype(np.float32),
    )
    assert env.episode_length_buf.tolist() == [0, 1]
    assert env._exact_hard_terminated_buf.tolist() == [False, False]
    assert [core.ticks for core in cores] == [0, 1]
    assert [ledger.policy_ticks for ledger in env._event_ledgers] == [0, 1]

    # Row 0 terminates again while row 1 independently reaches its timeout;
    # both reset without freezing either row or requiring a whole-batch reset.
    second = env.diagnostic_step(torch.zeros((2, 31)))
    assert second.exact_hard_terminations.tolist() == [True, False]
    assert second.time_outs.tolist() == [False, True]
    assert second.episode_dones.tolist() == [True, True]
    assert second.episode_done_reasons == ("base_too_low", "time_out")
    assert second.reset_env_ids == (0, 1)
    assert env.episode_length_buf.tolist() == [0, 0]
    assert [core.ticks for core in cores] == [0, 0]

    third = env.diagnostic_step(torch.zeros((2, 31)))
    assert third.episode_dones.tolist() == [True, False]
    assert third.episode_done_reasons == ("base_too_low", None)
    assert third.reset_env_ids == (0,)
    assert env.episode_length_buf.tolist() == [0, 1]

    actions = torch.zeros((3, 2, 31))
    trace_a, receipt_a = env.run_diagnostic_rollout(actions)
    trace_b, receipt_b = env.run_diagnostic_rollout(actions)
    np.testing.assert_array_equal(trace_a, trace_b)
    assert receipt_a["trace_and_event_sha256"] == receipt_b[
        "trace_and_event_sha256"
    ]
    assert receipt_a["kind"] == "a3_mujoco_n1_diagnostic_vecenv_rollout_v4"
    terminal_descriptor = receipt_a["terminal_observation_trace"]
    assert terminal_descriptor["storage"] == "digest_only_not_returned"
    assert len(terminal_descriptor["sha256"]) == 64
    assert receipt_a["termination_transcript"][0]["episode_dones"] == [True, False]
    assert receipt_a["termination_transcript"][1]["time_outs"] == [False, True]
    assert receipt_a["question_source_sha256_by_env"] == ["d" * 64, "e" * 64]
    assert receipt_a["semantic"]["question_source_sha256_by_env"] == [
        "d" * 64,
        "e" * 64,
    ]
    assert receipt_a["returned_observation_semantics"] == (
        "post_compact_reset_next_observation"
    )

    # Independent verifier: receipt + returned trace are sufficient to
    # recompute the aggregate digest even though terminal observations are
    # deliberately represented by a digest-only descriptor.
    recomputed = hashlib.sha256()
    recomputed.update(
        json.dumps(
            receipt_a["semantic"], sort_keys=True, separators=(",", ":")
        ).encode()
    )
    recomputed.update(np.ascontiguousarray(trace_a, dtype="<f8").tobytes())
    for key in (
        "terminal_observation_trace",
        "event_transcript",
        "native_physical_event_transcript",
        "event_ledger_transcript",
        "termination_transcript",
    ):
        recomputed.update(
            json.dumps(
                receipt_a[key], sort_keys=True, separators=(",", ":")
            ).encode()
        )
    assert recomputed.hexdigest() == receipt_a["trace_and_event_sha256"]

    alternate_question_b = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="f" * 64,
    )
    alternate_env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=(_SelectiveFallenCore(0), _SelectiveFallenCore(1)),
        robot_tape=tape,
        questions=(question_a, alternate_question_b),
    )
    alternate_trace, alternate_receipt = alternate_env.run_diagnostic_rollout(actions)
    np.testing.assert_array_equal(trace_a, alternate_trace)
    assert alternate_receipt["question_source_sha256_by_env"] == [
        "d" * 64,
        "f" * 64,
    ]
    assert alternate_receipt["trace_and_event_sha256"] != receipt_a[
        "trace_and_event_sha256"
    ]


def test_compact_reset_failure_invalidates_vecenv_until_full_reset():
    torch = pytest.importorskip("torch")

    class _ResetFailingFallenCore(_FakeCore):
        fail_compact_reset = False

        def reset(self, *, robot_tape, question):
            if self.fail_compact_reset:
                raise RuntimeError("synthetic reset failure")
            return super().reset(robot_tape=robot_tape, question=question)

        def step(self, action):
            row = super().step(action)
            row["plant"] = _plant_row(pelvis_height_m=0.49)
            return row

    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((2, 31), dtype=np.float64),
        source_sha256="c" * 64,
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    core = _ResetFailingFallenCore(0)
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=(core,), robot_tape=tape, questions=(question,)
    )
    core.fail_compact_reset = True
    with pytest.raises(vec_env.VecEnvContractError, match="compact reset failed"):
        env.diagnostic_step(torch.zeros((1, 31)))
    assert env._has_reset is False
    with pytest.raises(vec_env.VecEnvContractError, match="must be reset"):
        env.diagnostic_step(torch.zeros((1, 31)))
    core.fail_compact_reset = False
    observations, _ = env.reset()
    assert observations.shape == (1, 76)


def test_robot_table_terminal_snapshot_resets_only_hit_env_and_clears_new_latch():
    torch = pytest.importorskip("torch")

    class _TableOnceWithSurvivorContactCore(_FakeCore):
        emitted_table_guard = False

        def step(self, action):
            row = super().step(action)
            if self.index == 0 and not self.emitted_table_guard:
                self.emitted_table_guard = True
                row["plant"] = _plant_row(
                    robot_hit_table_substep=True,
                    robot_hit_table_first_substep=2,
                )
            elif self.index == 1:
                row["plant"] = _plant_row(
                    table_contact_pairs=1,
                    table_contact_substeps=1,
                    first_table_contact_pair="survivor_robot~table",
                )
            return row

    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((5, 31), dtype=np.float64),
        source_sha256="c" * 64,
    )
    questions = (
        SimpleNamespace(
            scene_binding_sha256="b" * 64,
            source_sha256="d" * 64,
        ),
        SimpleNamespace(
            scene_binding_sha256="b" * 64,
            source_sha256="e" * 64,
        ),
    )
    cores = (
        _TableOnceWithSurvivorContactCore(0),
        _TableOnceWithSurvivorContactCore(1),
    )
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=cores, robot_tape=tape, questions=questions
    )
    first = env.diagnostic_step(torch.zeros((2, 31)))
    assert first.episode_dones.tolist() == [True, False]
    assert first.exact_hard_termination_reasons == ("robot_hit_table", None)
    assert first.reset_env_ids == (0,)
    assert first.per_env_ledgers[0]["first_exact_hard_termination"] == {
        "policy_tick": 0,
        "sample_timing": "physics_substep",
        "physics_substep": 2,
        "robot_hit_table_first_substep": 2,
        "reason": "robot_hit_table",
        "all_reasons": ["robot_hit_table"],
    }
    assert env.episode_length_buf.tolist() == [0, 1]
    assert [ledger.policy_ticks for ledger in env._event_ledgers] == [0, 1]

    # Caller mutation must not leak into either the reset row or the survivor's
    # still-live episode ledger.
    first.per_env_ledgers[0]["first_exact_hard_termination"][
        "all_reasons"
    ].append("caller_corruption")
    first.per_env_ledgers[1]["first_robot_obstacle_contact"][
        "pair"
    ] = "caller_corruption"

    second = env.diagnostic_step(torch.zeros((2, 31)))
    assert second.episode_dones.tolist() == [False, False]
    assert second.exact_hard_termination_reasons == (None, None)
    assert second.reset_env_ids == ()
    assert second.per_env_ledgers[0]["first_exact_hard_termination"] is None
    assert second.per_env_ledgers[0]["latches"]["robot_table_keepout_seen"] is False
    assert second.per_env_ledgers[1]["first_robot_obstacle_contact"] == {
        "policy_tick": 0,
        "pair": "survivor_robot~table",
    }
    assert env.episode_length_buf.tolist() == [1, 2]
