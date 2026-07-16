"""Dependency-light regression tests for the schema-3 training/export contract."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
)
SPEC = importlib.util.spec_from_file_location("training_contract_under_test", MODULE_PATH)
TC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TC)


@dataclass(frozen=True)
class _Term:
    name: str
    dim: int


@dataclass(frozen=True)
class _ActorContract:
    name: str = "test_actor"
    obs_mode: str = "test_mode"
    total_dim: int = 3
    terms: tuple = (_Term("foo", 2), _Term("bar", 1))


class _PolicyCfg:
    history_length = None

    @staticmethod
    def to_dict():
        return {"foo": {"history_length": 0}, "bar": {"history_length": 2}}


class _Manager:
    def __init__(self, terms):
        self._terms = terms

    def get_term(self, name):
        return self._terms[name]


def _env(*, joints=3, num_envs=2, action_ids=slice(None), policy_dt=0.02):
    joint_names = [f"j{i}" for i in range(joints)]
    body_names = ["pelvis", "torso", "wrist", "unused"]
    data = SimpleNamespace(
        joint_names=joint_names,
        default_joint_pos_nominal=np.linspace(-0.1, 0.1, joints),
        default_joint_stiffness=np.tile(np.linspace(10.0, 20.0, joints), (num_envs, 1)),
        default_joint_damping=np.tile(np.linspace(1.0, 2.0, joints), (num_envs, 1)),
        default_joint_armature=np.tile(
            np.linspace(0.001, 0.01, joints), (num_envs, 1)
        ),
        default_joint_friction_coeff=np.tile(
            np.linspace(0.1, 0.3, joints), (num_envs, 1)
        ),
        joint_vel_limits=np.tile(np.linspace(8.0, 12.0, joints), (num_envs, 1)),
        joint_effort_limits=np.tile(np.linspace(30.0, 60.0, joints), (num_envs, 1)),
        soft_joint_pos_limits=np.tile(
            np.stack((-np.ones(joints), np.ones(joints)), axis=1)[None, :, :],
            (num_envs, 1, 1),
        ),
    )
    robot = SimpleNamespace(
        data=data,
        body_names=body_names,
        actuators={
            "all": SimpleNamespace(
                joint_indices=slice(None), is_implicit_model=True
            )
        },
    )
    action_cfg = SimpleNamespace(use_default_offset=True)
    action = SimpleNamespace(
        _joint_ids=action_ids,
        _scale=np.tile(np.linspace(0.1, 0.3, joints), (num_envs, 1)),
        _clamp_enabled=True,
        cfg=action_cfg,
    )
    motion_cfg = SimpleNamespace(
        body_names=["pelvis", "torso", "wrist"], anchor_body_name="torso"
    )
    motion = SimpleNamespace(
        cfg=motion_cfg,
        body_indexes=np.array([0, 1, 2]),
        motion=SimpleNamespace(
            seg_len=np.array([11, 13]),
            per_clip_fps=(50.0, 50.0),
            kinematics_contracts=[
                {
                    "schema_version": 2, "body_pos_point": "link_origin",
                    "body_lin_vel_point": "center_of_mass", "exact": True,
                    "body_names": list(body_names), "status": "declared_v2",
                },
                {
                    "schema_version": 2, "body_pos_point": "link_origin",
                    "body_lin_vel_point": "center_of_mass", "exact": True,
                    "body_names": list(body_names), "status": "declared_v2",
                },
            ],
            kinematics_contract_exact=True,
        ),
    )
    obs = SimpleNamespace(
        active_terms={"policy": ["foo", "bar"]},
        group_obs_term_dim={"policy": [(2,), (1,)]},
        group_obs_dim={"policy": (3,)},
        cfg=SimpleNamespace(policy=_PolicyCfg()),
    )
    joint_pos_cfg = SimpleNamespace(use_default_offset=True, clamp=True)
    cfg = SimpleNamespace(
        decimation=4,
        actions=SimpleNamespace(joint_pos=joint_pos_cfg),
    )
    return SimpleNamespace(
        scene={"robot": robot},
        action_manager=_Manager({"joint_pos": action}),
        command_manager=_Manager({"motion": motion}),
        observation_manager=obs,
        cfg=cfg,
        physics_dt=0.005,
        step_dt=policy_dt,
    )


def _command_env(command_func):
    env = _env()
    env.observation_manager.active_terms = {"policy": ["command"]}
    env.observation_manager.group_obs_term_dim = {"policy": [(62,)]}
    env.observation_manager.group_obs_dim = {"policy": (62,)}
    env.observation_manager.cfg.policy = SimpleNamespace(
        history_length=None,
        to_dict=lambda: {"command": {"history_length": 0}},
    )
    env.cfg.observations = SimpleNamespace(
        policy=SimpleNamespace(command=SimpleNamespace(func=command_func))
    )
    actor = SimpleNamespace(
        name="deploy_parity",
        obs_mode="deploy_parity",
        total_dim=62,
        terms=(_Term("command", 62),),
    )
    return env, actor


def test_actor_leg_ref_mask_fact_uses_exact_callable_identity(monkeypatch):
    import functools

    def canonical_unmasked(env, command_name):
        raise NotImplementedError

    def canonical_masked(env, command_name):
        raise NotImplementedError

    monkeypatch.setattr(
        TC,
        "_canonical_actor_leg_ref_mask_callables",
        lambda: ((canonical_unmasked, False), (canonical_masked, True)),
    )
    env, actor = _command_env(canonical_unmasked)
    facts = TC.runtime_execution_facts(env, actor)
    assert "actor_leg_ref_mask" not in facts
    assert facts["actor_leg_ref_mask_provenance_epoch"] == 1

    env.cfg.observations.policy.command.func = functools.partial(canonical_masked)
    facts = TC.runtime_execution_facts(env, actor)
    assert facts["actor_leg_ref_mask"] is True
    assert tuple(
        key
        for key in facts
        if key not in ("actor_leg_ref_mask_provenance_epoch", "actor_leg_ref_mask")
    ) == TC.RUNTIME_EXECUTION_KEYS

    # A partial carrying any bound argument is a different configured observation term.  Looking
    # only at ``.func`` would launder it as the canonical callable and mint a false epoch-1 fact.
    for nonempty_partial in (
        functools.partial(canonical_unmasked, object()),
        functools.partial(canonical_unmasked, command_name="different_command"),
        functools.partial(functools.partial(canonical_masked), command_name="different_command"),
    ):
        env.cfg.observations.policy.command.func = nonempty_partial
        with pytest.raises(RuntimeError, match="bound args/kwargs"):
            TC.runtime_execution_facts(env, actor)

    # functools.wraps copies marker-like attributes and names, but the wrapper is not authority.
    @functools.wraps(canonical_masked)
    def renamed_masked_command(env, command_name):
        raise NotImplementedError

    renamed_masked_command.actor_leg_ref_mask = True
    renamed_masked_command.actor_leg_ref_mask_provenance_epoch = 1
    env.cfg.observations.policy.command.func = renamed_masked_command
    with pytest.raises(RuntimeError, match="not one of the two canonical"):
        TC.runtime_execution_facts(env, actor)
    legacy = TC.runtime_execution_facts(
        env, actor, allow_legacy_actor_leg_ref_mask_ambiguity=True
    )
    assert "actor_leg_ref_mask_provenance_epoch" not in legacy
    assert "actor_leg_ref_mask" not in legacy


def test_actor_leg_ref_mask_rejects_partial_subclass_with_overridden_call(monkeypatch):
    import functools

    def canonical_unmasked(env, command_name):
        raise NotImplementedError

    def canonical_masked(env, command_name):
        raise NotImplementedError

    class SemanticOverridePartial(functools.partial):
        def __call__(self, env, command_name):
            return "different-command-semantics"

    monkeypatch.setattr(
        TC,
        "_canonical_actor_leg_ref_mask_callables",
        lambda: ((canonical_unmasked, False), (canonical_masked, True)),
    )
    disguised = SemanticOverridePartial(canonical_unmasked)
    assert disguised.args == () and disguised.keywords == {}
    assert disguised(None, "motion") == "different-command-semantics"
    env, actor = _command_env(disguised)
    with pytest.raises(RuntimeError, match="not one of the two canonical"):
        TC.runtime_execution_facts(env, actor)


@pytest.mark.parametrize("donor_value", [None, "0", "1"])
def test_actor_leg_ref_mask_metadata_is_checkpoint_authoritative_and_only_when_true(donor_value):
    metadata = {
        "keep": "value",
        "training_contract_exact": "1",
        "training_contract_sha256": "b" * 64,
        "source_checkpoint_sha256": "a" * 64,
    }
    if donor_value is not None:
        metadata["actor_leg_ref_mask"] = donor_value

    masked_contract = {
        "actor_obs_term_names": ["command"],
        "actor_obs_term_dims": [62],
        "actor_leg_ref_mask_provenance_epoch": 1,
        "actor_leg_ref_mask": True,
    }
    TC.bind_actor_leg_ref_mask_metadata(metadata, masked_contract)
    assert metadata["actor_leg_ref_mask_provenance_epoch"] == "1"
    assert metadata["actor_leg_ref_mask"] == "1"
    assert metadata["training_contract_exact"] == "0"
    assert metadata["actor_leg_ref_mask_provenance_sha256"] == (
        TC.actor_leg_ref_mask_provenance_sha256(
            training_contract_sha256="b" * 64,
            source_checkpoint_sha256="a" * 64,
            masked=True,
        )
    )

    # Unmasked and legacy checkpoints clear any stale donor value. False is deliberately treated
    # as absence here; the schema validator separately rejects serializing False in a contract.
    unmasked_contract = dict(masked_contract)
    unmasked_contract.pop("actor_leg_ref_mask")
    metadata["training_contract_exact"] = "1"
    TC.bind_actor_leg_ref_mask_metadata(metadata, unmasked_contract)
    assert "actor_leg_ref_mask" not in metadata
    assert metadata["actor_leg_ref_mask_provenance_epoch"] == "1"
    assert metadata["training_contract_exact"] == "1"
    assert metadata["actor_leg_ref_mask_provenance_sha256"] == (
        TC.actor_leg_ref_mask_provenance_sha256(
            training_contract_sha256="b" * 64,
            source_checkpoint_sha256="a" * 64,
            masked=False,
        )
    )
    TC.bind_actor_leg_ref_mask_metadata(metadata, {})
    assert "actor_leg_ref_mask" not in metadata
    assert "actor_leg_ref_mask_provenance_epoch" not in metadata
    assert "actor_leg_ref_mask_provenance_sha256" not in metadata
    metadata["actor_leg_ref_mask"] = "1"
    TC.bind_actor_leg_ref_mask_metadata(metadata, None)
    assert "actor_leg_ref_mask" not in metadata


def test_runtime_contract_extracts_actual_execution_values():
    facts = TC.runtime_execution_facts(_env(), _ActorContract())
    assert tuple(facts) == TC.RUNTIME_EXECUTION_KEYS
    assert facts["action_joint_ids"] == [0, 1, 2]
    assert facts["joint_names"] == ["j0", "j1", "j2"]
    assert facts["default_joint_pos"] == pytest.approx([-0.1, 0.0, 0.1])
    assert facts["joint_actuator_types"] == ["implicit"] * 3
    assert facts["joint_armature"] == pytest.approx([0.001, 0.0055, 0.01])
    assert facts["joint_friction_coefficients"] == pytest.approx([0.1, 0.2, 0.3])
    assert facts["joint_velocity_limits"] == pytest.approx([8.0, 10.0, 12.0])
    assert facts["joint_friction_backend"] == "physx"
    assert facts["joint_friction_semantics"] == "load_dependent_spatial_force_coefficient"
    assert facts["joint_friction_units"] == "dimensionless"
    assert facts["qdes_joint_pos_limits"] == [[-1.0, 1.0]] * 3
    assert facts["physics_step_dt_s"] == 0.005
    assert facts["policy_step_dt_s"] == 0.02
    assert facts["control_decimation"] == 4
    assert facts["actor_obs_term_names"] == ["foo", "bar"]
    assert facts["actor_obs_term_dims"] == [2, 1]
    assert facts["observation_history_lengths"] == [1, 2]
    assert facts["articulation_body_names"] == ["pelvis", "torso", "wrist", "unused"]
    assert facts["body_names"] == ["pelvis", "torso", "wrist"]
    assert facts["anchor_body_index"] == 1
    assert facts["motion_segment_lengths"] == [11, 13]
    assert facts["motion_clip_fps"] == [50.0, 50.0]


def test_motion_fps_and_body_order_are_formal_runtime_guards():
    bad_fps = _env()
    bad_fps.command_manager.get_term("motion").motion.per_clip_fps = (50.0, 49.0)
    with pytest.raises(RuntimeError, match="must all equal policy rate"):
        TC.runtime_execution_facts(bad_fps, _ActorContract())

    bad_order = _env()
    bad_order.command_manager.get_term("motion").motion.kinematics_contracts[1][
        "body_names"
    ] = ["torso", "pelvis", "wrist", "unused"]
    contract = {
        "schema_version": 3,
        **TC.runtime_execution_facts(bad_order, _ActorContract()),
        "racket_control_point": "pingpang_red_Link_origin_v1",
        "racket_control_point_offset_wrist_m": [0.21021, 0.032078, 0.032036],
    }
    with pytest.raises(ValueError, match="body_names do not match the runtime articulation"):
        TC.validate_schema3_contract(contract)


def test_soft_limit_env0_selection_uses_rank_not_num_envs_vs_num_joints():
    # Regression: [31 envs, 31 joints, 2] used to be mistaken for [31 joints, 2].
    facts = TC.runtime_execution_facts(
        _env(joints=31, num_envs=31),
        SimpleNamespace(
            name="wide", obs_mode="wide", total_dim=3,
            terms=(_Term("foo", 2), _Term("bar", 1)),
        ),
    )
    assert len(facts["qdes_joint_pos_limits"]) == 31
    assert facts["qdes_joint_pos_limits"][30] == [-1.0, 1.0]


def test_nonidentity_action_order_is_rejected():
    with pytest.raises(RuntimeError, match="identity action/articulation order"):
        TC.runtime_execution_facts(_env(action_ids=np.array([1, 0, 2])), _ActorContract())


def test_actuator_integration_must_cover_each_joint_exactly():
    missing = _env()
    missing.scene["robot"].actuators["all"].joint_indices = np.array([0, 1])
    with pytest.raises(RuntimeError, match="unresolved for joints"):
        TC.runtime_execution_facts(missing, _ActorContract())

    overlap = _env()
    overlap.scene["robot"].actuators = {
        "a": SimpleNamespace(joint_indices=np.array([0, 1]), is_implicit_model=True),
        "b": SimpleNamespace(joint_indices=np.array([1, 2]), is_implicit_model=False),
    }
    with pytest.raises(RuntimeError, match="multiple actuator groups"):
        TC.runtime_execution_facts(overlap, _ActorContract())


def test_timing_and_actor_layout_are_runtime_guards():
    with pytest.raises(RuntimeError, match="policy dt"):
        TC.runtime_execution_facts(_env(policy_dt=0.021), _ActorContract())
    bad_actor = SimpleNamespace(
        name="bad", obs_mode="bad", total_dim=3,
        terms=(_Term("bar", 1), _Term("foo", 2)),
    )
    with pytest.raises(RuntimeError, match="does not match"):
        TC.runtime_execution_facts(_env(), bad_actor)


def _schema3_contract():
    return {
        "schema_version": 3,
        **TC.runtime_execution_facts(_env(), _ActorContract()),
        "racket_control_point": "pingpang_red_Link_origin_v1",
        "racket_control_point_offset_wrist_m": [0.21021, 0.032078, 0.032036],
    }


def _planner_revision_schema3_contract():
    contract = _schema3_contract()
    contract["strike_phase_per_clip"] = [0.4, 0.6]
    profile = {
        "policy_dt_s": 0.02,
        "min_tts_s": 0.1,
        "max_tts_s": 2.0,
        "max_phase_rate_per_s": 4.0,
        "max_phase_acceleration_per_s2": 20.0,
        "max_deadline_revision_delta_s": 0.25,
        "max_position_revision_delta_m": 0.1,
        "max_velocity_revision_delta_mps": 0.5,
        "max_normal_revision_delta_rad": 0.2,
        "normal_unit_tolerance": 0.0001,
        "early_deadline_tolerance_s": 1e-9,
        "contract_version": "phase_governor_v1",
        "schema_version": 1,
    }
    profile_bytes = (
        json.dumps(profile, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    contract["planner_task_revision"] = {
        "enabled": True,
        "revision_schema_version": 1,
        "governor": {
            "contract_version": "phase_governor_v1",
            "schema_version": 1,
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "profile": profile,
        },
        "initial_tts_range_s": [0.25, 1.7],
    }
    contract["planner_task_revision_training"] = {
        "initial_tts_sampling_semantics": (
            "explicit_weighted_mixture_over_initial_tts_range_s"
        ),
        "initial_tts_mixture": {
            "contract_version": "initial_tts_mixture_v1",
            "components": [
                {"name": "late_stress", "range_s": [0.25, 0.49], "weight": 0.15},
                {"name": "baseline_0p5", "range_s": [0.5, 0.5], "weight": 0.20},
                {"name": "fast_deploy", "range_s": [0.5, 0.9], "weight": 0.30},
                {"name": "broad_arrival", "range_s": [0.9, 1.7], "weight": 0.35},
            ],
        },
        "initial_feasibility_gate": (
            "normalized_phase_rate_and_acceleration_envelope_only"
        ),
        "dynamics_certified_action_tau_min_bound": False,
        "timing_exam_semantics": {
            "0.5_s": "required_baseline_gate",
            "below_0.5_s": "stress_diagnostic_not_support_floor",
        },
        "position_std_m": 0.02,
        "velocity_std_mps": 0.1,
        "normal_std_rad": 0.05,
        "tts_std_s": 0.1,
        "truth_fields_immutable": [
            "question_bank_row",
            "physical_ball",
            "reward_target",
            "critic_target",
        ],
        "actor_revision_fields": [
            "target_position",
            "target_velocity",
            "signed_target_normal",
            "time_to_strike",
        ],
    }
    return contract


def test_planner_revision_metadata_is_complete_canonical_and_profile_bound():
    contract = _planner_revision_schema3_contract()
    metadata = TC.planner_task_revision_metadata(contract)
    assert metadata == json.dumps(
        contract["planner_task_revision"],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    decoded = json.loads(metadata)
    assert set(decoded) == {
        "enabled",
        "revision_schema_version",
        "governor",
        "initial_tts_range_s",
    }
    assert set(decoded["governor"]["profile"]) == {
        "policy_dt_s",
        "min_tts_s",
        "max_tts_s",
        "max_phase_rate_per_s",
        "max_phase_acceleration_per_s2",
        "max_deadline_revision_delta_s",
        "max_position_revision_delta_m",
        "max_velocity_revision_delta_mps",
        "max_normal_revision_delta_rad",
        "normal_unit_tolerance",
        "early_deadline_tolerance_s",
        "contract_version",
        "schema_version",
    }
    profile_bytes = (
        json.dumps(
            decoded["governor"]["profile"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(profile_bytes).hexdigest() == decoded["governor"][
        "profile_sha256"
    ]
    # Same fixed vector is hard-coded in C++ test_pp_onnx_policy.cpp; this catches cross-language
    # key-order/number-format drift instead of letting each side self-sign its own serialization.
    assert decoded["governor"]["profile_sha256"] == (
        "3ebaca6b7f6ce1e841d5cf62aab3df38c796093ce563b6bfb4ef3048a0dd886a"
    )
    # Validation also proves the checkpoint-bound strike-frame inputs are complete.
    TC.validate_schema3_contract_structure(contract)


@pytest.mark.parametrize("keep", ["planner_task_revision", "planner_task_revision_training"])
def test_planner_revision_half_configuration_fails_closed(keep):
    contract = _planner_revision_schema3_contract()
    drop = (
        "planner_task_revision_training"
        if keep == "planner_task_revision"
        else "planner_task_revision"
    )
    contract.pop(drop)
    with pytest.raises(ValueError, match="half-configured"):
        TC.planner_task_revision_metadata(contract)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda contract: contract["planner_task_revision"].pop("initial_tts_range_s"),
            "missing fields",
        ),
        (
            lambda contract: contract["planner_task_revision"]["governor"]["profile"].pop(
                "max_phase_rate_per_s"
            ),
            "missing fields",
        ),
        (
            lambda contract: contract["planner_task_revision"]["governor"]["profile"].__setitem__(
                "max_phase_acceleration_per_s2", 21.0
            ),
            "does not bind",
        ),
        (
            lambda contract: contract.__setitem__("strike_phase_per_clip", [0.4]),
            "one checkpoint-bound strike phase",
        ),
    ],
)
def test_planner_revision_missing_partial_and_tamper_fail_closed(mutation, message):
    contract = _planner_revision_schema3_contract()
    mutation(contract)
    with pytest.raises(ValueError, match=message):
        TC.planner_task_revision_metadata(contract)


def test_planner_revision_range_and_policy_clock_mismatch_fail_closed():
    contract = _planner_revision_schema3_contract()
    contract["planner_task_revision"]["initial_tts_range_s"] = [0.05, 1.5]
    with pytest.raises(ValueError, match="strictly ordered inside"):
        TC.planner_task_revision_metadata(contract)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda contract: contract["planner_task_revision_training"].pop(
                "initial_tts_mixture"
            ),
            "missing fields",
        ),
        (
            lambda contract: contract["planner_task_revision_training"].__setitem__(
                "initial_tts_mixture", ["contract_version", "components"]
            ),
            "must be an object",
        ),
        (
            lambda contract: contract["planner_task_revision_training"][
                "initial_tts_mixture"
            ]["components"][0].__setitem__("weight", 0.14),
            "weights must sum to 1",
        ),
        (
            lambda contract: contract["planner_task_revision_training"][
                "initial_tts_mixture"
            ]["components"][0].__setitem__("range_s", [0.3, 0.49]),
            "support must equal",
        ),
        (
            lambda contract: contract["planner_task_revision_training"][
                "initial_tts_mixture"
            ]["components"][1].__setitem__("range_s", [0.5, 0.51]),
            "exact 0.5 s point mass",
        ),
    ],
)
def test_planner_revision_initial_tts_mixture_fails_closed(mutation, message):
    contract = _planner_revision_schema3_contract()
    mutation(contract)
    with pytest.raises(ValueError, match=message):
        TC.planner_task_revision_metadata(contract)

    contract = _planner_revision_schema3_contract()
    profile = contract["planner_task_revision"]["governor"]["profile"]
    profile["policy_dt_s"] = 0.01
    canonical = (
        json.dumps(profile, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    contract["planner_task_revision"]["governor"]["profile_sha256"] = hashlib.sha256(
        canonical
    ).hexdigest()
    with pytest.raises(ValueError, match="disagrees with schema-3 policy_step_dt_s"):
        TC.planner_task_revision_metadata(contract)


def test_legacy_contract_does_not_acquire_planner_revision_metadata():
    contract = _schema3_contract()
    assert TC.planner_task_revision_metadata(contract) is None
    donor_metadata = {"planner_task_revision": '{"enabled":true}', "other": "kept"}
    assert TC.bind_planner_task_revision_metadata(donor_metadata, contract) is None
    assert donor_metadata == {"other": "kept"}

    enabled_contract = _planner_revision_schema3_contract()
    encoded = TC.bind_planner_task_revision_metadata(donor_metadata, enabled_contract)
    assert encoded == TC.planner_task_revision_metadata(enabled_contract)
    assert donor_metadata["planner_task_revision"] == encoded

    diagnostic_v2 = {"schema_version": 2}
    assert TC.planner_task_revision_metadata(diagnostic_v2) is None

    # Presence on an old sidecar is not an opt-in; only checkpoint-bound schema 3 may carry it.
    diagnostic_v2["planner_task_revision"] = _planner_revision_schema3_contract()[
        "planner_task_revision"
    ]
    diagnostic_v2["planner_task_revision_training"] = _planner_revision_schema3_contract()[
        "planner_task_revision_training"
    ]
    with pytest.raises(ValueError, match="requires schema-3"):
        TC.planner_task_revision_metadata(diagnostic_v2)


def _qdot_hinge_schema3_contract():
    contract = {
        "schema_version": 3,
        **TC.runtime_execution_facts(_env(joints=31), _ActorContract()),
        "racket_control_point": "pingpang_red_Link_origin_v1",
        "racket_control_point_offset_wrist_m": [0.21021, 0.032078, 0.032036],
    }
    contract["joint_velocity_limit_hinge_reward"] = {
        "schema_version": 1,
        "enabled": True,
        "weight": -0.25,
        "margin": 0.85,
        "asset_name": "robot",
        "joint_count": 31,
        "joint_order": "runtime_articulation_identity",
        "velocity_limit_source": "runtime_execution_facts.joint_velocity_limits",
        "formula": "mean(relu(abs(qd)/joint_velocity_limits-margin)^2)",
    }
    return contract


def _command_schema3_contract():
    contract = _schema3_contract()
    contract.update({
        "actor_obs_contract": "deploy_parity",
        "actor_obs_mode": "deploy_parity",
        "actor_obs_total_dim": 62,
        "actor_obs_term_names": ["command"],
        "actor_obs_term_dims": [62],
        "observation_history_lengths": [1],
        "actor_leg_ref_mask_provenance_epoch": 1,
    })
    return contract


def _diagnostic_schema3_contract():
    contract = _schema3_contract()
    contract["motion_kinematics_contracts"] = [
        {
            "schema_version": None,
            "body_pos_point": "link_origin",
            "body_lin_vel_point": "link_origin",
            "body_names": None,
            "exact": False,
            "status": "legacy_link_origin_velocity_diagnostic_only",
            "link_fd_max_abs_mps": 1.0e-6,
            "max_ang_radps": 2.0,
        }
        for _ in contract["motion_kinematics_contracts"]
    ]
    contract["motion_kinematics_exact"] = False
    return contract


def test_per_clip_velocity_points_preserve_explicit_and_legacy_assumed_com_semantics():
    contracts = [
        {"body_lin_vel_point": "center_of_mass", "status": "declared_v2"},
        {"body_lin_vel_point": None, "status": "legacy_unbound_assumed_com"},
        {
            "body_lin_vel_point": "link_origin",
            "status": "legacy_link_origin_velocity_diagnostic_only",
        },
    ]
    assert TC.resolve_motion_body_lin_vel_points(contracts) == (
        "center_of_mass", "center_of_mass", "link_origin",
    )

    for bad in (
        [{"body_lin_vel_point": None, "status": "unknown_legacy"}],
        [{"body_lin_vel_point": "inertial_origin", "status": "declared"}],
    ):
        with pytest.raises(ValueError, match="unresolved null|unknown body_lin_vel_point"):
            TC.resolve_motion_body_lin_vel_points(bad)


def test_schema3_inexact_assumed_com_remains_structural_but_not_formal():
    contract = _diagnostic_schema3_contract()
    contract["motion_kinematics_contracts"][0]["body_lin_vel_point"] = None
    contract["motion_kinematics_contracts"][0]["status"] = "legacy_unbound_assumed_com"
    TC.validate_schema3_contract_structure(contract)
    assert TC.resolve_motion_body_lin_vel_points(
        contract["motion_kinematics_contracts"]
    ) == ("center_of_mass", "link_origin")
    with pytest.raises(ValueError, match="formal lineage requires"):
        TC.validate_schema3_contract(contract)


def test_schema3_requires_every_execution_field_and_rejects_other_formal_versions():
    contract = _schema3_contract()
    TC.validate_schema3_contract(contract)
    for key in (*TC.RUNTIME_EXECUTION_KEYS, *TC.SCHEMA3_TASK_KEYS):
        broken = dict(contract)
        broken.pop(key)
        with pytest.raises(ValueError, match="missing execution facts"):
            TC.validate_schema3_contract(broken)
    with pytest.raises(ValueError, match="unsupported formal"):
        TC.validate_schema3_contract({**contract, "schema_version": 2})
    with pytest.raises(ValueError, match="unsupported formal"):
        TC.validate_schema3_contract({**contract, "schema_version": 4})


@pytest.mark.parametrize("invalid", [False, None, 0, 1, "1"])
def test_schema3_actor_leg_ref_mask_is_only_when_true(invalid):
    contract = _command_schema3_contract()
    contract["actor_leg_ref_mask"] = invalid
    with pytest.raises(ValueError, match="actor_leg_ref_mask must be true when present"):
        TC.validate_schema3_contract_structure(contract)

    contract["actor_leg_ref_mask"] = True
    TC.validate_schema3_contract_structure(contract)


@pytest.mark.parametrize("invalid", [False, None, 0, 2, "1", 1.0])
def test_schema3_command_contract_requires_exact_mask_provenance_epoch(invalid):
    contract = _command_schema3_contract()
    contract["actor_leg_ref_mask_provenance_epoch"] = invalid
    with pytest.raises(ValueError, match="provenance_epoch"):
        TC.validate_schema3_contract_structure(contract)

    contract.pop("actor_leg_ref_mask_provenance_epoch")
    with pytest.raises(ValueError, match="masked and unmasked checkpoints are indistinguishable"):
        TC.validate_schema3_contract_structure(contract)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"actor_obs_term_names": ["command", "command"],
          "actor_obs_term_dims": [31, 31],
          "observation_history_lengths": [1, 1]}, "names must be unique"),
        ({"actor_obs_term_dims": [62.0]}, "dims must be positive integers"),
        ({"actor_obs_term_dims": [True]}, "dims must be positive integers"),
        ({"observation_history_lengths": [0]}, "history lengths must be positive integers"),
        ({"actor_obs_total_dim": 61}, "must equal the sum"),
        ({"actor_obs_term_names": ["command", "extra"]}, "equal-length arrays"),
    ],
)
def test_schema3_actor_layout_cannot_bypass_mask_epoch(updates, message):
    contract = _command_schema3_contract()
    contract.update(updates)
    with pytest.raises(ValueError, match=message):
        TC.validate_schema3_contract_structure(contract)


def test_diagnostic_schema3_is_structurally_valid_but_never_formal_exact():
    contract = _diagnostic_schema3_contract()
    TC.validate_schema3_contract_structure(contract)
    with pytest.raises(ValueError, match="formal lineage requires motion_kinematics_exact=true"):
        TC.validate_schema3_contract(contract)


def test_diagnostic_schema3_rejects_malformed_or_self_promoting_motion_facts():
    mismatch = _diagnostic_schema3_contract()
    mismatch["motion_kinematics_exact"] = True
    with pytest.raises(ValueError, match="disagrees with the per-clip"):
        TC.validate_schema3_contract_structure(mismatch)

    malformed = _diagnostic_schema3_contract()
    malformed["motion_kinematics_contracts"][0].pop("status")
    with pytest.raises(ValueError, match="status must be non-empty"):
        TC.validate_schema3_contract_structure(malformed)

    promoted = _diagnostic_schema3_contract()
    promoted["motion_kinematics_contracts"][0]["exact"] = True
    with pytest.raises(ValueError, match="claims exact without"):
        TC.validate_schema3_contract_structure(promoted)

    bad_face = _diagnostic_schema3_contract()
    bad_face["face_command_enabled"] = "true"
    with pytest.raises(ValueError, match="face_command_enabled must be boolean"):
        TC.validate_schema3_contract_structure(bad_face)


def test_formal_face179_contract_freezes_shared_a_frame_and_striking_face_signs():
    contract = _schema3_contract()
    contract.update({
        "actor_obs_contract": "deploy_parity_face179",
        "face_command_enabled": True,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1.0, -1.0],
    })
    TC.validate_schema3_contract_structure(contract)

    for signs in (None, [1.0, 1.0], [-1.0, 1.0], [True, -1.0]):
        broken = dict(contract)
        if signs is None:
            broken.pop("mount_normal_sign_per_clip")
        else:
            broken["mount_normal_sign_per_clip"] = signs
        with pytest.raises(ValueError, match=r"mount_normal_sign_per_clip=\[\+1,-1\]"):
            TC.validate_schema3_contract_structure(broken)

    wrong_pair = dict(contract, face_command_pairing="legacy_signed_vs_A")
    with pytest.raises(ValueError, match="requires shared_plus_y"):
        TC.validate_schema3_contract_structure(wrong_pair)

    disabled = dict(contract, face_command_enabled=False)
    with pytest.raises(ValueError, match="requires face_command_enabled=true"):
        TC.validate_schema3_contract_structure(disabled)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("joint_actuator_types", ["implicit", "implicit", "mystery"], "implicit\\|explicit"),
        ("joint_effort_limits", [30.0, 0.0, 60.0], "positive"),
        ("joint_armature", [0.0, -0.1, 0.0], "non-negative"),
        ("joint_friction_coefficients", [0.1, float("nan"), 0.2], "non-negative"),
        ("joint_velocity_limits", [8.0, 0.0, 12.0], "positive"),
        ("joint_friction_backend", "mujoco", "must be physx"),
        ("joint_friction_units", "N*m", "dimensionless"),
    ],
)
def test_schema3_rejects_unproven_actuator_plant_values(key, value, message):
    contract = _schema3_contract()
    contract[key] = value
    with pytest.raises(ValueError, match=message):
        TC.validate_schema3_contract(contract)


def test_schema3_validates_optional_qdot_limit_hinge_contract():
    contract = _qdot_hinge_schema3_contract()
    TC.validate_schema3_contract(contract)

    disabled = _qdot_hinge_schema3_contract()
    disabled["joint_velocity_limit_hinge_reward"] = {
        **disabled["joint_velocity_limit_hinge_reward"],
        "enabled": False,
        "weight": 0.0,
    }
    TC.validate_schema3_contract(disabled)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema_version", 1.0, "schema_version must be integer 1"),
        ("weight", 0.1, "weight must be finite and <= 0"),
        ("margin", 1.0, r"margin must be finite and in \(0, 1\)"),
        ("enabled", False, "enabled disagrees with weight"),
        ("joint_count", 30, "identity 31-joint order"),
        ("joint_count", 31.0, "identity 31-joint order"),
        ("joint_order", "sorted_by_name", "joint_order must be exactly"),
        ("formula", "action_rate_l2", "formula must be exactly"),
    ],
)
def test_schema3_rejects_qdot_limit_hinge_contract_drift(key, value, message):
    contract = _qdot_hinge_schema3_contract()
    contract["joint_velocity_limit_hinge_reward"] = {
        **contract["joint_velocity_limit_hinge_reward"],
        key: value,
    }
    with pytest.raises(ValueError, match=message):
        TC.validate_schema3_contract(contract)


def test_checkpoint_binding_requires_sha_schema_and_exact_lineage():
    digest = "a" * 64
    exact = {
        "infos": {
            TC.CHECKPOINT_CONTRACT_SCHEMA_KEY: 3,
            TC.CHECKPOINT_CONTRACT_SHA_KEY: digest,
            TC.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY: 1,
        }
    }
    TC.require_checkpoint_contract_binding(exact, schema=3, sha256=digest)
    inexact = {"infos": {**exact["infos"], TC.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY: 0}}
    TC.require_checkpoint_contract_binding(
        inexact, schema=3, sha256=digest, require_lineage_exact=False
    )
    assert TC.checkpoint_contract_lineage_exact(inexact) is False
    with pytest.raises(ValueError, match="exact-lineage"):
        TC.require_checkpoint_contract_binding(inexact, schema=3, sha256=digest)
    with pytest.raises(ValueError, match="not bound"):
        TC.require_checkpoint_contract_binding(exact, schema=3, sha256="b" * 64)
    assert TC.checkpoint_claims_contract(exact)
    assert TC.checkpoint_claims_contract(
        {"infos": {TC.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY: 0}}
    )
    assert not TC.checkpoint_claims_contract({"infos": {}})


def test_export_paths_do_not_promote_schema2_or_unknown_schemas():
    exporter = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/utils/exporter.py"
    ).read_text(encoding="utf-8")
    standalone = (ROOT / "scripts/standalone_onnx_export.py").read_text(encoding="utf-8")
    assert "training_contract_schema >= 2" not in exporter
    assert "contract_schema >= 2" not in standalone
    assert 'training_contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION' in exporter
    assert 'contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION' in standalone
    assert 'metadata["training_contract_exact"] = "1"' in exporter
    assert '"training_contract_exact": "1" if training_contract_lineage_exact else "0"' in standalone
    assert "bind_actor_leg_ref_mask_metadata" in standalone
    assert "bind_actor_leg_ref_mask_metadata" in exporter
    for field in (
        "qdes_joint_pos_limits",
        "physics_step_dt_s",
        "policy_step_dt_s",
        "control_decimation",
        "joint_actuator_types",
        "joint_armature",
        "joint_friction_coefficients",
        "joint_velocity_limits",
        "joint_friction_semantics",
        "racket_control_point_offset_wrist_m",
        "face_command_enabled",
        "face_command_pairing",
        "motion_allow_legacy_link_origin_velocity",
        "motion_body_lin_vel_points",
        "motion_kinematics_exact",
        "training_contract_schema_version",
        "training_contract_sha256",
        "source_checkpoint_sha256",
    ):
        assert field in exporter
        assert field in standalone
    assert "_donor_activation" in standalone
    assert "donor graph has ambiguous activation operators" in standalone
    assert 'donor_meta["motion_body_lin_vel_points"] = ",".join(' in standalone
    assert 'donor_meta.pop("motion_body_lin_vel_points", None)' in standalone
    assert "ACTOR_LEG_REF_MASK_PROVENANCE_KEY" in exporter
    assert "bind_actor_leg_ref_mask_metadata" in standalone
    for source in (exporter, standalone):
        assert "validate_schema3_contract_structure" in source
        assert "checkpoint_claims_contract" in source
        assert "if training_contract_lineage_exact" in source


def test_training_hard_contract_traces_face_pairing_and_legacy_motion_opt_in():
    train = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert '"face_command_enabled": bool(getattr(racket, "face_command", False))' in train
    assert '"face_command_pairing": attr(racket, "face_command_pairing", "shared_plus_y")' in train
    assert '"motion_allow_legacy_link_origin_velocity": bool(' in train
    assert "contract_lineage_exact = ckpt is None and motion_kinematics_exact" in train
    tracking_cfg = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    ).read_text(encoding="utf-8")
    observations = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_observations.py"
    ).read_text(encoding="utf-8")
    contract_source = MODULE_PATH.read_text(encoding="utf-8")
    assert "command = ObsTerm(func=mdp.generated_commands" in tracking_cfg
    assert "def generated_commands_actor_leg_masked(" in observations
    assert "from isaaclab.envs.mdp import generated_commands" in contract_source
    assert "if func is canonical" in contract_source


def test_runner_embeds_schema_sha_and_lineage_in_checkpoint_infos():
    source = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py"
    ).read_text(encoding="utf-8")
    assert "infos[CHECKPOINT_CONTRACT_SCHEMA_KEY]" in source
    assert "infos[CHECKPOINT_CONTRACT_SHA_KEY]" in source
    assert "infos[CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY]" in source
    train = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert "require_checkpoint_contract_binding(" in train
    assert "training_contract_lineage_exact=contract_lineage_exact" in train
