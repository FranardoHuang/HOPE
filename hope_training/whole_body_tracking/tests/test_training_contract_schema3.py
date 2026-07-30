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


def _env(
    *,
    joints=3,
    num_envs=2,
    action_ids=slice(None),
    policy_dt=0.02,
    finite_qdes_projection=False,
    finite_projection_inset=0.05,
):
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
    action_cfg = SimpleNamespace(
        use_default_offset=True,
        project_finite_preclamp_qdes_without_termination=(
            finite_qdes_projection
        ),
        finite_projection_soft_envelope_inset_fraction=(
            finite_projection_inset
        ),
    )
    action = SimpleNamespace(
        _joint_ids=action_ids,
        _scale=np.tile(np.linspace(0.1, 0.3, joints), (num_envs, 1)),
        _clamp_enabled=True,
        cfg=action_cfg,
        finite_preclamp_qdes_projection_enabled=finite_qdes_projection,
        finite_projection_soft_envelope_inset_fraction=(
            finite_projection_inset if finite_qdes_projection else 0.0
        ),
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
    joint_pos_cfg = SimpleNamespace(
        use_default_offset=True,
        clamp=True,
        project_finite_preclamp_qdes_without_termination=(
            finite_qdes_projection
        ),
        finite_projection_soft_envelope_inset_fraction=(
            finite_projection_inset
        ),
    )
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
    assert TC.FINITE_PRECLAMP_QDES_PROJECTION_KEY not in facts
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


def test_runtime_projection_fact_is_true_only_and_runtime_verified():
    enabled = _env(finite_qdes_projection=True)
    facts = TC.runtime_execution_facts(enabled, _ActorContract())
    assert facts[TC.FINITE_PRECLAMP_QDES_PROJECTION_KEY] is True
    assert (
        facts[TC.FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY]
        == 0.05
    )
    assert tuple(
        key
        for key in facts
        if key
        not in (
            TC.FINITE_PRECLAMP_QDES_PROJECTION_KEY,
            TC.FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY,
        )
    ) == TC.RUNTIME_EXECUTION_KEYS

    enabled.action_manager.get_term(
        "joint_pos"
    ).finite_preclamp_qdes_projection_enabled = False
    with pytest.raises(RuntimeError, match="config/runtime facts disagree"):
        TC.runtime_execution_facts(enabled, _ActorContract())

    missing_runtime = _env(finite_qdes_projection=True)
    del missing_runtime.action_manager.get_term(
        "joint_pos"
    ).finite_preclamp_qdes_projection_enabled
    with pytest.raises(RuntimeError, match="exposes no runtime projection property"):
        TC.runtime_execution_facts(missing_runtime, _ActorContract())

    non_boolean = _env(finite_qdes_projection=True)
    non_boolean.action_manager.get_term(
        "joint_pos"
    ).finite_preclamp_qdes_projection_enabled = 1
    with pytest.raises(RuntimeError, match="must be an exact boolean"):
        TC.runtime_execution_facts(non_boolean, _ActorContract())

    mismatched_inset = _env(finite_qdes_projection=True)
    mismatched_inset.action_manager.get_term(
        "joint_pos"
    ).finite_projection_soft_envelope_inset_fraction = 0.04
    with pytest.raises(RuntimeError, match="inset config/runtime facts disagree"):
        TC.runtime_execution_facts(mismatched_inset, _ActorContract())


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


_SOFT_LIMIT_V2_FORMULA = (
    "sum(where(u>0,penalty_floor+(1-penalty_floor)*"
    "(1-exp(-shape_rate*clamp(u,0,1)))/(1-exp(-shape_rate)),0));"
    "u=relu(m_eff-min(q-lo,hi-q)/(hi-lo))/m_eff;"
    "m_eff=min(margin_frac,min(default_q-lo,hi-default_q)/(hi-lo)-stance_eps);"
    "require_all(m_eff>margin_floor)"
)


def _schema3_soft_limit_v2_contract():
    contract = {
        "schema_version": 3,
        **TC.runtime_execution_facts(_env(joints=31), _ActorContract()),
        "racket_control_point": "pingpang_red_Link_origin_v1",
        "racket_control_point_offset_wrist_m": [0.21021, 0.032078, 0.032036],
    }
    shared = {
        "schema_version": 2,
        "enabled": True,
        "probe_enabled": True,
        "activation_ledger": "weight_independent_control_step_counters",
        "weight": -40.0,
        "margin_frac": 0.08,
        "penalty_floor": 0.25,
        "shape_rate": 4.0,
        "stance_eps": 0.005,
        "margin_floor": 0.005,
        "joint_count": 31,
        "joint_order": "runtime_articulation_identity",
        "position_limit_source": "articulation.data.soft_joint_pos_limits",
        "default_stance_source": "articulation.data.default_joint_pos",
        "formula": _SOFT_LIMIT_V2_FORMULA,
        "aggregation": "sum_all_31_joints",
        "per_joint_cap": 1.0,
        "gate": "dense_every_control_step",
    }
    contract["qdes_limit_barrier_reward"] = {
        **shared,
        "term_name": "qdes_limit_barrier",
        "probe_term_name": "qdes_limit_barrier_probe",
        "term_callable": (
            "whole_body_tracking.tasks.tracking.mdp.qdes_limit_barrier_v2"
        ),
        "probe_callable": (
            "whole_body_tracking.tasks.tracking.mdp.qdes_limit_barrier_v2_probe"
        ),
        "action_name": "joint_pos",
        "position_source": "joint_pos.processed_actions",
    }
    contract["actual_joint_limit_barrier_reward"] = {
        **shared,
        "term_name": "joint_limit",
        "probe_term_name": "actual_joint_limit_barrier_probe",
        "term_callable": (
            "whole_body_tracking.tasks.tracking.mdp.actual_joint_limit_barrier_v2"
        ),
        "probe_callable": (
            "whole_body_tracking.tasks.tracking.mdp."
            "actual_joint_limit_barrier_v2_probe"
        ),
        "asset_name": "robot",
        "position_source": "articulation.data.joint_pos",
    }
    return contract


def test_schema3_soft_limit_v2_roundtrip_binds_both_independent_channels():
    contract = _schema3_soft_limit_v2_contract()
    TC.validate_schema3_contract_structure(contract)
    TC.validate_schema3_contract(contract)


@pytest.mark.parametrize(
    ("channel", "key", "value", "message"),
    [
        ("qdes_limit_barrier_reward", "probe_callable", "lookalike", "probe_callable"),
        ("qdes_limit_barrier_reward", "penalty_floor", 0.0, "penalty_floor"),
        ("actual_joint_limit_barrier_reward", "term_name", "joint_limit_v1", "term_name"),
        ("actual_joint_limit_barrier_reward", "weight", -20.0, "weight must match"),
        ("actual_joint_limit_barrier_reward", "margin_frac", 0.1, "margin_frac must match"),
        ("actual_joint_limit_barrier_reward", "penalty_floor", 0.5, "penalty_floor must match"),
    ],
)
def test_schema3_soft_limit_v2_rejects_callable_and_cross_channel_drift(
    channel, key, value, message
):
    contract = _schema3_soft_limit_v2_contract()
    contract[channel][key] = value
    with pytest.raises(ValueError, match=message):
        TC.validate_schema3_contract_structure(contract)


def test_schema3_soft_limit_v2_requires_floor_and_actual_channel():
    missing_floor = _schema3_soft_limit_v2_contract()
    missing_floor["qdes_limit_barrier_reward"].pop("penalty_floor")
    with pytest.raises(ValueError, match="missing fields"):
        TC.validate_schema3_contract_structure(missing_floor)

    missing_actual = _schema3_soft_limit_v2_contract()
    missing_actual.pop("actual_joint_limit_barrier_reward")
    with pytest.raises(ValueError, match="requires the independent"):
        TC.validate_schema3_contract_structure(missing_actual)


def test_soft_limit_v1_v2_contract_sha_drift_cannot_exact_resume():
    v2 = _schema3_soft_limit_v2_contract()
    v2_sha = hashlib.sha256(
        json.dumps(
            v2, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    # A legacy donor cannot claim the fresh schema-2 sidecar hash.  This is the exact
    # checkpoint gate used by train.py after its full hard-contract equality check.
    donor = {
        "infos": {
            TC.CHECKPOINT_CONTRACT_SCHEMA_KEY: 3,
            TC.CHECKPOINT_CONTRACT_SHA_KEY: "a" * 64,
            TC.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY: 1,
        }
    }
    with pytest.raises(ValueError, match="not bound"):
        TC.require_checkpoint_contract_binding(donor, schema=3, sha256=v2_sha)


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
        # 2026-07-25 SUM 裁定:与 train.py/validator 同步;旧 mean 串 sidecar 应 fail loud
        "formula": "sum(relu(abs(qd)/joint_velocity_limits-margin)^2)",
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


def _action_ball_diagnostic_schema3_contract():
    contract = _schema3_contract()
    contract[TC.FINITE_PRECLAMP_QDES_PROJECTION_KEY] = True
    contract[TC.FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY] = 0.05
    contract["target_mode"] = "action_ball"
    contract["actor_obs_contract"] = "action_ball_n2"
    contract["actor_obs_mode"] = "deploy_parity"
    contract["actor_obs_total_dim"] = 183
    contract["actor_obs_term_names"] = ["action_ball_policy"]
    contract["actor_obs_term_dims"] = [183]
    contract["observation_history_lengths"] = [1]
    contract["action_ball_training"] = {
        "schema_version": 1,
        "authorization": {
            "diagnostic_unauthorized": True,
            "formal_evidence_prohibited": True,
            "curriculum_promotion_prohibited": True,
            "exact_export_prohibited": True,
            "formal_judge_prohibited": True,
        },
        "runtime": {
            "diagnostic_unauthorized": True,
            "evaluator_authority": {
                "diagnostic_unauthorized": True,
                "formal_authority_available": False,
                "formal_launch_requires_code_pinned_receipt": True,
                "runtime_or_manifest_may_self_authorize": False,
                "authority_binding": {"kind": "diagnostic"},
                "authority_state_owner_sha256": "a" * 64,
            },
        },
        "motion_admission": {
            "diagnostic_unauthorized": True,
            "training_authorized": False,
        },
    }
    return contract


def _action_set_identity(
    *,
    profile_id="fixture_upper_nomove_n2",
    table_pose=False,
    table_pose_twist=False,
    heading_task=False,
    teacher_start=False,
):
    assert (
        sum(
            bool(value)
            for value in (
                table_pose,
                table_pose_twist,
                heading_task,
                teacher_start,
            )
        )
        <= 1
    )
    if teacher_start:
        actor_obs_contract = (
            "action_ball_table_pose_twist_heading_task_teacher_start_n2"
        )
        actor_obs_width = 196
    elif heading_task:
        actor_obs_contract = (
            "action_ball_table_pose_twist_heading_task_n2"
        )
        actor_obs_width = 195
    elif table_pose_twist:
        actor_obs_contract = "action_ball_table_pose_twist_n2"
        actor_obs_width = 195
    elif table_pose:
        actor_obs_contract = "action_ball_table_pose_n2"
        actor_obs_width = 192
    else:
        actor_obs_contract = "action_ball_n2"
        actor_obs_width = 183
    action_ids = ["bh_loop", "bh_block"]
    action_uids = [101, 202]
    order_digest = TC._action_ball_order_uid_digest(action_ids, action_uids)
    row = {
        "schema_version": 1,
        "kind": TC.ACTION_BALL_ACTION_SET_CONTRACT_KIND,
        "profile_id": profile_id,
        "expected_n": 2,
        "scope": "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": action_ids,
        "ordered_action_uids": action_uids,
        "order_uid_digest_sha256": order_digest,
        "manifest_path": "configs/action_ball_fixture_n2.json",
        "manifest_sha256": "c" * 64,
        "experiment_name": "agibot_a3_hope_action_ball_fixture_n2",
        "actor_obs_contract": actor_obs_contract,
        "actor_obs_width": actor_obs_width,
        "namespace_identity": f"n2-{order_digest[:12]}",
    }
    row["contract_sha256"] = TC._action_ball_canonical_sha256(row)
    row["contract_source_path"] = TC.ACTION_BALL_ACTION_SET_SOURCE_PATH
    row["contract_source_sha256"] = "d" * 64
    return row


def _action_ball_formal_schema3_contract():
    contract = _action_ball_diagnostic_schema3_contract()
    action_ball = contract["action_ball_training"]
    action_ball["authorization"] = {
        key: False for key in action_ball["authorization"]
    }
    action_ball["runtime"].pop("diagnostic_unauthorized")
    evaluator = action_ball["runtime"]["evaluator_authority"]
    evaluator.pop("diagnostic_unauthorized")
    evaluator["formal_authority_available"] = True
    action_ball["motion_admission"] = {
        "authorization_purpose": "training",
        "certificate_sha256": "b" * 64,
    }
    identity = _action_set_identity()
    action_ball["action_set_identity"] = identity
    action_ball["preflight"] = {
        "manifest": {
            "path": identity["manifest_path"],
            "file_sha256": identity["manifest_sha256"],
        },
        "prototype": {"scope": identity["scope"]},
        "mobility_mode": identity["mobility_mode"],
        "action_order": list(identity["ordered_action_ids"]),
        "action_uids": list(identity["ordered_action_uids"]),
        "action_bindings": [
            {"action_id": action_id, "action_uid": action_uid}
            for action_id, action_uid in zip(
                identity["ordered_action_ids"],
                identity["ordered_action_uids"],
            )
        ],
    }
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
    for invalid_schema in (3.0, "3", True):
        with pytest.raises(ValueError, match="plain integer"):
            TC.validate_schema3_contract_structure(
                {**contract, "schema_version": invalid_schema}
            )


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


def test_action_ball_diagnostic_authorization_is_structural_but_never_formal():
    contract = _action_ball_diagnostic_schema3_contract()
    TC.validate_schema3_contract_structure(contract)
    assert TC.validate_action_ball_training_authorization(contract) is True
    with pytest.raises(
        ValueError,
        match="formal validation rejects diagnostic_unauthorized",
    ):
        TC.validate_schema3_contract(contract)


def test_action_ball_requires_true_immutable_projection_fact():
    contract = _action_ball_diagnostic_schema3_contract()
    contract.pop(TC.FINITE_PRECLAMP_QDES_PROJECTION_KEY)
    with pytest.raises(
        ValueError,
        match="missing the immutable finite pre-clamp q_des projection",
    ):
        TC.validate_schema3_contract_structure(contract)

    contract[TC.FINITE_PRECLAMP_QDES_PROJECTION_KEY] = False
    with pytest.raises(ValueError, match="exact boolean true"):
        TC.validate_schema3_contract_structure(contract)

    contract = _action_ball_diagnostic_schema3_contract()
    contract.pop(TC.FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY)
    with pytest.raises(ValueError, match="soft-envelope inset fact"):
        TC.validate_schema3_contract_structure(contract)

    legacy = _schema3_contract()
    legacy[TC.FINITE_PRECLAMP_QDES_PROJECTION_KEY] = True
    with pytest.raises(ValueError, match="ActionBall-only"):
        TC.validate_schema3_contract_structure(legacy)


def test_action_ball_formal_authorization_still_passes_formal_validation():
    contract = _action_ball_formal_schema3_contract()
    assert TC.validate_action_ball_training_authorization(contract) is False
    TC.validate_schema3_contract(contract)
    metadata = {
        TC.ACTION_BALL_DIAGNOSTIC_METADATA_KEY: "donor-value",
        TC.FORMAL_EVIDENCE_BOOKABLE_METADATA_KEY: "0",
    }
    assert (
        TC.bind_action_ball_diagnostic_metadata(
            metadata,
            contract,
            lineage_exact=True,
        )
        is False
    )
    assert TC.ACTION_BALL_DIAGNOSTIC_METADATA_KEY not in metadata
    assert TC.FORMAL_EVIDENCE_BOOKABLE_METADATA_KEY not in metadata


def test_action_ball_action_set_runtime_rejects_swapped_order_and_cross_n():
    identity = _action_set_identity()
    kwargs = {
        "actor_obs_contract": "action_ball_n2",
        "actor_obs_width": 183,
        "manifest_path": identity["manifest_path"],
        "manifest_sha256": identity["manifest_sha256"],
        "scope": "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": identity["ordered_action_ids"],
        "ordered_action_uids": identity["ordered_action_uids"],
        "experiment_name": identity["experiment_name"],
    }
    assert (
        TC.validate_action_ball_action_set_runtime_identity(identity, **kwargs)
        == identity
    )

    swapped = dict(kwargs)
    swapped["ordered_action_ids"] = list(reversed(identity["ordered_action_ids"]))
    with pytest.raises(ValueError, match="disagrees with live runtime"):
        TC.validate_action_ball_action_set_runtime_identity(identity, **swapped)

    cross_n = dict(kwargs)
    cross_n["actor_obs_contract"] = "action_ball_n1"
    cross_n["actor_obs_width"] = 182
    with pytest.raises(ValueError, match="disagrees with live runtime"):
        TC.validate_action_ball_action_set_runtime_identity(identity, **cross_n)


def test_action_ball_table_pose_layout_is_accepted_with_exact_n_and_width():
    contract = _action_ball_diagnostic_schema3_contract()
    contract["actor_obs_contract"] = "action_ball_table_pose_n2"
    contract["actor_obs_total_dim"] = 192
    contract["actor_obs_term_dims"] = [192]
    assert TC.validate_action_ball_training_authorization(contract) is True

    wrong_width = dict(contract)
    wrong_width["actor_obs_total_dim"] = 183
    with pytest.raises(ValueError, match="does not match"):
        TC.validate_action_ball_training_authorization(wrong_width)

    identity = _action_set_identity(table_pose=True)
    kwargs = {
        "actor_obs_contract": "action_ball_table_pose_n2",
        "actor_obs_width": 192,
        "manifest_path": identity["manifest_path"],
        "manifest_sha256": identity["manifest_sha256"],
        "scope": "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": identity["ordered_action_ids"],
        "ordered_action_uids": identity["ordered_action_uids"],
        "experiment_name": identity["experiment_name"],
    }
    assert (
        TC.validate_action_ball_action_set_runtime_identity(identity, **kwargs)
        == identity
    )

    cross_n = dict(identity)
    cross_n["actor_obs_contract"] = "action_ball_table_pose_n1"
    cross_n["actor_obs_width"] = 191
    code_row = {
        key: cross_n[key]
        for key in TC._ACTION_BALL_ACTION_SET_CODE_KEYS
        if key != "contract_sha256"
    }
    cross_n["contract_sha256"] = TC._action_ball_canonical_sha256(code_row)
    with pytest.raises(ValueError, match="exact N=2"):
        TC.validate_action_ball_action_set_identity_block(cross_n)


def test_action_ball_table_pose_twist_layout_is_accepted_with_exact_n_and_width():
    contract = _action_ball_diagnostic_schema3_contract()
    contract["actor_obs_contract"] = "action_ball_table_pose_twist_n2"
    contract["actor_obs_total_dim"] = 195
    contract["actor_obs_term_dims"] = [195]
    assert TC.validate_action_ball_training_authorization(contract) is True

    wrong_width = dict(contract)
    wrong_width["actor_obs_total_dim"] = 192
    with pytest.raises(ValueError, match="does not match"):
        TC.validate_action_ball_training_authorization(wrong_width)

    identity = _action_set_identity(table_pose_twist=True)
    kwargs = {
        "actor_obs_contract": "action_ball_table_pose_twist_n2",
        "actor_obs_width": 195,
        "manifest_path": identity["manifest_path"],
        "manifest_sha256": identity["manifest_sha256"],
        "scope": "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": identity["ordered_action_ids"],
        "ordered_action_uids": identity["ordered_action_uids"],
        "experiment_name": identity["experiment_name"],
    }
    assert (
        TC.validate_action_ball_action_set_runtime_identity(identity, **kwargs)
        == identity
    )

    cross_n = dict(identity)
    cross_n["actor_obs_contract"] = "action_ball_table_pose_twist_n1"
    cross_n["actor_obs_width"] = 194
    code_row = {
        key: cross_n[key]
        for key in TC._ACTION_BALL_ACTION_SET_CODE_KEYS
        if key != "contract_sha256"
    }
    cross_n["contract_sha256"] = TC._action_ball_canonical_sha256(code_row)
    with pytest.raises(ValueError, match="exact N=2"):
        TC.validate_action_ball_action_set_identity_block(cross_n)


def test_frame_consistent_action_ball_layout_is_accepted_with_exact_width():
    contract = _action_ball_diagnostic_schema3_contract()
    contract["actor_obs_contract"] = (
        "action_ball_table_pose_twist_heading_task_n2"
    )
    contract["actor_obs_total_dim"] = 195
    contract["actor_obs_term_dims"] = [195]
    assert TC.validate_action_ball_training_authorization(contract) is True

    wrong_width = dict(contract)
    wrong_width["actor_obs_total_dim"] = 194
    with pytest.raises(ValueError, match="does not match"):
        TC.validate_action_ball_training_authorization(wrong_width)

    identity = _action_set_identity(heading_task=True)
    kwargs = {
        "actor_obs_contract": (
            "action_ball_table_pose_twist_heading_task_n2"
        ),
        "actor_obs_width": 195,
        "manifest_path": identity["manifest_path"],
        "manifest_sha256": identity["manifest_sha256"],
        "scope": "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": identity["ordered_action_ids"],
        "ordered_action_uids": identity["ordered_action_uids"],
        "experiment_name": identity["experiment_name"],
    }
    assert (
        TC.validate_action_ball_action_set_runtime_identity(identity, **kwargs)
        == identity
    )


def test_teacher_start_action_ball_layout_is_accepted_with_exact_width():
    contract = _action_ball_diagnostic_schema3_contract()
    contract["actor_obs_contract"] = (
        "action_ball_table_pose_twist_heading_task_teacher_start_n2"
    )
    contract["actor_obs_total_dim"] = 196
    contract["actor_obs_term_dims"] = [196]
    assert TC.validate_action_ball_training_authorization(contract) is True

    wrong_width = dict(contract)
    wrong_width["actor_obs_total_dim"] = 195
    with pytest.raises(ValueError, match="does not match"):
        TC.validate_action_ball_training_authorization(wrong_width)

    identity = _action_set_identity(teacher_start=True)
    kwargs = {
        "actor_obs_contract": (
            "action_ball_table_pose_twist_heading_task_teacher_start_n2"
        ),
        "actor_obs_width": 196,
        "manifest_path": identity["manifest_path"],
        "manifest_sha256": identity["manifest_sha256"],
        "scope": "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": identity["ordered_action_ids"],
        "ordered_action_uids": identity["ordered_action_uids"],
        "experiment_name": identity["experiment_name"],
    }
    assert (
        TC.validate_action_ball_action_set_runtime_identity(identity, **kwargs)
        == identity
    )


def test_action_ball_formal_contract_crossbinds_manifest_and_preflight_order():
    contract = _action_ball_formal_schema3_contract()
    contract["action_ball_training"]["preflight"]["manifest"][
        "file_sha256"
    ] = "e" * 64
    with pytest.raises(ValueError, match="disagrees with live runtime"):
        TC.validate_schema3_contract_structure(contract)

    contract = _action_ball_formal_schema3_contract()
    contract["action_ball_training"]["preflight"]["action_bindings"].reverse()
    with pytest.raises(ValueError, match="bindings disagree"):
        TC.validate_schema3_contract_structure(contract)


def test_action_ball_metadata_replaces_donor_identity_only_for_exact_formal_lineage():
    formal = _action_ball_formal_schema3_contract()
    identity = formal["action_ball_training"]["action_set_identity"]
    metadata = {
        key: "laundered-donor-value"
        for key in TC.ACTION_BALL_ACTION_SET_METADATA_KEYS
    }
    metadata.update(
        {
            "action_set_profile_id": "laundered-legacy-alias",
            "action_set_contract_sha256": "laundered-legacy-alias",
        }
    )
    assert TC.bind_action_ball_action_set_metadata(
        metadata, formal, lineage_exact=True
    )
    assert metadata["action_ball_profile_id"] == identity["profile_id"]
    assert json.loads(metadata["action_ball_action_order"]) == identity[
        "ordered_action_ids"
    ]
    assert json.loads(
        metadata["action_ball_ordered_action_uids"]
    ) == identity["ordered_action_uids"]
    assert (
        metadata["action_ball_manifest_sha256"]
        == identity["manifest_sha256"]
    )
    assert (
        metadata["action_ball_action_set_contract_sha256"]
        == identity["contract_sha256"]
    )
    assert "action_set_profile_id" not in metadata
    assert "action_set_contract_sha256" not in metadata

    diagnostic_metadata = {
        key: "laundered-donor-value"
        for key in TC.ACTION_BALL_ACTION_SET_METADATA_KEYS
    }
    assert not TC.bind_action_ball_action_set_metadata(
        diagnostic_metadata,
        _action_ball_diagnostic_schema3_contract(),
        lineage_exact=False,
    )
    assert not set(TC.ACTION_BALL_ACTION_SET_METADATA_KEYS).intersection(
        diagnostic_metadata
    )


def test_action_ball_launch_claim_binds_registry_source_and_exact_identity(
    tmp_path,
):
    checkout = tmp_path / "checkout"
    source = checkout / TC.ACTION_BALL_ACTION_SET_SOURCE_PATH
    source.parent.mkdir(parents=True)
    identity = _action_set_identity()
    literal_keys = (
        "profile_id",
        "expected_n",
        "scope",
        "mobility_mode",
        "ordered_action_ids",
        "ordered_action_uids",
        "order_uid_digest_sha256",
        "manifest_path",
        "manifest_sha256",
        "experiment_name",
    )
    literal_row = {key: identity[key] for key in literal_keys}
    profile_policy = {
        "expected_n": identity["expected_n"],
        "scope": identity["scope"],
        "mobility_mode": identity["mobility_mode"],
        "required_action_ids": identity["ordered_action_ids"],
        "retired_action_ids": ["retired_action"],
    }
    source.write_text(
        (
            "ACTION_SET_PROFILE_POLICIES = "
            f"{ {identity['profile_id']: profile_policy}!r}\n"
            "ACTION_SET_CONTRACTS = "
            f"{ {identity['profile_id']: literal_row}!r}\n"
        ),
        encoding="utf-8",
    )
    code_identity = {
        key: value
        for key, value in identity.items()
        if key not in ("contract_source_path", "contract_source_sha256")
    }
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {
        "source_checkout": str(checkout),
        "launch_profile": identity["profile_id"],
        "ordered_action_ids": identity["ordered_action_ids"],
        "action_set_contract": code_identity,
        "manifest": {
            "path": identity["manifest_path"],
            "sha256": identity["manifest_sha256"],
        },
        "runtime_code_sha256": {
            TC.ACTION_BALL_ACTION_SET_SOURCE_PATH: source_sha,
        },
    }
    base_contract = {
        "schema_version": 1,
        "kind": "action_ball_python_nosite_argv_contract_v1",
        "bootstrap": {
            "path": str(tmp_path / "bootstrap.py"),
            "byte_count": 1,
            "sha256": "b" * 64,
        },
        "entrypoint": {
            "path": str(tmp_path / "entrypoint.py"),
            "byte_count": 1,
            "sha256": "e" * 64,
        },
        "import_roots": [
            {
                "path": str(tmp_path / "imports"),
                "tree_sha256": "f" * 64,
                "file_count": 1,
                "total_size_bytes": 1,
            }
        ],
        "entrypoint_argv": [
            "train-entrypoint",
        ],
    }
    base_raw = TC._canonical_action_ball_json(base_contract).encode(
        "utf-8"
    )
    import base64

    base_argv = [
        "/usr/bin/python3",
        "-I",
        "-B",
        "-S",
        "-c",
        "trampoline",
        str(tmp_path / "bootstrap.py"),
        "b" * 64,
        hashlib.sha256(base_raw).hexdigest(),
        base64.b64encode(base_raw).decode("ascii"),
    ]
    payload["argv_without_launch_claim"] = base_argv
    payload["isolated_training_entrypoint"] = {
        "nosite_argv_contract": base_contract,
        "nosite_argv_contract_sha256": base_argv[8],
    }
    claim_sha = TC._action_ball_canonical_sha256(payload)
    no_site_contract = json.loads(json.dumps(base_contract))
    no_site_contract["entrypoint_argv"].append(
        f"++training_launch_claim_sha256={claim_sha}"
    )
    no_site_raw = TC._canonical_action_ball_json(no_site_contract).encode(
        "utf-8"
    )
    argv = list(base_argv)
    argv[8] = hashlib.sha256(no_site_raw).hexdigest()
    argv[9] = base64.b64encode(no_site_raw).decode("ascii")
    claim = {
        "schema_version": TC.ACTION_BALL_LAUNCH_CLAIM_SCHEMA_VERSION,
        "kind": TC.ACTION_BALL_LAUNCH_CLAIM_KIND,
        "launch_claim_sha256": claim_sha,
        "canonical_payload": payload,
        "argv": argv,
        "confirmation_claim_sha256": claim_sha,
    }
    claim_path = tmp_path / "launch_claim.json"
    claim_path.write_text(
        json.dumps(claim, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = TC.load_action_ball_action_set_identity_from_launch_claim(
        claim_path,
        expected_claim_sha256=claim_sha,
        actual_argv=tuple(argv),
    )
    assert loaded["contract_source_sha256"] == source_sha
    assert loaded["ordered_action_ids"] == identity["ordered_action_ids"]

    source.write_text("ACTION_SET_CONTRACTS = {'drift': {}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source bytes differ"):
        TC.load_action_ball_action_set_identity_from_launch_claim(
            claim_path,
            expected_claim_sha256=claim_sha,
            actual_argv=tuple(argv),
        )


def test_action_ball_launch_claim_rejects_actual_kernel_argv_and_policy_drift(
    tmp_path,
):
    checkout = tmp_path / "checkout"
    source = checkout / TC.ACTION_BALL_ACTION_SET_SOURCE_PATH
    source.parent.mkdir(parents=True)
    identity = _action_set_identity()
    literal_keys = (
        "profile_id",
        "expected_n",
        "scope",
        "mobility_mode",
        "ordered_action_ids",
        "ordered_action_uids",
        "order_uid_digest_sha256",
        "manifest_path",
        "manifest_sha256",
        "experiment_name",
    )
    literal_row = {key: identity[key] for key in literal_keys}
    policy = {
        "expected_n": identity["expected_n"],
        "scope": identity["scope"],
        "mobility_mode": identity["mobility_mode"],
        "required_action_ids": identity["ordered_action_ids"],
        "retired_action_ids": [],
    }
    source.write_text(
        (
            f"ACTION_SET_PROFILE_POLICIES = "
            f"{ {identity['profile_id']: policy}!r}\n"
            f"ACTION_SET_CONTRACTS = "
            f"{ {identity['profile_id']: literal_row}!r}\n"
        ),
        encoding="utf-8",
    )
    code_identity = {
        key: value
        for key, value in identity.items()
        if key not in ("contract_source_path", "contract_source_sha256")
    }
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {
        "source_checkout": str(checkout),
        "launch_profile": identity["profile_id"],
        "ordered_action_ids": identity["ordered_action_ids"],
        "action_set_contract": code_identity,
        "manifest": {
            "path": identity["manifest_path"],
            "sha256": identity["manifest_sha256"],
        },
        "runtime_code_sha256": {
            TC.ACTION_BALL_ACTION_SET_SOURCE_PATH: source_sha,
        },
    }
    base_no_site = {
        "schema_version": 1,
        "kind": "action_ball_python_nosite_argv_contract_v1",
        "bootstrap": {"path": "/b", "byte_count": 1, "sha256": "b" * 64},
        "entrypoint": {"path": "/e", "byte_count": 1, "sha256": "e" * 64},
        "import_roots": [
            {
                "path": "/i",
                "tree_sha256": "f" * 64,
                "file_count": 1,
                "total_size_bytes": 1,
            }
        ],
        "entrypoint_argv": ["train-entrypoint"],
    }
    import base64

    base_raw = TC._canonical_action_ball_json(base_no_site).encode("utf-8")
    base_argv = [
        "/python",
        "-I",
        "-B",
        "-S",
        "-c",
        "trampoline",
        "/b",
        "b" * 64,
        hashlib.sha256(base_raw).hexdigest(),
        base64.b64encode(base_raw).decode("ascii"),
    ]
    payload["argv_without_launch_claim"] = base_argv
    payload["isolated_training_entrypoint"] = {
        "nosite_argv_contract": base_no_site,
        "nosite_argv_contract_sha256": base_argv[8],
    }
    claim_sha = TC._action_ball_canonical_sha256(payload)
    no_site = json.loads(json.dumps(base_no_site))
    no_site["entrypoint_argv"].append(
        f"++training_launch_claim_sha256={claim_sha}"
    )
    raw = TC._canonical_action_ball_json(no_site).encode("utf-8")
    argv = list(base_argv)
    argv[8] = hashlib.sha256(raw).hexdigest()
    argv[9] = base64.b64encode(raw).decode("ascii")
    claim = {
        "schema_version": TC.ACTION_BALL_LAUNCH_CLAIM_SCHEMA_VERSION,
        "kind": TC.ACTION_BALL_LAUNCH_CLAIM_KIND,
        "launch_claim_sha256": claim_sha,
        "canonical_payload": payload,
        "argv": argv,
        "confirmation_claim_sha256": claim_sha,
    }
    claim_path = tmp_path / "launch_claim.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    drifted = list(argv)
    drifted[0] = "/other-python"
    with pytest.raises(ValueError, match="actual kernel argv"):
        TC.load_action_ball_action_set_identity_from_launch_claim(
            claim_path,
            expected_claim_sha256=claim_sha,
            actual_argv=drifted,
        )

    policy["required_action_ids"] = list(
        reversed(identity["ordered_action_ids"])
    )
    source.write_text(
        (
            f"ACTION_SET_PROFILE_POLICIES = "
            f"{ {identity['profile_id']: policy}!r}\n"
            f"ACTION_SET_CONTRACTS = "
            f"{ {identity['profile_id']: literal_row}!r}\n"
        ),
        encoding="utf-8",
    )
    payload["runtime_code_sha256"][
        TC.ACTION_BALL_ACTION_SET_SOURCE_PATH
    ] = hashlib.sha256(source.read_bytes()).hexdigest()
    payload["isolated_training_entrypoint"][
        "nosite_argv_contract"
    ] = base_no_site
    payload["isolated_training_entrypoint"][
        "nosite_argv_contract_sha256"
    ] = base_argv[8]
    claim_sha = TC._action_ball_canonical_sha256(payload)
    no_site = json.loads(json.dumps(base_no_site))
    no_site["entrypoint_argv"].append(
        f"++training_launch_claim_sha256={claim_sha}"
    )
    raw = TC._canonical_action_ball_json(no_site).encode("utf-8")
    argv[8] = hashlib.sha256(raw).hexdigest()
    argv[9] = base64.b64encode(raw).decode("ascii")
    claim.update(
        {
            "launch_claim_sha256": claim_sha,
            "canonical_payload": payload,
            "argv": argv,
            "confirmation_claim_sha256": claim_sha,
        }
    )
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    with pytest.raises(ValueError, match="profile policy"):
        TC.load_action_ball_action_set_identity_from_launch_claim(
            claim_path,
            expected_claim_sha256=claim_sha,
            actual_argv=argv,
        )


def test_action_ball_diagnostic_evaluator_rejects_unknown_rights():
    contract = _action_ball_diagnostic_schema3_contract()
    contract["action_ball_training"]["runtime"]["evaluator_authority"][
        "can_publish_formal_score"
    ] = True
    with pytest.raises(ValueError, match="unknown fields"):
        TC.validate_schema3_contract_structure(contract)


def test_action_ball_authorization_block_cannot_be_stripped_or_orphaned():
    stripped = _action_ball_formal_schema3_contract()
    stripped.pop("action_ball_training")
    with pytest.raises(ValueError, match="missing the mandatory"):
        TC.validate_schema3_contract(stripped)

    orphaned = _action_ball_formal_schema3_contract()
    orphaned["target_mode"] = "solved"
    orphaned["actor_obs_contract"] = "deploy_parity_face179"
    with pytest.raises(ValueError, match="requires target_mode='action_ball'"):
        TC.validate_schema3_contract_structure(orphaned)

    mismatched_actor = _action_ball_formal_schema3_contract()
    mismatched_actor["actor_obs_contract"] = "action_ball_n05"
    with pytest.raises(ValueError, match=r"action_ball_n<N>"):
        TC.validate_schema3_contract_structure(mismatched_actor)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("action_ball_training", "schema_version"),
            1.0,
            "schema_version must be integer 1",
        ),
        (
            (
                "action_ball_training",
                "authorization",
                "diagnostic_unauthorized",
            ),
            1.0,
            "must be an exact boolean",
        ),
        (
            (
                "action_ball_training",
                "authorization",
                "formal_judge_prohibited",
            ),
            "true",
            "must be an exact boolean",
        ),
        (
            ("action_ball_training", "runtime", "diagnostic_unauthorized"),
            "1",
            "must be an exact boolean",
        ),
        (
            (
                "action_ball_training",
                "runtime",
                "evaluator_authority",
                "formal_authority_available",
            ),
            0,
            "must be an exact boolean",
        ),
        (
            (
                "action_ball_training",
                "motion_admission",
                "training_authorized",
            ),
            0.0,
            "must be an exact boolean",
        ),
    ],
)
def test_action_ball_authorization_rejects_float_and_truthy_lookalikes(
    path, value, message
):
    contract = json.loads(json.dumps(_action_ball_diagnostic_schema3_contract()))
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        TC.validate_schema3_contract_structure(contract)


def test_action_ball_diagnostic_export_metadata_is_exact_zero_and_non_bookable():
    contract = _action_ball_diagnostic_schema3_contract()
    metadata = {
        "training_contract_exact": "1",
        TC.ACTION_BALL_DIAGNOSTIC_METADATA_KEY: "donor-value",
        TC.FORMAL_EVIDENCE_BOOKABLE_METADATA_KEY: "1",
    }
    assert (
        TC.bind_action_ball_diagnostic_metadata(
            metadata,
            contract,
            lineage_exact=False,
        )
        is True
    )
    assert metadata["training_contract_exact"] == "0"
    assert metadata[TC.ACTION_BALL_DIAGNOSTIC_METADATA_KEY] == "1"
    assert metadata[TC.FORMAL_EVIDENCE_BOOKABLE_METADATA_KEY] == "0"

    with pytest.raises(
        ValueError,
        match="cannot claim training_contract_lineage_exact=1",
    ):
        TC.bind_action_ball_diagnostic_metadata(
            {},
            contract,
            lineage_exact=True,
        )
    with pytest.raises(ValueError, match="exact boolean"):
        TC.bind_action_ball_diagnostic_metadata(
            {},
            contract,
            lineage_exact=0,
        )


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


def test_schema3_checkpoint_binding_requires_explicit_zero_or_one_lineage():
    digest = "a" * 64
    checkpoint = {
        "infos": {
            TC.CHECKPOINT_CONTRACT_SCHEMA_KEY: 3,
            TC.CHECKPOINT_CONTRACT_SHA_KEY: digest,
        }
    }
    with pytest.raises(ValueError, match="explicitly declare"):
        TC.require_checkpoint_contract_binding(
            checkpoint,
            schema=3,
            sha256=digest,
            require_lineage_exact=False,
        )


@pytest.mark.parametrize("invalid_lineage", [1.0, "1", True, False, 2, -1])
def test_checkpoint_binding_rejects_non_plain_integer_lineage(invalid_lineage):
    digest = "a" * 64
    checkpoint = {
        "infos": {
            TC.CHECKPOINT_CONTRACT_SCHEMA_KEY: 3,
            TC.CHECKPOINT_CONTRACT_SHA_KEY: digest,
            TC.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY: invalid_lineage,
        }
    }
    with pytest.raises(ValueError, match="plain integer 0/1"):
        TC.require_checkpoint_contract_binding(
            checkpoint,
            schema=3,
            sha256=digest,
            require_lineage_exact=False,
        )


@pytest.mark.parametrize("invalid_schema", [3.0, "3", True])
def test_checkpoint_binding_rejects_non_plain_integer_schema(invalid_schema):
    digest = "a" * 64
    checkpoint = {
        "infos": {
            TC.CHECKPOINT_CONTRACT_SCHEMA_KEY: invalid_schema,
            TC.CHECKPOINT_CONTRACT_SHA_KEY: digest,
            TC.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY: 0,
        }
    }
    with pytest.raises(ValueError, match="not bound"):
        TC.require_checkpoint_contract_binding(
            checkpoint,
            schema=3,
            sha256=digest,
            require_lineage_exact=False,
        )


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
    assert "bind_action_ball_diagnostic_metadata" in standalone
    assert "bind_action_ball_diagnostic_metadata" in exporter
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
    assert "contract_lineage_exact = _action_ball_contract_lineage_exact(" in train
    assert "source_lineage_exact=ckpt is None" in train
    assert "diagnostic_unauthorized=action_ball_diagnostic_unauthorized" in train
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
