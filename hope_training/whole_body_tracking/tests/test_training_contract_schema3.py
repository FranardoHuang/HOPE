"""Dependency-light regression tests for the schema-3 training/export contract."""

from __future__ import annotations

import importlib.util
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


def test_actor_leg_ref_mask_fact_is_only_when_true_and_survives_partial_wrapping():
    import functools

    # Unmasked env (no observations cfg at all): key absent, ordering untouched.
    facts = TC.runtime_execution_facts(_env(), _ActorContract())
    assert "actor_leg_ref_mask" not in facts

    def generated_commands_actor_leg_masked(env, command_name):  # name-based detection leg
        raise NotImplementedError

    env = _env()
    env.cfg.observations = SimpleNamespace(
        policy=SimpleNamespace(
            command=SimpleNamespace(func=generated_commands_actor_leg_masked)
        )
    )
    facts = TC.runtime_execution_facts(env, _ActorContract())
    assert facts["actor_leg_ref_mask"] is True
    assert tuple(k for k in facts if k != "actor_leg_ref_mask") == TC.RUNTIME_EXECUTION_KEYS

    # Marker-attribute detection must survive a rename AND a functools.partial wrapper.
    def renamed_masked_command(env, command_name):
        raise NotImplementedError

    renamed_masked_command.actor_leg_ref_mask = True
    env.cfg.observations.policy.command.func = functools.partial(renamed_masked_command)
    facts = TC.runtime_execution_facts(env, _ActorContract())
    assert facts["actor_leg_ref_mask"] is True


@pytest.mark.parametrize("donor_value", [None, "0", "1"])
def test_actor_leg_ref_mask_metadata_is_checkpoint_authoritative_and_only_when_true(donor_value):
    metadata = {"keep": "value"}
    if donor_value is not None:
        metadata["actor_leg_ref_mask"] = donor_value

    TC.bind_actor_leg_ref_mask_metadata(metadata, {"actor_leg_ref_mask": True})
    assert metadata == {"keep": "value", "actor_leg_ref_mask": "1"}

    # Unmasked and legacy checkpoints clear any stale donor value. False is deliberately treated
    # as absence here; the schema validator separately rejects serializing False in a contract.
    TC.bind_actor_leg_ref_mask_metadata(metadata, {})
    assert metadata == {"keep": "value"}
    metadata["actor_leg_ref_mask"] = "1"
    TC.bind_actor_leg_ref_mask_metadata(metadata, None)
    assert metadata == {"keep": "value"}


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
    contract = _schema3_contract()
    contract["actor_leg_ref_mask"] = invalid
    with pytest.raises(ValueError, match="actor_leg_ref_mask must be true when present"):
        TC.validate_schema3_contract_structure(contract)

    contract["actor_leg_ref_mask"] = True
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
