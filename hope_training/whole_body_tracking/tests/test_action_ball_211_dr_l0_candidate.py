"""Static fail-closed contract for the fresh A211/C211 DR-L0 leaves."""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "cfg/task"
REPO_ROOT = ROOT.parents[1]
MANIFEST = (
    REPO_ROOT
    / "configs/action_ball_n1_measured_20260803"
    / "action_ball_211_dr_l0_learnability_candidate.v1.json"
)
TRAIN = ROOT / "scripts/train.py"
TRAINING_CONTRACT = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TC = _load_module("action_ball_dr_l0_training_contract_under_test", TRAINING_CONTRACT)


def _load_train_module():
    pytest.importorskip("hydra")
    source_root = str((ROOT / "source/whole_body_tracking").resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    return _load_module("action_ball_dr_l0_train_under_test", TRAIN)

LEAVES = {
    "A": (
        "HOPEPingPongActionBallA211VendorV2N1DRL0Learnability",
        "HOPEPingPongActionBallA211VendorV2N1Learnability",
        "action_ball_a211",
    ),
    "C": (
        "HOPEPingPongActionBallC211VendorV2N1DRL0Learnability",
        "HOPEPingPongActionBallC211VendorV2N1Learnability",
        "action_ball_c211",
    ),
}


def _raw_task(name: str) -> dict:
    return yaml.safe_load((TASK_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def _domain_rand_whitelist() -> tuple[str, ...]:
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"), filename=str(TRAIN))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_DOMAIN_RAND_KEYS"
            for target in node.targets
        )
    )
    return tuple(ast.literal_eval(assignment.value))


@pytest.mark.parametrize("side", ("A", "C"))
def test_dr_l0_is_a_new_leaf_and_does_not_mutate_retained_dr_parent(side: str):
    leaf_name, parent_name, _actor_contract = LEAVES[side]
    leaf = _raw_task(leaf_name)
    parent = _raw_task(parent_name)

    assert leaf["name"] == leaf_name
    assert leaf["defaults"] == [f"{parent_name}@_here_", "_self_"]
    assert leaf["domain_rand"] == {
        "stable_ready_plant": True,
        "startup_physics_material": False,
        "startup_joint_default_pos": False,
        "policy_observation_corruption": False,
    }
    for candidate_only_key in (
        "startup_physics_material",
        "startup_joint_default_pos",
        "policy_observation_corruption",
    ):
        assert candidate_only_key not in parent["domain_rand"]


def test_manifest_spells_exact_adopted_state_and_all_deferred_axes():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "BOUND_FRESH_DIAGNOSTIC_LAUNCH"
    assert manifest["shared_finalizer"] == {
        "absent_joint_offset_decoder": "action_ball_dr_l0_exact_zero_decoder",
        "status": "IMPLEMENTED_HOST_TESTED",
    }
    assert manifest["fresh_lineage_required"] is True
    assert manifest["authorization"] == {
        "diagnostic_unauthorized": True,
        "formal": False,
        "hardware": False,
        "main_setting": False,
        "promotion": False,
        "ready": False,
        "runtime_launch": True,
    }
    assert manifest["exact_leaf_fields"] == {
        "task.domain_rand.policy_observation_corruption": False,
        "task.domain_rand.stable_ready_plant": True,
        "task.domain_rand.startup_joint_default_pos": False,
        "task.domain_rand.startup_physics_material": False,
    }
    assert manifest["required_post_finalizer_state"] == {
        "actions.joint_pos.control_step_action_delay_max": 0,
        "actions.joint_pos.control_step_action_delay_min": 0,
        "events.add_joint_default_pos": None,
        "events.base_com": None,
        "events.combined_push": None,
        "events.combined_push_sweep": None,
        "events.force_push": None,
        "events.force_push_sweep": None,
        "events.physics_material": None,
        "events.push_robot": None,
        "events.randomize_link_mass": None,
        "events.randomize_pd_gains": None,
        "force_push.enable": False,
        "motion.joint_position_range": [0.0, 0.0],
        "motion.pose_range_all_six_axes": [0.0, 0.0],
        "motion.stand_start_yaw_range": [0.0, 0.0],
        "motion.velocity_range_all_six_axes": [0.0, 0.0],
        "observations.policy.enable_corruption": False,
        "push.enable": False,
        "lateral_perturbation_runtime_spec": None,
        "racket.action_ball_target_observation_noise": False,
        "racket.all_target_transport_mutations": 0.0,
    }
    assert set(manifest["deferred_axes"]) == {
        "action_delay",
        "armature_joint_friction_torque_limit_randomization",
        "joint_default_position",
        "link_mass",
        "observation_corruption",
        "pd_gain",
        "push",
        "reset_state_noise",
        "robot_physics_material",
        "task_or_target_noise",
        "torso_com",
    }
    assert manifest["policy_exploration"]["changed_by_dr_l0"] is False
    assert manifest["runtime_integration_blockers"] == []
    resolved = manifest["resolved_finalizer_contract"]
    assert resolved == {
        "contract_sha256": TC.action_ball_dr_l0_contract_sha256(),
        "hard_contract_identity": "action_ball_dr_l0_exact_all_off_v1",
        "source": (
            "whole_body_tracking.utils.training_contract."
            "action_ball_dr_l0_contract_payload"
        ),
    }
    assert TC.action_ball_dr_l0_contract_payload()["identity"] == resolved[
        "hard_contract_identity"
    ]


@pytest.mark.parametrize("side", ("A", "C"))
def test_hydra_candidate_preserves_a_c_mdp_and_fixed_noise_axes(side: str):
    hydra = pytest.importorskip("hydra")
    leaf_name, _parent_name, actor_contract = LEAVES[side]
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "cfg").resolve()),
    ):
        task = hydra.compose(
            config_name="train", overrides=[f"task={leaf_name}"]
        ).task

    assert task.name == leaf_name
    assert task.actor_obs_contract == actor_contract
    assert task.env.num_envs == 4096
    assert task.domain_rand.stable_ready_plant is True
    assert task.domain_rand.startup_physics_material is False
    assert task.domain_rand.startup_joint_default_pos is False
    assert task.domain_rand.policy_observation_corruption is False
    assert task.actions.control_step_action_delay_min == 0
    assert task.actions.control_step_action_delay_max == 0
    assert task.push.enable is False
    assert task.force_push.enable is False
    assert tuple(task.motion.joint_position_range) == (0.0, 0.0)
    assert tuple(task.motion.stand_start_yaw_range) == (0.0, 0.0)
    for reset_group in (task.motion.pose_range, task.motion.velocity_range):
        assert set(reset_group) == {"x", "y", "z", "roll", "pitch", "yaw"}
        assert all(tuple(reset_group[axis]) == (0.0, 0.0) for axis in reset_group)
    assert task.racket.action_ball_target_source == (
        "online_solver" if side == "A" else "direct_ball"
    )
    assert task.racket.action_ball_initial_center_single_question is True
    assert task.racket.action_ball_target_observation_noise is False
    for field in (
        "achieved_target_mix_prob",
        "midswing_resample_prob",
        "target_delay_steps",
        "target_jitter_pos_per_s",
        "target_jitter_vel_per_s",
        "target_noise_white",
        "target_noise_ar1_sigma",
        "target_dropout_prob",
        "target_post_strike_dropout_s",
        "target_bias_per_swing",
    ):
        assert task.racket[field] == pytest.approx(0.0)


def test_shared_finalizer_keys_and_manifest_resolve_the_same_launch_contract():
    """The bound diagnostic manifest must name the live finalizer bytes."""

    candidate_keys = {
        "startup_physics_material",
        "startup_joint_default_pos",
        "policy_observation_corruption",
    }
    assert candidate_keys.issubset(_domain_rand_whitelist())

    source = TRAIN.read_text(encoding="utf-8")
    assert "action_ball_dr_l0_exact_all_off_v1" in source
    assert "ACTION_BALL_DR_L0_ZERO_DECODER_SOURCE" in source
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["authorization"]["runtime_launch"] is True
    assert manifest["runtime_integration_blockers"] == []
    assert manifest["resolved_finalizer_contract"]["contract_sha256"] == (
        TC.action_ball_dr_l0_contract_sha256()
    )
    train = _load_train_module()
    assert train._action_ball_dr_l0_contract_payload() == (
        TC.action_ball_dr_l0_contract_payload()
    )


def _finalizer_env():
    sentinel = object()
    zero_ranges = {
        axis: (0.0, 0.0)
        for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    return types.SimpleNamespace(
        events=types.SimpleNamespace(
            physics_material=sentinel,
            add_joint_default_pos=sentinel,
            base_com=None,
            randomize_link_mass=None,
            randomize_pd_gains=None,
            push_robot=None,
            force_push=None,
            force_push_sweep=None,
            combined_push=None,
            combined_push_sweep=None,
        ),
        observations=types.SimpleNamespace(
            policy=types.SimpleNamespace(enable_corruption=True)
        ),
        actions=types.SimpleNamespace(
            joint_pos=types.SimpleNamespace(
                control_step_action_delay_min=0,
                control_step_action_delay_max=0,
            )
        ),
        commands=types.SimpleNamespace(
            motion=types.SimpleNamespace(
                joint_position_range=(0.0, 0.0),
                stand_start_yaw_range=(0.0, 0.0),
                pose_range=deepcopy(zero_ranges),
                velocity_range=deepcopy(zero_ranges),
            ),
            racket_target=types.SimpleNamespace(
                achieved_target_mix_prob=0.0,
                midswing_resample_prob=0.0,
                target_delay_steps=0,
                target_jitter_pos_per_s=0.0,
                target_jitter_vel_per_s=0.0,
                target_noise_white=0.0,
                target_noise_ar1_sigma=0.0,
                target_dropout_prob=0.0,
                target_post_strike_dropout_s=0.0,
                target_bias_per_swing=0.0,
                action_ball_target_observation_noise=False,
            ),
        ),
        push=types.SimpleNamespace(enable=False),
        force_push=types.SimpleNamespace(enable=False),
    )


@pytest.mark.parametrize("side", ("A", "C"))
def test_composed_a_c_dr_l0_finalizer_reaches_exact_all_off_state(side: str):
    train = _load_train_module()
    hydra = pytest.importorskip("hydra")
    leaf_name, _parent_name, _actor_contract = LEAVES[side]
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "cfg").resolve()),
    ):
        task = hydra.compose(
            config_name="train", overrides=[f"task={leaf_name}"]
        ).task

    env_cfg = _finalizer_env()
    applied = []
    assert train._apply_action_ball_dr_l0_finalizer(
        env_cfg, task.domain_rand, applied
    ) is True
    assert env_cfg.events.physics_material is None
    assert env_cfg.events.add_joint_default_pos is None
    assert env_cfg.events.base_com is None
    assert env_cfg.events.randomize_link_mass is None
    assert env_cfg.events.randomize_pd_gains is None
    assert env_cfg.events.push_robot is None
    assert env_cfg.events.force_push is None
    assert env_cfg.events.force_push_sweep is None
    assert env_cfg.observations.policy.enable_corruption is False
    assert len(applied) == 1
    assert "action_ball_dr_l0_exact_all_off_v1" in applied[0]
    assert getattr(
        env_cfg, train._ACTION_BALL_DR_L0_RUNTIME_ATTR
    ) == train._action_ball_dr_l0_contract_payload()


@pytest.mark.parametrize(
    "domain_rand",
    (
        {
            "stable_ready_plant": True,
            "startup_physics_material": False,
        },
        {
            "stable_ready_plant": True,
            "startup_physics_material": False,
            "startup_joint_default_pos": True,
            "policy_observation_corruption": False,
        },
        {
            "stable_ready_plant": False,
            "startup_physics_material": False,
            "startup_joint_default_pos": False,
            "policy_observation_corruption": False,
        },
    ),
)
def test_dr_l0_rejects_partial_enabled_or_unstable_mixed_configs(domain_rand):
    train = _load_train_module()
    with pytest.raises(train._OverrideError, match="ActionBall DR-L0"):
        train._apply_action_ball_dr_l0_finalizer(
            _finalizer_env(), domain_rand, []
        )


@pytest.mark.parametrize(
    "mutation",
    ("push", "reset_state", "target_transport", "action_delay"),
)
def test_dr_l0_finalizer_refuses_composed_non_l0_axes(mutation: str):
    train = _load_train_module()
    env_cfg = _finalizer_env()
    if mutation == "push":
        env_cfg.events.push_robot = object()
        env_cfg.push.enable = True
    elif mutation == "reset_state":
        env_cfg.commands.motion.pose_range["x"] = (-0.1, 0.1)
    elif mutation == "target_transport":
        env_cfg.commands.racket_target.target_dropout_prob = 0.01
    else:
        env_cfg.actions.joint_pos.control_step_action_delay_max = 1
    dr = {
        "stable_ready_plant": True,
        "startup_physics_material": False,
        "startup_joint_default_pos": False,
        "policy_observation_corruption": False,
    }
    with pytest.raises(train._OverrideError, match="DR-L0"):
        train._apply_action_ball_dr_l0_finalizer(env_cfg, dr, [])


@pytest.mark.parametrize("side", ("A", "C"))
def test_normal_a_c_leaf_retains_material_joint_offset_and_corruption(side: str):
    train = _load_train_module()
    hydra = pytest.importorskip("hydra")
    _leaf_name, parent_name, _actor_contract = LEAVES[side]
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "cfg").resolve()),
    ):
        task = hydra.compose(
            config_name="train", overrides=[f"task={parent_name}"]
        ).task

    env_cfg = _finalizer_env()
    material = env_cfg.events.physics_material
    joint_offset = env_cfg.events.add_joint_default_pos
    assert train._apply_action_ball_dr_l0_finalizer(
        env_cfg, task.domain_rand, []
    ) is False
    assert env_cfg.events.physics_material is material
    assert env_cfg.events.add_joint_default_pos is joint_offset
    assert env_cfg.observations.policy.enable_corruption is True
    assert not hasattr(env_cfg, train._ACTION_BALL_DR_L0_RUNTIME_ATTR)


_JOINT_NAMES = [f"joint_{index:02d}" for index in range(31)]


def _shared_ready_bootstrap(decoder: dict) -> dict:
    ready = [0.1] * 31
    return {
        "schema_version": 1,
        "kind": TC.ACTION_BALL_POLICY_BOOTSTRAP_KIND,
        "action_count": 1,
        "action_order": ["take_061_unit04_bh"],
        "joint_names": list(_JOINT_NAMES),
        "ready_source": {
            "semantics": "motion.joint_pos[motion.seg_start[action_slot]]",
            "canonical_ready_sha256": "",
            "canonical_ready_fk_sha256": "",
            "motion_sha256_per_action": ["a" * 64],
            "shared_ready_joint_pos": ready,
            "shared_ready_joint_pos_sha256": TC.action_ball_shared_ready_sha256(
                action_order=["take_061_unit04_bh"],
                joint_names=_JOINT_NAMES,
                shared_ready_joint_pos=ready,
            ),
        },
        "decoder": decoder,
        "initialization": {
            "fresh_only": True,
            "resume_overwrite_prohibited": True,
            "output_layer_weight": "zeros",
            "output_layer_bias": "decoder.normalized_bias",
            "init_noise_std": 0.02,
            "sigma_envelope": 4.0,
        },
        "hard_inner_guard": {
            "limit_source": "articulation.data.joint_pos_limits",
            "margin_rad": 0.0,
            "margin_fraction": 0.02,
            "hard_lower": [-2.0] * 31,
            "hard_upper": [2.0] * 31,
            "hard_inner_lower": [-1.92] * 31,
            "hard_inner_upper": [1.92] * 31,
        },
    }


def _dr_l0_decoder() -> dict:
    return {
        "semantics": "q_des=default_joint_pos+action_scale*action",
        "use_default_offset": True,
        "default_joint_pos": [0.0] * 31,
        "action_scale": [0.25] * 31,
        "normalized_bias": [0.4] * 31,
        "startup_offset_delta_source": TC.ACTION_BALL_DR_L0_ZERO_DECODER_SOURCE,
        "startup_offset_delta_identity": (
            TC.action_ball_dr_l0_zero_decoder_identity(joint_names=_JOINT_NAMES)
        ),
        "startup_offset_delta_lower": [0.0] * 31,
        "startup_offset_delta_upper": [0.0] * 31,
    }


def test_bootstrap_validator_accepts_reproducible_dr_l0_zero_decoder_identity():
    decoder = _dr_l0_decoder()
    validated = TC.validate_action_ball_policy_bootstrap(
        _shared_ready_bootstrap(decoder), expected_action_count=1
    )
    identity = validated["decoder"]["startup_offset_delta_identity"]
    assert identity["kind"] == TC.ACTION_BALL_DR_L0_ZERO_DECODER_KIND
    assert len(identity["content_sha256"]) == 64
    assert identity == TC.action_ball_dr_l0_zero_decoder_identity(
        joint_names=_JOINT_NAMES
    )


@pytest.mark.parametrize("mutation", ("content_sha", "nonzero_envelope"))
def test_bootstrap_validator_rejects_forged_or_nonzero_dr_l0_decoder(mutation: str):
    decoder = _dr_l0_decoder()
    if mutation == "content_sha":
        decoder["startup_offset_delta_identity"]["content_sha256"] = "f" * 64
    else:
        decoder["startup_offset_delta_upper"][7] = 0.01
    with pytest.raises(ValueError, match="DR-L0"):
        TC.validate_action_ball_policy_bootstrap(_shared_ready_bootstrap(decoder))


def test_train_bootstrap_source_requires_event_absence_only_for_dr_l0(monkeypatch):
    train = _load_train_module()
    package = types.ModuleType("whole_body_tracking")
    package.__path__ = []
    utils = types.ModuleType("whole_body_tracking.utils")
    utils.__path__ = []
    utils.training_contract = TC
    package.utils = utils
    monkeypatch.setitem(sys.modules, "whole_body_tracking", package)
    monkeypatch.setitem(sys.modules, "whole_body_tracking.utils", utils)
    monkeypatch.setitem(
        sys.modules, "whole_body_tracking.utils.training_contract", TC
    )
    zero_env = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            events=types.SimpleNamespace(add_joint_default_pos=None)
        )
    )
    decoder = train._action_ball_startup_offset_decoder_contract(
        zero_env, _JOINT_NAMES, dr_l0_zero_decoder=True
    )
    expected = _dr_l0_decoder()
    assert decoder == {
        key: expected[key]
        for key in (
            "startup_offset_delta_source",
            "startup_offset_delta_identity",
            "startup_offset_delta_lower",
            "startup_offset_delta_upper",
        )
    }
    assert decoder["startup_offset_delta_lower"] == [0.0] * 31
    assert decoder["startup_offset_delta_upper"] == [0.0] * 31

    def randomize_joint_default_pos():
        pass

    sampled_env = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            events=types.SimpleNamespace(
                add_joint_default_pos=types.SimpleNamespace(
                    func=randomize_joint_default_pos,
                    params={
                        "pos_distribution_params": (-0.01, 0.01),
                        "operation": "add",
                        "distribution": "uniform",
                    },
                )
            )
        )
    )
    sampled = train._action_ball_startup_offset_decoder_contract(
        sampled_env, _JOINT_NAMES, dr_l0_zero_decoder=False
    )
    assert sampled == {
        "startup_offset_delta_source": "events.add_joint_default_pos.uniform_add",
        "startup_offset_delta_lower": [-0.01] * 31,
        "startup_offset_delta_upper": [0.01] * 31,
    }
    with pytest.raises(RuntimeError, match="requires events.add_joint_default_pos=None"):
        train._action_ball_startup_offset_decoder_contract(
            sampled_env, _JOINT_NAMES, dr_l0_zero_decoder=True
        )
    with pytest.raises(RuntimeError, match="explicit startup"):
        train._action_ball_startup_offset_decoder_contract(
            zero_env, _JOINT_NAMES, dr_l0_zero_decoder=False
        )


def test_runtime_contract_reopens_all_off_state_and_binds_fresh_lineage():
    train = _load_train_module()
    env_cfg = _finalizer_env()
    dr = {
        "stable_ready_plant": True,
        "startup_physics_material": False,
        "startup_joint_default_pos": False,
        "policy_observation_corruption": False,
    }
    assert train._apply_action_ball_dr_l0_finalizer(env_cfg, dr, []) is True
    bootstrap = _shared_ready_bootstrap(_dr_l0_decoder())
    contract = train._action_ball_dr_l0_runtime_contract(
        env_cfg, policy_bootstrap=bootstrap
    )
    assert contract == train._action_ball_dr_l0_contract_payload()
    assert contract["lineage"] == {
        "binding": "training_contract_sha256",
        "fresh_checkpoint_required": True,
        "fresh_normalizers_required": True,
        "retained_dr_resume_forbidden": True,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "event_reenabled",
        "push_reenabled",
        "corruption_reenabled",
        "reset_noise_reenabled",
        "target_noise_reenabled",
        "delay_reenabled",
        "no_bootstrap",
        "nonzero_decoder",
    ),
)
def test_runtime_contract_rejects_post_finalizer_drift_or_old_lineage(mutation: str):
    train = _load_train_module()
    env_cfg = _finalizer_env()
    dr = {
        "stable_ready_plant": True,
        "startup_physics_material": False,
        "startup_joint_default_pos": False,
        "policy_observation_corruption": False,
    }
    assert train._apply_action_ball_dr_l0_finalizer(env_cfg, dr, []) is True
    bootstrap = _shared_ready_bootstrap(_dr_l0_decoder())
    if mutation == "event_reenabled":
        env_cfg.events.randomize_pd_gains = object()
    elif mutation == "push_reenabled":
        env_cfg.events.push_robot = object()
        env_cfg.push.enable = True
    elif mutation == "corruption_reenabled":
        env_cfg.observations.policy.enable_corruption = True
    elif mutation == "reset_noise_reenabled":
        env_cfg.commands.motion.velocity_range["yaw"] = (-0.1, 0.1)
    elif mutation == "target_noise_reenabled":
        env_cfg.commands.racket_target.target_noise_white = 0.01
    elif mutation == "delay_reenabled":
        env_cfg.actions.joint_pos.control_step_action_delay_max = 1
    elif mutation == "no_bootstrap":
        bootstrap = None
    else:
        bootstrap["decoder"]["startup_offset_delta_upper"][4] = 0.01
    with pytest.raises(RuntimeError, match="DR-L0"):
        train._action_ball_dr_l0_runtime_contract(
            env_cfg, policy_bootstrap=bootstrap
        )


def test_manifest_binds_checkpoint_and_normalizer_lineage_to_dr_l0_contract():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["runtime_hard_contract"] == {
        "all_off_state": (
            "events + proprio corruption + six-axis reset ranges + target "
            "transport + action delay + all push writers"
        ),
        "checkpoint_and_normalizer_binding": "training_contract_sha256",
        "fresh_checkpoint_required": True,
        "fresh_normalizers_required": True,
        "identity": "action_ball_dr_l0_exact_all_off_v1",
        "retained_dr_resume_forbidden": True,
        "startup_offset_delta": (
            "ordered 31-D exact zero with reproducible decoder identity"
        ),
    }
