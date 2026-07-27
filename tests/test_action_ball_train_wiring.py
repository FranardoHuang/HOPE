"""Host-only coverage for the formal action-ball training boundary."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_PATH = (
    ROOT / "hope_training" / "whole_body_tracking" / "scripts" / "train.py"
)
MDP_PATH = (
    ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)


class NS(types.SimpleNamespace):
    pass


class FakeObsTerm:
    def __init__(self, func=None, params=None):
        self.func = func
        self.params = dict(params or {})


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_mod():
    if importlib.util.find_spec("hydra") is None:
        hydra = types.ModuleType("hydra")
        hydra.__spec__ = importlib.util.spec_from_loader(
            "hydra", loader=None
        )
        hydra.main = lambda *_args, **_kwargs: lambda function: function
        sys.modules["hydra"] = hydra
    if importlib.util.find_spec("omegaconf") is None:
        omegaconf = types.ModuleType("omegaconf")
        omegaconf.__spec__ = importlib.util.spec_from_loader(
            "omegaconf", loader=None
        )
        omegaconf.ListConfig = list
        omegaconf.OmegaConf = NS(
            resolve=lambda *_args, **_kwargs: None,
            set_struct=lambda *_args, **_kwargs: None,
        )
        sys.modules["omegaconf"] = omegaconf
    return _load_by_path("_action_ball_train_under_test", TRAIN_PATH)


def _fake_loaded(action_ids, *, paths=None, digests=None):
    action_ids = tuple(action_ids)
    paths = paths or tuple(f"motions/{name}.npz" for name in action_ids)
    digests = digests or tuple("a" * 64 for _ in action_ids)
    actions = tuple(
        NS(
            action_id=name,
            action_uid=1000 + index,
            family=f"family_{index}",
            motion_path=paths[index],
            motion_sha256=digests[index],
            strike_phase=0.25 + 0.5 * index / max(1, len(action_ids)),
            mount_normal_sign=1 if index % 2 == 0 else -1,
        )
        for index, name in enumerate(action_ids)
    )
    return NS(
        file_sha256="b" * 64,
        canonical_sha256="c" * 64,
        source_path=pathlib.Path("/review/action_ball.json"),
        referenced_assets=None,
        manifest=NS(
            manifest_id="action-ball-test",
            action_order=action_ids,
            actions=actions,
            mobility_mode="move",
        ),
    )


def _action_ball_env(action_ids):
    racket = NS(
        target_mode="action_ball",
        action_ball_manifest_path="/review/action_ball.json",
        action_ball_manifest_sha256="b" * 64,
        action_ball_policy_contract_sha256="c" * 64,
        action_ball_evaluator_launch_receipt_path=(
            "configs/action_ball_evaluator_launch.json"
        ),
        action_ball_evaluator_launch_receipt_file_sha256="d" * 64,
        action_ball_seed=7,
        action_ball_pool_refill_rows=16,
        action_ball_fixed_direction=True,
        face_command=True,
        face_command_pairing="shared_plus_y",
        clip_names_per_clip=(),
        strike_phase_per_clip=(),
        mount_normal_sign_per_clip=(),
        achieved_target_mix_prob=0.0,
        midswing_resample_prob=0.0,
        target_delay_steps=0,
        target_delay_tts_mode="live",
        target_jitter_pos_per_s=0.0,
        target_jitter_vel_per_s=0.0,
        target_noise_white=0.0,
        target_noise_ar1_sigma=0.0,
        target_dropout_prob=0.0,
        target_post_strike_dropout_s=0.0,
        target_bias_per_swing=0.0,
        question_bank="",
        cq_anchor_bank="",
        exam_bank="",
        question_bank_allow_legacy=False,
        virtual_ball=True,
        vb_metrics_only=False,
        physical_ball=False,
        shadow_ball=False,
        shadow_table=False,
        planner_revision_enabled=False,
        racket_body_name="pingpang_red_Link",
        wrist_body_name="right_wrist_yaw_Link",
        mount_offset=(0.21021, 0.032078, 0.032036),
        mount_quat=(1.0, 0.0, 0.0, 0.0),
        mount_normal_axis=1,
        vb_table_near_x=0.5,
        vb_table_surface_z=0.76,
    )
    motion = NS(
        canonical_ready_mode=True,
        canonical_registry_repo_root="",
        canonical_registry_path="configs/canonical_registry.json",
        canonical_registry_sha256="1" * 64,
        canonical_registry_alignment_sha256="2" * 64,
        canonical_ready_sha256="3" * 64,
        canonical_ready_fk_sha256="4" * 64,
        canonical_promotion_certificate_path="certs/promotion.json",
        wrap_teleport=False,
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=17,
        clip_switch_prob=0.0,
        speed_scale_range=(1.0, 1.0),
        speed_scale_per_clip=None,
        event_timing_mode="disabled",
        planner_revision_enabled=False,
        joint_position_range=(0.0, 0.0),
        pose_range={},
        velocity_range={},
        clip_family_per_clip=None,
    )
    broad_regex = (
        r"^(?!left_ankle_roll_Link$)(?!right_ankle_roll_Link$).+$"
    )
    table_prim = "{ENV_REGEX_NS}/TableObstacle"
    return NS(
        obs_mode="hitter_footwork",
        face_command_obs=True,
        station_obs=False,
        physical_ball=False,
        table_obstacle=True,
        table_obstacle_prim=table_prim,
        scene=NS(
            racket_table_contact=NS(
                prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_Link",
                filter_prim_paths_expr=[table_prim],
            ),
            table_obstacle=NS(
                prim_path=table_prim,
                init_state=NS(pos=(1.87, 0.0, 0.735)),
                spawn=NS(
                    size=(2.74, 1.525, 0.05),
                    collision_props=NS(collision_enabled=True),
                ),
            ),
        ),
        terminations=NS(
            base_fell_tilt=NS(
                func="bad_orientation",
                time_out=False,
                params={"limit_angle": 0.7},
            ),
            base_too_low=NS(
                func="root_height_below_minimum",
                time_out=False,
                params={"minimum_height": 0.5},
            ),
            robot_hit_table=NS(
                func="robot_hit_table",
                time_out=False,
                params={
                    "sensor_cfg": NS(
                        name="contact_forces", body_names=[broad_regex]
                    ),
                    "filtered_sensor_cfg": NS(
                        name="racket_table_contact"
                    ),
                    "asset_cfg": NS(
                        name="robot", body_names=[broad_regex]
                    ),
                    "near_x": 0.5,
                    "surface_z": 0.76,
                    "force_threshold": 1.0,
                    "margin": 0.02,
                },
            ),
        ),
        commands=NS(racket_target=racket, motion=motion),
        observations=NS(policy=NS()),
        rewards=NS(motion_global_anchor_pos=None),
    )


def _task(action_ids):
    return {
        "actor_obs_contract": f"action_ball_n{len(action_ids)}",
        "racket": {"clip_names": list(action_ids)},
    }


def _stub_obs_imports(monkeypatch):
    managers = types.ModuleType("isaaclab.managers")
    managers.ObservationTermCfg = FakeObsTerm
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.managers = managers
    mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    mdp.racket_target_normal_cmd = object()
    mdp.action_one_hot = object()
    tracking = types.ModuleType("whole_body_tracking.tasks.tracking")
    tracking.mdp = mdp
    tasks = types.ModuleType("whole_body_tracking.tasks")
    tasks.tracking = tracking
    root = types.ModuleType("whole_body_tracking")
    root.tasks = tasks
    for name, module in (
        ("isaaclab", isaaclab),
        ("isaaclab.managers", managers),
        ("whole_body_tracking", root),
        ("whole_body_tracking.tasks", tasks),
        ("whole_body_tracking.tasks.tracking", tracking),
        ("whole_body_tracking.tasks.tracking.mdp", mdp),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return mdp


def _install_real_evaluator_module(monkeypatch):
    """Expose dependency-light evaluator code under its production import path."""

    curriculum = _load_by_path(
        f"_action_ball_curriculum_{id(monkeypatch)}",
        MDP_PATH / "action_ball_curriculum.py",
    )
    monkeypatch.setitem(sys.modules, "action_ball_curriculum", curriculum)
    evaluator = _load_by_path(
        f"_action_ball_evaluation_{id(monkeypatch)}",
        MDP_PATH / "action_ball_evaluation.py",
    )

    root = types.ModuleType("whole_body_tracking")
    tasks = types.ModuleType("whole_body_tracking.tasks")
    tracking = types.ModuleType("whole_body_tracking.tasks.tracking")
    mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    root.tasks = tasks
    tasks.tracking = tracking
    tracking.mdp = mdp
    mdp.action_ball_evaluation = evaluator
    for name, module in (
        ("whole_body_tracking", root),
        ("whole_body_tracking.tasks", tasks),
        ("whole_body_tracking.tasks.tracking", tracking),
        ("whole_body_tracking.tasks.tracking.mdp", mdp),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return curriculum, evaluator


def _stub_preflight(monkeypatch, train_mod, action_ids):
    loaded = _fake_loaded(action_ids)
    monkeypatch.setattr(
        train_mod,
        "_load_action_ball_manifest_from_cfg",
        lambda *_args: loaded,
    )
    monkeypatch.setattr(
        train_mod,
        "_action_ball_preflight_contract",
        lambda *_args: {"sha256": "d" * 64},
    )
    monkeypatch.setattr(
        train_mod,
        "_validate_action_ball_static_motion_admission",
        lambda *_args, **_kwargs: {
            "certificate_sha256": "f" * 64,
        },
    )
    monkeypatch.setattr(
        train_mod,
        "_load_action_ball_evaluator_launch_from_cfg",
        lambda *_args, **_kwargs: {
            "launch_receipt_canonical_sha256": "e" * 64,
        },
    )
    return loaded


def _runtime_contract_fixture(train_mod):
    action_ids = ("a", "b")
    motion_sha = ("1" * 64, "2" * 64)
    profile_sha = ("3" * 64, "4" * 64)
    sampler_sha = "5" * 64
    adapter_sha = "6" * 64
    runtime_sha = "7" * 64
    physics_payload = {
        "schema_version": 1,
        "kind": "test.physics",
        "table": {"surface_z": 0.76},
    }
    physics_sha = train_mod._canonical_contract_sha256(physics_payload)
    solver_payload = {
        "schema_version": 1,
        "kind": "test.solver",
        "physics_profile_sha256": physics_sha,
        "acceptance": {"fixed_direction": True},
    }
    solver_sha = train_mod._canonical_contract_sha256(solver_payload)
    preflight = {
        "schema_version": 1,
        "manifest": {
            "path": "configs/action_ball.json",
            "file_sha256": "8" * 64,
            "canonical_sha256": "9" * 64,
            "manifest_id": "test-manifest",
        },
        "mobility_mode": "move",
        "action_order": list(action_ids),
        "action_uids": [101, 102],
        "action_bindings": [
            {
                "action_id": name,
                "action_uid": 101 + slot,
                "action_slot": slot,
                "family": "forehand" if slot == 0 else "backhand",
                "motion_path": f"motions/{name}.npz",
                "motion_sha256": motion_sha[slot],
                "sampling_profile_sha256": profile_sha[slot],
                "strike_phase": 0.5,
                "mount_normal_sign": 1 if slot == 0 else -1,
            }
            for slot, name in enumerate(action_ids)
        ],
        "prototype": {
            "path": "configs/prototypes.json",
            "scope": "full",
            "sha256": "a" * 64,
        },
        "profile_adapter": {
            "contract": {"schema_version": 1},
            "sha256": adapter_sha,
        },
        "sampler": {
            "contract_sha256": sampler_sha,
            "seed": 11,
            "pool_refill_rows": 32,
        },
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "curriculum": {
            "config": {
                "target_failure_rate": 0.1,
                "failure_band_half_width": 0.02,
            },
            "config_sha256": "b" * 64,
        },
        "holdout": {"seed": 9, "samples_per_action": 64, "split_id": "x"},
        "fixed_direction": True,
        "initial_episode_length_randomization": False,
        "policy_contract_sha256": "c" * 64,
    }
    preflight["sha256"] = train_mod._canonical_contract_sha256(preflight)
    bindings = [
        {
            "action_uid": row["action_uid"],
            "action_slot": row["action_slot"],
            "motion_path": row["motion_path"],
            "motion_sha256": row["motion_sha256"],
            "profile_sha256": row["sampling_profile_sha256"],
        }
        for row in preflight["action_bindings"]
    ]
    registry_sha = train_mod._canonical_contract_sha256(
        {
            "runtime_contract_sha256": runtime_sha,
            "pins": {
                "manifest_sha256": preflight["manifest"]["file_sha256"],
                "sampler_sha256": sampler_sha,
                "physics_sha256": physics_sha,
                "solver_sha256": solver_sha,
            },
            "mobility_mode": "move",
            "bindings": bindings,
        }
    )
    motion_cfg = NS(
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=19,
        canonical_registry_sha256="e" * 64,
        canonical_registry_alignment_sha256="f" * 64,
        canonical_ready_sha256="0" * 64,
        canonical_ready_fk_sha256="1" * 64,
    )
    racket_cfg = NS(cq_overdraw=1.5, cq_max_redraw_rounds=4)
    sha_columns = {
        name: ["2" * 64, "3" * 64]
        for name in (
            "canonical_source_manifest_sha256_per_clip",
            "canonical_build_manifest_sha256_per_clip",
            "canonical_applicability_manifest_sha256_per_clip",
            "canonical_evidence_manifest_sha256_per_clip",
            "canonical_training_config_sha256_per_clip",
            "canonical_adoption_manifest_sha256_per_clip",
        )
    }
    unsigned = {
        "schema_version": 1,
        "kind": (
            "whole_body_tracking.RacketTargetCommand."
            "action_ball_hard_contract"
        ),
        "manifest": copy.deepcopy(preflight["manifest"]),
        "mobility_mode": "move",
        "action_order": list(action_ids),
        "action_uids": [101, 102],
        "bindings": bindings,
        "prototype": copy.deepcopy(preflight["prototype"]),
        "profiles": {
            "adapter_contract_sha256": adapter_sha,
            "profile_sha256": list(profile_sha),
            "sampler_contract_sha256": sampler_sha,
        },
        "sampling": {
            "action_ball_seed": 11,
            "pool_refill_rows": 32,
            "balanced_clip_sampling": True,
            "balanced_clip_sampling_seed": 19,
            "external_overdraw_multiplier": 1.5,
            "maximum_external_proposal_rounds": 4,
        },
        "solver": {"payload": solver_payload, "sha256": solver_sha},
        "physics": {"payload": physics_payload, "sha256": physics_sha},
        "curriculum": {
            "config": copy.deepcopy(preflight["curriculum"]["config"]),
            "policy_contract_sha256": "c" * 64,
            "frozen_checkpoint_evidence_required": True,
            "live_rollout_advances_curriculum": False,
        },
        "runtime": {
            "runtime_contract_sha256": runtime_sha,
            "registry_sha256": registry_sha,
            "fixed_direction": True,
            "wrap_teleport": False,
        },
        "motion_admission": {
            "canonical_registry_sha256": "e" * 64,
            "canonical_registry_alignment_sha256": "f" * 64,
            "canonical_ready_sha256": "0" * 64,
            "canonical_ready_fk_sha256": "1" * 64,
            **sha_columns,
            "canonical_evidence_level_per_clip": ["E2", "E2"],
        },
    }
    contract = copy.deepcopy(unsigned)
    contract["canonical_sha256"] = train_mod._canonical_contract_sha256(
        unsigned
    )
    return contract, preflight, racket_cfg, motion_cfg, runtime_sha


def _rehash_runtime_contract(train_mod, contract):
    unsigned = copy.deepcopy(contract)
    unsigned.pop("canonical_sha256", None)
    contract["canonical_sha256"] = train_mod._canonical_contract_sha256(
        unsigned
    )


def test_action_ball_yaml_keys_are_whitelisted_and_consumed(train_mod):
    racket = {
        "action_ball_manifest_path",
        "action_ball_manifest_sha256",
        "action_ball_policy_contract_sha256",
        "action_ball_evaluator_launch_receipt_path",
        "action_ball_evaluator_launch_receipt_file_sha256",
        "action_ball_seed",
        "action_ball_pool_refill_rows",
        "action_ball_fixed_direction",
    }
    motion = {
        "canonical_ready_mode",
        "canonical_registry_path",
        "canonical_registry_repo_root",
        "canonical_registry_sha256",
        "canonical_registry_alignment_sha256",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "canonical_promotion_certificate_path",
        "joint_position_range",
        "pose_range",
        "velocity_range",
    }
    assert racket <= set(train_mod._RACKET_KEYS)
    assert motion <= set(train_mod._MOTION_KEYS)
    tree = ast.parse(inspect.getsource(train_mod._apply_task_overrides))
    consumed = {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_set_attr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }
    # Loop-translated canonical string/map fields are intentionally visible as
    # their literal field tuples rather than Constant call arguments.
    assert racket <= consumed
    source = inspect.getsource(train_mod._apply_task_overrides)
    assert all(repr(name) in source or f'"{name}"' in source for name in motion)


def test_action_ball_explicit_trusted_root_must_not_depend_on_cwd(train_mod):
    with pytest.raises(
        train_mod._OverrideError, match="must be absolute"
    ):
        train_mod._action_ball_repo_root(
            NS(canonical_registry_repo_root=".")
        )
    assert train_mod._action_ball_repo_root(
        NS(canonical_registry_repo_root=str(ROOT))
    ) == ROOT.resolve()


def _static_motion_admission_fixture(
    train_mod, monkeypatch, tmp_path, *, action_count, trusted
):
    registry_path = tmp_path / "configs" / "registry.json"
    certificate_path = tmp_path / "certs" / "promotion.json"
    ready_path = tmp_path / "motions" / "ready.json"
    ready_fk_path = tmp_path / "motions" / "ready_fk.json"
    for path, payload in (
        (registry_path, b'{"schema_version":2}\n'),
        (certificate_path, b'{"schema_version":2}\n'),
        (ready_path, b"ready\n"),
        (ready_fk_path, b"ready-fk\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    certificate_sha = hashlib.sha256(
        certificate_path.read_bytes()
    ).hexdigest()
    action_ids = tuple(f"action_{index}" for index in range(action_count))
    action_bindings = []
    entries = []
    motion_files = []
    for index, action_id in enumerate(action_ids):
        relative_path = f"motions/{action_id}.npz"
        motion_path = tmp_path / relative_path
        motion_path.write_bytes(f"motion-{index}".encode("ascii"))
        motion_sha = hashlib.sha256(motion_path.read_bytes()).hexdigest()
        family = "forehand" if index % 2 == 0 else "backhand"
        strike_phase = 0.25 + (0.5 * index / max(1, action_count))
        mount_sign = 1 if index % 2 == 0 else -1
        action_bindings.append(
            {
                "action_id": action_id,
                "action_uid": 1000 + index,
                "action_slot": index,
                "family": family,
                "motion_path": relative_path,
                "motion_sha256": motion_sha,
                "sampling_profile_sha256": hashlib.sha256(
                    f"profile-{index}".encode("ascii")
                ).hexdigest(),
                "strike_phase": strike_phase,
                "mount_normal_sign": mount_sign,
            }
        )
        entries.append(
            NS(
                motion_id=action_id,
                npz_path_text=relative_path,
                npz_sha256=motion_sha,
                family=family,
                strike_phase=strike_phase,
                mount_normal_sign=float(mount_sign),
            )
        )
        motion_files.append(str(motion_path.resolve()))

    preflight = {
        "action_order": list(action_ids),
        "action_bindings": action_bindings,
        "prototype": {"scope": "full"},
    }
    alignment_sha = "2" * 64
    ready_sha = "3" * 64
    ready_fk_sha = "4" * 64
    motion_cfg = NS(
        canonical_registry_repo_root=str(tmp_path),
        canonical_registry_path="configs/registry.json",
        canonical_registry_sha256=registry_sha,
        canonical_registry_alignment_sha256=alignment_sha,
        canonical_ready_sha256=ready_sha,
        canonical_ready_fk_sha256=ready_fk_sha,
        canonical_promotion_certificate_path="certs/promotion.json",
    )

    class FakeRegistry:
        pass

    class FakeTables:
        pass

    registry = FakeRegistry()
    registry.schema_version = 2
    registry.path = registry_path.resolve()
    registry.repo_root = tmp_path.resolve()
    registry.registry_sha256 = registry_sha
    registry.registry_digest_pinned = True
    registry.motion_ids = action_ids
    registry.scope = "full"
    registry.bank_id = "generic-test-bank"
    registry.entries = tuple(entries)
    registry.canonical_ready_path = ready_path.resolve()
    registry.canonical_ready_fk_path = ready_fk_path.resolve()

    tables = FakeTables()
    tables.registry_sha256 = registry_sha
    tables.alignment_sha256 = alignment_sha
    tables.canonical_ready_sha256 = ready_sha
    tables.canonical_ready_fk_sha256 = ready_fk_sha
    tables.authorization_purpose = "training"
    tables.motion_ids = action_ids
    tables.scope = "full"
    tables.bank_id = registry.bank_id
    tables.motion_file = tuple(motion_files)
    tables.clip_family_per_clip = tuple(
        row["family"] for row in action_bindings
    )
    tables.strike_phase_per_clip = tuple(
        row["strike_phase"] for row in action_bindings
    )
    tables.mount_normal_sign_per_clip = tuple(
        row["mount_normal_sign"] for row in action_bindings
    )

    def load_training_adopted_registry(*args, **kwargs):
        assert pathlib.Path(args[0]).resolve() == registry_path.resolve()
        assert pathlib.Path(args[1]).resolve() == certificate_path.resolve()
        assert kwargs["expected_registry_sha256"] == registry_sha
        assert (
            kwargs["expected_promotion_certificate_sha256"]
            == certificate_sha
        )
        return registry, tables

    admission = NS(
        TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256=(
            frozenset({certificate_sha}) if trusted else frozenset()
        ),
        _binding_sha256=lambda _binding: "5" * 64,
    )
    module = NS(
        motion_admission=admission,
        REGISTRY_SCHEMA_VERSION=1,
        GENERIC_REGISTRY_SCHEMA_VERSION=2,
        CanonicalMotionBankRegistry=FakeRegistry,
        CanonicalRuntimeTables=FakeTables,
        load_training_adopted_registry=load_training_adopted_registry,
        bank_promotion_binding=lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        train_mod,
        "_load_action_ball_canonical_registry_module",
        lambda _repo_root: module,
    )
    return preflight, motion_cfg, certificate_sha


def test_action_ball_motion_pre_gym_requires_nonempty_code_trust(
    train_mod, monkeypatch, tmp_path
):
    preflight, motion_cfg, _certificate_sha = (
        _static_motion_admission_fixture(
            train_mod,
            monkeypatch,
            tmp_path,
            action_count=1,
            trusted=False,
        )
    )
    with pytest.raises(RuntimeError, match="code trust set is empty"):
        train_mod._validate_action_ball_static_motion_admission(
            preflight,
            motion_cfg=motion_cfg,
        )


def test_action_ball_motion_pre_gym_accepts_generic_v2_arbitrary_n_exact_rows(
    train_mod, monkeypatch, tmp_path
):
    preflight, motion_cfg, certificate_sha = (
        _static_motion_admission_fixture(
            train_mod,
            monkeypatch,
            tmp_path,
            action_count=93,
            trusted=True,
        )
    )
    result = train_mod._validate_action_ball_static_motion_admission(
        preflight,
        motion_cfg=motion_cfg,
    )
    assert result["registry_schema_version"] == 2
    assert result["certificate_sha256"] == certificate_sha
    assert len(result["motion_rows"]) == 93
    assert [row["motion_id"] for row in result["motion_rows"]] == (
        preflight["action_order"]
    )

    drifted = copy.deepcopy(preflight)
    drifted["action_order"].reverse()
    with pytest.raises(RuntimeError, match="ordered motion_ids"):
        train_mod._validate_action_ball_static_motion_admission(
            drifted,
            motion_cfg=motion_cfg,
        )


def _opaque_motion_admission_fixture(train_mod, monkeypatch):
    action_uids = [1001, 1002]
    motion_rows = [
        {
            "motion_id": f"action_{index}",
            "action_uid": action_uids[index],
            "action_slot": index,
            "motion_path": f"motions/action_{index}.npz",
            "motion_sha256": str(index + 1) * 64,
            "profile_sha256": str(index + 3) * 64,
        }
        for index in range(2)
    ]
    static = {
        "registry_schema_version": 2,
        "bank_id": "generic-test-bank",
        "scope": "full",
        "registry_path": "configs/registry.json",
        "registry_sha256": "5" * 64,
        "alignment_sha256": "6" * 64,
        "canonical_ready_path": "motions/ready.json",
        "canonical_ready_sha256": "7" * 64,
        "canonical_ready_fk_path": "motions/ready_fk.json",
        "canonical_ready_fk_sha256": "8" * 64,
        "certificate_path": "certs/promotion.json",
        "certificate_sha256": "9" * 64,
        "promotion_binding_sha256": "a" * 64,
        "motion_rows": motion_rows,
    }
    monkeypatch.setattr(
        train_mod,
        "_validate_action_ball_static_motion_admission",
        lambda *_args, **_kwargs: copy.deepcopy(static),
    )
    source_paths = {
        "commands": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/commands.py"
        ),
        "action_ball_runtime": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/"
            "action_ball_runtime.py"
        ),
        "canonical_motion_registry": (
            "hope_training/whole_body_tracking/scripts/"
            "canonical_motion_registry.py"
        ),
        "canonical_motion_admission": (
            "hope_training/whole_body_tracking/scripts/"
            "canonical_motion_admission.py"
        ),
    }
    source_receipts = {
        name: {
            "path": path,
            "sha256": hashlib.sha256(
                (ROOT / path).read_bytes()
            ).hexdigest(),
        }
        for name, path in source_paths.items()
    }
    runtime_sha = "b" * 64
    registry_sha = "c" * 64
    owner_sha = "d" * 64
    unsigned = {
        "schema_version": 1,
        "kind": (
            "whole_body_tracking.MotionCommand."
            "action_ball_motion_admission"
        ),
        "authorization_purpose": "training",
        "trusted_repo_root": str(ROOT.resolve()),
        "opaque_capability": {
            "capability_type": "TrustedMotionAdmission",
            "purpose": "training",
            "promotion_binding_sha256": static[
                "promotion_binding_sha256"
            ],
            "certificate_path": static["certificate_path"],
            "certificate_sha256": static["certificate_sha256"],
        },
        "canonical_bank": {
            "bank_id": static["bank_id"],
            "scope": static["scope"],
            "registry_path": static["registry_path"],
            "registry_sha256": static["registry_sha256"],
            "alignment_sha256": static["alignment_sha256"],
            "canonical_ready_path": static["canonical_ready_path"],
            "canonical_ready_sha256": static[
                "canonical_ready_sha256"
            ],
            "canonical_ready_fk_path": static[
                "canonical_ready_fk_path"
            ],
            "canonical_ready_fk_sha256": static[
                "canonical_ready_fk_sha256"
            ],
            "motion_rows": copy.deepcopy(motion_rows),
        },
        "runtime_binding": {
            "runtime_contract_sha256": runtime_sha,
            "broker_state_schema_version": 4,
            "broker_registry_sha256": registry_sha,
            "provider_state_owner_sha256": owner_sha,
            "ordered_action_uids": action_uids,
            "manifest_rows_are_identity_only": True,
        },
        "implementation_sources": source_receipts,
    }
    receipt = copy.deepcopy(unsigned)
    receipt["canonical_sha256"] = train_mod._canonical_contract_sha256(
        unsigned
    )
    return (
        receipt,
        {"action_uids": action_uids},
        NS(canonical_registry_repo_root=str(ROOT)),
        runtime_sha,
        registry_sha,
        owner_sha,
    )


def test_action_ball_opaque_motion_receipt_revalidates_code_and_runtime(
    train_mod, monkeypatch
):
    (
        receipt,
        preflight,
        motion_cfg,
        runtime_sha,
        registry_sha,
        owner_sha,
    ) = _opaque_motion_admission_fixture(train_mod, monkeypatch)
    assert (
        train_mod._validate_action_ball_motion_admission_receipt(
            receipt,
            preflight=preflight,
            motion_cfg=motion_cfg,
            expected_runtime_contract_sha256=runtime_sha,
            expected_broker_state_schema_version=4,
            expected_broker_registry_sha256=registry_sha,
            expected_provider_state_owner_sha256=owner_sha,
        )
        is receipt
    )

    drifted = copy.deepcopy(receipt)
    drifted["runtime_binding"]["ordered_action_uids"].reverse()
    drifted["canonical_sha256"] = train_mod._canonical_contract_sha256(
        {
            key: value
            for key, value in drifted.items()
            if key != "canonical_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="runtime binding"):
        train_mod._validate_action_ball_motion_admission_receipt(
            drifted,
            preflight=preflight,
            motion_cfg=motion_cfg,
            expected_runtime_contract_sha256=runtime_sha,
            expected_broker_state_schema_version=4,
            expected_broker_registry_sha256=registry_sha,
            expected_provider_state_owner_sha256=owner_sha,
        )


@pytest.mark.parametrize("action_count", (1, 5, 93))
def test_action_ball_appends_exact_arbitrary_n_tail(
    train_mod, monkeypatch, action_count
):
    action_ids = tuple(f"action_{index}" for index in range(action_count))
    env_cfg = _action_ball_env(action_ids)
    _stub_preflight(monkeypatch, train_mod, action_ids)
    mdp = _stub_obs_imports(monkeypatch)
    applied = []

    train_mod._finalize_action_ball_training_cfg(
        env_cfg, _task(action_ids), applied
    )

    terms = list(vars(env_cfg.observations.policy).items())
    assert [name for name, _term in terms] == [
        "racket_target_normal_cmd",
        "action_one_hot",
    ]
    assert terms[0][1].func is mdp.racket_target_normal_cmd
    assert terms[1][1].func is mdp.action_one_hot
    assert terms[1][1].params["expected_actions"] == action_count
    assert env_cfg.commands.racket_target.clip_names_per_clip == action_ids
    assert len(env_cfg.commands.motion.clip_family_per_clip) == action_count
    assert any(f"action_one_hot(+{action_count})" in line for line in applied)


def test_action_ball_actor_action_count_is_bounded(
    train_mod, monkeypatch
):
    action_ids = tuple(f"action_{index}" for index in range(1025))
    env_cfg = _action_ball_env(action_ids)
    _stub_preflight(monkeypatch, train_mod, action_ids)
    with pytest.raises(train_mod._OverrideError, match="at most 1024"):
        train_mod._finalize_action_ball_training_cfg(
            env_cfg, _task(action_ids), []
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda env: setattr(
                env.commands.racket_target, "virtual_ball", False
            ),
            "virtual_ball=true",
        ),
        (
            lambda env: setattr(
                env.commands.racket_target, "achieved_target_mix_prob", 0.1
            ),
            "achieved_target_mix_prob",
        ),
        (
            lambda env: setattr(
                env.commands.motion, "canonical_ready_mode", False
            ),
            "canonical_ready_mode",
        ),
        (
            lambda env: setattr(env.commands.motion, "wrap_teleport", True),
            "wrap_teleport",
        ),
        (
            lambda env: setattr(
                env.commands.motion, "speed_scale_range", (0.8, 1.0)
            ),
            "native motion speed",
        ),
        (
            lambda env: setattr(env, "table_obstacle", False),
            "table_obstacle=true",
        ),
        (
            lambda env: setattr(
                env.commands.motion, "joint_position_range", (-0.1, 0.1)
            ),
            "joint_position_range",
        ),
        (
            lambda env: setattr(
                env.commands.racket_target, "planner_revision_enabled", True
            ),
            "planner_revision",
        ),
    ),
)
def test_action_ball_incompatible_recipe_fails_before_tail(
    train_mod, monkeypatch, mutate, message
):
    action_ids = ("a", "b")
    env_cfg = _action_ball_env(action_ids)
    _stub_preflight(monkeypatch, train_mod, action_ids)
    _stub_obs_imports(monkeypatch)
    mutate(env_cfg)
    with pytest.raises(train_mod._OverrideError, match=message):
        train_mod._finalize_action_ball_training_cfg(
            env_cfg, _task(action_ids), []
        )
    assert vars(env_cfg.observations.policy) == {}


def test_action_ball_manifest_order_and_physical_truth_fail_closed(
    train_mod, monkeypatch
):
    env_cfg = _action_ball_env(("a", "b"))
    _stub_preflight(monkeypatch, train_mod, ("b", "a"))
    _stub_obs_imports(monkeypatch)
    with pytest.raises(train_mod._OverrideError, match="action_order"):
        train_mod._finalize_action_ball_training_cfg(
            env_cfg, _task(("a", "b")), []
        )

    env_cfg = _action_ball_env(("a", "b"))
    _stub_preflight(monkeypatch, train_mod, ("a", "b"))
    env_cfg.commands.racket_target.physical_ball = True
    with pytest.raises(train_mod._OverrideError, match="switches disagree"):
        train_mod._finalize_action_ball_training_cfg(
            env_cfg, _task(("a", "b")), []
        )


def test_action_ball_allows_only_pinned_solver_cq_knobs(
    train_mod, monkeypatch
):
    action_ids = ("a", "b")
    env_cfg = _action_ball_env(action_ids)
    _stub_preflight(monkeypatch, train_mod, action_ids)
    _stub_obs_imports(monkeypatch)
    task = _task(action_ids)
    task["racket"].update(
        {
            "cq_overdraw": 1.5,
            "cq_n_iters": 8,
            "cq_tol_m": 0.04,
            "cq_speed_budget": 7.0,
            "cq_max_redraw_rounds": 4,
        }
    )
    train_mod._finalize_action_ball_training_cfg(env_cfg, task, [])

    env_cfg = _action_ball_env(action_ids)
    _stub_preflight(monkeypatch, train_mod, action_ids)
    task = _task(action_ids)
    task["racket"]["cq_seed"] = 3
    with pytest.raises(
        train_mod._OverrideError, match="legacy-CQ producer"
    ):
        train_mod._finalize_action_ball_training_cfg(env_cfg, task, [])


def test_action_ball_resolved_motion_path_and_bytes_are_exact(
    train_mod, monkeypatch, tmp_path
):
    action_ids = ("a", "b")
    files = []
    assets = []
    digests = []
    relative = []
    for index in range(2):
        path = tmp_path / f"motion_{index}.npz"
        path.write_bytes(f"motion-{index}".encode())
        files.append(str(path))
        relative.append(path.name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests.append(digest)
        assets.append(NS(resolved_path=path.resolve()))
    loaded = _fake_loaded(
        action_ids, paths=tuple(relative), digests=tuple(digests)
    )
    loaded.referenced_assets = NS(motions=tuple(assets))
    env_cfg = _action_ball_env(action_ids)
    monkeypatch.setattr(
        train_mod,
        "_load_action_ball_manifest_from_cfg",
        lambda *_args: loaded,
    )
    train_mod._validate_action_ball_motion_sources(env_cfg, files)

    with pytest.raises(train_mod._OverrideError, match="path mismatch"):
        train_mod._validate_action_ball_motion_sources(
            env_cfg, list(reversed(files))
        )


def test_action_ball_runtime_hard_contract_accepts_exact_payload(train_mod):
    contract, preflight, racket_cfg, motion_cfg, runtime_sha = (
        _runtime_contract_fixture(train_mod)
    )
    assert (
        train_mod._validate_action_ball_runtime_hard_contract(
            contract,
            preflight=preflight,
            racket_cfg=racket_cfg,
            motion_cfg=motion_cfg,
            expected_runtime_contract_sha256=runtime_sha,
        )
        is contract
    )


def _evaluator_launch_fixture(
    train_mod, monkeypatch, tmp_path, *, trust=True
):
    _contract, preflight, _racket, _motion, _runtime_sha = (
        _runtime_contract_fixture(train_mod)
    )
    curriculum, evaluator = _install_real_evaluator_module(monkeypatch)
    relative_source = pathlib.PurePosixPath(
        "hope_training/whole_body_tracking/closed_attempt_source.py"
    )
    source = tmp_path.joinpath(*relative_source.parts)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"reviewed closed-attempt source\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    profiles = tuple(
        curriculum.ActionProfileKey(
            action_uid=row["action_uid"],
            profile_sha256=row["sampling_profile_sha256"],
            mobility=preflight["mobility_mode"],
        )
        for row in preflight["action_bindings"]
    )
    launch = evaluator.launch_receipt_document(
        curriculum_contract_sha256=preflight["profile_adapter"]["sha256"],
        profile_order=profiles,
        sampler_sha256=preflight["sampler"]["contract_sha256"],
        solver_sha256=preflight["solver_profile_sha256"],
        policy_contract_sha256=preflight["policy_contract_sha256"],
        attempt_source_contract_sha256="d" * 64,
        attempt_source_path=relative_source.as_posix(),
        attempt_source_sha256=source_sha,
    )
    launch_sha = train_mod._canonical_contract_sha256(launch)
    evaluator.TRUSTED_FROZEN_EVALUATOR_LAUNCH_RECEIPT_SHA256 = (
        frozenset({launch_sha}) if trust else frozenset()
    )
    return evaluator, launch, launch_sha, preflight, source


def test_action_ball_evaluator_launch_is_code_pinned_and_reopens_source(
    train_mod, monkeypatch, tmp_path
):
    _evaluator, launch, launch_sha, preflight, _source = (
        _evaluator_launch_fixture(
            train_mod, monkeypatch, tmp_path, trust=True
        )
    )
    result = train_mod._validate_action_ball_evaluator_launch_receipt(
        launch,
        declared_launch_sha256=launch_sha,
        preflight=preflight,
        solver_sha256=preflight["solver_profile_sha256"],
        repo_root=tmp_path,
    )
    assert result["launch_receipt_canonical_sha256"] == launch_sha
    assert (
        result["authority_binding"]["launch_receipt_sha256"]
        == launch_sha
    )


def test_action_ball_evaluator_launch_rejects_empty_trust_and_source_drift(
    train_mod, monkeypatch, tmp_path
):
    _evaluator, launch, launch_sha, preflight, source = (
        _evaluator_launch_fixture(
            train_mod, monkeypatch, tmp_path, trust=False
        )
    )
    with pytest.raises(RuntimeError, match="not code-pinned"):
        train_mod._validate_action_ball_evaluator_launch_receipt(
            launch,
            declared_launch_sha256=launch_sha,
            preflight=preflight,
            solver_sha256=preflight["solver_profile_sha256"],
            repo_root=tmp_path,
        )

    evaluator, launch, launch_sha, preflight, source = (
        _evaluator_launch_fixture(
            train_mod, monkeypatch, tmp_path / "drift", trust=True
        )
    )
    del evaluator
    source.write_bytes(b"changed after launch receipt\n")
    with pytest.raises(RuntimeError, match="source bytes drifted"):
        train_mod._validate_action_ball_evaluator_launch_receipt(
            launch,
            declared_launch_sha256=launch_sha,
            preflight=preflight,
            solver_sha256=preflight["solver_profile_sha256"],
            repo_root=tmp_path / "drift",
        )


def test_action_ball_evaluator_launch_rejects_profile_or_pin_drift(
    train_mod, monkeypatch, tmp_path
):
    _evaluator, launch, launch_sha, preflight, _source = (
        _evaluator_launch_fixture(
            train_mod, monkeypatch, tmp_path, trust=True
        )
    )
    reordered = copy.deepcopy(launch)
    reordered["profile_order"].reverse()
    with pytest.raises(RuntimeError, match="profile_order"):
        train_mod._validate_action_ball_evaluator_launch_receipt(
            reordered,
            declared_launch_sha256=launch_sha,
            preflight=preflight,
            solver_sha256=preflight["solver_profile_sha256"],
            repo_root=tmp_path,
        )
    with pytest.raises(RuntimeError, match="receipt SHA mismatch"):
        train_mod._validate_action_ball_evaluator_launch_receipt(
            launch,
            declared_launch_sha256="f" * 64,
            preflight=preflight,
            solver_sha256=preflight["solver_profile_sha256"],
            repo_root=tmp_path,
        )


def test_action_ball_evaluator_receipt_is_loaded_before_gym_from_exact_file(
    train_mod, monkeypatch, tmp_path
):
    _evaluator, launch, launch_sha, preflight, _source = (
        _evaluator_launch_fixture(
            train_mod, monkeypatch, tmp_path, trust=True
        )
    )
    relative = pathlib.PurePosixPath("configs/evaluator_launch.json")
    receipt_file = tmp_path.joinpath(*relative.parts)
    receipt_file.parent.mkdir(parents=True)
    receipt_file.write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    file_sha = hashlib.sha256(receipt_file.read_bytes()).hexdigest()
    racket_cfg = NS(
        action_ball_evaluator_launch_receipt_path=relative.as_posix(),
        action_ball_evaluator_launch_receipt_file_sha256=file_sha,
    )
    result = train_mod._load_action_ball_evaluator_launch_from_cfg(
        racket_cfg,
        NS(canonical_registry_repo_root=str(tmp_path)),
        preflight=preflight,
    )
    assert result["file_sha256"] == file_sha
    assert result["launch_receipt_canonical_sha256"] == launch_sha
    assert result["launch_receipt"] == launch


def test_action_ball_evaluator_receipt_pre_gym_rejects_missing_duplicate_or_symlink(
    train_mod, monkeypatch, tmp_path
):
    _evaluator, launch, _launch_sha, preflight, _source = (
        _evaluator_launch_fixture(
            train_mod, monkeypatch, tmp_path, trust=True
        )
    )
    motion_cfg = NS(canonical_registry_repo_root=str(tmp_path))
    with pytest.raises(
        train_mod._OverrideError, match="requires.*receipt_path"
    ):
        train_mod._load_action_ball_evaluator_launch_from_cfg(
            NS(
                action_ball_evaluator_launch_receipt_path="",
                action_ball_evaluator_launch_receipt_file_sha256="",
            ),
            motion_cfg,
            preflight=preflight,
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(
        b'{"schema_version":1,"schema_version":1}\n'
    )
    duplicate_sha = hashlib.sha256(duplicate.read_bytes()).hexdigest()
    with pytest.raises(
        train_mod._OverrideError, match="duplicate JSON key"
    ):
        train_mod._load_action_ball_evaluator_launch_from_cfg(
            NS(
                action_ball_evaluator_launch_receipt_path="duplicate.json",
                action_ball_evaluator_launch_receipt_file_sha256=duplicate_sha,
            ),
            motion_cfg,
            preflight=preflight,
        )

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"value":NaN}\n')
    nonfinite_sha = hashlib.sha256(nonfinite.read_bytes()).hexdigest()
    with pytest.raises(train_mod._OverrideError, match="non-finite JSON"):
        train_mod._load_action_ball_evaluator_launch_from_cfg(
            NS(
                action_ball_evaluator_launch_receipt_path="nonfinite.json",
                action_ball_evaluator_launch_receipt_file_sha256=(
                    nonfinite_sha
                ),
            ),
            motion_cfg,
            preflight=preflight,
        )

    target = tmp_path / "target.json"
    target.write_text(json.dumps(launch), encoding="utf-8")
    link = tmp_path / "linked.json"
    link.symlink_to(target)
    target_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(train_mod._OverrideError, match="symbolic link"):
        train_mod._load_action_ball_evaluator_launch_from_cfg(
            NS(
                action_ball_evaluator_launch_receipt_path="linked.json",
                action_ball_evaluator_launch_receipt_file_sha256=target_sha,
            ),
            motion_cfg,
            preflight=preflight,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda contract: contract.pop("sampling"),
            "invalid keys",
        ),
        (
            lambda contract: contract["sampling"].__setitem__(
                "action_ball_seed", 12
            ),
            "runtime sampling",
        ),
        (
            lambda contract: contract["bindings"].reverse(),
            "runtime bindings",
        ),
        (
            lambda contract: contract["curriculum"].__setitem__(
                "policy_contract_sha256", "d" * 64
            ),
            "runtime curriculum",
        ),
        (
            lambda contract: contract["physics"]["payload"].__setitem__(
                "unbound_mutation", True
            ),
            "does not authenticate",
        ),
    ),
)
def test_action_ball_runtime_hard_contract_rejects_drift(
    train_mod, mutate, message
):
    contract, preflight, racket_cfg, motion_cfg, runtime_sha = (
        _runtime_contract_fixture(train_mod)
    )
    mutate(contract)
    _rehash_runtime_contract(train_mod, contract)
    with pytest.raises(RuntimeError, match=message):
        train_mod._validate_action_ball_runtime_hard_contract(
            contract,
            preflight=preflight,
            racket_cfg=racket_cfg,
            motion_cfg=motion_cfg,
            expected_runtime_contract_sha256=runtime_sha,
        )


def test_action_ball_policy_sha_binds_exact_ppo_recipe(train_mod):
    agent_cfg = NS(
        to_dict=lambda: {
            "num_steps_per_env": 24,
            "empirical_normalization": False,
            "policy": {"actor_hidden_dims": [256, 128]},
            "algorithm": {
                "learning_rate": 0.0003,
                "rnd_cfg": None,
                "symmetry_cfg": None,
            },
        }
    )
    recipe = train_mod._action_ball_agent_recipe(agent_cfg)
    preflight = {"policy_contract_sha256": recipe["sha256"]}
    assert (
        recipe["recipe"]["runner"]["init_at_random_ep_len"] is False
    )
    assert (
        train_mod._validate_action_ball_policy_recipe(
            preflight, agent_cfg
        )
        == recipe
    )
    preflight["policy_contract_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="policy contract SHA"):
        train_mod._validate_action_ball_policy_recipe(
            preflight, agent_cfg
        )


def test_action_ball_build_requires_opaque_motion_admission_source(train_mod):
    source = inspect.getsource(train_mod._build_training_hard_contract)
    assert "action_ball_motion_admission_hard_contract" in source
    assert "motion_admission_receipt = admission_fn()" in source
    assert "action_ball_motion_admission_hard_contract() receipt" in source


def test_action_ball_disables_only_initial_episode_randomization(train_mod):
    run_source = TRAIN_PATH.read_text(encoding="utf-8")
    assert (
        "init_at_random_ep_len=not action_ball_training" in run_source
    )
    preflight_source = inspect.getsource(
        train_mod._action_ball_preflight_contract
    )
    assert '"initial_episode_length_randomization": False' in (
        preflight_source
    )
