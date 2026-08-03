"""Host-only regressions for action-ball launch/runtime contract composition."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts" / "train.py"
TRAIN_SOURCE = TRAIN_PATH.read_text(encoding="utf-8")
TRAIN_TREE = ast.parse(TRAIN_SOURCE, filename=str(TRAIN_PATH))


def _train_functions(names, namespace):
    wanted = set(names)
    nodes = [
        node
        for node in TRAIN_TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def test_vendor_hctrl_is_composed_before_live_schema3_extraction():
    override = next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_task_overrides"
    )
    source = ast.get_source_segment(TRAIN_SOURCE, override)
    assert '"physx_control_position_limit_inset_fraction"' in source
    assert "must be the exact vendor-only value 0.02" in source
    assert (
        "env_cfg.actions.joint_pos.physx_control_position_limit_inset_fraction"
        in source
    )
    # The hard contract is extracted only from the fully instantiated env/action term.  The
    # training-contract helper then cross-checks this composed value against the live PhysX
    # getter; no launcher-supplied metadata can mint the block on its own.
    assert TRAIN_SOURCE.count("runtime_execution_facts(env, actor_contract)") >= 2


def test_table_attribution_top_level_override_reapplies_post_init_table_binding():
    override = next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_task_overrides"
    )
    source = ast.get_source_segment(TRAIN_SOURCE, override)
    assert '_get(task, "table_contact_attribution_diagnostic")' in source
    assert '"task.table_contact_attribution_diagnostic"' in source
    assert "_as_explicit_bool(" in source
    assert 'env_cfg.table_contact_attribution_diagnostic = _enabled' in source
    assert 'getattr(env_cfg, "table_robot_keepout", False) is True' in source
    assert '_params.get("attribution_diagnostic") is _enabled' in source
    assert '_params.get("attribution_command_name") == "racket_target"' in source
    # Both table_obstacle and the later diagnostic override must re-run the
    # idempotent installer because Hydra task.* overrides arrive after
    # env_cfg.__post_init__ has already constructed the DoneTerm.
    assert source.count("_apply_table(env_cfg)") == 2


def test_task_first_safety_gate_binds_table_attribution_params_exactly():
    validator = next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_task_first_safety_semantics"
    )
    source = ast.get_source_segment(TRAIN_SOURCE, validator)
    assert "_validate_task_first_table_attribution_params(" in source
    assert "action_ball=action_ball" in source


def test_action_set_identity_makes_resume_reject_profile_n_order_and_manifest_drift():
    (contract_diff,) = _train_functions(["_contract_diff"], {})
    base = {
        "action_ball_training": {
            "action_set_identity": {
                "profile_id": "profile_n2",
                "expected_n": 2,
                "ordered_action_ids": ["a", "b"],
                "ordered_action_uids": [11, 22],
                "manifest_sha256": "a" * 64,
            }
        }
    }
    variants = []
    for key, value in (
        ("profile_id", "other_profile"),
        ("expected_n", 1),
        ("ordered_action_ids", ["b", "a"]),
        ("manifest_sha256", "b" * 64),
    ):
        variant = json.loads(json.dumps(base))
        variant["action_ball_training"]["action_set_identity"][key] = value
        variants.append((key, variant))
    for key, variant in variants:
        diffs = contract_diff(base, variant)
        assert diffs
        assert any(
            f"action_ball_training.action_set_identity.{key}" in item
            for item in diffs
        )


def test_formal_action_set_claim_is_verified_before_scene_construction():
    load = TRAIN_SOURCE.index(
        "load_action_ball_action_set_identity_from_launch_claim("
    )
    required = TRAIN_SOURCE.index(
        "training_launch_claim_path/SHA and a verified code-owned"
    )
    scene = TRAIN_SOURCE.index("env = gym.make(task_id")
    assert load < required < scene


def test_action_ball_manifest_scope_owns_effective_body_imitation_recipe():
    finalizer = next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finalize_action_ball_training_cfg"
    )
    source = ast.get_source_segment(TRAIN_SOURCE, finalizer)
    assert "manifest.prototype.scope" in source
    assert "task.rewards.full_body_mimic" in source
    assert "configured_full_body is not expected_full_body" in source
    for body_name in (
        "pelvis_link",
        "left_hip_roll_Link",
        "left_knee_Link",
        "left_ankle_roll_Link",
        "right_hip_roll_Link",
        "right_knee_Link",
        "right_ankle_roll_Link",
    ):
        assert body_name in source
    assert "upper-scope ActionBall must not imitate lower-body" in source
    assert "full-scope ActionBall must include the exact" in source


def test_only_trainable_a211_fixed_questions_keep_exact_strike_curriculum():
    finalizer = next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finalize_action_ball_training_cfg"
    )
    source = ast.get_source_segment(TRAIN_SOURCE, finalizer)
    assert "if a211_trainable:" in source
    assert "if not bool(getattr(racket_cfg, attr, False))" in source
    assert 'adaptive_sigma_source != "ball_exact_strike"' in source
    assert "trainable A211 fixed-question curriculum requires" in source
    assert "fixed-question target ablations other than" in source
    assert "trainable A211 require" in source


def test_formal_runtime_hook_is_ready_and_owns_exact_sidecar_signature():
    command_path = (
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "hope_commands.py"
    )
    source = command_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(command_path))
    command = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RacketTargetCommand"
    )
    ready = next(
        node
        for node in command.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id
            == "ACTION_BALL_FROZEN_EVALUATOR_RUNTIME_V1_READY"
            for target in node.targets
        )
    )
    assert isinstance(ready.value, ast.Constant)
    assert ready.value.value is True
    hook = next(
        node
        for node in command.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "action_ball_frozen_evaluator_execute_v1"
    )
    assert [argument.arg for argument in hook.args.kwonlyargs] == [
        "request_document",
        "vector_env",
        "runner",
        "deterministic_policy",
        "expected_task_id",
        "expected_policy_generation",
        "expected_proposal_sampler_contract_sha256",
        "progress_callback",
        "request_deadline_monotonic_ns",
    ]
    hook_source = ast.get_source_segment(source, hook)
    assert "sample_frozen_evaluation_proposal(" in hook_source
    assert "cross_engine_physical_truth(" in hook_source
    assert "_action_ball_pool.request" not in hook_source
    assert "reward" not in hook_source.lower()
    solver = next(
        node
        for node in command.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_action_ball_frozen_eval_solve"
    )
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "solve_proposals"
        for node in ast.walk(solver)
    ) == 1
    solver_source = ast.get_source_segment(source, solver)
    assert "host_packet = result.proposal_host_packet" in solver_source
    assert "residual_rows = host_packet.residual_rows" in solver_source
    assert ".detach().cpu()" not in solver_source
    assert ".item()" not in solver_source


def test_formal_runtime_refreshes_installed_reference_and_covers_close_tick():
    command_path = (
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "hope_commands.py"
    )
    source = command_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(command_path))
    command = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RacketTargetCommand"
    )
    hook = next(
        node
        for node in command.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "action_ball_frozen_evaluator_execute_v1"
    )
    refresh = next(
        node
        for node in command.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_action_ball_frozen_eval_refresh_motion_reference"
    )
    hook_source = ast.get_source_segment(source, hook)
    refresh_source = ast.get_source_segment(source, refresh)
    assert (
        hook_source.index("_action_ball_frozen_eval_install(")
        < hook_source.index("scene.update(0.0)")
        < hook_source.index(
            "_action_ball_frozen_eval_refresh_motion_reference("
        )
        < hook_source.index("_compute_racket_state()")
        < hook_source.index("get_observations()")
    )
    assert (
        hook_source.index("_compute_strike_timing()")
        < hook_source.index("get_observations()")
    )
    assert "motion.body_pos_relative_w[env_ids] =" in refresh_source
    assert "motion.body_quat_relative_w[env_ids] =" in refresh_source
    assert "_update_command(" not in refresh_source
    assert "time_steps" not in refresh_source
    assert "math.ceil(" in hook_source
    assert "math.floor(" not in hook_source
    budget_anchor = hook_source.index("budgets =")
    loop_anchor = hook_source.index(
        "for step_index in range(max(budgets))"
    )
    budget_source = hook_source[budget_anchor:loop_anchor]
    assert "pre_swing_wait_s" in budget_source
    assert "scaled_t_cycle_s" in budget_source
    assert "_action_ball_attempt_close_margin_s" in budget_source


def test_frozen_task_receipt_rejects_rehashed_ball_tick_task_and_rate_drift():
    command_path = (
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "hope_commands.py"
    )
    tree = ast.parse(
        command_path.read_text(encoding="utf-8"),
        filename=str(command_path),
    )
    wanted = {
        "_action_ball_canonical_sha256",
        "_action_ball_frozen_eval_receipt",
        "_action_ball_assert_frozen_eval_receipt",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"hashlib": hashlib, "json": json}
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(command_path), "exec"), namespace)
    issue = namespace["_action_ball_frozen_eval_receipt"]
    verify = namespace["_action_ball_assert_frozen_eval_receipt"]
    kind = "action_ball_frozen_evaluation_task"
    expected = {
        "sampler_sample_receipt": {
            "incoming_velocity_w_mps": [-4.0, 0.0, 0.2],
        },
        "task": {"racket_site_target_w_m": [0.2, 0.3, 1.0]},
        "teacher": {
            "time_to_contact_tick": 41,
            "teacher_rate": 1.1,
        },
    }
    receipt = issue(kind=kind, content=expected)
    assert verify(
        receipt, kind=kind, expected_content=expected
    ) == receipt["receipt_sha256"]
    mutations = (
        ("sampler_sample_receipt", "incoming_velocity_w_mps", [-3.9, 0.0, 0.2]),
        ("task", "racket_site_target_w_m", [0.2, 0.4, 1.0]),
        ("teacher", "time_to_contact_tick", 42),
        ("teacher", "teacher_rate", 1.2),
    )
    for group, field, value in mutations:
        forged_content = json.loads(json.dumps(expected))
        forged_content[group][field] = value
        forged = issue(kind=kind, content=forged_content)
        with pytest.raises(ValueError, match="live proposal/task source"):
            verify(forged, kind=kind, expected_content=expected)


def test_action_ball_preflight_binds_production_mixture_and_policy_tick():
    preflight = next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_action_ball_preflight_contract"
    )
    assert any(
        argument.arg == "policy_dt_s"
        for argument in preflight.args.kwonlyargs
    )
    sampler_calls = [
        node
        for node in ast.walk(preflight)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ActionBallSampler"
    ]
    assert len(sampler_calls) == 1
    keywords = {
        keyword.arg: keyword.value
        for keyword in sampler_calls[0].keywords
        if keyword.arg is not None
    }
    assert set(keywords) == {
        "seed",
        "sampling_mixture",
        "contact_time_step_s",
    }
    mixture = keywords["sampling_mixture"]
    assert (
        isinstance(mixture, ast.Call)
        and isinstance(mixture.func, ast.Name)
        and mixture.func.id == "SamplingMixture"
        and not mixture.args
        and not mixture.keywords
    )
    step = keywords["contact_time_step_s"]
    assert (
        isinstance(step, ast.Call)
        and isinstance(step.func, ast.Name)
        and step.func.id == "float"
        and len(step.args) == 1
        and isinstance(step.args[0], ast.Name)
        and step.args[0].id == "policy_dt_s"
    )


def test_preflight_reads_the_same_float32_nonzero_ready_root_z_as_runtime(
    tmp_path,
):
    class OverrideError(RuntimeError):
        pass

    (read_ready_z,) = _train_functions(
        ("_action_ball_ready_root_z_by_slot",),
        {
            "hashlib": hashlib,
            "math": math,
            "_OverrideError": OverrideError,
        },
    )
    motion = tmp_path / "motion.npz"
    body_pos_w = np.zeros((2, 2, 3), dtype=np.float64)
    body_pos_w[0, 1, 2] = 0.9123456789
    np.savez(
        motion,
        body_names=np.asarray(["torso_Link", "pelvis_link"]),
        body_pos_w=body_pos_w,
    )
    raw_sha256 = hashlib.sha256(motion.read_bytes()).hexdigest()
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(
            actions=(SimpleNamespace(action_id="backhand"),)
        ),
        referenced_assets=SimpleNamespace(
            motions=(
                SimpleNamespace(
                    resolved_path=motion,
                    sha256=raw_sha256,
                ),
            )
        ),
    )

    actual = read_ready_z(
        loaded,
        SimpleNamespace(body_names=["pelvis_link", "torso_Link"]),
    )

    assert actual == (float(np.float32(0.9123456789)),)
    assert actual[0] != 0.0


def test_diagnostic_authorization_is_explicit_and_never_formal_lineage():
    (
        exact_dict,
        authorization_contract,
        validate_authorization,
        lineage_exact,
    ) = _train_functions(
        (
            "_action_ball_exact_dict",
            "_action_ball_training_authorization_contract",
            "_validate_action_ball_training_authorization",
            "_action_ball_contract_lineage_exact",
        ),
        {},
    )
    del exact_dict

    diagnostic = {
        "authorization": authorization_contract(True),
        "runtime": {
            "diagnostic_unauthorized": True,
            "evaluator_authority": {
                "diagnostic_unauthorized": True,
                "formal_authority_available": False,
                "formal_launch_requires_code_pinned_receipt": True,
                "runtime_or_manifest_may_self_authorize": False,
                "authority_binding": {"kind": "diagnostic"},
                "authority_state_owner_sha256": _digest(
                    "diagnostic-state-owner"
                ),
            },
        },
        "motion_admission": {
            "diagnostic_unauthorized": True,
            "training_authorized": False,
        },
    }
    assert validate_authorization(diagnostic) is True
    assert all(diagnostic["authorization"].values())
    assert (
        lineage_exact(
            source_lineage_exact=True,
            motion_kinematics_exact=True,
            diagnostic_unauthorized=True,
        )
        is False
    )

    formal = {
        "authorization": authorization_contract(False),
        "runtime": {
            "evaluator_authority": {
                "formal_authority_available": True,
            }
        },
        "motion_admission": {"authorization_purpose": "training"},
    }
    assert validate_authorization(formal) is False
    assert not any(formal["authorization"].values())
    assert (
        lineage_exact(
            source_lineage_exact=True,
            motion_kinematics_exact=True,
            diagnostic_unauthorized=False,
        )
        is True
    )
    assert (
        lineage_exact(
            source_lineage_exact=False,
            motion_kinematics_exact=True,
            diagnostic_unauthorized=False,
        )
        is False
    )

    contradictory = json.loads(json.dumps(diagnostic))
    contradictory["authorization"]["formal_evidence_prohibited"] = False
    with pytest.raises(RuntimeError, match="contradictory"):
        validate_authorization(contradictory)

    cross_view_drift = json.loads(json.dumps(diagnostic))
    cross_view_drift["runtime"]["diagnostic_unauthorized"] = False
    with pytest.raises(RuntimeError, match="disagrees across"):
        validate_authorization(cross_view_drift)

    with pytest.raises(RuntimeError, match="exact bool"):
        lineage_exact(
            source_lineage_exact=1,
            motion_kinematics_exact=True,
            diagnostic_unauthorized=False,
        )


def _install_action_ball_modules(monkeypatch, *, arm_catalog, authority, trust):
    package_names = (
        "whole_body_tracking",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.tracking",
        "whole_body_tracking.tasks.tracking.mdp",
    )
    packages = {}
    for name in package_names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        packages[name] = module
    sampling = ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_sampling"
    )
    sampling.ARM_CATALOG_SHA256 = arm_catalog
    evaluator = ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_evaluation"
    )
    evaluator.FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256 = authority
    evaluator.TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256 = frozenset(
        trust
    )
    runtime = ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_runtime"
    )
    runtime.BROKER_STATE_SCHEMA_VERSION = 4
    runtime.TASK_RECEIPT_TIMING_AUTHORITY = (
        "per_swing_task_receipt_v5_exact_face_contact"
    )
    reference_guard = ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_reference_guard"
    )
    reference_guard.REFERENCE_GUARD_CONTRACT_PAYLOAD = {
        "kind": "test-reference-guard",
        "schema_version": 1,
    }
    reference_guard.REFERENCE_GUARD_CONTRACT_SHA256 = _canonical(
        reference_guard.REFERENCE_GUARD_CONTRACT_PAYLOAD
    )

    def validate_reference_guard_mode(value):
        if value not in ("phase_gated", "metrics_only"):
            raise ValueError("invalid reference_guard_mode")
        return value

    reference_guard.validate_reference_guard_mode = (
        validate_reference_guard_mode
    )
    monkeypatch.setitem(sys.modules, sampling.__name__, sampling)
    monkeypatch.setitem(sys.modules, evaluator.__name__, evaluator)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setitem(
        sys.modules, reference_guard.__name__, reference_guard
    )
    packages[
        "whole_body_tracking.tasks.tracking.mdp"
    ].action_ball_evaluation = evaluator
    packages[
        "whole_body_tracking.tasks.tracking.mdp"
    ].action_ball_runtime = runtime
    return reference_guard


def test_formal_cross_check_accepts_the_complete_runtime_payload(monkeypatch):
    arm_catalog = _digest("arm-catalog")
    evaluator_authority_sha = _digest("evaluator-authority")
    evaluator_launch_sha = _digest("evaluator-launch")
    reference_guard = _install_action_ball_modules(
        monkeypatch,
        arm_catalog=arm_catalog,
        authority=evaluator_authority_sha,
        trust=(evaluator_launch_sha,),
    )

    namespace = {
        "hashlib": hashlib,
        "json": json,
        "math": math,
    }
    (
        canonical,
        exact_dict,
        sha256,
        json_equal,
        assert_json_equal,
        content_receipt,
        validate_reference_guard_contract,
        validate,
    ) = _train_functions(
        (
            "_canonical_contract_sha256",
            "_action_ball_exact_dict",
            "_action_ball_sha256",
            "_action_ball_json_equal",
            "_action_ball_assert_json_equal",
            "_action_ball_content_receipt",
            "_validate_action_ball_reference_guard_contract",
            "_validate_action_ball_runtime_hard_contract",
        ),
        namespace,
    )
    del (
        exact_dict,
        sha256,
        json_equal,
        assert_json_equal,
        content_receipt,
        validate_reference_guard_contract,
    )

    manifest_sha = _digest("manifest")
    manifest_canonical_sha = _digest("manifest-canonical")
    motion_sha = _digest("motion")
    profile_sha = _digest("profile-with-nonzero-ready-z")
    adapter_sha = _digest("adapter")
    sampler_sha = _digest("sampler")
    policy_sha = _digest("policy")
    prototype_sha = _digest("prototype")
    evaluator_file_sha = _digest("evaluator-file")
    solver_payload = {
        "physics_profile_sha256": _digest("physics-placeholder")
    }
    physics_payload = {"kind": "physics"}
    physics_sha = canonical(physics_payload)
    solver_payload["physics_profile_sha256"] = physics_sha
    solver_sha = canonical(solver_payload)
    action_uid = 19
    binding = {
        "action_uid": action_uid,
        "action_slot": 0,
        "motion_path": "motions/backhand.npz",
        "motion_sha256": motion_sha,
        "profile_sha256": profile_sha,
    }
    curriculum_config = {"target_failure_rate": 0.1}
    preflight = {
        "schema_version": 1,
        "manifest": {
            "path": "configs/action_ball.json",
            "file_sha256": manifest_sha,
            "canonical_sha256": manifest_canonical_sha,
            "manifest_id": "formal",
        },
        "mobility_mode": "no_move",
        "action_order": ["backhand"],
        "action_uids": [action_uid],
        "ready_root_z_by_slot_m": [0.91],
        "action_bindings": [
            {
                "action_id": "backhand",
                "action_uid": action_uid,
                "action_slot": 0,
                "family": "backhand",
                "motion_path": "motions/backhand.npz",
                "motion_sha256": motion_sha,
                "sampling_profile_sha256": profile_sha,
                "strike_phase": 0.5,
                "mount_normal_sign": -1,
            }
        ],
        "prototype": {
            "path": "configs/prototype.json",
            "scope": "upper",
            "sha256": prototype_sha,
        },
        "profile_adapter": {
            "contract": {"schema_version": 2},
            "sha256": adapter_sha,
        },
        "sampler": {
            "contract_sha256": sampler_sha,
            "arm_catalog_sha256": arm_catalog,
            "seed": 7,
            "pool_refill_rows": 16,
        },
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "curriculum": {
            "config": curriculum_config,
            "config_sha256": canonical(curriculum_config),
        },
        "holdout": {"samples_per_action": 768},
        "fixed_direction": True,
        "initial_episode_length_randomization": False,
        "policy_contract_sha256": policy_sha,
        "evaluator_launch": {
            "path": "configs/evaluator.json",
            "file_sha256": evaluator_file_sha,
        },
    }
    preflight["sha256"] = canonical(preflight)

    domain_sources = {
        name: _digest(name)
        for name in (
            "hope_commands.py",
            "action_ball_curriculum.py",
            "action_ball_runtime.py",
        )
    }
    domain_payload = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.domain_claim_authority",
        "implementation_source_sha256": domain_sources,
        "manifest_sha256": manifest_sha,
        "adapter_contract_sha256": adapter_sha,
        "action_uids": [action_uid],
        "profile_sha256": [profile_sha],
        "mobility_mode": "no_move",
        "curriculum_config": curriculum_config,
        "policy_contract_sha256": policy_sha,
        "schedule": {
            "claim_barrier": "true_reset_only",
            "domain_source": (
                "frozen_ActionBallCurriculum.expected_domains"
            ),
            "selection": "per_action_round_robin",
            "training_selector": False,
            "live_rollout_updates_curriculum": False,
        },
    }
    domain_sha = canonical(domain_payload)
    state_schema = 6
    state_owner_sha = canonical(
        {
            "schema_version": state_schema,
            "kind": (
                "whole_body_tracking.RacketTargetCommand."
                "action_ball_mutable_state_owner"
            ),
            "action_uids": [action_uid],
            "sampler_contract_sha256": sampler_sha,
            "domain_authority_contract_sha256": domain_sha,
            "solver_contract_sha256": solver_sha,
        }
    )
    runtime_contract_sha = _digest("runtime-contract")
    registry_sha = canonical(
        {
            "runtime_contract_sha256": runtime_contract_sha,
            "pins": {
                "manifest_sha256": manifest_sha,
                "sampler_sha256": sampler_sha,
                "domain_authority_sha256": domain_sha,
                "physics_sha256": physics_sha,
                "solver_sha256": solver_sha,
            },
            "mobility_mode": "no_move",
            "bindings": [binding],
        }
    )
    implementation_sources = {
        name: _digest(f"runtime-{name}")
        for name in (
            "hope_commands.py",
            "action_ball_curriculum.py",
            "action_ball_evaluation.py",
            "action_ball_manifest.py",
            "action_ball_profile_adapter.py",
            "action_ball_runtime.py",
                "action_ball_sampling.py",
                "continuous_questions.py",
                "racket_contact_geometry.py",
                "stroke_adapt_torch.py",
                "virtual_ball.py",
        )
    }
    evaluator_verified = {
        "path": "configs/evaluator.json",
        "file_sha256": evaluator_file_sha,
        "launch_receipt": {"kind": "frozen-evaluator"},
        "launch_receipt_canonical_sha256": evaluator_launch_sha,
        "authority_binding": {"binding": "exact"},
        "authority_state_owner_sha256": _digest(
            "evaluator-state-owner"
        ),
        "attempt_source_state_owner_sha256": _digest(
            "attempt-source-state-owner"
        ),
        "coordinator_state_owner_sha256": _digest(
            "coordinator-state-owner"
        ),
        "inbox_root": "/tmp/action-ball-inbox",
        "inbox_owner_id": "owner",
        "inbox_run_id": "run",
        "sidecar_launch_receipt_path": "configs/sidecar.json",
        "sidecar_launch_receipt_file_sha256": _digest("sidecar-file"),
        "sidecar_launch_receipt_content_sha256": _digest(
            "sidecar-content"
        ),
        "sidecar_code_path": "scripts/action_ball_frozen_eval_sidecar.py",
        "sidecar_code_sha256": _digest("sidecar-code"),
        "drain_reset_launch": {
            "kind": "frozen-evaluator-drain-reset",
            "sha256": _digest("drain-reset"),
        },
    }
    namespace["_action_ball_repo_root"] = lambda motion_cfg: ROOT
    namespace["_validate_action_ball_mdp_source_map"] = (
        lambda value, **kwargs: value
    )
    admission_calls = []

    def validate_motion_admission(value, **kwargs):
        admission_calls.append((value, kwargs))
        return value

    namespace["_validate_action_ball_motion_admission_receipt"] = (
        validate_motion_admission
    )
    namespace["_load_action_ball_evaluator_launch_from_cfg"] = (
        lambda *args, **kwargs: evaluator_verified
    )

    motion_cfg = SimpleNamespace(
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=11,
        hold_steps_range=(0, 0),
        stand_start_min_hold=0,
        post_swing_min_hold=0,
        stagger_initial_clock=False,
        speed_scale_range=(1.0, 1.0),
        speed_scale_per_clip=None,
        planner_revision_enabled=False,
    )
    racket_cfg = SimpleNamespace(
        cq_overdraw=2.0,
        cq_max_redraw_rounds=5,
        action_ball_frozen_eval_interval_updates=25,
        reference_guard_mode="phase_gated",
    )
    runtime = {
        "schema_version": 1,
        "kind": (
            "whole_body_tracking.RacketTargetCommand."
            "action_ball_hard_contract"
        ),
        "manifest": preflight["manifest"],
        "mobility_mode": "no_move",
        "action_order": ["backhand"],
        "action_uids": [action_uid],
        "bindings": [binding],
        "prototype": preflight["prototype"],
        "profiles": {
            "adapter_contract_sha256": adapter_sha,
            "profile_sha256": [profile_sha],
            "arm_catalog_sha256": arm_catalog,
            "sampler_contract_sha256": sampler_sha,
        },
        "sampling": {
            "action_ball_seed": 7,
            "pool_refill_rows": 16,
            "balanced_clip_sampling": True,
            "balanced_clip_sampling_seed": 11,
            "external_overdraw_multiplier": 2.0,
            "maximum_external_proposal_rounds": 5,
        },
        "timing": {
            "authority": (
                "per_swing_task_receipt_v5_exact_face_contact"
            ),
            "policy_dt_s": 0.02,
            "attempt_close_margin_s": 0.02,
            "episode_length_s": 10.0,
            "time_to_strike_source": (
                "MotionCommand.action_ball_time_to_contact_remaining_s"
            ),
            "legacy_motion_time_owners": {
                "hold_steps_range": [0, 0],
                "stand_start_min_hold": 0,
                "post_swing_min_hold": 0,
                "stagger_initial_clock": False,
                "speed_scale_range": [1.0, 1.0],
                "speed_scale_per_clip": None,
                "planner_revision_enabled": False,
            },
        },
        "reference_guard": {
            "mode": "phase_gated",
            "contract_payload": (
                reference_guard.REFERENCE_GUARD_CONTRACT_PAYLOAD
            ),
            "contract_sha256": (
                reference_guard.REFERENCE_GUARD_CONTRACT_SHA256
            ),
        },
        "solver": {"payload": solver_payload, "sha256": solver_sha},
        "physics": {"payload": physics_payload, "sha256": physics_sha},
        "domain_authority": {
            "payload": domain_payload,
            "sha256": domain_sha,
        },
        "mutable_state_owner": {
            "schema_version": state_schema,
            "state_owner_sha256": state_owner_sha,
            "protocol_views": [
                "domain_claim_authority",
                "birth_provider",
                "task_solver",
            ],
            "checkpoint_state_is_mutable": True,
            "mutable_state_sha256_is_not_a_hard_contract_pin": True,
        },
        "curriculum": {
            "config": curriculum_config,
            "policy_contract_sha256": policy_sha,
            "frozen_checkpoint_evidence_required": True,
            "live_rollout_advances_curriculum": False,
        },
        "evaluator_authority": {
            "authority_contract_sha256": evaluator_authority_sha,
            "trusted_launch_receipt_sha256": [evaluator_launch_sha],
            "evaluator_launch_receipt_path": evaluator_verified["path"],
            "evaluator_launch_receipt_file_sha256": evaluator_file_sha,
            "evaluator_launch_receipt": evaluator_verified[
                "launch_receipt"
            ],
            "launch_receipt_canonical_sha256": evaluator_launch_sha,
            "authority_binding": evaluator_verified["authority_binding"],
            "authority_state_owner_sha256": evaluator_verified[
                "authority_state_owner_sha256"
            ],
            "attempt_source_state_owner_sha256": evaluator_verified[
                "attempt_source_state_owner_sha256"
            ],
            "coordinator_state_owner_sha256": evaluator_verified[
                "coordinator_state_owner_sha256"
            ],
            "inbox_root": evaluator_verified["inbox_root"],
            "inbox_owner_id": evaluator_verified["inbox_owner_id"],
            "inbox_run_id": evaluator_verified["inbox_run_id"],
            "sidecar_launch_receipt_path": evaluator_verified[
                "sidecar_launch_receipt_path"
            ],
            "sidecar_launch_receipt_file_sha256": evaluator_verified[
                "sidecar_launch_receipt_file_sha256"
            ],
            "sidecar_launch_receipt_content_sha256": evaluator_verified[
                "sidecar_launch_receipt_content_sha256"
            ],
            "sidecar_code_path": evaluator_verified[
                "sidecar_code_path"
            ],
            "sidecar_code_sha256": evaluator_verified[
                "sidecar_code_sha256"
            ],
            "drain_reset": evaluator_verified["drain_reset_launch"],
            "evaluation_interval_updates": 25,
            "formal_authority_available": True,
            "formal_launch_requires_code_pinned_receipt": True,
            "runtime_or_manifest_may_self_authorize": False,
        },
        "runtime": {
            "runtime_contract_sha256": runtime_contract_sha,
            "registry_sha256": registry_sha,
            "implementation_source_sha256": implementation_sources,
            "fixed_direction": True,
            "wrap_teleport": False,
        },
        "motion_admission": {"opaque": "identity"},
    }
    runtime["canonical_sha256"] = canonical(runtime)

    assert validate(
        runtime,
        preflight=preflight,
        racket_cfg=racket_cfg,
        motion_cfg=motion_cfg,
        expected_runtime_contract_sha256=runtime_contract_sha,
    ) == runtime
    assert admission_calls == [
        (
            runtime["motion_admission"],
            {
                "preflight": preflight,
                "motion_cfg": motion_cfg,
                "expected_runtime_contract_sha256": runtime_contract_sha,
                "expected_broker_state_schema_version": 4,
                "expected_broker_registry_sha256": registry_sha,
                "expected_provider_state_owner_sha256": state_owner_sha,
            },
        )
    ]

    drifted = json.loads(json.dumps(runtime))
    drifted["profiles"]["profile_sha256"] = [_digest("zero-ready-z")]
    drifted["canonical_sha256"] = canonical(
        {
            key: value
            for key, value in drifted.items()
            if key != "canonical_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="runtime profiles disagrees"):
        validate(
            drifted,
            preflight=preflight,
            racket_cfg=racket_cfg,
            motion_cfg=motion_cfg,
            expected_runtime_contract_sha256=runtime_contract_sha,
        )
