"""Host-only seams for action -> ball -> fixed-action task runtime wiring.

The Isaac command module is intentionally not imported on developer hosts.  We execute its pure
contract builders from the shipped AST and statically pin the simulator-facing transaction order.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMAND_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "hope_commands.py"
)
MOTION_COMMAND_PATH = COMMAND_PATH.with_name("commands.py")
SOURCE = COMMAND_PATH.read_text(encoding="utf-8")
MOTION_SOURCE = MOTION_COMMAND_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(COMMAND_PATH))
COMMAND_CLASS = next(
    node
    for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
)


def _module_functions(names, namespace):
    wanted = set(names)
    nodes = [
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
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
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def _module_class(name, namespace):
    node = next(
        node
        for node in TREE.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return namespace[name]


def _method(name):
    return next(
        node
        for node in COMMAND_CLASS.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _method_source(name):
    return ast.get_source_segment(SOURCE, _method(name))


def _valid_recipe(**overrides):
    values = {
        "target_mode": "action_ball",
        "question_bank": "",
        "question_bank_allow_legacy": False,
        "cq_anchor_bank": "",
        "exam_bank": "",
        "achieved_target_mix_prob": 0.0,
        "midswing_resample_prob": 0.0,
        "planner_revision_enabled": False,
        "target_delay_steps": 0,
        "target_delay_tts_mode": "live",
        "target_jitter_pos_per_s": 0.0,
        "target_jitter_vel_per_s": 0.0,
        "target_noise_white": 0.0,
        "target_noise_ar1_sigma": 0.0,
        "target_dropout_prob": 0.0,
        "target_post_strike_dropout_s": 0.0,
        "target_bias_per_swing": 0.0,
        "physical_ball": False,
        "physical_ball_impulse": False,
        "shadow_ball": False,
        "shadow_table": False,
        "virtual_ball": True,
        "vb_metrics_only": False,
        "face_command": True,
        "face_command_pairing": "shared_plus_y",
        "action_ball_fixed_direction": True,
        "clip_names_per_clip": ("fh_drive",),
        "action_ball_manifest_path": "configs/action_ball.json",
        "action_ball_manifest_sha256": "a" * 64,
        "action_ball_policy_contract_sha256": "b" * 64,
        "action_ball_evaluator_launch_receipt_path": (
            "configs/action_ball_evaluator_launch.json"
        ),
        "action_ball_evaluator_launch_receipt_file_sha256": "c" * 64,
        "action_ball_sidecar_launch_receipt_path": (
            "configs/action_ball_sidecar_launch.json"
        ),
        "action_ball_sidecar_launch_receipt_file_sha256": "d" * 64,
        "action_ball_drain_reset_launch_receipt_path": (
            "configs/action_ball_drain_reset_launch.json"
        ),
        "action_ball_drain_reset_launch_receipt_file_sha256": "e" * 64,
        "action_ball_evaluation_inbox_root": "/tmp/action-ball-eval",
        "action_ball_evaluation_owner_id": "Franco",
        "action_ball_evaluation_run_id": "run-001",
        "action_ball_frozen_eval_interval_updates": 100,
        "action_ball_seed": 7,
        "action_ball_pool_refill_rows": 16,
        "cq_n_iters": 24,
        "cq_max_redraw_rounds": 5,
        "cq_tol_m": 0.08,
        "cq_speed_budget": 3.4,
        "cq_overdraw": 2.0,
        "vb_rollout_h": 0.002,
        "vb_rollout_steps": 1000,
        "vb_capture_radius": 0.09,
        "vb_min_approach_speed": 0.25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_action_ball_recipe_is_independent_and_requires_one_virtual_scorer():
    (check,) = _module_functions(
        ("_assert_action_ball_recipe_is_coherent",),
        {"math": math, "Path": Path},
    )
    check(_valid_recipe())
    check(
        _valid_recipe(
            virtual_ball=False,
            vb_metrics_only=True,
        )
    )
    with pytest.raises(ValueError, match="authoritative virtual scorer"):
        check(
            _valid_recipe(
                virtual_ball=False,
                vb_metrics_only=False,
            )
        )
    for field, value in {
        "question_bank": "old_bank.npz",
        "physical_ball": True,
        "planner_revision_enabled": True,
        "action_ball_fixed_direction": False,
        "face_command": False,
    }.items():
        with pytest.raises(ValueError, match="incoherent launch recipe"):
            check(_valid_recipe(**{field: value}))

    # The explicit action-ball guard must not reinterpret legacy task_first.
    check(SimpleNamespace(target_mode="task_first"))


def test_diagnostic_action_ball_omits_formal_evaluator_stack_only():
    (check,) = _module_functions(
        ("_assert_action_ball_recipe_is_coherent",),
        {"math": math, "Path": Path},
    )
    omitted = {
        "action_ball_evaluator_launch_receipt_path": "",
        "action_ball_evaluator_launch_receipt_file_sha256": "",
        "action_ball_sidecar_launch_receipt_path": "",
        "action_ball_sidecar_launch_receipt_file_sha256": "",
        "action_ball_drain_reset_launch_receipt_path": "",
        "action_ball_drain_reset_launch_receipt_file_sha256": "",
        "action_ball_evaluation_inbox_root": "",
        "action_ball_evaluation_owner_id": "",
        "action_ball_evaluation_run_id": "",
    }
    check(
        _valid_recipe(
            action_ball_diagnostic_unauthorized=True,
            **omitted,
        )
    )
    with pytest.raises(
        ValueError,
        match="action_ball_evaluator_launch_receipt_path is empty",
    ):
        check(
            _valid_recipe(
                action_ball_diagnostic_unauthorized=False,
                **omitted,
            )
        )
    with pytest.raises(
        ValueError,
        match="action_ball_diagnostic_unauthorized must be an exact boolean",
    ):
        check(
            _valid_recipe(
                action_ball_diagnostic_unauthorized=1,
                **omitted,
            )
        )


def test_action_ball_runtime_waits_for_command_manager_construction():
    method = _method("_ensure_action_ball_runtime_initialized")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    ensure = namespace["_ensure_action_ball_runtime_initialized"]

    calls = []
    command = SimpleNamespace(
        _action_ball_enabled=True,
        _action_ball_runtime_initialized=False,
        _action_ball_runtime_initializing=False,
        _env=SimpleNamespace(),
        _initialize_action_ball_runtime=lambda: calls.append("initialized"),
    )
    with pytest.raises(RuntimeError, match="CommandManager construction"):
        ensure(command)
    assert calls == []
    command._env.command_manager = object()
    ensure(command)
    assert calls == ["initialized"]
    assert command._action_ball_runtime_initialized is True
    assert command._action_ball_runtime_initializing is False
    ensure(command)
    assert calls == ["initialized"]

    init_source = _method_source("__init__")
    assert "self._action_ball_runtime_initialized = False" in init_source
    for caller in (
        "action_ball_hard_contract",
        "_resample_command",
        "_update_command",
        "_update_metrics",
    ):
        assert (
            "self._ensure_action_ball_runtime_initialized()"
            in _method_source(caller)
        )


def test_diagnostic_motion_payload_is_snapshotted_before_runtime_binding(
    tmp_path,
):
    motion_tree = ast.parse(MOTION_SOURCE, filename=str(MOTION_COMMAND_PATH))
    motion_class = next(
        node
        for node in motion_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MotionCommand"
    )
    snapshot_method = next(
        node
        for node in motion_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_snapshot_diagnostic_motion_bytes"
    )
    module = ast.Module(body=[snapshot_method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Path": Path, "hashlib": hashlib}
    exec(compile(module, str(MOTION_COMMAND_PATH), "exec"), namespace)
    snapshot = namespace["_snapshot_diagnostic_motion_bytes"]

    motion_path = tmp_path / "motion.npz"
    payload = b"diagnostic-motion-bytes"
    motion_path.write_bytes(payload)
    command = SimpleNamespace(
        _motion_files=(str(motion_path),),
        _motion_file_sha256=(hashlib.sha256(payload).hexdigest(),),
    )
    assert snapshot(command) == (payload,)

    command._motion_file_sha256 = ("0" * 64,)
    with pytest.raises(
        ValueError,
        match="changed between initial hashing and MotionLoader adoption",
    ):
        snapshot(command)

    init_method = next(
        node
        for node in motion_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    init_source = ast.get_source_segment(MOTION_SOURCE, init_method)
    diagnostic_branch = init_source[
        init_source.index(
            "if self.canonical_ready_mode and diagnostic_unauthorized:"
        ):
        init_source.index(
            "elif self.canonical_ready_mode:"
        )
    ]
    assert (
        "self._snapshot_diagnostic_motion_bytes()"
        in diagnostic_branch
    )
    assert "self._motion_payloads = None" not in diagnostic_branch


def test_training_builds_deferred_action_ball_contract_before_hook_audit():
    train_source = (
        ROOT / "scripts" / "train.py"
    ).read_text(encoding="utf-8")
    action_branch = train_source[
        train_source.index(
            'str(getattr(racket, "target_mode", "")) == "action_ball"'
        ) :
        train_source.index(
            "from whole_body_tracking.tasks.tracking.mdp.action_ball_runtime import",
            train_source.index(
                'str(getattr(racket, "target_mode", "")) == "action_ball"'
            ),
        )
    ]
    assert action_branch.index(
        "runtime_action_ball = runtime_contract_fn()"
    ) < action_branch.index("non_explicit = []")


def test_action_ball_live_racket_site_geometry_is_fail_closed():
    (check,) = _module_functions(
        ("_assert_action_ball_racket_site_contract",),
        {
            "Sequence": __import__("collections.abc").abc.Sequence,
            "math": math,
        },
    )
    geometry = SimpleNamespace(
        RACKET_SITE_OFFSET_WRIST_M=(0.21021, 0.032078, 0.032036),
    )

    def valid_cfg(**overrides):
        values = {
            "racket_body_name": "pingpang_red_Link",
            "wrist_body_name": "right_wrist_yaw_Link",
            "mount_offset": (0.21021, 0.032078, 0.032036),
            "mount_quat": (1.0, 0.0, 0.0, 0.0),
            "mount_normal_axis": 1,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    check(valid_cfg(), geometry)
    # The tolerance covers serialization noise only; it cannot authorize a
    # second geometric transform.
    check(
        valid_cfg(
            mount_offset=(0.21021 + 5.0e-13, 0.032078, 0.032036)
        ),
        geometry,
    )
    for overrides in (
        {"racket_body_name": "right_wrist_yaw_Link"},
        {"wrist_body_name": "right_wrist_pitch_Link"},
        {
            "mount_offset": (
                0.21021 + 1.0e-9,
                0.032078,
                0.032036,
            )
        },
        {"mount_offset": (math.nan, 0.032078, 0.032036)},
        {"mount_quat": (0.0, 1.0, 0.0, 0.0)},
        {"mount_normal_axis": True},
        {"mount_normal_axis": 2},
    ):
        with pytest.raises(
            ValueError,
            match="live racket-site geometry contract mismatch",
        ):
            check(valid_cfg(**overrides), geometry)

    initialize = _method_source("_initialize_action_ball_runtime")
    assert initialize.index(
        "_assert_action_ball_racket_site_contract"
    ) < initialize.index("load_action_ball_manifest(")


def test_evaluator_launch_file_is_exact_tracked_regular_json(tmp_path):
    strict_json, read_tracked = _module_functions(
        (
            "_action_ball_strict_json_bytes",
            "_action_ball_read_tracked_regular_file",
        ),
        {
            "hashlib": hashlib,
            "json": json,
            "os": __import__("os"),
            "stat": __import__("stat"),
            "Path": Path,
        },
    )
    receipt = tmp_path / "launch.json"
    raw = b'{"schema_version":1,"nested":{"value":2}}\n'
    receipt.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    resolved, relative, loaded = read_tracked(
        repo_root=tmp_path,
        relative_path="launch.json",
        expected_file_sha256=digest,
        name="test receipt",
    )
    assert resolved == receipt.resolve()
    assert relative == "launch.json"
    assert loaded == raw
    assert strict_json(raw, name="test receipt")["nested"] == {"value": 2}

    with pytest.raises(ValueError, match="repeats JSON key"):
        strict_json(
            b'{"schema_version":1,"nested":{"value":1,"value":2}}',
            name="test receipt",
        )
    with pytest.raises(ValueError, match="file SHA mismatch"):
        read_tracked(
            repo_root=tmp_path,
            relative_path="launch.json",
            expected_file_sha256="0" * 64,
            name="test receipt",
        )
    link = tmp_path / "launch-link.json"
    try:
        link.symlink_to(receipt.name)
    except (OSError, NotImplementedError):
        pytest.skip("host cannot create symlinks")
    with pytest.raises(ValueError, match="symbolic link"):
        read_tracked(
            repo_root=tmp_path,
            relative_path=link.name,
            expected_file_sha256=digest,
            name="test receipt",
        )


def test_evaluator_launch_receipt_binds_arm_catalog_and_scheduler():
    loader = ast.get_source_segment(
        SOURCE,
        next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_action_ball_load_evaluator_authority"
        ),
    )
    assert '"arm_catalog_sha256"' in loader
    assert '"scheduler_contract_sha256"' in loader
    assert "13 frozen keys" in loader
    assert "arm_catalog_sha256=arm_catalog_sha256" in loader
    assert (
        "scheduler_contract_sha256=scheduler_contract_sha256"
        in loader
    )


def _solver_cfg(**overrides):
    values = {
        "cq_n_iters": 24,
        "cq_tol_m": 0.08,
        "cq_speed_budget": 3.4,
        "cq_max_redraw_rounds": 5,
        "cq_overdraw": 2.0,
        "vb_rollout_h": 0.002,
        "vb_rollout_steps": 1000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_solver_profile_hashes_executable_speed_face_and_contact_fit_contract():
    canonical, build = _module_functions(
        (
            "_action_ball_canonical_sha256",
            "action_ball_solver_profile_contract",
        ),
        {
            "hashlib": hashlib,
            "json": json,
            "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION": 2,
        },
    )
    sources = {
        "hope_commands.py": "0" * 64,
        "continuous_questions.py": "1" * 64,
        "stroke_adapt_torch.py": "2" * 64,
        "virtual_ball.py": "3" * 64,
        "racket_contact_geometry.py": "5" * 64,
    }
    geometry_contract = {
        "payload": {
            "schema_version": 2,
            "kind": "exact_face_contact_v2",
        },
        "sha256": "6" * 64,
    }
    contract = build(
        _solver_cfg(),
        physics_profile_sha256="4" * 64,
        source_sha256=sources,
        contact_geometry_contract=geometry_contract,
        net_top_z=0.9325,
    )
    # Compatibility fence: adding the exact-N1 objective must not perturb the
    # ordinary N5/N73 solver receipt when the optional identity is absent.
    # This digest was minted from the pre-counter payload represented by the
    # fixed inputs above; keeping it literal catches even subtle key additions.
    assert contract["sha256"] == (
        "c80a115826f7d9628230a2c709d4a36085297053d1b775005482ef3cbbc2975b"
    )
    assert contract == build(
        _solver_cfg(),
        physics_profile_sha256="4" * 64,
        source_sha256=sources,
        contact_geometry_contract=geometry_contract,
        net_top_z=0.9325,
        counter_rally_objective_profile_sha256=None,
        counter_rally_venue_physics_sha256=None,
    )
    assert "counter_rally" not in contract["payload"]
    assert contract["sha256"] == canonical(contract["payload"])
    solve = contract["payload"]["solve"]
    assert solve == {
        "n_iters": 24,
        "tol_m": 0.08,
        "global_speed_budget_mps": 3.4,
        "max_external_proposal_rounds": 5,
        "external_overdraw_multiplier": 2.0,
    }
    acceptance = contract["payload"]["acceptance"]
    assert (
        acceptance["speed"]["upper"]
        == "min_selected_prototype_speed_max_and_global_speed_budget_inclusive"
    )
    assert (
        acceptance["face"]["physical_face_mapping"]
        == "physical_B_face=solved_raw_A_face*selected_prototype_face_sign"
    )
    assert acceptance["contact_normal_speed_fit"] == {
        "definition": (
            "-dot(v_ball_in-v_racket,selected_physical_B_face_normal)"
        ),
        "minimum_mps_inclusive": 1.4,
        "maximum_mps_inclusive": 7.2,
        "rejection_reason": "contact_normal_speed_out_of_fit",
    }
    assert acceptance["ordered_rejection_reason_schema"][-6:] == [
        "contact_normal_speed_out_of_fit",
        "teacher_site_rate_geometry_unsolved",
        "teacher_rate_out_of_bounds",
        "pre_swing_wait_out_of_bounds",
        "cycle_exceeds_episode_horizon",
        "ball_birth_not_beyond_net",
    ]
    assert acceptance["incoming_birth"] == {
        "predicate": (
            "contact_x_w_plus_abs_v_in_x_times_ttc_linear_lower_"
            "bound_at_or_beyond_net_plane_plus_margin"
        ),
        "net_margin_m": 0.05,
        "rejection_reason": "ball_birth_not_beyond_net",
    }
    assert contract["payload"]["batch_semantics"] == {
        "row_separable": True,
        "required_parity": (
            "bitwise_equal_per_row_for_full_single_permuted_and_chunked_batches"
        ),
        "resume_replay_grouping": "arbitrary_pending_receipt_batch",
    }
    assert build(
        _solver_cfg(cq_speed_budget=3.3),
        physics_profile_sha256="4" * 64,
        source_sha256=sources,
        contact_geometry_contract=geometry_contract,
        net_top_z=0.9325,
    )["sha256"] != contract["sha256"]
    assert build(
        _solver_cfg(cq_overdraw=1.5),
        physics_profile_sha256="4" * 64,
        source_sha256=sources,
        contact_geometry_contract=geometry_contract,
        net_top_z=0.9325,
    )["sha256"] != contract["sha256"]
    drifted_geometry = {
        "payload": dict(geometry_contract["payload"]),
        "sha256": "7" * 64,
    }
    assert build(
        _solver_cfg(),
        physics_profile_sha256="4" * 64,
        source_sha256=sources,
        contact_geometry_contract=drifted_geometry,
        net_top_z=0.9325,
    )["sha256"] != contract["sha256"]


def test_counter_rally_solver_contract_appends_exact_ordered_rejections_and_identity(
    monkeypatch,
):
    canonical, build = _module_functions(
        (
            "_action_ball_canonical_sha256",
            "action_ball_solver_profile_contract",
        ),
        {
            "hashlib": hashlib,
            "json": json,
            "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION": 2,
        },
    )
    counter_reasons = (
        "reverse_ray_not_opponent_bound",
        "landing_depth_outside_table",
        "landing_depth_not_opponent_half",
        "landing_behind_contact",
        "reverse_ray_misses_table",
        "incoming_speed_outside_venue_support",
        "target_speed_outside_venue_support",
    )
    counter_module_name = (
        "whole_body_tracking.tasks.tracking.mdp.counter_rally"
    )
    counter_module = types.ModuleType(counter_module_name)
    counter_module.COUNTER_RALLY_SOLVER_REJECTION_REASON_SCHEMA = (
        counter_reasons
    )
    monkeypatch.setitem(sys.modules, counter_module_name, counter_module)

    ordinary_sources = {
        "hope_commands.py": "0" * 64,
        "continuous_questions.py": "1" * 64,
        "stroke_adapt_torch.py": "2" * 64,
        "virtual_ball.py": "3" * 64,
        "racket_contact_geometry.py": "5" * 64,
    }
    geometry_contract = {
        "payload": {
            "schema_version": 2,
            "kind": "exact_face_contact_v2",
        },
        "sha256": "6" * 64,
    }
    ordinary = build(
        _solver_cfg(),
        physics_profile_sha256="4" * 64,
        source_sha256=ordinary_sources,
        contact_geometry_contract=geometry_contract,
        net_top_z=0.9325,
    )
    sources = {
        **ordinary_sources,
        "counter_rally.py": "7" * 64,
        "counter_rally_torch.py": "8" * 64,
    }
    objective_sha = "9" * 64
    venue_sha = "a" * 64
    counter = build(
        _solver_cfg(),
        physics_profile_sha256="4" * 64,
        source_sha256=sources,
        contact_geometry_contract=geometry_contract,
        net_top_z=0.9325,
        counter_rally_objective_profile_sha256=objective_sha,
        counter_rally_venue_physics_sha256=venue_sha,
    )
    ordinary_reasons = ordinary["payload"]["acceptance"][
        "ordered_rejection_reason_schema"
    ]
    counter_reason_schema = counter["payload"]["acceptance"][
        "ordered_rejection_reason_schema"
    ]
    assert counter_reason_schema[: len(ordinary_reasons)] == ordinary_reasons
    assert tuple(counter_reason_schema[-7:]) == counter_reasons
    assert len(counter_reason_schema) == len(ordinary_reasons) + 7
    assert counter["payload"]["counter_rally"] == {
        "mode": "exact_n1_fixed_action_reverse_ray",
        "objective_profile_sha256": objective_sha,
        "venue_physics_sha256": venue_sha,
        "precheck_before_ordinary_solver": True,
        "selector_or_action_switching": False,
    }
    assert counter["payload"]["implementation_source_sha256"] == sources
    assert counter["sha256"] == canonical(counter["payload"])
    assert counter["sha256"] != ordinary["sha256"]

    with pytest.raises(
        ValueError,
        match="requires objective and venue-physics SHA together",
    ):
        build(
            _solver_cfg(),
            physics_profile_sha256="4" * 64,
            source_sha256=sources,
            contact_geometry_contract=geometry_contract,
            net_top_z=0.9325,
            counter_rally_objective_profile_sha256=objective_sha,
        )


def test_domain_authority_contract_pins_behavior_but_not_mutable_cursors():
    canonical, build = _module_functions(
        (
            "_action_ball_canonical_sha256",
            "action_ball_domain_authority_contract",
        ),
        {
            "hashlib": hashlib,
            "json": json,
            "Sequence": tuple,
            "_ACTION_BALL_DOMAIN_AUTHORITY_SCHEMA_VERSION": 1,
        },
    )
    sources = {
        "hope_commands.py": "0" * 64,
        "action_ball_curriculum.py": "1" * 64,
        "action_ball_runtime.py": "2" * 64,
    }
    contract = build(
        manifest_sha256="3" * 64,
        adapter_contract_sha256="4" * 64,
        action_uids=(7, 9),
        profile_sha256=("5" * 64, "6" * 64),
        mobility_mode="move",
        curriculum_config={"target_failure_rate": 0.2},
        policy_contract_sha256="7" * 64,
        source_sha256=sources,
    )
    assert contract["sha256"] == canonical(contract["payload"])
    assert contract["payload"]["schedule"] == {
        "claim_barrier": "true_reset_only",
        "domain_source": "frozen_ActionBallCurriculum.expected_domains",
        "selection": "per_action_round_robin",
        "training_selector": False,
        "live_rollout_updates_curriculum": False,
    }
    encoded = json.dumps(contract, sort_keys=True)
    assert "cursor" not in encoded
    assert "provider_birth" not in encoded
    assert "sampler_draw" not in encoded
    with pytest.raises(ValueError, match="one profile SHA"):
        build(
            manifest_sha256="3" * 64,
            adapter_contract_sha256="4" * 64,
            action_uids=(7, 9),
            profile_sha256=("5" * 64,),
            mobility_mode="move",
            curriculum_config={},
            policy_contract_sha256="7" * 64,
            source_sha256=sources,
        )


def test_physics_profile_includes_exact_venue_bytes_and_full_table_geometry(tmp_path):
    canonical, sha_file, build = _module_functions(
        (
            "_action_ball_canonical_sha256",
            "_action_ball_sha256_file",
            "action_ball_physics_profile_contract",
        ),
        {
            "hashlib": hashlib,
            "json": json,
            "Path": Path,
            "_ACTION_BALL_PHYSICS_PROFILE_SCHEMA_VERSION": 1,
        },
    )
    venue = tmp_path / "venue.yaml"
    venue.write_text("physics: exact\n", encoding="utf-8")
    prm = SimpleNamespace(
        source_path=venue,
        **{
            name: float(index + 1) / 10.0
            for index, name in enumerate(
                (
                    "k_d",
                    "k_m",
                    "g",
                    "ball_radius",
                    "inertia_coeff",
                    "paddle_a_t",
                    "paddle_b_t",
                    "paddle_mu",
                    "paddle_e_g1",
                    "paddle_e_g2",
                )
            )
        },
    )
    cfg = SimpleNamespace(
        vb_table_surface_z=0.76,
        vb_min_landing_depth=0.15,
        vb_capture_radius=0.09,
        vb_min_approach_speed=0.25,
        vb_rollout_h=0.002,
        vb_rollout_steps=1000,
    )
    contract = build(
        cfg,
        prm,
        repo_root=tmp_path,
        surface_z=0.78,
        net_x=1.87,
        net_top_z=0.9325,
        opponent_near_x=0.5,
        opponent_far_x=3.24,
        table_half_width=0.7625,
    )
    assert contract["sha256"] == canonical(contract["payload"])
    assert contract["payload"]["venue_source"] == {
        "path": "venue.yaml",
        "file_sha256": sha_file(venue),
    }
    geometry = contract["payload"]["geometry_and_grading"]
    assert geometry["opponent_near_x_m"] == 0.5
    assert geometry["net_x_m"] == 1.87
    assert geometry["opponent_far_x_m"] == 3.24
    assert geometry["table_half_width_m"] == 0.7625


def test_runtime_transaction_consumes_birth_only_on_reset_and_installs_one_tuple():
    initialize = _method_source("_initialize_action_ball_runtime")
    sample = _method_source("_sample_targets_action_ball")
    install = _method_source("_action_ball_commit_install")
    assert "pool.bind_birth_authority(broker)" in initialize
    assert "Path.cwd()" not in initialize
    assert "consume_many_true_reset" in sample
    assert "reserve_true_reset" not in sample
    assert "commit_true_reset" not in sample
    assert "if true_reset:" in sample
    assert "wrap changed its frozen action/birth identity" in sample
    assert sample.index("_action_ball_close_attempts") < sample.index(
        "_action_ball_pool.request"
    )
    assert sample.index("_action_ball_pool.request") < sample.index(
        "_action_ball_commit_install"
    )
    for field in (
        "racket_target_pos_w",
        "racket_target_vel_w",
        "racket_target_normal_w",
        "target_normal_cmd",
        "base_target_pos_w",
        "vb_vel_in_w",
        "vb_spin_in_w",
        "_vb_target_xy_per_env",
    ):
        assert f"self.{field}[ids]" in install
    assert install.index("if any(not bool(torch.isfinite") < install.index(
        "self.racket_target_pos_w[ids]"
    )
    assert '_ACTION_BALL_LEDGER_NAMES.index("I")' in install
    assert '_ACTION_BALL_LEDGER_NAMES.index("S")' in install


def test_startup_pins_native_site_speed_and_disables_legacy_time_owners():
    initialize = _method_source("_initialize_action_ball_runtime")
    assert "ARM_KEYS as CURRICULUM_ARM_KEYS" in initialize
    assert "ARM_KEYS as SAMPLER_ARM_KEYS" in initialize
    assert (
        "tuple(CURRICULUM_ARM_KEYS) != tuple(SAMPLER_ARM_KEYS)"
        in initialize
    )
    assert (
        "profile.reference_racket_site_speed_mps"
        in initialize
    )
    assert (
        "torch.linalg.norm("
        "\n                    self._ref_racket_vel_w_per_clip[slot]"
        in initialize
    )
    assert "abs_tol=1.0e-6" in initialize
    assert (
        "profile reference racket-site speed differs"
        in initialize
    )
    for literal in (
        '"hold_steps_range": (0, 0)',
        '"stand_start_min_hold": 0',
        '"post_swing_min_hold": 0',
        '"stagger_initial_clock": False',
    ):
        assert literal in initialize


def test_pending_task_age_counts_only_completed_physics_ticks():
    """Reset resolves at age 0; wrap resolves at dt after one real tick."""

    motion_begin = MOTION_SOURCE.index(
        "if self._resampling_from_wrap and not self.cfg.wrap_teleport:"
    )
    motion_end = MOTION_SOURCE.index(
        "\n        if self.canonical_ready_mode:", motion_begin
    )
    wrap = MOTION_SOURCE[motion_begin:motion_end]
    assert "_begin_action_ball_task_pending(" in wrap
    assert "elapsed_s=float(self._env.step_dt)" in wrap
    advance_begin = MOTION_SOURCE.index(
        "def _advance_action_ball_task_timing("
    )
    advance_end = MOTION_SOURCE.index(
        "\n    def _write_canonical_ready_state", advance_begin
    )
    advance = MOTION_SOURCE[advance_begin:advance_end]
    assert "active_before" in advance
    assert (
        "advancing = active_before_resolve & ~cycle_due_before"
        in " ".join(advance.split())
    )

    install = _method_source("_action_ball_commit_install")
    assert "pending_elapsed_s = 0.0" in install
    assert "float(self._env.step_dt) if self._resample_is_wrap" not in install
    assert "float(receipt.time_to_contact_s) - pending_elapsed_s" in install
    timing = _method_source("_compute_strike_timing")
    assert "& ~wrapped" in timing
    assert "action_ball_time_to_contact_remaining_s" in timing


def test_pool_solver_adapter_delegates_4096_rows_to_one_batch_callback():
    adapter_type = _module_class("_ActionBallPoolSolverAdapter", {})
    calls = {"scalar": 0, "batch": 0, "shape": None}

    def scalar(_request):
        calls["scalar"] += 1

    def batch(requests):
        calls["batch"] += 1
        calls["shape"] = len(requests)
        return tuple(range(len(requests)))

    adapter = adapter_type(
        solver_contract_sha256="a" * 64,
        state_owner_sha256="b" * 64,
        solve=scalar,
        solve_many=batch,
        assert_emitted_sample=lambda _receipt: None,
        assert_emitted_tasks=lambda _receipts: None,
        emitted_task_count_for=lambda _uid: 0,
        task_transcript_for_birth=lambda _birth: (0, "0" * 64),
        assert_proposal_assignments=lambda _assignments: None,
        sample_highwater_for=lambda _uid: (-1, 0),
        state_getter=lambda: {"cursor": 0},
        state_loader=lambda _state: None,
    )
    result = adapter.solve_many(range(4096))
    assert result[0] == 0
    assert result[-1] == 4095
    assert calls == {"scalar": 0, "batch": 1, "shape": 4096}


def test_domain_provider_and_solver_views_share_one_exact_mutable_state_owner():
    domain_type = _module_class("_ActionBallDomainAuthorityAdapter", {})
    provider_type = _module_class("_ActionBallBirthProviderAdapter", {})
    solver_type = _module_class("_ActionBallPoolSolverAdapter", {})
    box = {"state": {"cursor": 0, "provider_births": [], "sampler": {}}}

    def state_getter():
        return json.loads(json.dumps(box["state"]))

    def state_loader(value):
        box["state"] = json.loads(json.dumps(value))

    owner = "c" * 64
    domain = domain_type(
        domain_authority_contract_sha256="d" * 64,
        state_owner_sha256=owner,
        claim=lambda uid: ("claim", uid),
        domain_cursor_for=lambda _uid: 0,
        state_getter=state_getter,
        state_loader=state_loader,
    )
    provider = provider_type(
        sampler_contract_sha256="e" * 64,
        state_owner_sha256=owner,
        provide=lambda request: ("birth", request),
        assert_issued_birth=lambda _receipt: None,
        birth_highwater_for=lambda _uid: (-1, 0),
        state_getter=state_getter,
        state_loader=state_loader,
    )
    solver = solver_type(
        solver_contract_sha256="f" * 64,
        state_owner_sha256=owner,
        solve=lambda request: ("task", request),
        solve_many=lambda requests: tuple(("task", row) for row in requests),
        assert_emitted_sample=lambda _receipt: None,
        assert_emitted_tasks=lambda _receipts: None,
        emitted_task_count_for=lambda _uid: 0,
        task_transcript_for_birth=lambda _birth: (0, "0" * 64),
        assert_proposal_assignments=lambda _assignments: None,
        sample_highwater_for=lambda _uid: (-1, 0),
        state_getter=state_getter,
        state_loader=state_loader,
    )
    assert {
        domain.state_owner_sha256,
        provider.state_owner_sha256,
        solver.state_owner_sha256,
    } == {owner}
    replacement = {
        "cursor": 9,
        "provider_births": [{"birth": "1"}],
        "sampler": {"draw": 17},
    }
    provider.load_state_dict(replacement)
    assert domain.state_dict() == replacement
    assert solver.state_dict() == replacement
    assert domain.claim_for_action(7) == ("claim", 7)
    assert domain.domain_cursor_for(7) == 0
    assert provider("request") == ("birth", "request")
    assert provider.birth_highwater_for(7) == (-1, 0)
    solver.assert_emitted_tasks(())
    solver.assert_proposal_assignments(())
    assert solver.emitted_task_count_for(7) == 0
    assert solver.task_transcript_for_birth("birth") == (0, "0" * 64)
    assert solver.sample_highwater_for(7) == (-1, 0)


class _FakeTensor:
    def __init__(self, values, *, dtype=None):
        self.values = list(values)
        self.dtype = dtype

    def clone(self):
        return _FakeTensor(self.values, dtype=self.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)

    def sum(self):
        return _FakeScalar(sum(self.values))

    def index_select(self, _dim, indices):
        return _FakeTensor(
            [self.values[index] for index in indices.values],
            dtype=self.dtype,
        )

    def __getitem__(self, index):
        value = self.values[index]
        if isinstance(value, (tuple, list)):
            return _FakeTensor(value, dtype=self.dtype)
        return _FakeScalar(value)


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def test_opaque_task_ref_rejects_forged_stale_and_cross_env_refs(
    monkeypatch,
):
    runtime_module = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_runtime"
    )

    @dataclass(frozen=True)
    class TaskRef:
        env_id: int
        reset_generation: int
        swing_generation: int
        action_uid: int
        action_slot: int
        birth_sha256: str
        sample_sha256: str
        task_sha256: str

    class TaskReceipt:
        def __init__(self):
            self.env_id = 0
            self.reset_generation = 2
            self.swing_generation = 3
            self.action_uid = 7
            self.action_slot = 0
            self.birth_sha256 = "1" * 64
            self.sample_sha256 = "2" * 64
            self.canonical_sha256 = "3" * 64

        def to_dict(self):
            return {"receipt": self}

        @classmethod
        def from_dict(cls, value):
            return value["receipt"]

        def assert_birth(self, birth):
            assert birth.canonical_sha256 == self.birth_sha256

        def task_ref(self):
            return TaskRef(
                env_id=self.env_id,
                reset_generation=self.reset_generation,
                swing_generation=self.swing_generation,
                action_uid=self.action_uid,
                action_slot=self.action_slot,
                birth_sha256=self.birth_sha256,
                sample_sha256=self.sample_sha256,
                task_sha256=self.canonical_sha256,
            )

    runtime_module.ActionTaskReceiptRef = TaskRef
    runtime_module.ActionBallTaskReceipt = TaskReceipt
    monkeypatch.setitem(
        sys.modules, runtime_module.__name__, runtime_module
    )
    namespace = {}
    module = ast.Module(
        body=[
            _method("_action_ball_task_receipt_for_env"),
            _method("action_ball_task_ref_for_env"),
            _method("action_ball_resolve_task_ref"),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)

    receipt = TaskReceipt()
    birth = SimpleNamespace(canonical_sha256=receipt.birth_sha256)
    command = SimpleNamespace(
        num_envs=2,
        _action_ball_enabled=True,
        _action_ball_task_by_env=[receipt, None],
        _action_ball_birth_by_env=[birth, None],
        _action_ball_attempt_active=_FakeTensor([True, False]),
        _action_ball_action_uid=_FakeTensor([7, -1]),
        _action_ball_action_slot=_FakeTensor([0, -1]),
        _action_ball_reset_generation=_FakeTensor([2, 0]),
        _action_ball_swing_generation=_FakeTensor([3, 0]),
        _action_ball_attempt_action=_FakeTensor([0, -1]),
    )
    command._action_ball_task_receipt_for_env = types.MethodType(
        namespace["_action_ball_task_receipt_for_env"], command
    )
    command.action_ball_task_ref_for_env = types.MethodType(
        namespace["action_ball_task_ref_for_env"], command
    )
    command.action_ball_resolve_task_ref = types.MethodType(
        namespace["action_ball_resolve_task_ref"], command
    )
    ref = command.action_ball_task_ref_for_env(0)
    assert command.action_ball_resolve_task_ref(ref) is receipt
    with pytest.raises(ValueError, match="exact opaque ref"):
        command.action_ball_resolve_task_ref(
            SimpleNamespace(env_id=0)
        )
    with pytest.raises(RuntimeError, match="stale"):
        command.action_ball_resolve_task_ref(
            replace(ref, task_sha256="4" * 64)
        )
    with pytest.raises(RuntimeError, match="stale"):
        command.action_ball_resolve_task_ref(
            replace(ref, env_id=1)
        )


def _load_action_ball_dependency_modules(monkeypatch):
    module_dir = COMMAND_PATH.parent
    result = {}
    for basename in (
        "action_ball_curriculum",
        "action_ball_evaluation",
        "action_ball_sampling",
        "action_ball_runtime",
    ):
        name = f"whole_body_tracking.tasks.tracking.mdp.{basename}"
        spec = importlib.util.spec_from_file_location(
            name, module_dir / f"{basename}.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        result[basename] = module
    return result


def _host_sampling_profile(sampling, action_uid=101):
    return sampling.SamplingProfile(
        action_uid=action_uid,
        contact_offset_center_b_yaw_m=(0.55, 0.12, 0.82),
        contact_offset_std_lower_initial_m=(0.005, 0.01, 0.01),
        contact_offset_std_lower_max_m=(0.02, 0.12, 0.16),
        contact_offset_std_upper_initial_m=(0.004, 0.02, 0.01),
        contact_offset_std_upper_max_m=(0.02, 0.10, 0.15),
        contact_offset_min_b_yaw_m=(0.45, -0.20, 0.55),
        contact_offset_max_b_yaw_m=(0.65, 0.35, 1.10),
        time_to_contact_center_s=1.20,
        time_to_contact_std_lower_initial_s=0.01,
        time_to_contact_std_lower_max_s=0.15,
        time_to_contact_std_upper_initial_s=0.02,
        time_to_contact_std_upper_max_s=0.30,
        time_to_contact_min_s=1.05,
        time_to_contact_max_s=1.60,
        incoming_direction_center_b_yaw=(-1.0, 0.0, 0.0),
        incoming_direction_tangent_u_b_yaw=(0.0, 1.0, 0.0),
        incoming_direction_tangent_v_b_yaw=(0.0, 0.0, -1.0),
        incoming_direction_tangent_u_neg_initial_deg=0.5,
        incoming_direction_tangent_u_neg_max_deg=8.0,
        incoming_direction_tangent_u_pos_initial_deg=0.6,
        incoming_direction_tangent_u_pos_max_deg=7.0,
        incoming_direction_tangent_v_neg_initial_deg=0.7,
        incoming_direction_tangent_v_neg_max_deg=6.0,
        incoming_direction_tangent_v_pos_initial_deg=0.8,
        incoming_direction_tangent_v_pos_max_deg=5.0,
        incoming_inbound_axis_b_yaw=(-1.0, 0.0, 0.0),
        incoming_inbound_min_cosine=0.8,
        incoming_speed_center_mps=4.0,
        incoming_speed_std_lower_initial_mps=0.05,
        incoming_speed_std_lower_max_mps=1.2,
        incoming_speed_std_upper_initial_mps=0.06,
        incoming_speed_std_upper_max_mps=1.0,
        incoming_speed_min_mps=1.6,
        incoming_speed_max_mps=7.0,
        spin_direction_center_b_yaw=(0.0, 1.0, 0.0),
        spin_direction_tangent_u_b_yaw=(0.0, 0.0, 1.0),
        spin_direction_tangent_v_b_yaw=(1.0, 0.0, 0.0),
        spin_direction_tangent_u_neg_initial_deg=0.0,
        spin_direction_tangent_u_neg_max_deg=35.0,
        spin_direction_tangent_u_pos_initial_deg=0.0,
        spin_direction_tangent_u_pos_max_deg=30.0,
        spin_direction_tangent_v_neg_initial_deg=0.0,
        spin_direction_tangent_v_neg_max_deg=25.0,
        spin_direction_tangent_v_pos_initial_deg=0.0,
        spin_direction_tangent_v_pos_max_deg=20.0,
        spin_magnitude_center_radps=15.0,
        spin_magnitude_std_lower_initial_radps=0.2,
        spin_magnitude_std_lower_max_radps=8.0,
        spin_magnitude_std_upper_initial_radps=0.3,
        spin_magnitude_std_upper_max_radps=9.0,
        spin_magnitude_min_radps=0.0,
        spin_magnitude_max_radps=40.0,
        base_spawn_center_w_m=(-0.10, 0.05, 0.0),
        base_spawn_std_lower_initial_m=(0.005, 0.005, 0.0),
        base_spawn_std_lower_max_m=(0.15, 0.20, 0.0),
        base_spawn_std_upper_initial_m=(0.006, 0.007, 0.0),
        base_spawn_std_upper_max_m=(0.12, 0.18, 0.0),
        base_spawn_min_w_m=(-0.50, -0.40, 0.0),
        base_spawn_max_w_m=(0.30, 0.50, 0.0),
        base_travel_center_b_yaw_m=(0.0, 0.0, 0.0),
        base_travel_std_lower_initial_m=(0.0, 0.0, 0.0),
        base_travel_std_lower_max_m=(0.0, 0.0, 0.0),
        base_travel_std_upper_initial_m=(0.0, 0.0, 0.0),
        base_travel_std_upper_max_m=(0.0, 0.0, 0.0),
        base_travel_min_b_yaw_m=(0.0, 0.0, 0.0),
        base_travel_max_b_yaw_m=(0.0, 0.0, 0.0),
        landing_aim_center_w_xy_m=(2.55, 0.0),
        landing_aim_std_lower_initial_m=(0.01, 0.01),
        landing_aim_std_lower_max_m=(0.25, 0.35),
        landing_aim_std_upper_initial_m=(0.02, 0.01),
        landing_aim_std_upper_max_m=(0.20, 0.30),
        landing_aim_min_w_xy_m=(2.20, -0.55),
        landing_aim_max_w_xy_m=(2.90, 0.55),
        reference_t_hit_s=0.80,
        reference_t_cycle_s=1.60,
        reference_racket_site_speed_mps=6.0,
        reaction_margin_s=0.05,
        teacher_rate_min=0.80,
        teacher_rate_max=1.20,
        mobility_mode="no_move",
    )


def _runtime_birth_for_sampler(
    runtime,
    *,
    sampler,
    sampler_birth,
    binding,
    pins,
    registry_sha256,
    env_id,
    reset_generation,
    domain_epoch,
    levels,
):
    runtime_levels = runtime.ActionDomainLevels.from_dict(levels.as_dict())
    claim = runtime.ActionDomainClaim(
        authority_contract_sha256=pins.domain_authority_sha256,
        action_uid=binding.action_uid,
        domain_epoch=domain_epoch,
        domain_levels=runtime_levels,
        levels_sha256=runtime_levels.canonical_sha256,
        arm_catalog_sha256=runtime.ARM_CATALOG_SHA256,
        profile_sha256=binding.profile_sha256,
        mobility_mode="no_move",
    )
    return runtime.ActionBirthReceipt(
        registry_sha256=registry_sha256,
        env_id=env_id,
        reset_generation=reset_generation,
        action_uid=binding.action_uid,
        action_slot=binding.action_slot,
        domain_epoch=domain_epoch,
        domain_claim_sha256=claim.canonical_sha256,
        domain_authority_sha256=pins.domain_authority_sha256,
        domain_levels=runtime_levels,
        levels_sha256=runtime_levels.canonical_sha256,
        arm_catalog_sha256=runtime.ARM_CATALOG_SHA256,
        sampler_birth_sha256=sampler_birth.birth_id,
        sampler_birth_index=sampler_birth.birth_index,
        sampler_draw_start=sampler_birth.draw_start,
        sampler_draw_end=sampler_birth.draw_end,
        mobility_mode="no_move",
        base_yaw_rad=sampler_birth.base_yaw_rad,
        base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        base_spawn_w_m=sampler_birth.base_start_w_m,
        manifest_sha256=pins.manifest_sha256,
        sampler_sha256=sampler.sampler_contract_sha256,
        profile_sha256=binding.profile_sha256,
        motion_sha256=binding.motion_sha256,
        physics_sha256=pins.physics_sha256,
        solver_sha256=pins.solver_sha256,
    )


def test_no_move_goal_is_the_episode_spawn_not_the_environment_origin(
    monkeypatch,
):
    modules = _load_action_ball_dependency_modules(monkeypatch)
    sampling = modules["action_ball_sampling"]
    profile = _host_sampling_profile(sampling, action_uid=101)
    sampler = sampling.ActionBallSampler((profile,), seed=29)
    levels = sampling.DomainLevels()
    birth = sampler.reserve_birth(
        action_uid=101,
        domain_epoch=0,
        levels=levels,
        base_yaw_rad=0.0,
    )
    sample = sampler.sample(
        birth=birth,
        action_uid=101,
        domain_epoch=0,
        levels=levels,
        base_yaw_rad=0.0,
    )
    assert birth.base_start_w_m != (0.0, 0.0, 0.0)
    assert sample.mobility_mode == "no_move"
    assert sample.base_goal_w_m == birth.base_start_w_m
    install = _method_source("_action_ball_commit_install")
    assert (
        "self.base_target_pos_w[ids] = "
        "origins[:, :2] + base_goal_local[:, :2]"
        in " ".join(install.split())
    )


def test_sampler_birth_parser_round_trips_current_arm_catalog_and_mixture(
    monkeypatch,
):
    modules = _load_action_ball_dependency_modules(monkeypatch)
    sampling = modules["action_ball_sampling"]
    profile = _host_sampling_profile(sampling, action_uid=101)
    levels = sampling.DomainLevels()

    method = _method("_action_ball_parse_sampler_birth")
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[method], type_ignores=[])
            ),
            str(COMMAND_PATH),
            "exec",
        ),
        namespace,
    )
    parse = namespace["_action_ball_parse_sampler_birth"]
    if isinstance(parse, staticmethod):
        parse = parse.__func__

    for mixture in (None, sampling.SamplingMixture()):
        sampler = sampling.ActionBallSampler(
            (profile,),
            seed=31,
            sampling_mixture=mixture,
        )
        birth = sampler.reserve_birth(
            action_uid=101,
            domain_epoch=0,
            levels=levels,
            base_yaw_rad=0.0,
        )
        serialized = birth.to_receipt()
        assert parse(serialized) == birth
        assert serialized["arm_catalog_sha256"] == sampling.ARM_CATALOG_SHA256

        bad_arm_catalog = json.loads(json.dumps(serialized))
        bad_arm_catalog["arm_catalog_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="arm catalog"):
            parse(bad_arm_catalog)

        if mixture is not None:
            missing_sampling = json.loads(json.dumps(serialized))
            del missing_sampling["sampling"]
            with pytest.raises(ValueError, match="birth_id"):
                parse(missing_sampling)


def test_retired_birth_history_rejects_resigned_runtime_assignment(monkeypatch):
    modules = _load_action_ball_dependency_modules(monkeypatch)
    sampling = modules["action_ball_sampling"]
    runtime = modules["action_ball_runtime"]
    uid = 101
    profile = _host_sampling_profile(sampling, uid)
    sampler = sampling.ActionBallSampler((profile,), seed=17)
    pins = runtime.RuntimePins(
        manifest_sha256="1" * 64,
        sampler_sha256=sampler.sampler_contract_sha256,
        domain_authority_sha256="2" * 64,
        physics_sha256="3" * 64,
        solver_sha256="4" * 64,
    )
    binding = runtime.ActionBinding(
        action_uid=uid,
        action_slot=0,
        motion_path="vendor_assets/action.npz",
        motion_sha256="5" * 64,
        profile_sha256=profile.sha256,
    )
    broker = runtime.ActionBirthBroker((binding,), pins, "no_move")
    levels = sampling.DomainLevels()
    births = []
    providers = {}
    for generation in (1, 2):
        sampler_birth = sampler.reserve_birth(
            action_uid=uid,
            domain_epoch=generation,
            levels=levels,
            base_yaw_rad=0.0,
        )
        receipt = _runtime_birth_for_sampler(
            runtime,
            sampler=sampler,
            sampler_birth=sampler_birth,
            binding=binding,
            pins=pins,
            registry_sha256=broker.registry_sha256,
            env_id=0,
            reset_generation=generation,
            domain_epoch=generation,
            levels=levels,
        )
        births.append(receipt)
        providers[receipt.canonical_sha256] = {
            "runtime_birth": receipt,
            "sampler_birth": sampler_birth,
        }

    method = _method("_action_ball_assert_issued_birth")
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[method], type_ignores=[])
            ),
            str(COMMAND_PATH),
            "exec",
        ),
        namespace,
    )
    command = SimpleNamespace(
        _action_ball_bindings=(binding,),
        _action_ball_pins=pins,
        _action_ball_manifest=SimpleNamespace(mobility_mode="no_move"),
        _action_ball_broker=broker,
        _action_ball_sampler=sampler,
        # Both generations remain in the append-only assignment transcript
        # even though only generation 2 would be active after retirement.
        _action_ball_provider_births=providers,
        _action_ball_provider_history={
            receipt.canonical_sha256: receipt for receipt in births
        },
    )
    namespace["_action_ball_assert_issued_birth"](command, births[0])
    namespace["_action_ball_assert_issued_birth"](command, births[1])
    forged = replace(births[0], env_id=9, reset_generation=7)
    with pytest.raises(RuntimeError, match="assignment transcript"):
        namespace["_action_ball_assert_issued_birth"](command, forged)


def test_proposal_assignment_replay_covers_rejected_rows_and_old_births(
    monkeypatch,
):
    modules = _load_action_ball_dependency_modules(monkeypatch)
    sampling = modules["action_ball_sampling"]
    runtime = modules["action_ball_runtime"]
    uid = 101
    profile = _host_sampling_profile(sampling, uid)
    sampler = sampling.ActionBallSampler((profile,), seed=23)
    pins = runtime.RuntimePins(
        manifest_sha256="1" * 64,
        sampler_sha256=sampler.sampler_contract_sha256,
        domain_authority_sha256="2" * 64,
        physics_sha256="3" * 64,
        solver_sha256="4" * 64,
    )
    binding = runtime.ActionBinding(
        action_uid=uid,
        action_slot=0,
        motion_path="vendor_assets/action.npz",
        motion_sha256="5" * 64,
        profile_sha256=profile.sha256,
    )
    broker = runtime.ActionBirthBroker((binding,), pins, "no_move")
    levels = sampling.DomainLevels()
    runtime_births = []
    sample_indices = []
    for generation, sample_count in ((1, 2), (2, 1)):
        sampler_birth = sampler.reserve_birth(
            action_uid=uid,
            domain_epoch=generation,
            levels=levels,
            base_yaw_rad=0.0,
        )
        runtime_birth = _runtime_birth_for_sampler(
            runtime,
            sampler=sampler,
            sampler_birth=sampler_birth,
            binding=binding,
            pins=pins,
            registry_sha256=broker.registry_sha256,
            env_id=0,
            reset_generation=generation,
            domain_epoch=generation,
            levels=levels,
        )
        runtime_births.append(runtime_birth)
        owned = []
        for _ in range(sample_count):
            sample = sampler.sample(
                birth=sampler_birth,
                action_uid=uid,
                domain_epoch=generation,
                levels=levels,
                base_yaw_rad=0.0,
            )
            owned.append(sample.sample_index)
        sample_indices.append(tuple(owned))

    assignments = (
        runtime.ActionSampleAssignment(
            birth=runtime_births[0],
            refill_index=1,
            proposal_sample_indices=sample_indices[0],
        ),
        runtime.ActionSampleAssignment(
            birth=runtime_births[1],
            refill_index=1,
            proposal_sample_indices=sample_indices[1],
        ),
    )
    method = _method(
        "_action_ball_assert_proposal_assignments_against"
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[method], type_ignores=[])
            ),
            str(COMMAND_PATH),
            "exec",
        ),
        namespace,
    )
    provider_history = {
        birth.canonical_sha256: birth for birth in runtime_births
    }
    before = sampler.state_dict()
    namespace["_action_ball_assert_proposal_assignments_against"](
        SimpleNamespace(),
        assignments,
        sampler=sampler,
        provider_history=provider_history,
    )
    assert sampler.state_dict() == before

    wrong_birth = runtime.ActionSampleAssignment(
        birth=runtime_births[1],
        refill_index=2,
        proposal_sample_indices=(sample_indices[0][0],),
    )
    with pytest.raises(ValueError, match="different episode birth"):
        namespace["_action_ball_assert_proposal_assignments_against"](
            SimpleNamespace(),
            (wrong_birth,),
            sampler=sampler,
            provider_history=provider_history,
        )
    assert sampler.state_dict() == before


@pytest.mark.parametrize(
    ("solver_speed", "receipts_per_birth", "timing_reason", "sample_v_in_x"),
    (
        (1.0, 1, None, -5.0),
        (2.0, 0, "teacher_rate_out_of_bounds", -5.0),
        (1.0, 0, "ball_birth_not_beyond_net", -1.0),
    ),
)
def test_refill_many_flattens_4096_births_and_rejects_timing_pre_issue(
    monkeypatch,
    solver_speed,
    receipts_per_birth,
    timing_reason,
    sample_v_in_x,
):
    refill_many = _method("_action_ball_refill_pool_many")
    namespace = {
        "math": math,
        "torch": SimpleNamespace(
            long="long",
            tensor=lambda values, dtype=None, device=None: _FakeTensor(
                values, dtype=dtype
            ),
        ),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            refill_many,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)

    runtime_module = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_runtime"
    )

    class TaskReceipt:
        @staticmethod
        def from_birth(birth, **kwargs):
            return SimpleNamespace(
                birth_sha256=birth.canonical_sha256,
                action_uid=7,
                canonical_sha256=kwargs["sample_sha256"],
                **kwargs,
            )

    class RefillBatch:
        def __init__(
            self,
            *,
            action_uid,
            proposed_count,
            proposal_sample_indices,
            receipts,
        ):
            self.action_uid = action_uid
            self.proposed_count = proposed_count
            self.proposal_sample_indices = proposal_sample_indices
            self.receipts = receipts

    runtime_module.ActionBallTaskReceipt = TaskReceipt
    runtime_module.ActionPoolRefillBatch = RefillBatch
    runtime_module.ActionBallContractError = ValueError

    def derive_action_teacher_site_timing(
        *,
        racket_site_velocity_w_mps,
        time_to_contact_s,
        reference_t_hit_s,
        reference_t_cycle_s,
        reference_racket_site_speed_mps,
        reaction_margin_s,
        teacher_rate_min,
        teacher_rate_max,
    ):
        del reaction_margin_s, teacher_rate_min, teacher_rate_max
        required = math.sqrt(
            sum(
                float(value) ** 2
                for value in racket_site_velocity_w_mps
            )
        )
        rate = required / reference_racket_site_speed_mps
        hit = reference_t_hit_s / rate
        cycle = reference_t_cycle_s / rate
        return SimpleNamespace(
            required_racket_site_speed_mps=required,
            teacher_rate=rate,
            scaled_t_hit_s=hit,
            scaled_t_cycle_s=cycle,
            pre_swing_wait_s=time_to_contact_s - hit,
        )

    runtime_module.derive_action_teacher_site_timing = (
        derive_action_teacher_site_timing
    )
    runtime_module.extend_task_transcript_sha256 = (
        lambda prior, task: hashlib.sha256(
            f"{prior}:{task}".encode("ascii")
        ).hexdigest()
    )
    solver_module = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.continuous_questions"
    )
    solver_calls = []

    def solve_proposals(
        clip_ids,
        contact,
        incoming,
        spin,
        aim,
        ref_normal,
        **_kwargs,
    ):
        sizes = {
            len(tensor.values)
            for tensor in (
                clip_ids,
                contact,
                incoming,
                spin,
                aim,
                ref_normal,
            )
        }
        assert sizes == {4096}
        solver_calls.append(4096)
        return SimpleNamespace(
            ok=_FakeTensor([True] * 4096),
            reason_counts={},
            proposals=SimpleNamespace(
                reason_code=_FakeTensor([0] * 4096)
            ),
            v_racket=_FakeTensor(
                [(solver_speed, 0.0, 0.0)] * 4096
            ),
            n_racket=_FakeTensor([(0.0, 1.0, 0.0)] * 4096),
            resid_m=_FakeTensor([0.0] * 4096),
        )

    solver_module.solve_proposals = solve_proposals
    solver_module.BALL_BIRTH_NET_MARGIN_M = 0.05
    solver_module.ball_birth_x_lower_bound_m = (
        lambda contact_x, v_in_x, ttc: float(contact_x)
        + abs(float(v_in_x)) * float(ttc)
    )
    geometry_module = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.racket_contact_geometry"
    )

    class ExactFaceContactGeometryError(ValueError):
        def __init__(self, reason):
            self.reason = reason
            super().__init__(reason)

    def solve_exact_face_contact(
        *,
        ball_contact_w_m,
        racket_face_center_velocity_w_mps,
        solved_raw_a_normal_w,
        mount_normal_sign,
        reference_racket_quat_wxyz,
        reference_racket_angular_velocity_w_radps,
        reference_racket_site_speed_mps,
        teacher_rate_min,
        teacher_rate_max,
    ):
        del (
            solved_raw_a_normal_w,
            reference_racket_angular_velocity_w_radps,
        )
        required = math.sqrt(
            sum(
                float(value) ** 2
                for value in racket_face_center_velocity_w_mps
            )
        )
        rate = required / float(reference_racket_site_speed_mps)
        if not float(teacher_rate_min) <= rate <= float(teacher_rate_max):
            raise ExactFaceContactGeometryError(
                "teacher_rate_out_of_bounds"
            )
        return SimpleNamespace(
            geometry_source_sha256="9" * 64,
            mount_normal_sign=int(mount_normal_sign),
            racket_command_quat_wxyz=tuple(
                reference_racket_quat_wxyz
            ),
            racket_site_target_w_m=tuple(ball_contact_w_m),
            racket_face_center_velocity_w_mps=tuple(
                racket_face_center_velocity_w_mps
            ),
            racket_site_velocity_w_mps=tuple(
                racket_face_center_velocity_w_mps
            ),
            racket_command_angular_velocity_w_radps=(0.0, 0.0, 0.0),
            teacher_rate=rate,
        )

    geometry_module.ExactFaceContactGeometryError = (
        ExactFaceContactGeometryError
    )
    geometry_module.solve_exact_face_contact = solve_exact_face_contact
    geometry_module.GEOMETRY_SOURCE_SHA256 = "9" * 64
    mdp_package = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp"
    )
    mdp_package.__path__ = []
    mdp_package.racket_contact_geometry = geometry_module
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    monkeypatch.setitem(sys.modules, solver_module.__name__, solver_module)
    monkeypatch.setitem(
        sys.modules, geometry_module.__name__, geometry_module
    )
    monkeypatch.setitem(sys.modules, mdp_package.__name__, mdp_package)

    class Sampler:
        def __init__(self):
            self.calls = 0

        def sample(self, **_kwargs):
            index = self.calls
            self.calls += 1
            return SimpleNamespace(
                sample_id=f"{index:064x}",
                sample_index=index,
                draw_start=index * 18,
                draw_end=(index + 1) * 18,
                verify_sample_id=lambda: None,
                base_spawn_latent_w_m=(0.0, 0.0, 0.0),
                base_travel_latent_b_yaw_m=(0.0, 0.0, 0.0),
                contact_offset_from_base_goal_b_yaw_m=(0.0, 0.0, 1.0),
                contact_w_m=(0.0, 0.0, 1.0),
                time_to_contact_s=0.5,
                incoming_speed_mps=1.0,
                incoming_direction_b_yaw=(-1.0, 0.0, 0.0),
                incoming_velocity_w_mps=(sample_v_in_x, 0.0, 0.0),
                spin_magnitude_radps=0.0,
                spin_direction_b_yaw=(1.0, 0.0, 0.0),
                spin_w_radps=(0.0, 0.0, 0.0),
                landing_aim_w_xy_m=(2.0, 0.0),
                base_goal_w_m=(0.0, 0.0, 0.0),
            )

    sampler = Sampler()
    notes = {"P": 0, "A": 0}

    def note(_slot, name, amount):
        notes[name] += amount

    command = SimpleNamespace(
        cfg=SimpleNamespace(
            cq_max_redraw_rounds=1,
            cq_overdraw=1.0,
            vb_rollout_h=0.002,
            vb_rollout_steps=1000,
        ),
        device="cpu",
        _action_ball_planes=(0.78, 1.87, 0.9325),
        _ref_racket_normal_raw_w_per_clip=_FakeTensor(
            [(0.0, 1.0, 0.0)], dtype="float"
        ),
        _action_ball_bindings=(SimpleNamespace(action_uid=7),),
        _action_ball_solver_contract={
            "payload": {
                "acceptance": {
                        "ordered_rejection_reason_schema": (
                            "no_landing",
                            "teacher_site_rate_geometry_unsolved",
                            "teacher_rate_out_of_bounds",
                            "pre_swing_wait_out_of_bounds",
                            "cycle_exceeds_episode_horizon",
                            "ball_birth_not_beyond_net",
                        )
                }
            }
        },
        _action_ball_sampler=sampler,
        _action_ball_reject_counts={7: {}},
        _action_ball_prototypes=object(),
        _action_ball_prm=object(),
        _action_ball_solver_cfg=object(),
        _action_ball_timing=((0.2, 1.0),),
        _action_ball_bundle=SimpleNamespace(
            profiles=(
                SimpleNamespace(
                    reference_t_hit_s=0.2,
                    reference_t_cycle_s=0.4,
                    reference_racket_site_speed_mps=1.0,
                    reaction_margin_s=0.05,
                    teacher_rate_min=0.8,
                    teacher_rate_max=1.2,
                ),
            )
        ),
        _ref_racket_quat_w_per_clip=_FakeTensor(
            [(1.0, 0.0, 0.0, 0.0)], dtype="float"
        ),
        _ref_racket_ang_vel_w_per_clip=_FakeTensor(
            [(0.0, 0.0, 0.0)], dtype="float"
        ),
        _action_ball_mount_signs=(1,),
        _action_ball_attempt_close_margin_s=0.02,
        _action_ball_episode_length_s=2.0,
        _action_ball_note=note,
        _action_ball_provider_history={},
        _action_ball_task_transcript_by_birth={},
        _action_ball_emitted_task_count_by_uid={7: 0},
    )
    providers = {}
    requests = []
    for env_id in range(4096):
        birth = SimpleNamespace(
            canonical_sha256=f"{env_id + 1:064x}",
            domain_epoch=0,
            base_yaw_rad=0.0,
            base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            base_spawn_w_m=(0.0, 0.0, 0.0),
        )
        providers[birth.canonical_sha256] = {
            "runtime_birth": birth,
            "sampler_birth": object(),
            "levels": object(),
        }
        command._action_ball_provider_history[
            birth.canonical_sha256
        ] = SimpleNamespace(action_uid=7)
        command._action_ball_task_transcript_by_birth[
            birth.canonical_sha256
        ] = (0, "0" * 64)
        requests.append(
            SimpleNamespace(
                birth=birth,
                action_uid=7,
                action_slot=0,
                minimum_receipts=1,
                swing_generation_start=0,
            )
        )
    command._action_ball_provider_births = providers
    batches = namespace["_action_ball_refill_pool_many"](
        command, tuple(requests)
    )
    assert solver_calls == [4096]
    assert sampler.calls == 4096
    assert len(batches) == 4096
    assert all(batch.proposed_count == 1 for batch in batches)
    assert batches[0].proposal_sample_indices == (0,)
    assert batches[-1].proposal_sample_indices == (4095,)
    assert all(
        len(batch.receipts) == receipts_per_birth
        for batch in batches
    )
    assert notes == {
        "P": 4096,
        "A": 4096 * receipts_per_birth,
    }
    assert command._action_ball_emitted_task_count_by_uid[7] == (
        4096 * receipts_per_birth
    )
    if timing_reason is None:
        assert command._action_ball_reject_counts == {7: {}}
    else:
        assert command._action_ball_reject_counts == {
            7: {timing_reason: 4096}
        }
        assert all(
            count == 0
            for count, _root in (
                command._action_ball_task_transcript_by_birth.values()
            )
        )


def test_live_emitted_sample_hook_delegates_full_identity_and_rejects_resigned_forgery(
    monkeypatch,
):
    method = _method("_action_ball_assert_emitted_sample")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    runtime_module = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_runtime"
    )

    class Receipt:
        def __init__(self, identity):
            self._identity = json.loads(json.dumps(identity))
            self.birth_sha256 = "1" * 64
            self.sampler_birth_sha256 = "2" * 64

        def to_dict(self):
            return {"identity": self._identity}

        @classmethod
        def from_dict(cls, value):
            return cls(value["identity"])

        def sampler_identity_receipt(self):
            return json.loads(json.dumps(self._identity))

        def __eq__(self, other):
            return isinstance(other, Receipt) and self._identity == other._identity

    runtime_module.ActionBallTaskReceipt = Receipt
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    canonical = {
        "sample_id": "3" * 64,
        "sample_index": 41,
        "draw_start": 700,
        "draw_end": 717,
        "landing_aim_w_xy_m": [2.4, 0.1],
        "contact_w_m": [0.1, -0.2, 1.1],
        "contact_offset_from_base_goal_b_yaw_m": [0.1, -0.2, 1.1],
        "spin_magnitude_radps": 4.0,
        "spin_direction_b_yaw": [0.0, 1.0, 0.0],
        "spin_w_radps": [0.0, 4.0, 0.0],
    }

    class Sampler:
        def __init__(self):
            self.calls = []
            self.state = {"draw": 717, "sample_count": 42}

        def assert_issued_sample(self, identity):
            self.calls.append(json.loads(json.dumps(identity)))
            if identity != canonical:
                raise ValueError("deterministic issued replay mismatch")

    sampler = Sampler()
    birth = SimpleNamespace(
        canonical_sha256="1" * 64,
        birth_id="2" * 64,
    )
    command = SimpleNamespace(
        _action_ball_provider_births={
            "1" * 64: {
                "runtime_birth": birth,
                "sampler_birth": birth,
            }
        },
        _action_ball_sampler=sampler,
    )
    original_state = json.loads(json.dumps(sampler.state))
    namespace["_action_ball_assert_emitted_sample"](
        command, Receipt(canonical)
    )
    assert sampler.state == original_state
    for patch in (
        {"landing_aim_w_xy_m": [2.5, 0.1]},
        {
            "contact_w_m": [0.2, -0.2, 1.1],
            "contact_offset_from_base_goal_b_yaw_m": [0.2, -0.2, 1.1],
        },
        {
            "spin_magnitude_radps": 5.0,
            "spin_direction_b_yaw": [1.0, 0.0, 0.0],
            "spin_w_radps": [5.0, 0.0, 0.0],
        },
    ):
        forged = {**canonical, **patch, "sample_id": "4" * 64}
        with pytest.raises(ValueError, match="issued replay mismatch"):
            namespace["_action_ball_assert_emitted_sample"](
                command, Receipt(forged)
            )
        assert sampler.state == original_state
    assert len(sampler.calls) == 4


def test_fixed_action_refill_accounts_rejects_upstream_of_policy_attempts():
    refill = _method_source("_action_ball_refill_pool_many")
    assert refill.index('"P", len(samples)') < refill.index(
        "solve_proposals("
    )
    assert "protos=self._action_ball_prototypes" in refill
    assert "base_quat=base_quat" in refill
    assert "cfg=self._action_ball_solver_cfg" in refill
    assert "orient_normal" not in refill
    assert "admission/rejection counts do not conserve proposals" in refill
    assert refill.index('"A", len(indices)') < refill.index(
        "ActionBallTaskReceipt.from_birth"
    )
    assert "swing_generation=(" in refill
    for forbidden in ('"F"', '"U_table"', '"U_fall"', '"U_collision"'):
        assert forbidden not in refill


def test_exact_n1_refill_prechecks_before_ordinary_solver_and_nests_task_identity():
    refill = _method_source("_action_ball_refill_pool_many")
    normalized = " ".join(refill.split())
    proposed = refill.index(
        'self._action_ball_note(state["slot"], "P", len(samples))'
    )
    conditional_precheck = refill.index(
        "if counter_rally_enabled:",
        refill.index("counter_rally_tasks ="),
    )
    precheck = refill.index("counter_rally_precheck(", conditional_precheck)
    ordinary_solve = refill.index("result = solve_proposals(", precheck)
    assert proposed < conditional_precheck < precheck < ordinary_solve
    assert (
        "if not precheck.eligible_for_solver:"
        in refill[precheck:ordinary_solve]
    )
    assert "reject_counts[reason]" in refill[precheck:ordinary_solve]
    assert "continue" in refill[precheck:ordinary_solve]
    assert (
        "flat_samples = eligible_samples"
        in refill[precheck:ordinary_solve]
    )
    assert (
        "flat_state_indices = eligible_state_indices"
        in refill[precheck:ordinary_solve]
    )
    assert "counter_rally_tasks = eligible_tasks" in refill[
        precheck:ordinary_solve
    ]
    assert (
        "frozen_action_uid=int( request.action_uid )"
        in normalized
    )
    assert (
        "solver_action_uid=int( self._action_ball_bindings[ "
        "state[\"slot\"] ].action_uid )"
        in normalized
    )

    receipt_begin = refill.index("counter_rally_task=(")
    receipt_end = refill.index(
        ")", refill.index("target_baseline_speed_mps=(", receipt_begin)
    )
    nested_receipt = refill[receipt_begin:receipt_end]
    assert "counter_rally_task_identity_type(" in nested_receipt
    for field in (
        "objective_profile_sha256=",
        "return_direction_env_xy=",
        "target_baseline_speed_mps=",
    ):
        assert field in nested_receipt
    assert receipt_begin > ordinary_solve


def test_exact_n1_install_validates_every_nested_identity_before_any_live_write():
    install = _method_source("_action_ball_commit_install")
    identity_check = install.index("receipt.require_counter_rally_task(")
    all_receipts = install.index("for receipt in receipts", identity_check)
    finite_check = install.index(
        "if any(not bool(torch.isfinite(value).all())"
    )
    first_live_write = install.index(
        "self.racket_target_pos_w[ids] ="
    )
    assert identity_check < all_receipts < finite_check < first_live_write
    validation_prefix = install[:first_live_write]
    for live_target in (
        "self.racket_target_pos_w[ids] =",
        "self.vb_vel_in_w[ids] =",
        "self._vb_target_xy_per_env[ids] =",
        "self._counter_rally_return_direction_env_xy[ids] =",
        "self._action_ball_task_by_env[",
    ):
        assert live_target not in validation_prefix
    mutation_suffix = install[first_live_write:]
    assert (
        "self._counter_rally_return_direction_env_xy[ids]"
        in mutation_suffix
    )
    assert (
        "self._counter_rally_target_baseline_speed_mps[ids]"
        in mutation_suffix
    )
    assert "self._counter_rally_task_identity_by_env[int(env)]" in (
        mutation_suffix
    )


def test_exact_n1_fitted_rollout_uses_only_actual_contact_rows_and_clears_cache():
    evaluate = _method_source("_vb_evaluate")
    clear_terms = evaluate.index(
        "self._counter_rally_reward_terms.zero_()"
    )
    clear_accepted = evaluate.index(
        "self._counter_rally_accepted.zero_()"
    )
    no_strike_return = evaluate.index(
        "if not bool(exact_strike.any()):"
    )
    assert clear_terms < clear_accepted < no_strike_return

    achieved_contact = evaluate.index(
        "v_plus, w_plus = _vb.predict_paddle_contact("
    )
    contact_gather = evaluate.index(
        "contact_ids = torch.where(gate)[0]"
    )
    fitted_rollout = evaluate.index(
        "counter_outcome = rollout_counter_rally_torch("
    )
    reward_scatter = evaluate.index(
        "self._counter_rally_reward_terms[",
        fitted_rollout,
    )
    assert (
        achieved_contact
        < contact_gather
        < fitted_rollout
        < reward_scatter
    )
    rally_block = evaluate[contact_gather:reward_scatter]
    for gathered_actual in (
        ")[contact_ids],",
        "v_plus[contact_ids]",
        "w_plus[contact_ids]",
        "self._vb_target_xy_per_env[contact_ids]",
    ):
        assert gathered_actual in rally_block
    assert "if int(contact_ids.numel()) > 0:" in rally_block
    assert "_ref_racket" not in rally_block
    assert "teacher" not in rally_block
    assert "dt_s=0.001" in rally_block
    assert "binding=self._counter_rally_torch_binding" in rally_block


def test_action_ball_runtime_has_no_training_selector_or_legacy_ball_producer():
    initialize = _method_source("_initialize_action_ball_runtime")
    refill = _method_source("_action_ball_refill_pool_many")
    recipe = _method_source("_sample_targets_action_ball")
    combined = "\n".join((initialize, refill, recipe))
    assert "solve_proposals(" in refill
    for forbidden in (
        "generate_continuous_questions",
        "_apply_question_bank_targets",
        "select_and_fit",
        "question_bank",
        "action_selector",
        "planner_selector",
    ):
        assert forbidden not in combined
    assert "balanced_clip_sampling=True" in initialize
    assert "balanced capability collection" in initialize


def test_production_runtime_explicitly_binds_20_60_20_sampling_on_init_and_resume():
    initialize = _method_source("_initialize_action_ball_runtime")
    decode = _method_source("_action_ball_decode_solver_mutable_state")
    load = _method_source("_action_ball_load_exact_resume_state_dict")
    combined = "\n".join((initialize, decode, load))
    assert "SamplingMixture" in initialize
    # One live constructor, one mutable-state verifier, and two disposable
    # exact-resume constructors all bind the same explicit production mix.
    assert combined.count("sampling_mixture=SamplingMixture()") == 4


def test_reference_termination_phase_is_per_env_true_reset_latched_and_resumed():
    initialize = _method_source("_initialize_action_ball_runtime")
    gate = _method_source("action_ball_reference_terminations_enabled")
    sample = _method_source("_sample_targets_action_ball")
    save = _method_source("_action_ball_exact_resume_state_dict")
    load = _method_source("_action_ball_load_exact_resume_state_dict")
    assert "_action_ball_reference_term_center_latch = torch.ones" in initialize
    assert "return self._action_ball_reference_term_center_latch" in gate
    assert "clip_id" not in gate
    assert "if true_reset:" in sample
    assert (
        "self._action_ball_reference_term_center_latch[ids]"
        in sample
    )
    assert sample.index("_action_ball_commit_install(") < sample.index(
        "self._action_ball_reference_term_center_latch[ids]"
    )
    assert '"reference_term_center_latch"' in save
    assert '"reference_term_center_latch"' in load
    assert (
        "self._action_ball_reference_term_center_latch.copy_("
        in load
    )


def test_hard_joint_outcomes_preserve_raw_overlap_and_never_enter_difficulty():
    classify = _method_source("_action_ball_reset_outcome_masks")
    close = _method_source("_action_ball_close_attempts")
    for name in (
        "joint_actual_forbidden",
        "joint_qdes_forbidden",
        "robot_hit_table",
        "_PHYSICAL_FALL_TERMINATION_TERMS",
        "_REFERENCE_TERMINATION_TERMS",
    ):
        assert name in classify
    assert classify.index("joint_actual =") < classify.index(
        "joint_qdes ="
    ) < classify.index("fall =") < classify.index("table =")
    assert "reference_failure" in classify
    assert "unattributed" in classify
    assert "if bool(unattributed.any())" in classify
    assert "collision = torch.zeros" in classify
    assert "fabricate ``U_collision``" in classify
    assert "terminated" in classify
    joint_qdes_expression = classify[
        classify.index("joint_qdes ="):classify.index("fall =")
    ]
    table_expression = classify[
        classify.index("table ="):classify.index("named_unsafe =")
    ]
    assert "~joint_actual" not in joint_qdes_expression
    assert "~joint_actual" not in table_expression
    assert "~joint_qdes" not in table_expression
    for ledger_name in (
        '"U_joint_qdes"',
        '"U_joint_actual"',
        '"U_table"',
        '"U_fall"',
        '"U_collision"',
    ):
        assert ledger_name in close
    assert "unsafe_union = (" in close
    safe_expression = close[
        close.index("safe = active"):close.index("legal = safe")
    ]
    assert "~unsafe_union" in safe_expression
    failed_expression = close[
        close.index("failed ="):close.index("unclassified_terminated")
    ]
    assert "joint_qdes" not in failed_expression
    assert "joint_actual" not in failed_expression
    assert "unsafe_unique" in close
    assert "unsafe_max > unsafe_unique" in close
    assert "unsafe_unique > unsafe_sum" in close


def test_exact_resume_captures_every_receipt_tape_queue_and_generation_without_io():
    save = _method_source("_action_ball_exact_resume_state_dict")
    stage = _method_source("_action_ball_stage_resume_curriculum")
    load = _method_source("_action_ball_load_exact_resume_state_dict")
    for key in (
        '"hard_contract"',
        '"solver"',
        '"physics"',
        '"curriculum"',
        '"mutable_state"',
        '"broker"',
        '"pool"',
        '"ledger"',
        '"env_state"',
        '"integrity_sha256"',
    ):
        assert key in save
    for vector in (
        '"action_uid"',
        '"action_slot"',
        '"reset_generation"',
        '"swing_generation"',
        '"attempt_active"',
        '"attempt_action_slot"',
        '"attempt_legal"',
        '"reference_term_center_latch"',
    ):
        assert vector in save
    assert "staged_curriculum.load_state_dict" in stage
    assert "staged_broker.load_state_dict" in load
    assert "staged_pool.load_state_dict" in load
    assert "_action_ball_stage_resume_curriculum" in load
    assert "_action_ball_load_frozen_evaluation_runtime" in stage
    assert "_action_ball_load_drain_reset_authority" in stage
    assert "staged_coordinator.load_state_dict" in stage
    assert "reconcile_published_request" in stage
    assert "arm_catalog_sha256=ARM_CATALOG_SHA256" in stage
    assert (
        "scheduler_contract_sha256=("
        in stage
        and "self._action_ball_curriculum.scheduler_contract_sha256"
        in stage
    )
    assert "if diagnostic_unauthorized:" in stage
    diagnostic_branch = stage[
        stage.index("if diagnostic_unauthorized:"):
        stage.index("else:", stage.index("if diagnostic_unauthorized:"))
    ]
    assert "_action_ball_load_frozen_evaluation_runtime" not in diagnostic_branch
    assert "_action_ball_decode_solver_mutable_state" in load
    assert "sample_highwater_for=" in load
    assert load.index("_action_ball_stage_resume_curriculum") < load.index(
        "staged_broker.load_state_dict"
    ) < load.index("staged_pool.load_state_dict")
    # V4 rebuilds and validates the whole evaluator/curriculum authority graph off to the side,
    # then swaps that staged graph into the live object before restoring the broker/pool views.
    assert load.index(
        'self._action_ball_curriculum = staged_runtime["curriculum"]'
    ) < load.index(
        "self._action_ball_broker.load_state_dict"
    ) < load.index("self._action_ball_pool.load_state_dict")
    staged_assert_birth = load[
        load.index("def _staged_assert_birth"):
        load.index("def _staged_assert_sample")
    ]
    staged_assert_sample = load[
        load.index("def _staged_assert_sample"):
        load.index("staged_domain =")
    ]
    assert "ActionBallSampler(" not in staged_assert_birth
    assert "ActionBallSampler(" not in staged_assert_sample
    assert 'staged_shared["sampler"].assert_issued_birth' in staged_assert_birth
    assert 'staged_shared["sampler"].assert_issued_sample' in staged_assert_sample
    assert 'restored_ledger[x_row][slot] += 1' in load
    assert "self._action_ball_attempt_active.zero_()" in load
    assert "self._action_ball_task_by_env = [None] * self.num_envs" in load
    for forbidden in (
        "solve_proposals",
        "write_root_state_to_sim",
        "write_joint_state_to_sim",
        "_action_ball_commit_install",
        "._action_ball_sampler.sample(",
    ):
        assert forbidden not in load


def _stage_resume_curriculum_method(namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            _method("_action_ball_stage_resume_curriculum"),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return namespace["_action_ball_stage_resume_curriculum"]


def test_diagnostic_exact_resume_round_trips_without_formal_receipt(
    monkeypatch,
):
    modules = _load_action_ball_dependency_modules(monkeypatch)
    curriculum = modules["action_ball_curriculum"]
    key = curriculum.ActionProfileKey(
        action_uid=101,
        profile_sha256="1" * 64,
        mobility="no_move",
    )
    config = curriculum.BallCurriculumConfig()
    scheduler = curriculum.ArmSchedulerConfig()
    source = curriculum.ActionBallCurriculum(
        contract_sha256="2" * 64,
        profile_order=(key,),
        sampler_sha256="3" * 64,
        solver_sha256="4" * 64,
        policy_contract_sha256="5" * 64,
        config=config,
        scheduler_config=scheduler,
    )
    state = source.state_dict()
    launch = {"diagnostic_unauthorized": True}

    stage = _stage_resume_curriculum_method(
        {
            "_action_ball_load_frozen_evaluation_runtime": (
                lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError(
                        "diagnostic resume attempted to read a formal "
                        "evaluator receipt"
                    )
                )
            ),
            "_action_ball_load_drain_reset_authority": (
                lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError(
                        "diagnostic resume attempted to read a formal "
                        "drain/reset receipt"
                    )
                )
            ),
        }
    )
    command = SimpleNamespace(
        _action_ball_diagnostic_unauthorized=True,
        _action_ball_evaluator_launch=launch,
        _action_ball_bundle=SimpleNamespace(contract_sha256="2" * 64),
        _action_ball_profile_keys=(key,),
        _action_ball_sampler=SimpleNamespace(
            sampler_contract_sha256="3" * 64
        ),
        _action_ball_solver_contract={"sha256": "4" * 64},
        _action_ball_curriculum=source,
        cfg=SimpleNamespace(
            action_ball_policy_contract_sha256="5" * 64
        ),
    )
    frozen = {
        "schema_version": 1,
        "diagnostic_unauthorized": True,
        "last_request_step": -1,
        "profile_cursor": 0,
        "next_kind_by_uid": {},
        "coordinator": None,
        "drain_source": None,
    }
    restored = stage(command, state, frozen)
    assert restored["curriculum"].state_dict() == state
    assert restored["curriculum"].scheduler_contract_sha256 == (
        scheduler.contract_sha256
    )
    assert restored["evaluator_authority"] is None
    assert restored["coordinator"] is None


def test_formal_exact_resume_reopens_v4_inbox_and_drain_authorities():
    stage = _method_source("_action_ball_stage_resume_curriculum")
    assert "_action_ball_load_frozen_evaluation_runtime(" in stage
    assert "_action_ball_load_drain_reset_authority(" in stage
    assert "staged_coordinator.load_state_dict(" in stage
    assert "staged_drain_source.load_state_dict(" in stage
    assert "staged_coordinator.reconcile_published_request()" in stage
    assert '"coordinator": staged_coordinator' in stage
    assert '"drain_authority": staged_drain_authority' in stage
    assert '"recovered_request": recovered_request' in stage
    assert "FrozenEvaluatorAuthority" not in stage


def test_hard_contract_is_path_stable_and_hashes_random_schedule():
    contract = _method_source("action_ball_hard_contract")
    assert ".relative_to(self._action_ball_repo_root).as_posix()" in contract
    assert '"sampling"' in contract
    for key in (
        '"action_ball_seed"',
        '"pool_refill_rows"',
        '"balanced_clip_sampling"',
        '"balanced_clip_sampling_seed"',
        '"external_overdraw_multiplier"',
        '"maximum_external_proposal_rounds"',
    ):
        assert key in contract
    assert "source_path.as_posix()" not in contract
    assert 'payload["canonical_sha256"] = _action_ball_canonical_sha256(payload)' in contract


def test_legal_result_only_latches_onto_an_installed_action_ball_attempt():
    book = _method_source("_vb_book_strike_step")
    assert "if self._action_ball_enabled:" in book
    assert (
        "legal & self._action_ball_attempt_active"
        in " ".join(book.split())
    )
