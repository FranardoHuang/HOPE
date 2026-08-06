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
        # S0 新增的必填开关: 课程自有的 level-zero 单题(所有 32 个 domain level
        # 全零时把物理字段钉在 profile 中心), 校验器要求它是精确 bool。
        # 出厂的 A211/C211 诊断 YAML 都是 true, 这里照抄真实发射配方。
        "action_ball_initial_center_single_question": True,
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


def test_diagnostic_fast_path_keeps_functional_solver_and_forces_one_row():
    initialize = _method_source("_initialize_action_ball_runtime")
    refill = _method_source("_action_ball_refill_pool_many")
    diagnostic_state = _method_source(
        "_action_ball_exact_resume_state_dict"
    )

    assert (
        "1\n            if diagnostic_unauthorized\n"
        "            else int(self.cfg.action_ball_pool_refill_rows)"
        in initialize
    )
    assert (
        "1.0\n            if diagnostic_unauthorized\n"
        "            else float(self.cfg.cq_overdraw)"
        in initialize
    )
    assert (
        "_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS"
        in initialize
    )
    assert "diagnostic_unauthorized=diagnostic_unauthorized" in initialize
    assert (
        "self._action_ball_sampler.sample_many_prevalidated("
        in refill
    )
    assert "self._action_ball_sampler.sample(" in refill
    assert "result = solve_proposals(" in refill
    assert "_solve_proposals_diagnostic_host_only(" in refill
    assert "_DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY" in refill
    assert "_diagnostic_prevalidated_authority=(" in refill
    assert "if diagnostic_unauthorized" in refill
    assert "solver_float_values = torch.tensor(" in refill
    assert refill.count("solver_float_values[") == 5
    assert refill.count("].view(solver_row_count, ") == 5
    assert "self._action_ball_effective_cq_overdraw" in refill
    assert (
        "self._action_ball_effective_cq_max_redraw_rounds"
        in refill
    )
    assert "host_packet = result.proposal_host_packet" in refill
    assert (
        "racket_velocity_rows = host_packet.racket_velocity_rows"
        in refill
    )
    assert (
        "racket_normal_rows = host_packet.racket_normal_rows"
        in refill
    )
    assert "residual_rows = host_packet.residual_rows" in refill
    assert ".cpu()" not in refill
    # Diagnostic authorization and checkpoint recoverability are independent.
    # A211/C211 remain promotion-unauthorized, but now serialize the complete
    # mutable sampler/cache/WAIT/task/latch graph for strict preflight plus a
    # mandatory fresh-reset continuation.
    assert '"exact_resume_supported": False' not in diagnostic_state
    assert "action_ball_diagnostic_checkpoint" not in diagnostic_state
    for exact_key in (
        '"curriculum"',
        '"mutable_state"',
        '"broker"',
        '"pool"',
        '"env_state"',
        '"task_wait"',
        '"runtime_latches"',
    ):
        assert exact_key in diagnostic_state

    reserve_start = MOTION_SOURCE.index(
        "    def _reserve_action_ball_true_reset("
    )
    reserve_end = MOTION_SOURCE.index(
        "    def _rollback_action_ball_true_reset(", reserve_start
    )
    reserve = MOTION_SOURCE[reserve_start:reserve_end]
    assert "diagnostic_fast_path" in reserve
    assert (
        "else self._action_ball_birth_broker.state_dict()" in reserve
    )
    assert "if not diagnostic_fast_path:" in reserve


def test_action_ball_runtime_waits_for_command_manager_construction():
    method = _method("_ensure_action_ball_runtime_initialized")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    ensure = namespace["_ensure_action_ball_runtime_initialized"]

    calls = []
    command = None

    def validate_binding():
        calls.append("validated")
        # The real validator asks Racket for its shared exact-state digest,
        # which re-enters action_ball_hard_contract().  Publication of the
        # initialized flag must make that nested ensure a no-op.
        ensure(command)

    command = SimpleNamespace(
        _action_ball_enabled=True,
        _action_ball_runtime_initialized=False,
        _action_ball_runtime_initializing=False,
        _env=SimpleNamespace(),
        _initialize_action_ball_runtime=lambda: calls.append("initialized"),
        _motion=lambda: SimpleNamespace(
            validate_action_ball_task_authority_binding=validate_binding
        ),
    )
    with pytest.raises(RuntimeError, match="CommandManager construction"):
        ensure(command)
    assert calls == []
    command._env.command_manager = object()
    ensure(command)
    assert calls == ["initialized", "validated"]
    assert command._action_ball_runtime_initialized is True
    assert command._action_ball_runtime_initializing is False
    ensure(command)
    assert calls == ["initialized", "validated"]

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


def test_teacher_start_actor_read_lazily_initializes_once_before_motion_access():
    methods = [
        _method("_ensure_action_ball_runtime_initialized"),
        _method("actor_time_to_teacher_start_s"),
    ]
    module = ast.Module(body=methods, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    ensure = namespace["_ensure_action_ball_runtime_initialized"]
    read_teacher_start = namespace["actor_time_to_teacher_start_s"]

    events = []

    class TensorStub:
        def to(self, *, dtype):
            events.append(("to", dtype))
            return self

    wait = TensorStub()
    motion = SimpleNamespace(
        action_ball_pre_swing_wait_remaining_s=wait,
        validate_action_ball_task_authority_binding=lambda: events.append(
            "validated"
        ),
    )

    def motion_getter():
        events.append("motion")
        return motion

    command = SimpleNamespace(
        _action_ball_enabled=True,
        _action_ball_runtime_initialized=False,
        _action_ball_runtime_initializing=False,
        _env=SimpleNamespace(command_manager=object()),
        _initialize_action_ball_runtime=lambda: events.append("initialized"),
        _motion=motion_getter,
        time_to_strike=SimpleNamespace(dtype="float32"),
    )
    command._ensure_action_ball_runtime_initialized = types.MethodType(
        ensure, command
    )

    assert read_teacher_start(command) is wait
    assert events == [
        "initialized",
        "motion",
        "validated",
        "motion",
        ("to", "float32"),
    ]

    events.clear()
    assert read_teacher_start(command) is wait
    assert events == ["motion", ("to", "float32")]


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


def test_diagnostic_motion_binding_emits_unauthorized_canonical_receipt():
    motion_tree = ast.parse(MOTION_SOURCE, filename=str(MOTION_COMMAND_PATH))
    motion_class = next(
        node
        for node in motion_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MotionCommand"
    )
    contract_method = next(
        node
        for node in motion_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "action_ball_motion_admission_hard_contract"
    )
    module = ast.Module(body=[contract_method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "hashlib": hashlib,
        "_canonical_json_bytes": lambda value: json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8"),
    }
    exec(compile(module, str(MOTION_COMMAND_PATH), "exec"), namespace)
    contract = namespace[
        "action_ball_motion_admission_hard_contract"
    ]

    command = SimpleNamespace(
        _action_ball_birth_broker=object(),
        _action_ball_trusted_repo_root=Path("/repo"),
        _action_ball_runtime_module_bound=object(),
        _canonical_diagnostic_unauthorized=True,
        _motion_file_sha256=("1" * 64,),
    )
    first = contract(command)
    second = contract(command)
    assert first == second
    assert first["diagnostic_unauthorized"] is True
    assert first["training_authorized"] is False
    assert first["motion_file_sha256"] == ["1" * 64]
    assert first["canonical_sha256"] == hashlib.sha256(
        namespace["_canonical_json_bytes"](
            {
                key: value
                for key, value in first.items()
                if key != "canonical_sha256"
            }
        )
    ).hexdigest()

    command._motion_file_sha256 = ("2" * 64,)
    assert contract(command)["canonical_sha256"] != first[
        "canonical_sha256"
    ]


def test_action_ball_sample_revalidates_registry_on_birth_and_task():
    source = _method_source("_sample_targets_action_ball")
    registry_argument = (
        "registry_sha256=self._action_ball_broker.registry_sha256"
    )
    assert source.count(registry_argument) == 2


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


#: A stand-in for the sealed per-symbol surface.  The solver profile only reads
#: five fields out of it, so a fixture is enough here; the surface's own
#: coverage gates are tested in test_action_ball_solver_semantic_surface.py.
_FIXTURE_SEMANTIC_SURFACE = {
    "payload": {
        "kind": "whole_body_tracking.action_ball.solver_semantic_surface",
        "schema_version": 1,
        "symbol_digest_algorithm": "fixture",
        "coverage_policy": {
            "pinned_sources": [
                "hope_commands.py",
                "continuous_questions.py",
            ]
        },
        "covered": {"hope_commands.py": {"a": "b", "c": "d"}},
    },
    "sha256": "9" * 64,
}


def _solver_profile_namespace():
    """Namespace with the module's real schema version and direction constant.

    Re-typing ``3`` here is how this test rotted for a whole schema bump: it
    kept asserting against a v2 payload the shipped code had stopped building.
    """

    namespace = {"hashlib": hashlib, "json": json}
    _module_assignments(
        (
            "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION",
            "_ACTION_BALL_SOLVER_FIXED_DIRECTION",
        ),
        namespace,
    )
    return namespace


def test_solver_profile_hashes_executable_speed_face_and_contact_fit_contract(
    monkeypatch,
):
    namespace = _solver_profile_namespace()
    canonical, knobs, build = _module_functions(
        (
            "_action_ball_canonical_sha256",
            "action_ball_declared_solver_knobs",
            "action_ball_solver_profile_contract",
        ),
        namespace,
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
        semantic_surface=_FIXTURE_SEMANTIC_SURFACE,
        source_sha256=sources,
        contact_geometry_contract=geometry_contract,
        net_top_z=0.9325,
    )
    # Compatibility fence: adding the exact-N1 objective must not perturb the
    # ordinary N5/N73 solver receipt when the optional identity is absent.
    # This digest was minted from the schema-v3 payload represented by the fixed
    # inputs above; keeping it literal catches even subtle key additions.
    assert contract["sha256"] == (
        "dd10cac3caabef5a4af744be65847d15de3b2b3adf1ea950dee71e97f1ac85b1"
    )
    assert contract["payload"]["solve"] == knobs(_solver_cfg())
    assert contract == build(
        _solver_cfg(),
        physics_profile_sha256="4" * 64,
        semantic_surface=_FIXTURE_SEMANTIC_SURFACE,
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
        semantic_surface=_FIXTURE_SEMANTIC_SURFACE,
        source_sha256=sources,
        contact_geometry_contract=geometry_contract,
        net_top_z=0.9325,
    )["sha256"] != contract["sha256"]
    assert build(
        _solver_cfg(cq_overdraw=1.5),
        physics_profile_sha256="4" * 64,
        semantic_surface=_FIXTURE_SEMANTIC_SURFACE,
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
        semantic_surface=_FIXTURE_SEMANTIC_SURFACE,
        source_sha256=sources,
        contact_geometry_contract=drifted_geometry,
        net_top_z=0.9325,
    )["sha256"] != contract["sha256"]
    # A different sealed surface is a different solver profile: that binding is
    # the entire point of schema v3.
    assert build(
        _solver_cfg(),
        physics_profile_sha256="4" * 64,
        semantic_surface={
            "payload": dict(_FIXTURE_SEMANTIC_SURFACE["payload"]),
            "sha256": "a" * 64,
        },
        source_sha256=sources,
        contact_geometry_contract=geometry_contract,
        net_top_z=0.9325,
    )["sha256"] != contract["sha256"]


def test_counter_rally_solver_contract_appends_exact_ordered_rejections_and_identity(
    monkeypatch,
):
    canonical, _knobs, build = _module_functions(
        (
            "_action_ball_canonical_sha256",
            "action_ball_declared_solver_knobs",
            "action_ball_solver_profile_contract",
        ),
        _solver_profile_namespace(),
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
        semantic_surface=_FIXTURE_SEMANTIC_SURFACE,
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
        semantic_surface=_FIXTURE_SEMANTIC_SURFACE,
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
    # v3 dropped the whole-file map for the six adjudicated solver sources; the
    # counter-rally pair has never had a symbol-level adjudication, so it is the
    # only thing left on a coarse whole-file pin -- and it says so by name.
    assert counter["payload"]["unadjudicated_whole_file_sha256"] == {
        "counter_rally.py": "7" * 64,
        "counter_rally_torch.py": "8" * 64,
    }
    assert "implementation_source_sha256" not in counter["payload"]
    assert counter["sha256"] == canonical(counter["payload"])
    assert counter["sha256"] != ordinary["sha256"]

    with pytest.raises(
        ValueError,
        match="requires objective and venue-physics SHA together",
    ):
        build(
            _solver_cfg(),
            physics_profile_sha256="4" * 64,
            semantic_surface=_FIXTURE_SEMANTIC_SURFACE,
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
    namespace = {"hashlib": hashlib, "json": json, "Path": Path}
    (param_names,) = _module_assignments(
        ("_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES",), namespace
    )
    namespace["_ACTION_BALL_PHYSICS_PROFILE_SCHEMA_VERSION"] = 1
    canonical, sha_file, build = _module_functions(
        (
            "_action_ball_canonical_sha256",
            "_action_ball_sha256_file",
            "action_ball_physics_profile_contract",
        ),
        namespace,
    )
    # The ten declared venue numbers are one named tuple shared with the runtime
    # cross-check; enumerating them again here would be a third copy.
    assert param_names == (
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
    venue = tmp_path / "venue.yaml"
    venue.write_text("physics: exact\n", encoding="utf-8")
    prm = SimpleNamespace(
        source_path=venue,
        **{
            name: float(index + 1) / 10.0
            for index, name in enumerate(param_names)
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


def test_reset_task_identity_is_packed_once_and_reused_across_python_seams():
    sample = _method_source("_sample_targets_action_ball")
    close = _method_source("_action_ball_close_attempts")
    retire = _method_source("_action_ball_retire_previous_births")
    install = _method_source("_action_ball_commit_install")

    assert "host_identity_rows = tuple(" in sample
    assert "torch.stack(" in sample
    for field in (
        "ids,",
        "action_slots,",
        "action_uids,",
        "reset_generations,",
        "swing_generations,",
        "previous_swing_generations,",
        "attempt_active.to(dtype=torch.long),",
    ):
        assert field in sample
    assert sample.count(".cpu()") == 1
    assert sample.count(".tolist()") == 1
    assert ".item()" not in sample

    assert "active_host_env_ids=active_host_env_ids" in sample
    assert "host_env_ids=host_env_ids" in sample
    assert "for env_id in active_host_env_ids:" in close
    assert "for env_id in host_env_ids:" in retire
    assert "zip(host_env_ids, births, receipts)" in install
    # The global curriculum drain has no reset identity row pack, so the
    # retirement seam must retain its one explicit compatibility fallback.
    assert "if host_env_ids is None:" in retire
    assert "ids.detach().cpu().tolist()" in retire


def test_diagnostic_install_packet_matches_fixed_receipt_tape():
    torch = pytest.importorskip("torch")

    pack_rows, bool_packet = _module_functions(
        (
            "_action_ball_pack_diagnostic_install_rows",
            "_action_ball_host_bool_packet",
        ),
        {"torch": torch},
    )

    receipts = (
        SimpleNamespace(
            ball_contact_w_m=(1.0, 2.0, 3.0),
            racket_site_target_w_m=(4.0, 5.0, 6.0),
            base_goal_w_m=(7.0, 8.0, 9.0),
            racket_face_center_velocity_w_mps=(10.0, 11.0, 12.0),
            racket_site_velocity_w_mps=(13.0, 14.0, 15.0),
            racket_command_quat_wxyz=(16.0, 17.0, 18.0, 19.0),
            racket_normal_w=(20.0, 21.0, 22.0),
            incoming_velocity_w_mps=(23.0, 24.0, 25.0),
            incoming_spin_w_radps=(26.0, 27.0, 28.0),
            landing_aim_w_xy_m=(29.0, 30.0),
            time_to_contact_s=31.0,
        ),
        SimpleNamespace(
            ball_contact_w_m=(101.0, 102.0, 103.0),
            racket_site_target_w_m=(104.0, 105.0, 106.0),
            base_goal_w_m=(107.0, 108.0, 109.0),
            racket_face_center_velocity_w_mps=(110.0, 111.0, 112.0),
            racket_site_velocity_w_mps=(113.0, 114.0, 115.0),
            racket_command_quat_wxyz=(116.0, 117.0, 118.0, 119.0),
            racket_normal_w=(120.0, 121.0, 122.0),
            incoming_velocity_w_mps=(123.0, 124.0, 125.0),
            incoming_spin_w_radps=(126.0, 127.0, 128.0),
            landing_aim_w_xy_m=(129.0, 130.0),
            time_to_contact_s=131.0,
        ),
    )
    identities = (
        SimpleNamespace(
            return_direction_env_xy=(32.0, 33.0),
            target_baseline_speed_mps=34.0,
        ),
        SimpleNamespace(
            return_direction_env_xy=(132.0, 133.0),
            target_baseline_speed_mps=134.0,
        ),
    )
    packed = pack_rows(
        receipts=receipts,
        counter_rally_identities=identities,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    expected = torch.tensor(
        (
            tuple(float(value) for value in range(1, 35)),
            tuple(float(value) for value in range(101, 135)),
        ),
        dtype=torch.float64,
    )
    assert torch.equal(packed, expected)
    assert torch.equal(packed[:, 0:3], expected[:, 0:3])
    assert torch.equal(packed[:, 15:19], expected[:, 15:19])
    assert torch.equal(packed[:, 19:22], expected[:, 19:22])
    assert torch.equal(packed[:, 28:30], expected[:, 28:30])
    assert torch.equal(packed[:, 30], expected[:, 30])
    assert torch.equal(packed[:, 31:33], expected[:, 31:33])
    assert torch.equal(packed[:, 33], expected[:, 33])

    without_counter = pack_rows(
        receipts=receipts,
        counter_rally_identities=None,
        dtype=torch.float64,
        device="cpu",
    )
    assert torch.equal(without_counter, expected[:, :31])
    assert bool_packet(
        (
            torch.tensor(True),
            torch.tensor(False),
            torch.tensor(True),
        )
    ) == (True, False, True)


def test_diagnostic_reset_install_batches_transfers_and_predicates():
    pack = ast.get_source_segment(
        SOURCE,
        next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_action_ball_pack_diagnostic_install_rows"
        ),
    )
    install = _method_source("_action_ball_commit_install")
    close = _method_source("_action_ball_close_attempts")
    classify = _method_source("_action_ball_reset_outcome_masks")

    assert 'device="cpu"' in pack
    assert pack.count(".to(device=device)") == 1
    assert "_action_ball_pack_diagnostic_install_rows(" in install
    diagnostic_install = install[
        install.index("if self._action_ball_diagnostic_unauthorized:"):
        install.index(
            "\n        else:",
            install.index("if self._action_ball_diagnostic_unauthorized:"),
        )
    ]
    assert "torch.tensor(" not in diagnostic_install
    assert "_action_ball_host_bool_packet(" in install
    assert "torch.isfinite(initial_tts).all()" in install
    assert "torch.all(initial_tts > 0.0)" in install

    assert close.count("_action_ball_host_bool_packet(") == 2
    assert "_defer_unattributed_validation=True" in close
    assert "~unattributed.any()" in close
    assert "~unclassified_terminated.any()" in close
    assert "torch.all(unsafe_max <= unsafe_unique)" in close
    assert "torch.all(unsafe_unique <= unsafe_sum)" in close
    # Default/formal evaluation still synchronously enforces attribution at
    # the original boundary instead of entering the diagnostic packet path.
    assert "if not _defer_unattributed_validation:" in classify
    assert "if bool(unattributed.any())" in classify


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
        pool_owns_birth_task_transcripts=False,
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
        claim_many=lambda uids: tuple(
            ("claim", uid) for uid in uids
        ),
        domain_cursor_for=lambda _uid: 0,
        state_getter=state_getter,
        state_loader=state_loader,
    )
    provider = provider_type(
        sampler_contract_sha256="e" * 64,
        state_owner_sha256=owner,
        provide=lambda request: ("birth", request),
        provide_many=lambda requests: tuple(
            ("birth", request) for request in requests
        ),
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
        pool_owns_birth_task_transcripts=False,
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
    assert domain.claim_many_for_actions((7, 9)) == (
        ("claim", 7),
        ("claim", 9),
    )
    assert domain.domain_cursor_for(7) == 0
    assert provider("request") == ("birth", "request")
    assert provider.provide_many(("a", "b")) == (
        ("birth", "a"),
        ("birth", "b"),
    )
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

    def view(self, row_count, column_count):
        assert row_count * column_count == len(self.values)
        return _FakeTensor(
            [
                tuple(
                    self.values[
                        row * column_count : (row + 1) * column_count
                    ]
                )
                for row in range(row_count)
            ],
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


def test_action_ball_reference_host_rows_are_bound_once_as_immutable_copies():
    cache_method = _method("_action_ball_cache_reference_host_rows")
    module = ast.Module(body=[cache_method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    cache_rows = namespace["_action_ball_cache_reference_host_rows"]

    class ReferenceTensor:
        def __init__(self, values, *, shape, device="cuda:0"):
            self.values = [list(row) for row in values]
            self.shape = shape
            self.device = device
            self.dtype = "float32"
            self.cpu_calls = 0

        def detach(self):
            return self

        def cpu(self):
            self.cpu_calls += 1
            return self

        def tolist(self):
            return [list(row) for row in self.values]

    action = SimpleNamespace(
        action_id="bh_block",
        action_uid=7,
        motion_path="motions/bh_block.npz",
        motion_sha256="1" * 64,
    )
    binding = SimpleNamespace(
        action_uid=7,
        action_slot=0,
        motion_path="motions/bh_block.npz",
        motion_sha256="1" * 64,
        profile_sha256="2" * 64,
    )
    quat = ReferenceTensor(
        ((1.0, 0.0, 0.0, 0.0),),
        shape=(1, 4),
    )
    omega = ReferenceTensor(
        ((0.1, 0.2, 0.3),),
        shape=(1, 3),
    )
    velocity = ReferenceTensor(
        ((1.1, 1.2, 1.3),),
        shape=(1, 3),
    )
    raw_normal = ReferenceTensor(
        ((0.0, 1.0, 0.0),),
        shape=(1, 3),
    )
    command = SimpleNamespace(
        device="cuda:0",
        _action_ball_manifest=SimpleNamespace(actions=(action,)),
        _action_ball_bindings=(binding,),
        _action_ball_bundle=SimpleNamespace(
            profile_sha256=("2" * 64,)
        ),
        _ref_racket_quat_w_per_clip=quat,
        _ref_racket_ang_vel_w_per_clip=omega,
        _ref_racket_vel_w_per_clip=velocity,
        _ref_racket_normal_raw_w_per_clip=raw_normal,
    )

    cache_rows(command)
    assert quat.cpu_calls == 1
    assert omega.cpu_calls == 1
    assert velocity.cpu_calls == 1
    assert raw_normal.cpu_calls == 1
    assert command._action_ball_reference_host_identity == (
        (
            "bh_block",
            7,
            0,
            "motions/bh_block.npz",
            "1" * 64,
            "2" * 64,
        ),
    )
    assert command._action_ball_reference_quat_host_rows == (
        (1.0, 0.0, 0.0, 0.0),
    )
    assert command._action_ball_reference_omega_host_rows == (
        (0.1, 0.2, 0.3),
    )
    assert command._action_ball_reference_velocity_host_rows == (
        (1.1, 1.2, 1.3),
    )
    assert command._action_ball_reference_raw_normal_host_rows == (
        (0.0, 1.0, 0.0),
    )
    assert isinstance(
        command._action_ball_reference_quat_host_rows, tuple
    )
    assert isinstance(
        command._action_ball_reference_quat_host_rows[0], tuple
    )

    # The cached receipt rows must not remain a mutable tensor/list view.
    quat.values[0][0] = -1.0
    omega.values[0][0] = 9.0
    velocity.values[0][0] = 9.0
    raw_normal.values[0][0] = 9.0
    assert command._action_ball_reference_quat_host_rows[0][0] == 1.0
    assert command._action_ball_reference_omega_host_rows[0][0] == 0.1
    assert command._action_ball_reference_velocity_host_rows[0][0] == 1.1
    assert command._action_ball_reference_raw_normal_host_rows[0][0] == 0.0


@pytest.mark.parametrize(
    ("binding_patch", "quat_shape", "omega_device", "message"),
    (
        (
            {"action_uid": 8},
            (1, 4),
            "cuda:0",
            "identity differs",
        ),
        (
            {},
            (1, 3),
            "cuda:0",
            "shape/device/dtype mismatch",
        ),
        (
            {},
            (1, 4),
            "cpu",
            "shape/device/dtype mismatch",
        ),
    ),
)
def test_action_ball_reference_host_cache_fails_closed_before_copy(
    binding_patch,
    quat_shape,
    omega_device,
    message,
):
    cache_method = _method("_action_ball_cache_reference_host_rows")
    module = ast.Module(body=[cache_method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    cache_rows = namespace["_action_ball_cache_reference_host_rows"]

    class ReferenceTensor:
        dtype = "float32"

        def __init__(self, values, *, shape, device):
            self.values = values
            self.shape = shape
            self.device = device
            self.cpu_calls = 0

        def detach(self):
            return self

        def cpu(self):
            self.cpu_calls += 1
            return self

        def tolist(self):
            return self.values

    action = SimpleNamespace(
        action_id="bh_block",
        action_uid=7,
        motion_path="motions/bh_block.npz",
        motion_sha256="1" * 64,
    )
    binding_values = {
        "action_uid": 7,
        "action_slot": 0,
        "motion_path": "motions/bh_block.npz",
        "motion_sha256": "1" * 64,
        "profile_sha256": "2" * 64,
        **binding_patch,
    }
    quat = ReferenceTensor(
        [[1.0, 0.0, 0.0, 0.0]],
        shape=quat_shape,
        device="cuda:0",
    )
    omega = ReferenceTensor(
        [[0.1, 0.2, 0.3]],
        shape=(1, 3),
        device=omega_device,
    )
    velocity = ReferenceTensor(
        [[1.1, 1.2, 1.3]],
        shape=(1, 3),
        device=omega_device,
    )
    raw_normal = ReferenceTensor(
        [[0.0, 1.0, 0.0]],
        shape=(1, 3),
        device=omega_device,
    )
    command = SimpleNamespace(
        device="cuda:0",
        _action_ball_manifest=SimpleNamespace(actions=(action,)),
        _action_ball_bindings=(SimpleNamespace(**binding_values),),
        _action_ball_bundle=SimpleNamespace(
            profile_sha256=("2" * 64,)
        ),
        _ref_racket_quat_w_per_clip=quat,
        _ref_racket_ang_vel_w_per_clip=omega,
        _ref_racket_vel_w_per_clip=velocity,
        _ref_racket_normal_raw_w_per_clip=raw_normal,
    )
    with pytest.raises(RuntimeError, match=message):
        cache_rows(command)
    assert quat.cpu_calls == 0
    assert omega.cpu_calls == 0
    assert velocity.cpu_calls == 0
    assert raw_normal.cpu_calls == 0


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
        _action_ball_fixed_view_enabled=False,
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
        # This is the exact per-birth scope: the assignment transcript below is
        # the authority.  Live-births-only runs keep no such transcript and are
        # covered separately in test_action_ball_task_transcript_scope.py.
        _action_ball_birth_catalogs_are_live_only=lambda: False,
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
    (
        "solver_speed",
        "receipts_per_birth",
        "timing_reason",
        "sample_v_in_x",
        "solver_admit_counts",
        "effective_rounds",
        "sample_field_override",
        "base_quat_mode",
        "reverse_requests",
        "expected_pack_error",
    ),
    (
        (1.0, 1, None, -5.0, None, 1, None, "identity", False, None),
        (
            2.0,
            0,
            "teacher_rate_out_of_bounds",
            -5.0,
            None,
            1,
            None,
            "identity",
            False,
            None,
        ),
        (
            1.0,
            0,
            "ball_birth_not_beyond_net",
            -1.0,
            None,
            1,
            None,
            "identity",
            False,
            None,
        ),
        (
            1.0,
            1,
            None,
            -5.0,
            (2048, 1024, 512, 256, 256),
            64,
            None,
            "identity",
            False,
            None,
        ),
        (
            1.0,
            0,
            None,
            -5.0,
            (0, 0, 0, 0, 0),
            5,
            None,
            "identity",
            False,
            None,
        ),
        (
            1.0,
            0,
            None,
            -5.0,
            None,
            1,
            ("contact_w_m", (0.0, 0.0)),
            "identity",
            False,
            "contact_w_m row 0 must have exactly 3 values, got 2",
        ),
        (
            1.0,
            0,
            None,
            -5.0,
            None,
            1,
            None,
            "extra_nan",
            False,
            "base_quat_wxyz row 0 must have exactly 4 values, got 5",
        ),
        (
            1.0,
            1,
            None,
            -5.0,
            None,
            1,
            None,
            "distinct",
            True,
            None,
        ),
    ),
)
def test_refill_many_flattens_4096_births_and_rejects_timing_pre_issue(
    monkeypatch,
    solver_speed,
    receipts_per_birth,
    timing_reason,
    sample_v_in_x,
    solver_admit_counts,
    effective_rounds,
    sample_field_override,
    base_quat_mode,
    reverse_requests,
    expected_pack_error,
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
    # The refill entry point now cross-checks the sealed profile against the
    # objects it is about to solve with, so the real check travels with it.
    _module_assignments(
        (
            "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION",
            "_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS",
            "_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES",
        ),
        namespace,
    )
    _module_functions(
        ("action_ball_assert_solver_runtime_matches_declaration",), namespace
    )
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
    solver_knobs = {
        "n_iters": 24,
        "tol_m": 0.08,
        "global_speed_budget_mps": 3.4,
        "max_external_proposal_rounds": int(effective_rounds),
        "external_overdraw_multiplier": 1.0,
    }
    ball_params = {
        name: float(index + 1) / 10.0
        for index, name in enumerate(
            namespace["_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES"]
        )
    }
    solver_cfg = SimpleNamespace(
        tol_m=solver_knobs["tol_m"],
        n_iters=solver_knobs["n_iters"],
        speed_budget=solver_knobs["global_speed_budget_mps"],
        max_redraw_rounds=solver_knobs["max_external_proposal_rounds"],
        fixed_direction=True,
    )

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
    runtime_module._diagnostic_prevalidated_task_receipt_from_birth = (
        lambda birth, **kwargs: TaskReceipt.from_birth(
            birth,
            **kwargs,
        )
    )

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
    # The refill entry point's declaration cross-check reads these three live,
    # so this stub serves them out of continuous_questions.py itself.
    for _constant in (
        "BALL_BIRTH_NET_MARGIN_M",
        "CONTACT_NORMAL_SPEED_MIN_MPS",
        "CONTACT_NORMAL_SPEED_MAX_MPS",
    ):
        setattr(
            solver_module,
            _constant,
            _source_constant(CONTINUOUS_QUESTIONS_PATH, _constant),
        )
    solver_calls = []
    observed_base_quat_rows = []

    def solve_proposals(
        clip_ids,
        contact,
        incoming,
        spin,
        aim,
        ref_normal,
        **_kwargs,
    ):
        assert (
            _kwargs["_diagnostic_prevalidated_authority"]
            is solver_module._DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY
        )
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
        assert len(sizes) == 1
        row_count = sizes.pop()
        call_index = len(solver_calls)
        solver_calls.append(row_count)
        observed_base_quat_rows.append(
            tuple(_kwargs["base_quat"].values)
        )
        admitted_count = (
            row_count
            if solver_admit_counts is None
            else solver_admit_counts[call_index]
        )
        rejected_count = row_count - admitted_count
        admitted_values = (
            [True] * admitted_count
            + [False] * rejected_count
        )
        reason_values = (
            [-1] * admitted_count
            + [0] * rejected_count
        )
        velocity_values = [
            (solver_speed, 0.0, 0.0)
        ] * row_count
        normal_values = [(0.0, 1.0, 0.0)] * row_count
        residual_values = [0.0] * row_count
        return SimpleNamespace(
            ok=_FakeTensor(admitted_values),
            reason_counts=(
                {}
                if rejected_count == 0
                else {"no_landing": rejected_count}
            ),
            proposals=SimpleNamespace(
                reason_code=_FakeTensor(reason_values)
            ),
            v_racket=_FakeTensor(velocity_values),
            n_racket=_FakeTensor(normal_values),
            resid_m=_FakeTensor(residual_values),
            proposal_host_packet=SimpleNamespace(
                reason_codes=tuple(reason_values),
                admitted=tuple(admitted_values),
                racket_velocity_rows=tuple(velocity_values),
                racket_normal_rows=tuple(normal_values),
                residual_rows=tuple(residual_values),
            ),
        )

    def solve_proposals_diagnostic_host_only(*args, **kwargs):
        result = solve_proposals(*args, **kwargs)
        return result.proposal_host_packet, result.reason_counts

    solver_module.solve_proposals = solve_proposals
    solver_module._solve_proposals_diagnostic_host_only = (
        solve_proposals_diagnostic_host_only
    )
    solver_module._DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY = object()
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
            sample = SimpleNamespace(
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
            if sample_field_override is not None:
                field_name, value = sample_field_override
                setattr(sample, field_name, value)
            return sample

        def sample_many_prevalidated(self, *, count, **kwargs):
            return tuple(
                self.sample(**kwargs) for _ in range(count)
            )

    sampler = Sampler()
    notes = {"P": 0, "A": 0}

    def note(_slot, name, amount):
        notes[name] += amount

    command = SimpleNamespace(
        cfg=SimpleNamespace(
            cq_max_redraw_rounds=int(effective_rounds),
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
                "kind": (
                    "whole_body_tracking.continuous_questions.solve_proposals"
                ),
                "schema_version": namespace[
                    "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION"
                ],
                "fixed_direction": True,
                "solve": dict(solver_knobs),
                "integrator": {"h_s": 0.002, "n_steps": 1000},
                "acceptance": {
                        "landing": {"tol_m": solver_knobs["tol_m"]},
                        "net": {"ball_center_net_top_z_m": 0.9325},
                        "contact_normal_speed_fit": {
                            "minimum_mps_inclusive": _source_constant(
                                CONTINUOUS_QUESTIONS_PATH,
                                "CONTACT_NORMAL_SPEED_MIN_MPS",
                            ),
                            "maximum_mps_inclusive": _source_constant(
                                CONTINUOUS_QUESTIONS_PATH,
                                "CONTACT_NORMAL_SPEED_MAX_MPS",
                            ),
                        },
                        "incoming_birth": {
                            "net_margin_m": _source_constant(
                                CONTINUOUS_QUESTIONS_PATH,
                                "BALL_BIRTH_NET_MARGIN_M",
                            )
                        },
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
        _action_ball_physics_contract={
            "payload": {
                "kind": "whole_body_tracking.action_ball.physics_and_scorer",
                "virtual_ball_params": dict(ball_params),
                "geometry_and_grading": {
                    "ball_center_surface_z_m": 0.78,
                    "net_x_m": 1.87,
                    "ball_center_net_top_z_m": 0.9325,
                },
                "scorer_integrator": {"h_s": 0.002, "n_steps": 1000},
            }
        },
        _action_ball_sampler=sampler,
        _action_ball_reject_counts={7: {}},
        _action_ball_prototypes=object(),
        _action_ball_prm=SimpleNamespace(**ball_params),
        _action_ball_solver_cfg=solver_cfg,
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
        _action_ball_reference_quat_host_rows=(
            (1.0, 0.0, 0.0, 0.0),
        ),
        _action_ball_reference_omega_host_rows=(
            (0.0, 0.0, 0.0),
        ),
        _action_ball_mount_signs=(1,),
        _action_ball_attempt_close_margin_s=0.02,
        _action_ball_episode_length_s=2.0,
        _action_ball_note=note,
        _action_ball_provider_history={},
        _action_ball_task_transcript_by_birth={},
        _action_ball_emitted_task_count_by_uid={7: 0},
        _action_ball_diagnostic_unauthorized=True,
        _action_ball_effective_cq_max_redraw_rounds=effective_rounds,
        _action_ball_effective_cq_overdraw=1.0,
    )
    providers = {}
    requests = []
    for env_id in range(4096):
        if base_quat_mode == "identity":
            base_yaw_rad = 0.0
            base_quat_wxyz = (1.0, 0.0, 0.0, 0.0)
        elif base_quat_mode == "extra_nan":
            base_yaw_rad = 0.0
            base_quat_wxyz = (1.0, 0.0, 0.0, 0.0, float("nan"))
        elif base_quat_mode == "distinct":
            half_yaw = 0.5 * float(env_id) * 1.0e-4
            base_yaw_rad = 2.0 * half_yaw
            base_quat_wxyz = (
                math.cos(half_yaw),
                0.0,
                0.0,
                math.sin(half_yaw),
            )
        else:
            raise AssertionError(f"unknown base_quat_mode {base_quat_mode!r}")
        birth = SimpleNamespace(
            canonical_sha256=f"{env_id + 1:064x}",
            action_uid=7,
            domain_epoch=0,
            base_yaw_rad=base_yaw_rad,
            base_quat_wxyz=base_quat_wxyz,
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
    if reverse_requests:
        requests.reverse()
    if expected_pack_error is not None:
        with pytest.raises(RuntimeError, match=expected_pack_error):
            namespace["_action_ball_refill_pool_many"](
                command, tuple(requests)
            )
        assert solver_calls == []
        assert observed_base_quat_rows == []
        return
    batches = namespace["_action_ball_refill_pool_many"](
        command, tuple(requests)
    )
    if base_quat_mode == "distinct":
        assert observed_base_quat_rows == [
            tuple(
                request.birth.base_quat_wxyz
                for request in requests
            )
        ]
    if solver_admit_counts is None:
        expected_solver_calls = [4096]
    else:
        unresolved = 4096
        expected_solver_calls = []
        for admitted_count in solver_admit_counts:
            expected_solver_calls.append(unresolved)
            unresolved -= admitted_count
    assert solver_calls == expected_solver_calls
    assert sampler.calls == sum(expected_solver_calls)
    assert len(batches) == 4096
    assert batches[0].proposal_sample_indices[0] == 0
    if solver_admit_counts is None:
        assert all(batch.proposed_count == 1 for batch in batches)
        assert batches[-1].proposal_sample_indices == (4095,)
    else:
        assert sum(
            batch.proposed_count for batch in batches
        ) == sum(expected_solver_calls)
    assert all(
        len(batch.receipts) == receipts_per_birth
        for batch in batches
    )
    assert notes == {
        "P": sum(expected_solver_calls),
        "A": 4096 * receipts_per_birth,
    }
    assert command._action_ball_emitted_task_count_by_uid[7] == (
        4096 * receipts_per_birth
    )
    if solver_admit_counts is not None:
        expected_rejections = (
            sum(expected_solver_calls)
            - sum(solver_admit_counts)
        )
        assert command._action_ball_reject_counts == {
            7: {"no_landing": expected_rejections}
        }
    elif timing_reason is None:
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
    assert '"protos": self._action_ball_prototypes' in refill
    assert '"base_quat": base_quat' in refill
    assert '"cfg": self._action_ball_solver_cfg' in refill
    assert refill.count("**solver_kwargs") == 2
    assert "orient_normal" not in refill
    assert "admission/rejection counts do not conserve proposals" in refill
    assert "int(admitted.sum().item())" not in refill
    assert "sum(bool(value) for value in admitted_rows)" in refill
    assert "host_packet = result.proposal_host_packet" in refill
    assert "reason_codes = list(host_packet.reason_codes)" in refill
    assert "admitted_rows = list(host_packet.admitted)" in refill
    assert ".detach().cpu()" not in refill
    assert ".item()" not in refill
    assert "_ref_racket_quat_w_per_clip" not in refill
    assert "_ref_racket_ang_vel_w_per_clip" not in refill
    assert "_action_ball_reference_quat_host_rows" in refill
    assert "_action_ball_reference_omega_host_rows" in refill
    receipt_call = refill.index(
        "task_receipt_from_birth(",
        refill.index('"A", len(indices)'),
    )
    assert refill.index('"A", len(indices)') < receipt_call
    assert refill.index(
        "admission/rejection counts do not conserve proposals"
    ) < receipt_call
    assert refill.index(
        "exact-face geometry returned an unpinned"
    ) < receipt_call
    assert refill.index(
        "producer timing prefilter disagrees"
    ) < receipt_call
    assert refill.index(
        "aggregate and per-row rejection ledgers disagree"
    ) < receipt_call
    assert (
        "_diagnostic_prevalidated_task_receipt_from_birth"
        in refill
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
    assert "self._counter_rally_task_identity_by_env[env_id]" in (
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
        "if not exact_any:"
    )
    assert clear_terms < clear_accepted < no_strike_return
    formal_orphan = evaluate.index(
        "if bool((exact_strike & ~active).any()):"
    )
    formal_motion_read = evaluate.index(
        "current_clip = self._motion().clip_id.to(",
        formal_orphan,
    )
    formal_identity = evaluate.index(
        "if bool(active_identity_drift.any()):",
        formal_motion_read,
    )
    angular_velocity_read = evaluate.index(
        "current_angular_velocity = (",
        formal_identity,
    )
    assert (
        formal_orphan
        < formal_motion_read
        < formal_identity
        < angular_velocity_read
        < no_strike_return
    )

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


def test_task_timing_is_resolved_before_first_reset_observation_can_escape():
    sample = _method_source("_sample_targets_action_ball")
    assert (
        sample.index("_action_ball_commit_install(")
        < sample.index("motion.resolve_action_ball_task_timing_now(ids)")
        < sample.index(
            "self._action_ball_reference_term_center_latch[ids]"
        )
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
    assert 'restored_ledger[x_row][slot] += 1' not in load
    assert "self._action_ball_attempt_active.copy_(" in load
    assert "self._action_ball_task_by_env = tasks" in load
    assert "resume_reset_exclusion.copy_(" in load
    for forbidden in (
        "solve_proposals",
        "write_root_state_to_sim",
        "write_joint_state_to_sim",
        "_action_ball_commit_install",
        "._action_ball_sampler.sample(",
    ):
        assert forbidden not in load


def test_action_ball_exact_resume_preflight_is_read_only_and_precedes_commit():
    initialize = _method_source("_initialize_action_ball_runtime")
    validate = _method_source(
        "_action_ball_validate_exact_resume_state_dict"
    )
    load = _method_source("_action_ball_load_exact_resume_state_dict")

    assert (
        "self.validate_exact_resume_state_dict = (" in initialize
    )
    assert "_validate_only=True" in validate
    validation_boundary = load.index("if _validate_only:")
    first_live_commit = load.index(
        "self._action_ball_curriculum = staged_runtime"
    )
    assert validation_boundary < first_live_commit
    for staged_component in (
        "staged_task_wait_state",
        "staged_runtime_latches",
        "staged_curriculum",
        "staged_sampler",
        "staged_broker",
        "staged_pool",
    ):
        assert load.index(staged_component) < validation_boundary


def test_action_ball_resume_requires_fresh_reset_and_retires_open_attempt_as_x():
    load = _method_source("_action_ball_load_exact_resume_state_dict")
    close = _method_source("_action_ball_close_attempts")

    assert "resume_reset_exclusion.copy_(" in load
    assert "self._action_ball_attempt_active" in load
    assert "active = installed_active & ~resume_exclusion" in close
    assert "unattributed &= active" in close
    assert "clamped[resume_exclusion]" in close
    assert 'additions["X"]' in close
    assert "self._action_ball_resume_reset_exclusion[ids] = False" in close


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


def test_split_ready_wait_uses_one_task_valid_bit_and_no_transition_driver():
    arm_wait = _method_source("_action_ball_arm_task_wait")
    assert "motion.bind_action_ball_public_task_valid(" in arm_wait
    assert "motion._action_ball_time_to_contact_s[ids] += wait_s" in arm_wait
    assert "motion._action_ball_pre_swing_wait_s[ids] += wait_s" in arm_wait
    assert "_action_ball_task_valid[ids] = False" in arm_wait
    assert "connector" not in SOURCE
    assert "connector" not in MOTION_SOURCE
    assert "teacher-frame0 transition" not in MOTION_SOURCE

    motion_tree = ast.parse(MOTION_SOURCE, filename=str(MOTION_COMMAND_PATH))
    motion_class = next(
        node
        for node in motion_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MotionCommand"
    )
    methods = {
        node.name: ast.get_source_segment(MOTION_SOURCE, node)
        for node in motion_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "return ~task_valid" in methods["_action_ball_safe_ready_wait_mask"]
    assert "_action_ball_dynamic_ready_physical_joint_pos_rad" in methods["joint_pos"]
    assert "_action_ball_safe_ready_body_pos_w" in methods["body_pos_w"]
    assert "_action_ball_safe_ready_body_quat_w" in methods["body_quat_w"]
    for name in (
        "joint_vel",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "anchor_lin_vel_w",
        "anchor_ang_vel_w",
    ):
        assert "_action_ball_safe_ready_wait_mask()" in methods[name]
    assert "return torch.ones(" in methods["imitation_eligible"]


def test_hidden_wait_is_ineligible_and_active_miss_closes_zero_over_c():
    close = " ".join(_method_source("_action_ball_close_attempts").split())
    assert "active = active & self._action_ball_task_valid[ids]" in close
    assert '"C": torch.bincount(clamped[active]' in close
    assert '"H": torch.bincount(clamped[hit]' in close
    assert "hit = active & self._action_ball_attempt_hit[ids]" in close

    strike = " ".join(_method_source("_vb_book_strike_step").split())
    assert "hit = hit & self._action_ball_task_valid" in strike
    sparse = " ".join(
        _method_source("_book_sparse_reward_eligibility").split()
    )
    for name in (
        "exact_strike",
        "capture",
        "net_clear",
        "landing_valid",
        "legal_return",
    ):
        assert f"{name} = {name} & task_valid" in sparse


def test_every_live_birth_receipt_declares_the_sampler_initial_center_law():
    """Both live birth sites must record the law the sampler drew under.

    ``ActionBallSampler`` collapses the whole plan to the literal centre point
    while ``initial_center_single_question`` is on and all 32 curriculum arms
    are exactly zero, so the quota slot no longer picks the stratum.  The
    receipt gate can only judge that row if the receipt says which law applied.
    A birth site that forgets the keyword silently falls back to ``False`` and
    is then rejected by the quota comparison at the first level-zero reset --
    i.e. it breaks training, not a test.  Assert over *every* construction call
    so a future third site cannot be added without the keyword.
    """

    calls = [
        node
        for node in ast.walk(COMMAND_CLASS)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ActionBirthReceipt"
    ]
    assert len(calls) == 2, "birth receipt construction sites moved"
    for call in calls:
        keywords = {
            keyword.arg: keyword
            for keyword in call.keywords
            if keyword.arg is not None
        }
        assert "initial_center_single_question" in keywords
        value = " ".join(
            ast.get_source_segment(
                SOURCE, keywords["initial_center_single_question"]
            ).split()
        )
        # The value has to come from the sampler that produced the birth, not
        # from a literal or from the receipt's own default.
        assert (
            "self._action_ball_sampler.initial_center_single_question"
            in value
        )
        # Legacy births carry no mixture at all and the receipt refuses the
        # flag there, so the sampler law is gated on the mixture being present.
        assert "sampler_birth.sampling_mixture is not None" in value


# --------------------------------------------------------------------------- #
# The declaration/actual bridge.                                              #
#                                                                              #
# 人话:pin 封的是 payload 里"声明的数字";真正喂给求解器的数字要经过一条传递线,   #
# 而那条线住在 ``_initialize_action_ball_runtime``(语义面以 ``runtime_wiring``   #
# 为由排除的 1700 多行)。下面这组测试盯的就是这条线:声明必须只有一个出处、       #
# 映射必须只有一处、活值和声明必须逐字段相等,不等就 fail-closed。                #
# --------------------------------------------------------------------------- #
CONTINUOUS_QUESTIONS_PATH = COMMAND_PATH.with_name("continuous_questions.py")


def _module_assignments(names, namespace):
    """Execute named module-level assignments out of the shipped source.

    Reading the real assignment beats re-typing its value in the test: a test
    that carries its own copy of a constant stops being evidence the moment the
    two disagree, which is the exact failure this whole area exists to stop.
    """

    wanted = set(names)
    nodes = [
        node
        for node in TREE.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in wanted
            for target in node.targets
        )
    ]
    found = {
        target.id
        for node in nodes
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert found >= wanted, wanted - found
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def _source_constant(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level constant of {path}")


def _install_continuous_questions_stub(monkeypatch):
    """Serve the three covered acceptance constants out of their own source.

    ``hope_commands`` reads them at call time so that the payload cannot carry a
    stale hand-typed copy.  The test reads them the same way, for the same
    reason.
    """

    module_name = "whole_body_tracking.tasks.tracking.mdp.continuous_questions"
    stub = types.ModuleType(module_name)
    for constant in (
        "BALL_BIRTH_NET_MARGIN_M",
        "CONTACT_NORMAL_SPEED_MIN_MPS",
        "CONTACT_NORMAL_SPEED_MAX_MPS",
    ):
        setattr(
            stub, constant, _source_constant(CONTINUOUS_QUESTIONS_PATH, constant)
        )
    monkeypatch.setitem(sys.modules, module_name, stub)
    return stub


@pytest.fixture
def bridge(monkeypatch):
    """The four covered symbols of the declaration/actual bridge, plus a stub cq.

    ``continuous_questions`` is stubbed with values read out of its own source,
    not with numbers typed here.
    """

    namespace = {"hashlib": hashlib, "json": json}
    _module_assignments(
        (
            "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION",
            "_ACTION_BALL_PHYSICS_PROFILE_SCHEMA_VERSION",
            "_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS",
            "_ACTION_BALL_SOLVER_FIXED_DIRECTION",
            "_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES",
        ),
        namespace,
    )
    namespace["Path"] = Path
    (
        knobs,
        build_cfg,
        check,
        solver_contract,
        physics_contract,
        canonical,
        _sha_file,
    ) = _module_functions(
        (
            "action_ball_declared_solver_knobs",
            "action_ball_solver_cfg_from_declaration",
            "action_ball_assert_solver_runtime_matches_declaration",
            "action_ball_solver_profile_contract",
            "action_ball_physics_profile_contract",
            "_action_ball_canonical_sha256",
            "_action_ball_sha256_file",
        ),
        namespace,
    )
    stub = _install_continuous_questions_stub(monkeypatch)
    return SimpleNamespace(
        knobs=knobs,
        build_cfg=build_cfg,
        check=check,
        solver_contract=solver_contract,
        physics_contract=physics_contract,
        canonical=canonical,
        constants=namespace,
        cq=stub,
    )


class _FakeContinuousQuestionCfg:
    def __init__(
        self, *, tol_m, n_iters, speed_budget, max_redraw_rounds, fixed_direction
    ):
        self.tol_m = tol_m
        self.n_iters = n_iters
        self.speed_budget = speed_budget
        self.max_redraw_rounds = max_redraw_rounds
        self.fixed_direction = fixed_direction


def _bridge_world(bridge, tmp_path, **cfg_overrides):
    """One consistent (sealed payload, live object) pair, built the honest way."""

    venue = tmp_path / "venue.yaml"
    venue.write_text("physics: exact\n", encoding="utf-8")
    prm = SimpleNamespace(
        source_path=venue,
        **{
            name: float(index + 1) / 10.0
            for index, name in enumerate(
                bridge.constants["_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES"]
            )
        },
    )
    cfg = _valid_recipe(
        vb_table_surface_z=0.76,
        vb_min_landing_depth=0.15,
        **cfg_overrides,
    )
    planes = (0.78, 1.87, 0.9325)
    physics = bridge.physics_contract(
        cfg,
        prm,
        repo_root=tmp_path,
        surface_z=planes[0],
        net_x=planes[1],
        net_top_z=planes[2],
        opponent_near_x=0.5,
        opponent_far_x=3.24,
        table_half_width=0.7625,
    )
    solver = bridge.solver_contract(
        cfg,
        physics_profile_sha256=physics["sha256"],
        semantic_surface={
            "payload": {
                "kind": (
                    "whole_body_tracking.action_ball.solver_semantic_surface"
                ),
                "schema_version": 1,
                "symbol_digest_algorithm": "test",
                "coverage_policy": {"pinned_sources": ["hope_commands.py"]},
                "covered": {"hope_commands.py": {"a": "b"}},
            },
            "sha256": "9" * 64,
        },
        source_sha256={},
        contact_geometry_contract={"payload": {}, "sha256": "6" * 64},
        net_top_z=planes[2],
    )
    return SimpleNamespace(
        cfg=cfg,
        prm=prm,
        planes=planes,
        solver=solver,
        physics=physics,
        solver_cfg=bridge.build_cfg(cfg, _FakeContinuousQuestionCfg),
    )


def _run_check(bridge, world, **overrides):
    kwargs = {
        "solver_declaration": world.solver["payload"],
        "physics_declaration": world.physics["payload"],
        "solver_cfg": world.solver_cfg,
        "prm": world.prm,
        "planes": world.planes,
        "rollout_h": float(world.cfg.vb_rollout_h),
        "rollout_steps": int(world.cfg.vb_rollout_steps),
        "overdraw": float(world.cfg.cq_overdraw),
        "maximum_rounds": int(world.cfg.cq_max_redraw_rounds),
        "diagnostic_unauthorized": False,
        "call_site": "test",
    }
    kwargs.update(overrides)
    return bridge.check(**kwargs)


def test_declared_solver_knobs_are_the_only_source_of_the_payloads_solve_block(
    bridge, tmp_path
):
    world = _bridge_world(bridge, tmp_path)
    declared = bridge.knobs(world.cfg)
    assert world.solver["payload"]["solve"] == declared
    assert world.solver["payload"]["acceptance"]["landing"]["tol_m"] == (
        declared["tol_m"]
    )
    # The two contact-fit bounds used to be hand-typed literals in the payload.
    # They now read the live constants, so the declaration cannot go stale.
    # These three stay literal in the payload (the offline pinner mints it from
    # a git revision's source text), so what keeps them honest is the runtime
    # cross-check, not the builder.  Assert both halves.
    fit = world.solver["payload"]["acceptance"]["contact_normal_speed_fit"]
    assert fit["minimum_mps_inclusive"] == bridge.cq.CONTACT_NORMAL_SPEED_MIN_MPS
    assert fit["maximum_mps_inclusive"] == bridge.cq.CONTACT_NORMAL_SPEED_MAX_MPS
    assert world.solver["payload"]["acceptance"]["incoming_birth"][
        "net_margin_m"
    ] == bridge.cq.BALL_BIRTH_NET_MARGIN_M
    assert _run_check(bridge, world)["compared_field_count"] > 0
    assert world.solver["payload"]["fixed_direction"] is (
        bridge.constants["_ACTION_BALL_SOLVER_FIXED_DIRECTION"]
    )


def test_solver_cfg_mapping_carries_the_declared_knobs_and_nothing_else(
    bridge, tmp_path
):
    world = _bridge_world(bridge, tmp_path)
    declared = bridge.knobs(world.cfg)
    assert world.solver_cfg.tol_m == declared["tol_m"]
    assert world.solver_cfg.n_iters == declared["n_iters"]
    assert world.solver_cfg.speed_budget == declared["global_speed_budget_mps"]
    assert world.solver_cfg.max_redraw_rounds == (
        declared["max_external_proposal_rounds"]
    )
    assert world.solver_cfg.fixed_direction is (
        bridge.constants["_ACTION_BALL_SOLVER_FIXED_DIRECTION"]
    )


def test_runtime_cross_check_admits_the_honest_wiring_and_names_what_it_compared(
    bridge, tmp_path
):
    world = _bridge_world(bridge, tmp_path)
    receipt = _run_check(bridge, world)
    assert receipt["call_site"] == "test"
    assert receipt["diagnostic_unauthorized"] is False
    assert receipt["compared_field_count"] == len(receipt["compared_fields"])
    assert len(set(receipt["compared_fields"])) == len(
        receipt["compared_fields"]
    )
    # Every declared number, not a sample of them.
    for name in bridge.constants["_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES"]:
        assert "physics.virtual_ball_params.%s" % name in receipt[
            "compared_fields"
        ]
    for name in (
        "solver.fixed_direction",
        "solver.solve.n_iters",
        "solver.solve.tol_m",
        "solver.solve.global_speed_budget_mps",
        "solver.solve.max_external_proposal_rounds",
        "solver.solve.external_overdraw_multiplier",
        "solver.solve.max_external_proposal_rounds.effective",
        "solver.acceptance.net.ball_center_net_top_z_m",
        "physics.geometry_and_grading.net_x_m",
    ):
        assert name in receipt["compared_fields"], name


# The three mutations that escaped the per-symbol pin when the mapping lived
# inline in the excluded wiring function.  Here they are applied to the LIVE
# object only -- i.e. the wiring hands the solver something the sealed payload
# does not declare -- which is precisely the shape a static digest cannot see.
ESCAPED_WIRING_MUTATIONS = (
    ("tol_m", 0.5, "solver.solve.tol_m"),
    ("speed_budget", 2.0, "solver.solve.global_speed_budget_mps"),
)


@pytest.mark.parametrize(
    "field,factor,expected_name",
    ESCAPED_WIRING_MUTATIONS,
    ids=[name for name, _, _ in ESCAPED_WIRING_MUTATIONS],
)
def test_runtime_cross_check_catches_a_scaled_solver_knob(
    bridge, tmp_path, field, factor, expected_name
):
    world = _bridge_world(bridge, tmp_path)
    setattr(world.solver_cfg, field, getattr(world.solver_cfg, field) * factor)
    with pytest.raises(RuntimeError) as error:
        _run_check(bridge, world)
    assert expected_name in str(error.value)
    assert "fields drifted" in str(error.value)


def test_runtime_cross_check_catches_a_bumped_iteration_count(bridge, tmp_path):
    world = _bridge_world(bridge, tmp_path)
    world.solver_cfg.n_iters = world.solver_cfg.n_iters + 5
    with pytest.raises(RuntimeError, match="solver.solve.n_iters"):
        _run_check(bridge, world)


def test_runtime_cross_check_catches_the_flipped_fixed_direction_flag(
    bridge, tmp_path
):
    """The edit that made the declaration AND an exclusion reason false at once."""

    world = _bridge_world(bridge, tmp_path)
    world.solver_cfg.fixed_direction = False
    with pytest.raises(RuntimeError, match="solver.fixed_direction"):
        _run_check(bridge, world)


def test_runtime_cross_check_catches_a_doctored_plane_and_ball_parameter(
    bridge, tmp_path
):
    world = _bridge_world(bridge, tmp_path)
    with pytest.raises(
        RuntimeError, match="acceptance.net.ball_center_net_top_z_m"
    ):
        _run_check(
            bridge,
            world,
            planes=(world.planes[0], world.planes[1], world.planes[2] + 0.01),
        )
    drifted = SimpleNamespace(
        **{
            name: getattr(world.prm, name)
            for name in bridge.constants["_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES"]
        }
    )
    drifted.paddle_mu = drifted.paddle_mu * 1.5
    with pytest.raises(RuntimeError, match="virtual_ball_params.paddle_mu"):
        _run_check(bridge, world, prm=drifted)


def test_runtime_cross_check_catches_a_widened_overdraw_and_redraw_budget(
    bridge, tmp_path
):
    world = _bridge_world(bridge, tmp_path)
    with pytest.raises(RuntimeError, match="external_overdraw_multiplier"):
        _run_check(bridge, world, overdraw=float(world.cfg.cq_overdraw) * 2.0)
    with pytest.raises(
        RuntimeError, match=r"max_external_proposal_rounds\.effective"
    ):
        _run_check(
            bridge,
            world,
            maximum_rounds=int(world.cfg.cq_max_redraw_rounds) + 1,
        )


def test_the_diagnostic_exemption_is_exactly_two_named_constants(
    bridge, tmp_path
):
    """A diagnostic run may depart from the declaration in two places only."""

    world = _bridge_world(bridge, tmp_path)
    rounds = bridge.constants[
        "_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS"
    ]
    _run_check(
        bridge,
        world,
        diagnostic_unauthorized=True,
        overdraw=1.0,
        maximum_rounds=rounds,
    )
    # The exemption does not become a licence to hand over anything at all.
    with pytest.raises(RuntimeError, match="external_overdraw_multiplier"):
        _run_check(
            bridge,
            world,
            diagnostic_unauthorized=True,
            overdraw=1.5,
            maximum_rounds=rounds,
        )
    with pytest.raises(RuntimeError, match="solver.solve.tol_m"):
        _run_check(
            bridge,
            world,
            diagnostic_unauthorized=True,
            overdraw=1.0,
            maximum_rounds=rounds,
            solver_cfg=_FakeContinuousQuestionCfg(
                tol_m=world.solver_cfg.tol_m * 0.5,
                n_iters=world.solver_cfg.n_iters,
                speed_budget=world.solver_cfg.speed_budget,
                max_redraw_rounds=world.solver_cfg.max_redraw_rounds,
                fixed_direction=world.solver_cfg.fixed_direction,
            ),
        )


def test_runtime_cross_check_refuses_a_payload_that_is_not_the_sealed_profile(
    bridge, tmp_path
):
    world = _bridge_world(bridge, tmp_path)
    stale = dict(world.solver["payload"])
    stale["schema_version"] = 2
    with pytest.raises(RuntimeError, match="not the schema v"):
        _run_check(bridge, world, solver_declaration=stale)
    wrong_kind = dict(world.physics["payload"])
    wrong_kind["kind"] = "something.else"
    with pytest.raises(RuntimeError, match="not the physics/scorer profile"):
        _run_check(bridge, world, physics_declaration=wrong_kind)


def test_every_solve_entry_point_runs_the_cross_check_before_it_solves():
    """The load-bearing half: the boot call is not the one that matters.

    ``_initialize_action_ball_runtime`` is excluded from the semantic surface,
    so a call placed only there could be deleted without moving the pin.  These
    three methods are covered, so the call sites below are pinned.
    """

    for method in (
        "_action_ball_refill_pool_many",
        "_action_ball_replay_emitted_tasks",
        "_action_ball_frozen_eval_solve",
    ):
        source = _method_source(method)
        assert (
            "action_ball_assert_solver_runtime_matches_declaration(" in source
        ), method
        assert f'call_site="{method}"' in source
        check_at = source.index(
            "action_ball_assert_solver_runtime_matches_declaration("
        )
        solve_at = source.index("solve_proposals(")
        assert check_at < solve_at, (
            f"{method} solves before it checks the declaration"
        )


@pytest.mark.parametrize(
    "constant,field",
    (
        (
            "CONTACT_NORMAL_SPEED_MIN_MPS",
            "solver.acceptance.contact_normal_speed_fit.minimum_mps_inclusive",
        ),
        (
            "CONTACT_NORMAL_SPEED_MAX_MPS",
            "solver.acceptance.contact_normal_speed_fit.maximum_mps_inclusive",
        ),
        (
            "BALL_BIRTH_NET_MARGIN_M",
            "solver.acceptance.incoming_birth.net_margin_m",
        ),
    ),
)
def test_a_payload_literal_that_drifts_from_its_constant_is_refused(
    bridge, tmp_path, monkeypatch, constant, field
):
    """The declaration may be a literal, but it may not become a lie.

    The offline pinner mints this payload from a revision's source text, so the
    builder cannot import the live constant.  What replaces that is this
    comparison: change the constant without changing the declaration (or the
    other way round) and the run refuses before it draws a question.
    """

    world = _bridge_world(bridge, tmp_path)
    monkeypatch.setattr(bridge.cq, constant, getattr(bridge.cq, constant) + 1.0)
    with pytest.raises(RuntimeError) as error:
        _run_check(bridge, world)
    assert field in str(error.value)
