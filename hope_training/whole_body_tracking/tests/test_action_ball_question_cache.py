import copy
import importlib.util
import ast
from pathlib import Path
import sys
import types

import pytest


PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_question_cache.py"
)
SPEC = importlib.util.spec_from_file_location(
    "action_ball_question_cache_test_target", PATH
)
Q = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = Q
SPEC.loader.exec_module(Q)

COMMAND_PATH = PATH.with_name("hope_commands.py")


def _answer(value=1.0):
    return Q.CachedQuestionAnswer.from_values(
        reason_code=-1,
        admitted=True,
        racket_velocity=(value, value + 1.0, value + 2.0),
        racket_normal=(0.0, 1.0, 0.0),
        residual=0.001,
    )


def _payload(**overrides):
    payload = {
        "action_uid": 101,
        "domain_epoch": 0,
        "domain_levels": {"incoming_speed_lower": 0.0},
        "sampling_stratum": "domain",
        "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "contact_w_m": [0.5, 0.1, 0.9],
        "incoming_velocity_w_mps": [-4.0, 0.0, 0.0],
        "physics_sha256": "1" * 64,
    }
    payload.update(overrides)
    return payload


def test_q_q_qprime_counts_one_novel_then_hit_then_novel():
    cache = Q.ExactCurriculumQuestionCache((101,))
    q = Q.exact_question_sha256(_payload())
    q_prime = Q.exact_question_sha256(
        _payload(contact_w_m=[0.51, 0.1, 0.9])
    )
    answer = _answer()

    assert cache.peek(action_uid=101, question_sha256=q) is None
    cache.install_novel(
        action_uid=101, question_sha256=q, answer=answer
    )
    assert cache.note_hit(action_uid=101, question_sha256=q) == answer
    cache.install_novel(
        action_uid=101, question_sha256=q_prime, answer=_answer(2.0)
    )

    assert cache.consumer_hit_count == 1
    assert cache.novel_producer_count == 2
    assert cache.peek(action_uid=101, question_sha256=q) is None
    assert cache.peek(action_uid=101, question_sha256=q_prime) is not None


def test_static_q_reset_calls_once_then_qprime_invalidates_and_calls_once():
    """Novel-producer count is the exact inverse-solver call denominator."""

    cache = Q.ExactCurriculumQuestionCache((101,))
    q = Q.exact_question_sha256(_payload())
    for reset_index in range(8):
        answer = cache.peek(action_uid=101, question_sha256=q)
        if answer is None:
            cache.install_novel(
                action_uid=101, question_sha256=q, answer=_answer()
            )
        else:
            cache.note_hit(action_uid=101, question_sha256=q)
    assert cache.novel_producer_count == 1  # first Q reset: one solve
    assert cache.consumer_hit_count == 7  # unchanged Q resets: zero solves

    q_prime = Q.exact_question_sha256(
        _payload(domain_levels={"incoming_speed_lower": 0.25})
    )
    assert cache.peek(action_uid=101, question_sha256=q_prime) is None
    cache.install_novel(
        action_uid=101, question_sha256=q_prime, answer=_answer(2.0)
    )
    assert cache.novel_producer_count == 2  # Q' semantic change: one solve


@pytest.mark.parametrize(
    "change",
    [
        {"domain_epoch": 1},
        {"domain_levels": {"incoming_speed_lower": 0.25}},
        {"sampling_stratum": "frontier"},
        {"base_quat_wxyz": [0.999, 0.0, 0.0, 0.0447]},
        {"physics_sha256": "2" * 64},
        {"incoming_velocity_w_mps": [-4.1, 0.0, 0.0]},
    ],
)
def test_level_band_base_physics_and_continuous_question_changes_miss(change):
    cache = Q.ExactCurriculumQuestionCache((101,))
    q = Q.exact_question_sha256(_payload())
    cache.install_novel(
        action_uid=101, question_sha256=q, answer=_answer()
    )
    changed = Q.exact_question_sha256(_payload(**change))
    assert changed != q
    assert cache.peek(action_uid=101, question_sha256=changed) is None


def test_hot_and_cold_resume_are_exact_for_catalog_hot_rows():
    cache = Q.ExactCurriculumQuestionCache((101, 202))
    q1 = Q.exact_question_sha256(_payload())
    q2 = Q.exact_question_sha256(_payload(action_uid=202))
    cache.install_novel(
        action_uid=101, question_sha256=q1, answer=_answer()
    )
    cache.install_novel(
        action_uid=202, question_sha256=q2, answer=_answer(3.0)
    )
    cache.note_hit(action_uid=101, question_sha256=q1)
    sealed = cache.state_dict()

    resumed = Q.ExactCurriculumQuestionCache((202, 101))
    resumed.load_state_dict(copy.deepcopy(sealed))
    assert resumed.state_dict() == sealed
    assert resumed.note_hit(action_uid=202, question_sha256=q2) == _answer(3.0)
    assert resumed.consumer_hit_count == 2
    assert resumed.novel_producer_count == 2


def test_restore_is_atomic_and_corruption_does_not_mutate_live_cache():
    cache = Q.ExactCurriculumQuestionCache((101,))
    q = Q.exact_question_sha256(_payload())
    cache.install_novel(
        action_uid=101, question_sha256=q, answer=_answer()
    )
    before = cache.state_dict()
    corrupt = copy.deepcopy(before)
    corrupt["rows"][0]["answer"]["float64_bits_hex"] = "00"
    # Re-signing proves structural validation, not just outer integrity, is
    # what rejects the forged checkpoint.
    payload = {
        key: corrupt[key]
        for key in corrupt
        if key != "integrity_sha256"
    }
    corrupt["integrity_sha256"] = Q._canonical_sha256(payload)
    with pytest.raises(ValueError, match="wrong width"):
        cache.load_state_dict(corrupt)
    assert cache.state_dict() == before


def test_in_batch_duplicate_counts_consumer_without_duplicate_row():
    cache = Q.ExactCurriculumQuestionCache((101,))
    q = Q.exact_question_sha256(_payload())
    answer = _answer()
    cache.install_novel(
        action_uid=101, question_sha256=q, answer=answer
    )
    cache.note_in_batch_reuse(
        action_uid=101, question_sha256=q, answer=answer
    )
    assert cache.consumer_hit_count == 1
    assert cache.novel_producer_count == 1
    assert len(cache.state_dict()["rows"]) == 1


def test_mixed_same_action_questions_coexist_until_owning_birth_retires():
    cache = Q.ExactCurriculumQuestionCache((101,))
    q = Q.exact_question_sha256(_payload())
    q_prime = Q.exact_question_sha256(
        _payload(contact_w_m=[0.51, 0.1, 0.9])
    )
    birth_q = "a" * 64
    birth_q_prime = "b" * 64
    cache.install_novel(
        action_uid=101,
        question_sha256=q,
        answer=_answer(),
        birth_sha256=birth_q,
    )
    cache.install_novel(
        action_uid=101,
        question_sha256=q_prime,
        answer=_answer(2.0),
        birth_sha256=birth_q_prime,
    )

    assert cache.peek(action_uid=101, question_sha256=q) == _answer()
    assert cache.peek(action_uid=101, question_sha256=q_prime) == _answer(2.0)
    assert cache.row_count == 2
    assert cache.active_birth_count == 2

    cold = Q.ExactCurriculumQuestionCache((101,))
    cold.load_state_dict(cache.state_dict())
    assert cold.peek(action_uid=101, question_sha256=q) == _answer()
    assert cold.peek(action_uid=101, question_sha256=q_prime) == _answer(2.0)

    cold.retire_births((birth_q,))
    assert cold.peek(action_uid=101, question_sha256=q) is None
    assert cold.peek(action_uid=101, question_sha256=q_prime) == _answer(2.0)
    assert cold.row_count == 1
    assert cold.active_birth_count == 1
    cold.retire_births((birth_q_prime,))
    assert cold.peek(action_uid=101, question_sha256=q_prime) == _answer(2.0)
    assert cold.row_count == 1
    assert cold.active_birth_count == 0


def test_runtime_deduplicates_before_solver_and_commits_cache_last():
    tree = ast.parse(COMMAND_PATH.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_action_ball_refill_pool_many"
    )
    source = ast.get_source_segment(COMMAND_PATH.read_text(), method)
    assert source is not None
    assert "unique_miss_indices" in source
    assert "selector = torch.tensor" in source
    assert "miss_clip_ids = clip_ids.index_select" in source
    assert "solve_proposals(\n                            miss_clip_ids" in source
    assert "_solve_proposals_diagnostic_host_only(\n                                miss_clip_ids" in source
    assert "note_in_batch_reuse" in source
    # The live cache is updated only after task receipts, transcript counts,
    # and rejection ledgers have all been built successfully.
    assert source.rfind("live_question_cache.load_state_dict") > source.rfind(
        "_action_ball_emitted_task_count_by_uid"
    )


def test_command_checkpoint_and_cold_resume_round_trip_the_exact_cache():
    source = COMMAND_PATH.read_text()
    tree = ast.parse(source)
    wanted = {
        "_action_ball_solver_mutable_state_dict",
        "_action_ball_decode_solver_mutable_state",
        "_action_ball_load_solver_mutable_state",
        "_action_ball_load_exact_resume_state_dict",
    }
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    }
    state = methods["_action_ball_solver_mutable_state_dict"]
    decode = methods["_action_ball_decode_solver_mutable_state"]
    commit = methods["_action_ball_load_solver_mutable_state"]
    cold = methods["_action_ball_load_exact_resume_state_dict"]
    assert state is not None and '"exact_question_cache": (' in state
    assert decode is not None and 'state["exact_question_cache"]' in decode
    assert "staged_question_cache.load_state_dict" in decode
    assert "staged_question_cache.active_birth_sha256s" in decode
    assert commit is not None and "staged_question_cache.state_dict()" in commit
    assert cold is not None and "question_cache=staged_shared[\"decoded\"][1]" in cold


def test_direct_ball_branch_precedes_all_inverse_solver_branches():
    tree = ast.parse(COMMAND_PATH.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_action_ball_refill_pool_many"
    )
    source = ast.get_source_segment(COMMAND_PATH.read_text(), method)
    assert source is not None
    direct = source.index('if target_source == "direct_ball":')
    uncached = source.index("elif staged_question_cache is None:", direct)
    solve = source.index("_solve_proposals_diagnostic_host_only(", uncached)
    assert direct < uncached < solve
    direct_block = source[direct:uncached]
    assert "ProposalHostPacket(" in direct_block
    assert "solve_proposals(" not in direct_block
    assert "_solve_proposals_diagnostic_host_only(" not in direct_block


def test_4096_identical_questions_solve_once_then_hot_and_cold_hit_and_qprime_solves_once():
    """Report the actual producer-call denominator for Q/Q/Q'/cold restore."""

    cache = Q.ExactCurriculumQuestionCache((101,))
    solve_calls = 0

    def consume(payload):
        nonlocal solve_calls
        digest = Q.exact_question_sha256(payload)
        answer = cache.peek(action_uid=101, question_sha256=digest)
        if answer is None:
            solve_calls += 1
            answer = _answer(float(solve_calls))
            cache.install_novel(
                action_uid=101,
                question_sha256=digest,
                answer=answer,
            )
        else:
            cache.note_hit(action_uid=101, question_sha256=digest)
        return answer

    q = _payload()
    for _ in range(4096):
        consume(q)
    assert solve_calls == 1
    assert cache.novel_producer_count == 1
    assert cache.consumer_hit_count == 4095

    for _ in range(4096):
        consume(q)
    assert solve_calls == 1
    assert cache.consumer_hit_count == 8191

    cold = Q.ExactCurriculumQuestionCache((101,))
    cold.load_state_dict(cache.state_dict())
    digest = Q.exact_question_sha256(q)
    assert cold.peek(action_uid=101, question_sha256=digest) == _answer(1.0)
    assert solve_calls == 1

    consume(_payload(contact_w_m=[0.52, 0.1, 0.9]))
    assert solve_calls == 2
    assert cache.novel_producer_count == 2


def _load_replay_harness():
    tree = ast.parse(COMMAND_PATH.read_text())
    helper_names = {
        "_action_ball_semantic_levels",
        "_action_ball_exact_question_payload",
    }
    helpers = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    command_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
    )
    replay = next(
        copy.deepcopy(node)
        for node in command_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_action_ball_replay_emitted_tasks"
    )
    harness = ast.ClassDef(
        name="ReplayHarness",
        bases=[],
        keywords=[],
        body=[replay],
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *helpers,
            harness,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": pytest.importorskip("torch")}
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return namespace


def test_cache_enabled_pure_assertion_replays_mixed_4096_keys_without_inverse(
    monkeypatch,
):
    """Pool/cold-resume purity assertions must not erase the cache speedup."""

    loaded = _load_replay_harness()
    ReplayHarness = loaded["ReplayHarness"]
    build_payload = loaded["_action_ball_exact_question_payload"]

    class Levels:
        def to_dict(self):
            return {"incoming_speed_lower": 0.0}

    levels = Levels()
    birth = types.SimpleNamespace(
        domain_epoch=0,
        domain_levels=levels,
        base_yaw_rad=0.0,
        base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        base_spawn_w_m=(0.0, 0.0, 1.0),
        manifest_sha256="1" * 64,
        profile_sha256="2" * 64,
        motion_sha256="3" * 64,
        physics_sha256="4" * 64,
        solver_sha256="5" * 64,
        canonical_sha256="6" * 64,
    )

    class Receipt:
        def __init__(self, index):
            self.index = index
            self.canonical_sha256 = f"{index + 1:064x}"
            self.birth_sha256 = birth.canonical_sha256
            self.action_uid = 101
            self.action_slot = 0
            self.birth_sampling_stratum = "domain"
            self.birth_sampling_levels = levels
            self.birth_frontier_arm = None
            self.sampling_stratum = "domain"
            self.sampling_levels = levels
            self.frontier_arm = None
            self.base_goal_w_m = (0.0, 0.0, 1.0)
            self.base_travel_latent_b_yaw_m = (0.0, 0.0, 0.0)
            self.ball_contact_w_m = (
                (0.5 if index % 2 == 0 else 0.51),
                0.1,
                0.9,
            )
            self.time_to_contact_s = 0.8
            self.incoming_velocity_w_mps = (-4.0, 0.0, 0.0)
            self.incoming_spin_w_radps = (0.0, 10.0, 0.0)
            self.landing_aim_w_xy_m = (1.0, 0.0)
            self.racket_face_center_velocity_w_mps = (
                (1.0, 2.0, 3.0)
                if index % 2 == 0
                else (4.0, 5.0, 6.0)
            )
            self.racket_normal_w = (0.0, 1.0, 0.0)
            self.solver_residual_m = 0.001
            self.sampler_birth_sha256 = "7" * 64

        def to_dict(self):
            return {"receipt": self}

        @classmethod
        def from_dict(cls, value):
            return value["receipt"]

        def assert_contract(self, **_kwargs):
            return None

        def assert_birth(self, candidate):
            assert candidate is birth

        def sampler_identity_receipt(self):
            return self.index

    runtime_module = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_runtime"
    )
    runtime_module.ActionBallTaskReceipt = Receipt
    solver_calls = {"count": 0}

    def forbidden_solve(*_args, **_kwargs):
        solver_calls["count"] += 1
        raise AssertionError("cache assertion must not call inverse solver")

    continuous_module = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.continuous_questions"
    )
    continuous_module.solve_proposals = forbidden_solve
    monkeypatch.setitem(
        sys.modules,
        runtime_module.__name__,
        runtime_module,
    )
    monkeypatch.setitem(
        sys.modules,
        continuous_module.__name__,
        continuous_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.tasks.tracking.mdp.action_ball_question_cache",
        Q,
    )

    receipts = tuple(Receipt(index) for index in range(4096))
    cache = Q.ExactCurriculumQuestionCache((101,))
    for receipt in receipts[:2]:
        payload = build_payload(
            action_uid=101,
            action_slot=0,
            birth=birth,
            sample=receipt,
            mount_normal_sign=1,
        )
        digest = Q.exact_question_sha256(payload)
        answer = Q.CachedQuestionAnswer.from_values(
            reason_code=-1,
            admitted=True,
            racket_velocity=receipt.racket_face_center_velocity_w_mps,
            racket_normal=(0.0, 1.0, 0.0),
            residual=0.001,
        )
        cache.install_novel(
            action_uid=101,
            question_sha256=digest,
            answer=answer,
            birth_sha256=birth.canonical_sha256,
        )
    assert cache.row_count == 2
    cold = Q.ExactCurriculumQuestionCache((101,))
    cold.load_state_dict(cache.state_dict())

    sampler = types.SimpleNamespace(
        assert_issued_samples=lambda rows: len(rows) == 4096
    )
    harness = ReplayHarness()
    harness._action_ball_bindings = (object(),)
    harness._action_ball_pins = object()
    harness._action_ball_manifest = types.SimpleNamespace(mobility_mode="no_move")
    harness._action_ball_broker = types.SimpleNamespace(registry_sha256="8" * 64)
    harness._counter_rally_enabled = False
    harness._action_ball_target_source = "online_solver"
    harness._action_ball_mount_signs = (1,)
    harness._action_ball_assert_emitted_task_reference_and_timing = (
        lambda rows: len(rows) == 4096
    )

    harness._action_ball_replay_emitted_tasks(
        receipts,
        sampler=sampler,
        provider_history={birth.canonical_sha256: birth},
        question_cache=cache,
    )
    harness._action_ball_replay_emitted_tasks(
        receipts,
        sampler=sampler,
        provider_history={birth.canonical_sha256: birth},
        question_cache=cold,
    )
    assert solver_calls["count"] == 0
