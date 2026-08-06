"""Which per-birth bookkeeping a run keeps, and what may be reconciled against it.

人话:诊断跑(A211/C211 那种 4096 格的)每次出生**故意**不写两本"逐出生存档"
(``provider_history`` 和逐出生任务 transcript)。以前存 checkpoint 时却拿这两本空账
去核对"发出去多少任务"这个一直在涨的计数器,于是每个诊断跑都在 update 0 之后存盘那一
刻炸掉。这里钉死修复后的范围:

* 诊断跑(live-births-only)干净状态**必须**能序列化;
* 但它的计数器仍然要对账 —— 换成对"接纳提案账 A",少一条多一条都必须红;
* 精确跑那条严格对账**一个字没动**;
* 两种 scope 的 checkpoint 不能互相 resume,而且 live-births-only 的 checkpoint
  根本不能拿去做精确续跑。

每个测试都写成"检查粗一个档次就通不过"的形状:把范围守卫删掉、把对账整段删掉、或者
把 scope 牌子降级成"只要是个已知值就行",都会让下面某一条变红。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

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
SOURCE = COMMAND_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(COMMAND_PATH))
COMMAND_CLASS = next(
    node
    for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
)

_MODULE_FUNCTIONS = ("_action_ball_canonical_sha256",)
_MODULE_CONSTANTS = (
    "_ACTION_BALL_SOLVER_STATE_SCHEMA_VERSION",
    "_ACTION_BALL_LEDGER_NAMES",
    "_ACTION_BALL_TASK_TRANSCRIPT_SCOPE_EXACT",
    "_ACTION_BALL_TASK_TRANSCRIPT_SCOPE_DIAGNOSTIC",
    "_ACTION_BALL_TASK_TRANSCRIPT_SCOPES",
)
_METHODS = (
    "_action_ball_birth_catalogs_are_live_only",
    "_action_ball_task_transcript_scope",
    "_action_ball_online_solver_owns_admitted_task_counts",
    "_action_ball_admitted_proposal_row",
    "_action_ball_expected_admitted_task_counts_live_only",
    "_action_ball_live_ledger",
    "_action_ball_solver_mutable_state_dict",
    "_action_ball_decode_solver_mutable_state",
    "_action_ball_load_exact_resume_state_dict",
)


def _harness_namespace():
    """Execute exactly the module constants/functions the methods below close over."""

    body = []
    wanted_constants = set(_MODULE_CONSTANTS)
    wanted_functions = set(_MODULE_FUNCTIONS)
    found_constants = set()
    found_functions = set()
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & wanted_constants:
                body.append(node)
                found_constants |= names & wanted_constants
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            body.append(node)
            found_functions.add(node.name)
    assert found_constants == wanted_constants, sorted(
        wanted_constants - found_constants
    )
    assert found_functions == wanted_functions, sorted(
        wanted_functions - found_functions
    )
    namespace = {"__name__": "hope_commands_scope_harness"}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            ast.Import(names=[ast.alias("hashlib")]),
            ast.Import(names=[ast.alias("json")]),
            *body,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return namespace


NAMESPACE = _harness_namespace()
SCOPE_EXACT = NAMESPACE["_ACTION_BALL_TASK_TRANSCRIPT_SCOPE_EXACT"]
SCOPE_DIAGNOSTIC = NAMESPACE["_ACTION_BALL_TASK_TRANSCRIPT_SCOPE_DIAGNOSTIC"]
LEDGER_NAMES = NAMESPACE["_ACTION_BALL_LEDGER_NAMES"]


def _command_class():
    """Rehost the real method bodies on a bare class, with no simulator imports."""

    nodes = []
    for name in _METHODS:
        node = next(
            item
            for item in COMMAND_CLASS.body
            if isinstance(item, ast.FunctionDef) and item.name == name
        )
        nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = dict(NAMESPACE)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return type(
        "ScopeHarnessCommand",
        (),
        {name: namespace[name] for name in _METHODS},
    )


COMMAND_CLASS_OBJ = _command_class()


class _HostRow(list):
    """Minimal stand-in for one long ledger row on a torch tensor."""

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


def _ledger(*, proposed, admitted):
    rows = []
    for name in LEDGER_NAMES:
        if name == "P":
            rows.append(_HostRow(proposed))
        elif name == "A":
            rows.append(_HostRow(admitted))
        else:
            rows.append(_HostRow([0] * len(admitted)))
    return rows


def _command(
    *,
    live_only,
    admitted_task_counts,
    admitted=(4096,),
    proposed=(4096,),
    provider_history=None,
    task_transcript_by_birth=None,
):
    command = COMMAND_CLASS_OBJ()
    command._action_ball_diagnostic_unauthorized = bool(live_only)
    command._action_ball_fixed_view_enabled = False
    command._action_ball_banded_question_bank = None
    command._action_ball_immutable_tape = None
    command._action_ball_bundle = SimpleNamespace(action_uids=(7,))
    command._action_ball_provider_births = {}
    command._action_ball_provider_history = dict(provider_history or {})
    command._action_ball_task_transcript_by_birth = dict(
        task_transcript_by_birth or {}
    )
    command._action_ball_emitted_task_count_by_uid = dict(admitted_task_counts)
    command._action_ball_domain_cursor_by_uid = {7: 4096}
    command._action_ball_reject_counts = {7: {}}
    command._action_ball_exact_question_cache = None
    command._action_ball_state_owner_sha256 = "0" * 64
    command._action_ball_domain_authority_contract = {"sha256": "1" * 64}
    command._action_ball_sampler = SimpleNamespace(
        sampler_contract_sha256="2" * 64,
        state_dict=lambda: {"kind": "sampler-state"},
    )
    command._action_ball_solver_contract = {
        "sha256": "3" * 64,
        "payload": {
            "acceptance": {"ordered_rejection_reason_schema": ("no_landing",)}
        },
    }
    command._action_ball_curriculum = SimpleNamespace(
        state_dict=lambda: {"state_sha256": "4" * 64}
    )
    command._action_ball_ledger = _ledger(
        proposed=list(proposed), admitted=list(admitted)
    )
    return command


def _exact_transcript_catalog(*, admitted_count):
    """One birth carrying ``admitted_count`` admitted tasks in both catalogs."""

    birth_sha = "a" * 64
    history = {
        birth_sha: SimpleNamespace(
            action_uid=7,
            canonical_sha256=birth_sha,
            to_dict=lambda: {"birth": birth_sha},
        )
    }
    transcripts = {birth_sha: (admitted_count, "b" * 64)}
    return history, transcripts


def test_live_births_only_run_can_serialize_its_clean_solver_state():
    """The original blocker: diagnostic update-0 save must stop dying.

    Deleting the live-births-only branch (going back to reconciling against the
    intentionally empty transcript) puts this back in the red.
    """

    command = _command(
        live_only=True,
        admitted_task_counts={7: 4096},
        admitted=(4096,),
        proposed=(4096,),
    )
    payload = command._action_ball_solver_mutable_state_dict()
    assert payload["task_transcript_scope"] == SCOPE_DIAGNOSTIC
    assert payload["task_transcripts"] == []
    assert payload["provider_history"] == []
    assert payload["proposal_ledger"]["A"] == [4096]


def test_live_births_only_run_still_rejects_a_real_admitted_task_drift():
    """Scope moved, strength did not: one extra admitted task must still be red.

    Deleting the reconciliation instead of re-scoping it makes this pass
    silently, which is the whole point of the test.
    """

    command = _command(
        live_only=True,
        admitted_task_counts={7: 4097},
        admitted=(4096,),
        proposed=(4096,),
    )
    with pytest.raises(RuntimeError, match="admitted-proposal ledger"):
        command._action_ball_solver_mutable_state_dict()

    lost = _command(
        live_only=True,
        admitted_task_counts={7: 4095},
        admitted=(4096,),
        proposed=(4096,),
    )
    with pytest.raises(RuntimeError, match="admitted-proposal ledger"):
        lost._action_ball_solver_mutable_state_dict()


def test_live_births_only_run_rejects_a_per_birth_catalog_row():
    """The empty catalogs are a claim, so a row appearing in them is drift."""

    history, transcripts = _exact_transcript_catalog(admitted_count=0)
    command = _command(
        live_only=True,
        admitted_task_counts={7: 0},
        admitted=(0,),
        proposed=(0,),
        provider_history=history,
        task_transcript_by_birth=transcripts,
    )
    with pytest.raises(RuntimeError, match="live-births-only"):
        command._action_ball_solver_mutable_state_dict()


def test_exact_run_reconciliation_is_unchanged_and_still_catches_drift():
    """The strict per-birth reconciliation must be byte-for-byte as strong."""

    history, transcripts = _exact_transcript_catalog(admitted_count=4096)
    clean = _command(
        live_only=False,
        admitted_task_counts={7: 4096},
        admitted=(4096,),
        proposed=(4096,),
        provider_history=history,
        task_transcript_by_birth=transcripts,
    )
    payload = clean._action_ball_solver_mutable_state_dict()
    assert payload["task_transcript_scope"] == SCOPE_EXACT
    assert payload["task_transcripts"] == [
        {
            "birth_sha256": "a" * 64,
            "admitted_count": 4096,
            "transcript_sha256": "b" * 64,
        }
    ]

    drifted = _command(
        live_only=False,
        admitted_task_counts={7: 4097},
        admitted=(4097,),
        proposed=(4097,),
        provider_history=history,
        task_transcript_by_birth=transcripts,
    )
    with pytest.raises(RuntimeError, match="per-birth task transcript"):
        drifted._action_ball_solver_mutable_state_dict()


def _decode(command, payload):
    return command._action_ball_decode_solver_mutable_state(payload)


def test_a_diagnostic_checkpoint_cannot_be_decoded_by_an_exact_run():
    """Cross-scope resume must name the reason instead of zeroing what is missing.

    A guard that only checked "is this one of the two known scope names" would
    let the diagnostic payload through here.
    """

    producer = _command(
        live_only=True,
        admitted_task_counts={7: 4096},
        admitted=(4096,),
        proposed=(4096,),
    )
    payload = producer._action_ball_solver_mutable_state_dict()
    assert payload["task_transcript_scope"] == SCOPE_DIAGNOSTIC

    exact_run = _command(
        live_only=False,
        admitted_task_counts={7: 0},
        admitted=(0,),
        proposed=(0,),
    )
    with pytest.raises(ValueError) as excinfo:
        _decode(exact_run, payload)
    message = str(excinfo.value)
    assert "task-transcript scope mismatch" in message
    assert SCOPE_DIAGNOSTIC in message and SCOPE_EXACT in message


def test_an_exact_checkpoint_cannot_be_decoded_by_a_live_births_only_run():
    """The refusal is symmetric; neither side may silently adopt the other."""

    history, transcripts = _exact_transcript_catalog(admitted_count=4096)
    producer = _command(
        live_only=False,
        admitted_task_counts={7: 4096},
        admitted=(4096,),
        proposed=(4096,),
        provider_history=history,
        task_transcript_by_birth=transcripts,
    )
    payload = producer._action_ball_solver_mutable_state_dict()

    live_only_run = _command(
        live_only=True,
        admitted_task_counts={7: 0},
        admitted=(0,),
        proposed=(0,),
    )
    with pytest.raises(ValueError, match="task-transcript scope mismatch"):
        _decode(live_only_run, payload)


def test_matching_scope_gets_past_the_scope_gate():
    """Positive control: the gate must fire on the scope, not on everything."""

    producer = _command(
        live_only=True,
        admitted_task_counts={7: 4096},
        admitted=(4096,),
        proposed=(4096,),
    )
    payload = producer._action_ball_solver_mutable_state_dict()
    consumer = _command(
        live_only=True,
        admitted_task_counts={7: 0},
        admitted=(0,),
        proposed=(0,),
    )
    with pytest.raises(Exception) as excinfo:
        _decode(consumer, payload)
    assert "task-transcript scope" not in str(excinfo.value)


def test_state_without_the_scope_brand_is_refused_by_name():
    """An unbranded payload cannot prove its empty catalogs were deliberate."""

    producer = _command(
        live_only=True,
        admitted_task_counts={7: 4096},
        admitted=(4096,),
        proposed=(4096,),
    )
    payload = producer._action_ball_solver_mutable_state_dict()
    payload.pop("task_transcript_scope")
    consumer = _command(
        live_only=True,
        admitted_task_counts={7: 0},
        admitted=(0,),
        proposed=(0,),
    )
    with pytest.raises(ValueError, match="predates the task-transcript scope"):
        _decode(consumer, payload)


def test_live_births_only_run_refuses_exact_resume_outright():
    """A checkpoint with no per-birth transcript cannot serve an exact resume.

    The A211/C211 launch claims already declare ``resume_prohibited`` /
    ``fresh_only``; the runtime now says the same thing instead of discovering
    it halfway through a restore.
    """

    live_only_run = _command(
        live_only=True,
        admitted_task_counts={7: 0},
        admitted=(0,),
        proposed=(0,),
    )
    with pytest.raises(ValueError, match="cannot serve an exact resume"):
        live_only_run._action_ball_load_exact_resume_state_dict(
            {}, strict=True
        )

    exact_run = _command(
        live_only=False,
        admitted_task_counts={7: 0},
        admitted=(0,),
        proposed=(0,),
    )
    with pytest.raises(Exception) as excinfo:
        exact_run._action_ball_load_exact_resume_state_dict({}, strict=True)
    assert "cannot serve an exact resume" not in str(excinfo.value)


def test_a_banded_question_bank_run_must_keep_the_counter_at_zero():
    """Only the online proposal solver may move the admitted-task counter."""

    command = _command(
        live_only=True,
        admitted_task_counts={7: 12},
        admitted=(12,),
        proposed=(12,),
    )
    command._action_ball_banded_question_bank = object()
    with pytest.raises(RuntimeError, match="admitted-proposal ledger"):
        command._action_ball_solver_mutable_state_dict()

    quiet = _command(
        live_only=True,
        admitted_task_counts={7: 0},
        admitted=(12,),
        proposed=(12,),
    )
    quiet._action_ball_banded_question_bank = object()
    payload = quiet._action_ball_solver_mutable_state_dict()
    assert payload["task_transcript_scope"] == SCOPE_DIAGNOSTIC


def test_the_pool_is_told_who_owns_the_per_birth_roots():
    """A live-births-only run must declare the roots pool-owned, or the pool asks
    a catalog that was never written and the checkpoint dies one frame later."""

    initialize = ast.get_source_segment(
        SOURCE,
        next(
            node
            for node in COMMAND_CLASS.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_initialize_action_ball_runtime"
        ),
    )
    online = initialize[initialize.index("_ActionBallPoolSolverAdapter(") :]
    online = online[: online.index("\n        else:")]
    assert "pool_owns_birth_task_transcripts=bool(" in online
    assert "diagnostic_unauthorized" in online

    adapter_class = next(
        node
        for node in TREE.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_ActionBallPoolSolverAdapter"
    )
    namespace = dict(NAMESPACE)
    module = ast.Module(body=[adapter_class], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    kwargs = dict(
        solver_contract_sha256="a" * 64,
        state_owner_sha256="b" * 64,
        solve=lambda _request: None,
        solve_many=lambda _requests: (),
        assert_emitted_sample=lambda _receipt: None,
        assert_emitted_tasks=lambda _receipts: None,
        emitted_task_count_for=lambda _uid: 0,
        task_transcript_for_birth=lambda _birth: (0, "0" * 64),
        assert_proposal_assignments=lambda _assignments: None,
        sample_highwater_for=lambda _uid: (-1, 0),
        state_getter=lambda: {},
        state_loader=lambda _state: None,
    )
    adapter_type = namespace["_ActionBallPoolSolverAdapter"]
    assert (
        adapter_type(
            pool_owns_birth_task_transcripts=True, **kwargs
        ).pool_owns_birth_task_transcripts
        is True
    )
    assert (
        adapter_type(
            pool_owns_birth_task_transcripts=False, **kwargs
        ).pool_owns_birth_task_transcripts
        is False
    )
    with pytest.raises(TypeError):
        adapter_type(pool_owns_birth_task_transcripts=1, **kwargs)


def test_scope_is_a_property_of_the_run_not_of_the_provider_entry_point():
    """Both the scalar and the batched provider seams must agree on the scope."""

    scalar = ast.get_source_segment(
        SOURCE,
        next(
            node
            for node in COMMAND_CLASS.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_action_ball_provide_birth"
        ),
    )
    assert "_action_ball_birth_catalogs_are_live_only()" in scalar
    index = scalar.index("_action_ball_birth_catalogs_are_live_only()")
    tail = scalar[index:]
    assert "self._action_ball_provider_history[receipt_sha256] = receipt" in tail
