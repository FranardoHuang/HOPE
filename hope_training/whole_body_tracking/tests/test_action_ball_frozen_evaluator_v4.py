from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


MDP = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MDP / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load("action_ball_runtime")
C = _load("action_ball_curriculum")
E = _load("action_ball_evaluation")

CONTRACT = "a" * 64
SAMPLER = "b" * 64
SOLVER = "c" * 64
POLICY = "d" * 64
SOURCE_CONTRACT = "8" * 64
SOURCE_PATH = "tests/exact_frozen_attempt_source.py"
SOURCE_SHA = "6" * 64
SCHEDULER = C.ArmSchedulerConfig().contract_sha256
KEY = C.ActionProfileKey(7, "f" * 64, "no_move")
LEVELS = (0.0,) * len(C.ARM_KEYS)


def _digest(*parts):
    return hashlib.sha256(
        ":".join(str(part) for part in parts).encode("ascii")
    ).hexdigest()


class ExactAttemptSource:
    source_contract_sha256 = SOURCE_CONTRACT
    source_code_sha256 = SOURCE_SHA
    source_path = SOURCE_PATH

    def __init__(
        self,
        *,
        rejects=(),
        terminal_signals=None,
    ):
        self.rejects = frozenset(rejects)
        self.terminal_signals = dict(terminal_signals or {})
        self.state_owner_sha256 = _digest(
            "source-owner",
            ",".join(str(value) for value in sorted(self.rejects)),
            ",".join(
                f"{index}={signals.to_dict()}"
                for index, signals in sorted(
                    self.terminal_signals.items()
                )
            ),
        )
        self.issued = {}
        self.solver = {}
        self.lifecycle = {}
        self.terminal = {}
        self.tamper_solver = False
        self.revision = 0

    def state_fingerprint(self):
        return self.revision

    def state_dict(self):
        return {
            "issued": dict(sorted(self.issued.items())),
            "solver": dict(sorted(self.solver.items())),
            "lifecycle": dict(sorted(self.lifecycle.items())),
            "terminal": dict(sorted(self.terminal.items())),
            "revision": self.revision,
        }

    def load_state_dict(self, state):
        self.issued = dict(state["issued"])
        self.solver = dict(state["solver"])
        self.lifecycle = dict(state["lifecycle"])
        self.terminal = dict(state["terminal"])
        self.revision = state["revision"]

    def issue_proposal(self, request):
        slot = request.proposal_offset % 5
        stratum = (
            "center"
            if slot == 0
            else "frontier"
            if slot == 4
            else "interior"
        )
        proposal = R.FrozenIssuedProposal.create(
            reservation_sha256=request.reservation_sha256,
            source_contract_sha256=self.source_contract_sha256,
            sample_receipt_sha256=_digest(
                "sample",
                request.reservation_sha256,
            ),
            birth_receipt_sha256=_digest(
                "birth",
                request.reservation_sha256,
            ),
            action_uid=request.action_uid,
            profile_sha256=request.profile_sha256,
            mobility_mode=request.mobility_mode,
            domain_epoch=request.domain_epoch,
            levels_sha256=request.domain_levels.canonical_sha256,
            sample_index=request.sample_index,
            birth_index=request.birth_index,
            sampling_stratum=stratum,
            frontier_arm=(
                (
                    request.selected_arm_key
                    or R.ARM_KEYS[0]
                )
                if stratum == "frontier"
                else ""
            ),
        )
        self.issued[request.reservation_sha256] = (
            proposal.source_receipt_sha256
        )
        self.revision += 1
        return proposal

    def assert_exact_proposal(self, request, proposal):
        if (
            self.issued.get(request.reservation_sha256)
            != proposal.source_receipt_sha256
        ):
            raise ValueError("proposal is absent from exact issued tape")
        proposal.assert_request(request)

    def solver_event(self, request, proposal):
        rejected = request.proposal_offset in self.rejects
        event = R.FrozenSolverEvent.create(
            proposal_receipt_sha256=proposal.source_receipt_sha256,
            source_contract_sha256=self.source_contract_sha256,
            disposition="rejected" if rejected else "admitted",
            reject_reason="geometry_unreachable" if rejected else "",
            task_receipt_sha256=(
                ""
                if rejected
                else _digest("task", proposal.source_receipt_sha256)
            ),
        )
        if not self.tamper_solver:
            self.solver[proposal.source_receipt_sha256] = (
                event.event_receipt_sha256
            )
            self.revision += 1
        return event

    def assert_solver_event(self, request, proposal, event):
        del request
        if (
            self.solver.get(proposal.source_receipt_sha256)
            != event.event_receipt_sha256
        ):
            raise ValueError("solver event is absent from exact source tape")

    def lifecycle_event(
        self,
        request,
        proposal,
        solver,
        stage,
    ):
        del request
        event = R.FrozenLifecycleEvent.create(
            proposal_receipt_sha256=proposal.source_receipt_sha256,
            task_receipt_sha256=solver.task_receipt_sha256,
            source_contract_sha256=self.source_contract_sha256,
            stage=stage,
        )
        self.lifecycle[
            f"{proposal.source_receipt_sha256}:{stage}"
        ] = event.event_receipt_sha256
        self.revision += 1
        return event

    def assert_lifecycle_event(
        self,
        request,
        proposal,
        solver,
        event,
    ):
        del request, solver
        if (
            self.lifecycle.get(
                f"{proposal.source_receipt_sha256}:{event.stage}"
            )
            != event.event_receipt_sha256
        ):
            raise ValueError(
                "lifecycle event is absent from exact source tape"
            )

    def terminal_event(self, request, proposal, solver):
        signals = self.terminal_signals.get(
            request.proposal_offset,
            R.FrozenTerminalSignals(legal_return=True),
        )
        event = R.FrozenTerminalEvent.create(
            proposal_receipt_sha256=proposal.source_receipt_sha256,
            task_receipt_sha256=solver.task_receipt_sha256,
            source_contract_sha256=self.source_contract_sha256,
            signals=signals,
        )
        self.terminal[proposal.source_receipt_sha256] = (
            event.event_receipt_sha256
        )
        self.revision += 1
        return event

    def assert_terminal_event(
        self,
        request,
        proposal,
        solver,
        event,
    ):
        del request, solver
        if (
            self.terminal.get(proposal.source_receipt_sha256)
            != event.event_receipt_sha256
        ):
            raise ValueError(
                "terminal event is absent from exact source tape"
            )


def _launch():
    return E.launch_receipt_document_v4(
        curriculum_contract_sha256=CONTRACT,
        profile_order=(KEY,),
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=SCHEDULER,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        attempt_source_contract_sha256=SOURCE_CONTRACT,
        attempt_source_path=SOURCE_PATH,
        attempt_source_sha256=SOURCE_SHA,
    )


def test_v4_launch_accepts_equivalent_module_key_but_rejects_duck_types():
    alias = "action_ball_curriculum_equivalent_contract_test"
    spec = importlib.util.spec_from_file_location(
        alias, MDP / "action_ball_curriculum.py"
    )
    equivalent = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[alias] = equivalent
    try:
        spec.loader.exec_module(equivalent)
        key = equivalent.ActionProfileKey(7, "f" * 64, "no_move")
        assert type(key) is not C.ActionProfileKey
        launch = E.launch_receipt_document_v4(
            curriculum_contract_sha256=CONTRACT,
            profile_order=(key,),
            arm_catalog_sha256=C.ARM_CATALOG_SHA256,
            scheduler_contract_sha256=SCHEDULER,
            sampler_sha256=SAMPLER,
            solver_sha256=SOLVER,
            policy_contract_sha256=POLICY,
            attempt_source_contract_sha256=SOURCE_CONTRACT,
            attempt_source_path=SOURCE_PATH,
            attempt_source_sha256=SOURCE_SHA,
        )
        assert launch["profile_order"] == [KEY.as_dict()]
    finally:
        sys.modules.pop(alias, None)

    for fake in (
        KEY.as_dict(),
        SimpleNamespace(**KEY.as_dict()),
    ):
        with pytest.raises(ValueError, match="ActionProfileKey"):
            E.launch_receipt_document_v4(
                curriculum_contract_sha256=CONTRACT,
                profile_order=(fake,),
                arm_catalog_sha256=C.ARM_CATALOG_SHA256,
                scheduler_contract_sha256=SCHEDULER,
                sampler_sha256=SAMPLER,
                solver_sha256=SOLVER,
                policy_contract_sha256=POLICY,
                attempt_source_contract_sha256=SOURCE_CONTRACT,
                attempt_source_path=SOURCE_PATH,
                attempt_source_sha256=SOURCE_SHA,
            )


def _authority(source=None):
    launch = _launch()
    digest = E._canonical_sha256(launch)
    E.TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256 = (
        frozenset(
            set(
                E.TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256
            )
            | {digest}
        )
    )
    return E.FrozenEvaluatorV4Authority.from_trusted_launch_receipt(
        launch,
        attempt_source=source or ExactAttemptSource(),
    )


def _open(authority, snapshot, role, *, selected_arm_key=""):
    return authority.open_window(
        snapshot=snapshot,
        key=KEY,
        evidence_role=role,
        domain_epoch=3,
        stratum="marginal" if selected_arm_key else "center",
        selected_arm_key=selected_arm_key,
        selection_round=9,
        arm_levels=LEVELS,
        rho=0.0,
    )


def _finish(authority, session, count):
    for _ in range(count):
        handle = authority.issue_next(session)
        disposition = authority.capture_solver(handle)
        if disposition == "admitted":
            authority.capture_install(handle)
            authority.capture_start(handle)
            authority.capture_terminal(handle)
    return authority.finalize_window(session)


def test_v4_policy_generation_is_the_external_ppo_generation():
    authority = _authority()
    initial = authority.freeze_checkpoint(
        b"ppo-update-0",
        policy_generation=0,
    )
    first = authority.freeze_checkpoint(
        b"ppo-update-99",
        policy_generation=99,
    )
    second = authority.freeze_checkpoint(
        b"ppo-update-199",
        policy_generation=199,
    )
    assert initial.generation == 0
    assert first.generation == 99
    assert second.generation == 199
    assert authority.policy_snapshot(99).checkpoint_sha256 == (
        first.checkpoint_sha256
    )
    assert authority.policy_snapshot(0).checkpoint_sha256 == (
        initial.checkpoint_sha256
    )

    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="increase monotonically",
    ):
        authority.freeze_checkpoint(
            b"duplicate-or-regressed",
            policy_generation=199,
        )

    state = authority.state_dict()
    restored = _authority()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.policy_snapshot(199).checkpoint_sha256 == (
        second.checkpoint_sha256
    )


def test_v4_launch_and_source_are_exactly_code_pinned(monkeypatch):
    launch = _launch()
    monkeypatch.setattr(
        E,
        "TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256",
        frozenset(),
    )
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="not code-pinned",
    ):
        E.FrozenEvaluatorV4Authority.from_trusted_launch_receipt(
            launch,
            attempt_source=ExactAttemptSource(),
        )

    digest = E._canonical_sha256(launch)
    E.TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256 = (
        frozenset({digest})
    )
    source = ExactAttemptSource()
    source.source_code_sha256 = "0" * 64
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="source_code_sha256",
    ):
        E.FrozenEvaluatorV4Authority.from_trusted_launch_receipt(
            launch,
            attempt_source=source,
        )

    legacy = E.launch_receipt_document(
        curriculum_contract_sha256=CONTRACT,
        profile_order=(KEY,),
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=SCHEDULER,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        attempt_source_contract_sha256=SOURCE_CONTRACT,
        attempt_source_path=SOURCE_PATH,
        attempt_source_sha256=SOURCE_SHA,
    )
    with pytest.raises(ValueError, match="invalid keys"):
        E.FrozenEvaluatorV4Authority.from_trusted_launch_receipt(
            legacy,
            attempt_source=ExactAttemptSource(),
        )


def test_v4_privately_allocates_checkpoint_generation_and_disjoint_ranges():
    authority = _authority()
    snapshot = authority.freeze_checkpoint(b"checkpoint bytes, not a sha")
    assert snapshot.checkpoint_sha256 == hashlib.sha256(
        b"checkpoint bytes, not a sha"
    ).hexdigest()
    assert snapshot.generation == 1

    canary_session = _open(
        authority,
        snapshot,
        "frozen_canary",
    )
    canary = _finish(
        authority,
        canary_session,
        E.V4_CANARY_PROPOSALS,
    )
    heldout_session = _open(
        authority,
        snapshot,
        "frozen_heldout",
    )
    heldout = _finish(
        authority,
        heldout_session,
        E.V4_HELDOUT_PROPOSALS,
    )
    assert canary.evidence.seed_block_start == 0
    assert canary.evidence.seed_block_end_exclusive == 320
    assert heldout.evidence.seed_block_start == 320
    assert heldout.evidence.seed_block_end_exclusive == 1280
    assert canary.evidence.sample_id_end_exclusive == (
        heldout.evidence.sample_id_start
    )
    assert canary.evidence.unique_birth_count == 320
    assert heldout.evidence.unique_birth_count == 960

    release = authority.issue_release(
        canary=canary,
        heldout=heldout,
    )
    assert release.release_authorized
    assert release.schema_version == 4
    assert authority.assert_release_receipt(release) == (
        canary.evidence,
        heldout.evidence,
    )
    authority.consume_release(release)
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="stale",
    ):
        authority.assert_release_receipt(release)


def test_v4_api_rejects_self_reported_identity_outcome_and_new_band():
    authority = _authority()
    snapshot = authority.freeze_checkpoint(b"checkpoint")
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="opaque policy snapshot",
    ):
        authority.open_window(
            snapshot="e" * 64,
            key=KEY,
            evidence_role="frozen_canary",
            domain_epoch=0,
            stratum="center",
            selected_arm_key="",
            selection_round=0,
            arm_levels=LEVELS,
            rho=0.0,
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        authority.open_window(
            snapshot=snapshot,
            key=KEY,
            evidence_role="frozen_canary",
            domain_epoch=0,
            stratum="center",
            selected_arm_key="",
            selection_round=0,
            arm_levels=LEVELS,
            rho=0.0,
            policy_generation=99,
        )
    session = _open(
        authority,
        snapshot,
        "frozen_canary",
        selected_arm_key="incoming_speed_upper",
    )
    handle = authority.issue_next(session)
    authority.capture_solver(handle)
    authority.capture_install(handle)
    authority.capture_start(handle)
    with pytest.raises(TypeError, match="unexpected keyword"):
        authority.capture_terminal(
            handle,
            terminal_outcome="legal_return",
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        authority.capture_terminal(
            handle,
            in_new_band=True,
        )


def test_v4_terminal_precedence_and_new_band_are_code_derived():
    source = ExactAttemptSource(
        terminal_signals={
            0: R.FrozenTerminalSignals(
                joint_actual_limit=True,
                joint_qdes_limit=True,
                fall=True,
                table_hit=True,
                collision=True,
                legal_return=True,
            ),
            1: R.FrozenTerminalSignals(
                joint_qdes_limit=True,
                table_hit=True,
                legal_return=True,
            ),
        }
    )
    authority = _authority(source)
    snapshot = authority.freeze_checkpoint(b"joint-precedence")
    session = _open(
        authority,
        snapshot,
        "frozen_canary",
        selected_arm_key="incoming_speed_upper",
    )
    capability = _finish(
        authority,
        session,
        E.V4_CANARY_PROPOSALS,
    )
    ledger = capability.evidence.ledger
    assert ledger.U_joint_actual == 1
    assert ledger.U_joint_qdes == 2
    assert ledger.U_table == 2
    assert ledger.U_fall == 1
    assert ledger.U_collision == 1
    assert ledger.L == 318
    # Deterministic five-slot mixture in the exact source puts every fifth
    # proposal in the selected signed-arm frontier.
    assert ledger.NB == 64
    assert ledger.NB_F == 0
    rows = authority.attempt_rows(session)
    assert rows[0]["terminal_outcome"] == "joint_actual_limit"
    assert rows[0]["terminal_signals"]["table_hit"] is True
    assert rows[1]["terminal_outcome"] == "joint_qdes_limit"
    assert rows[1]["terminal_signals"]["table_hit"] is True
    assert sum(bool(row["in_new_band"]) for row in rows) == 64


def test_v4_reject_reason_conserves_ledger_and_tamper_is_rejected():
    source = ExactAttemptSource(rejects=range(10))
    authority = _authority(source)
    snapshot = authority.freeze_checkpoint(b"solver-rejects")
    session = _open(authority, snapshot, "frozen_canary")
    capability = _finish(
        authority,
        session,
        E.V4_CANARY_PROPOSALS,
    )
    ledger = capability.evidence.ledger
    assert (ledger.P, ledger.A, ledger.I, ledger.S, ledger.C) == (
        320,
        310,
        310,
        310,
        310,
    )
    rows = authority.attempt_rows(session)
    assert sum(
        row["reject_reason"] == "geometry_unreachable"
        for row in rows
    ) == 10

    bad_source = ExactAttemptSource()
    bad_authority = _authority(bad_source)
    bad_snapshot = bad_authority.freeze_checkpoint(b"tamper")
    bad_session = _open(
        bad_authority,
        bad_snapshot,
        "frozen_canary",
    )
    handle = bad_authority.issue_next(bad_session)
    bad_source.tamper_solver = True
    with pytest.raises(ValueError, match="absent"):
        bad_authority.capture_solver(handle)


def test_v4_optional_stopping_fails_and_crash_burns_ranges():
    authority = _authority()
    snapshot = authority.freeze_checkpoint(b"crash")
    session = _open(authority, snapshot, "frozen_canary")
    for _ in range(10):
        handle = authority.issue_next(session)
        authority.capture_solver(handle)
        authority.capture_install(handle)
        authority.capture_start(handle)
        authority.capture_terminal(handle)
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="optional stopping",
    ):
        authority.finalize_window(session)
    assert authority.burn_unfinished_after_crash(session) == 310
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="safe-closed floor",
    ):
        authority.finalize_window(session)

    # The failed fixed window still burns all 320 private allocations.
    next_session = _open(
        authority,
        snapshot,
        "frozen_heldout",
    )
    rows = authority.attempt_rows(next_session)
    assert rows[0]["seed"] == 320
    assert rows[0]["sample_index"] == 320
    assert rows[0]["birth_index"] == 320


def test_legacy_schema3_can_never_enter_v4_release():
    launch = E.launch_receipt_document(
        curriculum_contract_sha256=CONTRACT,
        profile_order=(KEY,),
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=SCHEDULER,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        attempt_source_contract_sha256=SOURCE_CONTRACT,
        attempt_source_path=SOURCE_PATH,
        attempt_source_sha256=SOURCE_SHA,
    )
    E.TRUSTED_FROZEN_EVALUATOR_LAUNCH_RECEIPT_SHA256 = (
        frozenset({E._canonical_sha256(launch)})
    )
    legacy = E.FrozenEvaluatorAuthority.from_trusted_launch_receipt(
        launch
    )
    assert not legacy.release_authorized
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="legacy schema-3",
    ):
        legacy.assert_release_receipt(object())

    v4 = _authority()
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="opaque schema-4",
    ):
        v4.assert_release_receipt(deepcopy(launch))


def test_v4_scheduler_transcript_is_trusted_single_use_and_resumable():
    authority = _authority()
    snapshot = authority.freeze_checkpoint(b"scheduler-resume")
    session = _open(authority, snapshot, "scheduler")
    for _ in range(5):
        handle = authority.issue_next(session)
        authority.capture_solver(handle)
        authority.capture_install(handle)
        authority.capture_start(handle)
        authority.capture_terminal(handle)
    allocation_sha = session._allocation_sha256
    state = deepcopy(authority.state_dict())
    authority.load_state_dict(state)
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="another or restored",
    ):
        authority.issue_next(session)

    resumed = authority.pending_session(allocation_sha)
    for _ in range(E.V4_SCHEDULER_PROPOSALS - 5):
        handle = authority.issue_next(resumed)
        authority.capture_solver(handle)
        authority.capture_install(handle)
        authority.capture_start(handle)
        authority.capture_terminal(handle)
    capability = authority.finalize_window(resumed)
    assert not capability.release_authorized
    inspected = authority.assert_scheduler_capabilities_many(
        {KEY: capability}
    )
    assert inspected[KEY][0].ledger.P == 100
    assert len(inspected[KEY][1]) == 100
    authority.consume_scheduler_capabilities_many({KEY: capability})
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="stale or consumed",
    ):
        authority.assert_scheduler_capabilities_many(
            {KEY: capability}
        )
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="scheduler ingest rejects",
    ):
        canary_session = _open(
            authority,
            snapshot=authority.policy_snapshot(1),
            role="frozen_canary",
        )
        canary = _finish(
            authority,
            canary_session,
            E.V4_CANARY_PROPOSALS,
        )
        authority.assert_scheduler_capabilities_many({KEY: canary})


def test_v4_pending_release_reacquires_after_exact_resume_and_old_is_stale():
    authority = _authority()
    snapshot = authority.freeze_checkpoint(b"release-resume")
    canary = _finish(
        authority,
        _open(authority, snapshot, "frozen_canary"),
        E.V4_CANARY_PROPOSALS,
    )
    heldout = _finish(
        authority,
        _open(authority, snapshot, "frozen_heldout"),
        E.V4_HELDOUT_PROPOSALS,
    )
    release = authority.issue_release(
        canary=canary,
        heldout=heldout,
    )
    release_id = release.release_id
    state = deepcopy(authority.state_dict())
    authority.load_state_dict(state)
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="foreign, stale",
    ):
        authority.assert_release_receipt(release)
    reacquired = authority.pending_release(release_id)
    assert authority.assert_release_receipts_many(
        {KEY: reacquired}
    )[KEY] == (
        canary.evidence,
        heldout.evidence,
    )
    authority.consume_releases_many({KEY: reacquired})
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="not pending",
    ):
        authority.pending_release(release_id)


def test_v4_state_rejects_tamper_and_legacy_migration():
    authority = _authority()
    snapshot = authority.freeze_checkpoint(b"state-tamper")
    session = _open(authority, snapshot, "scheduler")
    authority.burn_unfinished_after_crash(session)
    state = deepcopy(authority.state_dict())

    tampered = deepcopy(state)
    tampered["windows"][0]["attempts"][0]["request"]["seed"] = 999
    unsigned = dict(tampered)
    unsigned.pop("state_sha256")
    tampered["state_sha256"] = E._canonical_sha256(unsigned)
    with pytest.raises(
        (E.FrozenEvaluationAuthorityError, ValueError),
        match="reservation SHA|allocation",
    ):
        authority.load_state_dict(tampered)

    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="legacy/unsupported",
    ):
        authority.load_state_dict(
            {
                **state,
                "schema_version": 3,
            }
        )
