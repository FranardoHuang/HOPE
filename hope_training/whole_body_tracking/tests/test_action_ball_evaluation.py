from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys

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


C = _load("action_ball_curriculum")
E = _load("action_ball_evaluation")

CONTRACT = "a" * 64
SAMPLER = "b" * 64
SOLVER = "c" * 64
POLICY = "d" * 64
CHECKPOINT = "e" * 64
ATTEMPT_SOURCE = "8" * 64
ATTEMPT_SOURCE_PATH = "tests/frozen_attempt_source.py"
ATTEMPT_SOURCE_SHA = "6" * 64
SCHEDULER = C.ArmSchedulerConfig().contract_sha256
KEY = C.ActionProfileKey(7, "f" * 64, "no_move")


def _digest(namespace, index):
    return hashlib.sha256(f"{namespace}:{index}".encode("ascii")).hexdigest()


def _launch():
    return E.launch_receipt_document(
        curriculum_contract_sha256=CONTRACT,
        profile_order=(KEY,),
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=SCHEDULER,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        attempt_source_contract_sha256=ATTEMPT_SOURCE,
        attempt_source_path=ATTEMPT_SOURCE_PATH,
        attempt_source_sha256=ATTEMPT_SOURCE_SHA,
    )


def _authority():
    launch = _launch()
    launch_sha = E._canonical_sha256(launch)
    E.TRUSTED_FROZEN_EVALUATOR_LAUNCH_RECEIPT_SHA256 = frozenset(
        set(E.TRUSTED_FROZEN_EVALUATOR_LAUNCH_RECEIPT_SHA256)
        | {launch_sha}
    )
    return E.FrozenEvaluatorAuthority.from_trusted_launch_receipt(launch)


def _attempts(
    authority,
    *,
    count,
    sample_offset,
    birth_offset,
    duplicate_birth=False,
    failures=0,
):
    return tuple(
        authority.record_attempt(
            sample_receipt_sha256=_digest("sample", sample_offset + index),
            birth_receipt_sha256=_digest(
                "birth",
                birth_offset if duplicate_birth else birth_offset + index,
            ),
            solver_admitted=True,
            installed=True,
            started=True,
            closed=True,
            terminal_outcome=(
                "safe_nonreturn" if index < failures else "legal_return"
            ),
        )
        for index in range(count)
    )


def _issue(
    authority,
    *,
    role="frozen_canary",
    count=256,
    sample_offset=0,
    birth_offset=0,
    seq=1,
    window_id="window-1",
    duplicate_birth=False,
    failures=0,
):
    return authority.issue_window(
        key=KEY,
        policy_checkpoint_sha256=CHECKPOINT,
        policy_generation=1,
        evidence_role=role,
        domain_epoch=0,
        stratum="center",
        selected_arm_key="",
        selection_round=0,
        arm_levels=(0.0,) * len(C.ARM_KEYS),
        rho=0.0,
        seed_block_start=1_000_000 + sample_offset,
        seed_block_end_exclusive=1_000_000 + sample_offset + count,
        sample_id_start=sample_offset,
        sample_id_end_exclusive=sample_offset + count,
        seq=seq,
        window_id=window_id,
        ordered_attempt_receipts=_attempts(
            authority,
            count=count,
            sample_offset=sample_offset,
            birth_offset=birth_offset,
            duplicate_birth=duplicate_birth,
            failures=failures,
        ),
    )


def test_launch_is_exact_and_code_pinned_with_arm_scheduler_identity():
    launch = _launch()
    assert set(launch) == {
        "schema_version",
        "kind",
        "authority_contract_sha256",
        "curriculum_contract_sha256",
        "profile_order",
        "arm_catalog_sha256",
        "scheduler_contract_sha256",
        "sampler_sha256",
        "solver_sha256",
        "policy_contract_sha256",
        "attempt_source_contract_sha256",
        "attempt_source_path",
        "attempt_source_sha256",
    }
    assert launch["arm_catalog_sha256"] == C.ARM_CATALOG_SHA256
    assert launch["scheduler_contract_sha256"] == SCHEDULER
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="not code-pinned",
    ):
        E.FrozenEvaluatorAuthority.from_trusted_launch_receipt(launch)
    with pytest.raises(ValueError, match="catalog"):
        E.launch_receipt_document(
            curriculum_contract_sha256=CONTRACT,
            profile_order=(KEY,),
            arm_catalog_sha256="0" * 64,
            scheduler_contract_sha256=SCHEDULER,
            sampler_sha256=SAMPLER,
            solver_sha256=SOLVER,
            policy_contract_sha256=POLICY,
            attempt_source_contract_sha256=ATTEMPT_SOURCE,
            attempt_source_path=ATTEMPT_SOURCE_PATH,
            attempt_source_sha256=ATTEMPT_SOURCE_SHA,
        )


def test_diagnostic_authority_cannot_record_issue_or_bind():
    authority = E.FrozenEvaluatorAuthority(
        curriculum_contract_sha256=CONTRACT,
        profile_order=(KEY,),
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=SCHEDULER,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
    )
    with pytest.raises(E.FrozenEvaluationAuthorityError, match="cannot record"):
        authority.record_attempt(
            sample_receipt_sha256="1" * 64,
            birth_receipt_sha256="2" * 64,
            solver_admitted=True,
            installed=True,
            started=True,
            closed=True,
            terminal_outcome="legal_return",
        )
    with pytest.raises(E.FrozenEvaluationAuthorityError, match="cannot issue"):
        authority.issue_window(
            key=KEY,
            policy_checkpoint_sha256=CHECKPOINT,
            policy_generation=1,
            evidence_role="scheduler",
            domain_epoch=0,
            stratum="center",
            selected_arm_key="",
            selection_round=0,
            arm_levels=(0.0,) * len(C.ARM_KEYS),
            rho=0.0,
            seed_block_start=0,
            seed_block_end_exclusive=1,
            sample_id_start=0,
            sample_id_end_exclusive=1,
            seq=1,
            window_id="diagnostic",
            ordered_attempt_receipts=(),
        )
    with pytest.raises(E.FrozenEvaluationAuthorityError, match="not formal"):
        authority.assert_binding(
            curriculum_contract_sha256=CONTRACT,
            profile_order=(KEY,),
            arm_catalog_sha256=C.ARM_CATALOG_SHA256,
            scheduler_contract_sha256=SCHEDULER,
            sampler_sha256=SAMPLER,
            solver_sha256=SOLVER,
            policy_contract_sha256=POLICY,
        )


def test_window_derives_ledger_and_binds_ordered_sample_and_birth_roots():
    authority = _authority()
    capability = _issue(authority, failures=26)
    rows = authority.attempt_rows_many({KEY: capability})[KEY]
    assert len(rows) == 256
    assert capability.ledger.P == 256
    assert capability.ledger.L == 230
    assert capability.ledger.F == 26
    assert capability.unique_birth_count == 256
    assert capability.sample_receipt_root_sha256 == (
        E.ordered_sample_receipt_root(
            tuple(row["sample_receipt_sha256"] for row in rows)
        )
    )
    assert capability.birth_receipt_root_sha256 == (
        E.ordered_birth_receipt_root(
            tuple(row["birth_receipt_sha256"] for row in rows)
        )
    )
    with pytest.raises(TypeError, match="unexpected keyword.*ledger"):
        authority.issue_window(
            key=KEY,
            policy_checkpoint_sha256=CHECKPOINT,
            policy_generation=1,
            evidence_role="scheduler",
            domain_epoch=0,
            stratum="center",
            selected_arm_key="",
            selection_round=0,
            arm_levels=(0.0,) * len(C.ARM_KEYS),
            rho=0.0,
            seed_block_start=3_000,
            seed_block_end_exclusive=3_001,
            sample_id_start=3_000,
            sample_id_end_exclusive=3_001,
            seq=2,
            window_id="caller-ledger",
            ordered_attempt_receipts=(),
            ledger=C.BallOutcomeLedger(
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0
            ),
        )


def test_scheduler_allows_birth_multiplicity_but_frozen_windows_do_not():
    authority = _authority()
    scheduler = _issue(
        authority,
        role="scheduler",
        count=20,
        duplicate_birth=True,
    )
    assert scheduler.unique_birth_count == 1
    authority.consume_many({KEY: scheduler})

    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="one unique birth",
    ):
        _issue(
            authority,
            role="frozen_canary",
            count=256,
            sample_offset=20,
            birth_offset=20,
            seq=2,
            window_id="duplicate-frozen-birth",
            duplicate_birth=True,
        )


def test_completed_formal_window_round_trips_as_aggregate_only():
    authority = _authority()
    capability = _issue(authority)
    evidence = capability.evidence
    authority.consume_many(
        {KEY: capability},
        retain_formal_window_sha256=(),
    )
    state = deepcopy(authority.state_dict())
    assert state["pending"] == []
    assert len(state["consumed"]) == 1
    compact = state["consumed"][0]
    assert compact["attempt_storage"] == "formal_compact"
    assert compact["ordered_attempts"] is None
    assert compact["evidence"]["ledger"] == evidence.ledger.as_dict()
    assert (
        compact["evidence"]["sample_receipt_root_sha256"]
        == evidence.sample_receipt_root_sha256
    )
    assert (
        compact["evidence"]["birth_receipt_root_sha256"]
        == evidence.birth_receipt_root_sha256
    )

    restored = _authority()
    restored.load_state_dict(state)
    restored.assert_formal_retention(())
    assert restored.state_dict() == state


def test_same_action_receipts_ranges_and_frozen_births_are_disjoint():
    authority = _authority()
    first = _issue(authority)
    authority.consume_many({KEY: first})

    reused_samples = _attempts(
        authority,
        count=256,
        sample_offset=0,
        birth_offset=10_000,
    )
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="sample receipt",
    ):
        authority.issue_window(
            key=KEY,
            policy_checkpoint_sha256=CHECKPOINT,
            policy_generation=1,
            evidence_role="frozen_heldout",
            domain_epoch=0,
            stratum="center",
            selected_arm_key="",
            selection_round=0,
            arm_levels=(0.0,) * len(C.ARM_KEYS),
            rho=0.0,
            seed_block_start=2_000_000,
            seed_block_end_exclusive=2_000_256,
            sample_id_start=10_000,
            sample_id_end_exclusive=10_256,
            seq=2,
            window_id="reused-samples",
            ordered_attempt_receipts=reused_samples,
        )

    reused_births = tuple(
        authority.record_attempt(
            sample_receipt_sha256=_digest("sample", 20_000 + index),
            birth_receipt_sha256=_digest("birth", index),
            solver_admitted=True,
            installed=True,
            started=True,
            closed=True,
            terminal_outcome="legal_return",
        )
        for index in range(768)
    )
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="frozen windows reuse a birth",
    ):
        authority.issue_window(
            key=KEY,
            policy_checkpoint_sha256=CHECKPOINT,
            policy_generation=1,
            evidence_role="frozen_heldout",
            domain_epoch=0,
            stratum="center",
            selected_arm_key="",
            selection_round=0,
            arm_levels=(0.0,) * len(C.ARM_KEYS),
            rho=0.0,
            seed_block_start=3_000_000,
            seed_block_end_exclusive=3_000_768,
            sample_id_start=20_000,
            sample_id_end_exclusive=20_768,
            seq=2,
            window_id="reused-births",
            ordered_attempt_receipts=reused_births,
        )


def test_capability_is_single_use_foreign_safe_and_exact_resume_invalidates_old():
    authority = _authority()
    capability = _issue(authority)
    other = _authority()
    with pytest.raises(E.FrozenEvaluationAuthorityError, match="another"):
        other.inspect_many({KEY: capability})

    state = deepcopy(authority.state_dict())
    authority.load_state_dict(state)
    with pytest.raises(E.FrozenEvaluationAuthorityError, match="restored"):
        authority.inspect_many({KEY: capability})
    restored = authority.pending_capability(capability.capability_id)
    authority.consume_many({KEY: restored})
    with pytest.raises(E.FrozenEvaluationAuthorityError, match="stale"):
        authority.inspect_many({KEY: restored})


def test_policy_generation_is_monotonic_and_checkpoint_identity_is_unique():
    authority = _authority()
    first = _issue(authority)
    authority.consume_many({KEY: first})
    attempts = _attempts(
        authority,
        count=20,
        sample_offset=1_000,
        birth_offset=1_000,
    )
    kwargs = dict(
        key=KEY,
        evidence_role="scheduler",
        domain_epoch=0,
        stratum="center",
        selected_arm_key="",
        selection_round=0,
        arm_levels=(0.0,) * len(C.ARM_KEYS),
        rho=0.0,
        seed_block_start=2_001_000,
        seed_block_end_exclusive=2_001_020,
        sample_id_start=1_000,
        sample_id_end_exclusive=1_020,
        seq=2,
        window_id="checkpoint-alias",
        ordered_attempt_receipts=attempts,
    )
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="multiple checkpoints",
    ):
        authority.issue_window(
            policy_checkpoint_sha256="9" * 64,
            policy_generation=1,
            **kwargs,
        )

    advanced = authority.issue_window(
        policy_checkpoint_sha256="9" * 64,
        policy_generation=2,
        **{**kwargs, "window_id": "generation-2"},
    )
    authority.consume_many({KEY: advanced})
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="cannot regress",
    ):
        authority.issue_window(
            policy_checkpoint_sha256=CHECKPOINT,
            policy_generation=1,
            **{
                **kwargs,
                "sample_id_start": 2_000,
                "sample_id_end_exclusive": 2_020,
                "seed_block_start": 2_002_000,
                "seed_block_end_exclusive": 2_002_020,
                "seq": 3,
                "window_id": "regressed",
                "ordered_attempt_receipts": _attempts(
                    authority,
                    count=20,
                    sample_offset=2_000,
                    birth_offset=2_000,
                ),
            },
        )


def test_rehashed_attempt_tamper_and_legacy_state_fail_closed():
    authority = _authority()
    _issue(authority)
    state = deepcopy(authority.state_dict())
    state["pending"][0]["ordered_attempts"][0][
        "birth_receipt_sha256"
    ] = "9" * 64
    unsigned = dict(state)
    unsigned.pop("state_sha256")
    state["state_sha256"] = E._canonical_sha256(unsigned)
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="sample/birth receipt evidence mismatch",
    ):
        authority.load_state_dict(state)

    legacy = deepcopy(authority.state_dict())
    legacy["schema_version"] = 1
    unsigned = dict(legacy)
    unsigned.pop("state_sha256")
    legacy["state_sha256"] = E._canonical_sha256(unsigned)
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="unsupported",
    ):
        authority.load_state_dict(legacy)
