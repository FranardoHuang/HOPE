from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

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


def _load(name, filename=None):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        MDP / (filename or f"{name}.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = _load("action_ball_curriculum")
E = _load("action_ball_evaluation")
S = _load("action_ball_sampling_for_curriculum", "action_ball_sampling.py")

CONTRACT = "a" * 64
SAMPLER = "b" * 64
SOLVER = "c" * 64
POLICY = "d" * 64
CHECKPOINT = "e" * 64
ATTEMPT_SOURCE = "8" * 64
ATTEMPT_SOURCE_SHA = "7" * 64


def _key(action_uid=1, mobility="move", profile_char="f"):
    return C.ActionProfileKey(
        action_uid,
        profile_char * 64,
        mobility,
    )


def _system(keys=None, *, scheduler_config=None, config=None):
    keys = (_key(),) if keys is None else tuple(keys)
    scheduler_config = scheduler_config or C.ArmSchedulerConfig()
    launch = E.launch_receipt_document(
        curriculum_contract_sha256=CONTRACT,
        profile_order=keys,
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=scheduler_config.contract_sha256,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        attempt_source_contract_sha256=ATTEMPT_SOURCE,
        attempt_source_path="tests/frozen_attempt_source.py",
        attempt_source_sha256=ATTEMPT_SOURCE_SHA,
    )
    launch_sha = E._canonical_sha256(launch)
    E.TRUSTED_FROZEN_EVALUATOR_LAUNCH_RECEIPT_SHA256 = frozenset(
        set(E.TRUSTED_FROZEN_EVALUATOR_LAUNCH_RECEIPT_SHA256)
        | {launch_sha}
    )
    authority = E.FrozenEvaluatorAuthority.from_trusted_launch_receipt(
        launch
    )
    curriculum = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=keys,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=config or C.BallCurriculumConfig(),
        scheduler_config=scheduler_config,
        evaluator_authority=authority,
    )
    return curriculum, authority


def _digest(namespace, action_uid, index):
    return hashlib.sha256(
        f"{namespace}:{action_uid}:{index}".encode("ascii")
    ).hexdigest()


class EvidenceFactory:
    def __init__(self):
        self.seq = 0
        self.cursor = {}
        self.birth_cursor = {}

    def issue(
        self,
        curriculum,
        authority,
        key,
        *,
        role,
        count,
        domain=None,
        stratum=None,
        failures=0,
        solver_rejects=0,
        table_hits=0,
        other_unsafe=0,
        duplicate_birth=False,
        new_band=0,
    ):
        if domain is None:
            if role == "scheduler":
                domains = curriculum.scheduler_domains(key)
                domain = next(
                    item
                    for item in domains
                    if stratum is None or item.stratum == stratum
                )
            else:
                domain = curriculum.selected_formal_domain(key)
                assert domain is not None
        uid = key.action_uid
        sample_start = self.cursor.get(uid, 0)
        birth_start = self.birth_cursor.get(uid, 0)
        attempts = []
        safe_cursor = 0
        for index in range(count):
            admitted = index >= solver_rejects
            terminal = None
            if admitted:
                if safe_cursor < table_hits:
                    terminal = "table_hit"
                elif safe_cursor < table_hits + other_unsafe:
                    terminal = "fall"
                elif safe_cursor < table_hits + other_unsafe + failures:
                    terminal = "safe_nonreturn"
                else:
                    terminal = "legal_return"
                safe_cursor += 1
            attempts.append(
                authority.record_attempt(
                    sample_receipt_sha256=_digest(
                        "sample",
                        uid,
                        sample_start + index,
                    ),
                    birth_receipt_sha256=_digest(
                        "birth",
                        uid,
                        (
                            birth_start
                            if duplicate_birth
                            else birth_start + index
                        ),
                    ),
                    solver_admitted=admitted,
                    installed=admitted,
                    started=admitted,
                    closed=admitted,
                    terminal_outcome=terminal,
                    infrastructure_invalid=False,
                    in_new_band=index < new_band,
                )
            )
        self.seq += 1
        capability = authority.issue_window(
            key=key,
            policy_checkpoint_sha256=CHECKPOINT,
            policy_generation=1,
            evidence_role=role,
            domain_epoch=domain.domain_epoch,
            stratum=domain.stratum,
            selected_arm_key=domain.selected_arm_key,
            selection_round=domain.selection_round,
            arm_levels=domain.arm_levels,
            rho=domain.rho,
            seed_block_start=10_000_000 + sample_start,
            seed_block_end_exclusive=10_000_000 + sample_start + count,
            sample_id_start=sample_start,
            sample_id_end_exclusive=sample_start + count,
            seq=self.seq,
            window_id=f"{uid}:{self.seq}:{role}:{domain.stratum}",
            ordered_attempt_receipts=tuple(attempts),
        )
        self.cursor[uid] = sample_start + count
        self.birth_cursor[uid] = birth_start + (
            1 if duplicate_birth else count
        )
        return capability


def _ring_len(curriculum, key, arm):
    progress = curriculum._progress[key]
    return len(
        curriculum._new_band_ring_rows(progress, C.ARM_KEYS.index(arm))
    )


def _ring_failures(curriculum, key, arm):
    progress = curriculum._progress[key]
    rows = curriculum._new_band_ring_rows(
        progress, C.ARM_KEYS.index(arm)
    )
    return sum(
        1
        for row in rows
        if row["terminal_outcome"] == "safe_nonreturn"
    )


def _fill_ring(curriculum, authority, factory, key, *, ring_failures=0):
    """Feed scheduler windows until the selected arm's ring is full with the
    requested failure count.

    Each window carries exactly ring-size new-band safe-closed rows, and the
    ring keeps only the newest ring-size rows, so one window per arm decides
    that arm's ring content; the loop ends when the currently selected arm
    matches the request.
    """

    ring_size = curriculum.scheduler_config.new_band_ring_size
    for _ in range(6 * len(C.ARM_KEYS)):
        arm = curriculum.selected_arm(key)
        assert arm
        if (
            _ring_len(curriculum, key, arm) >= ring_size
            and _ring_failures(curriculum, key, arm) == ring_failures
        ):
            return arm
        capability = factory.issue(
            curriculum,
            authority,
            key,
            role="scheduler",
            count=ring_size,
            stratum=f"marginal:{arm}",
            failures=ring_failures,
            new_band=ring_size,
        )
        curriculum.observe_scheduler({key: capability})
    raise AssertionError("selected arm ring never filled")


def _certify(
    curriculum,
    authority,
    factory,
    key,
    *,
    failures=0,
    table_hits=0,
):
    domain = curriculum.selected_formal_domain(key)
    assert domain is not None
    canary = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_canary",
        count=256,
        domain=domain,
        failures=min(failures, 25),
        table_hits=table_hits,
    )
    canary_decision = curriculum.update_selected({key: canary})[0]
    heldout = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_heldout",
        count=768,
        domain=domain,
        failures=failures,
        table_hits=table_hits,
    )
    heldout_decision = curriculum.update_selected({key: heldout})[0]
    return canary_decision, heldout_decision


def _completed_no_move_state():
    key = _key(mobility="no_move")
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    while curriculum.phase(key) == "marginal":
        _fill_ring(
            curriculum,
            authority,
            factory,
            key,
            ring_failures=3,
        )
        _certify(
            curriculum,
            authority,
            factory,
            key,
            failures=77,
        )
    _certify(
        curriculum,
        authority,
        factory,
        key,
        failures=77,
    )
    assert curriculum.phase(key) == "steady"
    return curriculum.state_dict()


def _remap_evidence(document, *, key, seq, window_id):
    return C.BallDomainEvidence.create(
        key=key,
        arm_catalog_sha256=document["arm_catalog_sha256"],
        scheduler_contract_sha256=document[
            "scheduler_contract_sha256"
        ],
        sampler_sha256=document["sampler_sha256"],
        solver_sha256=document["solver_sha256"],
        policy_contract_sha256=document["policy_contract_sha256"],
        policy_checkpoint_sha256=document[
            "policy_checkpoint_sha256"
        ],
        policy_generation=document["policy_generation"],
        evidence_role=document["evidence_role"],
        domain_epoch=document["domain_epoch"],
        stratum=document["stratum"],
        selected_arm_key=document["selected_arm_key"],
        selection_round=document["selection_round"],
        arm_levels=tuple(document["arm_levels"]),
        rho=document["rho"],
        seed_block_start=document["seed_block_start"],
        seed_block_end_exclusive=document["seed_block_end_exclusive"],
        sample_id_start=document["sample_id_start"],
        sample_id_end_exclusive=document["sample_id_end_exclusive"],
        sample_receipt_root_sha256=document[
            "sample_receipt_root_sha256"
        ],
        unique_birth_count=document["unique_birth_count"],
        birth_receipt_root_sha256=document[
            "birth_receipt_root_sha256"
        ],
        seq=seq,
        window_id=window_id,
        ledger=C.BallOutcomeLedger(**document["ledger"]),
    )


def _synthesize_completed_state(template_state, keys):
    curriculum, _ = _system(keys)
    state = deepcopy(curriculum.state_dict())
    template_progress = template_state["progress"][0]
    authority_state = state["evaluator_authority_state"]
    state_owner = authority_state["state_owner_sha256"]
    consumed = []
    progress_rows = []
    global_seq = 0

    template_events = sorted(
        [
            ("formal", receipt)
            for receipt in template_progress["formal_receipts"]
        ]
        + [
            ("scheduler", receipt)
            for receipt in template_progress["scheduler_receipts"]
        ],
        key=lambda item: item[1]["evidence"]["seq"],
    )
    for key in keys:
        progress = deepcopy(template_progress)
        progress["key"] = key.as_dict()
        progress["formal_receipts"] = []
        progress["scheduler_receipts"] = []
        progress["pending_canary_window_sha256"] = None
        event_chain = "0" * 64
        last_certified = None
        for event_kind, template_receipt in template_events:
            global_seq += 1
            template_evidence = template_receipt["evidence"]
            window_id = (
                f"compact-n93:{key.action_uid}:{global_seq}:"
                f"{template_evidence['evidence_role']}:"
                f"{template_evidence['stratum']}"
            )
            evidence = _remap_evidence(
                template_evidence,
                key=key,
                seq=global_seq,
                window_id=window_id,
            )
            evidence_document = evidence._hash_document()
            if event_kind == "formal":
                progress["formal_receipts"].append(
                    {
                        "evidence": deepcopy(evidence_document),
                        "window_sha256": evidence.window_sha256,
                        "certified": template_receipt["certified"],
                    }
                )
                if template_receipt["certified"]:
                    last_certified = curriculum._certificate(evidence)
                attempt_storage = "formal_compact"
                ordered_attempts = None
            else:
                attempts = deepcopy(template_receipt["attempts"])
                progress["scheduler_receipts"].append(
                    {
                        "evidence": deepcopy(evidence_document),
                        "window_sha256": evidence.window_sha256,
                        "attempts": attempts,
                    }
                )
                attempt_storage = "full"
                ordered_attempts = deepcopy(attempts)
            event_chain = hashlib.sha256(
                (event_chain + evidence.window_sha256).encode("ascii")
            ).hexdigest()
            capability_id = E._canonical_sha256(
                {
                    "state_owner_sha256": state_owner,
                    "window_sha256": evidence.window_sha256,
                    "sample_receipt_root_sha256": (
                        evidence.sample_receipt_root_sha256
                    ),
                    "birth_receipt_root_sha256": (
                        evidence.birth_receipt_root_sha256
                    ),
                    "unique_birth_count": evidence.unique_birth_count,
                }
            )
            consumed.append(
                {
                    "capability_id": capability_id,
                    "evidence": deepcopy(evidence_document),
                    "window_sha256": evidence.window_sha256,
                    "attempt_storage": attempt_storage,
                    "ordered_attempts": ordered_attempts,
                }
            )
        progress["event_hash_chain_sha256"] = event_chain
        progress["last_certified"] = last_certified
        progress_rows.append(progress)

    authority_chain = "0" * 64
    for window in consumed:
        authority_chain = hashlib.sha256(
            (
                authority_chain + window["capability_id"]
            ).encode("ascii")
        ).hexdigest()
    authority_state["pending"] = []
    authority_state["consumed"] = consumed
    authority_state["consumed_hash_chain_sha256"] = authority_chain
    unsigned_authority = dict(authority_state)
    unsigned_authority.pop("state_sha256")
    authority_state["state_sha256"] = E._canonical_sha256(
        unsigned_authority
    )
    state["progress"] = progress_rows
    state["evaluator_authority_state"] = authority_state
    state["evaluator_authority_state_sha256"] = C._canonical_sha256(
        authority_state
    )
    unsigned = dict(state)
    unsigned.pop("state_sha256")
    state["state_sha256"] = C._canonical_sha256(unsigned)
    return curriculum, state


def test_arm_catalog_exactly_matches_sampler_and_no_move_disables_four():
    assert len(C.ARM_KEYS) == 32
    assert C.ARM_KEYS == S.ARM_KEYS
    assert C.ARM_CATALOG_SHA256 == S.ARM_CATALOG_SHA256
    assert C.ARM_KEYS[:2] == (
        "time_to_contact_lower",
        "time_to_contact_upper",
    )
    assert {
        "incoming_direction_u_neg",
        "incoming_direction_u_pos",
        "incoming_direction_v_neg",
        "incoming_direction_v_pos",
        "spin_direction_u_neg",
        "spin_direction_u_pos",
        "spin_direction_v_neg",
        "spin_direction_v_pos",
    }.issubset(C.ARM_KEYS)
    move = _key(mobility="move")
    no_move = _key(mobility="no_move")
    assert len(move.enabled_arms) == 32
    assert len(no_move.enabled_arms) == 28
    assert set(C.BASE_TRAVEL_ARMS).isdisjoint(no_move.enabled_arms)


def test_manifest_config_stays_separate_from_code_frozen_scheduler_and_heldout():
    config = C.BallCurriculumConfig()
    assert set(config.as_dict()) == {
        "min_proposals",
        "min_safe_closed",
        "target_failure_rate",
        "failure_band_half_width",
        "min_solver_admit_rate",
        "min_install_rate",
        "min_start_rate",
        "min_close_rate",
        "max_other_unsafe_rate",
        "confidence_z",
        "max_center_failures",
    }
    assert config.min_proposals == 256
    assert config.heldout_min_proposals == 768
    assert config.heldout_min_safe_closed == 768
    with pytest.raises(ValueError, match="cannot be below 256"):
        C.BallCurriculumConfig(min_proposals=255)
    assert C.ArmSchedulerConfig().rolling_window == 100
    with pytest.raises(ValueError, match="fixed at 100"):
        C.ArmSchedulerConfig(rolling_window=99)
    scheduler = C.ArmSchedulerConfig()
    assert scheduler.new_band_ring_size == 30
    assert scheduler.as_dict()["new_band_ring_size"] == 30
    with pytest.raises(ValueError, match="fixed at 30"):
        C.ArmSchedulerConfig(new_band_ring_size=29)


def test_public_evidence_is_diagnostic_and_canary_alone_cannot_advance():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    capability = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_canary",
        count=256,
    )
    before = curriculum.state_dict()["progress"]
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="opaque capabilities",
    ):
        curriculum.update_selected({key: capability.evidence})
    assert curriculum.state_dict()["progress"] == before

    decision = curriculum.update_selected({key: capability})[0]
    assert decision.kind == "canary_pass"
    assert curriculum.phase(key) == "center"
    assert set(curriculum.frontiers(key).values()) == {0.0}


def test_only_in_flight_canary_retains_attempt_rows():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    domain = curriculum.selected_formal_domain(key)
    canary = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_canary",
        count=256,
        domain=domain,
    )
    curriculum.update_selected({key: canary})
    in_flight = curriculum.state_dict()["evaluator_authority_state"]
    assert len(in_flight["consumed"]) == 1
    assert in_flight["consumed"][0]["attempt_storage"] == "full"
    assert len(in_flight["consumed"][0]["ordered_attempts"]) == 256

    heldout = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_heldout",
        count=768,
        domain=domain,
    )
    curriculum.update_selected({key: heldout})
    completed = curriculum.state_dict()["evaluator_authority_state"]
    assert len(completed["consumed"]) == 2
    assert all(
        row["attempt_storage"] == "formal_compact"
        and row["ordered_attempts"] is None
        for row in completed["consumed"]
    )


def test_heldout_requires_matching_canary_and_fixed_256_768_floors():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    heldout = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_heldout",
        count=768,
    )
    with pytest.raises(ValueError, match="prior frozen canary"):
        curriculum.update_selected({key: heldout})

    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    too_small = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_canary",
        count=255,
    )
    with pytest.raises(ValueError, match="below 256"):
        curriculum.update_selected({key: too_small})

    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    domain = curriculum.selected_formal_domain(key)
    canary = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_canary",
        count=256,
        domain=domain,
    )
    curriculum.update_selected({key: canary})
    too_small_heldout = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_heldout",
        count=767,
        domain=domain,
    )
    with pytest.raises(ValueError, match="below 768"):
        curriculum.update_selected({key: too_small_heldout})


def test_center_then_signed_marginals_expand_independently():
    # Band [0.15, 0.45], ring 30, direct count: <=floor(4.5)=4 failures is
    # too easy (expand), 5..13 is in band (lock), >=14 is too hard.
    wide = C.BallCurriculumConfig(
        target_failure_rate=0.3,
        failure_band_half_width=0.15,
    )
    key = _key()
    curriculum, authority = _system((key,), config=wide)
    factory = EvidenceFactory()
    _, center = _certify(curriculum, authority, factory, key)
    assert center.kind == "center_pass"
    assert curriculum.phase(key) == "marginal"
    assert curriculum.selected_arm(key) == "time_to_contact_lower"

    lower_arm = _fill_ring(
        curriculum, authority, factory, key, ring_failures=0
    )
    canary, lower = _certify(curriculum, authority, factory, key)
    assert canary.kind == "canary_pass"
    assert lower.kind == "expand_marginal"
    assert curriculum.frontiers(key)[lower_arm] == 0.25

    upper_arm = _fill_ring(
        curriculum, authority, factory, key, ring_failures=9
    )
    assert upper_arm != lower_arm or (
        # An expanded arm re-probes at the next level with an empty ring.
        curriculum.frontiers(key)[upper_arm] == 0.25
    )
    _, upper = _certify(
        curriculum,
        authority,
        factory,
        key,
        failures=230,
    )
    assert upper.kind == "lock_marginal"
    assert curriculum.frontiers(key)[upper_arm] in (0.25, 0.5)


def test_marginal_promotion_requires_full_new_band_ring():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    assert curriculum.phase(key) == "marginal"
    arm = curriculum.selected_arm(key)
    before = curriculum.frontiers(key)

    # 29 < 30 ring rows: the heldout must hold the arm open, not bound it.
    capability = factory.issue(
        curriculum,
        authority,
        key,
        role="scheduler",
        count=30,
        stratum=f"marginal:{arm}",
        failures=0,
        new_band=29,
    )
    curriculum.observe_scheduler({key: capability})
    for _ in range(4 * len(C.ARM_KEYS)):
        if curriculum.selected_arm(key) == arm:
            break
        other = curriculum.selected_arm(key)
        other_window = factory.issue(
            curriculum,
            authority,
            key,
            role="scheduler",
            count=30,
            stratum=f"marginal:{other}",
            failures=0,
            new_band=0,
        )
        curriculum.observe_scheduler({key: other_window})
    assert curriculum.selected_arm(key) == arm
    assert _ring_len(curriculum, key, arm) == 29
    _, decision = _certify(curriculum, authority, factory, key)
    assert decision.kind == "new_band_ring_incomplete"
    assert curriculum.frontiers(key) == before
    progress = curriculum._progress[key]
    assert progress.arm_status[C.ARM_KEYS.index(arm)] == "probing"


def test_marginal_ring_failure_count_binds_and_uses_exactly_30_rows():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    arm = _fill_ring(
        curriculum, authority, factory, key, ring_failures=4
    )
    # Direct count on the default [0.075, 0.125] band with ring 30:
    # 4 >= floor(0.125*30)+1 = 4 is too hard, even though the heldout
    # window failure rate (77/768 ~ 10%) sits inside the band.
    before = curriculum.frontiers(key)
    _, decision = _certify(
        curriculum, authority, factory, key, failures=77
    )
    assert decision.kind == "bound_marginal"
    assert curriculum.frontiers(key) == before
    assert (
        curriculum._progress[key].arm_status[C.ARM_KEYS.index(arm)]
        == "decided"
    )


def test_marginal_ring_direct_count_thresholds_at_default_band():
    # f10 band [0.075, 0.125], ring 30: 2 -> expand, 3 -> lock (Franco
    # 2026-07-28 second ruling: direct counting, no Wilson).
    for failures, expected_kind in ((2, "expand_marginal"), (3, "lock_marginal")):
        key = _key()
        curriculum, authority = _system((key,))
        factory = EvidenceFactory()
        _certify(curriculum, authority, factory, key)
        arm = _fill_ring(
            curriculum, authority, factory, key, ring_failures=failures
        )
        _, decision = _certify(
            curriculum, authority, factory, key, failures=77
        )
        assert decision.kind == expected_kind, (failures, decision.kind)
        assert curriculum.frontiers(key)[arm] == 0.25


def test_no_move_never_schedules_or_certifies_base_travel():
    key = _key(mobility="no_move")
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    assert all(
        domain.selected_arm_key not in C.BASE_TRAVEL_ARMS
        for domain in curriculum.scheduler_domains(key)
    )
    state = curriculum.state_dict()["progress"][0]
    for arm in C.BASE_TRAVEL_ARMS:
        index = C.ARM_KEYS.index(arm)
        assert state["arm_status"][index] == "disabled"
        assert state["arm_frontier_indices"][index] == 0


def test_all_enabled_marginals_reach_joint_then_steady_without_shadow_axis():
    key = _key(mobility="no_move")
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    visited = []
    while curriculum.phase(key) == "marginal":
        arm = _fill_ring(
            curriculum,
            authority,
            factory,
            key,
            ring_failures=3,
        )
        assert arm == curriculum.selected_arm(key)
        assert arm and arm not in visited
        visited.append(arm)
        _certify(
            curriculum,
            authority,
            factory,
            key,
            failures=77,
        )
    assert sorted(visited) == sorted(key.enabled_arms)
    assert curriculum.phase(key) == "joint"
    joint = curriculum.selected_formal_domain(key)
    assert joint is not None
    assert joint.stratum == "joint"
    assert joint.rho == 0.25
    assert all(
        level == (0.0 if arm in C.BASE_TRAVEL_ARMS else 0.0625)
        for arm, level in zip(C.ARM_KEYS, joint.arm_levels)
    )
    _, decision = _certify(
        curriculum,
        authority,
        factory,
        key,
        failures=77,
    )
    assert decision.kind == "enter_steady"
    assert curriculum.phase(key) == "steady"
    assert curriculum.joint_rho(key) == 0.25


def test_scheduler_keeps_stage_rejects_out_of_policy_failure_and_rolls_latest_100():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    arm = C.ARM_KEYS[0]
    stratum = f"marginal:{arm}"

    rejected = factory.issue(
        curriculum,
        authority,
        key,
        role="scheduler",
        count=100,
        stratum=stratum,
        solver_rejects=10,
        failures=40,
    )
    curriculum.observe_scheduler({key: rejected})
    progress = curriculum._progress[key]
    ledger = curriculum._recent_ledger(progress, 0)
    assert (ledger.P, ledger.A, ledger.F) == (100, 90, 40)
    assert not curriculum._scheduler_eligible(ledger)

    clean = factory.issue(
        curriculum,
        authority,
        key,
        role="scheduler",
        count=100,
        stratum=stratum,
        failures=0,
    )
    curriculum.observe_scheduler({key: clean})
    ledger = curriculum._recent_ledger(curriculum._progress[key], 0)
    assert (ledger.P, ledger.A, ledger.F) == (100, 100, 0)
    assert curriculum._scheduler_eligible(ledger)


def test_scheduler_score_uses_failure_ucb_after_hard_eligibility_gates():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)

    for index, arm in enumerate(C.ARM_KEYS):
        capability = factory.issue(
            curriculum,
            authority,
            key,
            role="scheduler",
            count=100,
            stratum=f"marginal:{arm}",
            failures=(30 if index == 0 else 0 if index == 1 else 10),
        )
        curriculum.observe_scheduler({key: capability})

    progress = curriculum._progress[key]
    progress.selection_round = 32
    progress.last_selected_round = (32,) * len(C.ARM_KEYS)
    curriculum._reselect_arm(key, progress)
    assert progress.selection_round == 33
    assert progress.selected_arm_key == C.ARM_KEYS[1]
    assert (
        C.wilson_interval(
            curriculum._recent_ledger(progress, 1).F,
            curriculum._recent_ledger(progress, 1).safe_closed,
            z=curriculum.config.confidence_z,
        ).upper
        < C.wilson_interval(
            curriculum._recent_ledger(progress, 0).F,
            curriculum._recent_ledger(progress, 0).safe_closed,
            z=curriculum.config.confidence_z,
        ).upper
    )


def test_forced_exploration_is_deterministic_and_bounds_starvation():
    key = _key()
    curriculum, _ = _system((key,))
    progress = curriculum._progress[key]
    progress.phase = "marginal"
    progress.arm_status = ("probing",) * len(C.ARM_KEYS)
    progress.selection_round = 4
    rounds = [4] * len(C.ARM_KEYS)
    rounds[7] = 0
    progress.last_selected_round = tuple(rounds)

    # All arms are under-sampled equally.  Oldest wins, catalog index breaks ties.
    curriculum._reselect_arm(key, progress)
    assert progress.selection_round == 5
    assert progress.selected_arm_key == C.ARM_KEYS[7]

    progress.selection_round = (
        curriculum.scheduler_config.max_gap_factor * len(C.ARM_KEYS)
    )
    progress.last_selected_round = tuple(
        0 if index in (3, 9) else progress.selection_round
        for index in range(len(C.ARM_KEYS))
    )
    curriculum._reselect_arm(key, progress)
    assert progress.selected_arm_key == C.ARM_KEYS[3]


def test_table_hit_is_zero_tolerance_in_frozen_heldout():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    before = curriculum.frontiers(key)
    domain = curriculum.selected_formal_domain(key)
    canary = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_canary",
        count=256,
        domain=domain,
    )
    assert curriculum.update_selected({key: canary})[0].kind == "canary_pass"
    heldout = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_heldout",
        count=769,
        domain=domain,
        table_hits=1,
    )
    decision = curriculum.update_selected({key: heldout})[0]
    assert decision.kind == "bound_marginal"
    assert "table_hit_zero_tolerance" in decision.blockers
    assert curriculum.frontiers(key) == before


def test_exact_resume_restores_pending_heldout_and_invalidates_old_capability():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    domain = curriculum.selected_formal_domain(key)
    canary = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_canary",
        count=256,
        domain=domain,
    )
    curriculum.update_selected({key: canary})
    heldout = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_heldout",
        count=768,
        domain=domain,
    )
    state = deepcopy(curriculum.state_dict())

    resumed, resumed_authority = _system((key,))
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state
    with pytest.raises(E.FrozenEvaluationAuthorityError, match="another"):
        resumed.update_selected({key: heldout})
    restored = resumed_authority.pending_capability(
        heldout.capability_id
    )
    assert resumed.update_selected({key: restored})[0].kind == "center_pass"


def test_rehashed_replay_forgery_and_legacy_schema_fail_closed():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    scheduler = factory.issue(
        curriculum,
        authority,
        key,
        role="scheduler",
        count=20,
        stratum=f"marginal:{C.ARM_KEYS[0]}",
    )
    curriculum.observe_scheduler({key: scheduler})
    state = deepcopy(curriculum.state_dict())
    state["progress"][0]["selected_arm_key"] = C.ARM_KEYS[-1]
    unsigned = dict(state)
    unsigned.pop("state_sha256")
    state["state_sha256"] = C._canonical_sha256(unsigned)

    resumed, _ = _system((key,))
    with pytest.raises(ValueError, match="deterministic replay"):
        resumed.load_state_dict(state)

    legacy = deepcopy(curriculum.state_dict())
    legacy["schema_version"] = 4
    unsigned = dict(legacy)
    unsigned.pop("state_sha256")
    legacy["state_sha256"] = C._canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="legacy seven-axis"):
        resumed.load_state_dict(legacy)


def test_ninety_three_profiles_round_trip_without_fixed_action_count():
    keys = tuple(
        _key(
            action_uid=index + 1,
            mobility="move",
            profile_char="abcdef"[index % 6],
        )
        for index in range(93)
    )
    curriculum, _ = _system(keys)
    state = curriculum.state_dict()
    assert len(state["progress"]) == 93
    resumed, _ = _system(keys)
    resumed.load_state_dict(deepcopy(state))
    assert resumed.state_dict() == state


def test_compact_full_course_n1_and_n93_size_and_latency_gates():
    template_state = _completed_no_move_state()
    template_raw = json.dumps(
        template_state,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    consumed = template_state["evaluator_authority_state"]["consumed"]
    formal_rows = [
        row
        for row in consumed
        if row["attempt_storage"] == "formal_compact"
    ]
    scheduler_rows = [
        row for row in consumed if row["attempt_storage"] == "full"
    ]
    assert len(formal_rows) == 60
    assert all(
        row["ordered_attempts"] is None for row in formal_rows
    )
    # Franco 2026-07-28 new-band ring: scheduler windows retain their exact
    # attempt rows (the ring state), one full 30-row window per enabled arm.
    assert len(scheduler_rows) == 28
    assert all(
        len(row["ordered_attempts"]) == 30 for row in scheduler_rows
    )
    assert len(template_raw) < 2 * 1024 * 1024
    projected_n93_bytes = len(template_raw) * 93
    assert projected_n93_bytes < 128 * 1024 * 1024

    keys = tuple(
        _key(
            action_uid=index + 1,
            mobility="no_move",
            profile_char="abcdef"[index % 6],
        )
        for index in range(93)
    )
    resumed, state = _synthesize_completed_state(template_state, keys)
    encode_started = time.perf_counter()
    raw = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    encode_seconds = time.perf_counter() - encode_started
    assert len(raw) < 128 * 1024 * 1024
    assert len(raw) <= projected_n93_bytes
    assert encode_seconds < 5.0

    load_started = time.perf_counter()
    resumed.load_state_dict(deepcopy(state))
    load_seconds = time.perf_counter() - load_started
    save_started = time.perf_counter()
    round_trip = resumed.state_dict()
    save_seconds = time.perf_counter() - save_started
    assert round_trip == state
    # The ring course replays 93 x 28 scheduler windows (30 attempt rows
    # each) on load; measured ~13-16 s on a busy pod, so the gate carries
    # honest headroom instead of flaking on machine load.
    assert load_seconds < 30.0
    assert save_seconds < 5.0


def test_compact_root_forgery_and_deleted_event_fail_closed():
    state = _completed_no_move_state()
    key = _key(mobility="no_move")

    forged = deepcopy(state)
    authority_state = forged["evaluator_authority_state"]
    authority_state["consumed"][0]["evidence"][
        "sample_receipt_root_sha256"
    ] = "9" * 64
    unsigned_authority = dict(authority_state)
    unsigned_authority.pop("state_sha256")
    authority_state["state_sha256"] = E._canonical_sha256(
        unsigned_authority
    )
    forged["evaluator_authority_state_sha256"] = C._canonical_sha256(
        authority_state
    )
    unsigned = dict(forged)
    unsigned.pop("state_sha256")
    forged["state_sha256"] = C._canonical_sha256(unsigned)
    resumed, _ = _system((key,))
    with pytest.raises(
        (ValueError, E.FrozenEvaluationAuthorityError),
        match="hash mismatch",
    ):
        resumed.load_state_dict(forged)

    deleted = deepcopy(state)
    deleted_progress = deleted["progress"][0]
    deleted_progress["formal_receipts"].pop()
    event_chain = "0" * 64
    for receipt in deleted_progress["formal_receipts"]:
        event_chain = hashlib.sha256(
            (
                event_chain + receipt["window_sha256"]
            ).encode("ascii")
        ).hexdigest()
    deleted_progress["event_hash_chain_sha256"] = event_chain
    deleted_authority = deleted["evaluator_authority_state"]
    deleted_authority["consumed"].pop()
    authority_chain = "0" * 64
    for window in deleted_authority["consumed"]:
        authority_chain = hashlib.sha256(
            (
                authority_chain + window["capability_id"]
            ).encode("ascii")
        ).hexdigest()
    deleted_authority["consumed_hash_chain_sha256"] = authority_chain
    unsigned_authority = dict(deleted_authority)
    unsigned_authority.pop("state_sha256")
    deleted_authority["state_sha256"] = E._canonical_sha256(
        unsigned_authority
    )
    deleted["evaluator_authority_state_sha256"] = C._canonical_sha256(
        deleted_authority
    )
    unsigned = dict(deleted)
    unsigned.pop("state_sha256")
    deleted["state_sha256"] = C._canonical_sha256(unsigned)
    resumed, _ = _system((key,))
    with pytest.raises(ValueError, match="deterministic replay"):
        resumed.load_state_dict(deleted)
