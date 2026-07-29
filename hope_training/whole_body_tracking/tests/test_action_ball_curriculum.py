from copy import deepcopy
import hashlib
import importlib.util
import inspect
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
DRAIN_SOURCE_CONTRACT = "1" * 64
DRAIN_SOURCE_SHA = "2" * 64
BROKER_CONTRACT = "3" * 64
ATTEMPT_POOL_CONTRACT = "4" * 64
TASK_POOL_CONTRACT = "5" * 64
ENV_RESET_CONTRACT = "6" * 64
DRAIN_SOURCE_PATH = "tests/exact_drain_reset_source.py"


class ExactDrainResetSource:
    """Test double for a code-pinned coordinator that owns a live fence."""

    def __init__(self, env_count=4, consumed=()):
        self.env_count = env_count
        self.active_attempts = 0
        self.reserved_attempts = 0
        self.active_births = 0
        self.pending_task_receipts = 0
        self.broker_reset_generation = 1
        self.attempt_pool_reset_generation = 1
        self.task_receipt_pool_reset_generation = 1
        self.env_reset_generation = 1
        self.reset_count = env_count
        self.reset_participant_ids = list(range(env_count))
        self.broker_state_root_sha256 = "9" * 64
        self.attempt_pool_state_root_sha256 = "a" * 64
        self.task_receipt_pool_state_root_sha256 = "b" * 64
        self.env_reset_state_root_sha256 = "c" * 64
        self._fence = None
        self.consumed = [deepcopy(item) for item in consumed]

    def binding_document(self):
        return {
            "runtime_source_contract_sha256": DRAIN_SOURCE_CONTRACT,
            "runtime_source_path": DRAIN_SOURCE_PATH,
            "runtime_source_sha256": DRAIN_SOURCE_SHA,
            "broker_contract_sha256": BROKER_CONTRACT,
            "attempt_pool_contract_sha256": ATTEMPT_POOL_CONTRACT,
            "task_receipt_pool_contract_sha256": TASK_POOL_CONTRACT,
            "env_reset_contract_sha256": ENV_RESET_CONTRACT,
        }

    def _bitmap(self):
        return C._canonical_sha256(
            {
                "schema_version": 1,
                "reset_generation": self.env_reset_generation,
                "env_count": self.env_count,
                "reset_participant_ids": self.reset_participant_ids,
            }
        )

    def capture_drain_reset(self, request):
        if self._fence is not None:
            raise RuntimeError("drain/reset fence already held")
        fence_id = C._canonical_sha256(
            {
                "request": request,
                "generation": self.env_reset_generation,
                "capture_index": len(self.consumed),
            }
        )
        snapshot = {
            "schema_version": 1,
            "kind": "action_ball_global_pre_reset_snapshot",
            "request_sha256": C._canonical_sha256(request),
            "old_global_state_root_sha256": request[
                "old_global_state_root_sha256"
            ],
            "target_global_state_root_sha256": request[
                "target_global_state_root_sha256"
            ],
            "published_domain_set_root_sha256": request[
                "published_domain_set_root_sha256"
            ],
            "release_set_root_sha256": request[
                "release_set_root_sha256"
            ],
            "evidence_set_root_sha256": request[
                "evidence_set_root_sha256"
            ],
            "policy_checkpoint_sha256": request[
                "policy_checkpoint_sha256"
            ],
            "policy_generation": request["policy_generation"],
            "broker_reset_generation": self.broker_reset_generation,
            "attempt_pool_reset_generation": (
                self.attempt_pool_reset_generation
            ),
            "task_receipt_pool_reset_generation": (
                self.task_receipt_pool_reset_generation
            ),
            "env_reset_generation": self.env_reset_generation,
            "active_attempts": self.active_attempts,
            "reserved_attempts": self.reserved_attempts,
            "active_births": self.active_births,
            "pending_task_receipts": self.pending_task_receipts,
            "reset_count": self.reset_count,
            "env_count": self.env_count,
            "reset_participant_ids": list(self.reset_participant_ids),
            "reset_bitmap_sha256": self._bitmap(),
            "fence_id_sha256": fence_id,
            "broker_state_root_sha256": (
                self.broker_state_root_sha256
            ),
            "attempt_pool_state_root_sha256": (
                self.attempt_pool_state_root_sha256
            ),
            "task_receipt_pool_state_root_sha256": (
                self.task_receipt_pool_state_root_sha256
            ),
            "env_reset_state_root_sha256": (
                self.env_reset_state_root_sha256
            ),
        }
        if (
            self.active_attempts == 0
            and self.reserved_attempts == 0
            and self.active_births == 0
            and self.pending_task_receipts == 0
            and len(
                {
                    self.broker_reset_generation,
                    self.attempt_pool_reset_generation,
                    self.task_receipt_pool_reset_generation,
                    self.env_reset_generation,
                }
            )
            == 1
            and self.reset_count == self.env_count
            and self.reset_participant_ids
            == list(range(self.env_count))
        ):
            self._fence = {
                "fence_id_sha256": fence_id,
                "snapshot": snapshot,
            }
        return snapshot

    def _assert_fence(self, token):
        if self._fence is None:
            raise RuntimeError("drain/reset fence is absent")
        if (
            token["fence_id_sha256"]
            != self._fence["fence_id_sha256"]
        ):
            raise RuntimeError("drain/reset fence identity changed")
        snapshot = self._fence["snapshot"]
        for field in (
            "broker_reset_generation",
            "attempt_pool_reset_generation",
            "task_receipt_pool_reset_generation",
            "env_reset_generation",
            "active_attempts",
            "reserved_attempts",
            "active_births",
            "pending_task_receipts",
            "reset_count",
            "env_count",
            "reset_participant_ids",
            "reset_bitmap_sha256",
            "broker_state_root_sha256",
            "attempt_pool_state_root_sha256",
            "task_receipt_pool_state_root_sha256",
            "env_reset_state_root_sha256",
        ):
            current = (
                self._bitmap()
                if field == "reset_bitmap_sha256"
                else list(self.reset_participant_ids)
                if field == "reset_participant_ids"
                else getattr(self, field)
            )
            if snapshot[field] != current or token[field] != current:
                raise RuntimeError(f"drain/reset {field} changed under fence")

    def commit_drain_reset(self, token, publish_noexcept):
        self._assert_fence(token)
        publish_noexcept()
        self.consumed.append(token)
        self._fence = None
        return {
            "schema_version": 1,
            "kind": "action_ball_drain_reset_commit",
            "token_sha256": token["token_sha256"],
            "fence_id_sha256": token["fence_id_sha256"],
            "published": True,
        }

    def abort_drain_reset(self, token):
        self._assert_fence(token)
        self._fence = None
        return {
            "schema_version": 1,
            "kind": "action_ball_drain_reset_abort",
            "token_sha256": token["token_sha256"],
            "fence_id_sha256": token["fence_id_sha256"],
            "aborted": True,
        }

    def begin_birth(self):
        if self._fence is not None:
            raise RuntimeError("new work blocked by drain/reset fence")
        self.active_births += 1

    def assert_consumed_drain_reset(self, ordered_token_documents):
        return list(ordered_token_documents) == self.consumed


def _drain_launch(keys, scheduler_config=None):
    scheduler = scheduler_config or C.ArmSchedulerConfig()
    return C.drain_reset_launch_receipt_document(
        curriculum_contract_sha256=CONTRACT,
        profile_order=keys,
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=scheduler.contract_sha256,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        runtime_source_contract_sha256=DRAIN_SOURCE_CONTRACT,
        runtime_source_path=DRAIN_SOURCE_PATH,
        runtime_source_sha256=DRAIN_SOURCE_SHA,
        broker_contract_sha256=BROKER_CONTRACT,
        attempt_pool_contract_sha256=ATTEMPT_POOL_CONTRACT,
        task_receipt_pool_contract_sha256=TASK_POOL_CONTRACT,
        env_reset_contract_sha256=ENV_RESET_CONTRACT,
    )


def _drain_authority(keys, source=None, scheduler_config=None):
    source = source or ExactDrainResetSource()
    launch = _drain_launch(keys, scheduler_config)
    launch_sha = C._canonical_sha256(launch)
    C.TRUSTED_DRAIN_RESET_LAUNCH_RECEIPT_SHA256 = frozenset(
        set(C.TRUSTED_DRAIN_RESET_LAUNCH_RECEIPT_SHA256)
        | {launch_sha}
    )
    return (
        C.DrainResetAuthority.from_trusted_launch_receipt(
            launch, runtime_source=source
        ),
        source,
    )


def _drain_consumed(state):
    authority_state = state["drain_reset_authority_state"]
    assert authority_state is not None
    return deepcopy(authority_state["consumed"])


def _v4_fixture():
    name = "action_ball_frozen_evaluator_v4_fixture"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name(
        "test_action_ball_frozen_evaluator_v4.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _v4_system_and_release(source=None):
    fixture = _v4_fixture()
    key = fixture.KEY
    authority = fixture._authority(source)
    drain_authority, _ = _drain_authority((key,))
    curriculum = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=(key,),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=C.BallCurriculumConfig(),
        scheduler_config=C.ArmSchedulerConfig(),
        evaluator_authority=authority,
        drain_reset_authority=drain_authority,
    )
    domain = curriculum.selected_formal_domain(key)
    snapshot = authority.freeze_checkpoint(b"curriculum-v4-checkpoint")
    canary_session = authority.open_window(
        snapshot=snapshot,
        key=key,
        evidence_role="frozen_canary",
        domain_epoch=domain.domain_epoch,
        stratum=domain.stratum,
        selected_arm_key=domain.selected_arm_key,
        selection_round=domain.selection_round,
        arm_levels=domain.arm_levels,
        rho=domain.rho,
    )
    canary = fixture._finish(
        authority,
        canary_session,
        E.V4_CANARY_PROPOSALS,
    )
    heldout_session = authority.open_window(
        snapshot=snapshot,
        key=key,
        evidence_role="frozen_heldout",
        domain_epoch=domain.domain_epoch,
        stratum=domain.stratum,
        selected_arm_key=domain.selected_arm_key,
        selection_round=domain.selection_round,
        arm_levels=domain.arm_levels,
        rho=domain.rho,
    )
    heldout = fixture._finish(
        authority,
        heldout_session,
        E.V4_HELDOUT_PROPOSALS,
    )
    release = authority.issue_release(
        canary=canary,
        heldout=heldout,
    )
    return curriculum, authority, key, release


def _v4_system_for_keys(keys):
    fixture = _v4_fixture()
    launch = E.launch_receipt_document_v4(
        curriculum_contract_sha256=CONTRACT,
        profile_order=keys,
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=(
            C.ArmSchedulerConfig().contract_sha256
        ),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        attempt_source_contract_sha256=fixture.SOURCE_CONTRACT,
        attempt_source_path=fixture.SOURCE_PATH,
        attempt_source_sha256=fixture.SOURCE_SHA,
    )
    launch_sha = E._canonical_sha256(launch)
    E.TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256 = (
        frozenset(
            set(
                E.TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256
            )
            | {launch_sha}
        )
    )
    evaluator = E.FrozenEvaluatorV4Authority.from_trusted_launch_receipt(
        launch,
        attempt_source=fixture.ExactAttemptSource(),
    )
    drain, _ = _drain_authority(keys)
    curriculum = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=keys,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=C.BallCurriculumConfig(),
        scheduler_config=C.ArmSchedulerConfig(),
        evaluator_authority=evaluator,
        drain_reset_authority=drain,
    )
    return curriculum, evaluator


def _v4_release_for_snapshot(curriculum, authority, snapshot, key):
    fixture = _v4_fixture()
    domain = curriculum.selected_formal_domain(key)
    canary_session = authority.open_window(
        snapshot=snapshot,
        key=key,
        evidence_role="frozen_canary",
        domain_epoch=domain.domain_epoch,
        stratum=domain.stratum,
        selected_arm_key=domain.selected_arm_key,
        selection_round=domain.selection_round,
        arm_levels=domain.arm_levels,
        rho=domain.rho,
    )
    canary = fixture._finish(
        authority, canary_session, E.V4_CANARY_PROPOSALS
    )
    heldout_session = authority.open_window(
        snapshot=snapshot,
        key=key,
        evidence_role="frozen_heldout",
        domain_epoch=domain.domain_epoch,
        stratum=domain.stratum,
        selected_arm_key=domain.selected_arm_key,
        selection_round=domain.selection_round,
        arm_levels=domain.arm_levels,
        rho=domain.rho,
    )
    heldout = fixture._finish(
        authority, heldout_session, E.V4_HELDOUT_PROPOSALS
    )
    return authority.issue_release(canary=canary, heldout=heldout)


def _key(action_uid=1, mobility="move", profile_char="f"):
    return C.ActionProfileKey(
        action_uid,
        profile_char * 64,
        mobility,
    )


def _system(
    keys=None,
    *,
    scheduler_config=None,
    config=None,
    drain_consumed=(),
):
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
    drain_authority, _ = _drain_authority(
        keys,
        source=ExactDrainResetSource(consumed=drain_consumed),
        scheduler_config=scheduler_config,
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
        drain_reset_authority=drain_authority,
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
        curriculum._observe_scheduler_legacy_for_test({key: capability})
    raise AssertionError("selected arm ring never filled")


def _certify(
    curriculum,
    authority,
    factory,
    key,
    *,
    failures=0,
    table_hits=0,
    new_band=None,
    commit=True,
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
    canary_decision = curriculum._stage_legacy_capabilities_for_test(
        {key: canary}
    )[0]
    if new_band is None:
        heldout_new_band = 768 if domain.selected_arm_key else 0
    else:
        heldout_new_band = new_band
    heldout = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_heldout",
        count=768,
        domain=domain,
        failures=failures,
        table_hits=table_hits,
        new_band=heldout_new_band,
    )
    heldout_decision = curriculum._stage_legacy_capabilities_for_test(
        {key: heldout}
    )[0]
    if commit:
        pending = curriculum.pending_domain_release(key)
        assert pending is not None
        token = curriculum.issue_global_pre_reset_barrier()
        receipts = curriculum.commit_release(token)
        assert len(receipts) == 1
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
        for event_kind, template_receipt in template_events:
            model_progress = curriculum._progress[key]
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
                curriculum._apply_formal_evidence(
                    key, model_progress, evidence
                )
                assert (
                    model_progress.formal_receipts[-1].certified
                    == template_receipt["certified"]
                )
                if model_progress.pending_release is not None:
                    barrier = curriculum.issue_global_pre_reset_barrier()
                    curriculum.commit_release(barrier)
                attempt_storage = "formal_compact"
                ordered_attempts = None
            else:
                attempts = deepcopy(template_receipt["attempts"])
                normalized = tuple(
                    C._validated_attempt_row(
                        item,
                        name=f"synthetic scheduler[{index}]",
                    )
                    for index, item in enumerate(attempts)
                )
                model_progress.scheduler_receipts += (
                    C._SchedulerReceipt(
                        evidence=evidence,
                        attempts=normalized,
                    ),
                )
                curriculum._reselect_arm(key, model_progress)
                attempt_storage = "full"
                ordered_attempts = deepcopy(attempts)
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
        progress_rows.append(
            curriculum._progress_row(
                key, curriculum._progress[key]
            )
        )

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
    state["next_barrier_serial"] = curriculum._next_barrier_serial
    state["issued_global_pre_reset_barriers"] = []
    state["evaluator_authority_state"] = authority_state
    state["evaluator_authority_state_sha256"] = C._canonical_sha256(
        authority_state
    )
    drain_state = curriculum._drain_reset_authority.state_dict()
    state["drain_reset_authority_state"] = drain_state
    state["drain_reset_authority_state_sha256"] = C._canonical_sha256(
        drain_state
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
    assert config.heldout_min_new_band == 154
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


def test_release_api_has_no_caller_counts_and_production_is_fail_closed():
    signature = inspect.signature(
        C.ActionBallCurriculum.issue_global_pre_reset_barrier
    )
    assert tuple(signature.parameters) == ("self",)
    curriculum = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=(_key(),),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=C.BallCurriculumConfig(),
    )
    assert curriculum.release_authorized is False
    with pytest.raises(
        C.DrainResetAuthorityError, match="authority is not bound"
    ):
        curriculum.issue_global_pre_reset_barrier()
    with pytest.raises(TypeError):
        curriculum.issue_global_pre_reset_barrier(active_births=0)


def test_drain_reset_authority_requires_code_pin_and_exact_source(monkeypatch):
    keys = (_key(),)
    launch = _drain_launch(keys)
    monkeypatch.setattr(
        C,
        "TRUSTED_DRAIN_RESET_LAUNCH_RECEIPT_SHA256",
        frozenset(),
    )
    with pytest.raises(
        C.DrainResetAuthorityError, match="not code-pinned"
    ):
        C.DrainResetAuthority.from_trusted_launch_receipt(
            launch, runtime_source=ExactDrainResetSource()
        )

    monkeypatch.setattr(
        C,
        "TRUSTED_DRAIN_RESET_LAUNCH_RECEIPT_SHA256",
        frozenset({C._canonical_sha256(launch)}),
    )
    source = ExactDrainResetSource()
    source.binding_document = lambda: {
        **ExactDrainResetSource().binding_document(),
        "runtime_source_sha256": "f" * 64,
    }
    with pytest.raises(
        C.DrainResetAuthorityError, match="binding mismatch"
    ):
        C.DrainResetAuthority.from_trusted_launch_receipt(
            launch, runtime_source=source
        )


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
        match="schema-4 evaluator authority",
    ):
        curriculum.update_selected({key: capability.evidence})
    assert curriculum.state_dict()["progress"] == before

    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="schema-4 evaluator authority",
    ):
        curriculum.update_selected({key: capability})
    decision = curriculum._stage_legacy_capabilities_for_test(
        {key: capability}
    )[0]
    assert decision.kind == "canary_pass"
    assert curriculum.phase(key) == "center"
    assert set(curriculum.frontiers(key).values()) == {0.0}


def test_public_stage_accepts_only_v4_opaque_release_and_does_not_publish():
    curriculum, authority, key, release = _v4_system_and_release()
    before = curriculum.selected_formal_domain(key)
    decision = curriculum.stage_selected({key: release})[0]
    assert decision.kind == "center_pass"
    assert curriculum.selected_formal_domain(key) == before
    pending = curriculum.pending_domain_release(key)
    assert pending is not None
    assert pending.canary_window_sha256 == (
        release.canary_window_sha256
    )
    assert pending.heldout_window_sha256 == (
        release.heldout_window_sha256
    )
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="stale",
    ):
        authority.assert_release_receipt(release)


def test_v4_blocked_canary_pair_is_consumed_but_never_stages_release():
    fixture = _v4_fixture()
    source = fixture.ExactAttemptSource(
        terminal_signals={
            0: fixture.R.FrozenTerminalSignals(table_hit=True),
        }
    )
    curriculum, _, key, release = _v4_system_and_release(source)
    decision = curriculum.stage_selected({key: release})[0]
    assert decision.kind == "canary_blocked"
    assert "table_hit_zero_tolerance" in decision.blockers
    assert curriculum.pending_domain_release(key) is None
    state = deepcopy(curriculum.state_dict())

    authority = fixture._authority(
        fixture.ExactAttemptSource(
            terminal_signals={
                0: fixture.R.FrozenTerminalSignals(table_hit=True),
            }
        )
    )
    drain_authority, _ = _drain_authority((key,))
    resumed = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=(key,),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=C.BallCurriculumConfig(),
        scheduler_config=C.ArmSchedulerConfig(),
        evaluator_authority=authority,
        drain_reset_authority=drain_authority,
    )
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state
    assert resumed.phase(key) == "center"


def test_scheduler_ingest_accepts_only_exact_v4_transcript():
    fixture = _v4_fixture()
    curriculum, authority, key, release = _v4_system_and_release()
    curriculum.stage_selected({key: release})
    pending = curriculum.pending_domain_release(key)
    barrier = curriculum.issue_global_pre_reset_barrier()
    curriculum.commit_release(barrier)
    selected = curriculum.selected_arm(key)
    domain = next(
        item
        for item in curriculum.scheduler_domains(key)
        if item.selected_arm_key == selected
    )
    snapshot = authority.freeze_checkpoint(b"scheduler-v4-checkpoint")
    session = authority.open_window(
        snapshot=snapshot,
        key=key,
        evidence_role="scheduler",
        domain_epoch=domain.domain_epoch,
        stratum=domain.stratum,
        selected_arm_key=domain.selected_arm_key,
        selection_round=domain.selection_round,
        arm_levels=domain.arm_levels,
        rho=domain.rho,
    )
    capability = fixture._finish(
        authority,
        session,
        E.V4_SCHEDULER_PROPOSALS,
    )
    assert curriculum.observe_scheduler({key: capability})
    with pytest.raises(
        E.FrozenEvaluationAuthorityError,
        match="stale or consumed",
    ):
        curriculum.observe_scheduler({key: capability})
    state = deepcopy(curriculum.state_dict())
    resumed_authority = fixture._authority()
    resumed_drain_authority, _ = _drain_authority(
        (key,),
        source=ExactDrainResetSource(
            consumed=_drain_consumed(state)
        ),
    )
    resumed = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=(key,),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=C.BallCurriculumConfig(),
        scheduler_config=C.ArmSchedulerConfig(),
        evaluator_authority=resumed_authority,
        drain_reset_authority=resumed_drain_authority,
    )
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state


def test_v4_two_profile_release_is_one_atomic_global_epoch_publish():
    fixture = _v4_fixture()
    keys = (
        fixture.KEY,
        C.ActionProfileKey(8, "e" * 64, "no_move"),
    )
    launch = E.launch_receipt_document_v4(
        curriculum_contract_sha256=CONTRACT,
        profile_order=keys,
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=(
            C.ArmSchedulerConfig().contract_sha256
        ),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        attempt_source_contract_sha256=fixture.SOURCE_CONTRACT,
        attempt_source_path=fixture.SOURCE_PATH,
        attempt_source_sha256=fixture.SOURCE_SHA,
    )
    launch_sha = E._canonical_sha256(launch)
    E.TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256 = (
        frozenset(
            set(
                E.TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256
            )
            | {launch_sha}
        )
    )
    authority = E.FrozenEvaluatorV4Authority.from_trusted_launch_receipt(
        launch,
        attempt_source=fixture.ExactAttemptSource(),
    )
    drain_authority, _ = _drain_authority(keys)
    curriculum = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=keys,
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=C.BallCurriculumConfig(),
        scheduler_config=C.ArmSchedulerConfig(),
        evaluator_authority=authority,
        drain_reset_authority=drain_authority,
    )
    snapshot = authority.freeze_checkpoint(b"two-profile-checkpoint")
    releases = {}
    before = {}
    for key in keys:
        domain = curriculum.selected_formal_domain(key)
        before[key] = domain
        canary_session = authority.open_window(
            snapshot=snapshot,
            key=key,
            evidence_role="frozen_canary",
            domain_epoch=domain.domain_epoch,
            stratum=domain.stratum,
            selected_arm_key=domain.selected_arm_key,
            selection_round=domain.selection_round,
            arm_levels=domain.arm_levels,
            rho=domain.rho,
        )
        canary = fixture._finish(
            authority,
            canary_session,
            E.V4_CANARY_PROPOSALS,
        )
        heldout_session = authority.open_window(
            snapshot=snapshot,
            key=key,
            evidence_role="frozen_heldout",
            domain_epoch=domain.domain_epoch,
            stratum=domain.stratum,
            selected_arm_key=domain.selected_arm_key,
            selection_round=domain.selection_round,
            arm_levels=domain.arm_levels,
            rho=domain.rho,
        )
        heldout = fixture._finish(
            authority,
            heldout_session,
            E.V4_HELDOUT_PROPOSALS,
        )
        releases[key] = authority.issue_release(
            canary=canary,
            heldout=heldout,
        )
    curriculum.stage_selected(releases)
    pending = {
        key: curriculum.pending_domain_release(key) for key in keys
    }
    assert all(
        curriculum.selected_formal_domain(key) == before[key]
        for key in keys
    )
    token = curriculum.issue_global_pre_reset_barrier()
    receipts = curriculum.commit_release(token)
    assert len(receipts) == 2
    assert {item.barrier_token_sha256 for item in receipts} == {
        token.token_sha256
    }
    assert {item.commit_serial for item in receipts} == {
        token.barrier_serial
    }
    assert all(curriculum.domain_epoch(key) == 1 for key in keys)


def test_pending_release_rejects_new_policy_snapshot_before_consumption():
    fixture = _v4_fixture()
    keys = (
        fixture.KEY,
        C.ActionProfileKey(9, "9" * 64, "move"),
    )
    curriculum, authority = _v4_system_for_keys(keys)
    snapshot_1 = authority.freeze_checkpoint(b"pending-snapshot-one")
    release_1 = _v4_release_for_snapshot(
        curriculum, authority, snapshot_1, keys[0]
    )
    curriculum.stage_selected({keys[0]: release_1})
    assert curriculum.pending_domain_release(keys[0]) is not None

    snapshot_2 = authority.freeze_checkpoint(b"pending-snapshot-two")
    release_2 = _v4_release_for_snapshot(
        curriculum, authority, snapshot_2, keys[1]
    )
    with pytest.raises(
        ValueError, match="differs from pending releases"
    ):
        curriculum.stage_selected({keys[1]: release_2})
    # The preflight runs before the authority's atomic consume.
    assert authority.assert_release_receipt(release_2)
    assert curriculum.pending_domain_release(keys[1]) is None


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
    curriculum._stage_legacy_capabilities_for_test({key: canary})
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
    curriculum._stage_legacy_capabilities_for_test({key: heldout})
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
        curriculum._stage_legacy_capabilities_for_test({key: heldout})

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
        curriculum._stage_legacy_capabilities_for_test({key: too_small})

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
    curriculum._stage_legacy_capabilities_for_test({key: canary})
    too_small_heldout = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_heldout",
        count=767,
        domain=domain,
    )
    with pytest.raises(ValueError, match="below 768"):
        curriculum._stage_legacy_capabilities_for_test(
            {key: too_small_heldout}
        )


def test_outcome_ledger_allows_overlapping_raw_unsafe_channels():
    ledger = C.BallOutcomeLedger(
        P=1,
        A=1,
        I=1,
        S=1,
        C=1,
        L=0,
        F=0,
        U_table=1,
        U_fall=0,
        U_collision=0,
        X=0,
        U_joint_qdes=0,
        U_joint_actual=1,
    )
    assert ledger.unsafe == 1
    assert ledger.U_table == 1
    assert ledger.U_joint_actual == 1


def test_schema4_scheduler_ledger_retains_overlapping_raw_safety_signals():
    row = C._validated_attempt_row_v4(
        {
            "sample_receipt_sha256": _digest("sample", 1, 0),
            "birth_receipt_sha256": _digest("birth", 1, 0),
            "solver_admitted": True,
            "installed": True,
            "started": True,
            "closed": True,
            "terminal_outcome": "joint_actual_limit",
            "infrastructure_invalid": False,
            "in_new_band": False,
            "terminal_signals": {
                "infrastructure_invalid": False,
                "joint_actual_limit": True,
                "joint_qdes_limit": False,
                "fall": False,
                "table_hit": True,
                "collision": False,
                "legal_return": False,
            },
        },
        name="scheduler[0]",
    )
    ledger = C._ledger_from_attempt_rows((row,))
    assert ledger.unsafe == 1
    assert ledger.U_joint_actual == 1
    assert ledger.U_table == 1
    assert ledger.L == 0
    assert ledger.F == 0


def test_single_pass_scheduler_ledger_matches_reference_reduction():
    """The optimized reduction is byte-for-byte equivalent to the old scans."""

    rows = []
    for index, terminal in enumerate(
        (
            "legal_return",
            "safe_nonreturn",
            "table_hit",
            "fall",
            "collision",
            "joint_qdes_limit",
            "joint_actual_limit",
        )
    ):
        rows.append(
            C._validated_attempt_row(
                {
                    "sample_receipt_sha256": _digest(
                        "ledger-sample", 1, index
                    ),
                    "birth_receipt_sha256": _digest(
                        "ledger-birth", 1, index
                    ),
                    "solver_admitted": True,
                    "installed": True,
                    "started": True,
                    "closed": True,
                    "terminal_outcome": terminal,
                    "infrastructure_invalid": False,
                    "in_new_band": index < 2,
                },
                name=f"legacy[{index}]",
            )
        )
    rows.append(
        C._validated_attempt_row_v4(
            {
                "sample_receipt_sha256": _digest(
                    "ledger-sample", 1, 100
                ),
                "birth_receipt_sha256": _digest(
                    "ledger-birth", 1, 100
                ),
                "solver_admitted": True,
                "installed": True,
                "started": True,
                "closed": True,
                "terminal_outcome": "joint_actual_limit",
                "infrastructure_invalid": False,
                "in_new_band": False,
                "terminal_signals": {
                    "infrastructure_invalid": False,
                    "joint_actual_limit": True,
                    "joint_qdes_limit": True,
                    "fall": False,
                    "table_hit": True,
                    "collision": False,
                    "legal_return": False,
                },
            },
            name="overlap[0]",
        )
    )
    rows.append(
        C._validated_attempt_row(
            {
                "sample_receipt_sha256": _digest(
                    "ledger-sample", 1, 101
                ),
                "birth_receipt_sha256": _digest(
                    "ledger-birth", 1, 101
                ),
                "solver_admitted": False,
                "installed": False,
                "started": False,
                "closed": False,
                "terminal_outcome": None,
                "infrastructure_invalid": True,
                "in_new_band": False,
            },
            name="infrastructure[0]",
        )
    )

    terminals = {name: 0 for name in C._TERMINALS}
    new_band = 0
    new_band_failures = 0
    for row in rows:
        terminal = row["terminal_outcome"]
        if terminal is not None:
            terminals[terminal] += 1
        if C._ring_eligible(row):
            new_band += 1
            new_band_failures += terminal == "safe_nonreturn"

    def raw_signal_count(signal):
        count = 0
        for row in rows:
            signals = row.get("terminal_signals")
            count += (
                row["terminal_outcome"] == signal
                if signals is None
                else bool(signals[signal])
            )
        return count

    reference = C.BallOutcomeLedger(
        P=len(rows),
        A=sum(bool(row["solver_admitted"]) for row in rows),
        I=sum(bool(row["installed"]) for row in rows),
        S=sum(bool(row["started"]) for row in rows),
        C=sum(bool(row["closed"]) for row in rows),
        L=terminals["legal_return"],
        F=terminals["safe_nonreturn"],
        U_table=raw_signal_count("table_hit"),
        U_fall=raw_signal_count("fall"),
        U_collision=raw_signal_count("collision"),
        X=sum(bool(row["infrastructure_invalid"]) for row in rows),
        U_joint_qdes=raw_signal_count("joint_qdes_limit"),
        U_joint_actual=raw_signal_count("joint_actual_limit"),
        NB=new_band,
        NB_F=new_band_failures,
    )
    assert C._ledger_from_attempt_rows(tuple(rows)) == reference


def test_outcome_ledger_requires_raw_signal_for_every_unsafe_closure():
    with pytest.raises(
        ValueError, match="needs at least one raw sticky safety signal"
    ):
        C.BallOutcomeLedger(
            P=1,
            A=1,
            I=1,
            S=1,
            C=1,
            L=0,
            F=0,
            U_table=0,
            U_fall=0,
            U_collision=0,
            X=0,
        )


def test_center_then_signed_marginals_expand_independently():
    # The formal heldout interval, not scheduler history, decides each
    # independently signed candidate.  Zero failures is too easy; 230/768
    # overlaps the [0.15, 0.45] band and therefore locks.
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

    lower_arm = curriculum.selected_arm(key)
    canary, lower = _certify(curriculum, authority, factory, key)
    assert canary.kind == "canary_pass"
    assert lower.kind == "expand_marginal"
    assert curriculum.frontiers(key)[lower_arm] == 0.25

    upper_arm = curriculum.selected_arm(key)
    assert upper_arm != lower_arm or (
        # An expanded arm may be re-probed at its next level.
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


def test_marginal_release_does_not_require_scheduler_ring():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    assert curriculum.phase(key) == "marginal"
    arm = curriculum.selected_arm(key)
    assert _ring_len(curriculum, key, arm) == 0
    _, decision = _certify(curriculum, authority, factory, key)
    assert decision.kind == "expand_marginal"
    assert curriculum.frontiers(key)[arm] == 0.25


@pytest.mark.parametrize(
    ("scheduler_failures", "heldout_failures", "expected_kind", "frontier"),
    (
        (30, 0, "expand_marginal", 0.25),
        (0, 768, "bound_marginal", 0.0),
    ),
)
def test_scheduler_ring_has_no_formal_release_authority(
    scheduler_failures,
    heldout_failures,
    expected_kind,
    frontier,
):
    # Deliberately contradict the scheduler ring with the frozen heldout.
    # Only the heldout verdict may move or bind the formal frontier.
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    arm = _fill_ring(
        curriculum,
        authority,
        factory,
        key,
        ring_failures=scheduler_failures,
    )
    _, decision = _certify(
        curriculum,
        authority,
        factory,
        key,
        failures=heldout_failures,
    )
    assert decision.kind == expected_kind
    assert curriculum.frontiers(key)[arm] == frontier


@pytest.mark.parametrize(
    ("heldout_failures", "expected_kind", "frontier"),
    (
        (0, "expand_marginal", 0.25),
        (77, "lock_marginal", 0.25),
        (768, "bound_marginal", 0.0),
    ),
)
def test_frozen_heldout_decides_marginal_expand_lock_or_bound(
    heldout_failures,
    expected_kind,
    frontier,
):
    # These synthetic legacy windows mark all 768 heldout rows as belonging to
    # the selected action-axis-side new band.  Zero failures has UCB below
    # 7.5%, 77 failures overlaps the band, and 768 failures has LCB above
    # 12.5%.  In particular, 768/768 may never expand.
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    arm = curriculum.selected_arm(key)
    _, decision = _certify(
        curriculum,
        authority,
        factory,
        key,
        failures=heldout_failures,
    )
    assert decision.kind == expected_kind
    assert curriculum.frontiers(key)[arm] == frontier
    if expected_kind == "bound_marginal":
        assert (
            curriculum._progress[key].arm_status[C.ARM_KEYS.index(arm)]
            == "decided"
        )


def test_marginal_uses_new_band_when_whole_domain_would_expand():
    # The 35 failures are only 4.6% of the full 768-row heldout, whose Wilson
    # UCB lies below the 7.5% lower edge and would incorrectly expand.  They
    # are 18.2% of this action-axis-side's 192-row new band, whose Wilson LCB
    # lies above the 12.5% upper edge and must bound the candidate.
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    arm = curriculum.selected_arm(key)

    _, decision = _certify(
        curriculum,
        authority,
        factory,
        key,
        failures=35,
        new_band=192,
    )

    whole_domain = C.wilson_interval(
        35,
        768,
        z=curriculum.config.confidence_z,
    )
    new_band = C.wilson_interval(
        35,
        192,
        z=curriculum.config.confidence_z,
    )
    assert whole_domain.upper < curriculum.config.failure_band[0]
    assert new_band.lower > curriculum.config.failure_band[1]
    assert decision.policy_failure == new_band
    assert decision.kind == "bound_marginal"
    assert curriculum.frontiers(key)[arm] == 0.0


@pytest.mark.parametrize(
    ("new_band", "expected_kind", "expected_frontier", "blocked"),
    (
        (153, "bound_marginal", 0.0, True),
        (154, "expand_marginal", 0.25, False),
    ),
)
def test_marginal_requires_minimum_frozen_heldout_new_band_rows(
    new_band,
    expected_kind,
    expected_frontier,
    blocked,
):
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    arm = curriculum.selected_arm(key)

    _, decision = _certify(
        curriculum,
        authority,
        factory,
        key,
        new_band=new_band,
    )

    assert curriculum.config.heldout_min_new_band == 154
    assert decision.kind == expected_kind
    assert (
        "new_band_safe_closed_below_gate" in decision.blockers
    ) is blocked
    assert curriculum.frontiers(key)[arm] == expected_frontier


def test_scheduler_ring_is_retained_for_candidate_scheduling_only():
    for failures in (0, 30):
        key = _key()
        curriculum, authority = _system((key,))
        factory = EvidenceFactory()
        _certify(curriculum, authority, factory, key)
        arm = _fill_ring(
            curriculum, authority, factory, key, ring_failures=failures
        )
        assert _ring_len(curriculum, key, arm) == 30
        assert _ring_failures(curriculum, key, arm) == failures
        assert curriculum.frontiers(key)[arm] == 0.0


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
        arm = curriculum.selected_arm(key)
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
    curriculum._observe_scheduler_legacy_for_test({key: rejected})
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
    curriculum._observe_scheduler_legacy_for_test({key: clean})
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
        curriculum._observe_scheduler_legacy_for_test({key: capability})

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
    assert curriculum._stage_legacy_capabilities_for_test(
        {key: canary}
    )[0].kind == "canary_pass"
    heldout = factory.issue(
        curriculum,
        authority,
        key,
        role="frozen_heldout",
        count=769,
        domain=domain,
        table_hits=1,
    )
    decision = curriculum._stage_legacy_capabilities_for_test(
        {key: heldout}
    )[0]
    assert decision.kind == "bound_marginal"
    assert "table_hit_zero_tolerance" in decision.blockers
    assert curriculum.frontiers(key) == before


def test_heldout_stages_then_global_barrier_publishes_once_only():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    before_domain = curriculum.selected_formal_domain(key)
    before_state = curriculum.frontiers(key)
    _, decision = _certify(
        curriculum,
        authority,
        factory,
        key,
        commit=False,
    )
    pending = curriculum.pending_domain_release(key)
    assert pending is not None
    assert decision.kind == "center_pass"
    assert curriculum.phase(key) == "center"
    assert curriculum.frontiers(key) == before_state
    assert curriculum.selected_formal_domain(key) == before_domain
    assert pending.from_domain_epoch == before_domain.domain_epoch
    assert pending.to_domain_epoch == before_domain.domain_epoch + 1

    source = curriculum._drain_reset_authority._runtime_source
    source.active_births = 1
    with pytest.raises(
        C.DrainResetAuthorityError, match="active work"
    ):
        curriculum.issue_global_pre_reset_barrier()
    source.active_births = 0
    source.reset_count = 3
    with pytest.raises(
        C.DrainResetAuthorityError, match="partial reset"
    ):
        curriculum.issue_global_pre_reset_barrier()
    source.reset_count = source.env_count

    drain_receipt = curriculum.issue_global_pre_reset_barrier()
    with pytest.raises(RuntimeError, match="new work blocked"):
        source.begin_birth()
    with pytest.raises(
        C.DrainResetAuthorityError, match="opaque DrainResetReceipt"
    ):
        curriculum.commit_release(drain_receipt._token)
    receipt = curriculum.commit_release(drain_receipt)[0]
    assert receipt.release_id_sha256 == pending.release_id_sha256
    assert curriculum.phase(key) == "marginal"
    assert curriculum.domain_epoch(key) == pending.to_domain_epoch
    assert curriculum.pending_domain_release(key) is None
    with pytest.raises(
        C.DrainResetAuthorityError, match="stale, forged, or consumed"
    ):
        curriculum.commit_release(drain_receipt)


def test_runtime_source_not_callers_owns_every_drain_and_reset_gate():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(
        curriculum,
        authority,
        factory,
        key,
        commit=False,
    )
    source = curriculum._drain_reset_authority._runtime_source
    cases = (
        ("active_attempts", 1, "active work"),
        ("reserved_attempts", 1, "active work"),
        ("active_births", 1, "active work"),
        ("pending_task_receipts", 1, "active work"),
        ("attempt_pool_reset_generation", 2, "one reset generation"),
        ("reset_count", source.env_count - 1, "partial reset"),
        (
            "reset_participant_ids",
            list(range(source.env_count - 1)),
            "N-of-N",
        ),
    )
    for field, invalid, message in cases:
        original = deepcopy(getattr(source, field))
        setattr(source, field, invalid)
        with pytest.raises(C.DrainResetAuthorityError, match=message):
            curriculum.issue_global_pre_reset_barrier()
        assert curriculum.issued_global_pre_reset_barriers() == ()
        setattr(source, field, original)

    original_bitmap = source._bitmap
    source._bitmap = lambda: "f" * 64
    with pytest.raises(
        C.DrainResetAuthorityError, match="bitmap"
    ):
        curriculum.issue_global_pre_reset_barrier()
    source._bitmap = original_bitmap


def test_fenced_commit_rechecks_generation_before_publication():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(
        curriculum,
        authority,
        factory,
        key,
        commit=False,
    )
    source = curriculum._drain_reset_authority._runtime_source
    before = curriculum.selected_formal_domain(key)
    receipt = curriculum.issue_global_pre_reset_barrier()
    source.env_reset_generation += 1
    with pytest.raises(RuntimeError, match="changed under fence"):
        curriculum.commit_release(receipt)
    assert curriculum.selected_formal_domain(key) == before
    assert curriculum.pending_domain_release(key) is not None


def test_v4_exact_resume_covers_pending_barrier_and_committed_release():
    fixture = _v4_fixture()
    curriculum, _, key, release = _v4_system_and_release()
    curriculum.stage_selected({key: release})
    pending_state = deepcopy(curriculum.state_dict())

    authority_1 = fixture._authority()
    drain_authority_1, _ = _drain_authority((key,))
    resumed_1 = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=(key,),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=C.BallCurriculumConfig(),
        scheduler_config=C.ArmSchedulerConfig(),
        evaluator_authority=authority_1,
        drain_reset_authority=drain_authority_1,
    )
    resumed_1.load_state_dict(pending_state)
    assert resumed_1.state_dict() == pending_state
    assert resumed_1.pending_domain_release(key) is not None
    old_receipt = resumed_1.issue_global_pre_reset_barrier()
    with pytest.raises(
        C.DrainResetAuthorityError, match="cannot checkpoint a live"
    ):
        resumed_1.state_dict()

    authority_2 = fixture._authority()
    drain_authority_2, _ = _drain_authority((key,))
    resumed_2 = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=(key,),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=C.BallCurriculumConfig(),
        scheduler_config=C.ArmSchedulerConfig(),
        evaluator_authority=authority_2,
        drain_reset_authority=drain_authority_2,
    )
    resumed_2.load_state_dict(pending_state)
    with pytest.raises(
        C.DrainResetAuthorityError, match="another or restored authority"
    ):
        resumed_2.commit_release(old_receipt)
    redrained = resumed_2.issue_global_pre_reset_barrier()
    resumed_2.commit_release(redrained)
    committed_state = deepcopy(resumed_2.state_dict())

    authority_3 = fixture._authority()
    drain_authority_3, _ = _drain_authority(
        (key,),
        source=ExactDrainResetSource(
            consumed=_drain_consumed(committed_state)
        ),
    )
    resumed_3 = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=(key,),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=C.BallCurriculumConfig(),
        scheduler_config=C.ArmSchedulerConfig(),
        evaluator_authority=authority_3,
        drain_reset_authority=drain_authority_3,
    )
    resumed_3.load_state_dict(committed_state)
    assert resumed_3.state_dict() == committed_state
    assert resumed_3.phase(key) == "marginal"
    assert resumed_3.domain_epoch(key) == 1


def test_rehashed_drain_history_cannot_replace_source_consumed_transcript():
    key = _key()
    curriculum, authority = _system((key,))
    factory = EvidenceFactory()
    _certify(curriculum, authority, factory, key)
    state = deepcopy(curriculum.state_dict())
    trusted_consumed = _drain_consumed(state)

    forged = deepcopy(state)
    drain_state = forged["drain_reset_authority_state"]
    drain_state["consumed"] = []
    drain_state["consumed_hash_chain_sha256"] = "0" * 64
    unsigned_drain = dict(drain_state)
    unsigned_drain.pop("state_sha256")
    drain_state["state_sha256"] = C._canonical_sha256(unsigned_drain)
    forged["drain_reset_authority_state_sha256"] = (
        C._canonical_sha256(drain_state)
    )
    unsigned = dict(forged)
    unsigned.pop("state_sha256")
    forged["state_sha256"] = C._canonical_sha256(unsigned)

    resumed, _ = _system(
        (key,), drain_consumed=trusted_consumed
    )
    with pytest.raises(
        ValueError, match="authority and release history differ"
    ):
        resumed.load_state_dict(forged)


@pytest.mark.parametrize(
    ("terminal", "field", "blocker"),
    (
        (
            "joint_qdes_limit",
            "U_joint_qdes",
            "joint_qdes_limit_zero_tolerance",
        ),
        (
            "joint_actual_limit",
            "U_joint_actual",
            "joint_actual_limit_zero_tolerance",
        ),
    ),
)
def test_joint_limit_unsafe_is_zero_tolerance_not_difficulty_failure(
    terminal, field, blocker
):
    rows = [
        {
            "sample_receipt_sha256": _digest("sample", 1, index),
            "birth_receipt_sha256": _digest("birth", 1, index),
            "solver_admitted": True,
            "installed": True,
            "started": True,
            "closed": True,
            "terminal_outcome": terminal if index == 0 else "legal_return",
            "infrastructure_invalid": False,
            "in_new_band": False,
        }
        for index in range(256)
    ]
    ledger = C._ledger_from_attempt_rows(
        tuple(
            C._validated_attempt_row(row, name=f"attempt[{index}]")
            for index, row in enumerate(rows)
        )
    )
    assert getattr(ledger, field) == 1
    assert ledger.F == 0
    assert ledger.safe_closed == 255
    curriculum, _ = _system()
    metrics = curriculum._metrics(ledger)
    blockers = metrics[6]
    assert blocker in blockers
    assert metrics[8] is False
    assert metrics[9] is True


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
    curriculum._stage_legacy_capabilities_for_test({key: canary})
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
        resumed._stage_legacy_capabilities_for_test({key: heldout})
    restored = resumed_authority.pending_capability(
        heldout.capability_id
    )
    assert resumed._stage_legacy_capabilities_for_test(
        {key: restored}
    )[0].kind == "center_pass"


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
    curriculum._observe_scheduler_legacy_for_test({key: scheduler})
    state = deepcopy(curriculum.state_dict())
    state["progress"][0]["selected_arm_key"] = C.ARM_KEYS[-1]
    unsigned = dict(state)
    unsigned.pop("state_sha256")
    state["state_sha256"] = C._canonical_sha256(unsigned)

    resumed, _ = _system(
        (key,), drain_consumed=_drain_consumed(state)
    )
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


def test_compact_cached_multi_profile_replay_is_exact_and_tamper_closed():
    template_state = _completed_no_move_state()
    keys = tuple(
        _key(
            action_uid=index + 1,
            mobility="no_move",
            profile_char="abc"[index],
        )
        for index in range(3)
    )
    resumed, state = _synthesize_completed_state(template_state, keys)
    resumed.load_state_dict(deepcopy(state))
    assert resumed.state_dict() == state

    forged = deepcopy(state)
    forged["progress"][1]["selected_arm_key"] = C.ARM_KEYS[0]
    unsigned = dict(forged)
    unsigned.pop("state_sha256")
    forged["state_sha256"] = C._canonical_sha256(unsigned)
    fresh, _ = _system(
        keys, drain_consumed=_drain_consumed(state)
    )
    with pytest.raises(ValueError, match="deterministic replay"):
        fresh.load_state_dict(forged)


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
    with pytest.raises(
        ValueError,
        match=(
            "deterministic replay|release receipt is not replay-reachable|"
            "release receipt heldout window is absent"
        ),
    ):
        resumed.load_state_dict(deleted)


def test_counter_rally_objective_mask_disables_both_landing_y_arms_only():
    key = C.ActionProfileKey(101, "1" * 64, "move")
    config = C.BallCurriculumConfig(
        objective_inactive_arms=(
            "landing_aim_y_lower",
            "landing_aim_y_upper",
        )
    )
    curriculum = C.ActionBallCurriculum(
        contract_sha256=CONTRACT,
        profile_order=(key,),
        sampler_sha256=SAMPLER,
        solver_sha256=SOLVER,
        policy_contract_sha256=POLICY,
        config=config,
    )
    statuses = dict(
        zip(C.ARM_KEYS, curriculum._progress[key].arm_status)
    )
    assert statuses["landing_aim_y_lower"] == "disabled"
    assert statuses["landing_aim_y_upper"] == "disabled"
    assert statuses["landing_aim_x_lower"] == "pending"
    assert statuses["landing_aim_x_upper"] == "pending"
    assert set(config.active_arm_keys(mobility="move")) == (
        set(C.ARM_KEYS)
        - {"landing_aim_y_lower", "landing_aim_y_upper"}
    )

    legacy = C.BallCurriculumConfig()
    assert "objective_inactive_arms" not in legacy.as_dict()
    with pytest.raises(ValueError, match="only reviewed"):
        C.BallCurriculumConfig(
            objective_inactive_arms=("landing_aim_y_lower",)
        )
