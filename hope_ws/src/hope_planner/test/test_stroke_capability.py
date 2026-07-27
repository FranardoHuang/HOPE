"""Capability artifact and fail-closed N-action selector contract.

Run:
  PYTHONPATH=hope_ws/src/hope_planner \
    python -m pytest hope_ws/src/hope_planner/test/test_stroke_capability.py -q
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import pytest

from hope_planner.action_catalog import ActionCatalog
from hope_planner.stroke_capability import (
    CapabilityArtifact,
    CapabilityContractError,
    CandidateEvidence,
    SelectorProfile,
    select_action,
)


def _sha(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


CURRENT = {
    "policy_sha256": _sha("policy"),
    "task_sha256": _sha("task"),
    "reward_sha256": _sha("reward"),
    "heldout_sha256": _sha("heldout"),
    "model_sha256": _sha("model"),
    "calibration_sha256": _sha("calibration"),
}
QUERY_SHA256 = _sha("query")


@dataclass(frozen=True)
class _Action:
    action_id: str
    action_uid: int
    slot: int


class _Catalog:
    """Minimal realization of the action_catalog.py consumer interface."""

    def __init__(self, count, label="catalog", uid_offset=10_000):
        self.actions = tuple(
            _Action(action_id=f"stroke_{index:03d}", action_uid=uid_offset + index + 1, slot=index)
            for index in range(count)
        )
        self.catalog_sha256 = _sha(f"{label}:{count}:{uid_offset}")
        self._by_uid = {action.action_uid: action for action in self.actions}

    def by_uid(self, uid):
        return self._by_uid[uid]


class _ExplodingRepr:
    def __repr__(self):
        raise RuntimeError("repr must not be called by canonical hashing")


def _real_catalog(count):
    return ActionCatalog.build(
        [
            {
                "action_id": f"stroke_{index:03d}",
                "family": "forehand" if index % 2 == 0 else "backhand",
                "content_sha256": _sha(f"motion-content-{index}"),
            }
            for index in range(count)
        ]
    )


def _artifact(catalog):
    return CapabilityArtifact.create(catalog=catalog, **CURRENT)


def _profile(catalog, **overrides):
    values = {
        "min_support": 20,
        "max_ood_score": 0.4,
        "min_lcb_success": 0.5,
        "delta_tie": 0.05,
        "priority_by_uid": {action.action_uid: action.slot for action in catalog.actions},
    }
    values.update(overrides)
    return SelectorProfile(**values)


def _candidates(catalog, lcbs=None):
    if lcbs is None:
        lcbs = [0.60 + 0.001 * action.slot for action in catalog.actions]
    return tuple(
        CandidateEvidence(
            action_uid=action.action_uid,
            hard_ok=True,
            hard_reason="",
            support_count=100,
            ood_score=0.1,
            lcb_success=lcb,
        )
        for action, lcb in zip(catalog.actions, lcbs)
    )


def _candidate_receipt(action_uid, query_sha256=QUERY_SHA256):
    return _sha("candidate:{}:{}".format(action_uid, query_sha256))


def _select(
    catalog,
    candidates,
    profile=None,
    artifact=None,
    query_sha256=QUERY_SHA256,
    expected_profile_sha256=None,
    **sha_overrides,
):
    current = dict(CURRENT)
    current.update(sha_overrides)
    current_artifact = artifact or _artifact(catalog)
    current_profile = profile or _profile(catalog)
    bound_candidates = tuple(
        candidate
        if candidate.evidence_sha256 is not None
        else candidate.bind_receipts(
            artifact_sha256=(
                current_artifact.artifact_sha256
                if candidate.artifact_sha256 is None
                else candidate.artifact_sha256
            ),
            query_sha256=(
                query_sha256
                if candidate.query_sha256 is None
                else candidate.query_sha256
            ),
            candidate_sha256=(
                _candidate_receipt(candidate.action_uid, query_sha256)
                if candidate.candidate_sha256 is None
                else candidate.candidate_sha256
            ),
        )
        for candidate in candidates
    )
    return select_action(
        catalog,
        current_artifact,
        current_profile,
        bound_candidates,
        query_sha256=query_sha256,
        expected_profile_sha256=(
            current_profile.profile_sha256
            if expected_profile_sha256 is None
            else expected_profile_sha256
        ),
        **current,
    )


def test_artifact_round_trip_and_canonical_self_hash():
    catalog = _Catalog(5)
    artifact = _artifact(catalog)
    raw = artifact.to_mapping()
    assert raw["schema_version"] == 1
    assert raw["catalog_sha256"] == catalog.catalog_sha256
    assert raw["action_uids"] == [action.action_uid for action in catalog.actions]
    assert CapabilityArtifact.from_mapping(raw) == artifact

    tampered = dict(raw)
    tampered["policy_sha256"] = _sha("tampered")
    with pytest.raises(CapabilityContractError, match="artifact_sha256"):
        CapabilityArtifact.from_mapping(tampered)


@pytest.mark.parametrize("mutation", ["missing", "extra", "tuple_uids", "bad_schema"])
def test_artifact_mapping_is_strict(mutation):
    raw = _artifact(_Catalog(2)).to_mapping()
    if mutation == "missing":
        raw.pop("reward_sha256")
    elif mutation == "extra":
        raw["comment"] = "not part of schema v1"
    elif mutation == "tuple_uids":
        raw["action_uids"] = tuple(raw["action_uids"])
    else:
        raw["schema_version"] = 2
    with pytest.raises(CapabilityContractError):
        CapabilityArtifact.from_mapping(raw)


@pytest.mark.parametrize(
    "field",
    [
        "policy_sha256",
        "task_sha256",
        "reward_sha256",
        "heldout_sha256",
        "model_sha256",
        "calibration_sha256",
    ],
)
def test_artifact_rejects_every_runtime_sha_mismatch(field):
    catalog = _Catalog(2)
    current = dict(CURRENT)
    current[field] = _sha("wrong-" + field)
    with pytest.raises(CapabilityContractError, match=field):
        _artifact(catalog).assert_compatible(catalog=catalog, **current)


def test_artifact_rejects_catalog_sha_and_slot_identity_mismatch():
    catalog = _Catalog(2, label="a")
    other_sha = _Catalog(2, label="b")
    with pytest.raises(CapabilityContractError, match="catalog_sha256"):
        _artifact(catalog).assert_compatible(catalog=other_sha, **CURRENT)

    other_uids = _Catalog(2, label="a", uid_offset=20_000)
    # Isolate the per-slot UID receipt from the catalog digest check.
    other_uids.catalog_sha256 = catalog.catalog_sha256
    with pytest.raises(CapabilityContractError, match="action_uids"):
        _artifact(catalog).assert_compatible(catalog=other_uids, **CURRENT)


def test_profile_hash_binds_all_thresholds_and_complete_priority_map():
    catalog = _Catalog(2)
    first = _profile(catalog)
    reordered_priorities = {
        action.action_uid: action.slot
        for action in reversed(catalog.actions)
    }
    same_content = _profile(catalog, priority_by_uid=reordered_priorities)
    changed_gate = _profile(catalog, min_support=21)
    changed_priority = _profile(
        catalog,
        priority_by_uid={
            catalog.actions[0].action_uid: 1,
            catalog.actions[1].action_uid: 0,
        },
    )

    assert first.profile_sha256 == same_content.profile_sha256
    assert first.profile_sha256 != changed_gate.profile_sha256
    assert first.profile_sha256 != changed_priority.profile_sha256
    assert first.to_mapping()["schema_version"] == 1


@pytest.mark.parametrize("count", [1, 2, 5, 6, 93])
def test_selector_scales_by_catalog_not_a_hardcoded_action_count(count):
    catalog = _real_catalog(count)
    lcbs = [0.55 + 0.4 * index / max(1, count - 1) for index in range(count)]
    decision = _select(
        catalog,
        _candidates(catalog, lcbs),
        profile=_profile(
            catalog,
            delta_tie=0.0,
            priority_by_uid={action.action_uid: 0 for action in catalog.actions},
        ),
    )
    assert decision.selected_action_uid == catalog.actions[-1].action_uid
    assert decision.selected_action_id == catalog.actions[-1].action_id
    assert decision.selected_slot == count - 1
    assert len(decision.assessments) == count


def test_decision_carries_artifact_profile_query_and_candidate_receipts():
    catalog = _Catalog(2)
    artifact = _artifact(catalog)
    profile = _profile(catalog)
    decision = _select(
        catalog,
        _candidates(catalog),
        artifact=artifact,
        profile=profile,
    )

    assert decision.artifact_sha256 == artifact.artifact_sha256
    assert decision.profile_sha256 == profile.profile_sha256
    assert decision.query_sha256 == QUERY_SHA256
    assert [row.candidate_sha256 for row in decision.assessments] == [
        _candidate_receipt(action.action_uid)
        for action in catalog.actions
    ]
    assert all(len(row.evidence_sha256) == 64 for row in decision.assessments)
    assert len(decision.decision_sha256) == 64

    changed_row = replace(
        decision.assessments[0],
        lcb_success=float(decision.assessments[0].lcb_success) + 0.01,
    )
    with pytest.raises(CapabilityContractError, match="evidence receipt"):
        replace(
            decision,
            assessments=(changed_row,) + decision.assessments[1:],
        )


def test_unsafe_optimistic_top_priority_action_can_never_win():
    catalog = _Catalog(2)
    unsafe, safe = catalog.actions
    candidates = (
        # Numeric evidence is deliberately stronger than the safe action.  The
        # selector must reject on the hard gate before reading any of it.
        CandidateEvidence(
            unsafe.action_uid,
            hard_ok=False,
            hard_reason="table_collision",
            support_count=1_000_000,
            ood_score=0.0,
            lcb_success=1.0,
        ),
        CandidateEvidence(
            safe.action_uid,
            hard_ok=True,
            hard_reason="",
            support_count=100,
            ood_score=0.1,
            lcb_success=0.7,
        ),
    )
    profile = _profile(
        catalog,
        priority_by_uid={unsafe.action_uid: 0, safe.action_uid: 99},
    )
    decision = _select(catalog, candidates, profile=profile)
    assert decision.selected_action_uid == safe.action_uid
    assert decision.assessments[0].reason == "hard_reject"
    assert decision.assessments[0].hard_reason == "table_collision"


def test_hard_reject_precedes_even_invalid_numeric_diagnostics():
    catalog = _Catalog(2)
    candidates = (
        CandidateEvidence(
            catalog.actions[0].action_uid,
            hard_ok=False,
            hard_reason="no_reach",
            support_count=-1,
            ood_score=float("nan"),
            lcb_success=float("inf"),
        ),
        _candidates(catalog, [0.8, 0.8])[1],
    )
    decision = _select(catalog, candidates)
    assert decision.selected_slot == 1
    assert decision.assessments[0].reason == "hard_reject"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("support_count", -1),
        ("support_count", 10.5),
        ("ood_score", float("nan")),
        ("ood_score", float("inf")),
        ("ood_score", -0.01),
        ("ood_score", 1.01),
        ("lcb_success", float("nan")),
        ("lcb_success", float("-inf")),
        ("lcb_success", -0.01),
        ("lcb_success", 1.01),
    ],
)
def test_invalid_evidence_only_invalidates_its_candidate(field, bad):
    catalog = _Catalog(2)
    first = {
        "action_uid": catalog.actions[0].action_uid,
        "hard_ok": True,
        "hard_reason": "",
        "support_count": 100,
        "ood_score": 0.1,
        "lcb_success": 0.99,
    }
    first[field] = bad
    candidates = (
        CandidateEvidence(**first),
        CandidateEvidence(
            catalog.actions[1].action_uid,
            hard_ok=True,
            hard_reason="",
            support_count=100,
            ood_score=0.1,
            lcb_success=0.70,
        ),
    )
    decision = _select(catalog, candidates)
    assert decision.selected_slot == 1
    assert decision.assessments[0].reason == "invalid_evidence"


def test_explicit_all_invalid_low_support_ood_and_hard_abstentions():
    catalog = _Catalog(2)
    base = [
        {
            "action_uid": action.action_uid,
            "hard_ok": True,
            "hard_reason": "",
            "support_count": 100,
            "ood_score": 0.1,
            "lcb_success": 0.8,
        }
        for action in catalog.actions
    ]
    cases = []

    invalid = [dict(row, lcb_success=float("nan")) for row in base]
    cases.append((invalid, "abstain_all_invalid_evidence"))
    low = [dict(row, support_count=19) for row in base]
    cases.append((low, "abstain_all_low_support"))
    ood = [dict(row, ood_score=0.41) for row in base]
    cases.append((ood, "abstain_all_ood"))
    hard = [
        {
            "action_uid": action.action_uid,
            "hard_ok": False,
            "hard_reason": "unsafe",
        }
        for action in catalog.actions
    ]
    cases.append((hard, "abstain_all_hard_rejected"))

    for rows, expected in cases:
        decision = _select(
            catalog, tuple(CandidateEvidence(**row) for row in rows)
        )
        assert decision.abstained
        assert (decision.selected_uid, decision.selected_id, decision.selected_slot) == (
            0,
            "",
            -1,
        )
        assert decision.reason == expected


def test_mixed_failures_have_explicit_no_eligible_abstention():
    catalog = _Catalog(2)
    decision = _select(
        catalog,
        (
            CandidateEvidence(
                catalog.actions[0].action_uid,
                hard_ok=False,
                hard_reason="unsafe",
            ),
            CandidateEvidence(
                catalog.actions[1].action_uid,
                hard_ok=True,
                hard_reason="",
                support_count=0,
                ood_score=0.0,
                lcb_success=1.0,
            ),
        ),
    )
    assert decision.abstained
    assert decision.reason == "abstain_no_eligible_candidate"


def test_best_lcb_below_threshold_abstains_before_priority():
    catalog = _Catalog(2)
    decision = _select(catalog, _candidates(catalog, [0.49, 0.48]))
    assert decision.abstained
    assert decision.reason == "abstain_below_min_lcb"
    assert [row.reason for row in decision.assessments] == [
        "below_min_lcb",
        "below_min_lcb",
    ]


def test_priority_cannot_revive_action_below_minimum_lcb():
    catalog = _Catalog(2)
    above_floor, priority_favorite = catalog.actions
    profile = _profile(
        catalog,
        min_lcb_success=0.5,
        delta_tie=0.2,
        priority_by_uid={
            above_floor.action_uid: 99,
            priority_favorite.action_uid: 0,
        },
    )

    decision = _select(
        catalog,
        _candidates(catalog, [0.60, 0.49]),
        profile=profile,
    )

    assert decision.selected_action_uid == above_floor.action_uid
    assert decision.assessments[1].reason == "below_min_lcb"


def test_statistics_win_outside_priority_tie_bound():
    catalog = _Catalog(2)
    statistical_winner, priority_winner = catalog.actions
    profile = _profile(
        catalog,
        delta_tie=0.05,
        priority_by_uid={
            statistical_winner.action_uid: 99,
            priority_winner.action_uid: 0,
        },
    )
    decision = _select(catalog, _candidates(catalog, [0.90, 0.84]), profile=profile)
    assert decision.selected_action_uid == statistical_winner.action_uid


def test_priority_may_break_tie_inside_bound_but_lcb_breaks_equal_priority():
    catalog = _Catalog(2)
    statistical_winner, priority_winner = catalog.actions
    profile = _profile(
        catalog,
        delta_tie=0.05,
        priority_by_uid={
            statistical_winner.action_uid: 1,
            priority_winner.action_uid: 0,
        },
    )
    decision = _select(catalog, _candidates(catalog, [0.90, 0.86]), profile=profile)
    assert decision.selected_action_uid == priority_winner.action_uid

    equal_priority = _profile(
        catalog,
        delta_tie=0.05,
        priority_by_uid={action.action_uid: 0 for action in catalog.actions},
    )
    decision = _select(
        catalog, _candidates(catalog, [0.90, 0.86]), profile=equal_priority
    )
    assert decision.selected_action_uid == statistical_winner.action_uid


def test_priority_may_break_tie_at_exact_delta_boundary_only():
    catalog = _Catalog(2)
    statistical_winner, priority_winner = catalog.actions
    profile = _profile(
        catalog,
        delta_tie=0.25,
        priority_by_uid={
            statistical_winner.action_uid: 99,
            priority_winner.action_uid: 0,
        },
    )

    boundary = _select(
        catalog,
        _candidates(catalog, [0.75, 0.50]),
        profile=profile,
    )
    outside = _select(
        catalog,
        _candidates(
            catalog,
            [0.75, float.fromhex("0x1.fffffffffffffp-2")],
        ),
        profile=profile,
    )

    assert boundary.selected_action_uid == priority_winner.action_uid
    assert outside.selected_action_uid == statistical_winner.action_uid


def test_uid_is_the_final_stable_tie_breaker():
    catalog = _Catalog(2)
    profile = _profile(
        catalog,
        priority_by_uid={action.action_uid: 0 for action in catalog.actions},
    )
    decision = _select(catalog, _candidates(catalog, [0.8, 0.8]), profile=profile)
    assert decision.selected_action_uid == min(action.action_uid for action in catalog.actions)


def test_priority_map_must_cover_catalog_exactly():
    catalog = _Catalog(2)
    missing = SelectorProfile(
        min_support=1,
        max_ood_score=1.0,
        min_lcb_success=0.0,
        delta_tie=0.0,
        priority_by_uid={catalog.actions[0].action_uid: 0},
    )
    with pytest.raises(CapabilityContractError, match="cover the catalog exactly"):
        _select(catalog, _candidates(catalog), profile=missing)

    extra = SelectorProfile(
        min_support=1,
        max_ood_score=1.0,
        min_lcb_success=0.0,
        delta_tie=0.0,
        priority_by_uid={
            catalog.actions[0].action_uid: 0,
            catalog.actions[1].action_uid: 1,
            999_999: 2,
        },
    )
    with pytest.raises(CapabilityContractError, match="cover the catalog exactly"):
        _select(catalog, _candidates(catalog), profile=extra)


def test_hard_candidate_shape_is_strict():
    uid = _Catalog(1).actions[0].action_uid
    with pytest.raises(CapabilityContractError, match="non-empty hard_reason"):
        CandidateEvidence(uid, hard_ok=False, hard_reason="")
    with pytest.raises(CapabilityContractError, match="must provide"):
        CandidateEvidence(uid, hard_ok=True, hard_reason="", support_count=10)
    with pytest.raises(CapabilityContractError, match="hard_reason must be empty"):
        CandidateEvidence(
            uid,
            hard_ok=True,
            hard_reason="ok",
            support_count=10,
            ood_score=0.1,
            lcb_success=0.9,
        )


def test_candidate_sequence_rejects_duplicate_unknown_reordered_and_missing():
    catalog = _Catalog(2)
    good = _candidates(catalog)

    duplicate = (good[0], CandidateEvidence(
        good[0].action_uid,
        hard_ok=True,
        hard_reason="",
        support_count=100,
        ood_score=0.1,
        lcb_success=0.8,
    ))
    with pytest.raises(CapabilityContractError, match="duplicate"):
        _select(catalog, duplicate)

    unknown = (good[0], CandidateEvidence(
        999_999,
        hard_ok=True,
        hard_reason="",
        support_count=100,
        ood_score=0.1,
        lcb_success=0.8,
    ))
    with pytest.raises(CapabilityContractError, match="unknown"):
        _select(catalog, unknown)

    with pytest.raises(CapabilityContractError, match="slot order"):
        _select(catalog, tuple(reversed(good)))

    with pytest.raises(CapabilityContractError, match="exactly one row"):
        _select(catalog, good[:-1])


def test_candidate_evidence_receipts_are_required_and_match_current_inputs():
    catalog = _Catalog(2)
    artifact = _artifact(catalog)
    profile = _profile(catalog)
    bare = _candidates(catalog)

    with pytest.raises(CapabilityContractError, match=r"candidates\[0\]\.artifact_sha256"):
        select_action(
            catalog,
            artifact,
            profile,
            bare,
            query_sha256=QUERY_SHA256,
            expected_profile_sha256=profile.profile_sha256,
            **CURRENT,
        )

    stale_artifact = (
        replace(bare[0], artifact_sha256=_sha("stale-artifact")),
        bare[1],
    )
    with pytest.raises(CapabilityContractError, match="current capability artifact"):
        _select(catalog, stale_artifact, artifact=artifact, profile=profile)

    stale_query = (
        replace(bare[0], query_sha256=_sha("stale-query")),
        bare[1],
    )
    with pytest.raises(CapabilityContractError, match="current query"):
        _select(catalog, stale_query, artifact=artifact, profile=profile)

    duplicate_receipt = _sha("duplicate-candidate-receipt")
    duplicates = tuple(
        replace(candidate, candidate_sha256=duplicate_receipt)
        for candidate in bare
    )
    with pytest.raises(CapabilityContractError, match="duplicate candidate_sha256"):
        _select(catalog, duplicates, artifact=artifact, profile=profile)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda row: replace(row, lcb_success=0.01),
        lambda row: replace(row, hard_ok=False, hard_reason="unsafe"),
    ],
)
def test_evidence_self_hash_rejects_score_or_hard_gate_tampering(tamper):
    catalog = _Catalog(2)
    artifact = _artifact(catalog)
    first, second = _candidates(catalog)
    bound = first.bind_receipts(
        artifact_sha256=artifact.artifact_sha256,
        query_sha256=QUERY_SHA256,
        candidate_sha256=_candidate_receipt(first.action_uid),
    )

    with pytest.raises(CapabilityContractError, match="evidence_sha256"):
        _select(
            catalog,
            (tamper(bound), second),
            artifact=artifact,
        )


def test_unsupported_diagnostic_uses_stable_type_sentinel_not_repr():
    catalog = _Catalog(2)
    artifact = _artifact(catalog)
    uid = catalog.actions[0].action_uid
    first = CandidateEvidence(
        uid,
        hard_ok=False,
        hard_reason="unsafe",
        support_count=_ExplodingRepr(),
    ).bind_receipts(
        artifact_sha256=artifact.artifact_sha256,
        query_sha256=QUERY_SHA256,
        candidate_sha256=_candidate_receipt(uid),
    )
    second = CandidateEvidence(
        uid,
        hard_ok=False,
        hard_reason="unsafe",
        support_count=_ExplodingRepr(),
    ).bind_receipts(
        artifact_sha256=artifact.artifact_sha256,
        query_sha256=QUERY_SHA256,
        candidate_sha256=_candidate_receipt(uid),
    )

    assert first.evidence_sha256 == second.evidence_sha256
    decision = _select(
        catalog,
        (first, _candidates(catalog)[1]),
        artifact=artifact,
    )
    assert decision.selected_slot == 1


def test_unauthorized_profile_hash_fails_before_selection():
    catalog = _Catalog(2)
    profile = _profile(catalog)
    with pytest.raises(CapabilityContractError, match="authorized profile"):
        _select(
            catalog,
            _candidates(catalog),
            profile=profile,
            expected_profile_sha256=_sha("different-profile"),
        )


def test_decision_identity_must_match_unique_selected_assessment():
    catalog = _Catalog(2)
    decision = _select(catalog, _candidates(catalog))
    with pytest.raises(CapabilityContractError, match="selected decision identity"):
        replace(
            decision,
            selected_action_uid=999_999,
            selected_action_id="not_in_assessments",
            selected_slot=99,
        )

    selected_index = decision.selected_slot
    selected = decision.assessments[selected_index]
    with pytest.raises(CapabilityContractError, match="empty hard_reason"):
        replace(selected, hard_reason="table_collision")
    with pytest.raises(CapabilityContractError, match="valid numeric evidence"):
        replace(
            selected,
            support_count=-99,
            ood_score=2.0,
            lcb_success=-1.0,
        )

    forged = replace(
        selected,
        action_uid=999_999,
        action_id="forged",
    )
    assessments = list(decision.assessments)
    assessments[selected_index] = forged
    with pytest.raises(CapabilityContractError, match="evidence receipt"):
        replace(
            decision,
            selected_action_uid=forged.action_uid,
            selected_action_id=forged.action_id,
            assessments=tuple(assessments),
        )

    hard = tuple(
        CandidateEvidence(
            action.action_uid,
            hard_ok=False,
            hard_reason="unsafe",
        )
        for action in catalog.actions
    )
    abstention = _select(catalog, hard)
    with pytest.raises(CapabilityContractError, match="abstention reason"):
        replace(abstention, reason="abstain_all_ood")


def test_select_action_checks_artifact_before_scoring():
    catalog = _Catalog(2)
    with pytest.raises(CapabilityContractError, match="task_sha256"):
        _select(catalog, _candidates(catalog), task_sha256=_sha("stale-task"))
