#!/usr/bin/env python3
"""Strict loader for canonical motion marker authorities v2 and v3.

The v2 authority keeps six concepts separate:

* an ordinary nominal air-swing event;
* legacy ge50 and ge80 stationary-scan seeds;
* a nullable legacy preferred hint;
* a synthetic construction annotation;
* an explicit historical ``adv2c3`` comparator frame; and
* the mandatory post-retime behavior-rescan gate.

Ordinary event mappings are bound to exact frame-identity receipts.  A receipt
may prove full-frame identity modulo documented SE(2), or only temporal-index
lineage when grip bake changed the right wrist.  Neither classification creates
ball-contact truth or behavior authority.  Synthetic construction evidence can
never create a nominal or preferred behavior marker.

The v1 authority remains an immutable historical input.  Callers must
externally pin the exact v2 authority SHA-256.

Authority v3 (2026-07-27) exists for one reason: a recording made after
2026-07-10 has no v1 record and can never acquire one, so it cannot carry a v1
air-swing nominal event, and it must not pretend to.  v3 therefore adds

* an ``event_provenance`` discriminated union per row, whose third arm,
  ``derived_tool_scanned_contact``, records a frame a named tool scanned
  directly on THE BYTES THIS ROW BINDS -- closed on
  ``scan_artifact.input_source_sha256 == bound_recipe_source.sha256``, with no
  frame-identity receipt, because nothing is being mapped across clips;
* an optional ``derived_seed`` beside the legacy ge50/ge80 seeds; and
* an appendable row list whose ORDERED FIRST FIVE rows must still be the
  canonical five, byte-for-byte re-derivable from the v2 authority.

Two rules make v3 an honest widening rather than a hole:

* the profile is chosen by the CALLER, never by the document -- a file whose
  own ``authority_id`` disagrees with the caller's profile is rejected; and
* the fabrication guard (``_require_v1_content_agreement``): any row claiming
  v1 air-swing lineage is checked against the v1 authority's own bytes, not
  merely against the v1 SHA-256 string.  The v1 row must exist, its
  ``bound_recipe_source.sha256`` must equal this row's, and its nominal event
  frame must equal this row's.  Re-pointing a v1 receipt at different source
  bytes, or minting a v1 lineage for a recording v1 never saw, now fails.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class MarkerSemanticsError(ValueError):
    """The marker authority or one of its content bindings is invalid."""


CANONICAL_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)
SYNTHETIC_MOTION_ID = "fh_block_syn"
MARKER_AUTHORITY_SCHEMA_VERSION = 2
MARKER_AUTHORITY_ID = "canonical_motion_marker_semantics_v2_20260724"
MARKER_AUTHORITY_PATH = "configs/canonical_motion_marker_semantics_v2_20260724.json"
MARKER_AUTHORITY_REVIEW_STATUS = (
    "REVIEWED_FOR_CONTENT_ADDRESSED_FRAME_MAPPING_AND_LEGACY_SEED_PROVENANCE"
)
MARKER_AUTHORITY_SHA256 = (
    "0bf36c204162bf63e044db129bdb869129f36fb7a6d191a7c83effc4ba300929"
)
MARKER_AUTHORITY_V3_SCHEMA_VERSION = 3
MARKER_AUTHORITY_V3_ID = "canonical_motion_marker_semantics_v3_20260727"
MARKER_AUTHORITY_V3_PATH = "configs/canonical_motion_marker_semantics_v3_20260727.json"
MARKER_AUTHORITY_V3_REVIEW_STATUS = (
    "REVIEWED_FOR_VERBATIM_CANONICAL_PREFIX_AND_TOOL_DERIVED_CONTACT_PROVENANCE"
)
MARKER_AUTHORITY_V3APPEND_V12_PATH = (
    "configs/canonical_motion_marker_semantics_v3append_v12_20260729.json"
)
MARKER_AUTHORITY_V3APPEND_V12_MOTION_IDS = (
    "fh_loop_high",
    "v12_forehand_block",
)
LEGACY_AUTHORITY_PATH = "configs/canonical_motion_marker_semantics_v1_20260724.json"
LEGACY_AUTHORITY_SHA256 = (
    "186fa064c24ada5e7c36530e67096b6b8027b92159a7458f77d0517593e54252"
)

#: ``event_provenance.kind`` discriminants.  These are three DIFFERENT claims,
#: not three spellings of one claim.
EVENT_KIND_LEGACY_V1 = "legacy_v1_air_swing_event"
EVENT_KIND_DONOR_CONSTRUCTION = "donor_construction_annotation"
EVENT_KIND_DERIVED_CONTACT = "derived_tool_scanned_contact"

_DERIVED_CONTACT_SEMANTIC_KIND = "derived_kinematic_carrier_speed_peak_frame"
_DERIVED_FRAME_IDENTITY_REASON = (
    "SCANNED_DIRECTLY_ON_BOUND_SOURCE_NO_CROSS_CLIP_MAPPING"
)
_DERIVED_V1_LINEAGE_REASON = "NO_V1_RECORD_EXISTS_FOR_THIS_RECORDING"
_ALLOWED_DERIVED_RULE_IDS = frozenset(
    {
        # returnability / face-manifold scan lineage (the four legacy rows)
        "longest_contiguous_run_at_or_above_score_threshold",
        # kinematic carrier-speed scan run directly on the bound npz bytes
        "max_carrier_speed_at_or_above_min_height",
    }
)
_DERIVED_RETIMING_STAGES = frozenset({"source_timing", "post_retime_compiled_clip"})
TEMPORAL_INDEX_CLASSIFICATION = (
    "TEMPORAL_INDEX_LINEAGE_ONLY_NOT_POSE_OR_EVENT_SEMANTIC_TRANSFER"
)
FULL_FRAME_CLASSIFICATION = "FULL_FRAME_IDENTITY_MODULO_DOCUMENTED_SE2"

_SHA256_LENGTH = 64
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "authority_id",
        "review_status",
        "scope",
        "supersedes",
        "authorization",
        "semantic_contract",
        "shared_evidence",
        "motions",
    }
)
_SUPERSEDES_KEYS = frozenset({"path", "sha256", "role"})
_AUTHORIZATION_KEYS = frozenset(
    {
        "training_authorized",
        "behavior_authorized",
        "hardware_authorized",
        "artifact_promotion_authorized",
    }
)
_SEMANTIC_CONTRACT_KEYS = frozenset(
    {
        "nominal_event",
        "frame_identity_receipt",
        "legacy_ge50_seed",
        "legacy_ge80_seed",
        "legacy_preferred_seed",
        "construction_marker",
        "historical_adv2c3_comparator",
        "contact_truth_observed",
        "synthetic_behavior_from_donor_or_construction_forbidden",
        "automatic_event_to_window_aliasing_forbidden",
        "automatic_historical_adv2c3_recomputation_forbidden",
        "legacy_seed_is_certified_post_retime_window",
        "post_retime_behavior_rescan_required",
        "behavior_promotion_gate",
        "unresolved_or_mismatched_source_binding_policy",
    }
)
_SHARED_EVIDENCE_KEYS = frozenset(
    {
        "frame_identity_receipts_summary",
        "legacy_timing_table",
        "legacy_timing_generator",
        "historical_adv2c3_builder",
        "historical_adv2c3_manifest",
        "ordinary_scan_job_manifest",
        "s0_highpress_scan_job_manifest",
    }
)
_LOCAL_REF_KEYS = frozenset({"path", "sha256"})
_REMOTE_REF_KEYS = frozenset({"remote_path", "sha256"})
_TIMING_REF_KEYS = frozenset({"remote_path", "sha256", "legacy_label_warning"})
_ROW_KEYS = frozenset(
    {
        "motion_id",
        "bound_recipe_source",
        "nominal_event",
        "legacy_scan_evidence",
        "legacy_ge50_seed",
        "legacy_ge80_seed",
        "legacy_preferred_seed",
        "construction_marker",
        "historical_adv2c3_comparator",
        "post_retime_behavior_gate",
    }
)
_BOUND_SOURCE_KEYS = frozenset({"path", "sha256"})
_NOMINAL_EVENT_KEYS = frozenset(
    {
        "frame",
        "semantic_kind",
        "contact_truth_observed",
        "behavior_authorized",
        "legacy_evidence_json_pointer",
        "event_source_remote_path",
        "event_source_sha256",
        "mapping_to_bound_recipe_source",
        "frame_identity_receipt",
    }
)
_FRAME_RECEIPT_REF_KEYS = frozenset(
    {
        "path",
        "sha256",
        "receipt_id",
        "classification",
        "event_source_frame",
        "bound_source_frame",
        "event_source_sha256",
        "bound_source_sha256",
        "invariant_joint_count",
        "excluded_joint_indices",
        "full_pose_identity_claimed",
        "authority_v1_sha256",
    }
)
_SCAN_KEYS = frozenset(
    {
        "remote_path",
        "sha256",
        "input_source_remote_path",
        "input_source_sha256",
        "job_manifest_ref",
        "job_id",
        "scan_contract",
    }
)
_ORDINARY_SCAN_CONTRACT_KEYS = frozenset(
    {
        "stroke_side",
        "cli_overrides",
        "model",
        "span_rule",
        "default_ball_distribution",
    }
)
_S0_SCAN_CONTRACT_KEYS = frozenset(
    {"stroke_side", "cli_overrides", "model", "span_rule"}
)
_SYNTHETIC_SCAN_CONTRACT_KEYS = frozenset(
    {
        "stroke_side",
        "cli_overrides",
        "model",
        "span_rule",
        "transfer_to_synthetic_forehand_behavior",
    }
)
_GE_SEED_KEYS = frozenset(
    {
        "span_inclusive",
        "threshold_fraction",
        "seed_role",
        "source_scan_ref",
        "legacy_stationary_scan_only",
        "certified_post_retime_window",
    }
)
_PREFERRED_KEYS = frozenset(
    {
        "frame",
        "basis",
        "source_scan_ref",
        "behavior_authorized",
        "automatic_build_anchor_authorized",
    }
)
_CONSTRUCTION_KEYS = frozenset(
    {
        "annotation_frame",
        "annotation_semantics",
        "donor_motion_id",
        "donor_preferred_frame",
        "solve_span_inclusive",
        "upper_receipt",
        "upper_scoped_window",
        "full_receipt",
        "full_scoped_window",
        "behavior_nominal_created",
        "behavior_preferred_created",
        "transfer_to_synthetic_forehand_behavior",
    }
)
_HISTORICAL_KEYS = frozenset(
    {
        "frame",
        "recorded_basis",
        "source_authority_path",
        "source_authority_sha256",
        "source_json_pointer",
        "adopted",
        "recompute_forbidden",
        "role",
    }
)
_POST_RETIME_GATE_KEYS = frozenset(
    {
        "required",
        "status",
        "required_scopes",
        "legacy_seed_is_certified_window",
        "behavior_promotion_authorized",
    }
)
#: v3 only.  ``rescan`` is the slot the authority's own gate always implied but
#: never provided: somewhere to put the post-retime behavior rescan RESULT.  It
#: is defined here and accepted only as ``null`` -- see ``_validate_gate_rescan``.
_V3_POST_RETIME_GATE_KEYS = _POST_RETIME_GATE_KEYS | {"rescan"}
_V3_ROW_KEYS = frozenset(
    {
        "motion_id",
        "bound_recipe_source",
        "event_provenance",
        "legacy_ge50_seed",
        "legacy_ge80_seed",
        "legacy_preferred_seed",
        "derived_seed",
        "historical_adv2c3_comparator",
        "post_retime_behavior_gate",
    }
)
_V3_LEGACY_PROVENANCE_KEYS = frozenset(
    {"kind", "nominal_event", "legacy_scan_evidence", "construction_marker"}
)
_V3_DERIVED_PROVENANCE_KEYS = frozenset({"kind", "derived_contact"})
_DERIVED_CONTACT_KEYS = frozenset(
    {
        "frame",
        "semantic_kind",
        "contact_truth_observed",
        "behavior_authorized",
        "returnability_certified",
        "retiming_stage",
        "frame_identity",
        "v1_lineage",
        "derivation",
        "scan_artifact",
    }
)
_DERIVED_SEED_KEYS = frozenset(
    {
        "span_inclusive",
        "argmax_frame",
        "seed_role",
        "retiming_stage",
        "certified_post_retime_window",
        "derivation",
        "scan_artifact",
    }
)
_DERIVATION_KEYS = frozenset({"tool", "tool_sha256", "cli_argv", "rule_id", "thresholds"})
_SCAN_ARTIFACT_KEYS = frozenset({"path", "sha256", "input_source_sha256"})
_NOT_CLAIMED_KEYS = frozenset({"claimed", "reason"})
_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "summary_id",
        "publication_class",
        "authorization",
        "marker_authority_binding",
        "builder",
        "verification",
        "joint_contract",
        "receipts",
        "synthetic_face_construction",
        "non_claims",
    }
)
_SUMMARY_RECEIPT_KEYS = frozenset(
    {
        "motion_id",
        "event_source_frame",
        "authority_frame_json_pointer",
        "receipt_path",
        "receipt_sha256",
        "build_spec_sha256",
        "classification",
        "invariant_joint_count",
        "excluded_joint_indices",
        "verification",
    }
)

_EXPECTED_SEED_ROLES = {
    "fh_loop": "legacy_behavior_seed_requires_post_retime_rescan",
    "bh_loop_c": "legacy_behavior_seed_requires_post_retime_rescan",
    "fh_block_syn": "BACKHAND_CONSTRUCTION_SEED_ONLY_MANDATORY_FOREHAND_RESCAN",
    "bh_block": "legacy_behavior_seed_requires_post_retime_rescan",
    "s0_highpress": (
        "legacy_specialized_high_ball_seed_requires_post_retime_direction_rescan"
    ),
}
_EXPECTED_STROKE_SIDES = {
    "fh_loop": "forehand",
    "bh_loop_c": "backhand",
    "fh_block_syn": "backhand",
    "bh_block": "backhand",
    "s0_highpress": "backhand_proxy",
}
_EXPECTED_JOB_MANIFEST_REFS = {
    "fh_loop": "ordinary_scan_job_manifest",
    "bh_loop_c": "ordinary_scan_job_manifest",
    "fh_block_syn": "ordinary_scan_job_manifest",
    "bh_block": "ordinary_scan_job_manifest",
    "s0_highpress": "s0_highpress_scan_job_manifest",
}


@dataclass(frozen=True)
class AuthorityProfile:
    """What the CALLER expects an authority file to be.

    The profile is never read out of the document being validated.  A document
    that declares an ``authority_id`` other than the one the caller asked for is
    an error, not a request to switch validation regimes.
    """

    name: str
    schema_version: int
    authority_id: str
    path: str
    review_status: str
    appendable_rows: bool
    exact_appended_motion_ids: tuple[str, ...] | None


MARKER_AUTHORITY_PROFILES: Mapping[str, AuthorityProfile] = MappingProxyType(
    {
        "v2": AuthorityProfile(
            name="v2",
            schema_version=MARKER_AUTHORITY_SCHEMA_VERSION,
            authority_id=MARKER_AUTHORITY_ID,
            path=MARKER_AUTHORITY_PATH,
            review_status=MARKER_AUTHORITY_REVIEW_STATUS,
            appendable_rows=False,
            exact_appended_motion_ids=None,
        ),
        "v3": AuthorityProfile(
            name="v3",
            schema_version=MARKER_AUTHORITY_V3_SCHEMA_VERSION,
            authority_id=MARKER_AUTHORITY_V3_ID,
            path=MARKER_AUTHORITY_V3_PATH,
            review_status=MARKER_AUTHORITY_V3_REVIEW_STATUS,
            appendable_rows=True,
            exact_appended_motion_ids=None,
        ),
        "v3append_v12": AuthorityProfile(
            name="v3append_v12",
            schema_version=MARKER_AUTHORITY_V3_SCHEMA_VERSION,
            authority_id=MARKER_AUTHORITY_V3_ID,
            path=MARKER_AUTHORITY_V3APPEND_V12_PATH,
            review_status=MARKER_AUTHORITY_V3_REVIEW_STATUS,
            appendable_rows=True,
            exact_appended_motion_ids=MARKER_AUTHORITY_V3APPEND_V12_MOTION_IDS,
        ),
    }
)
#: Reverse table used by callers (the recipe loader) that only know a path.
MARKER_AUTHORITY_PROFILE_BY_PATH: Mapping[str, str] = MappingProxyType(
    {profile.path: name for name, profile in MARKER_AUTHORITY_PROFILES.items()}
)


@dataclass(frozen=True)
class DerivedProvenance:
    """Who computed this, on which exact bytes, with which written-down rule."""

    tool: str
    tool_sha256: str
    rule_id: str
    cli_argv: tuple[str, ...]
    scan_artifact_path: str
    scan_artifact_sha256: str
    scan_input_source_sha256: str
    retiming_stage: str


@dataclass(frozen=True)
class DerivedContact:
    """A contact-candidate frame scanned on the row's own bound source bytes.

    Not observed ball contact, not returnability-certified, not a behavior
    authorization, and explicitly not a v1 air-swing event.
    """

    frame: int
    semantic_kind: str
    returnability_certified: bool
    provenance: DerivedProvenance


@dataclass(frozen=True)
class DerivedSeed:
    """A tool-derived search window, the derived counterpart of ge50/ge80."""

    span: tuple[int, int]
    argmax_frame: int
    seed_role: str
    certified_post_retime_window: bool
    provenance: DerivedProvenance


@dataclass(frozen=True)
class FrameIdentityBinding:
    path: str
    sha256: str
    receipt_id: str
    classification: str
    invariant_joint_count: int
    excluded_joint_indices: tuple[int, ...]
    full_pose_identity_claimed: bool


@dataclass(frozen=True)
class ConstructionMarker:
    annotation_frame: int
    donor_preferred_frame: int
    solve_span: tuple[int, int]


@dataclass(frozen=True)
class MarkerSemanticsRow:
    """Separated, non-authorizing marker data for one source motion."""

    motion_id: str
    nominal_event: int | None
    ge50_seed: tuple[int, int] | None
    ge80_seed: tuple[int, int] | None
    preferred_seed: int | None
    construction_marker: ConstructionMarker | None
    historical_adv2c3_start: int | None
    bound_recipe_source_path: str
    bound_recipe_source_sha256: str
    source_scan_remote_path: str | None
    source_scan_sha256: str | None
    frame_identity: FrameIdentityBinding | None
    post_retime_behavior_gate_status: str
    # v3 additions.  Every v2 row keeps the v2 defaults, so a v2 authority
    # produces exactly the rows it produced before this field existed.
    event_provenance_kind: str = EVENT_KIND_LEGACY_V1
    derived_contact: DerivedContact | None = None
    derived_seed: DerivedSeed | None = None
    post_retime_rescan: None = None

    # -- the single anchor/window resolution law -------------------------
    # Both the compiler (_authority_markers) and the bank gate hand-wrote
    # their own if/elif over nominal_event / construction_marker.  Two copies
    # of a rule drift; this is the one copy.

    def contact_anchor(self) -> tuple[int | None, str]:
        """Return ``(frame, basis)`` for the row's contact anchor.

        Priority, fixed and explicit:

        1. a PROMOTED post-retime derived contact -- not reachable today, see
           ``post_retime_rescan``;
        2. a source-timing tool-derived contact;
        3. a legacy v1 air-swing nominal event;
        4. a donor construction annotation.

        Promotion is a per-row explicit act, never "take the newest".  In
        particular ``fh_loop``'s frame 61 does not move because this method
        exists.
        """

        if (
            self.derived_contact is not None
            and self.derived_contact.provenance.retiming_stage
            == "post_retime_compiled_clip"
        ):
            return self.derived_contact.frame, "promoted_post_retime_derived_contact"
        if self.derived_contact is not None:
            return self.derived_contact.frame, "derived_source_timing_contact"
        if self.nominal_event is not None:
            return self.nominal_event, EVENT_KIND_LEGACY_V1
        if self.construction_marker is not None:
            return (
                self.construction_marker.annotation_frame,
                EVENT_KIND_DONOR_CONSTRUCTION,
            )
        return None, "none"

    def search_window(self) -> tuple[tuple[int, int] | None, str]:
        """Return ``(span_inclusive, basis)`` for the compiler's search window."""

        if self.derived_seed is not None:
            return self.derived_seed.span, "derived_seed"
        if self.ge80_seed is not None:
            return self.ge80_seed, "legacy_ge80_seed"
        return None, "none"

    def authority_frames(self) -> tuple[int, ...]:
        """Every source-frame index this row asserts, for range checking."""

        frames: list[int] = []
        for span in (self.ge50_seed, self.ge80_seed):
            if span is not None:
                frames.extend(span)
        for scalar in (
            self.historical_adv2c3_start,
            self.nominal_event,
            self.preferred_seed,
        ):
            if scalar is not None:
                frames.append(scalar)
        if self.construction_marker is not None:
            frames.extend(
                (
                    self.construction_marker.annotation_frame,
                    self.construction_marker.donor_preferred_frame,
                    *self.construction_marker.solve_span,
                )
            )
        if self.derived_contact is not None:
            frames.append(self.derived_contact.frame)
        if self.derived_seed is not None:
            frames.extend((*self.derived_seed.span, self.derived_seed.argmax_frame))
        return tuple(frames)


@dataclass(frozen=True)
class MarkerSemantics:
    """Immutable, externally SHA-pinned v2 marker authority."""

    path: Path
    repo_root: Path
    sha256: str
    authority_id: str
    review_status: str
    legacy_authority_sha256: str
    rows: tuple[MarkerSemanticsRow, ...]
    profile: str = "v2"
    #: v3 only: the SHA-256 of the v2 authority whose five rows this file
    #: reproduces verbatim.  Recomputed from bytes at load time, not trusted.
    verbatim_prefix_authority_sha256: str | None = None

    def row(self, motion_id: str) -> MarkerSemanticsRow:
        matches = tuple(row for row in self.rows if row.motion_id == motion_id)
        if len(matches) != 1:
            raise MarkerSemanticsError(
                f"authority has {len(matches)} rows named {motion_id!r}"
            )
        return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MarkerSemanticsError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise MarkerSemanticsError(f"{label} keys must be strings")
    actual = frozenset(value)
    if actual != expected:
        raise MarkerSemanticsError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MarkerSemanticsError(
            f"{label} must be a non-empty string without edge whitespace"
        )
    return value


def _exact_string(value: Any, expected: str, label: str) -> str:
    text = _string(value, label)
    if text != expected:
        raise MarkerSemanticsError(f"{label} must equal {expected!r}")
    return text


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if (
        len(text) != _SHA256_LENGTH
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise MarkerSemanticsError(f"{label} must be a 64-character lowercase SHA-256")
    return text


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MarkerSemanticsError(f"{label} must be an integer >= {minimum}")
    return value


def _exact_bool(value: Any, expected: bool, label: str) -> bool:
    if not isinstance(value, bool) or value is not expected:
        raise MarkerSemanticsError(f"{label} must be exactly {expected}")
    return value


def _exact_number(value: Any, expected: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MarkerSemanticsError(f"{label} must be numeric")
    result = float(value)
    if result != expected:
        raise MarkerSemanticsError(f"{label} must equal {expected}")
    return result


def _inclusive_span(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise MarkerSemanticsError(f"{label} must be a two-integer inclusive span")
    start = _integer(value[0], f"{label}[0]")
    end = _integer(value[1], f"{label}[1]")
    if end < start:
        raise MarkerSemanticsError(f"{label} end precedes start")
    return start, end


def _repo_relative_path(value: Any, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise MarkerSemanticsError(f"{label} must be a normalized relative path")
    return text


def _repo_file(repo_root: Path, value: Any, label: str) -> tuple[str, Path]:
    text = _repo_relative_path(value, label)
    path = (repo_root / text).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise MarkerSemanticsError(f"{label} escapes repo_root") from exc
    if not path.is_file():
        raise MarkerSemanticsError(f"{label} does not exist: {path}")
    return text, path


def _remote_path(value: Any, label: str) -> str:
    text = _string(value, label)
    if not (text.startswith("pod2:/") or text.startswith("pod3:/")):
        raise MarkerSemanticsError(f"{label} must be an explicit pod2:/ or pod3:/ path")
    return text


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise MarkerSemanticsError(f"{label} must be a JSON string array")
    return tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MarkerSemanticsError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MarkerSemanticsError(
                    f"{label} contains non-finite JSON number {token!r}"
                )
            ),
        )
    except MarkerSemanticsError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MarkerSemanticsError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MarkerSemanticsError(f"{label} must contain one JSON object")
    return value


def _read_bound_json(
    repo_root: Path, reference: Any, label: str
) -> tuple[str, str, Mapping[str, Any]]:
    ref = _exact_keys(reference, _LOCAL_REF_KEYS, label)
    path_text, path = _repo_file(repo_root, ref["path"], f"{label}.path")
    expected = _sha256(ref["sha256"], f"{label}.sha256")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise MarkerSemanticsError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path_text, actual, _strict_json_bytes(payload, label)


def _validate_bound_file(
    repo_root: Path, reference: Any, label: str
) -> tuple[str, str]:
    ref = _exact_keys(reference, _LOCAL_REF_KEYS, label)
    path_text, path = _repo_file(repo_root, ref["path"], f"{label}.path")
    expected = _sha256(ref["sha256"], f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise MarkerSemanticsError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path_text, actual


def _validate_authorization(raw: Any, label: str) -> None:
    value = _exact_keys(raw, _AUTHORIZATION_KEYS, label)
    for key in sorted(_AUTHORIZATION_KEYS):
        _exact_bool(value[key], False, f"{label}.{key}")


def _validate_semantic_contract(raw: Any) -> None:
    value = _exact_keys(raw, _SEMANTIC_CONTRACT_KEYS, "semantic_contract")
    for key in (
        "nominal_event",
        "frame_identity_receipt",
        "legacy_ge50_seed",
        "legacy_ge80_seed",
        "legacy_preferred_seed",
        "construction_marker",
        "historical_adv2c3_comparator",
    ):
        _string(value[key], f"semantic_contract.{key}")
    _exact_bool(
        value["contact_truth_observed"],
        False,
        "semantic_contract.contact_truth_observed",
    )
    for key in (
        "synthetic_behavior_from_donor_or_construction_forbidden",
        "automatic_event_to_window_aliasing_forbidden",
        "automatic_historical_adv2c3_recomputation_forbidden",
        "post_retime_behavior_rescan_required",
    ):
        _exact_bool(value[key], True, f"semantic_contract.{key}")
    _exact_bool(
        value["legacy_seed_is_certified_post_retime_window"],
        False,
        "semantic_contract.legacy_seed_is_certified_post_retime_window",
    )
    _exact_string(
        value["behavior_promotion_gate"],
        "post_retime_behavior_rescan_per_motion_scope",
        "semantic_contract.behavior_promotion_gate",
    )
    _exact_string(
        value["unresolved_or_mismatched_source_binding_policy"],
        "fail_closed",
        "semantic_contract.unresolved_or_mismatched_source_binding_policy",
    )


def _validate_summary(
    raw: Mapping[str, Any], expected_sha256: str
) -> Mapping[str, Mapping[str, Any]]:
    summary = _exact_keys(raw, _SUMMARY_KEYS, "frame identity summary")
    if _integer(summary["schema_version"], "summary.schema_version") != 1:
        raise MarkerSemanticsError("frame identity summary schema must equal 1")
    _exact_string(
        summary["summary_id"],
        "ordinary_frame_identity_receipts_v1",
        "summary.summary_id",
    )
    _exact_string(
        summary["publication_class"],
        "evidence_only_not_artifact_authorization",
        "summary.publication_class",
    )
    _validate_authorization(summary["authorization"], "summary.authorization")
    marker_binding = summary["marker_authority_binding"]
    if not isinstance(marker_binding, Mapping):
        raise MarkerSemanticsError("summary.marker_authority_binding must be an object")
    if (
        _sha256(
            marker_binding.get("sha256"),
            "summary.marker_authority_binding.sha256",
        )
        != LEGACY_AUTHORITY_SHA256
    ):
        raise MarkerSemanticsError(
            "frame identity summary must bind the immutable v1 authority"
        )
    receipts = summary["receipts"]
    if not isinstance(receipts, list) or len(receipts) != 4:
        raise MarkerSemanticsError(
            "frame identity summary must contain four ordinary receipts"
        )
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(receipts):
        row = _exact_keys(raw_row, _SUMMARY_RECEIPT_KEYS, f"summary.receipts[{index}]")
        motion_id = _string(row["motion_id"], f"summary.receipts[{index}].motion_id")
        if motion_id in indexed or motion_id == SYNTHETIC_MOTION_ID:
            raise MarkerSemanticsError(
                "frame identity summary receipt motion IDs are invalid"
            )
        _sha256(row["receipt_sha256"], f"summary.receipts[{index}].sha256")
        _sha256(
            row["build_spec_sha256"],
            f"summary.receipts[{index}].build_spec_sha256",
        )
        _integer(
            row["event_source_frame"],
            f"summary.receipts[{index}].event_source_frame",
        )
        _repo_relative_path(
            row["receipt_path"],
            f"summary.receipts[{index}].receipt_path",
        )
        _string(
            row["authority_frame_json_pointer"],
            f"summary.receipts[{index}].authority_frame_json_pointer",
        )
        _exact_string(
            row["verification"],
            "PASS_REBUILT_EXACTLY",
            f"summary.receipts[{index}].verification",
        )
        indexed[motion_id] = row
    expected_ids = set(CANONICAL_MOTION_IDS) - {SYNTHETIC_MOTION_ID}
    if set(indexed) != expected_ids:
        raise MarkerSemanticsError(
            "frame identity summary ordinary motion membership changed"
        )
    if not expected_sha256:
        raise MarkerSemanticsError("summary SHA binding is empty")
    return MappingProxyType(indexed)


def _validate_shared_evidence(
    raw: Any, repo_root: Path
) -> Mapping[str, Mapping[str, Any]]:
    value = _exact_keys(raw, _SHARED_EVIDENCE_KEYS, "shared_evidence")
    _, summary_sha, summary = _read_bound_json(
        repo_root,
        value["frame_identity_receipts_summary"],
        "shared_evidence.frame_identity_receipts_summary",
    )
    summary_rows = _validate_summary(summary, summary_sha)
    for evidence_id in sorted(
        _SHARED_EVIDENCE_KEYS - {"frame_identity_receipts_summary"}
    ):
        expected_keys = (
            _TIMING_REF_KEYS
            if evidence_id == "legacy_timing_table"
            else _REMOTE_REF_KEYS
        )
        evidence = _exact_keys(
            value[evidence_id],
            expected_keys,
            f"shared_evidence.{evidence_id}",
        )
        _remote_path(
            evidence["remote_path"],
            f"shared_evidence.{evidence_id}.remote_path",
        )
        _sha256(evidence["sha256"], f"shared_evidence.{evidence_id}.sha256")
        if evidence_id == "legacy_timing_table":
            _string(
                evidence["legacy_label_warning"],
                "shared_evidence.legacy_timing_table.legacy_label_warning",
            )
    return summary_rows


def _direct_legacy_historical_starts(
    legacy_authority: Mapping[str, Any],
) -> Mapping[str, int]:
    motions = legacy_authority.get("motions")
    if not isinstance(motions, list) or len(motions) != len(CANONICAL_MOTION_IDS):
        raise MarkerSemanticsError(
            "immutable v1 authority historical motion list is invalid"
        )
    direct: dict[str, int] = {}
    for index, expected_motion_id in enumerate(CANONICAL_MOTION_IDS):
        row = motions[index]
        if not isinstance(row, Mapping) or row.get("motion_id") != expected_motion_id:
            raise MarkerSemanticsError(
                "immutable v1 authority historical motion order changed"
            )
        comparator = row.get("historical_adv2c3_start")
        if not isinstance(comparator, Mapping):
            raise MarkerSemanticsError(
                f"immutable v1 {expected_motion_id} historical record is invalid"
            )
        direct[expected_motion_id] = _integer(
            comparator.get("frame"),
            f"immutable v1 {expected_motion_id} historical frame",
        )
    return MappingProxyType(direct)


def _legacy_rows_by_motion_id(
    legacy_authority: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Index the immutable v1 authority's own rows by motion id.

    ``_direct_legacy_historical_starts`` has already proved the list is the
    canonical five in canonical order, so this only has to key them.
    """

    motions = legacy_authority["motions"]
    return MappingProxyType(
        {
            motion_id: motions[index]
            for index, motion_id in enumerate(CANONICAL_MOTION_IDS)
        }
    )


def _require_v1_content_agreement(
    legacy_rows: Mapping[str, Mapping[str, Any]],
    *,
    motion_id: str,
    bound_sha256: str,
    nominal_frame: int | None,
    label: str,
) -> None:
    """FABRICATION GUARD -- do not delete without reading this.

    ``_validate_frame_receipt`` compares two 64-character strings against the
    v1 authority SHA-256 and calls that "binding the immutable v1 authority".
    It never opens v1.  That inverted the honesty incentive: a row could bind
    ANY npz bytes, ship a freshly generated receipt and summary that agree with
    each other about those bytes, keep the v1 SHA string, and pass -- which is
    exactly how a 98-frame 2026-07-27 take could inherit a 116-frame
    2026-07-10 take's frame-61 marker.

    This function does what those string comparisons only pretended to do: it
    reads v1's own content at the pinned SHA and requires this row to agree
    with it on (a) existence, (b) the bound source bytes, and (c) the nominal
    event frame.

    Side effect, stated on the record: this permanently freezes the five
    canonical rows to the npz files v1 recorded for them.  Re-aiming an
    existing canonical source is no longer possible -- a new recording must
    arrive as a NEW row with ``derived_tool_scanned_contact`` provenance.  That
    is the intent, not an accident.
    """

    legacy_row = legacy_rows.get(motion_id)
    if legacy_row is None:
        raise MarkerSemanticsError(
            f"{label} claims v1 air-swing lineage, but the immutable v1 authority "
            f"has no record for {motion_id!r}.  v1 is frozen at 2026-07-10; a "
            "recording it never saw cannot acquire a v1 lineage.  Use "
            f"event_provenance.kind={EVENT_KIND_DERIVED_CONTACT!r} instead."
        )
    legacy_bound = legacy_row.get("bound_recipe_source")
    if not isinstance(legacy_bound, Mapping):
        raise MarkerSemanticsError(
            f"{label} immutable v1 record has no bound_recipe_source"
        )
    if legacy_bound.get("sha256") != bound_sha256:
        raise MarkerSemanticsError(
            f"{label} bound_recipe_source bytes disagree with the immutable v1 "
            f"record ({legacy_bound.get('sha256')!r} in v1, {bound_sha256!r} here). "
            "A v1 frame-identity receipt may not be re-pointed at different "
            "source bytes."
        )
    if nominal_frame is None:
        return
    legacy_event = legacy_row.get("nominal_event")
    if not isinstance(legacy_event, Mapping):
        raise MarkerSemanticsError(
            f"{label} claims a v1 nominal event that the v1 record does not have"
        )
    if legacy_event.get("frame") != nominal_frame:
        raise MarkerSemanticsError(
            f"{label} nominal event frame disagrees with the immutable v1 record "
            f"({legacy_event.get('frame')!r} in v1, {nominal_frame!r} here)"
        )


def _validate_scan(
    raw: Any,
    *,
    motion_id: str,
    bound_sha256: str,
) -> tuple[str, str]:
    label = f"motions[{motion_id}].legacy_scan_evidence"
    value = _exact_keys(raw, _SCAN_KEYS, label)
    remote_path = _remote_path(value["remote_path"], f"{label}.remote_path")
    scan_sha = _sha256(value["sha256"], f"{label}.sha256")
    _remote_path(
        value["input_source_remote_path"],
        f"{label}.input_source_remote_path",
    )
    input_sha = _sha256(value["input_source_sha256"], f"{label}.input_source_sha256")
    if input_sha != bound_sha256:
        raise MarkerSemanticsError(
            f"{label} input source must bind bound_recipe_source"
        )
    _exact_string(
        value["job_manifest_ref"],
        _EXPECTED_JOB_MANIFEST_REFS[motion_id],
        f"{label}.job_manifest_ref",
    )
    _string(value["job_id"], f"{label}.job_id")
    if motion_id == SYNTHETIC_MOTION_ID:
        keys = _SYNTHETIC_SCAN_CONTRACT_KEYS
    elif motion_id == "s0_highpress":
        keys = _S0_SCAN_CONTRACT_KEYS
    else:
        keys = _ORDINARY_SCAN_CONTRACT_KEYS
    contract = _exact_keys(value["scan_contract"], keys, f"{label}.scan_contract")
    _exact_string(
        contract["stroke_side"],
        _EXPECTED_STROKE_SIDES[motion_id],
        f"{label}.scan_contract.stroke_side",
    )
    _string_tuple(contract["cli_overrides"], f"{label}.scan_contract.cli_overrides")
    _string(contract["model"], f"{label}.scan_contract.model")
    _exact_string(
        contract["span_rule"],
        "longest_contiguous_run_at_or_above_score_threshold",
        f"{label}.scan_contract.span_rule",
    )
    if "default_ball_distribution" in contract:
        if contract["default_ball_distribution"] is not None:
            raise MarkerSemanticsError(
                f"{label}.default_ball_distribution must remain null"
            )
    if motion_id == SYNTHETIC_MOTION_ID:
        _exact_string(
            contract["transfer_to_synthetic_forehand_behavior"],
            "forbidden",
            f"{label}.scan_contract.transfer_to_synthetic_forehand_behavior",
        )
    return remote_path, scan_sha


def _validate_ge_seed(
    raw: Any,
    *,
    motion_id: str,
    label: str,
    threshold: float,
) -> tuple[int, int]:
    value = _exact_keys(raw, _GE_SEED_KEYS, label)
    span = _inclusive_span(value["span_inclusive"], f"{label}.span_inclusive")
    _exact_number(value["threshold_fraction"], threshold, f"{label}.threshold_fraction")
    _exact_string(
        value["seed_role"],
        _EXPECTED_SEED_ROLES[motion_id],
        f"{label}.seed_role",
    )
    _exact_string(
        value["source_scan_ref"],
        "legacy_scan_evidence",
        f"{label}.source_scan_ref",
    )
    _exact_bool(
        value["legacy_stationary_scan_only"],
        True,
        f"{label}.legacy_stationary_scan_only",
    )
    _exact_bool(
        value["certified_post_retime_window"],
        False,
        f"{label}.certified_post_retime_window",
    )
    return span


def _validate_frame_receipt(
    repo_root: Path,
    raw: Any,
    *,
    motion_id: str,
    nominal_frame: int,
    event_source_sha256: str,
    bound_source_sha256: str,
    summary_row: Mapping[str, Any],
) -> FrameIdentityBinding:
    label = f"motions[{motion_id}].nominal_event.frame_identity_receipt"
    reference = _exact_keys(raw, _FRAME_RECEIPT_REF_KEYS, label)
    path_text, receipt_sha, receipt = _read_bound_json(
        repo_root,
        {"path": reference["path"], "sha256": reference["sha256"]},
        label,
    )
    if path_text != summary_row["receipt_path"]:
        raise MarkerSemanticsError(
            f"{label}.path disagrees with frame identity summary"
        )
    if receipt_sha != summary_row["receipt_sha256"]:
        raise MarkerSemanticsError(
            f"{label}.sha256 disagrees with frame identity summary"
        )
    classification = _string(reference["classification"], f"{label}.classification")
    if classification != summary_row["classification"]:
        raise MarkerSemanticsError(f"{label}.classification disagrees with summary")
    if reference["receipt_id"] != receipt.get("receipt_id"):
        raise MarkerSemanticsError(f"{label}.receipt_id disagrees with receipt")
    if receipt.get("motion_id") != motion_id:
        raise MarkerSemanticsError(f"{label} motion_id disagrees with receipt")
    if receipt.get("publication_class") != "evidence_only_not_artifact_authorization":
        raise MarkerSemanticsError(f"{label} publication class changed")
    _validate_authorization(
        receipt.get("authorization"), f"{label}.receipt.authorization"
    )
    if receipt.get("mapping_role") != "ordinary_nominal_event_to_bound_recipe_source":
        raise MarkerSemanticsError(f"{label} mapping role changed")
    semantic = receipt.get("semantic_contract")
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("observed_ball_contact") is not False
        or semantic.get("behavior_authorized") is not False
    ):
        raise MarkerSemanticsError(f"{label} semantic authorization changed")
    try:
        event_mapping = receipt["event_mapping"]
        inputs = receipt["input_bindings"]
        invariant = receipt["invariant_joint_contract"]
        integrity = receipt["integrity_contract"]
        receipt_authority = receipt["event_authority_contract"]
    except KeyError as exc:
        raise MarkerSemanticsError(f"{label} receipt omitted {exc.args[0]!r}") from exc
    event_frame = _integer(
        reference["event_source_frame"], f"{label}.event_source_frame"
    )
    bound_frame = _integer(
        reference["bound_source_frame"], f"{label}.bound_source_frame"
    )
    if (
        event_frame != nominal_frame
        or bound_frame != nominal_frame
        or event_mapping.get("event_source_frame") != nominal_frame
        or event_mapping.get("bound_source_frame") != nominal_frame
        or event_mapping.get("identity_map") is not True
    ):
        raise MarkerSemanticsError(f"{label} frame identity mapping changed")
    for key, expected in (
        ("event_npz", event_source_sha256),
        ("bound_npz", bound_source_sha256),
    ):
        if (
            not isinstance(inputs.get(key), Mapping)
            or inputs[key].get("sha256") != expected
        ):
            raise MarkerSemanticsError(
                f"{label} {key} binding disagrees with authority"
            )
    if (
        reference["event_source_sha256"] != event_source_sha256
        or reference["bound_source_sha256"] != bound_source_sha256
    ):
        raise MarkerSemanticsError(f"{label} source SHA binding changed")
    invariant_rows = invariant.get("invariant_joints")
    excluded_rows = invariant.get("excluded_joints")
    if not isinstance(invariant_rows, list) or not isinstance(excluded_rows, list):
        raise MarkerSemanticsError(f"{label} invariant contract is invalid")
    invariant_count = _integer(
        reference["invariant_joint_count"],
        f"{label}.invariant_joint_count",
        minimum=1,
    )
    excluded_indices_raw = reference["excluded_joint_indices"]
    if not isinstance(excluded_indices_raw, list):
        raise MarkerSemanticsError(f"{label}.excluded_joint_indices must be an array")
    excluded_indices = tuple(
        _integer(value, f"{label}.excluded_joint_indices[{index}]")
        for index, value in enumerate(excluded_indices_raw)
    )
    actual_excluded = tuple(row.get("index") for row in excluded_rows)
    full_claim = reference["full_pose_identity_claimed"]
    if not isinstance(full_claim, bool):
        raise MarkerSemanticsError(
            f"{label}.full_pose_identity_claimed must be boolean"
        )
    if (
        invariant_count != len(invariant_rows)
        or excluded_indices != actual_excluded
        or integrity.get("full_pose_identity_claimed") is not full_claim
        or summary_row["invariant_joint_count"] != invariant_count
        or tuple(summary_row["excluded_joint_indices"]) != excluded_indices
    ):
        raise MarkerSemanticsError(f"{label} coverage contract changed")
    if full_claim:
        expected_classification = FULL_FRAME_CLASSIFICATION
        if invariant_count != 31 or excluded_indices:
            raise MarkerSemanticsError(f"{label} full identity requires all 31 joints")
    else:
        expected_classification = TEMPORAL_INDEX_CLASSIFICATION
        if invariant_count != 28 or excluded_indices != (26, 28, 30):
            raise MarkerSemanticsError(
                f"{label} temporal-only identity must exclude right wrist"
            )
    if classification != expected_classification:
        raise MarkerSemanticsError(f"{label} classification exceeds receipt evidence")
    if (
        reference["authority_v1_sha256"] != LEGACY_AUTHORITY_SHA256
        or receipt_authority.get("input_sha256") != LEGACY_AUTHORITY_SHA256
    ):
        raise MarkerSemanticsError(f"{label} must bind the immutable v1 authority")
    return FrameIdentityBinding(
        path=path_text,
        sha256=receipt_sha,
        receipt_id=str(reference["receipt_id"]),
        classification=classification,
        invariant_joint_count=invariant_count,
        excluded_joint_indices=excluded_indices,
        full_pose_identity_claimed=full_claim,
    )


def _validate_row(
    raw: Any,
    *,
    expected_motion_id: str,
    row_index: int,
    repo_root: Path,
    summary_rows: Mapping[str, Mapping[str, Any]],
    legacy_historical_starts: Mapping[str, int],
    legacy_rows: Mapping[str, Mapping[str, Any]],
) -> MarkerSemanticsRow:
    label = f"motions[{expected_motion_id}]"
    row = _exact_keys(raw, _ROW_KEYS, label)
    motion_id = _exact_string(
        row["motion_id"], expected_motion_id, f"{label}.motion_id"
    )
    bound = _exact_keys(
        row["bound_recipe_source"],
        _BOUND_SOURCE_KEYS,
        f"{label}.bound_recipe_source",
    )
    bound_path, bound_sha = _validate_bound_file(
        repo_root,
        bound,
        f"{label}.bound_recipe_source",
    )
    scan_path, scan_sha = _validate_scan(
        row["legacy_scan_evidence"],
        motion_id=motion_id,
        bound_sha256=bound_sha,
    )
    ge50 = _validate_ge_seed(
        row["legacy_ge50_seed"],
        motion_id=motion_id,
        label=f"{label}.legacy_ge50_seed",
        threshold=0.5,
    )
    ge80 = _validate_ge_seed(
        row["legacy_ge80_seed"],
        motion_id=motion_id,
        label=f"{label}.legacy_ge80_seed",
        threshold=0.8,
    )
    if not ge50[0] <= ge80[0] <= ge80[1] <= ge50[1]:
        raise MarkerSemanticsError(
            f"{label} legacy ge80 seed must be contained in ge50"
        )

    preferred = _exact_keys(
        row["legacy_preferred_seed"],
        _PREFERRED_KEYS,
        f"{label}.legacy_preferred_seed",
    )
    preferred_raw = preferred["frame"]
    preferred_frame = (
        None
        if preferred_raw is None
        else _integer(preferred_raw, f"{label}.legacy_preferred_seed.frame")
    )
    if preferred_frame is not None and not (ge50[0] <= preferred_frame <= ge50[1]):
        raise MarkerSemanticsError(f"{label} legacy preferred seed lies outside ge50")
    _string(preferred["basis"], f"{label}.legacy_preferred_seed.basis")
    _exact_string(
        preferred["source_scan_ref"],
        "legacy_scan_evidence",
        f"{label}.legacy_preferred_seed.source_scan_ref",
    )
    _exact_bool(
        preferred["behavior_authorized"],
        False,
        f"{label}.legacy_preferred_seed.behavior_authorized",
    )
    _exact_bool(
        preferred["automatic_build_anchor_authorized"],
        False,
        f"{label}.legacy_preferred_seed.automatic_build_anchor_authorized",
    )

    nominal_frame: int | None = None
    frame_identity: FrameIdentityBinding | None = None
    construction: ConstructionMarker | None = None
    if motion_id == SYNTHETIC_MOTION_ID:
        if row["nominal_event"] is not None:
            raise MarkerSemanticsError(
                "synthetic construction may not create nominal_event"
            )
        if preferred_frame is not None:
            raise MarkerSemanticsError(
                "synthetic construction may not create preferred behavior"
            )
        construction_raw = _exact_keys(
            row["construction_marker"],
            _CONSTRUCTION_KEYS,
            f"{label}.construction_marker",
        )
        annotation = _integer(
            construction_raw["annotation_frame"],
            f"{label}.construction_marker.annotation_frame",
        )
        solve_span = _inclusive_span(
            construction_raw["solve_span_inclusive"],
            f"{label}.construction_marker.solve_span_inclusive",
        )
        donor_preferred = _integer(
            construction_raw["donor_preferred_frame"],
            f"{label}.construction_marker.donor_preferred_frame",
        )
        if not (
            solve_span[0] <= annotation <= solve_span[1]
            and solve_span[0] <= ge50[0] <= ge50[1] <= solve_span[1]
            and ge50[0] <= donor_preferred <= ge50[1]
        ):
            raise MarkerSemanticsError(
                f"{label} construction markers exceed their solve/donor spans"
            )
        _exact_string(
            construction_raw["donor_motion_id"],
            "bh_block",
            f"{label}.construction_marker.donor_motion_id",
        )
        _string(
            construction_raw["annotation_semantics"],
            f"{label}.construction_marker.annotation_semantics",
        )
        for key in (
            "upper_receipt",
            "upper_scoped_window",
            "full_receipt",
            "full_scoped_window",
        ):
            _validate_bound_file(
                repo_root,
                construction_raw[key],
                f"{label}.construction_marker.{key}",
            )
        _exact_bool(
            construction_raw["behavior_nominal_created"],
            False,
            f"{label}.construction_marker.behavior_nominal_created",
        )
        _exact_bool(
            construction_raw["behavior_preferred_created"],
            False,
            f"{label}.construction_marker.behavior_preferred_created",
        )
        _exact_string(
            construction_raw["transfer_to_synthetic_forehand_behavior"],
            "forbidden",
            (f"{label}.construction_marker." "transfer_to_synthetic_forehand_behavior"),
        )
        construction = ConstructionMarker(
            annotation_frame=annotation,
            donor_preferred_frame=donor_preferred,
            solve_span=solve_span,
        )
    else:
        if row["construction_marker"] is not None:
            raise MarkerSemanticsError(f"{label}.construction_marker must be null")
        event = _exact_keys(
            row["nominal_event"],
            _NOMINAL_EVENT_KEYS,
            f"{label}.nominal_event",
        )
        nominal_frame = _integer(event["frame"], f"{label}.nominal_event.frame")
        semantic_kind = _string(
            event["semantic_kind"], f"{label}.nominal_event.semantic_kind"
        )
        if not semantic_kind.startswith("air_swing_"):
            raise MarkerSemanticsError(
                f"{label} nominal event must remain an air-swing label"
            )
        _exact_bool(
            event["contact_truth_observed"],
            False,
            f"{label}.nominal_event.contact_truth_observed",
        )
        _exact_bool(
            event["behavior_authorized"],
            False,
            f"{label}.nominal_event.behavior_authorized",
        )
        _string(
            event["legacy_evidence_json_pointer"],
            f"{label}.nominal_event.legacy_evidence_json_pointer",
        )
        _remote_path(
            event["event_source_remote_path"],
            f"{label}.nominal_event.event_source_remote_path",
        )
        event_source_sha = _sha256(
            event["event_source_sha256"],
            f"{label}.nominal_event.event_source_sha256",
        )
        if motion_id not in summary_rows:
            raise MarkerSemanticsError(
                f"{label} claims a v1 nominal event, but the frame-identity "
                f"summary has no ordinary receipt for {motion_id!r}"
            )
        summary_row = summary_rows[motion_id]
        frame_identity = _validate_frame_receipt(
            repo_root,
            event["frame_identity_receipt"],
            motion_id=motion_id,
            nominal_frame=nominal_frame,
            event_source_sha256=event_source_sha,
            bound_source_sha256=bound_sha,
            summary_row=summary_row,
        )
        _exact_string(
            event["mapping_to_bound_recipe_source"],
            frame_identity.classification,
            f"{label}.nominal_event.mapping_to_bound_recipe_source",
        )

    # FABRICATION GUARD.  Runs for every legacy-lineage row, synthetic included
    # (the synthetic row has no nominal event, so only the bound bytes are
    # checked).  See _require_v1_content_agreement for why this exists.
    _require_v1_content_agreement(
        legacy_rows,
        motion_id=motion_id,
        bound_sha256=bound_sha,
        nominal_frame=nominal_frame,
        label=label,
    )

    historical = _exact_keys(
        row["historical_adv2c3_comparator"],
        _HISTORICAL_KEYS,
        f"{label}.historical_adv2c3_comparator",
    )
    historical_frame = _integer(
        historical["frame"],
        f"{label}.historical_adv2c3_comparator.frame",
    )
    if historical_frame != legacy_historical_starts[motion_id]:
        raise MarkerSemanticsError(
            f"{label} historical comparator disagrees with immutable v1 "
            "direct record"
        )
    _string(
        historical["recorded_basis"],
        f"{label}.historical_adv2c3_comparator.recorded_basis",
    )
    _exact_string(
        historical["source_authority_path"],
        LEGACY_AUTHORITY_PATH,
        f"{label}.historical_adv2c3_comparator.source_authority_path",
    )
    _exact_string(
        historical["source_authority_sha256"],
        LEGACY_AUTHORITY_SHA256,
        f"{label}.historical_adv2c3_comparator.source_authority_sha256",
    )
    _exact_string(
        historical["source_json_pointer"],
        f"/motions/{row_index}/historical_adv2c3_start/frame",
        f"{label}.historical_adv2c3_comparator.source_json_pointer",
    )
    _exact_bool(
        historical["adopted"],
        False,
        f"{label}.historical_adv2c3_comparator.adopted",
    )
    _exact_bool(
        historical["recompute_forbidden"],
        True,
        f"{label}.historical_adv2c3_comparator.recompute_forbidden",
    )
    _exact_string(
        historical["role"],
        "historical_comparator_only_not_default",
        f"{label}.historical_adv2c3_comparator.role",
    )

    gate = _exact_keys(
        row["post_retime_behavior_gate"],
        _POST_RETIME_GATE_KEYS,
        f"{label}.post_retime_behavior_gate",
    )
    _exact_bool(gate["required"], True, f"{label}.post_retime_behavior_gate.required")
    expected_status = (
        "PENDING_NEW_FOREHAND_POST_RETIME_BEHAVIOR_RESCAN"
        if motion_id == SYNTHETIC_MOTION_ID
        else "PENDING_POST_RETIME_BEHAVIOR_RESCAN"
    )
    _exact_string(
        gate["status"],
        expected_status,
        f"{label}.post_retime_behavior_gate.status",
    )
    if gate["required_scopes"] != ["upper", "full"]:
        raise MarkerSemanticsError(
            f"{label}.post_retime_behavior_gate requires upper/full"
        )
    _exact_bool(
        gate["legacy_seed_is_certified_window"],
        False,
        (f"{label}.post_retime_behavior_gate." "legacy_seed_is_certified_window"),
    )
    _exact_bool(
        gate["behavior_promotion_authorized"],
        False,
        f"{label}.post_retime_behavior_gate.behavior_promotion_authorized",
    )
    return MarkerSemanticsRow(
        motion_id=motion_id,
        nominal_event=nominal_frame,
        ge50_seed=ge50,
        ge80_seed=ge80,
        preferred_seed=preferred_frame,
        construction_marker=construction,
        historical_adv2c3_start=historical_frame,
        bound_recipe_source_path=bound_path,
        bound_recipe_source_sha256=bound_sha,
        source_scan_remote_path=scan_path,
        source_scan_sha256=scan_sha,
        frame_identity=frame_identity,
        post_retime_behavior_gate_status=expected_status,
        event_provenance_kind=(
            EVENT_KIND_DONOR_CONSTRUCTION
            if construction is not None
            else EVENT_KIND_LEGACY_V1
        ),
    )


def _validate_not_claimed(raw: Any, expected_reason: str, label: str) -> None:
    value = _exact_keys(raw, _NOT_CLAIMED_KEYS, label)
    _exact_bool(value["claimed"], False, f"{label}.claimed")
    _exact_string(value["reason"], expected_reason, f"{label}.reason")


def _validate_derivation(
    repo_root: Path, raw: Any, label: str
) -> tuple[str, str, str, tuple[str, ...]]:
    value = _exact_keys(raw, _DERIVATION_KEYS, label)
    tool_text, tool_path = _repo_file(repo_root, value["tool"], f"{label}.tool")
    tool_sha = _sha256(value["tool_sha256"], f"{label}.tool_sha256")
    actual_tool_sha = sha256_file(tool_path)
    if actual_tool_sha != tool_sha:
        raise MarkerSemanticsError(
            f"{label}.tool_sha256 does not match the tool on disk: "
            f"declared {tool_sha}, actual {actual_tool_sha}"
        )
    rule_id = _string(value["rule_id"], f"{label}.rule_id")
    if rule_id not in _ALLOWED_DERIVED_RULE_IDS:
        raise MarkerSemanticsError(
            f"{label}.rule_id {rule_id!r} is not a registered derivation rule; "
            f"registered={sorted(_ALLOWED_DERIVED_RULE_IDS)}"
        )
    argv = _string_tuple(value["cli_argv"], f"{label}.cli_argv")
    if not isinstance(value["thresholds"], Mapping) or not value["thresholds"]:
        raise MarkerSemanticsError(f"{label}.thresholds must be a non-empty object")
    return tool_text, tool_sha, rule_id, argv


def _validate_scan_artifact(
    repo_root: Path,
    raw: Any,
    *,
    bound_sha256: str,
    label: str,
) -> tuple[str, str, Mapping[str, Any]]:
    """Bind the scan product and close it on the row's OWN bound bytes.

    This is what replaces a fabricated v1 lineage.  A legacy row proves its
    frame by pointing at an ancestor; a derived row proves its frame by
    pointing at the very bytes it binds.  Both the authority's claim and the
    artifact's own recorded input must equal ``bound_recipe_source.sha256``.
    """

    reference = _exact_keys(raw, _SCAN_ARTIFACT_KEYS, label)
    declared_input = _sha256(
        reference["input_source_sha256"], f"{label}.input_source_sha256"
    )
    if declared_input != bound_sha256:
        raise MarkerSemanticsError(
            f"{label}.input_source_sha256 must equal this row's "
            f"bound_recipe_source.sha256 ({bound_sha256}); a derived contact is "
            "only meaningful on the bytes it was scanned from"
        )
    path_text, artifact_sha, artifact = _read_bound_json(
        repo_root,
        {"path": reference["path"], "sha256": reference["sha256"]},
        label,
    )
    source_block = artifact.get("input_source")
    if (
        not isinstance(source_block, Mapping)
        or source_block.get("sha256") != bound_sha256
    ):
        raise MarkerSemanticsError(
            f"{label} artifact was not produced from this row's bound source bytes"
        )
    if artifact.get("publication_class") != "evidence_only_not_artifact_authorization":
        raise MarkerSemanticsError(f"{label} artifact publication class changed")
    _validate_authorization(artifact.get("authorization"), f"{label}.authorization")
    non_claims = artifact.get("non_claims")
    if not isinstance(non_claims, Mapping):
        raise MarkerSemanticsError(f"{label} artifact omitted non_claims")
    for key in (
        "observed_ball_contact",
        "returnability_certified",
        "post_retime_window",
        "v1_authority_lineage",
    ):
        if non_claims.get(key) is not False:
            raise MarkerSemanticsError(
                f"{label} artifact non_claims.{key} must remain false"
            )
    return path_text, artifact_sha, artifact


def _validate_derived_provenance(
    repo_root: Path,
    raw: Mapping[str, Any],
    *,
    bound_sha256: str,
    label: str,
    retiming_stage: str,
) -> tuple[DerivedProvenance, Mapping[str, Any]]:
    tool, tool_sha, rule_id, argv = _validate_derivation(
        repo_root, raw["derivation"], f"{label}.derivation"
    )
    artifact_path, artifact_sha, artifact = _validate_scan_artifact(
        repo_root,
        raw["scan_artifact"],
        bound_sha256=bound_sha256,
        label=f"{label}.scan_artifact",
    )
    artifact_derivation = artifact.get("derivation")
    if not isinstance(artifact_derivation, Mapping):
        raise MarkerSemanticsError(f"{label} artifact omitted its own derivation")
    if artifact_derivation.get("tool_sha256") != tool_sha:
        raise MarkerSemanticsError(
            f"{label} artifact was written by a different tool build than "
            f"{label}.derivation.tool_sha256 declares"
        )
    if artifact_derivation.get("rule_id") != rule_id:
        raise MarkerSemanticsError(
            f"{label} artifact rule_id disagrees with the authority's claim"
        )
    if tuple(artifact_derivation.get("cli_argv") or ()) != argv:
        raise MarkerSemanticsError(
            f"{label} artifact cli_argv disagrees with the authority's claim"
        )
    if artifact_derivation.get("thresholds") != raw["derivation"]["thresholds"]:
        raise MarkerSemanticsError(
            f"{label} artifact thresholds disagree with the authority's claim"
        )
    return (
        DerivedProvenance(
            tool=tool,
            tool_sha256=tool_sha,
            rule_id=rule_id,
            cli_argv=argv,
            scan_artifact_path=artifact_path,
            scan_artifact_sha256=artifact_sha,
            scan_input_source_sha256=bound_sha256,
            retiming_stage=retiming_stage,
        ),
        artifact,
    )


def _validate_derived_contact(
    repo_root: Path, raw: Any, *, bound_sha256: str, label: str
) -> DerivedContact:
    value = _exact_keys(raw, _DERIVED_CONTACT_KEYS, label)
    frame = _integer(value["frame"], f"{label}.frame")
    _exact_string(
        value["semantic_kind"],
        _DERIVED_CONTACT_SEMANTIC_KIND,
        f"{label}.semantic_kind",
    )
    # There is no ball in any of these recordings.  A derived contact is a
    # kinematic candidate, never observed truth, and never behavior authority.
    _exact_bool(
        value["contact_truth_observed"], False, f"{label}.contact_truth_observed"
    )
    _exact_bool(value["behavior_authorized"], False, f"{label}.behavior_authorized")
    _exact_bool(
        value["returnability_certified"], False, f"{label}.returnability_certified"
    )
    stage = _string(value["retiming_stage"], f"{label}.retiming_stage")
    if stage not in _DERIVED_RETIMING_STAGES:
        raise MarkerSemanticsError(
            f"{label}.retiming_stage must be one of {sorted(_DERIVED_RETIMING_STAGES)}"
        )
    if stage != "source_timing":
        raise MarkerSemanticsError(
            f"{label}.retiming_stage={stage!r} needs a compiled-clip binding and an "
            "explicit per-row promotion; that path is defined but not implemented"
        )
    # No frame-identity receipt, on purpose: a receipt exists to carry an event
    # measured on clip A over to clip B.  Nothing is being carried here.
    _validate_not_claimed(
        value["frame_identity"], _DERIVED_FRAME_IDENTITY_REASON, f"{label}.frame_identity"
    )
    _validate_not_claimed(
        value["v1_lineage"], _DERIVED_V1_LINEAGE_REASON, f"{label}.v1_lineage"
    )
    provenance, artifact = _validate_derived_provenance(
        repo_root,
        value,
        bound_sha256=bound_sha256,
        label=label,
        retiming_stage=stage,
    )
    result = artifact.get("result")
    if not isinstance(result, Mapping) or result.get("anchor_frame") != frame:
        raise MarkerSemanticsError(
            f"{label}.frame is not the frame the bound scan artifact reports"
        )
    if artifact.get("semantic_kind") != _DERIVED_CONTACT_SEMANTIC_KIND:
        raise MarkerSemanticsError(
            f"{label} artifact semantic_kind disagrees with the authority"
        )
    return DerivedContact(
        frame=frame,
        semantic_kind=_DERIVED_CONTACT_SEMANTIC_KIND,
        returnability_certified=False,
        provenance=provenance,
    )


def _validate_derived_seed(
    repo_root: Path, raw: Any, *, bound_sha256: str, label: str
) -> DerivedSeed:
    value = _exact_keys(raw, _DERIVED_SEED_KEYS, label)
    span = _inclusive_span(value["span_inclusive"], f"{label}.span_inclusive")
    argmax_frame = _integer(value["argmax_frame"], f"{label}.argmax_frame")
    if not span[0] <= argmax_frame <= span[1]:
        raise MarkerSemanticsError(f"{label}.argmax_frame lies outside its own span")
    _string(value["seed_role"], f"{label}.seed_role")
    stage = _string(value["retiming_stage"], f"{label}.retiming_stage")
    if stage != "source_timing":
        raise MarkerSemanticsError(
            f"{label}.retiming_stage must be 'source_timing'; post-retime seeds "
            "require an explicit promotion path that is not implemented"
        )
    _exact_bool(
        value["certified_post_retime_window"],
        False,
        f"{label}.certified_post_retime_window",
    )
    provenance, artifact = _validate_derived_provenance(
        repo_root,
        value,
        bound_sha256=bound_sha256,
        label=label,
        retiming_stage=stage,
    )
    result = artifact.get("result")
    if not isinstance(result, Mapping):
        raise MarkerSemanticsError(f"{label} artifact omitted its result block")
    if tuple(result.get("span_inclusive") or ()) != span:
        raise MarkerSemanticsError(
            f"{label}.span_inclusive is not the span the bound scan artifact reports"
        )
    if result.get("anchor_frame") != argmax_frame:
        raise MarkerSemanticsError(
            f"{label}.argmax_frame is not the frame the bound scan artifact reports"
        )
    return DerivedSeed(
        span=span,
        argmax_frame=argmax_frame,
        seed_role=str(value["seed_role"]),
        certified_post_retime_window=False,
        provenance=provenance,
    )


def _validate_gate_rescan(raw: Any, label: str) -> None:
    """The reserved post-retime rescan RESULT slot.

    The authority has always required a post-retime behavior rescan and has
    never had anywhere to record its answer, which is why wave-1 results ended
    up in ``cfg/strike_annotations.yaml`` instead.  The slot now exists.  It
    accepts only ``null``: promoting a rescan into an anchor needs machinery
    (compiled-clip binding, per-row promotion) that this change does not build,
    and a half-built promotion path is exactly the kind of thing that quietly
    moves ``fh_loop``'s anchor.
    """

    if raw is not None:
        raise MarkerSemanticsError(
            f"{label}.rescan must be null: the slot is reserved, and promoting a "
            "post-retime rescan to an anchor is not implemented"
        )


def _validate_derived_row(
    raw: Any,
    *,
    row_index: int,
    repo_root: Path,
) -> MarkerSemanticsRow:
    """One appended v3 row whose contact frame came from a tool, not from v1."""

    label = f"motions[{row_index}]"
    row = _exact_keys(raw, _V3_ROW_KEYS, label)
    motion_id = _string(row["motion_id"], f"{label}.motion_id")
    label = f"motions[{motion_id}]"
    if motion_id in CANONICAL_MOTION_IDS:
        raise MarkerSemanticsError(
            f"{label} reuses a canonical motion id outside the verbatim prefix"
        )
    bound = _exact_keys(
        row["bound_recipe_source"], _BOUND_SOURCE_KEYS, f"{label}.bound_recipe_source"
    )
    bound_path, bound_sha = _validate_bound_file(
        repo_root, bound, f"{label}.bound_recipe_source"
    )
    provenance = _exact_keys(
        row["event_provenance"],
        _V3_DERIVED_PROVENANCE_KEYS,
        f"{label}.event_provenance",
    )
    _exact_string(
        provenance["kind"],
        EVENT_KIND_DERIVED_CONTACT,
        f"{label}.event_provenance.kind",
    )
    derived_contact = _validate_derived_contact(
        repo_root,
        provenance["derived_contact"],
        bound_sha256=bound_sha,
        label=f"{label}.event_provenance.derived_contact",
    )
    for key in (
        "legacy_ge50_seed",
        "legacy_ge80_seed",
        "legacy_preferred_seed",
        "historical_adv2c3_comparator",
    ):
        if row[key] is not None:
            raise MarkerSemanticsError(
                f"{label}.{key} must be null on a derived row: legacy stationary "
                "scans and the v1 adv2c3 record do not exist for this recording"
            )
    if row["derived_seed"] is None:
        raise MarkerSemanticsError(
            f"{label}.derived_seed is mandatory on a derived row; without it the "
            "compiler has no search window at all"
        )
    derived_seed = _validate_derived_seed(
        repo_root,
        row["derived_seed"],
        bound_sha256=bound_sha,
        label=f"{label}.derived_seed",
    )
    gate = _exact_keys(
        row["post_retime_behavior_gate"],
        _V3_POST_RETIME_GATE_KEYS,
        f"{label}.post_retime_behavior_gate",
    )
    _exact_bool(gate["required"], True, f"{label}.post_retime_behavior_gate.required")
    expected_status = "PENDING_POST_RETIME_BEHAVIOR_RESCAN"
    _exact_string(
        gate["status"], expected_status, f"{label}.post_retime_behavior_gate.status"
    )
    if gate["required_scopes"] != ["upper", "full"]:
        raise MarkerSemanticsError(
            f"{label}.post_retime_behavior_gate requires upper/full"
        )
    _exact_bool(
        gate["legacy_seed_is_certified_window"],
        False,
        f"{label}.post_retime_behavior_gate.legacy_seed_is_certified_window",
    )
    _exact_bool(
        gate["behavior_promotion_authorized"],
        False,
        f"{label}.post_retime_behavior_gate.behavior_promotion_authorized",
    )
    _validate_gate_rescan(gate["rescan"], f"{label}.post_retime_behavior_gate")
    if derived_contact.provenance.scan_artifact_sha256 != (
        derived_seed.provenance.scan_artifact_sha256
    ):
        raise MarkerSemanticsError(
            f"{label} anchor and seed must come from one scan pass"
        )
    return MarkerSemanticsRow(
        motion_id=motion_id,
        nominal_event=None,
        ge50_seed=None,
        ge80_seed=None,
        preferred_seed=None,
        construction_marker=None,
        historical_adv2c3_start=None,
        bound_recipe_source_path=bound_path,
        bound_recipe_source_sha256=bound_sha,
        source_scan_remote_path=None,
        source_scan_sha256=None,
        frame_identity=None,
        post_retime_behavior_gate_status=expected_status,
        event_provenance_kind=EVENT_KIND_DERIVED_CONTACT,
        derived_contact=derived_contact,
        derived_seed=derived_seed,
    )


def _unwrap_v3_prefix_row(raw: Any, *, row_index: int) -> tuple[dict[str, Any], str]:
    """Rewrite one v3 canonical-prefix row into its exact v2 row shape.

    The point of unwrapping instead of reimplementing is that the canonical
    five then travel through ``_validate_row`` -- the same function, the same
    order, the same errors -- so "v3 does not change the old rows" is true by
    construction rather than by test coincidence.
    """

    label = f"motions[{row_index}]"
    row = _exact_keys(raw, _V3_ROW_KEYS, label)
    provenance = _exact_keys(
        row["event_provenance"],
        _V3_LEGACY_PROVENANCE_KEYS,
        f"{label}.event_provenance",
    )
    kind = _string(provenance["kind"], f"{label}.event_provenance.kind")
    if kind == EVENT_KIND_LEGACY_V1:
        if provenance["nominal_event"] is None:
            raise MarkerSemanticsError(
                f"{label}.event_provenance kind {kind!r} requires a nominal_event"
            )
    elif kind == EVENT_KIND_DONOR_CONSTRUCTION:
        if provenance["construction_marker"] is None:
            raise MarkerSemanticsError(
                f"{label}.event_provenance kind {kind!r} requires a construction_marker"
            )
    else:
        raise MarkerSemanticsError(
            f"{label}.event_provenance.kind {kind!r} is not permitted in the "
            "canonical verbatim prefix"
        )
    if row["derived_seed"] is not None:
        raise MarkerSemanticsError(
            f"{label}.derived_seed must be null in the canonical verbatim prefix; "
            "adding one would move an existing motion's search window"
        )
    gate = _exact_keys(
        row["post_retime_behavior_gate"],
        _V3_POST_RETIME_GATE_KEYS,
        f"{label}.post_retime_behavior_gate",
    )
    _validate_gate_rescan(gate["rescan"], f"{label}.post_retime_behavior_gate")
    return (
        {
            "motion_id": row["motion_id"],
            "bound_recipe_source": row["bound_recipe_source"],
            "nominal_event": provenance["nominal_event"],
            "legacy_scan_evidence": provenance["legacy_scan_evidence"],
            "legacy_ge50_seed": row["legacy_ge50_seed"],
            "legacy_ge80_seed": row["legacy_ge80_seed"],
            "legacy_preferred_seed": row["legacy_preferred_seed"],
            "construction_marker": provenance["construction_marker"],
            "historical_adv2c3_comparator": row["historical_adv2c3_comparator"],
            "post_retime_behavior_gate": {
                key: gate[key] for key in sorted(_POST_RETIME_GATE_KEYS)
            },
        },
        kind,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_verbatim_prefix(
    repo_root: Path, unwrapped: Sequence[Mapping[str, Any]]
) -> str:
    """Prove the canonical five were carried over, not retyped.

    The v2 authority bytes are re-read here and its SHA-256 becomes a
    load-bearing input to v3.  Any edit to an existing row -- a swapped source,
    a moved seed, a reworded role -- shows up as a diff on this comparison
    instead of as a silently different library.
    """

    path = (repo_root / MARKER_AUTHORITY_PATH).resolve()
    if not path.is_file():
        raise MarkerSemanticsError(
            f"v3 requires the v2 authority on disk to verify its prefix: {path}"
        )
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != MARKER_AUTHORITY_SHA256:
        raise MarkerSemanticsError(
            "the v2 authority bytes changed; v3's verbatim prefix cannot be verified"
        )
    v2 = _strict_json_bytes(payload, "v2 authority")
    v2_motions = v2["motions"]
    if not isinstance(v2_motions, list) or len(v2_motions) != len(
        CANONICAL_MOTION_IDS
    ):
        raise MarkerSemanticsError("the v2 authority motion list is not the five rows")
    for index, motion_id in enumerate(CANONICAL_MOTION_IDS):
        if _canonical_json(unwrapped[index]) != _canonical_json(v2_motions[index]):
            raise MarkerSemanticsError(
                f"motions[{motion_id}] is not a verbatim carry-over of the v2 row; "
                "v3 may append rows, never restate existing ones"
            )
    return actual


def _validate_v3_supersedes(raw: Any, repo_root: Path) -> None:
    chain = raw
    if not isinstance(chain, list) or len(chain) != 2:
        raise MarkerSemanticsError(
            "authority.supersedes must be a two-link chain: v1 then v2"
        )
    expected = (
        (LEGACY_AUTHORITY_PATH, LEGACY_AUTHORITY_SHA256, "immutable_historical_provenance_only"),
        (MARKER_AUTHORITY_PATH, MARKER_AUTHORITY_SHA256, "superseded_row_source_verbatim"),
    )
    for index, (path_text, sha_text, role) in enumerate(expected):
        link = _exact_keys(chain[index], _SUPERSEDES_KEYS, f"authority.supersedes[{index}]")
        _exact_string(link["path"], path_text, f"authority.supersedes[{index}].path")
        _exact_string(link["sha256"], sha_text, f"authority.supersedes[{index}].sha256")
        _exact_string(link["role"], role, f"authority.supersedes[{index}].role")
        _read_bound_json(
            repo_root,
            {"path": link["path"], "sha256": link["sha256"]},
            f"authority.supersedes[{index}]",
        )


def load_canonical_motion_markers(
    authority_path: str | Path,
    *,
    expected_authority_sha256: str,
    repo_root: str | Path | None = None,
    profile: str = "v2",
) -> MarkerSemantics:
    """Load one exact marker authority and all local content-addressed evidence.

    ``profile`` is the CALLER's expectation ("v2" or "v3").  It is deliberately
    not read from the document: a file whose entire purpose is content-addressed
    authority must not get to pick which rules it is judged by.
    """

    if profile not in MARKER_AUTHORITY_PROFILES:
        raise MarkerSemanticsError(
            f"unknown marker authority profile {profile!r}; "
            f"known={sorted(MARKER_AUTHORITY_PROFILES)}"
        )
    expected_profile = MARKER_AUTHORITY_PROFILES[profile]
    path = Path(authority_path).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else path.parents[1].resolve()
    )
    expected_sha = _sha256(expected_authority_sha256, "expected_authority_sha256")
    try:
        authority_bytes = path.read_bytes()
    except OSError as exc:
        raise MarkerSemanticsError(
            f"cannot read marker authority {path}: {exc}"
        ) from exc
    actual_sha = hashlib.sha256(authority_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise MarkerSemanticsError(
            "marker authority SHA-256 mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    raw = _exact_keys(
        _strict_json_bytes(authority_bytes, "marker authority"),
        _TOP_LEVEL_KEYS,
        "marker authority",
    )
    if _integer(raw["schema_version"], "authority.schema_version") != (
        expected_profile.schema_version
    ):
        raise MarkerSemanticsError(
            f"authority.schema_version must equal {expected_profile.schema_version}"
        )
    # The caller's profile decides; the document only gets to agree or fail.
    authority_id = _exact_string(
        raw["authority_id"], expected_profile.authority_id, "authority.authority_id"
    )
    review_status = _exact_string(
        raw["review_status"],
        expected_profile.review_status,
        "authority.review_status",
    )
    _string(raw["scope"], "authority.scope")
    if expected_profile.name == "v2":
        supersedes = _exact_keys(
            raw["supersedes"], _SUPERSEDES_KEYS, "authority.supersedes"
        )
        _exact_string(
            supersedes["path"],
            LEGACY_AUTHORITY_PATH,
            "authority.supersedes.path",
        )
        _exact_string(
            supersedes["sha256"],
            LEGACY_AUTHORITY_SHA256,
            "authority.supersedes.sha256",
        )
        _exact_string(
            supersedes["role"],
            "immutable_historical_provenance_only",
            "authority.supersedes.role",
        )
        legacy_reference = {
            "path": supersedes["path"],
            "sha256": supersedes["sha256"],
        }
    else:
        _validate_v3_supersedes(raw["supersedes"], root)
        legacy_reference = {
            "path": LEGACY_AUTHORITY_PATH,
            "sha256": LEGACY_AUTHORITY_SHA256,
        }
    legacy_path_text, legacy_sha, legacy_authority = _read_bound_json(
        root,
        legacy_reference,
        "authority.supersedes",
    )
    if (
        legacy_path_text != LEGACY_AUTHORITY_PATH
        or legacy_sha != LEGACY_AUTHORITY_SHA256
    ):
        raise MarkerSemanticsError("immutable v1 authority bytes changed")
    legacy_historical_starts = _direct_legacy_historical_starts(legacy_authority)
    legacy_rows = _legacy_rows_by_motion_id(legacy_authority)
    _validate_authorization(raw["authorization"], "authority.authorization")
    _validate_semantic_contract(raw["semantic_contract"])

    motions = raw["motions"]
    if not isinstance(motions, list):
        raise MarkerSemanticsError("authority.motions must be a JSON array")
    if expected_profile.appendable_rows:
        if len(motions) < len(CANONICAL_MOTION_IDS):
            raise MarkerSemanticsError(
                "authority.motions must begin with the canonical five rows"
            )
    elif len(motions) != len(CANONICAL_MOTION_IDS):
        raise MarkerSemanticsError("authority.motions must contain exactly five rows")
    declared_motion_ids = tuple(
        row.get("motion_id") if isinstance(row, Mapping) else None for row in motions
    )
    if expected_profile.exact_appended_motion_ids is not None:
        expected_motion_ids = (
            CANONICAL_MOTION_IDS + expected_profile.exact_appended_motion_ids
        )
        if declared_motion_ids != expected_motion_ids:
            raise MarkerSemanticsError(
                f"authority profile {expected_profile.name!r} requires exact ordered "
                f"motion ids {expected_motion_ids!r}; got {declared_motion_ids!r}"
            )
    # Validate the cheap structural identity before opening the ignored evidence
    # bundle.  A malformed profile must fail for its own ordered-ID defect even
    # on a machine that has not restored every historical frame-map receipt.
    summary_rows = _validate_shared_evidence(raw["shared_evidence"], root)
    prefix = motions[: len(CANONICAL_MOTION_IDS)]
    actual_ids = tuple(
        row.get("motion_id") if isinstance(row, Mapping) else None for row in prefix
    )
    if actual_ids != CANONICAL_MOTION_IDS:
        raise MarkerSemanticsError(
            "authority.motions canonical order changed; "
            f"expected={CANONICAL_MOTION_IDS!r}, got={actual_ids!r}"
        )

    verbatim_sha: str | None = None
    prefix_payloads: list[Mapping[str, Any]] = list(prefix)
    prefix_kinds = [EVENT_KIND_LEGACY_V1] * len(CANONICAL_MOTION_IDS)
    if expected_profile.appendable_rows:
        unwrapped: list[Mapping[str, Any]] = []
        kinds: list[str] = []
        for index, raw_row in enumerate(prefix):
            payload, kind = _unwrap_v3_prefix_row(raw_row, row_index=index)
            unwrapped.append(payload)
            kinds.append(kind)
        verbatim_sha = _require_verbatim_prefix(root, unwrapped)
        prefix_payloads = unwrapped
        prefix_kinds = kinds

    rows = [
        _validate_row(
            raw_row,
            expected_motion_id=motion_id,
            row_index=index,
            repo_root=root,
            summary_rows=summary_rows,
            legacy_historical_starts=legacy_historical_starts,
            legacy_rows=legacy_rows,
        )
        for index, (raw_row, motion_id) in enumerate(
            zip(prefix_payloads, CANONICAL_MOTION_IDS)
        )
    ]
    if expected_profile.appendable_rows:
        for row, declared in zip(rows, prefix_kinds):
            if row.event_provenance_kind != declared:
                raise MarkerSemanticsError(
                    f"motions[{row.motion_id}].event_provenance.kind {declared!r} "
                    f"disagrees with the row's actual content "
                    f"({row.event_provenance_kind!r})"
                )
        seen = set(CANONICAL_MOTION_IDS)
        for index, raw_row in enumerate(
            motions[len(CANONICAL_MOTION_IDS) :], start=len(CANONICAL_MOTION_IDS)
        ):
            appended = _validate_derived_row(
                raw_row, row_index=index, repo_root=root
            )
            if appended.motion_id in seen:
                raise MarkerSemanticsError(
                    f"authority.motions has duplicate motion_id "
                    f"{appended.motion_id!r}"
                )
            seen.add(appended.motion_id)
            rows.append(appended)
    return MarkerSemantics(
        path=path,
        repo_root=root,
        sha256=actual_sha,
        authority_id=authority_id,
        review_status=review_status,
        legacy_authority_sha256=LEGACY_AUTHORITY_SHA256,
        rows=tuple(rows),
        profile=expected_profile.name,
        verbatim_prefix_authority_sha256=verbatim_sha,
    )


load_marker_semantics = load_canonical_motion_markers


__all__ = [
    "CANONICAL_MOTION_IDS",
    "EVENT_KIND_DERIVED_CONTACT",
    "EVENT_KIND_DONOR_CONSTRUCTION",
    "EVENT_KIND_LEGACY_V1",
    "AuthorityProfile",
    "ConstructionMarker",
    "DerivedContact",
    "DerivedProvenance",
    "DerivedSeed",
    "FULL_FRAME_CLASSIFICATION",
    "FrameIdentityBinding",
    "LEGACY_AUTHORITY_PATH",
    "LEGACY_AUTHORITY_SHA256",
    "MARKER_AUTHORITY_ID",
    "MARKER_AUTHORITY_PATH",
    "MARKER_AUTHORITY_PROFILES",
    "MARKER_AUTHORITY_PROFILE_BY_PATH",
    "MARKER_AUTHORITY_REVIEW_STATUS",
    "MARKER_AUTHORITY_SCHEMA_VERSION",
    "MARKER_AUTHORITY_SHA256",
    "MARKER_AUTHORITY_V3_ID",
    "MARKER_AUTHORITY_V3_PATH",
    "MARKER_AUTHORITY_V3_REVIEW_STATUS",
    "MARKER_AUTHORITY_V3_SCHEMA_VERSION",
    "MARKER_AUTHORITY_V3APPEND_V12_MOTION_IDS",
    "MARKER_AUTHORITY_V3APPEND_V12_PATH",
    "MarkerSemantics",
    "MarkerSemanticsError",
    "MarkerSemanticsRow",
    "SYNTHETIC_MOTION_ID",
    "TEMPORAL_INDEX_CLASSIFICATION",
    "load_canonical_motion_markers",
    "load_marker_semantics",
    "sha256_file",
]
