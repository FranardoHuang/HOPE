#!/usr/bin/env python3
"""Reconstruct the real block pose matrix and solve one ready challenger.

This adapter is the only production path that may attach a
``ContactSourceProof`` to ``canonical_neutral_ready``.  It loads the
content-addressed canonical recipe, reconstructs upper/full backhand block
poses at source events opportunity-start/construction-donor-preferred/
nominal-event/opportunity-end, derives the forehand counterparts
through the exact seven-joint signed-face manifold, and binds every array to
the exact recipe, source, MuJoCo, URDF, site, and tool bytes.

The result is still only a restricted fixed-site/angular-minimax challenger.
The CLI never authorizes training or hardware, and publication remains subject
to the solver's exact collision/ground/support gates.  RunPod invocation must
be externally scheduled as nice-19, CPU-only work.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import canonical_body_scope as body_scope
import canonical_face_manifold as face
import canonical_motion_recipe as recipe_module
import canonical_mujoco_dynamics_gate as dynamics_gate
import canonical_neutral_ready as neutral
import audit_motion_npz
import motion_kinematics_contract
import mujoco_motion_player as motion_player
import racket_geometry_contract


class NeutralReadyAdapterError(RuntimeError):
    """The real recipe cannot produce an exact 16-row pose matrix."""


@dataclass(frozen=True)
class RealNeutralReadyInputs:
    """All content-bound inputs required by the neutral-ready solver."""

    recipe: recipe_module.CanonicalMotionRecipe
    backend: face.MujocoRightRacketBackend
    contacts: tuple[neutral.ContactPose, ...]
    contact_source_proof: neutral.ContactSourceProof
    ready_binding: neutral.ReadySourceBinding
    model_binding: neutral.ExactModelBinding


@dataclass(frozen=True)
class BlockPhaseMapBinding:
    """Reviewed lineage pin; f42 is not synthetic behavioral preference."""

    authority_path: str | Path
    expected_authority_sha256: str
    source_motion_sha256: str
    opportunity_start: int
    construction_donor_preferred: int
    nominal_event: int
    opportunity_end: int
    synthetic_behavior_preferred: None
    review_status: str

    def __post_init__(self) -> None:
        text = str(self.expected_authority_sha256)
        if (
            len(text) != 64
            or text != text.lower()
            or any(character not in "0123456789abcdef" for character in text)
        ):
            raise ValueError(
                "phase-map authority SHA-256 must be 64 lowercase hex digits"
            )
        source_sha = str(self.source_motion_sha256)
        if (
            len(source_sha) != 64
            or source_sha != source_sha.lower()
            or any(
                character not in "0123456789abcdef"
                for character in source_sha
            )
        ):
            raise ValueError(
                "phase-map source SHA-256 must be 64 lowercase hex digits"
            )
        values = self.frames
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values.values()
        ) or not (
            values["opportunity_start"]
            < values["construction_donor_preferred"]
            < values["nominal_event"]
            < values["opportunity_end"]
        ):
            raise ValueError(
                "phase map must be four strictly increasing integer frames"
            )
        if self.synthetic_behavior_preferred is not None:
            raise ValueError(
                "synthetic behavioral preferred frame must remain null"
            )
        if self.review_status != (
            "REVIEWED_FOR_LEGACY_SEED_PROVENANCE_ONLY"
        ):
            raise ValueError("phase authority review scope changed")

    @property
    def frames(self) -> dict[str, int]:
        return {
            "opportunity_start": self.opportunity_start,
            "construction_donor_preferred": (
                self.construction_donor_preferred
            ),
            "nominal_event": self.nominal_event,
            "opportunity_end": self.opportunity_end,
        }


def _sha256_file(path: Path) -> str:
    return neutral._read_real_file_snapshot(path, str(path))[1]


def _strict_json_from_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise NeutralReadyAdapterError(
                    f"{label} duplicates JSON key {key!r}"
                )
            output[key] = value
        return output

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise NeutralReadyAdapterError(
            f"cannot parse {label}: {exc}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise NeutralReadyAdapterError(f"{label} must be one JSON object")
    return decoded


def load_block_phase_map_binding(
    authority_path: str | Path,
    expected_sha256: str,
) -> BlockPhaseMapBinding:
    """Parse the reviewed lineage authority from the exact bytes hashed."""

    path = Path(authority_path).expanduser().absolute()
    expected = str(expected_sha256)
    try:
        authority_bytes, actual = neutral._read_real_file_snapshot(
            path, "reviewed marker authority"
        )
    except neutral.NeutralReadyError as exc:
        raise NeutralReadyAdapterError(str(exc)) from exc
    if actual != expected:
        raise NeutralReadyAdapterError(
            "reviewed block marker authority file/hash binding failed"
        )
    raw = _strict_json_from_bytes(
        authority_bytes, "block marker-semantics authority"
    )
    expected_keys = {
        "schema_version",
        "authority_id",
        "review_status",
        "scope",
        "authorization",
        "semantic_contract",
        "shared_evidence",
        "motions",
    }
    if set(raw) != expected_keys:
        raise NeutralReadyAdapterError(
            "block marker authority exact key set changed"
        )
    if (
        raw["schema_version"] != 1
        or raw["authority_id"]
        != "canonical_motion_marker_semantics_v1_20260724"
        or raw["review_status"]
        != "REVIEWED_FOR_LEGACY_SEED_PROVENANCE_ONLY"
        or raw["authorization"]
        != {
            "training_authorized": False,
            "behavior_authorized": False,
            "hardware_authorized": False,
            "artifact_promotion_authorized": False,
        }
    ):
        raise NeutralReadyAdapterError(
            "block marker authority identity/status changed"
        )
    semantic = raw["semantic_contract"]
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("contact_truth_observed") is not False
        or semantic.get("automatic_event_to_window_aliasing_forbidden")
        is not True
        or semantic.get(
            "automatic_adv2c3_recomputation_from_preferred_frame_forbidden"
        )
        is not True
        or semantic.get("post_retime_behavior_rescan_required") is not True
    ):
        raise NeutralReadyAdapterError(
            "block marker authority semantic contract changed"
        )
    motions = raw["motions"]
    if not isinstance(motions, list):
        raise NeutralReadyAdapterError(
            "block marker authority motions must be a list"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in motions:
        if not isinstance(row, Mapping) or not isinstance(
            row.get("motion_id"), str
        ):
            raise NeutralReadyAdapterError(
                "block marker authority motion row is malformed"
            )
        motion_id = str(row["motion_id"])
        if motion_id in by_id:
            raise NeutralReadyAdapterError(
                f"block marker authority duplicates {motion_id!r}"
            )
        by_id[motion_id] = row
    try:
        bh = by_id["bh_block"]
        fh = by_id["fh_block_syn"]
        bh_source = bh["bound_recipe_source"]
        fh_source = fh["bound_recipe_source"]
        bh_seed = bh["opportunity_seed"]
        fh_seed = fh["opportunity_seed"]
        bh_event = bh["nominal_event"]
        fh_event = fh["nominal_event"]
        manifold = fh["face_manifold"]
        ge80 = tuple(int(value) for value in bh_seed["ge80_span_inclusive"])
        fh_ge80 = tuple(
            int(value) for value in fh_seed["ge80_span_inclusive"]
        )
        construction = int(
            fh_seed["construction_donor_preferred_frame"]
        )
        nominal = int(fh_event["frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NeutralReadyAdapterError(
            f"block marker authority relevant rows are malformed: {exc}"
        ) from exc
    source_sha = str(bh_source.get("sha256"))
    if (
        bh_source != fh_source
        or ge80 != fh_ge80
        or len(ge80) != 2
        or bh_seed.get("preferred_frame") != construction
        or fh_seed.get("preferred_frame") is not None
        or bh_event.get("frame") != nominal
        or bh_event.get("contact_truth_observed") is not False
        or fh_event.get("contact_truth_observed") is not False
        or manifold.get("annotation_frame") != nominal
        or manifold.get("behavior_rescan_required") is not True
        or tuple(manifold.get("solve_span_inclusive", ())) != (34, 48)
    ):
        raise NeutralReadyAdapterError(
            "block marker authority BH/FH lineage facts changed"
        )
    frames = {
        "opportunity_start": ge80[0],
        "construction_donor_preferred": construction,
        "nominal_event": nominal,
        "opportunity_end": ge80[1],
    }
    try:
        binding = BlockPhaseMapBinding(
            authority_path=path,
            expected_authority_sha256=expected,
            source_motion_sha256=source_sha,
            opportunity_start=frames["opportunity_start"],
            construction_donor_preferred=frames[
                "construction_donor_preferred"
            ],
            nominal_event=frames["nominal_event"],
            opportunity_end=frames["opportunity_end"],
            synthetic_behavior_preferred=None,
            review_status=str(raw["review_status"]),
        )
    except (TypeError, ValueError) as exc:
        raise NeutralReadyAdapterError(
            f"block marker authority values are invalid: {exc}"
        ) from exc
    if _sha256_file(path) != expected:
        raise NeutralReadyAdapterError(
            "block marker authority changed during stable parse"
        )
    return binding


def _repo_real_file(
    repo_root: Path,
    relative_value: Any,
    label: str,
) -> tuple[Path, Path]:
    if not isinstance(relative_value, str) or not relative_value:
        raise NeutralReadyAdapterError(
            f"{label} must be a non-empty repository-relative path"
        )
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise NeutralReadyAdapterError(
            f"{label} must not be absolute or escape the repository"
        )
    candidate = (repo_root / relative).absolute()
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise NeutralReadyAdapterError(
            f"{label} cannot be resolved inside the repository: {exc}"
        ) from exc
    if resolved != candidate or not candidate.is_file():
        raise NeutralReadyAdapterError(
            f"{label} must be a real file with no symlink component"
        )
    return relative, candidate


def _snapshot_recipe_inputs(
    recipe_path: str | Path,
    *,
    repo_root: str | Path | None,
    expected_recipe_sha256: str,
) -> tuple[recipe_module.CanonicalMotionRecipe, dict[Path, str]]:
    """Parse recipe and every NPZ input from one private byte snapshot.

    The upstream generic loader historically hashes a path and later reopens
    the same pathname.  This adapter instead copies each already-hashed file
    descriptor snapshot into a private temporary repository, runs the strict
    loader there, then translates only path labels back to the original repo.
    Parsed arrays can therefore never come from bytes different from the proof
    hash even if an original pathname is swapped concurrently.
    """

    requested = Path(recipe_path).expanduser().absolute()
    root = (
        Path(repo_root).expanduser().absolute()
        if repo_root is not None
        else requested.parents[1]
    )
    try:
        resolved_root = root.resolve(strict=True)
        resolved_recipe = requested.resolve(strict=True)
        recipe_relative = resolved_recipe.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise NeutralReadyAdapterError(
            f"recipe/repository path binding failed: {exc}"
        ) from exc
    if root != resolved_root or requested != resolved_recipe:
        raise NeutralReadyAdapterError(
            "recipe and repo root must have no symlink components"
        )
    recipe_bytes, recipe_sha = neutral._read_real_file_snapshot(
        requested, "canonical motion recipe"
    )
    expected_recipe = str(expected_recipe_sha256)
    if recipe_sha != expected_recipe:
        raise NeutralReadyAdapterError(
            "canonical recipe SHA-256 does not match the independently "
            f"reviewed pin: {recipe_sha} != {expected_recipe}"
        )
    raw = _strict_json_from_bytes(recipe_bytes, "canonical motion recipe")
    try:
        reference_rows = [
            (
                raw["canonical_ready"]["path"],
                raw["canonical_ready"]["sha256"],
                "canonical ready",
            ),
            (
                raw["marker_authority"]["path"],
                raw["marker_authority"]["sha256"],
                "marker authority v2",
            ),
            (
                raw["model_contract"]["mjcf_path"],
                raw["model_contract"]["mjcf_sha256"],
                "MJCF root",
            ),
            (
                raw["model_contract"]["urdf_path"],
                raw["model_contract"]["urdf_sha256"],
                "URDF",
            ),
            (
                raw["model_contract"]["body_order_path"],
                raw["model_contract"]["body_order_sha256"],
                "body order",
            ),
            *[
                (
                    row["source_path"],
                    row["source_sha256"],
                    f"motion source {row['motion_id']}",
                )
                for row in raw["motion_specs"]
            ],
        ]
    except (KeyError, TypeError) as exc:
        raise NeutralReadyAdapterError(
            f"recipe input graph is malformed: {exc}"
        ) from exc
    try:
        authority_relative, authority_original = _repo_real_file(
            resolved_root, raw["marker_authority"]["path"], "marker authority v2"
        )
        authority_bytes, _ = neutral._read_real_file_snapshot(
            authority_original, "marker authority v2"
        )
        authority_raw = _strict_json_from_bytes(
            authority_bytes, "marker authority v2"
        )
    except (KeyError, TypeError) as exc:
        raise NeutralReadyAdapterError(
            f"marker authority reference is malformed: {exc}"
        ) from exc
    del authority_relative  # snapshot copying below re-derives each path

    def _collect_local_refs(value: Any, rows: list[tuple[Any, Any, str]]) -> None:
        # The v2 authority binds every local evidence file as {path, sha256};
        # remote provenance uses remote_path and is intentionally skipped.
        if isinstance(value, Mapping):
            path_value = value.get("path")
            sha_value = value.get("sha256")
            if isinstance(path_value, str) and isinstance(sha_value, str):
                rows.append(
                    (path_value, sha_value, "marker authority evidence")
                )
            for child in value.values():
                _collect_local_refs(child, rows)
        elif isinstance(value, (list, tuple)):
            for child in value:
                _collect_local_refs(child, rows)

    authority_reference_rows: list[tuple[Any, Any, str]] = []
    _collect_local_refs(authority_raw, authority_reference_rows)
    known = {str(row[0]) for row in reference_rows}
    for row in authority_reference_rows:
        if str(row[0]) not in known:
            known.add(str(row[0]))
            reference_rows.append(row)
    snapshots: dict[Path, tuple[Path, bytes, str]] = {}
    for relative_value, expected_value, label in reference_rows:
        relative, original = _repo_real_file(
            resolved_root, relative_value, label
        )
        expected = str(expected_value)
        if (
            len(expected) != 64
            or expected != expected.lower()
            or any(
                character not in "0123456789abcdef"
                for character in expected
            )
        ):
            raise NeutralReadyAdapterError(
                f"{label} SHA-256 pin is malformed"
            )
        payload, actual = neutral._read_real_file_snapshot(original, label)
        if actual != expected:
            raise NeutralReadyAdapterError(
                f"{label} SHA-256 mismatch: {actual} != {expected}"
            )
        prior = snapshots.get(original)
        if prior is not None and prior[2] != actual:
            raise NeutralReadyAdapterError(
                f"recipe gives conflicting hashes for {original}"
            )
        snapshots[original] = (relative, payload, actual)

    with tempfile.TemporaryDirectory(
        prefix="neutral-ready-recipe-snapshot-"
    ) as raw_snapshot_root:
        snapshot_root = Path(raw_snapshot_root)

        def write_snapshot(relative: Path, payload: bytes) -> Path:
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
                stream.flush()
            return destination

        snapshot_recipe = write_snapshot(recipe_relative, recipe_bytes)
        for relative, payload, _ in snapshots.values():
            destination = snapshot_root / relative
            if not destination.exists():
                write_snapshot(relative, payload)
        loaded = recipe_module.load_canonical_motion_recipe(
            snapshot_recipe, repo_root=snapshot_root
        )

    originals_by_relative = {
        relative: original
        for original, (relative, _, _) in snapshots.items()
    }
    ready_relative = Path(str(raw["canonical_ready"]["path"]))
    ready = replace(
        loaded.ready,
        path=originals_by_relative[ready_relative],
    )
    specs_by_id = {
        str(row["motion_id"]): row for row in raw["motion_specs"]
    }
    sources = tuple(
        replace(
            row,
            path=originals_by_relative[
                Path(str(specs_by_id[row.motion_id]["source_path"]))
            ],
        )
        for row in loaded.sources
    )
    model_paths = {
        name: originals_by_relative[
            Path(str(raw["model_contract"][f"{name}_path"]))
        ]
        for name in ("mjcf", "urdf", "body_order")
    }
    translated = recipe_module.CanonicalMotionRecipe(
        path=requested,
        repo_root=resolved_root,
        raw=loaded.raw,
        ready=ready,
        sources=sources,
        marker_semantics=loaded.marker_semantics,
        marker_authority_path=authority_original,
        marker_authority_sha256=loaded.marker_authority_sha256,
        model_paths=model_paths,
        model_hashes=loaded.model_hashes,
    )
    bindings = {requested: recipe_sha}
    bindings.update(
        {original: row[2] for original, row in snapshots.items()}
    )
    # Reject changes between the input snapshot and return to the caller.
    for original, expected in bindings.items():
        if _sha256_file(original) != expected:
            raise NeutralReadyAdapterError(
                f"recipe input changed during stable decode: {original}"
            )
    return translated, bindings


def _source_root(
    source: recipe_module.MotionSource,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        pelvis_index = tuple(motion_player.RUNTIME_BODY_NAMES).index(
            "pelvis_link"
        )
    except ValueError as exc:  # pragma: no cover - import-time invariant
        raise NeutralReadyAdapterError(
            "runtime body order lacks pelvis_link"
        ) from exc
    root_pos = np.asarray(
        source.clip.body_pos_w[:, pelvis_index], dtype=np.float64
    )
    root_quat = np.asarray(
        source.clip.body_quat_w[:, pelvis_index], dtype=np.float64
    )
    expected = source.clip.n_frames
    if root_pos.shape != (expected, 3) or root_quat.shape != (expected, 4):
        raise NeutralReadyAdapterError(
            "block source pelvis arrays do not match its frame count"
        )
    return root_pos, root_quat


def _preprocess_scope(
    recipe: recipe_module.CanonicalMotionRecipe,
    source: recipe_module.MotionSource,
    scope: str,
) -> body_scope.BodyScopeResult:
    root_pos, root_quat = _source_root(source)
    common = {
        "source_joint_pos": source.clip.joint_pos,
        "source_root_pos_w": root_pos,
        "source_root_quat_w": root_quat,
        "joint_names": dynamics_gate.RUNTIME_JOINT_NAMES,
        "canonical_ready_root_pos_w": recipe.ready.root_pos_w,
        "canonical_ready_root_quat_w": recipe.ready.root_quat_wxyz,
    }
    if scope == "upper":
        return body_scope.preprocess_body_scope(
            "upper",
            **common,
            canonical_ready_joint_pos=recipe.ready.joint_pos,
        )
    if scope == "full":
        return body_scope.preprocess_body_scope("full", **common)
    raise NeutralReadyAdapterError(f"unsupported scope {scope!r}")


def reviewed_block_phase_frames(
    backhand: recipe_module.MotionSource,
    forehand: recipe_module.MotionSource,
    marker_semantics: Any,
    binding: BlockPhaseMapBinding,
) -> dict[str, int]:
    """Validate four lineage events without inventing FH behavior preference."""

    backhand_row = marker_semantics.row("bh_block")
    forehand_row = marker_semantics.row("fh_block_syn")
    early, late = (int(value) for value in backhand_row.ge80_seed)
    if tuple(int(value) for value in forehand_row.ge80_seed) != (early, late):
        raise NeutralReadyAdapterError(
            "BH/FH legacy ge80 seed spans disagree"
        )
    if backhand_row.nominal_event is None:
        raise NeutralReadyAdapterError(
            "backhand block marker authority row lacks its nominal event"
        )
    construction = forehand_row.construction_marker
    if construction is None:
        raise NeutralReadyAdapterError(
            "synthetic forehand marker authority row lacks its construction marker"
        )
    if int(construction.annotation_frame) != int(backhand_row.nominal_event):
        raise NeutralReadyAdapterError(
            "synthetic construction annotation disagrees with the donor "
            "nominal event"
        )
    verified_binding = load_block_phase_map_binding(
        binding.authority_path, binding.expected_authority_sha256
    )
    if verified_binding != binding:
        raise NeutralReadyAdapterError(
            "caller phase-map values disagree with parsed authority bytes"
        )
    frames = binding.frames
    if binding.source_motion_sha256 != backhand.sha256:
        raise NeutralReadyAdapterError(
            "phase-map authority source hash disagrees with block source"
        )
    if (
        frames["opportunity_start"] != early
        or frames["opportunity_end"] != late
    ):
        raise NeutralReadyAdapterError(
            "reviewed opportunity start/end disagree with protected span"
        )
    if int(backhand_row.nominal_event) != frames["nominal_event"]:
        raise NeutralReadyAdapterError(
            "marker authority nominal event disagrees with reviewed lineage"
        )
    if int(construction.donor_preferred_frame) != frames[
        "construction_donor_preferred"
    ]:
        raise NeutralReadyAdapterError(
            "marker authority construction donor-preferred frame disagrees "
            "with reviewed lineage"
        )
    if binding.synthetic_behavior_preferred is not None:
        raise NeutralReadyAdapterError(
            "synthetic behavior preferred must remain unresolved/null"
        )
    return frames


def measure_exact_model_binding(
    recipe: recipe_module.CanonicalMotionRecipe,
    backend: face.MujocoRightRacketBackend,
) -> neutral.ExactModelBinding:
    """Measure pins for a separate review step; this does not approve them."""

    if type(backend) is not face.MujocoRightRacketBackend:
        raise NeutralReadyAdapterError(
            "real adapter requires the canonical MuJoCo backend type"
        )
    if tuple(backend.joint_names) != tuple(dynamics_gate.RUNTIME_JOINT_NAMES):
        raise NeutralReadyAdapterError(
            "real adapter backend joint order is not the canonical 31"
        )
    compiled_sha = dynamics_gate.compiled_model_signature(backend.model)
    compiled_mjb_sha = neutral._compiled_mjb_sha256(
        backend.model, backend._mujoco
    )
    limits_sha, _ = neutral.backend_limits_digest(backend)
    closure_sha, _ = neutral.mjcf_include_closure_digest(
        recipe.model_paths["mjcf"]
    )
    backend_contract_sha, _ = neutral.exact_backend_model_contract_digest(
        backend,
        compiled_model_sha256=compiled_sha,
        compiled_mjb_sha256=compiled_mjb_sha,
        backend_limits_sha256=limits_sha,
        mjcf_dependency_closure_sha256=closure_sha,
    )
    return neutral.ExactModelBinding(
        mjcf_path=recipe.model_paths["mjcf"],
        expected_mjcf_sha256=recipe.model_hashes["mjcf"],
        expected_compiled_model_sha256=compiled_sha,
        urdf_path=recipe.model_paths["urdf"],
        expected_urdf_sha256=recipe.model_hashes["urdf"],
        expected_backend_limits_sha256=limits_sha,
        expected_backend_model_contract_sha256=backend_contract_sha,
    )


def _contact(
    *,
    scope: str,
    phase: str,
    face_name: str,
    frame: int,
    joint_pos: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    source_path: Path,
    source_sha256: str,
    backend: face.MujocoRightRacketBackend,
) -> neutral.ContactPose:
    site_pos, site_rotation = backend.site_pose(
        joint_pos, root_pos_w, root_quat_w
    )
    provisional = neutral.ContactPose(
        scope=scope,
        phase=phase,
        face_name=face_name,
        joint_pos=np.asarray(joint_pos, dtype=np.float64).copy(),
        root_pos_w=np.asarray(root_pos_w, dtype=np.float64).copy(),
        root_quat_w=np.asarray(root_quat_w, dtype=np.float64).copy(),
        site_pos_w=np.asarray(site_pos, dtype=np.float64).copy(),
        site_rotation_w=np.asarray(site_rotation, dtype=np.float64).copy(),
        signed_face_normal_w=neutral._unit(
            np.asarray(site_rotation, dtype=np.float64)[:, 1],
            "adapter signed face normal",
        ),
        source_motion_path=source_path,
        source_motion_sha256=source_sha256,
        source_frame_index=frame,
        pose_content_sha256="0" * 64,
        pair_contract_sha256="0" * 64,
    )
    return neutral.ContactPose(
        **{
            **provisional.__dict__,
            "pose_content_sha256": neutral.contact_pose_digest(provisional),
        }
    )


def _proof_rows(
    contacts: tuple[neutral.ContactPose, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "label": row.label,
            "source_frame_index": int(row.source_frame_index),
            "pose_content_sha256": row.pose_content_sha256,
            "pair_contract_sha256": row.pair_contract_sha256,
            "joint_pos_sha256": neutral.array_content_sha256(row.joint_pos),
            "root_pos_w_sha256": neutral.array_content_sha256(row.root_pos_w),
            "root_quat_w_sha256": neutral.array_content_sha256(
                row.root_quat_w
            ),
            "site_pos_w_sha256": neutral.array_content_sha256(row.site_pos_w),
            "site_rotation_w_sha256": neutral.array_content_sha256(
                row.site_rotation_w
            ),
            "signed_face_normal_w_sha256": neutral.array_content_sha256(
                row.signed_face_normal_w
            ),
        }
        for row in contacts
    ]


def _file_binding(
    role: str,
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    resolved = path.expanduser().absolute()
    if not resolved.is_file() or resolved.is_symlink():
        raise NeutralReadyAdapterError(
            f"{role} must be a real regular file: {resolved}"
        )
    actual = _sha256_file(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise NeutralReadyAdapterError(
            f"{role} changed after its stable parse snapshot"
        )
    return {
        "role": role,
        "path": str(resolved),
        "sha256": actual,
    }


def load_real_neutral_ready_inputs(
    recipe_path: str | Path,
    *,
    expected_recipe_sha256: str,
    repo_root: str | Path | None = None,
    backend: face.MujocoRightRacketBackend | None = None,
    model_binding: neutral.ExactModelBinding | None = None,
    phase_map_binding: BlockPhaseMapBinding | None = None,
    face_config: face.FaceManifoldConfig | None = None,
) -> RealNeutralReadyInputs:
    """Reconstruct 16 lineage-only block pose rows from pinned sources."""

    recipe, recipe_input_bindings = _snapshot_recipe_inputs(
        recipe_path,
        repo_root=repo_root,
        expected_recipe_sha256=expected_recipe_sha256,
    )
    if backend is None:
        backend = face.MujocoRightRacketBackend(
            recipe.model_paths["mjcf"],
            dynamics_gate.RUNTIME_JOINT_NAMES,
            urdf_path=recipe.model_paths["urdf"],
        )
    if model_binding is None:
        raise NeutralReadyAdapterError(
            "real adapter requires independently reviewed compiled-model, "
            "backend-limit, and backend-contract pins; run the CLI's "
            "--measure-model-binding mode first, review/fix the values, then "
            "supply all three expected pins"
        )
    if (
        Path(model_binding.mjcf_path).expanduser().absolute()
        != recipe.model_paths["mjcf"].absolute()
        or model_binding.expected_mjcf_sha256
        != recipe.model_hashes["mjcf"]
        or Path(model_binding.urdf_path).expanduser().absolute()
        != recipe.model_paths["urdf"].absolute()
        or model_binding.expected_urdf_sha256
        != recipe.model_hashes["urdf"]
    ):
        raise NeutralReadyAdapterError(
            "reviewed model binding does not match the recipe MJCF/URDF pins"
        )
    backhand = recipe.source("bh_block")
    forehand = recipe.source("fh_block_syn")
    if (
        backhand.path != forehand.path
        or backhand.sha256 != forehand.sha256
    ):
        raise NeutralReadyAdapterError(
            "BH and synthetic FH block do not share exact source bytes"
        )
    if phase_map_binding is None:
        raise NeutralReadyAdapterError(
            "real adapter requires reviewed four-event lineage authority; "
            "synthetic behavior preference may not be invented"
        )
    phase_frames = reviewed_block_phase_frames(
        backhand, forehand, recipe.marker_semantics, phase_map_binding
    )

    velocity_fraction = float(
        recipe.raw["time_law"]["joint_velocity_limit_fraction"]
    )
    canonical_face_config = face.FaceManifoldConfig(
        mode="normal",
        velocity_limit_fraction=velocity_fraction,
    )
    if face_config is not None and asdict(face_config) != asdict(
        canonical_face_config
    ):
        raise NeutralReadyAdapterError(
            "publication adapter face config is code-fixed; caller overrides "
            "may be used only outside the verified source-proof path"
        )
    config = canonical_face_config
    face_contract = forehand.face_manifold
    if face_contract is None:
        raise NeutralReadyAdapterError(
            "synthetic forehand lacks its face-manifold recipe"
        )
    synthetic_construction = recipe.marker_semantics.row(
        "fh_block_syn"
    ).construction_marker
    if synthetic_construction is None:
        raise NeutralReadyAdapterError(
            "synthetic forehand marker authority row lacks its construction marker"
        )
    solve_start, solve_end = (
        int(value) for value in synthetic_construction.solve_span
    )
    solver_anchor = int(synthetic_construction.annotation_frame)
    if solver_anchor != phase_frames["nominal_event"]:
        raise NeutralReadyAdapterError(
            "face solver annotation disagrees with reviewed nominal event: "
            f"expected f{phase_frames['nominal_event']}, "
            f"got f{solver_anchor}"
        )

    contacts: list[neutral.ContactPose] = []
    source_array_rows: dict[str, Any] = {}
    face_summaries: dict[str, Any] = {}
    for scope in neutral.SCOPES:
        scoped = _preprocess_scope(recipe, backhand, scope)
        result = face.solve_face_flipped_window(
            scoped.joint_pos[solve_start : solve_end + 1],
            scoped.root_pos_w[solve_start : solve_end + 1],
            scoped.root_quat_w[solve_start : solve_end + 1],
            recipe.ready.joint_pos,
            fps=float(backhand.clip.fps),
            backend=backend,
            config=config,
            frame_indices=tuple(range(solve_start, solve_end + 1)),
            anchor_index=solver_anchor - solve_start,
        )
        synthetic_joint_pos = np.asarray(
            scoped.joint_pos, dtype=np.float64
        ).copy()
        synthetic_joint_pos[solve_start : solve_end + 1] = result.joint_pos
        face_summaries[scope] = result.summary()
        for phase in neutral.PHASES:
            frame = phase_frames[phase]
            source_array_rows[f"{scope}:{phase}"] = {
                "source_frame_index": frame,
                "raw_source_joint_pos_sha256": neutral.array_content_sha256(
                    backhand.clip.joint_pos[frame]
                ),
                "scoped_bh_joint_pos_sha256": neutral.array_content_sha256(
                    scoped.joint_pos[frame]
                ),
                "derived_fh_joint_pos_sha256": neutral.array_content_sha256(
                    synthetic_joint_pos[frame]
                ),
                "scoped_root_pos_w_sha256": neutral.array_content_sha256(
                    scoped.root_pos_w[frame]
                ),
                "scoped_root_quat_w_sha256": neutral.array_content_sha256(
                    scoped.root_quat_w[frame]
                ),
            }
            pair = [
                _contact(
                    scope=scope,
                    phase=phase,
                    face_name="bh",
                    frame=frame,
                    joint_pos=scoped.joint_pos[frame],
                    root_pos_w=scoped.root_pos_w[frame],
                    root_quat_w=scoped.root_quat_w[frame],
                    source_path=backhand.path,
                    source_sha256=backhand.sha256,
                    backend=backend,
                ),
                _contact(
                    scope=scope,
                    phase=phase,
                    face_name="fh",
                    frame=frame,
                    joint_pos=synthetic_joint_pos[frame],
                    root_pos_w=scoped.root_pos_w[frame],
                    root_quat_w=scoped.root_quat_w[frame],
                    source_path=backhand.path,
                    source_sha256=backhand.sha256,
                    backend=backend,
                ),
            ]
            pair_sha = neutral.contact_pair_contract_digest(pair[0], pair[1])
            contacts.extend(
                neutral.ContactPose(
                    **{
                        **row.__dict__,
                        "pair_contract_sha256": pair_sha,
                    }
                )
                for row in pair
            )
    ordered_contacts = tuple(contacts)
    if len(ordered_contacts) != (
        len(neutral.SCOPES) * len(neutral.PHASES) * len(neutral.FACES)
    ):
        raise NeutralReadyAdapterError(
            "real contact reconstruction did not produce exactly 16 rows"
        )

    file_bindings = [
        _file_binding("adapter_tool", Path(__file__).resolve()),
        _file_binding(
            "recipe",
            recipe.path,
            expected_sha256=recipe_input_bindings[recipe.path],
        ),
        _file_binding(
            "body_scope_tool", Path(body_scope.__file__).resolve()
        ),
        _file_binding("face_tool", Path(face.__file__).resolve()),
        _file_binding(
            "motion_source",
            backhand.path,
            expected_sha256=recipe_input_bindings[backhand.path],
        ),
        _file_binding(
            "canonical_ready",
            recipe.ready.path,
            expected_sha256=recipe_input_bindings[recipe.ready.path],
        ),
        _file_binding(
            "mjcf_root",
            recipe.model_paths["mjcf"],
            expected_sha256=recipe_input_bindings[
                recipe.model_paths["mjcf"]
            ],
        ),
        _file_binding(
            "urdf",
            recipe.model_paths["urdf"],
            expected_sha256=recipe_input_bindings[
                recipe.model_paths["urdf"]
            ],
        ),
        _file_binding(
            "phase_authority",
            Path(phase_map_binding.authority_path).expanduser().absolute(),
            expected_sha256=(
                phase_map_binding.expected_authority_sha256
            ),
        ),
        _file_binding(
            "neutral_solver_tool", Path(neutral.__file__).resolve()
        ),
        _file_binding(
            "motion_recipe_tool", Path(recipe_module.__file__).resolve()
        ),
        _file_binding(
            "dynamics_gate_tool", Path(dynamics_gate.__file__).resolve()
        ),
        _file_binding(
            "motion_player_tool", Path(motion_player.__file__).resolve()
        ),
        _file_binding(
            "motion_npz_decoder_tool",
            Path(audit_motion_npz.__file__).resolve(),
        ),
        _file_binding(
            "motion_kinematics_contract_tool",
            Path(motion_kinematics_contract.__file__).resolve(),
        ),
        _file_binding(
            "racket_geometry_contract_tool",
            Path(racket_geometry_contract.__file__).resolve(),
        ),
        _file_binding(
            "body_order",
            recipe.model_paths["body_order"],
            expected_sha256=recipe_input_bindings[
                recipe.model_paths["body_order"]
            ],
        ),
    ]
    proof_receipt: Mapping[str, Any] = {
        "schema_version": 1,
        "builder_contract": (
            "canonical_block_lineage_pose_reconstruction_v2"
        ),
        "file_bindings": file_bindings,
        "contact_matrix_sha256": neutral.contact_matrix_digest(
            ordered_contacts
        ),
        "contact_row_count": len(ordered_contacts),
        "phase_source_frames": phase_frames,
        "scope_contract": {
            "upper": (
                "canonical_body_scope.project_upper_body_scope_from_"
                "content_pinned_schema2"
            ),
            "full": (
                "canonical_body_scope.align_full_body_scope_from_"
                "content_pinned_schema2"
            ),
            "source_array_rows": source_array_rows,
            "legacy_recipe_source_anchor_frame": (
                phase_frames["nominal_event"]
            ),
            "legacy_anchor_semantics": (
                "nominal_air_swing_event_not_behavior_preferred_or_contact"
            ),
        },
        "face_contract": {
            "mode": "signed_raw_plus_y_flip_normal_hard_inplane_free",
            "solve_span": [solve_start, solve_end],
            "solver_anchor_annotation_frame": solver_anchor,
            "lineage_pose_frames": phase_frames,
            "synthetic_behavior_preferred_frame": None,
            "construction_donor_preferred_frame": phase_frames[
                "construction_donor_preferred"
            ],
            "post_retime_forehand_behavior_rescan_required": True,
            "contact_truth_observed": False,
            "single_axis_pi_overlay": False,
            "active_joint_names": list(face.RIGHT_STRIKE_CHAIN),
            "config": asdict(config),
            "scope_solver_summaries": face_summaries,
        },
        "model_binding": {
            "mjcf_path": str(
                Path(model_binding.mjcf_path).expanduser().absolute()
            ),
            "mjcf_sha256": model_binding.expected_mjcf_sha256,
            "compiled_model_sha256": (
                model_binding.expected_compiled_model_sha256
            ),
            "urdf_path": str(
                Path(model_binding.urdf_path).expanduser().absolute()
            ),
            "urdf_sha256": model_binding.expected_urdf_sha256,
            "backend_limits_sha256": (
                model_binding.expected_backend_limits_sha256
            ),
            "backend_model_contract_sha256": (
                model_binding.expected_backend_model_contract_sha256
            ),
            "racket_site_name": face.CANONICAL_RACKET_SITE,
            "normal_convention": neutral.NORMAL_CONVENTION,
        },
        "rows": _proof_rows(ordered_contacts),
    }
    proof = neutral.seal_contact_source_proof(
        proof_receipt, ordered_contacts
    )
    return RealNeutralReadyInputs(
        recipe=recipe,
        backend=backend,
        contacts=ordered_contacts,
        contact_source_proof=proof,
        ready_binding=neutral.ReadySourceBinding(
            path=recipe.ready.path,
            expected_sha256=recipe.ready.sha256,
        ),
        model_binding=model_binding,
    )


def verify_real_contact_source_reconstruction(
    proof: neutral.ContactSourceProof,
    contacts: tuple[neutral.ContactPose, ...],
    *,
    backend: face.RightRacketBackend,
    model_binding: neutral.ExactModelBinding,
) -> None:
    """Independently rerun source decode/scope/face/FK and compare every row."""

    if type(backend) is not face.MujocoRightRacketBackend:
        raise NeutralReadyAdapterError(
            "trusted reconstruction requires canonical MuJoCo backend"
        )
    raw_receipt = proof.receipt
    try:
        file_rows = raw_receipt["file_bindings"]
    except (KeyError, TypeError) as exc:
        raise NeutralReadyAdapterError(
            "source proof lacks file bindings"
        ) from exc
    role_paths: dict[str, Path] = {}
    role_hashes: dict[str, str] = {}
    for row in file_rows:
        role = str(row["role"])
        if role in role_paths:
            raise NeutralReadyAdapterError(
                f"source proof duplicates file role {role!r}"
            )
        role_paths[role] = Path(str(row["path"])).expanduser().absolute()
        role_hashes[role] = str(row["sha256"])
    expected_tool_paths = {
        "adapter_tool": Path(__file__).resolve(),
        "body_scope_tool": Path(body_scope.__file__).resolve(),
        "face_tool": Path(face.__file__).resolve(),
        "neutral_solver_tool": Path(neutral.__file__).resolve(),
        "motion_recipe_tool": Path(recipe_module.__file__).resolve(),
        "dynamics_gate_tool": Path(dynamics_gate.__file__).resolve(),
        "motion_player_tool": Path(motion_player.__file__).resolve(),
        "motion_npz_decoder_tool": Path(audit_motion_npz.__file__).resolve(),
        "motion_kinematics_contract_tool": Path(
            motion_kinematics_contract.__file__
        ).resolve(),
        "racket_geometry_contract_tool": Path(
            racket_geometry_contract.__file__
        ).resolve(),
    }
    for role, expected_path in expected_tool_paths.items():
        if role_paths.get(role) != expected_path:
            raise NeutralReadyAdapterError(
                f"source proof {role} is not the trusted implementation"
            )
    recipe_path = role_paths.get("recipe")
    if recipe_path is None:
        raise NeutralReadyAdapterError("source proof lacks recipe path")
    phase_authority_path = role_paths.get("phase_authority")
    phase_rows = raw_receipt.get("phase_source_frames")
    if phase_authority_path is None or "phase_authority" not in role_hashes:
        raise NeutralReadyAdapterError(
            "source proof lacks reviewed phase-map authority"
        )
    try:
        phase_binding = load_block_phase_map_binding(
            phase_authority_path, role_hashes["phase_authority"]
        )
    except (TypeError, ValueError) as exc:
        raise NeutralReadyAdapterError(
            f"source proof phase map cannot be reconstructed: {exc}"
        ) from exc
    if phase_rows != phase_binding.frames:
        raise NeutralReadyAdapterError(
            "source proof phase rows disagree with authority bytes"
        )
    face_contract = raw_receipt.get("face_contract")
    if not isinstance(face_contract, Mapping):
        raise NeutralReadyAdapterError(
            "source proof face contract is malformed"
        )
    raw_config = face_contract.get("config")
    if not isinstance(raw_config, Mapping):
        raise NeutralReadyAdapterError(
            "source proof lacks exact face solver config"
        )
    try:
        config = face.FaceManifoldConfig(**dict(raw_config))
    except (TypeError, ValueError) as exc:
        raise NeutralReadyAdapterError(
            f"source proof face config cannot be reconstructed: {exc}"
        ) from exc
    rebuilt = load_real_neutral_ready_inputs(
        recipe_path,
        expected_recipe_sha256=role_hashes["recipe"],
        repo_root=recipe_path.parents[1],
        backend=backend,
        model_binding=model_binding,
        phase_map_binding=phase_binding,
        face_config=config,
    )
    if (
        neutral.contact_matrix_digest(rebuilt.contacts)
        != neutral.contact_matrix_digest(contacts)
        or rebuilt.contact_source_proof.payload_sha256
        != proof.payload_sha256
        or json.dumps(
            rebuilt.contact_source_proof.receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        != json.dumps(
            proof.receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    ):
        raise NeutralReadyAdapterError(
            "trusted source replay did not exactly reproduce contacts/proof"
        )


def solve_real_neutral_ready_challenger(
    recipe_path: str | Path,
    *,
    expected_recipe_sha256: str,
    repo_root: str | Path | None = None,
    backend: face.MujocoRightRacketBackend | None = None,
    model_binding: neutral.ExactModelBinding | None = None,
    phase_map_binding: BlockPhaseMapBinding | None = None,
    face_config: face.FaceManifoldConfig | None = None,
    ready_config: neutral.NeutralReadyConfig | None = None,
) -> neutral.NeutralReadyResult:
    """Load real inputs, then run the restricted neutral-ready solver."""

    loaded = load_real_neutral_ready_inputs(
        recipe_path,
        expected_recipe_sha256=expected_recipe_sha256,
        repo_root=repo_root,
        backend=backend,
        model_binding=model_binding,
        phase_map_binding=phase_map_binding,
        face_config=face_config,
    )
    return neutral.solve_neutral_ready_candidate(
        loaded.recipe.ready.joint_pos,
        loaded.recipe.ready.root_pos_w,
        loaded.recipe.ready.root_quat_wxyz,
        loaded.contacts,
        backend=loaded.backend,
        ready_binding=loaded.ready_binding,
        model_binding=loaded.model_binding,
        contact_source_proof=loaded.contact_source_proof,
        config=ready_config,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one exact, no-clobber diagnostic neutral-ready challenger"
        )
    )
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--measure-model-binding", action="store_true")
    parser.add_argument("--expected-compiled-model-sha256")
    parser.add_argument("--expected-backend-limits-sha256")
    parser.add_argument("--expected-backend-model-contract-sha256")
    parser.add_argument("--phase-authority", type=Path)
    parser.add_argument("--expected-phase-authority-sha256")
    parser.add_argument("--random-restarts", type=int, default=8)
    parser.add_argument("--antipodal-circle-samples", type=int, default=36)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        recipe, _ = _snapshot_recipe_inputs(
            args.recipe,
            repo_root=args.repo_root,
            expected_recipe_sha256=args.expected_recipe_sha256,
        )
        backend = face.MujocoRightRacketBackend(
            recipe.model_paths["mjcf"],
            dynamics_gate.RUNTIME_JOINT_NAMES,
            urdf_path=recipe.model_paths["urdf"],
        )
        if args.measure_model_binding:
            measured = measure_exact_model_binding(recipe, backend)
            print(
                json.dumps(
                    {
                        "verdict": "MEASUREMENT_ONLY_REQUIRES_REVIEW",
                        "expected_compiled_model_sha256": (
                            measured.expected_compiled_model_sha256
                        ),
                        "expected_backend_limits_sha256": (
                            measured.expected_backend_limits_sha256
                        ),
                        "expected_backend_model_contract_sha256": (
                            measured.expected_backend_model_contract_sha256
                        ),
                        "mjcf_sha256": measured.expected_mjcf_sha256,
                        "urdf_sha256": measured.expected_urdf_sha256,
                        "publication_attempted": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.out is None:
            raise NeutralReadyAdapterError(
                "--out is required outside --measure-model-binding mode"
            )
        pins = (
            args.expected_compiled_model_sha256,
            args.expected_backend_limits_sha256,
            args.expected_backend_model_contract_sha256,
        )
        if any(value is None for value in pins):
            raise NeutralReadyAdapterError(
                "solve mode requires all three independently reviewed "
                "--expected-* model pins"
            )
        reviewed_binding = neutral.ExactModelBinding(
            mjcf_path=recipe.model_paths["mjcf"],
            expected_mjcf_sha256=recipe.model_hashes["mjcf"],
            expected_compiled_model_sha256=str(pins[0]),
            urdf_path=recipe.model_paths["urdf"],
            expected_urdf_sha256=recipe.model_hashes["urdf"],
            expected_backend_limits_sha256=str(pins[1]),
            expected_backend_model_contract_sha256=str(pins[2]),
        )
        phase_values = (
            args.phase_authority,
            args.expected_phase_authority_sha256,
        )
        if any(value is None for value in phase_values):
            raise NeutralReadyAdapterError(
                "solve mode requires phase authority path/hash and all four "
                "event frames decoded from those reviewed bytes"
            )
        reviewed_phase_map = load_block_phase_map_binding(
            args.phase_authority,
            str(args.expected_phase_authority_sha256),
        )
        result = solve_real_neutral_ready_challenger(
            args.recipe,
            expected_recipe_sha256=args.expected_recipe_sha256,
            repo_root=args.repo_root,
            backend=backend,
            model_binding=reviewed_binding,
            phase_map_binding=reviewed_phase_map,
            ready_config=neutral.NeutralReadyConfig(
                random_restarts=args.random_restarts,
                antipodal_circle_samples=args.antipodal_circle_samples,
            ),
        )
        published = neutral.publish_neutral_ready_candidate(result, args.out)
    except (
        NeutralReadyAdapterError,
        neutral.NeutralReadyError,
        recipe_module.MotionRecipeError,
        face.FaceManifoldError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "verdict": "INCOMPLETE_FAIL_CLOSED",
                    "error": str(exc),
                    "training_authorized": False,
                    "deploy_authorized": False,
                    "hardware_authorized": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "verdict": result.receipt["verdict"],
                "candidate_directory": str(published.directory),
                "candidate_npz_sha256": published.candidate_npz_sha256,
                "receipt_json_sha256": published.receipt_json_sha256,
                "training_authorized": False,
                "deploy_authorized": False,
                "hardware_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
