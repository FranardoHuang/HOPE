"""Exact native-MuJoCo port of the Isaac ActionBall robot/table guard.

The Isaac rule is a conservative pose keep-out, not a resolved-contact test:
43 pinned robot collision-component OBBs and one live racket-blade OBB are
broadened to world AABBs and compared with the five inflated table-assembly
AABBs at every physics substep.  This module consumes the same immutable
component artifact and fails closed if any source identity drifts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_TERMINATION_CONFIG = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)
EXPECTED_ISAAC_TERMINATION_CONFIG_SHA256 = (
    "a012013c39d769bb3be5383e821fa52edaf7cc973bfad81f9ff2435423357f42"
)
ISAAC_TERMINATION_CALLABLES = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/terminations.py"
)
EXPECTED_ISAAC_TERMINATION_CALLABLES_SHA256 = (
    "dfb6fc870a37d4af4d5c5fa9fa05dd854d0b64d4bbe72901b5585a3d3968b7d9"
)
ISAAC_ACTION_LATCH = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_actions.py"
)
EXPECTED_ISAAC_ACTION_LATCH_SHA256 = (
    "f42c1ab18eafe946a4b066198d711eda40d66300002123d94a5790a2e6d40b79"
)
CANONICAL_MJCF = (
    REPO_ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
EXPECTED_CANONICAL_MJCF_SHA256 = (
    "70c4fd6534f259d12990cef731cfdf8f8557f92fd0ca81cc4fc1c75a39336c0a"
)
MUJOCO_IDENTITY_MANIFEST = REPO_ROOT / "configs/a3_mujoco_identity_v2_20260803.json"
EXPECTED_MUJOCO_IDENTITY_MANIFEST_SHA256 = (
    "b8fc5deaaff8d213c2d077a0e7892b30d7f5a6c77c3d06dc029e3a2616d54d91"
)
CANONICAL_MUJOCO_IDENTITY_PY = (
    REPO_ROOT / "hope_training/whole_body_tracking/scripts/canonical_mujoco_identity.py"
)
EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256 = (
    "e43609988a371a76e5daab7545c608338ba159100c52cb50dc61b12a872fe2e1"
)
EXPECTED_PORTABLE_MUJOCO_IDENTITY_SHA256 = (
    "472219ae346d9217b7d1af860d462a18d6ed8507c5cbb9c0f1ddcd6f964dfd7a"
)
COLLISION_PROXY_ARTIFACT = (
    REPO_ROOT
    / "configs/a3_table_collision_proxy_20260731/"
    "a3_table_collision_components.v1.json"
)
EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256 = (
    "23e2f5b30bbba909f1123dc41f6c010354122b9837b4ef133a1c285a2cd78ca8"
)
EXPECTED_ACTION_BALL_TABLE_GEOMETRY_SHA256 = (
    "f6aab7524a3b6583ae7ced8da8b2b5d9d1bbe0ea0c72b3b688fefaf6ff66cc6a"
)

TABLE_GUARD_MARGIN_M = 0.02
COMPONENT_WORLD_AABB_GUARD_M = 1.0e-6
RACKET_BODY_NAME = "right_wrist_yaw_Link"
RACKET_BLADE_CENTER_OFFSET_WRIST_M = np.asarray(
    (0.206194, 0.025474, 0.028020), dtype=np.float64
)
RACKET_BLADE_LOCAL_HALF_AXES_M = np.diag(
    np.asarray((0.082, 0.008, 0.082), dtype=np.float64)
)
TABLE_ASSEMBLY_ROLES = ("top", "keepout", "net", "post_left", "post_right")
TABLE_CONTACT_BODY_NAMES = (
    "pelvis_link",
    "left_hip_pitch_Link",
    "right_hip_pitch_Link",
    "waist_yaw_Link",
    "left_hip_roll_Link",
    "right_hip_roll_Link",
    "waist_roll_Link",
    "left_hip_yaw_Link",
    "right_hip_yaw_Link",
    "torso_Link",
    "left_knee_Link",
    "right_knee_Link",
    "head_yaw_Link",
    "left_shoulder_pitch_Link",
    "right_shoulder_pitch_Link",
    "left_ankle_pitch_Link",
    "right_ankle_pitch_Link",
    "head_pitch_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_shoulder_yaw_Link",
    "right_shoulder_yaw_Link",
    "left_elbow_Link",
    "right_elbow_Link",
    "left_wrist_roll_Link",
    "right_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "right_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
)


class TableTerminationContractError(RuntimeError):
    """The exact Isaac robot/table termination cannot be reproduced."""


def _strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _sha256_file(path: Path, label: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise TableTerminationContractError(f"cannot read {label} source") from exc


def verify_isaac_source_authority() -> dict[str, str]:
    """Reopen the exact Isaac config, predicate, and sticky-latch sources."""

    config_sha = _sha256_file(ISAAC_TERMINATION_CONFIG, "Isaac termination config")
    callable_sha = _sha256_file(
        ISAAC_TERMINATION_CALLABLES, "Isaac termination callables"
    )
    action_latch_sha = _sha256_file(ISAAC_ACTION_LATCH, "Isaac action latch")
    if config_sha != EXPECTED_ISAAC_TERMINATION_CONFIG_SHA256:
        raise TableTerminationContractError(
            "Isaac robot/table termination config SHA-256 drifted"
        )
    if callable_sha != EXPECTED_ISAAC_TERMINATION_CALLABLES_SHA256:
        raise TableTerminationContractError(
            "Isaac robot/table termination callables SHA-256 drifted"
        )
    if action_latch_sha != EXPECTED_ISAAC_ACTION_LATCH_SHA256:
        raise TableTerminationContractError(
            "Isaac robot/table action-latch SHA-256 drifted"
        )
    return {
        "config_sha256": config_sha,
        "callables_sha256": callable_sha,
        "action_latch_sha256": action_latch_sha,
    }


def _owner_frame_contract(mujoco: Any, model: Any) -> dict[str, Any]:
    """Serialize the exact local frames that give the 43 OBB rows meaning."""

    body_ids: list[int] = []
    for name in TABLE_CONTACT_BODY_NAMES:
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        if body_id <= 0:
            raise TableTerminationContractError(
                f"MuJoCo model is missing exact A3 table-guard body {name!r}"
            )
        body_ids.append(body_id)
    if len(set(body_ids)) != len(TABLE_CONTACT_BODY_NAMES):
        raise TableTerminationContractError(
            "MuJoCo table-guard body mapping is not a 32-body bijection"
        )
    selected_names = dict(zip(body_ids, TABLE_CONTACT_BODY_NAMES))
    rows: list[dict[str, Any]] = []
    for name, body_id in zip(TABLE_CONTACT_BODY_NAMES, body_ids):
        parent_id = int(model.body_parentid[body_id])
        if parent_id == 0:
            parent_name = "__world__"
        else:
            parent_name = selected_names.get(parent_id)
            if parent_name is None:
                raise TableTerminationContractError(
                    f"table-guard body {name!r} has an unregistered parent"
                )
        position = np.asarray(model.body_pos[body_id], dtype=np.float64)
        quaternion = np.asarray(model.body_quat[body_id], dtype=np.float64)
        if (
            position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
            or not np.isclose(
                float(np.linalg.norm(quaternion)), 1.0, rtol=0.0, atol=1.0e-12
            )
        ):
            raise TableTerminationContractError(
                f"table-guard body {name!r} has a malformed owner-local frame"
            )
        rows.append(
            {
                "name": name,
                "parent": parent_name,
                "body_pos_m": position.tolist(),
                "body_quat_wxyz": quaternion.tolist(),
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "a3_table_collision_owner_local_frame_contract_v1",
        "body_rows": rows,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    payload["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    payload["body_ids"] = body_ids
    return payload


def _assert_owner_frame_contract_equal(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    if (
        expected.get("kind") != "a3_table_collision_owner_local_frame_contract_v1"
        or observed.get("kind") != expected.get("kind")
        or expected.get("body_rows") != observed.get("body_rows")
        or expected.get("content_sha256") != observed.get("content_sha256")
    ):
        raise TableTerminationContractError(
            "live MuJoCo owner-local body frames differ from the pre-registered plant"
        )


def _load_canonical_identity_module() -> Any:
    verifier_sha = _sha256_file(
        CANONICAL_MUJOCO_IDENTITY_PY, "canonical MuJoCo identity verifier"
    )
    if verifier_sha != EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256:
        raise TableTerminationContractError(
            "canonical MuJoCo identity verifier SHA-256 drifted"
        )
    module_name = "_table_termination_canonical_mujoco_identity"
    spec = importlib.util.spec_from_file_location(
        module_name, CANONICAL_MUJOCO_IDENTITY_PY
    )
    if spec is None or spec.loader is None:
        raise TableTerminationContractError(
            "cannot import canonical MuJoCo identity verifier"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - imported verifier is fail-closed
        raise TableTerminationContractError(
            "cannot load canonical MuJoCo identity verifier"
        ) from exc
    return module


@lru_cache(maxsize=2)
def _verified_registered_owner_frames(
    mujoco: Any, selected_root: str
) -> tuple[dict[str, Any], str, str]:
    """Compile the registered base once per process/toolchain and retain no model."""

    identity_module = _load_canonical_identity_module()
    verified = identity_module.verify_exact_mujoco_identity(
        mjcf_path=selected_root,
        expected_manifest_path=MUJOCO_IDENTITY_MANIFEST,
        trusted_expected_manifest_sha256=EXPECTED_MUJOCO_IDENTITY_MANIFEST_SHA256,
    )
    if verified.portable_identity_sha256 != EXPECTED_PORTABLE_MUJOCO_IDENTITY_SHA256:
        raise TableTerminationContractError(
            "portable MuJoCo plant identity SHA-256 drifted"
        )
    frames = verified.consume_verified_model(
        lambda base_model: _owner_frame_contract(mujoco, base_model)
    )
    return (
        frames,
        verified.portable_identity_sha256,
        verified.verification_receipt_sha256,
    )


def bind_pre_registered_owner_frames(
    mujoco: Any, model: Any, mjcf_path: Path | str
) -> dict[str, Any]:
    """Bind the live augmented scene to the registered base plant and OBB frames."""

    selected_root = Path(mjcf_path).expanduser().resolve()
    registered_root = CANONICAL_MJCF.resolve()
    if selected_root != registered_root:
        raise TableTerminationContractError(
            "exact robot/table termination requires the pre-registered root MJCF path"
        )
    if (
        _sha256_file(selected_root, "canonical root MJCF")
        != EXPECTED_CANONICAL_MJCF_SHA256
    ):
        raise TableTerminationContractError("canonical root MJCF SHA-256 drifted")
    if (
        _sha256_file(MUJOCO_IDENTITY_MANIFEST, "MuJoCo identity manifest")
        != EXPECTED_MUJOCO_IDENTITY_MANIFEST_SHA256
    ):
        raise TableTerminationContractError("MuJoCo identity manifest SHA-256 drifted")
    try:
        (
            expected_frames,
            portable_identity_sha256,
            verification_receipt_sha256,
        ) = _verified_registered_owner_frames(
            mujoco,
            str(selected_root),
        )
    except TableTerminationContractError:
        raise
    except Exception as exc:  # noqa: BLE001 - verifier exposes its own error type
        raise TableTerminationContractError(
            "pre-registered portable MuJoCo plant identity did not verify"
        ) from exc
    observed_frames = _owner_frame_contract(mujoco, model)
    _assert_owner_frame_contract_equal(expected_frames, observed_frames)
    return {
        "root_mjcf_path": str(selected_root),
        "root_mjcf_sha256": EXPECTED_CANONICAL_MJCF_SHA256,
        "identity_manifest_path": str(MUJOCO_IDENTITY_MANIFEST),
        "identity_manifest_sha256": EXPECTED_MUJOCO_IDENTITY_MANIFEST_SHA256,
        "identity_verifier_path": str(CANONICAL_MUJOCO_IDENTITY_PY),
        "identity_verifier_sha256": EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256,
        "portable_identity_sha256": portable_identity_sha256,
        "verification_receipt_sha256": verification_receipt_sha256,
        "owner_local_frame_sha256": expected_frames["content_sha256"],
    }


@dataclass(frozen=True)
class CollisionComponents:
    owner_indices: np.ndarray
    local_centers_m: np.ndarray
    local_half_axes_m: np.ndarray
    artifact_sha256: str
    content_sha256: str


@lru_cache(maxsize=4)
def _load_collision_components_cached(
    artifact_path: str, expected_file_sha256: str
) -> CollisionComponents:
    path = Path(artifact_path)
    try:
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise TableTerminationContractError(
            "robot/table collision proxy is not strict ASCII JSON"
        ) from exc
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_file_sha256:
        raise TableTerminationContractError(
            "robot/table collision proxy artifact SHA-256 drifted"
        )
    if not isinstance(document, dict):
        raise TableTerminationContractError("collision proxy root must be an object")
    if (
        document.get("schema_version") != 1
        or document.get("artifact_type") != "a3_table_collision_component_obb_v1"
        or tuple(document.get("body_order", ())) != TABLE_CONTACT_BODY_NAMES
    ):
        raise TableTerminationContractError(
            "collision proxy schema or exact 32-body order drifted"
        )
    content_sha = document.get("content_sha256")
    if not isinstance(content_sha, str) or len(content_sha) != 64:
        raise TableTerminationContractError("collision proxy content SHA is malformed")
    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != content_sha:
        raise TableTerminationContractError("collision proxy content SHA mismatch")
    components = document.get("components")
    if (
        not isinstance(components, list)
        or len(components) != 43
        or document.get("component_count") != 43
    ):
        raise TableTerminationContractError("collision proxy must contain 43 components")
    body_index = {name: index for index, name in enumerate(TABLE_CONTACT_BODY_NAMES)}
    component_ids = []
    owner_indices = []
    centers = []
    half_axes = []
    owner_coverage = set()
    for row in components:
        if not isinstance(row, dict):
            raise TableTerminationContractError("collision proxy component is malformed")
        component_id = row.get("component_id")
        owner = row.get("owner_body_name")
        try:
            center = np.asarray(row.get("local_center_owner_m"), dtype=np.float64)
            axes = np.asarray(row.get("local_half_axes_owner_m"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TableTerminationContractError(
                "collision proxy component geometry is non-numeric"
            ) from exc
        if (
            not isinstance(component_id, str)
            or not component_id
            or owner not in body_index
            or center.shape != (3,)
            or axes.shape != (3, 3)
            or not np.isfinite(center).all()
            or not np.isfinite(axes).all()
            or np.any(np.linalg.norm(axes, axis=1) <= 0.0)
        ):
            raise TableTerminationContractError(
                "collision proxy component metadata or geometry is malformed"
            )
        component_ids.append(component_id)
        owner_indices.append(body_index[str(owner)])
        centers.append(center)
        half_axes.append(axes)
        owner_coverage.add(str(owner))
    if (
        component_ids != sorted(component_ids)
        or len(set(component_ids)) != 43
        or owner_coverage != set(TABLE_CONTACT_BODY_NAMES)
    ):
        raise TableTerminationContractError(
            "collision proxy components are not canonical or body-complete"
        )
    arrays = (
        np.asarray(owner_indices, dtype=np.int64),
        np.asarray(centers, dtype=np.float64),
        np.asarray(half_axes, dtype=np.float64),
    )
    for value in arrays:
        value.setflags(write=False)
    return CollisionComponents(
        owner_indices=arrays[0],
        local_centers_m=arrays[1],
        local_half_axes_m=arrays[2],
        artifact_sha256=actual_sha,
        content_sha256=content_sha,
    )


def load_collision_components() -> CollisionComponents:
    return _load_collision_components_cached(
        str(COLLISION_PROXY_ARTIFACT), EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256
    )


def _validated_table_aabbs(geometry_contract: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(geometry_contract, Mapping):
        raise TableTerminationContractError("table geometry contract must be a mapping")
    payload = geometry_contract.get("payload")
    supplied_sha = geometry_contract.get("sha256")
    if not isinstance(payload, dict):
        raise TableTerminationContractError("table geometry payload is missing")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    actual_sha = hashlib.sha256(encoded).hexdigest()
    if (
        supplied_sha != actual_sha
        or actual_sha != EXPECTED_ACTION_BALL_TABLE_GEOMETRY_SHA256
    ):
        raise TableTerminationContractError("exact ActionBall table geometry SHA drifted")
    obstacles = payload.get("obstacles")
    if (
        not isinstance(obstacles, list)
        or len(obstacles) != 5
        or tuple(row.get("role") for row in obstacles) != TABLE_ASSEMBLY_ROLES
    ):
        raise TableTerminationContractError("table geometry is not the exact five-part assembly")
    centers = np.asarray(
        [row.get("center_mjcf_world_m") for row in obstacles], dtype=np.float64
    )
    extents = np.asarray(
        [row.get("full_extents_m") for row in obstacles], dtype=np.float64
    )
    if (
        centers.shape != (5, 3)
        or extents.shape != (5, 3)
        or not np.isfinite(centers).all()
        or not np.isfinite(extents).all()
        or np.any(extents <= 0.0)
    ):
        raise TableTerminationContractError("table geometry centers/extents are malformed")
    half = 0.5 * extents + TABLE_GUARD_MARGIN_M
    lo = centers - half
    hi = centers + half
    lo.setflags(write=False)
    hi.setflags(write=False)
    return lo, hi


def geometric_robot_table_hit(
    body_pos_w: Any,
    body_rotation_w: Any,
    components: CollisionComponents,
    aabb_lo: Any,
    aabb_hi: Any,
    *,
    racket_body_index: int,
) -> bool:
    """NumPy equivalent of Isaac's conservative broad-phase terminal kernel."""

    positions = np.asarray(body_pos_w, dtype=np.float64)
    rotations = np.asarray(body_rotation_w, dtype=np.float64)
    lo = np.asarray(aabb_lo, dtype=np.float64)
    hi = np.asarray(aabb_hi, dtype=np.float64)
    if (
        positions.shape != (32, 3)
        or rotations.shape != (32, 3, 3)
        or components.owner_indices.shape != (43,)
        or components.local_centers_m.shape != (43, 3)
        or components.local_half_axes_m.shape != (43, 3, 3)
        or lo.shape != (5, 3)
        or hi.shape != (5, 3)
        or type(racket_body_index) is not int
        or not 0 <= racket_body_index < 32
    ):
        raise TableTerminationContractError("robot/table guard runtime shapes drifted")
    if (
        not np.isfinite(positions).all()
        or not np.isfinite(rotations).all()
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(hi < lo)
    ):
        return True

    owner_pos = positions[components.owner_indices]
    owner_rotation = rotations[components.owner_indices]
    component_center = owner_pos + np.einsum(
        "cij,cj->ci", owner_rotation, components.local_centers_m
    )
    rotated_axes = np.einsum(
        "cij,ckj->cki", owner_rotation, components.local_half_axes_m
    )
    component_half = np.zeros_like(component_center)
    for local_axis in range(3):
        component_half += np.abs(rotated_axes[:, local_axis, :])
    component_half += COMPONENT_WORLD_AABB_GUARD_M
    component_lo = component_center - component_half
    component_hi = component_center + component_half
    component_overlap = np.ones((43, 5), dtype=bool)
    for axis in range(3):
        component_overlap &= (
            (component_hi[:, axis, None] >= lo[None, :, axis])
            & (component_lo[:, axis, None] <= hi[None, :, axis])
        )
    if bool(np.any(component_overlap)):
        return True

    racket_rotation = rotations[racket_body_index]
    blade_center = positions[racket_body_index] + (
        racket_rotation @ RACKET_BLADE_CENTER_OFFSET_WRIST_M
    )
    blade_axes = np.einsum(
        "ij,kj->ki", racket_rotation, RACKET_BLADE_LOCAL_HALF_AXES_M
    )
    blade_half = np.sum(np.abs(blade_axes), axis=0)
    blade_lo = blade_center - blade_half
    blade_hi = blade_center + blade_half
    return bool(
        np.any(np.all((blade_hi[None, :] >= lo) & (blade_lo[None, :] <= hi), axis=1))
    )


class ExactRobotTableGuard:
    """Run-static authority binding plus one exact per-substep pose sampler."""

    def __init__(
        self,
        mujoco: Any,
        model: Any,
        geometry_contract: Mapping[str, Any],
        *,
        mjcf_path: Path | str,
    ):
        self.source_receipt = verify_isaac_source_authority()
        self.identity_receipt = bind_pre_registered_owner_frames(
            mujoco, model, mjcf_path
        )
        self.components = load_collision_components()
        self.aabb_lo, self.aabb_hi = _validated_table_aabbs(geometry_contract)
        body_ids = _owner_frame_contract(mujoco, model)["body_ids"]
        self.body_ids = np.asarray(body_ids, dtype=np.int64)
        self.body_ids.setflags(write=False)
        self.racket_body_index = TABLE_CONTACT_BODY_NAMES.index(RACKET_BODY_NAME)

    def sample(self, data: Any) -> bool:
        positions = np.asarray(data.xpos[self.body_ids], dtype=np.float64)
        rotations = np.asarray(data.xmat[self.body_ids], dtype=np.float64).reshape(
            32, 3, 3
        )
        return geometric_robot_table_hit(
            positions,
            rotations,
            self.components,
            self.aabb_lo,
            self.aabb_hi,
            racket_body_index=self.racket_body_index,
        )


__all__ = [
    "CANONICAL_MJCF",
    "CANONICAL_MUJOCO_IDENTITY_PY",
    "COLLISION_PROXY_ARTIFACT",
    "CollisionComponents",
    "EXPECTED_ACTION_BALL_TABLE_GEOMETRY_SHA256",
    "EXPECTED_CANONICAL_MJCF_SHA256",
    "EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256",
    "EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256",
    "EXPECTED_ISAAC_ACTION_LATCH_SHA256",
    "EXPECTED_ISAAC_TERMINATION_CALLABLES_SHA256",
    "EXPECTED_ISAAC_TERMINATION_CONFIG_SHA256",
    "EXPECTED_MUJOCO_IDENTITY_MANIFEST_SHA256",
    "EXPECTED_PORTABLE_MUJOCO_IDENTITY_SHA256",
    "ExactRobotTableGuard",
    "ISAAC_ACTION_LATCH",
    "ISAAC_TERMINATION_CALLABLES",
    "ISAAC_TERMINATION_CONFIG",
    "MUJOCO_IDENTITY_MANIFEST",
    "TABLE_CONTACT_BODY_NAMES",
    "TABLE_GUARD_MARGIN_M",
    "TableTerminationContractError",
    "bind_pre_registered_owner_frames",
    "geometric_robot_table_hit",
    "load_collision_components",
    "verify_isaac_source_authority",
]
