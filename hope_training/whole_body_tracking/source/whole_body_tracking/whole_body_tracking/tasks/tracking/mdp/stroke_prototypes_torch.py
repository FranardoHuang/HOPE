"""Torch mirror of the stroke prototype table — SAME FILE, same sha256, aligned tensors.

人话:训练器和规划器读同一份 ``configs/stroke_prototypes_v1_*.json``、校验同一个 sha256。这边把
它摊平成一组对齐的张量,好让选择器在一个 batch 里同时判所有 env。字段含义、帧、单位见
``hope_ws/src/hope_planner/hope_planner/stroke_prototypes.py``——那份文档是这份数据的说明书。

Why the reader is duplicated instead of imported: the trainer runs on pods where the ROS package
``hope_planner`` is not installed. The BYTES are shared (one file, one sha256, one
``derived_sha256`` over the records), and ``tests/test_stroke_prototype_parity.py`` asserts the two
readers produce field-identical records — which is the checkable form of "one source of truth".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

try:
    from .racket_contact_geometry import GEOMETRY_SOURCE_SHA256
except ImportError:  # dependency-light by-path test/CLI import
    from racket_contact_geometry import GEOMETRY_SOURCE_SHA256


SCHEMA_VERSION = 2
LEGACY_SITE_VELOCITY_SCHEMA_VERSION = 1
FACE_CENTER_VELOCITY_POINT = "selected_rubber_face_center"

#: Fields the loader refuses to default. A missing one is a build error.
_REQUIRED_COMMON = (
    "motion_id", "scope", "family", "clip_index", "npz_sha256",
    "t_prepare_s", "t_prepare_min_s", "t_prepare_max_s",
    "band_b_x", "band_b_y", "band_z_w", "slack_b_xy_m", "slack_z_w_m",
    "p_contact_b", "n_hat_b", "face_sign", "priority", "enabled", "strike_phase",
    "contact_frame", "contact_window_frames",
)
_REQUIRED_V2 = (
    *_REQUIRED_COMMON,
    "racket_face_center_velocity_hat_b",
    "racket_face_center_elevation_deg",
    "racket_face_center_window_dir_cone_deg",
    "racket_face_center_speed_nominal_mps",
    "racket_face_center_speed_max_mps",
    "racket_face_center_speed_min_mps",
    "racket_face_center_v_star_cap_mps",
    "racket_face_center_v_dir_tol_deg",
    "racket_face_center_cos_normal_velocity",
)
_REQUIRED_V1 = (
    *_REQUIRED_COMMON,
    "v_hat_b",
    "elevation_deg",
    "speed_nominal_mps",
    "speed_max_mps",
    "speed_min_mps",
    "v_star_cap_mps",
    "v_dir_tol_deg",
)
_LEGACY_AMBIGUOUS_VELOCITY_FIELDS = frozenset(
    (
        "v_hat_b",
        "elevation_deg",
        "window_dir_cone_deg",
        "speed_nominal_mps",
        "speed_max_mps",
        "speed_min_mps",
        "v_star_cap_mps",
        "v_dir_tol_deg",
        "cos_nv",
    )
)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class StrokePrototypeTensors:
    """One scope's prototypes as (N, ...) tensors, row order == clip_id."""

    motion_ids: tuple
    families: tuple                 # "forehand" | "backhand"
    family_sign: torch.Tensor       # (N,) +1 forehand, -1 backhand
    v_hat_b: torch.Tensor           # (N, 3) unit, B_yaw
    elevation_deg: torch.Tensor     # (N,)
    speed_nominal: torch.Tensor     # (N,)
    speed_min: torch.Tensor         # (N,)
    speed_max: torch.Tensor         # (N,)
    v_star_cap: torch.Tensor        # (N,)
    v_dir_tol_deg: torch.Tensor     # (N,)
    t_prepare: torch.Tensor         # (N,)
    t_prepare_min: torch.Tensor     # (N,)
    t_prepare_max: torch.Tensor     # (N,)
    band_b_x: torch.Tensor          # (N, 2)
    band_b_y: torch.Tensor          # (N, 2)
    band_z_w: torch.Tensor          # (N, 2) W_floor
    slack_b_xy: torch.Tensor        # (N,)
    slack_z_w: torch.Tensor         # (N,)
    p_contact_b: torch.Tensor       # (N, 3)
    n_hat_b: torch.Tensor           # (N, 3)
    face_sign: torch.Tensor         # (N,)
    priority: torch.Tensor          # (N,) long
    enabled: torch.Tensor           # (N,) bool
    strike_phase: torch.Tensor      # (N,)
    contact_frame: torch.Tensor     # (N,) long
    contact_window: torch.Tensor    # (N, 2) long
    file_sha256: str = ""
    derived_sha256: str = ""
    scope: str = ""
    path: str = ""
    velocity_point_semantics: str = ""

    def __len__(self) -> int:
        return int(self.v_hat_b.shape[0])


def load_stroke_prototype_tensors(
    path,
    scope: str = "upper",
    device="cpu",
    expected_sha256=None,
    expected_motion_ids=None,
    expected_motion_sha256=None,
    allow_legacy_site_velocity: bool = False,
) -> StrokePrototypeTensors:
    """Read the shared prototype JSON for one scope into aligned tensors, fail-closed.

    ``expected_motion_ids`` and ``expected_motion_sha256`` pin every row
    against the loaded motion clips.  Schema v1 encoded official racket-site
    velocity in ambiguous generic fields; it is rejected unless an explicitly
    legacy caller opts in.  ActionBall must use schema v2 selected-rubber
    face-centre velocity.
    """
    path = str(path)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != str(expected_sha256):
        raise ValueError(
            f"{path}: stroke prototype file sha256 {actual} does not match the pinned "
            f"{expected_sha256} — trainer and planner would be reading different strokes"
        )
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    schema_version = int(doc.get("schema_version", -1))
    if schema_version == LEGACY_SITE_VELOCITY_SCHEMA_VERSION:
        if type(allow_legacy_site_velocity) is not bool:
            raise ValueError("allow_legacy_site_velocity must be bool")
        if not allow_legacy_site_velocity:
            raise ValueError(
                f"{path}: stroke prototype schema v1 contains legacy "
                "official-racket-site velocity; ActionBall exact-face "
                "contact requires schema v2 selected-rubber face-centre "
                "velocity"
            )
        required = _REQUIRED_V1
        direction_key = "v_hat_b"
        elevation_key = "elevation_deg"
        speed_nominal_key = "speed_nominal_mps"
        speed_min_key = "speed_min_mps"
        speed_max_key = "speed_max_mps"
        v_star_key = "v_star_cap_mps"
        direction_tolerance_key = "v_dir_tol_deg"
        velocity_point_semantics = "official_racket_site_legacy"
    elif schema_version == SCHEMA_VERSION:
        contract = doc.get("velocity_contract")
        expected_contract = {
            "direction_and_speed_point": FACE_CENTER_VELOCITY_POINT,
            "policy_control_point": "official_racket_site",
            "mapping": (
                "v_face_center=v_site+omega_world_cross_"
                "r_face_center_from_site_world"
            ),
            "site_velocity_authority": (
                "centered_position_fd_half_window_2_clamped_per_clip"
            ),
            "angular_velocity_authority": (
                "npz_body_ang_vel_w_at_right_wrist_yaw_Link"
            ),
            "direction_frame_authority": (
                "canonical_ready_root_yaw_at_frame_0"
            ),
            "geometry_source_sha256": GEOMETRY_SOURCE_SHA256,
        }
        if contract != expected_contract:
            raise ValueError(
                f"{path}: schema v2 velocity_contract must exactly bind "
                "selected-rubber face-centre velocity and the current "
                "exact-face geometry source"
            )
        required = _REQUIRED_V2
        direction_key = "racket_face_center_velocity_hat_b"
        elevation_key = "racket_face_center_elevation_deg"
        speed_nominal_key = "racket_face_center_speed_nominal_mps"
        speed_min_key = "racket_face_center_speed_min_mps"
        speed_max_key = "racket_face_center_speed_max_mps"
        v_star_key = "racket_face_center_v_star_cap_mps"
        direction_tolerance_key = (
            "racket_face_center_v_dir_tol_deg"
        )
        velocity_point_semantics = FACE_CENTER_VELOCITY_POINT
    else:
        raise ValueError(
            f"{path}: stroke prototype schema_version {doc.get('schema_version')!r} != "
            f"{SCHEMA_VERSION}"
        )
    scopes = doc.get("scopes")
    if not isinstance(scopes, dict) or scope not in scopes:
        raise ValueError(
            f"{path}: no prototype scope {scope!r} (have "
            f"{sorted(scopes) if isinstance(scopes, dict) else scopes!r})"
        )
    derived = canonical_sha256(scopes)
    if str(doc.get("derived_sha256", "")) != derived:
        raise ValueError(
            f"{path}: derived_sha256 {doc.get('derived_sha256')!r} does not match the sha256 of "
            f"the records it covers ({derived}) — a derived field was hand-edited; rebuild with "
            f"scripts/build_stroke_prototypes.py"
        )
    rows = scopes[scope]
    if not rows:
        raise ValueError(f"{path}: prototype scope {scope!r} is empty")
    for i, r in enumerate(rows):
        missing = [k for k in required if k not in r]
        if missing:
            raise ValueError(
                f"{path}: scope {scope!r} record {i} ({r.get('motion_id', '?')!r}) is missing "
                f"required field(s) {missing} — rebuild with scripts/build_stroke_prototypes.py"
            )
        if int(r["clip_index"]) != i:
            raise ValueError(
                f"{path}: scope {scope!r} record {i} ({r['motion_id']!r}) declares "
                f"clip_index={r['clip_index']} but sits at position {i}; clip_index IS the clip id"
            )
        if (
            schema_version == SCHEMA_VERSION
            and _LEGACY_AMBIGUOUS_VELOCITY_FIELDS.intersection(r)
        ):
            raise ValueError(
                f"{path}: schema v2 record {i} mixes legacy ambiguous "
                "site-velocity field names into the face-centre contract"
            )
        direction = tuple(float(value) for value in r[direction_key])
        direction_norm = sum(value * value for value in direction) ** 0.5
        if abs(direction_norm - 1.0) > 1.0e-6:
            raise ValueError(
                f"{path}: {r['motion_id']}/{scope} face-centre direction "
                f"must be unit length; got norm {direction_norm:.12g}"
            )
        if float(r[speed_min_key]) > float(r[speed_max_key]):
            raise ValueError(
                f"{path}: {r['motion_id']}/{scope} speed window "
                f"[{r[speed_min_key]}, {r[speed_max_key]}] is empty"
            )
    ids = tuple(str(r["motion_id"]) for r in rows)
    if expected_motion_ids is not None:
        want = tuple(str(m) for m in expected_motion_ids)
        if ids != want:
            raise ValueError(
                f"{path}: prototype scope {scope!r} lists clips {list(ids)} but the loaded motion "
                f"clip order is {list(want)} — align the prototype file with the motion_file order"
            )
    if expected_motion_sha256 is not None:
        expected_motion_sha256 = tuple(
            str(value) for value in expected_motion_sha256
        )
        observed_motion_sha256 = tuple(
            str(row["npz_sha256"]) for row in rows
        )
        if observed_motion_sha256 != expected_motion_sha256:
            raise ValueError(
                f"{path}: prototype scope {scope!r} NPZ SHA order "
                "differs from the loaded motion bytes"
            )

    def _t(key, dtype=torch.float32):
        return torch.tensor([r[key] for r in rows], dtype=dtype, device=device)

    return StrokePrototypeTensors(
        motion_ids=ids,
        families=tuple(str(r["family"]) for r in rows),
        family_sign=torch.tensor(
            [1.0 if str(r["family"]) == "forehand" else -1.0 for r in rows],
            dtype=torch.float32, device=device),
        v_hat_b=_t(direction_key),
        elevation_deg=_t(elevation_key),
        speed_nominal=_t(speed_nominal_key),
        speed_min=_t(speed_min_key),
        speed_max=_t(speed_max_key),
        v_star_cap=_t(v_star_key),
        v_dir_tol_deg=_t(direction_tolerance_key),
        t_prepare=_t("t_prepare_s"),
        t_prepare_min=_t("t_prepare_min_s"),
        t_prepare_max=_t("t_prepare_max_s"),
        band_b_x=_t("band_b_x"),
        band_b_y=_t("band_b_y"),
        band_z_w=_t("band_z_w"),
        slack_b_xy=_t("slack_b_xy_m"),
        slack_z_w=_t("slack_z_w_m"),
        p_contact_b=_t("p_contact_b"),
        n_hat_b=_t("n_hat_b"),
        face_sign=_t("face_sign"),
        priority=_t("priority", torch.long),
        enabled=torch.tensor([bool(r["enabled"]) for r in rows], dtype=torch.bool, device=device),
        strike_phase=_t("strike_phase"),
        contact_frame=_t("contact_frame", torch.long),
        contact_window=_t("contact_window_frames", torch.long),
        file_sha256=actual,
        derived_sha256=derived,
        scope=scope,
        path=path,
        velocity_point_semantics=velocity_point_semantics,
    )
