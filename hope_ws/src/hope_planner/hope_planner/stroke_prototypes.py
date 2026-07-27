"""Stroke prototypes — the ONE source of truth the selector and the adapter both read.

人话:一个动作(正手拉/反手拉/正手挡/反手挡/高压)在触球那一刻"往哪挥、挥多快、最快能多快、
准备要多久、能在哪个高度和左右范围触球、用哪一面",全部量自动作片段本身,写死在一个 JSON 里。
规划器(这个 numpy 模块)和训练器(``mdp/stroke_prototypes_torch.py``)读的是同一份文件、校验
同一个 sha256,所以两边永远不可能各拿一套动作参数。

The file is produced by ``hope_training/whole_body_tracking/scripts/build_stroke_prototypes.py``
from the compiled clips + their BUILD_MANIFEST; re-running the builder must reproduce it byte for
byte, so a hand-edited derived field fails the builder's ``--check``.

FRAMES (spec §0 — three frames exist in this system and are silently different; do not invent a
fourth):

* ``B_yaw``   base(pelvis) origin, yaw-only rotation.  ``v_hat_b``, ``p_contact_b``, ``band_b_x``,
  ``band_b_y`` and ``n_hat_b`` live here.
* ``W_floor`` env-local world, **z = 0 at the FLOOR**.  ``band_z_w`` lives here — it is a HEIGHT
  ABOVE THE FLOOR, which is the frame every consumer of a contact height already uses
  (``vb_table_surface_z`` = 0.76 etc.).
* ``W_table`` the planner's own world frame, z = 0 at the TABLE SURFACE.  A planner-side caller
  works in this frame and must declare the offset (``SelectorCfg.z_floor_offset_m``, = table
  height); nothing in this module guesses it.

Units are SI throughout; angles are stored in DEGREES in the JSON and kept in degrees on the
record (matching ``tilt_pitch_deg`` in ``strike_spec_planner``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1

#: Fields the loader refuses to fall back on.  A missing one is a build error, never a default.
_REQUIRED = (
    "motion_id", "scope", "family", "clip_index", "frames", "fps", "npz_sha256",
    "cycle_s", "contact_window_frames", "contact_frame", "strike_phase",
    "t_prepare_s", "t_recover_s", "deep_frame", "L_deep_m", "t_backswing_s",
    "t_prepare_min_s", "t_prepare_max_s",
    "v_hat_b", "elevation_deg", "window_dir_cone_deg", "speed_nominal_mps",
    "speed_max_mps", "speed_min_mps", "retime_range", "retime_binding", "v_star_cap_mps",
    "v_dir_tol_deg",
    "p_contact_b", "band_b_x", "band_b_y", "band_z_w", "base_height_at_contact_m",
    "slack_b_xy_m", "slack_z_w_m",
    "face_sign", "n_hat_b", "cos_nv",
    "priority", "enabled",
)


@dataclass(frozen=True)
class StrokePrototype:
    """One stroke, measured at its own contact frame.  See the module docstring for frames."""

    # --- A. identity / provenance -------------------------------------------------------
    motion_id: str
    scope: str                      # "upper" | "full"
    family: str                     # "forehand" | "backhand"
    clip_index: int                 # position in the loaded clip order == trainer clip_id
    frames: int
    fps: float
    npz_sha256: str

    # --- B. timing (s) ------------------------------------------------------------------
    cycle_s: float
    contact_window_frames: Tuple[int, int]
    contact_frame: int
    strike_phase: float
    t_prepare_s: float
    t_recover_s: float
    deep_frame: int
    L_deep_m: float
    t_backswing_s: float
    t_prepare_min_s: float          # t_prepare_s / retime_s_max — the FASTEST this stroke preps
    t_prepare_max_s: float          # t_prepare_s / retime_s_min — the SLOWEST

    # --- C. velocity identity (the adapter's invariant) ---------------------------------
    v_hat_b: np.ndarray             # (3,) unit, B_yaw
    elevation_deg: float            # asin(v_hat_b[2]) — the loop/block discriminator
    window_dir_cone_deg: float
    speed_nominal_mps: float
    speed_max_mps: float            # min(nominal*s_max, v_star_cap) — TWO measured ceilings only;
                                    # the deploy gate is reported (provenance.deploy_gate), not
                                    # folded in (owner's ruling 2026-07-27)
    speed_min_mps: float
    retime_range: Tuple[float, float]
    retime_binding: Tuple[str, str]
    v_star_cap_mps: float
    v_dir_tol_deg: float            # HUMAN

    # --- D. contact region (selector reachability only) ---------------------------------
    p_contact_b: np.ndarray         # (3,) B_yaw
    band_b_x: Tuple[float, float]   # MEASURED sweep over the contact window, B_yaw
    band_b_y: Tuple[float, float]
    band_z_w: Tuple[float, float]   # MEASURED, W_floor (height above the FLOOR)
    base_height_at_contact_m: float
    slack_b_xy_m: float             # HUMAN
    slack_z_w_m: float              # HUMAN

    # --- E. face -------------------------------------------------------------------------
    face_sign: float                # +1 red/+Y, -1 black/-Y
    n_hat_b: np.ndarray             # (3,) physical striking face normal at contact, B_yaw
    cos_nv: float

    # --- G. selection policy (HUMAN) -----------------------------------------------------
    priority: int
    enabled: bool

    # --- optional provenance / diagnostics ----------------------------------------------
    npz_filename: str = ""
    source_sha256: str = ""
    family_measured_side: str = ""
    y_b_at_contact_m: float = float("nan")
    source_window_frames: Tuple[int, int] = (-1, -1)

    def __post_init__(self) -> None:
        for name in ("v_hat_b", "p_contact_b", "n_hat_b"):
            v = np.asarray(getattr(self, name), dtype=float).reshape(3)
            object.__setattr__(self, name, v)
        if abs(float(np.linalg.norm(self.v_hat_b)) - 1.0) > 1e-6:
            raise ValueError(
                f"stroke prototype {self.motion_id}/{self.scope}: v_hat_b is not a unit vector "
                f"(|v_hat_b| = {float(np.linalg.norm(self.v_hat_b)):.6f})"
            )
        if float(self.v_hat_b[0]) <= 0.0:
            raise ValueError(
                f"stroke prototype {self.motion_id}/{self.scope}: v_hat_b points at "
                f"x={float(self.v_hat_b[0]):+.4f} <= 0, i.e. away from the opponent (+x). A stroke "
                f"whose blade travels backwards at contact can never return a ball"
            )
        if float(self.face_sign) not in (1.0, -1.0):
            raise ValueError(
                f"stroke prototype {self.motion_id}/{self.scope}: face_sign must be +1 or -1, "
                f"got {self.face_sign!r}"
            )
        if not (0.0 < self.speed_min_mps <= self.speed_max_mps):
            raise ValueError(
                f"stroke prototype {self.motion_id}/{self.scope}: speed window "
                f"[{self.speed_min_mps:.4f}, {self.speed_max_mps:.4f}] m/s is empty or non-positive"
            )
        if not (0.0 < self.t_prepare_min_s <= self.t_prepare_max_s):
            raise ValueError(
                f"stroke prototype {self.motion_id}/{self.scope}: prepare window "
                f"[{self.t_prepare_min_s:.4f}, {self.t_prepare_max_s:.4f}] s is empty"
            )
        if self.band_b_x[0] > self.band_b_x[1] or self.band_b_y[0] > self.band_b_y[1] \
                or self.band_z_w[0] > self.band_z_w[1]:
            raise ValueError(
                f"stroke prototype {self.motion_id}/{self.scope}: an inverted contact band "
                f"(x={self.band_b_x}, y={self.band_b_y}, z={self.band_z_w})"
            )
        if self.family not in ("forehand", "backhand"):
            raise ValueError(
                f"stroke prototype {self.motion_id}/{self.scope}: family must be 'forehand' or "
                f"'backhand', got {self.family!r}"
            )

    # --- convenience ---------------------------------------------------------------------
    @property
    def family_sign(self) -> float:
        """Deploy wire convention: +1 forehand, -1 backhand (node_runtime_contract:918-921)."""
        return 1.0 if self.family == "forehand" else -1.0

    @property
    def key(self) -> str:
        return f"{self.motion_id}/{self.scope}"


@dataclass(frozen=True)
class StrokePrototypeSet:
    """The ordered prototypes of ONE scope, plus the bytes they came from."""

    scope: str
    prototypes: Tuple[StrokePrototype, ...]
    file_sha256: str
    derived_sha256: str
    contact_rule: Dict[str, object]
    path: str = ""

    def __len__(self) -> int:
        return len(self.prototypes)

    def __iter__(self):
        return iter(self.prototypes)

    def __getitem__(self, i: int) -> StrokePrototype:
        return self.prototypes[i]

    @property
    def motion_ids(self) -> Tuple[str, ...]:
        return tuple(p.motion_id for p in self.prototypes)

    @property
    def clip_families(self) -> Tuple[str, ...]:
        return tuple(p.family for p in self.prototypes)

    def by_motion_id(self, motion_id: str) -> StrokePrototype:
        for p in self.prototypes:
            if p.motion_id == motion_id:
                return p
        raise KeyError(
            f"no stroke prototype {motion_id!r} in scope {self.scope!r}; have "
            f"{list(self.motion_ids)}"
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


def _record(raw: dict, path: str, scope: str, index: int) -> StrokePrototype:
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(
            f"{path}: scope {scope!r} record {index} ({raw.get('motion_id', '?')!r}) is missing "
            f"required field(s) {missing} — rebuild it with "
            f"scripts/build_stroke_prototypes.py; the loader never defaults a derived field"
        )
    if int(raw["clip_index"]) != index:
        raise ValueError(
            f"{path}: scope {scope!r} record {index} ({raw['motion_id']!r}) declares "
            f"clip_index={raw['clip_index']} but sits at position {index}. clip_index IS the "
            f"trainer's clip id — the file order and the field must agree"
        )
    return StrokePrototype(
        motion_id=str(raw["motion_id"]),
        scope=str(raw["scope"]),
        family=str(raw["family"]),
        clip_index=int(raw["clip_index"]),
        frames=int(raw["frames"]),
        fps=float(raw["fps"]),
        npz_sha256=str(raw["npz_sha256"]),
        cycle_s=float(raw["cycle_s"]),
        contact_window_frames=(int(raw["contact_window_frames"][0]),
                               int(raw["contact_window_frames"][1])),
        contact_frame=int(raw["contact_frame"]),
        strike_phase=float(raw["strike_phase"]),
        t_prepare_s=float(raw["t_prepare_s"]),
        t_recover_s=float(raw["t_recover_s"]),
        deep_frame=int(raw["deep_frame"]),
        L_deep_m=float(raw["L_deep_m"]),
        t_backswing_s=float(raw["t_backswing_s"]),
        t_prepare_min_s=float(raw["t_prepare_min_s"]),
        t_prepare_max_s=float(raw["t_prepare_max_s"]),
        v_hat_b=np.asarray(raw["v_hat_b"], dtype=float),
        elevation_deg=float(raw["elevation_deg"]),
        window_dir_cone_deg=float(raw["window_dir_cone_deg"]),
        speed_nominal_mps=float(raw["speed_nominal_mps"]),
        speed_max_mps=float(raw["speed_max_mps"]),
        speed_min_mps=float(raw["speed_min_mps"]),
        retime_range=(float(raw["retime_range"][0]), float(raw["retime_range"][1])),
        retime_binding=(str(raw["retime_binding"][0]), str(raw["retime_binding"][1])),
        v_star_cap_mps=float(raw["v_star_cap_mps"]),
        v_dir_tol_deg=float(raw["v_dir_tol_deg"]),
        p_contact_b=np.asarray(raw["p_contact_b"], dtype=float),
        band_b_x=(float(raw["band_b_x"][0]), float(raw["band_b_x"][1])),
        band_b_y=(float(raw["band_b_y"][0]), float(raw["band_b_y"][1])),
        band_z_w=(float(raw["band_z_w"][0]), float(raw["band_z_w"][1])),
        base_height_at_contact_m=float(raw["base_height_at_contact_m"]),
        slack_b_xy_m=float(raw["slack_b_xy_m"]),
        slack_z_w_m=float(raw["slack_z_w_m"]),
        face_sign=float(raw["face_sign"]),
        n_hat_b=np.asarray(raw["n_hat_b"], dtype=float),
        cos_nv=float(raw["cos_nv"]),
        priority=int(raw["priority"]),
        enabled=bool(raw["enabled"]),
        npz_filename=str(raw.get("npz_filename", "")),
        source_sha256=str(raw.get("source_sha256", "")),
        family_measured_side=str(raw.get("family_measured_side", "")),
        y_b_at_contact_m=float(raw.get("y_b_at_contact_m", float("nan"))),
        source_window_frames=(int(raw.get("source_window_frames", (-1, -1))[0]),
                              int(raw.get("source_window_frames", (-1, -1))[1])),
    )


def load_stroke_prototypes(
    path,
    scope: str = "upper",
    expected_sha256: Optional[str] = None,
    expected_motion_ids: Optional[Sequence[str]] = None,
) -> StrokePrototypeSet:
    """Read the prototype JSON for one scope, fail-closed.

    ``expected_sha256`` pins the file's bytes (the planner and the trainer both pass it, which is
    what makes "same bytes on both sides" checkable rather than asserted).  ``expected_motion_ids``
    pins the clip ORDER against whatever loaded the motion clips, so a prototype list that no
    longer lines up with the loaded clips is a construction error and not a silently wrong
    ``clip_index``.
    """
    path = str(path)
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != str(expected_sha256):
        raise ValueError(
            f"{path}: stroke prototype file sha256 {actual_sha} does not match the pinned "
            f"{expected_sha256} — planner and trainer would be reading different strokes"
        )
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    if int(doc.get("schema_version", -1)) != SCHEMA_VERSION:
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

    protos = tuple(_record(r, path, scope, i) for i, r in enumerate(scopes[scope]))
    if not protos:
        raise ValueError(f"{path}: prototype scope {scope!r} is empty")
    ids = [p.motion_id for p in protos]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate motion_id in scope {scope!r}: {ids}")
    if expected_motion_ids is not None:
        want = [str(m) for m in expected_motion_ids]
        if ids != want:
            raise ValueError(
                f"{path}: prototype scope {scope!r} lists clips {ids} but the loaded motion "
                f"clip order is {want}. clip_index is a positional index — align the prototype "
                f"file with the motion_file order, do not reorder one side silently"
            )
    return StrokePrototypeSet(
        scope=scope,
        prototypes=protos,
        file_sha256=actual_sha,
        derived_sha256=derived,
        contact_rule=dict(doc.get("contact_rule") or {}),
        path=path,
    )


def assert_strokes_are_distinguishable(protos: StrokePrototypeSet) -> None:
    """The direction cone may never be wide enough to turn one stroke into another.

    Spec §4.3: for two prototypes of the SAME family at different priorities, the adapter's
    stage-2 relaxation cone (``v_dir_tol_deg`` each way) must not span the gap between their
    velocity elevations, or "preserve the stroke's identity" would be a slogan.
    """
    enabled = [p for p in protos if p.enabled]
    for i, a in enumerate(enabled):
        for b in enabled[i + 1:]:
            if a.family != b.family or a.priority == b.priority:
                continue
            gap = abs(a.elevation_deg - b.elevation_deg)
            need = 2.0 * max(a.v_dir_tol_deg, b.v_dir_tol_deg)
            if gap <= need:
                raise ValueError(
                    f"stroke prototypes {a.key} (elevation {a.elevation_deg:+.1f} deg) and "
                    f"{b.key} ({b.elevation_deg:+.1f} deg) are the same family at different "
                    f"priorities but only {gap:.1f} deg apart, while the adapter may rotate the "
                    f"velocity direction by up to {need:.1f} deg total — the cone could turn one "
                    f"stroke into the other. Lower v_dir_tol_deg or drop one prototype"
                )


def direction_world(proto: StrokePrototype, base_yaw_rad: float) -> np.ndarray:
    """The prototype's contact-frame blade direction, rotated from B_yaw into the world.

    ``d_hat_w = R_z(yaw_base) . v_hat_b`` — the stroke's identity, the one thing the adapter is
    forbidden to change.
    """
    c, s = math.cos(float(base_yaw_rad)), math.sin(float(base_yaw_rad))
    v = proto.v_hat_b
    d = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=float)
    n = float(np.linalg.norm(d))
    if n <= 1e-12:                       # v_hat_b is validated unit at construction; total anyway
        raise ValueError(f"stroke prototype {proto.key}: v_hat_b has no direction")
    return d / n
