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

SCHEMA_VERSION = 1

#: Fields the loader refuses to default. A missing one is a build error.
_REQUIRED = (
    "motion_id", "scope", "family", "clip_index", "v_hat_b", "elevation_deg",
    "speed_nominal_mps", "speed_max_mps", "speed_min_mps", "v_star_cap_mps", "v_dir_tol_deg",
    "t_prepare_s", "t_prepare_min_s", "t_prepare_max_s",
    "band_b_x", "band_b_y", "band_z_w", "slack_b_xy_m", "slack_z_w_m",
    "p_contact_b", "n_hat_b", "face_sign", "priority", "enabled", "strike_phase",
    "contact_frame", "contact_window_frames",
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

    def __len__(self) -> int:
        return int(self.v_hat_b.shape[0])


def load_stroke_prototype_tensors(
    path,
    scope: str = "upper",
    device="cpu",
    expected_sha256=None,
    expected_motion_ids=None,
) -> StrokePrototypeTensors:
    """Read the shared prototype JSON for one scope into aligned tensors, fail-closed.

    ``expected_motion_ids`` pins the row order against the loaded motion clips: ``clip_index`` is a
    POSITIONAL index into the trainer's clip list, so a prototype file that no longer lines up with
    the clips is a construction error, not a silently wrong stroke.
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
    rows = scopes[scope]
    if not rows:
        raise ValueError(f"{path}: prototype scope {scope!r} is empty")
    for i, r in enumerate(rows):
        missing = [k for k in _REQUIRED if k not in r]
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
        if float(r["v_hat_b"][0]) <= 0.0:
            raise ValueError(
                f"{path}: {r['motion_id']}/{scope} v_hat_b points at x={float(r['v_hat_b'][0]):+.4f}"
                f" <= 0, away from the opponent — a stroke whose blade travels backwards at contact"
                f" can never return a ball"
            )
        if float(r["speed_min_mps"]) > float(r["speed_max_mps"]):
            raise ValueError(
                f"{path}: {r['motion_id']}/{scope} speed window "
                f"[{r['speed_min_mps']}, {r['speed_max_mps']}] is empty"
            )
    ids = tuple(str(r["motion_id"]) for r in rows)
    if expected_motion_ids is not None:
        want = tuple(str(m) for m in expected_motion_ids)
        if ids != want:
            raise ValueError(
                f"{path}: prototype scope {scope!r} lists clips {list(ids)} but the loaded motion "
                f"clip order is {list(want)} — align the prototype file with the motion_file order"
            )

    def _t(key, dtype=torch.float32):
        return torch.tensor([r[key] for r in rows], dtype=dtype, device=device)

    return StrokePrototypeTensors(
        motion_ids=ids,
        families=tuple(str(r["family"]) for r in rows),
        family_sign=torch.tensor(
            [1.0 if str(r["family"]) == "forehand" else -1.0 for r in rows],
            dtype=torch.float32, device=device),
        v_hat_b=_t("v_hat_b"),
        elevation_deg=_t("elevation_deg"),
        speed_nominal=_t("speed_nominal_mps"),
        speed_min=_t("speed_min_mps"),
        speed_max=_t("speed_max_mps"),
        v_star_cap=_t("v_star_cap_mps"),
        v_dir_tol_deg=_t("v_dir_tol_deg"),
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
    )
