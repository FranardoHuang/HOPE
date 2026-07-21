"""Stage-1 question-bank loading + selection (torch/numpy ONLY — no Isaac imports).

The bank is produced offline by ``scripts/gen_stage1_questions.py``: per clip, a FIXED contact
point plus an atomic per-question tuple ``(incoming velocity, incoming spin, demanded racket
velocity, demanded face normal)``. Keys are ``"<clip>/contact_pos_env"`` (3,),
``"<clip>/incoming_vel"`` / ``"<clip>/incoming_spin"`` / ``"<clip>/demanded_vel"`` /
``"<clip>/demanded_normal"`` (Q, 3) and optionally
``"<clip>/difficulty_deg"`` (Q,), with clip names ordered by clip_id (forehand=0, backhand=1).
Positions are tracking-env-frame (env origin at the robot spawn, table surface z=0.76); world
targets are ``env_origin + pos``. This module is deliberately standalone (like virtual_ball.py)
so the loading/selection logic is unit-testable without the training env.
"""

from __future__ import annotations

import json
import hashlib
import os
from typing import NamedTuple, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 3
SPLIT_ALGORITHM = "blake2b64-round1e-4-f64le-v1"
EXAM_FRAC = 0.2
SOURCE_FAMILY_CONTRACT = "stage1-source-family-v2"
PHYSICS_CONTRACT_FILES = (
    "configs/ball_physics_venue.yaml",
    "hope_ws/src/hope_planner/hope_planner/constants.py",
    "hope_ws/src/hope_planner/hope_planner/ball_contact.py",
    "hope_ws/src/hope_planner/hope_planner/ball_trajectory_predictor.py",
    "hope_ws/src/hope_planner/hope_planner/strike_spec_planner.py",
    "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
    "tasks/tracking/mdp/virtual_ball.py",
    "hope_training/whole_body_tracking/scripts/venue_ball_sampler.py",
)


def sha256_file(path: str) -> str:
    """Content hash used by bank provenance (streaming; no asset-size assumption)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value) -> str:
    """SHA-256 of deterministic JSON used for the source-family contract."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_repo_root(start: str | None = None) -> str:
    """Find the checkout root without importing the Isaac package hierarchy."""
    current = os.path.abspath(start or os.path.dirname(__file__))
    if os.path.isfile(current):
        current = os.path.dirname(current)
    while True:
        if os.path.isfile(os.path.join(current, "configs", "ball_physics_venue.yaml")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise ValueError("cannot locate repository root for Stage-1 physics contract")
        current = parent


def runtime_physics_contract(repo_root: str | None = None) -> dict:
    root = find_repo_root(repo_root)
    files = {}
    for relative in PHYSICS_CONTRACT_FILES:
        path = os.path.join(root, relative)
        if not os.path.isfile(path):
            raise ValueError(f"Stage-1 physics contract file is missing: {path}")
        files[relative] = sha256_file(path)
    return {"contract": "stage1-physics-runtime-v1", "files": files}


def validate_runtime_physics_contract(metadata: dict, repo_root: str | None = None) -> None:
    recorded = metadata.get("physics_contract")
    recorded_sha = str(metadata.get("physics_contract_sha256", ""))
    if not isinstance(recorded, dict) or canonical_sha256(recorded) != recorded_sha:
        raise ValueError("question bank physics contract/hash is missing or internally inconsistent")
    current = runtime_physics_contract(repo_root)
    if current != recorded:
        changed = sorted(
            key for key in set((recorded.get("files") or {})) | set(current["files"])
            if (recorded.get("files") or {}).get(key) != current["files"].get(key)
        )
        raise ValueError(
            "question bank was generated/scored with a different physics implementation; "
            f"changed files={changed}"
        )


def question_id(v_in_row) -> str:
    """Stable content id for a Stage-1 incoming-ball row.

    Explicit little-endian float64 closes the architecture-dependent byte-order hole in the
    former split function.  Quantization at 1e-4 m/s makes membership insensitive to harmless
    serialization precision while still separating physically distinct questions.
    """
    row = np.round(np.asarray(v_in_row, dtype=np.float64), 4).astype("<f8", copy=False)
    if row.shape != (3,) or not np.isfinite(row).all():
        raise ValueError(f"incoming velocity row must be finite shape (3,), got {row!r}")
    # IEEE -0.0 has a different byte representation from +0.0 but is the same physical
    # velocity. Canonicalize it or serialization details can put one question in two splits.
    row[row == 0.0] = 0.0
    return hashlib.blake2b(row.tobytes(order="C"), digest_size=8).hexdigest()


def question_split(v_in_row) -> str:
    value = int(question_id(v_in_row), 16) / 2.0**64
    return "exam" if value < EXAM_FRAC else "train"


def _is_hex_sha256(value) -> bool:
    """True only for one lowercase 64-hex SHA-256 string (fail-closed membership checks)."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def allowed_motion_shas(clip_info: dict) -> tuple:
    """One family's accepted motion SHA-256 set for runtime reconciliation.

    人话:一族题目允许绑定哪些 motion 文件。没有 ``motion_sha256_allowed``(现役所有题库)时
    退回单一 ``motion_sha256`` —— legacy 行为逐字节不变。重绑脚本
    (rebind_question_bank_motion_family.py)会把"触球行逐位相同"的 TOPP 烤入变速片段追加进
    列表;列表必须非空、逐项 64-hex、无重复、且包含原始 motion_sha256(答案是在原件上解出来
    的,原件永远合法),否则当场拒绝。
    """
    primary = clip_info.get("motion_sha256")
    extra = clip_info.get("motion_sha256_allowed")
    if extra is None:
        return (primary,)
    if (
        not isinstance(extra, (list, tuple))
        or not extra
        or any(not _is_hex_sha256(sha) for sha in extra)
    ):
        raise ValueError(
            f"motion_sha256_allowed must be a non-empty list of 64-hex SHA-256 strings, "
            f"got {extra!r}"
        )
    if len(set(extra)) != len(extra):
        raise ValueError(f"motion_sha256_allowed contains duplicates: {extra!r}")
    if primary not in extra:
        raise ValueError(
            f"motion_sha256_allowed {extra!r} must contain the primary motion_sha256 "
            f"{primary!r} (the exact motion the answers were solved on)"
        )
    return tuple(extra)


def validate_runtime_motion_contract(
    metadata: dict,
    motion_files: Sequence[str],
    segment_lengths: Sequence[int],
    strike_phases: Sequence[float],
    clip_families: Sequence[int] | None = None,
) -> None:
    """Bind a validated bank to the demonstrations/timing actually loaded by training/eval.

    A bank's answers are solved at one contact frame on one exact motion. Merely validating the
    bank internally is insufficient: pointing the task at a different file with the same clip
    name, or changing ``strike_phase_per_clip``, silently asks the answer at the wrong pose/time.
    This check is intentionally content-addressed (paths may differ across pods).

    ``clip_families=None``(现役所有臂)= 旧合同逐字节不变:第 i 个加载 clip 对账 clip_order
    第 i 名,SHA 必须等于该名下的单一 motion_sha256。传入族表(每个加载 clip 属于题库哪一族,
    长度 = 加载 clip 数)后改为族寻址对账:clip 数可以多于题库族数(6-clip 变速列表),每个
    clip 的 motion SHA 必须落在其族的允许列表(:func:`allowed_motion_shas`)里;帧数与击球
    锚帧仍须逐 clip 严格等于该族记录值 —— TOPP 烤入片段保持帧数/锚帧不变,这正是"题目答案
    按族复用合法"的前提,松不得。
    """
    order = list(metadata.get("clip_order") or [])
    if clip_families is None:
        if not (len(order) == len(motion_files) == len(segment_lengths) == len(strike_phases)):
            raise ValueError(
                "question-bank runtime contract length mismatch: "
                f"clips={len(order)}, files={len(motion_files)}, segments={len(segment_lengths)}, "
                f"strike phases={len(strike_phases)}"
            )
        names = list(order)
    else:
        families = list(clip_families)
        if not (len(families) == len(motion_files) == len(segment_lengths) == len(strike_phases)):
            raise ValueError(
                "question-bank family contract length mismatch: "
                f"families={len(families)}, files={len(motion_files)}, "
                f"segments={len(segment_lengths)}, strike phases={len(strike_phases)}"
            )
        names = []
        for fam in families:
            if isinstance(fam, bool) or float(fam) != float(int(fam)):
                raise ValueError(f"clip family indices must be integers, got {fam!r}")
            fam = int(fam)
            if not 0 <= fam < len(order):
                raise ValueError(
                    f"clip family index {fam} is outside the bank's clip_order "
                    f"{order!r} (valid: 0..{len(order) - 1})"
                )
            names.append(order[fam])
    clips = metadata.get("clips") or {}
    for name, path, runtime_n, runtime_phase in zip(
        names, motion_files, segment_lengths, strike_phases
    ):
        info = clips.get(name) or {}
        allowed = allowed_motion_shas(info)
        actual_sha = sha256_file(path)
        if actual_sha not in allowed:
            if len(allowed) == 1:
                raise ValueError(
                    f"question bank clip {name!r}: loaded motion SHA {actual_sha} != bank "
                    f"{info.get('motion_sha256')!r} ({path})"
                )
            raise ValueError(
                f"question bank clip {name!r}: loaded motion SHA {actual_sha} is not in the "
                f"family's allowed list {list(allowed)} ({path})"
            )
        runtime_n = int(runtime_n)
        runtime_phase = float(runtime_phase)
        if runtime_n != int(info.get("n_frames", -1)):
            raise ValueError(
                f"question bank clip {name!r}: runtime has {runtime_n} frames, bank records "
                f"{info.get('n_frames')!r}"
            )
        if not np.isfinite(runtime_phase) or not 0.0 <= runtime_phase <= 1.0:
            raise ValueError(
                f"question bank clip {name!r}: invalid runtime strike phase {runtime_phase!r}"
            )
        runtime_frame = int(round(runtime_phase * (runtime_n - 1)))
        if runtime_frame != int(info.get("anchor_frame", -1)):
            raise ValueError(
                f"question bank clip {name!r}: runtime strike phase {runtime_phase:.6g} -> "
                f"frame {runtime_frame}, bank answers are anchored at frame "
                f"{info.get('anchor_frame')!r}"
            )


class QuestionBank(NamedTuple):
    """Per-clip question tensors. Clips share Q_max rows; rows >= counts[c] are zero padding
    (never selected — :func:`select_questions` draws indices below the per-clip count)."""

    contact_pos: torch.Tensor  # (C, 3) fixed strike point per clip, tracking-env frame
    incoming_vel: torch.Tensor  # (C, Q_max, 3) incoming ball velocity at contact
    incoming_spin: torch.Tensor  # (C, Q_max, 3) incoming ball spin at contact (rad/s)
    demanded_vel: torch.Tensor  # (C, Q_max, 3) solved racket velocity per question
    demanded_normal: torch.Tensor  # (C, Q_max, 3) solved face normal per question
    counts: torch.Tensor  # (C,) long — valid question rows per clip
    difficulty_deg: torch.Tensor  # (C, Q_max) face-vs-clip-normal angle (0 where absent)
    metadata: dict  # validated schema-v3 manifest (empty only for explicit legacy loads)
    source_path: str  # absolute bank path used for checkpoint/export provenance


def load_question_bank(
    path: str,
    device: str | torch.device = "cpu",
    clip_names: Sequence[str] = ("forehand", "backhand"),
    allow_legacy: bool = False,
    expected_split: str | None = None,
) -> QuestionBank:
    """Load a stage-1 bank npz ONCE into per-clip float32 tensors on ``device``.

    Every name in ``clip_names`` must be present in the bank with >= 1 question — a missing or
    empty clip raises (do not train a clip on silent zeros). ``clip_names`` order defines the
    clip_id indexing (must match RacketTargetCommand._clip_names: 0=forehand, 1=backhand).

    Schema v3 additionally binds an explicit train/exam split, deterministic split algorithm,
    source-family hash, per-clip motion SHA/frame anchor and a passing torch closed-loop check.
    Training/evaluation callers MUST pass ``expected_split`` (``train``/``exam`` respectively).
    Older banks are accepted only with ``allow_legacy=True``; their missing spin tensor is then
    interpreted as zero solely to reproduce the old spin-blind generator.
    """
    if expected_split not in (None, "train", "exam"):
        raise ValueError(f"expected_split must be 'train', 'exam' or None, got {expected_split!r}")
    data = np.load(path)
    meta = None
    if "meta_json" in data:
        try:
            meta = json.loads(bytes(np.asarray(data["meta_json"], dtype=np.uint8)).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            meta = None
    schema = int(meta.get("schema_version", 0)) if meta is not None else 0
    strict_v3 = schema == SCHEMA_VERSION
    if not strict_v3 and not allow_legacy:
        if meta is None:
            raise ValueError(
                f"question bank {path!r} has no valid meta_json; regenerate with the current "
                f"gen_stage1_questions.py (or pass allow_legacy=True to load it anyway)"
            )
        raise ValueError(
            f"question bank {path!r}: schema_version={meta.get('schema_version')!r}, expected "
            f"{SCHEMA_VERSION}; regenerate or explicitly pass allow_legacy=True"
        )

    if strict_v3:
        required_meta = {
            "stage": "S1",
            "face_frame": "mount_plusY_A",
            "incoming_spin_mode": "zero",
            "spin_units": "rad/s",
        }
        bad = {k: (meta.get(k), expected) for k, expected in required_meta.items()
               if meta.get(k) != expected}
        if bad:
            raise ValueError(
                f"question bank {path!r}: invalid S1 metadata {bad}; regenerate with the current generator"
            )
        split = meta.get("split")
        if split not in ("train", "exam"):
            raise ValueError(
                f"question bank {path!r}: schema-v3 split must be explicit train/exam, got {split!r}"
            )
        if expected_split is not None and split != expected_split:
            raise ValueError(
                f"question bank {path!r}: split={split!r}, caller requires {expected_split!r}"
            )
        if meta.get("split_algorithm") != SPLIT_ALGORITHM:
            raise ValueError(
                f"question bank {path!r}: split_algorithm={meta.get('split_algorithm')!r}, "
                f"expected {SPLIT_ALGORITHM!r}"
            )
        if list(meta.get("clip_order") or []) != list(clip_names):
            raise ValueError(
                f"question bank {path!r}: clip_order={meta.get('clip_order')!r}, "
                f"expected {list(clip_names)!r}"
            )
        family = meta.get("source_family_contract")
        family_sha = meta.get("source_family_sha256")
        if not isinstance(family, dict) or canonical_sha256(family) != family_sha:
            raise ValueError(
                f"question bank {path!r}: source_family_contract/hash mismatch"
            )
        family_header = {
            "contract": SOURCE_FAMILY_CONTRACT,
            "stage": "S1", "face_frame": "mount_plusY_A",
            "incoming_spin_mode": "zero", "split_algorithm": SPLIT_ALGORITHM,
            "clip_order": list(clip_names),
        }
        family_bad = {key: (family.get(key), value) for key, value in family_header.items()
                      if family.get(key) != value}
        if family_bad:
            raise ValueError(
                f"question bank {path!r}: source-family header disagrees with bank {family_bad}"
            )
        physics_sha = str(meta.get("physics_sha256", ""))
        if (len(physics_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in physics_sha)
                or family.get("physics_sha256") != physics_sha):
            raise ValueError(
                f"question bank {path!r}: physics SHA is invalid or disagrees with source family"
            )
        physics_contract_sha = str(meta.get("physics_contract_sha256", ""))
        if family.get("physics_contract_sha256") != physics_contract_sha:
            raise ValueError(
                f"question bank {path!r}: physics runtime contract disagrees with source family"
            )
        validate_runtime_physics_contract(meta)
        expected_family_sampling = {
            "incoming": {
                "speed_range": list(meta.get("speed_range") or []),
                "vy_max": meta.get("vy_max"),
                "vz_range": list(meta.get("vz_range") or []),
            },
            "landing_env": list(meta.get("landing_env") or []),
            "near_x": meta.get("near_x"),
            "table_surface_z": meta.get("table_surface_z"),
            "speed_budget": meta.get("speed_budget"),
            "physics_sha256": physics_sha,
            "physics_contract_sha256": physics_contract_sha,
        }
        sampling_bad = {
            key: (family.get(key), expected)
            for key, expected in expected_family_sampling.items()
            if family.get(key) != expected
        }
        if sampling_bad:
            raise ValueError(
                f"question bank {path!r}: top-level sampling/physics metadata disagrees with "
                f"source family {sampling_bad}"
            )
        validation = meta.get("validation") or {}
        if (validation.get("torch_closed_loop_pass") is not True
                or validation.get("net_clearance_all_pass") is not True):
            raise ValueError(
                f"question bank {path!r}: no passing torch landing/net closed-loop proof"
            )
        max_landing_error = float(validation.get("max_landing_error_m", np.nan))
        if not np.isfinite(max_landing_error) or max_landing_error > 0.10:
            raise ValueError(
                f"question bank {path!r}: validation.max_landing_error_m="
                f"{max_landing_error!r}, required finite <= 0.10 m"
            )
        grip_by_clip = meta.get("grip_applied_per_clip") or {}
        yaw_by_clip = meta.get("rally_yaw_applied_per_clip") or {}
        if meta.get("grip_applied") is not True or meta.get("rally_yaw_applied") is not True or any(
            grip_by_clip.get(name) is not True or yaw_by_clip.get(name) is not True
            for name in clip_names
        ):
            raise ValueError(
                f"question bank {path!r}: every clip must prove grip calibration and rally-yaw "
                f"canonicalization; grip={grip_by_clip}, rally_yaw={yaw_by_clip}"
            )
    elif expected_split is not None:
        raise ValueError(
            f"question bank {path!r}: expected_split={expected_split!r} requires schema v3; "
            "legacy banks have no auditable split contract"
        )

    per_clip = []
    for name in clip_names:
        key = f"{name}/contact_pos_env"
        if key not in data:
            raise KeyError(
                f"question bank {path!r}: no entry for clip {name!r} (missing {key!r}); "
                f"regenerate with gen_stage1_questions.py --clip {name}:<motion.npz>"
            )
        contact = np.asarray(data[key], dtype=np.float64).reshape(3)
        incoming = np.asarray(data[f"{name}/incoming_vel"], dtype=np.float64).reshape(-1, 3)
        spin_key = f"{name}/incoming_spin"
        if spin_key in data:
            incoming_spin = np.asarray(data[spin_key], dtype=np.float64).reshape(-1, 3)
        elif allow_legacy:
            incoming_spin = np.zeros_like(incoming)
        else:
            raise KeyError(
                f"question bank {path!r} clip {name!r}: missing {spin_key!r}; regenerate schema v3"
            )
        vel = np.asarray(data[f"{name}/demanded_vel"], dtype=np.float64).reshape(-1, 3)
        nrm = np.asarray(data[f"{name}/demanded_normal"], dtype=np.float64).reshape(-1, 3)
        if len(vel) == 0 or not (incoming.shape == incoming_spin.shape == vel.shape == nrm.shape):
            raise ValueError(
                f"question bank {path!r} clip {name!r}: incoming_vel {incoming.shape}, "
                f"incoming_spin {incoming_spin.shape}, demanded_vel {vel.shape}, demanded_normal "
                f"{nrm.shape} must be matching non-empty (Q, 3) arrays"
            )
        arrays = (contact, incoming, incoming_spin, vel, nrm)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"question bank {path!r} clip {name!r}: non-finite value in question tuple")
        if strict_v3 and not np.allclose(incoming_spin, 0.0, atol=0.0, rtol=0.0):
            raise ValueError(
                f"question bank {path!r} clip {name!r}: S1 schema declares zero spin but contains "
                f"max |spin|={float(np.abs(incoming_spin).max()):.6g} rad/s"
            )
        dkey = f"{name}/difficulty_deg"
        if strict_v3 and dkey not in data:
            raise KeyError(
                f"question bank {path!r} clip {name!r}: schema-v3 requires {dkey!r}"
            )
        diff = (
            np.asarray(data[dkey], dtype=np.float64).reshape(-1)
            if dkey in data
            else np.zeros(len(vel), dtype=np.float32)
        )
        if len(diff) != len(vel):
            raise ValueError(
                f"question bank {path!r} clip {name!r}: difficulty_deg has {len(diff)} rows, "
                f"expected {len(vel)}"
            )
        if not np.isfinite(diff).all():
            raise ValueError(f"question bank {path!r} clip {name!r}: non-finite difficulty_deg")
        if strict_v3 and (np.any(diff < 0.0) or np.any(diff > 90.0 + 2e-3)):
            raise ValueError(
                f"question bank {path!r} clip {name!r}: A-frame difficulty must lie in [0, 90]"
            )
        if strict_v3:
            clip_info = (meta.get("clips") or {}).get(name) or {}
            required_clip = (
                "motion_basename", "motion_sha256", "n_frames", "anchor_frame",
                "anchor_phase", "question_count", "grip_mode", "grip_rotation_matrix",
                "rally_yaw_deg", "contact_pos_env", "clip_normal", "clip_vel",
            )
            missing = [key for key in required_clip if key not in clip_info]
            if missing:
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: missing provenance {missing}"
                )
            if int(clip_info["question_count"]) != len(vel):
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: metadata question_count="
                    f"{clip_info['question_count']} but arrays contain {len(vel)}"
                )
            motion_sha = str(clip_info["motion_sha256"])
            if len(motion_sha) != 64 or any(ch not in "0123456789abcdef" for ch in motion_sha):
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: invalid motion_sha256 {motion_sha!r}"
                )
            # 重绑扩展键(可选):有就必须自洽,坏列表当场拒绝;没有(现役题库)不加任何检查。
            if "motion_sha256_allowed" in clip_info:
                try:
                    allowed_motion_shas(clip_info)
                except ValueError as exc:
                    raise ValueError(
                        f"question bank {path!r} clip {name!r}: {exc}"
                    ) from exc
            family_clip = (family.get("clips") or {}).get(name) or {}
            clip_vel_key = f"{name}/clip_vel"
            if clip_vel_key not in data:
                raise KeyError(
                    f"question bank {path!r} clip {name!r}: missing {clip_vel_key!r}"
                )
            clip_vel = np.asarray(data[clip_vel_key], dtype=np.float64).reshape(3)
            if not np.isfinite(clip_vel).all():
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: non-finite clip velocity"
                )
            rounded_contact = np.round(contact, 12).tolist()
            rounded_clip_vel = np.round(clip_vel, 12).tolist()
            expected_family_clip = {
                "motion_sha256": motion_sha,
                "anchor_frame": int(clip_info["anchor_frame"]),
                "anchor_phase": float(clip_info["anchor_phase"]),
                "grip_mode": clip_info["grip_mode"],
                "grip_rotation_matrix": clip_info["grip_rotation_matrix"],
                "rally_yaw_deg": float(clip_info["rally_yaw_deg"]),
                "contact_pos_env": rounded_contact,
                "clip_normal": list(clip_info["clip_normal"]),
                "clip_vel": rounded_clip_vel,
            }
            mismatch = {key: (family_clip.get(key), value)
                        for key, value in expected_family_clip.items()
                        if family_clip.get(key) != value}
            if mismatch:
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: source-family provenance "
                    f"disagrees with clip metadata {mismatch}"
                )
            n_frames = int(clip_info["n_frames"])
            anchor_frame = int(clip_info["anchor_frame"])
            anchor_phase = float(clip_info["anchor_phase"])
            if n_frames < 2 or not 0 <= anchor_frame < n_frames or not np.isfinite(anchor_phase):
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: invalid frame/phase provenance"
                )
            if int(round(anchor_phase * (n_frames - 1))) != anchor_frame:
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: anchor_phase/frame disagree"
                )
            memberships = {question_split(row) for row in incoming}
            if memberships != {meta["split"]}:
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: rows have split memberships "
                    f"{sorted(memberships)}, manifest says {meta['split']!r}"
                )
            ids = [question_id(row) for row in incoming]
            if len(ids) != len(set(ids)):
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: duplicate incoming questions"
                )
            nrm_norm = np.linalg.norm(nrm, axis=1)
            if not np.allclose(nrm_norm, 1.0, atol=2e-4, rtol=0.0):
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: demanded normals are not unit length"
                )
            clip_normal_key = f"{name}/clip_normal"
            if clip_normal_key not in data:
                raise KeyError(
                    f"question bank {path!r} clip {name!r}: missing {clip_normal_key!r}"
                )
            clip_normal = np.asarray(data[clip_normal_key], dtype=np.float64).reshape(3)
            clip_normal_norm = float(np.linalg.norm(clip_normal))
            if (not np.isfinite(clip_normal).all()
                    or not np.isclose(clip_normal_norm, 1.0, atol=2e-4, rtol=0.0)):
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: clip_normal must already be unit "
                    f"length, got norm {clip_normal_norm:.8g}"
                )
            rounded_clip_normal = np.round(clip_normal, 12).tolist()
            geometry_mismatch = {}
            for key, actual in (
                ("contact_pos_env", rounded_contact),
                ("clip_normal", rounded_clip_normal),
                ("clip_vel", rounded_clip_vel),
            ):
                if clip_info.get(key) != actual or family_clip.get(key) != actual:
                    geometry_mismatch[key] = {
                        "array": actual,
                        "clip_meta": clip_info.get(key),
                        "family": family_clip.get(key),
                    }
            if geometry_mismatch:
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: geometry arrays/provenance disagree "
                    f"{geometry_mismatch}"
                )
            grip_rotation = clip_info.get("grip_rotation_matrix")
            if clip_info.get("grip_mode") == "registry" and grip_rotation is None:
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: registry grip lacks its rotation"
                )
            if clip_info.get("grip_mode") not in ("registry", "baked"):
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: unsupported strict grip mode "
                    f"{clip_info.get('grip_mode')!r}"
                )
            if grip_rotation is not None:
                grip_rotation_arr = np.asarray(grip_rotation, dtype=np.float64)
                if (grip_rotation_arr.shape != (3, 3)
                        or not np.isfinite(grip_rotation_arr).all()
                        or not np.allclose(grip_rotation_arr.T @ grip_rotation_arr, np.eye(3),
                                           atol=2e-6, rtol=0.0)
                        or not np.isclose(np.linalg.det(grip_rotation_arr), 1.0,
                                          atol=2e-6, rtol=0.0)):
                    raise ValueError(
                        f"question bank {path!r} clip {name!r}: invalid grip rotation matrix"
                    )
            expected_diff = np.degrees(np.arccos(np.clip(nrm @ clip_normal, -1.0, 1.0)))
            if not np.allclose(diff, expected_diff, atol=2e-3, rtol=0.0):
                raise ValueError(
                    f"question bank {path!r} clip {name!r}: difficulty_deg does not match normals"
                )
        per_clip.append((contact, incoming, incoming_spin, vel, nrm, diff))

    q_max = max(len(v) for _, v, _, _, _, _ in per_clip)
    n_clips = len(per_clip)
    contact_pos = torch.zeros(n_clips, 3)
    incoming_vel = torch.zeros(n_clips, q_max, 3)
    incoming_spin = torch.zeros(n_clips, q_max, 3)
    demanded_vel = torch.zeros(n_clips, q_max, 3)
    demanded_normal = torch.zeros(n_clips, q_max, 3)
    counts = torch.zeros(n_clips, dtype=torch.long)
    difficulty = torch.zeros(n_clips, q_max)
    for c, (contact, incoming, spin, vel, nrm, diff) in enumerate(per_clip):
        q = len(vel)
        contact_pos[c] = torch.from_numpy(contact)
        incoming_vel[c, :q] = torch.from_numpy(incoming)
        incoming_spin[c, :q] = torch.from_numpy(spin)
        demanded_vel[c, :q] = torch.from_numpy(vel)
        demanded_normal[c, :q] = torch.from_numpy(nrm)
        difficulty[c, :q] = torch.from_numpy(diff)
        counts[c] = q
    return QuestionBank(
        contact_pos=contact_pos.to(device),
        incoming_vel=incoming_vel.to(device),
        incoming_spin=incoming_spin.to(device),
        demanded_vel=demanded_vel.to(device),
        demanded_normal=demanded_normal.to(device),
        counts=counts.to(device),
        difficulty_deg=difficulty.to(device),
        metadata=(meta if strict_v3 else {}),
        source_path=os.path.abspath(path),
    )


def face_command_obs_vector(normal: torch.Tensor) -> torch.Tensor:
    """Contract-day 179-D face-command lane: ``[demanded normal (3), rho placeholder (1)]``.

    rho (the S3 spin-lane scalar) is reserved NOW, zero-filled, so the actor layout matches the
    frozen 175 -> 179 contract-day decision and S3 needs no further contract change / ladder
    retrain. Pure tensor helper (unit-tested without the env); the obs term wraps it.
    """
    return torch.cat([normal, normal.new_zeros(normal.shape[:-1] + (1,))], dim=-1)


def select_questions(
    bank: QuestionBank, clip_ids: torch.Tensor, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map uniform draws ``u`` in [0, 1) to one bank row per env for its clip.

    Returns ``(contact_pos, incoming_vel, incoming_spin, demanded_vel, demanded_normal,
    difficulty_deg)``, each row-selected by
    (clip_ids, floor(u * counts[clip])). The index is clamped below the per-clip count so zero
    padding rows are never selected. Pure tensor function — unit-tested without the env.
    """
    counts = bank.counts[clip_ids]
    q = (u * counts.to(u.dtype)).long().clamp_(min=0)
    q = torch.minimum(q, counts - 1)
    return (
        bank.contact_pos[clip_ids],
        bank.incoming_vel[clip_ids, q],
        bank.incoming_spin[clip_ids, q],
        bank.demanded_vel[clip_ids, q],
        bank.demanded_normal[clip_ids, q],
        bank.difficulty_deg[clip_ids, q],
    )
