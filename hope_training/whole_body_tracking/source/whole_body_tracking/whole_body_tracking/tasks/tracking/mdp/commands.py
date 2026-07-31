from __future__ import annotations

import hashlib
import io
import json
import math
import numpy as np
import os
import stat
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.mdp.event_timing import (
    EVENT_TIMING_MODE_DISABLED,
    EVENT_TIMING_MODE_POST_STRIKE_T1,
    EVENT_TIMING_MODES,
    EventTimingScheduler,
    load_event_schedule,
)
from whole_body_tracking.tasks.tracking.mdp.post_swing_teacher import (
    CAPTURE_CLAIM_KIND,
    CAPTURE_CLAIM_NAME,
    CAPTURE_CONTRACT,
    CAPTURE_RESULT_KIND,
    CAPTURE_RESULT_NAME,
    CAPTURE_STATE_NAME,
    PostSwingTeacherError,
    _canonical_json_bytes,
    _publish_bytes_no_clobber,
    load_post_swing_teacher_states,
    sha256_file,
)
from whole_body_tracking.tasks.tracking.mdp.planner_revision import (
    InitialTtsMixture,
    PLANNER_TASK_REVISION_SCHEMA_VERSION,
    PhaseGovernorProfile,
)


def _stand_start_yaw_samples(yaw_range, count: int, device):
    """Return stand-start yaw samples, or ``None`` for the byte-identical [0, 0] default.

    A degenerate non-zero range is a deterministic curriculum point, not an off switch.
    Avoiding an RNG draw there also makes fixed-yaw evaluation exactly reproducible.
    """
    yaw_lo, yaw_hi = (float(yaw_range[0]), float(yaw_range[1]))
    if yaw_lo == 0.0 and yaw_hi == 0.0:
        return None
    if yaw_lo == yaw_hi:
        return torch.full((count,), yaw_lo, device=device)
    return sample_uniform(yaw_lo, yaw_hi, (count,), device)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_CANONICAL_REGISTRY_RUNTIME_MODULE = None
_ACTION_BALL_RUNTIME_MODULE = None


class MotionLoader:
    """Loads one or more motion clips into a single concatenated time axis.

    Passing several files (HITTER unified policy: forehand + backhand) concatenates them along the time
    dimension and records per-clip ``seg_start`` / ``seg_len`` so the command can step/wrap/strike within
    one clip ("segment") at a time, selected per-env by swing type. A single file behaves exactly as
    before: one segment spanning the whole motion, ``time_step_total`` unchanged.
    """

    _KINEMATICS_SCHEMA = 2
    _KINEMATICS_CORE_KEYS = (
        "kinematics_schema_version", "body_pos_point", "body_lin_vel_point"
    )
    _KINEMATICS_BODY_NAMES_KEY = "body_names"

    @staticmethod
    def _meta_scalar(data, key: str) -> str:
        raw = np.asarray(data[key]).reshape(-1)
        if raw.size != 1:
            raise ValueError(f"motion metadata {key} must be scalar, got {np.asarray(data[key]).shape}")
        value = raw[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)

    @staticmethod
    def _meta_body_names(data, key: str) -> tuple[str, ...]:
        raw = np.asarray(data[key])
        if raw.ndim != 1:
            raise ValueError(f"motion metadata {key} must be one-dimensional, got {raw.shape}")
        names = []
        for value in raw.tolist():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            names.append(str(value))
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"motion metadata {key} must contain unique non-empty names")
        return tuple(names)

    @staticmethod
    def _fps_scalar(data, path: str) -> float:
        raw = np.asarray(data["fps"])
        if raw.size != 1:
            raise ValueError(f"{path}: fps must be scalar, got shape {raw.shape}")
        fps = float(raw.reshape(-1)[0])
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{path}: fps must be finite and positive, got {fps!r}")
        return fps

    @staticmethod
    def _validate_motion_array_shapes(
        data, path: str, articulation_body_count: int
    ) -> int:
        """Validate the shared time axis and full-articulation body-column shape."""

        expected_tail = {
            "body_pos_w": (articulation_body_count, 3),
            "body_quat_w": (articulation_body_count, 4),
            "body_lin_vel_w": (articulation_body_count, 3),
            "body_ang_vel_w": (articulation_body_count, 3),
        }
        arrays = {key: np.asarray(data[key]) for key in ("joint_pos", "joint_vel", *expected_tail)}
        if arrays["joint_pos"].ndim != 2 or arrays["joint_vel"].shape != arrays["joint_pos"].shape:
            raise ValueError(
                f"{path}: joint_pos/joint_vel must have the same (T,J) shape, got "
                f"{arrays['joint_pos'].shape}/{arrays['joint_vel'].shape}"
            )
        frame_count = int(arrays["joint_pos"].shape[0])
        if frame_count <= 0:
            raise ValueError(f"{path}: motion clip contains no frames")
        for key, tail in expected_tail.items():
            expected = (frame_count, *tail)
            if arrays[key].shape != expected:
                raise ValueError(f"{path}: {key} has shape {arrays[key].shape}, expected {expected}")
        return frame_count

    @classmethod
    def _kinematics_contract(
        cls,
        data,
        path: str,
        articulation_body_names: tuple[str, ...],
        *,
        allow_legacy_link_origin_velocity: bool = False,
    ) -> dict:
        """Validate body point semantics without guessing from a filename.

        Untagged historical Isaac clips remain loadable but exact-ineligible.
        Untagged legacy MuJoCo/retime clips have a decisive content signature:
        body_lin_vel_w == d(body_pos_w)/dt under meaningful angular motion.
        Those are fail-closed because MotionCommand rewards COM velocity.
        """

        files = set(data.files)
        present = [key in files for key in cls._KINEMATICS_CORE_KEYS]
        if any(present) and not all(present):
            raise ValueError(f"{path}: partial/malformed motion kinematics metadata")
        if not any(present) and cls._KINEMATICS_BODY_NAMES_KEY in files:
            raise ValueError(f"{path}: body_names exists without a kinematics schema")
        if all(present):
            schema_raw = np.asarray(data[cls._KINEMATICS_CORE_KEYS[0]]).reshape(-1)
            if schema_raw.size != 1:
                raise ValueError(
                    f"{path}: kinematics_schema_version must be scalar, got "
                    f"{np.asarray(data[cls._KINEMATICS_CORE_KEYS[0]]).shape}"
                )
            schema = int(schema_raw[0])
            pos_point = cls._meta_scalar(data, cls._KINEMATICS_CORE_KEYS[1])
            vel_point = cls._meta_scalar(data, cls._KINEMATICS_CORE_KEYS[2])
            if schema not in (1, cls._KINEMATICS_SCHEMA) or pos_point != "link_origin":
                raise ValueError(
                    f"{path}: unsupported motion kinematics contract "
                    f"schema={schema} pos={pos_point!r} vel={vel_point!r}"
                )
            if vel_point != "center_of_mass":
                raise ValueError(
                    f"{path}: body_lin_vel_point={vel_point!r}, but Isaac MotionCommand compares "
                    "against COM velocity. Run scripts/migrate_motion_kinematics.py with an explicit "
                    "--source-point; link-origin velocity must not enter formal training."
                )
            body_names = None
            if cls._KINEMATICS_BODY_NAMES_KEY in files:
                body_names = cls._meta_body_names(data, cls._KINEMATICS_BODY_NAMES_KEY)
                if body_names != articulation_body_names:
                    raise ValueError(
                        f"{path}: body_names/order does not match the runtime articulation: "
                        f"file={list(body_names)} runtime={list(articulation_body_names)}"
                    )
            if schema == cls._KINEMATICS_SCHEMA and body_names is None:
                raise ValueError(f"{path}: schema-{schema} motion is missing body_names")
            exact = schema == cls._KINEMATICS_SCHEMA and body_names is not None
            return {
                "schema_version": schema,
                "body_pos_point": pos_point,
                "body_lin_vel_point": vel_point,
                "body_names": None if body_names is None else list(body_names),
                "exact": exact,
                "status": "declared_v2" if exact else "legacy_v1_unbound_body_order",
            }

        pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        lin = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
        ang = np.asarray(data["body_ang_vel_w"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if pos.shape != lin.shape or ang.shape != lin.shape or len(pos) < 2 or fps <= 0.0:
            raise ValueError(f"{path}: invalid legacy motion arrays for point-semantics audit")
        link_fd = np.gradient(pos, 1.0 / fps, axis=0)
        fd_max = float(np.max(np.abs(lin - link_fd)))
        max_ang = float(np.max(np.linalg.norm(ang, axis=-1)))
        if max_ang > 0.2 and fd_max <= 1.0e-4:
            if not allow_legacy_link_origin_velocity:
                raise ValueError(
                    f"{path}: untagged body_lin_vel_w is numerically d(link-origin position)/dt "
                    f"(max residual {fd_max:.3e} m/s, max |omega| {max_ang:.2f} rad/s), but "
                    "MotionCommand rewards COM velocity. This is the pre-2026-07-10 V5/MuJoCo "
                    "converter signature. Migrate it explicitly with "
                    "scripts/migrate_motion_kinematics.py --source-point link_origin; refusing "
                    "to train on the wrong point."
                )
            return {
                "schema_version": None,
                "body_pos_point": "link_origin",
                "body_lin_vel_point": "link_origin",
                "body_names": None,
                "exact": False,
                "status": "legacy_link_origin_velocity_diagnostic_only",
                "link_fd_max_abs_mps": fd_max,
                "max_ang_radps": max_ang,
            }
        return {
            "schema_version": None, "body_pos_point": None, "body_lin_vel_point": None,
            "body_names": None,
            "exact": False, "status": "legacy_unbound_assumed_com",
            "link_fd_max_abs_mps": fd_max, "max_ang_radps": max_ang,
        }

    def __init__(
        self,
        motion_file,
        body_indexes: Sequence[int],
        *,
        motion_payloads: Sequence[bytes] | None = None,
        articulation_body_names: Sequence[str],
        selected_body_names: Sequence[str],
        device: str = "cpu",
        allow_legacy_link_origin_velocity: bool = False,
    ):
        files = [motion_file] if isinstance(motion_file, str) else list(motion_file)
        if not files:
            raise ValueError("MotionLoader needs at least one motion file")
        if motion_payloads is None:
            payloads: tuple[bytes | None, ...] = (None,) * len(files)
        else:
            try:
                payloads = tuple(motion_payloads)
            except TypeError as exc:
                raise ValueError(
                    "MotionLoader motion_payloads must be an ordered byte sequence"
                ) from exc
            if len(payloads) != len(files) or any(
                type(payload) is not bytes for payload in payloads
            ):
                raise ValueError(
                    "MotionLoader needs exactly one immutable bytes snapshot per motion file"
                )
        articulation_names = tuple(str(name) for name in articulation_body_names)
        selected_names = tuple(str(name) for name in selected_body_names)
        if (not articulation_names or len(set(articulation_names)) != len(articulation_names)
                or not selected_names or len(set(selected_names)) != len(selected_names)):
            raise ValueError("runtime articulation/selected body names must be non-empty and unique")
        indexes = [int(value) for value in (
            body_indexes.detach().cpu().tolist()
            if hasattr(body_indexes, "detach")
            else list(body_indexes)
        )]
        if len(indexes) != len(selected_names):
            raise ValueError(
                f"selected body indexes/names disagree: {indexes} vs {list(selected_names)}"
            )
        if any(index < 0 or index >= len(articulation_names) for index in indexes):
            raise ValueError(f"selected body index is outside articulation order: {indexes}")
        resolved_selected = tuple(articulation_names[index] for index in indexes)
        if resolved_selected != selected_names:
            raise ValueError(
                f"runtime selected body order mismatch: indexes resolve to {list(resolved_selected)}, "
                f"configured={list(selected_names)}"
            )
        jp, jv, bp, bq, bl, ba = [], [], [], [], [], []
        seg_lens = []
        self.kinematics_contracts = []
        per_clip_fps = []
        for f, payload in zip(files, payloads):
            if payload is None:
                if not os.path.isfile(f):
                    raise FileNotFoundError(f"Invalid motion file path: {f}")
                source = f
            else:
                source = io.BytesIO(payload)
            with np.load(source, allow_pickle=False) as data:
                fps = self._fps_scalar(data, f)
                per_clip_fps.append(fps)
                frame_count = self._validate_motion_array_shapes(
                    data, f, len(articulation_names)
                )
                _kin = self._kinematics_contract(
                    data,
                    f,
                    articulation_names,
                    allow_legacy_link_origin_velocity=allow_legacy_link_origin_velocity,
                )
                self.kinematics_contracts.append(_kin)
                if not _kin["exact"]:
                    print(
                        f"[MotionLoader WARN] {f}: legacy motion lacks a schema-2 bound body order; "
                        "allowed for checkpoint compatibility but formal lineage is exact-ineligible. "
                        "Migrate/re-export the clip with kinematics schema 2. "
                        f"audit={_kin}",
                        flush=True,
                    )
                jp.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
                jv.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
                bp.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
                bq.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
                bl.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
                ba.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
                seg_lens.append(frame_count)
        first_fps = per_clip_fps[0]
        if any(not math.isclose(value, first_fps, rel_tol=0.0, abs_tol=1.0e-12)
               for value in per_clip_fps[1:]):
            raise ValueError(f"motion clips have unequal fps values: {per_clip_fps}")
        self.fps = first_fps
        self.per_clip_fps = tuple(per_clip_fps)
        self.joint_pos = torch.cat(jp, dim=0)
        self.joint_vel = torch.cat(jv, dim=0)
        self._body_pos_w = torch.cat(bp, dim=0)
        self._body_quat_w = torch.cat(bq, dim=0)
        self._body_lin_vel_w = torch.cat(bl, dim=0)
        self._body_ang_vel_w = torch.cat(ba, dim=0)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        # Per-clip segment boundaries on the concatenated time axis.
        self.num_segments = len(seg_lens)
        self.seg_len = torch.tensor(seg_lens, dtype=torch.long, device=device)
        self.seg_start = torch.zeros(self.num_segments, dtype=torch.long, device=device)
        if self.num_segments > 1:
            self.seg_start[1:] = torch.cumsum(self.seg_len, dim=0)[:-1]
        self.kinematics_contract_exact = all(item["exact"] for item in self.kinematics_contracts)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


CLIP_FAMILY_FOREHAND = "forehand"
CLIP_FAMILY_BACKHAND = "backhand"
_CLIP_FAMILIES = (CLIP_FAMILY_FOREHAND, CLIP_FAMILY_BACKHAND)
_A3_CANONICAL_READY_JOINT_COUNT = 31


def resolve_clip_family_is_forehand(clip_family_per_clip, num_segments: int) -> tuple[bool, ...]:
    """Resolve the per-clip swing-family config into "is clip i a forehand?" booleans.

    人话:回答"第 i 个 clip 是正手还是反手"。没配表(None,现役所有在跑臂)就按老规矩推:
    单 clip 当正手、恰好 2 clip = (正手, 反手)——和四处写死的 ``clips == 0`` 判断逐字节同值;
    3 个及以上 clip 没配表直接报错,因为那正是变速正手变体会被悄悄当成反手训错的场景
    (spdmix v2 可行性备忘 2026-07-22 硬绑定一),宁可开机炸也不猜。配了表就整表核:长度要对上
    加载的 clip 数、值只认 "forehand"/"backhand"、正反手至少各一个,错了当场 ValueError。
    """
    nseg = int(num_segments)
    if nseg < 1:
        raise ValueError(f"clip family resolution needs at least one loaded clip, got {nseg}")
    if clip_family_per_clip is None:
        if nseg == 1:
            return (True,)
        if nseg == 2:
            return (True, False)
        raise ValueError(
            f"the loaded motion has {nseg} clips but task.motion.clip_family_per_clip is unset — "
            "the legacy 'clip 0 is the forehand, every other clip is the backhand' rule only ever "
            "matched the exact (forehand, backhand) 2-clip list; with more clips it would silently "
            "mislabel every extra forehand variant as a backhand (swing_sign/obs/target side all "
            "wrong). Declare one family per clip in motion_file order, e.g. "
            '["forehand","forehand","forehand","backhand","backhand","backhand"].'
        )
    families = tuple(str(value) for value in clip_family_per_clip)
    if len(families) != nseg:
        raise ValueError(
            f"clip_family_per_clip has {len(families)} entries but the loaded motion has {nseg} "
            "clip(s) — align it with the motion_file clip order (same order as "
            "strike_phase_per_clip / mount_normal_sign_per_clip)"
        )
    unknown = sorted(set(families) - set(_CLIP_FAMILIES))
    if unknown:
        raise ValueError(
            f"clip_family_per_clip entries must be one of {_CLIP_FAMILIES}, got {unknown}"
        )
    # The both-families rule is about the UNIFIED policy: with two or more clips, swing_sign, the
    # swing-type observation and the target side are all keyed off the family split, so a one-sided
    # table would train one lane and leave the other dead. A SINGLE-clip run has no split to key on
    # — swing_sign is one constant for every env — so the rule has nothing to protect there, and
    # applying it anyway leaves a single-clip arm no way to say which hand it is. It then falls into
    # the ``None`` default, which hardcodes "single clip is a forehand": every backhand-only arm
    # silently reports as a forehand and its per-side metrics read a structural 0.0000 while the
    # aggregate moves. 人话:一条只有反手的臂本来连"我是反手"都说不出口,只能被默认当成正手,
    # 于是逐侧指标恒为 0 —— 正是 07-26 把 45% 回球率读废的那个坑的镜像。
    if nseg >= 2 and (
        CLIP_FAMILY_FOREHAND not in families or CLIP_FAMILY_BACKHAND not in families
    ):
        raise ValueError(
            "clip_family_per_clip must contain at least one forehand and one backhand clip, got "
            f"{families} — the unified policy keys swing_sign, the swing-type observation and the "
            "target side off both families"
        )
    return tuple(value == CLIP_FAMILY_FOREHAND for value in families)


class _BalancedRoundRobinClipSampler:
    """Deterministic, exactly balanced clip allocation without touching global RNG.

    One seeded permutation defines a cyclic clip order. Every prefix of that
    infinite cycle gives each clip either ``floor(k / N)`` or ``ceil(k / N)``
    assignments, so the cumulative count spread is always at most one even
    when callers use different batch sizes.
    """

    _STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        num_segments: int,
        seed: int,
        clip_order: Sequence[str],
        device,
    ):
        if type(num_segments) is not int or num_segments < 1:
            raise ValueError(
                "balanced clip sampler num_segments must be a positive integer, "
                f"got {num_segments!r}"
            )
        if type(seed) is not int or not (0 <= seed < 2**63):
            raise ValueError(
                "balanced_clip_sampling_seed must be an integer in [0, 2**63)"
            )
        order = tuple(clip_order)
        if len(order) != num_segments or any(
            type(item) is not str for item in order
        ):
            raise ValueError(
                "balanced clip sampler clip_order must contain one path string per segment"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.num_segments = num_segments
        self.seed = seed
        self.clip_order = order
        self.device = torch.device(device)
        self.permutation = torch.randperm(
            num_segments, generator=generator, dtype=torch.long
        ).to(self.device)
        self.cursor = 0

    def sample(self, count: int) -> torch.Tensor:
        if type(count) is not int or count < 0:
            raise ValueError(
                "balanced clip sample count must be a non-negative integer, "
                f"got {count!r}"
            )
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        positions = (
            torch.arange(count, dtype=torch.long, device=self.device) + self.cursor
        ) % self.num_segments
        sampled = self.permutation[positions]
        self.cursor = (self.cursor + count) % self.num_segments
        return sampled

    def state_dict(self) -> dict:
        return {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "num_segments": self.num_segments,
            "seed": self.seed,
            "clip_order": self.clip_order,
            "permutation": tuple(
                int(value) for value in self.permutation.cpu().tolist()
            ),
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: dict):
        if type(state) is not dict:
            raise ValueError("balanced clip sampler state must be a dictionary")
        if state.get("schema_version") != self._STATE_SCHEMA_VERSION:
            raise ValueError(
                "balanced clip sampler state has an unsupported schema_version"
            )
        if state.get("num_segments") != self.num_segments:
            raise ValueError(
                "balanced clip sampler state num_segments does not match the loaded motion"
            )
        if state.get("seed") != self.seed:
            raise ValueError(
                "balanced clip sampler state seed does not match the configured seed"
            )
        if tuple(state.get("clip_order", ())) != self.clip_order:
            raise ValueError(
                "balanced clip sampler state clip_order does not match the loaded motion order"
            )
        permutation = state.get("permutation")
        if type(permutation) not in (tuple, list):
            raise ValueError(
                "balanced clip sampler state permutation must be an ordered sequence"
            )
        if (
            any(type(value) is not int for value in permutation)
            or sorted(permutation) != list(range(self.num_segments))
        ):
            raise ValueError(
                "balanced clip sampler state permutation is not a bijection of clip ids"
            )
        cursor = state.get("cursor")
        if type(cursor) is not int or not (0 <= cursor < self.num_segments):
            raise ValueError(
                "balanced clip sampler state cursor must be an integer inside the permutation"
            )
        # Restore saved bytes instead of regenerating, so exact resume survives
        # a future torch release changing randperm internals.
        self.permutation = torch.tensor(
            permutation, dtype=torch.long, device=self.device
        )
        self.cursor = cursor


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    _EXACT_RESUME_STATE_KIND = "whole_body_tracking.MotionCommand"
    _EXACT_RESUME_STATE_SCHEMA_VERSION = 2
    _ACTION_BALL_EXACT_RESUME_STATE_SCHEMA_VERSION = 4
    _ACTION_BALL_INT64_MAX = (1 << 63) - 1

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        canonical_ready_mode = getattr(self.cfg, "canonical_ready_mode", False)
        if type(canonical_ready_mode) is not bool:
            raise ValueError("canonical_ready_mode must be an exact boolean")
        self.canonical_ready_mode = canonical_ready_mode
        self._canonical_motion_registry = None
        self._canonical_motion_admission = None
        self._canonical_motion_promotion_binding = None
        self._canonical_motion_registry_module = None
        # Freeze Hydra/ListConfig/custom iterable input once.  Admission hashes
        # and MotionLoader must consume the same ordered path identity.
        self._motion_files = self._configured_motion_files(self.cfg.motion_file)
        # Bind the bytes actually admitted at construction, not whatever may happen to occupy
        # the same paths at checkpoint time. Exact resume must never pour an old curriculum or
        # replay ring into different clip content that reused the same filenames.
        self._motion_file_sha256 = tuple(
            sha256_file(path) for path in self._motion_files
        )
        racket_cfg_for_diag = getattr(
            getattr(getattr(env, "cfg", None), "commands", None),
            "racket_target",
            None,
        )
        diagnostic_unauthorized = getattr(
            racket_cfg_for_diag,
            "action_ball_diagnostic_unauthorized",
            False,
        )
        if type(diagnostic_unauthorized) is not bool:
            raise ValueError(
                "action_ball_diagnostic_unauthorized must be an exact boolean"
            )
        self._canonical_diagnostic_unauthorized = diagnostic_unauthorized
        if self.canonical_ready_mode and diagnostic_unauthorized:
            # Franco 2026-07-28 approved DIAGNOSTIC bypass: skip the registry
            # trust chain only.  The physical canonical-ready clip contract
            # (_validate_canonical_ready_clips) and the reset-curricula guard
            # below stay fully enforced — a bypassed run may not corrupt the
            # ready-entry geometry, it may only skip authorization.  Retain an
            # immutable snapshot even in diagnostic mode so MotionLoader and
            # the later action-ball broker bind the same bytes.  This snapshot
            # proves identity/TOCTOU closure only; it does not mint canonical
            # admission.
            print(
                "[MotionCommand] WARN canonical_ready_mode DIAGNOSTIC "
                "UNAUTHORIZED: registry/certificate admission bypassed; "
                "clip ready-entry contract still enforced",
                flush=True,
            )
            self._validate_canonical_ready_config()
            self._canonical_registry_tables = None
            self._motion_payloads = (
                self._snapshot_diagnostic_motion_bytes()
            )
        elif self.canonical_ready_mode:
            self._validate_canonical_ready_config()
            self._canonical_registry_tables = (
                self._load_and_validate_canonical_registry(env)
            )
            self._motion_payloads = self._snapshot_canonical_motion_bytes()
        else:
            # Default (non-canonical) motion_file channel: pre-branch behavior —
            # MotionLoader reads the raw NPZ paths directly, no code-owned trust
            # set. Admission is scoped to the canonical registry consumer above.
            self._motion_payloads = None
        self.motion = MotionLoader(
            self._motion_files,
            self.body_indexes,
            motion_payloads=self._motion_payloads,
            articulation_body_names=self.robot.body_names,
            selected_body_names=self.cfg.body_names,
            device=self.device,
            allow_legacy_link_origin_velocity=bool(
                self.cfg.allow_legacy_link_origin_velocity
            ),
        )
        if self.canonical_ready_mode:
            if not self._canonical_diagnostic_unauthorized:
                self._validate_canonical_registry_motion_bytes()
            self._validate_canonical_ready_clips()
        self._configure_action_ball_dynamic_ready()
        expected_fps = 1.0 / float(env.step_dt)
        if not math.isfinite(expected_fps) or not math.isclose(
            self.motion.fps, expected_fps, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(
                "motion fps must equal the policy rate exactly enough for one-frame-per-step "
                f"playback: clips={list(self.motion.per_clip_fps)} policy_hz={expected_fps:.12g}"
            )
        # GROUNDING preflight (2026-07-03): the actor obs consumes the RAW clip-world anchor quat,
        # and the racket-target boxes are planned in the +X-grounded frame — a clip that was never
        # re-grounded (frame-0 anchor yaw far from 0, e.g. registry v4 at ~+84 deg) trains a
        # TURN-AND-WALK policy whose footwork is undeployable without real base localization
        # (the 2026-07-03 model_9000 backward-jump lesson). Warn loudly; do not silently train.
        for _c in range(self.motion.num_segments):
            _q0 = self.motion.body_quat_w[int(self.motion.seg_start[_c]), self.motion_anchor_body_index]
            _w, _x, _y, _z = (float(_q0[0]), float(_q0[1]), float(_q0[2]), float(_q0[3]))
            _yaw0 = math.degrees(math.atan2(2.0 * (_w * _z + _x * _y), 1.0 - 2.0 * (_y * _y + _z * _z)))
            if abs(_yaw0) > 10.0:
                print(
                    f"[MotionCommand WARN] clip {_c} frame-0 anchor yaw = {_yaw0:+.1f} deg — this clip "
                    "was NOT re-grounded to +X (scripts/reground_hope_frame.py). Target boxes assume "
                    "+X grounding; training on it produces a turn-and-walk policy that needs "
                    "oracle/mocap localization at deploy. Pin registry_name to the re-grounded "
                    "lineage (hopex/v3) or re-ground and re-upload before training.",
                    flush=True,
                )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # R14 retiming: float shadow clock + per-env playback speed. Inactive at the default
        # (1.0, 1.0), keeping the integer-clock path byte-identical; when active, time_steps is
        # derived as round(time_steps_f) (matching the deploy clock's round() in time_step_for).
        _s_rng = tuple(float(x) for x in self.cfg.speed_scale_range)
        if len(_s_rng) != 2 or not (0.0 < _s_rng[0] <= _s_rng[1]):
            raise ValueError(f"speed_scale_range must be (lo, hi) with 0 < lo <= hi, got {self.cfg.speed_scale_range}")
        _s_lo, _s_hi = _s_rng
        self.retiming_active = not (_s_lo == 1.0 and _s_hi == 1.0)
        # FIXED per-clip playback speed (backhand-fix ablation 2026-07-08): e.g. (1.0, 0.8) plays
        # the backhand reference at 0.8x while the forehand stays 1.0x. Deterministic per clip
        # (no per-swing randomness), rides the same R14 float-clock path. Overrides
        # speed_scale_range sampling when set; None (default) = byte-identical legacy behavior.
        self._speed_per_clip = None
        if getattr(self.cfg, "speed_scale_per_clip", None) is not None:
            _spc = tuple(float(x) for x in self.cfg.speed_scale_per_clip)
            if any(s <= 0.0 for s in _spc):
                raise ValueError(f"speed_scale_per_clip must be positive, got {_spc}")
            if len(_spc) != self.motion.num_segments:
                raise ValueError(
                    f"speed_scale_per_clip has {len(_spc)} entries but the motion has "
                    f"{self.motion.num_segments} clip(s)")
            self._speed_per_clip = torch.tensor(_spc, device=self.device)
            self.retiming_active = True
            print(f"[MotionCommand] speed_scale_per_clip ACTIVE: {_spc} "
                  f"(fixed per-clip reference playback; overrides speed_scale_range)", flush=True)
        self.time_steps_f = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.speed_scale = torch.ones(self.num_envs, device=self.device)
        # Same-ball planner revisions (default OFF).  This is deliberately a separate clock from
        # the historical R14 random/fixed retimer: a revised deadline changes the phase rate of the
        # *same* physical ball task, never the clip, bank row, reward truth, or simulator state.
        # The RacketTargetCommand installs an immutable task identity and then submits one atomic
        # target/TTS revision per policy step.  This command consumes the accepted TTS on the next
        # step and advances a monotonic, acceleration-bounded reference phase.
        self.planner_revision_enabled = bool(
            getattr(self.cfg, "planner_revision_enabled", False)
        )
        self._planner_revision_profile: PhaseGovernorProfile | None = None
        self._planner_initial_tts_mixture: InitialTtsMixture | None = None
        if self.planner_revision_enabled:
            raw_profile = getattr(self.cfg, "planner_revision_profile", None)
            if not isinstance(raw_profile, dict):
                raise ValueError(
                    "planner_revision_enabled requires a complete planner_revision_profile mapping"
                )
            self._planner_revision_profile = PhaseGovernorProfile.from_mapping(raw_profile)
            initial_tts = tuple(
                float(value)
                for value in getattr(
                    self.cfg, "planner_revision_initial_tts_range_s", ()
                )
            )
            if (
                len(initial_tts) != 2
                or not math.isfinite(initial_tts[0])
                or not math.isfinite(initial_tts[1])
                or not (
                    self._planner_revision_profile.min_tts_s
                    <= initial_tts[0]
                    < initial_tts[1]
                    <= self._planner_revision_profile.max_tts_s
                )
            ):
                raise ValueError(
                    "planner_revision_initial_tts_range_s must be a non-degenerate ordered "
                    "finite pair inside "
                    "the complete profile TTS envelope"
                )
            raw_mixture = getattr(
                self.cfg, "planner_revision_initial_tts_mixture", None
            )
            if not isinstance(raw_mixture, dict):
                raise ValueError(
                    "planner_revision_enabled requires a complete "
                    "planner_revision_initial_tts_mixture mapping"
                )
            self._planner_initial_tts_mixture = InitialTtsMixture.from_mapping(
                raw_mixture
            )
            self._planner_initial_tts_mixture.validate_support(
                lo_s=initial_tts[0], hi_s=initial_tts[1]
            )
            if not math.isclose(
                self._planner_revision_profile.policy_dt_s,
                float(env.step_dt),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "planner revision profile policy_dt_s must equal the runtime policy step: "
                    f"profile={self._planner_revision_profile.policy_dt_s} runtime={env.step_dt}"
                )
            if self._speed_per_clip is not None or _s_rng != (1.0, 1.0):
                raise ValueError(
                    "planner revision phase governor is incompatible with R14 speed_scale_range/"
                    "speed_scale_per_clip; it owns the sole reference clock"
                )
            if str(getattr(self.cfg, "event_timing_mode", EVENT_TIMING_MODE_DISABLED)) \
                    != EVENT_TIMING_MODE_DISABLED:
                raise ValueError(
                    "planner revision phase governor is incompatible with event_timing_mode; "
                    "one task may have only one deadline owner"
                )
            if (
                tuple(int(value) for value in self.cfg.hold_steps_range) != (0, 0)
                or int(self.cfg.stand_start_min_hold) != 0
                or int(self.cfg.post_swing_min_hold) != 0
            ):
                raise ValueError(
                    "planner revision initial_tts_range_s owns preparation time; legacy random/min "
                    "hold clocks must all be zero"
                )
            # Force velocity scaling through the existing, audited retiming lane.  Unlike R14,
            # speed_scale is recomputed from the *actual* phase delta each step below.
            self.retiming_active = True
            n = self.num_envs
            self._planner_active = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._planner_control_epoch = torch.zeros(n, dtype=torch.long, device=self.device)
            self._planner_task_id = torch.zeros(n, dtype=torch.long, device=self.device)
            self._planner_task_revision = torch.full(
                (n,), -1, dtype=torch.long, device=self.device
            )
            self._planner_start_step = torch.zeros(n, device=self.device)
            self._planner_strike_step = torch.zeros(n, device=self.device)
            self._planner_phase_rate = torch.zeros(n, device=self.device)
            self._planner_slow_only_next = torch.zeros(
                n, dtype=torch.bool, device=self.device
            )
            self._planner_desired_tts = torch.zeros(n, device=self.device)
            self._planner_begin_tts = torch.zeros(n, device=self.device)
            self._planner_truth_tts = torch.zeros(n, device=self.device)
            # 带符号孪生时钟(2026-07-25):truth tts 是任务期限语义,触球后 clamp 钉 0
            # (obs/critic 读它,合同如此)。但 |tts|<=0.12 的击球窗掩码若也读它,窗就从
            # 触球一直开到 clip 收尾——随挥全程 ~50-100 步顶着 ±0.12 s 的设计语义,
            # position/normal 触球后停拍可薅、站稳包/face 税全程计费、模仿被 0.25x 捂嘴。
            # 窗掩码改读这条不截断的时钟:触球后照常转负,窗在 +0.12 s 如约关闭。
            # 非 active(reset 后新任务未装)置大正哨兵 = 窗关闭(fail-closed;
            # 旧行为是残留 0 → 空档期窗误开)。
            self._planner_truth_tts_signed = torch.full(
                (n,), 1.0e6, device=self.device
            )
            # Immutable task-begin envelope baseline.  A latest-value transport may legitimately
            # skip active revisions, so envelope checks may not depend on whichever revision the
            # consumer happened to observe previously.
            self._planner_begin_target_pos = torch.zeros(n, 3, device=self.device)
            self._planner_begin_target_vel = torch.zeros(n, 3, device=self.device)
            self._planner_begin_target_normal = torch.zeros(n, 3, device=self.device)
            self._planner_begin_target_normal[:, 0] = 1.0
            self.metrics["planner_revision_accepted"] = torch.zeros(n, device=self.device)
            self.metrics["planner_revision_rejected"] = torch.zeros(n, device=self.device)
            self.metrics["planner_phase_rate_per_s"] = torch.zeros(n, device=self.device)
            self.metrics["planner_truth_tts_s"] = torch.zeros(n, device=self.device)
        # Unified multi-clip (HITTER forehand+backhand) support. With one clip these are inert and the
        # behaviour below is byte-identical to the single-clip path. clip_id[env] selects which segment
        # (swing type) the env is currently imitating.
        self._multiseg = self.motion.num_segments > 1
        self.clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Action-conditioned ball-first is bound later by RacketTargetCommand, after both command
        # terms have been constructed from one admitted manifest.  Keeping every field ``None``
        # until that one-shot bind preserves the legacy reset path exactly, including its random
        # draws.  The broker owns immutable birth receipts; MotionCommand remains the sole owner of
        # articulation reset writes.
        self._action_ball_birth_broker = None
        self._action_ball_runtime_module_bound = None
        self._action_ball_trusted_repo_root = None
        self._action_ball_motion_admission_receipt_sha256 = None
        # Racket owns the solved per-swing task graph.  Motion receives only two public,
        # read-only callables: one frozen receipt accessor and one digest over Racket's complete
        # exact-resume payload.  No sampler/curriculum/pool object is retained here.
        self._action_ball_task_ref_for_env = None
        self._action_ball_task_receipt_resolver = None
        self._action_ball_shared_state_sha256_accessor = None
        self._action_ball_expected_shared_racket_state_sha256 = None
        self._action_ball_action_uids = None
        self._action_ball_motion_sha256 = None
        self._action_ball_segment_lengths = None
        self._action_ball_ready_root_z = None
        self._action_ball_ready_root_quat = None
        self._action_ball_reset_generation = None
        self._action_ball_swing_generation = None
        self._action_ball_birth_receipt_sha256 = None
        self._action_ball_seen_birth_receipts = None
        self._action_ball_active_task_refs = None
        self._action_ball_task_timing_active = None
        self._action_ball_diagnostic_pending_row_count = None
        self._action_ball_task_pending_elapsed_s = None
        self._action_ball_task_age_s = None
        self._action_ball_time_to_contact_s = None
        self._action_ball_teacher_rate = None
        self._action_ball_scaled_t_hit_s = None
        self._action_ball_scaled_t_cycle_s = None
        self._action_ball_pre_swing_wait_s = None
        balanced_clip_sampling = getattr(self.cfg, "balanced_clip_sampling", False)
        if type(balanced_clip_sampling) is not bool:
            raise ValueError("balanced_clip_sampling must be an exact boolean")
        self._balanced_clip_sampler: _BalancedRoundRobinClipSampler | None = None
        if balanced_clip_sampling:
            self._balanced_clip_sampler = _BalancedRoundRobinClipSampler(
                num_segments=int(self.motion.num_segments),
                seed=getattr(self.cfg, "balanced_clip_sampling_seed", 0),
                clip_order=self._motion_files,
                device=self.device,
            )
            print(
                "[MotionCommand] balanced_clip_sampling ACTIVE: "
                f"clips={self.motion.num_segments} "
                f"seed={self._balanced_clip_sampler.seed} "
                "(seeded round-robin clip allocation; exact count spread <= 1)",
                flush=True,
            )
        # 每 clip 的 forehand/backhand 家族表(spdmix v2 硬绑定一)。显式配置在这里整表校验
        # (boot fail-loud:长度==clip 数、值合法、正反手至少各一)并落成张量;None(默认,现役
        # 所有在跑臂)= 不建表、不打印、行为逐字节不变——查表方(clip_family_is_forehand)在第一次
        # 用到时按"单 clip 正手 / 恰 2 clip = (正手, 反手)"懒推导,>2 clip 缺表当场报错。
        self._clip_family_is_forehand_t: torch.Tensor | None = None
        if getattr(self.cfg, "clip_family_per_clip", None) is not None:
            self._clip_family_is_forehand_t = torch.tensor(
                resolve_clip_family_is_forehand(
                    self.cfg.clip_family_per_clip, int(self.motion.num_segments)
                ),
                dtype=torch.bool,
                device=self.device,
            )
            print(
                "[MotionCommand] clip_family_per_clip ACTIVE: "
                f"{tuple(str(value) for value in self.cfg.clip_family_per_clip)} "
                "(per-clip forehand/backhand lookup replaces the clips==0 hardcode)",
                flush=True,
            )
        # Robust per-step "this env just wrapped to a new swing" signal, consumed by the racket-target
        # command to resample its target. Replaces a time_steps<prev heuristic that fails when a clip
        # wrap jumps the index to a HIGHER segment start (forehand->backhand on the concatenated axis).
        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Pre-swing hold state (see cfg.hold_steps_range): while hold_counter > 0 the reference
        # clock is frozen at the swing's first frame ("waiting for the ball"). _update_command
        # decrements it. in_hold is exposed for rewards/metrics.
        self.hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # True only while _resample_command is being invoked from an intra-episode clip WRAP
        # (as opposed to a true episode reset) — wraps skip the RSI teleport (cfg.wrap_teleport).
        self._resampling_from_wrap = False
        # --- stagger_initial_clock (metric-sync fix 2026-07-09; default OFF = byte-identical) ------
        # Disease: 4096 envs constructed/resumed at the SAME instant + a low fall rate => they all
        # time out together, swing together, and reset together (episode_length sawtooth 52->485,
        # mass timeouts) — every EMA metric (fall rates, completion, return rates) then reads a
        # synchronized-queue oscillation instead of a steady rate. Cure, one flag, two one-shot
        # biases: (a) each env's FIRST true reset adds U[0, stagger_hold_max_steps] extra hold, so
        # the cohort's swing/strike phases spread within the first episode; (b) the first
        # _update_command after construction adds U[0, max_episode_length) to every env's episode
        # clock, so the FIRST timeouts — and every episode boundary after them — spread instead of
        # firing in one wave. 人话:开了它,4096 个 env 的"到点超时+挥拍节拍"被随机错开,EMA 指标
        # 不再集体振荡;默认关,现役跑法完全不受影响。
        self._stagger_hold_pending: torch.Tensor | None = None
        self._stagger_ep_pending = False
        if bool(getattr(self.cfg, "stagger_initial_clock", False)):
            self._stagger_hold_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._stagger_ep_pending = True
        # T1 continuous timing is deliberately a separate, fail-closed command path.  It reuses
        # native clip playback, exact pre-swing hold and no-wrap carry-state, but none of the
        # random wrap/switch/retiming mechanisms.  The schedule bytes are verified before any
        # event state exists; RacketTargetCommand later binds every immutable row to the loaded
        # train bank and supplies the native strike offset for each clip.
        self._event_timing_mode = str(
            getattr(self.cfg, "event_timing_mode", EVENT_TIMING_MODE_DISABLED)
        )
        if self._event_timing_mode not in EVENT_TIMING_MODES:
            raise ValueError(
                f"event_timing_mode must be one of {EVENT_TIMING_MODES}, "
                f"got {self._event_timing_mode!r}"
            )
        self._event_schedule = None
        self._event_scheduler: EventTimingScheduler | None = None
        self._event_native_strike_ticks: torch.Tensor | None = None
        if self._event_timing_mode == EVENT_TIMING_MODE_POST_STRIKE_T1:
            schedule_path = str(getattr(self.cfg, "event_timing_schedule", "") or "").strip()
            schedule_sha = str(
                getattr(self.cfg, "event_timing_schedule_sha256", "") or ""
            ).strip()
            if not schedule_path or not schedule_sha:
                raise ValueError(
                    "post_strike_t1 requires event_timing_schedule and its exact byte SHA-256"
                )
            if bool(getattr(self.cfg, "event_timing_repeat", False)):
                raise ValueError(
                    "post_strike_t1 rows may not repeat within an episode; materialize enough "
                    "immutable rows and reset only at the sequence boundary"
                )
            if bool(self.cfg.wrap_teleport):
                raise ValueError("post_strike_t1 requires wrap_teleport=false (carry state)")
            if float(self.cfg.clip_switch_prob) != 0.0:
                raise ValueError("post_strike_t1 requires clip_switch_prob=0")
            if bool(self.cfg.stagger_initial_clock):
                raise ValueError("post_strike_t1 requires stagger_initial_clock=false")
            if self.retiming_active:
                raise ValueError("post_strike_t1 requires native one-frame-per-step playback")
            if int(getattr(self.cfg, "rsi_skip_settle_frames", 0)) != 0:
                raise ValueError(
                    "post_strike_t1 event installs require rsi_skip_settle_frames=0; skipping "
                    "native clip frames would change immutable deadline feasibility"
                )
            self._event_schedule = load_event_schedule(schedule_path, schedule_sha)
            actual_rate = 1.0 / float(env.step_dt)
            if not math.isclose(
                actual_rate,
                float(self._event_schedule.policy_rate_hz),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "event schedule policy rate does not match the instantiated control rate: "
                    f"schedule={self._event_schedule.policy_rate_hz} runtime={actual_rate:.12g}"
                )
            bad_clips = sorted(
                {row.clip_id for row in self._event_schedule.rows}
                - set(range(int(self.motion.num_segments)))
            )
            if bad_clips:
                raise ValueError(
                    f"event schedule references unloaded motion clip ids {bad_clips}"
                )
            self._event_scheduler = EventTimingScheduler(
                self._event_schedule,
                num_envs=self.num_envs,
                device=self.device,
            )
        # A8: post-swing initial-state ring buffer (root state stored ORIGIN-RELATIVE in [:3] so a
        # snapshot from env B can seed env A; quats/velocities/joints are origin-invariant).
        # Tensors are allocated lazily at first capture (dof count comes from live robot data).
        self._post_swing_root: torch.Tensor | None = None
        self._post_swing_joint_pos: torch.Tensor | None = None
        self._post_swing_joint_vel: torch.Tensor | None = None
        self._post_swing_count = 0
        self._post_swing_ptr = 0
        self._post_swing_teacher_hard_contract: dict | None = None
        ready = getattr(self.cfg, "post_swing_require_ready_at_init", False)
        fail_fast = getattr(self.cfg, "post_swing_fail_fast_first_reset", False)
        require_readback = getattr(self.cfg, "post_swing_first_reset_require_readback", False)
        if any(type(value) is not bool for value in (ready, fail_fast, require_readback)):
            raise ValueError("post-swing teacher gates require exact booleans")
        self._post_swing_require_ready_at_init = ready
        self._post_swing_fail_fast_first_reset = fail_fast
        self._post_swing_first_reset_require_readback = require_readback
        min_count = getattr(self.cfg, "post_swing_first_reset_min_adopted_count", 1)
        min_fraction = getattr(self.cfg, "post_swing_first_reset_min_adopted_fraction", 0.0)
        tolerance = getattr(self.cfg, "post_swing_first_reset_selection_tolerance", 1.0)
        if type(min_count) is not int or min_count <= 0:
            raise ValueError("post_swing_first_reset_min_adopted_count must be a positive integer")
        if (
            type(min_fraction) is not float
            or not math.isfinite(min_fraction)
            or not 0.0 <= min_fraction <= 1.0
            or type(tolerance) is not float
            or not math.isfinite(tolerance)
            or not 0.0 <= tolerance <= 1.0
        ):
            raise ValueError("post-swing first-reset fractions must be finite JSON-style floats in [0,1]")
        self._post_swing_first_reset_min_adopted_count = min_count
        self._post_swing_first_reset_min_adopted_fraction = min_fraction
        self._post_swing_first_reset_selection_tolerance = tolerance
        if (
            require_readback
            or min_count != 1
            or min_fraction != 0.0
            or tolerance != 1.0
        ) and not fail_fast:
            raise ValueError(
                "post-swing first-reset acceptance thresholds require fail_fast_first_reset=true"
            )
        self._post_swing_first_reset_checked = False
        # The capture producer intentionally lives inside MotionCommand.  There is no reusable
        # writer that accepts caller-supplied arrays and no module-global Python "capability".
        # The only state snapshot is taken from live articulation tensors in the natural-wrap
        # branch below.  Its artifact makes the narrower, auditable claim that exact reviewed
        # source owned an O_EXCL namespace and emitted these bytes; it is not a cryptographic
        # proof that an unmodified Python runtime executed a particular callback.
        self._post_swing_capture_output_dir: Path | None = None
        self._post_swing_capture_target_count = 0
        self._post_swing_capture_motion_clips: list[dict] = []
        self._post_swing_capture_joint_names: list[str] = []
        self._post_swing_capture_producer_source_sha256: str | None = None
        self._post_swing_capture_runtime_hard_contract_sha256: str | None = None
        self._post_swing_capture_claim_sha256: str | None = None
        self._post_swing_capture_claim_fd: int | None = None
        self._post_swing_capture_roots: list[np.ndarray] = []
        self._post_swing_capture_joint_pos: list[np.ndarray] = []
        self._post_swing_capture_joint_vel: list[np.ndarray] = []
        self._post_swing_capture_count = 0
        self._post_swing_capture_complete = False
        # Activation accounting is kept outside ``metrics`` because command metrics are
        # instantaneous per-environment values, while these are event counts accumulated over
        # one PPO update.  MotionOnPolicyRunner consumes and resets them exactly once from its
        # existing per-update logger.  Integer device scalars avoid a host sync on every reset.
        self._post_swing_activation_counters = {
            name: torch.zeros((), dtype=torch.long, device=self.device)
            for name in (
                "post_swing_replay_buffer_not_ready_reset_count",
                "post_swing_replay_eligible_reset_count",
                "post_swing_replay_random_not_selected_reset_count",
                "post_swing_replay_selected_reset_count",
                "post_swing_replay_started_reset_count",
            )
        }
        self._load_post_swing_teacher_if_configured()
        self._configure_post_swing_capture_if_requested()
        # Reward-mechanism activation accounting.  These counters live on the motion command so
        # every imitation reward term can record into one per-update ledger without touching the
        # simulator or sampling another random number.  The unit of V1 is one environment sample
        # evaluated by the body-linear-velocity imitation term.  The unit of V2 is one
        # (imitation reward term, environment) sample inside the wide strike window; V2 therefore
        # counts every real scaled reward application rather than inferring activation from an
        # aggregate reward value.
        self._reward_activation_counters = {
            name: torch.zeros((), dtype=torch.long, device=self.device)
            for name in (
                "v1_velocity_mimic_eligible_sample_count",
                "v1_held_wrist_excluded_sample_count",
                "v2_strike_window_eligible_imitation_sample_count",
                "v2_quarter_scaled_strike_window_imitation_sample_count",
            )
        }
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["in_hold"] = torch.zeros(self.num_envs, device=self.device)
        if self._event_scheduler is not None:
            self.metrics["event_timing_armed"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_question_installed"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_question_unavailable"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_question_infeasible"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_deadline_due"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["event_opportunities_consumed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos_mean_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos_max_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel_mean_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel_max_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y", "z"):
            self.metrics[f"reference_anchor_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"robot_anchor_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"reference_anchor_lin_vel_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"robot_anchor_lin_vel_{axis}"] = torch.zeros(self.num_envs, device=self.device)

    @staticmethod
    def _range_is_exact_zero_pair(value) -> bool:
        try:
            pair = tuple(value)
        except TypeError:
            return False
        return len(pair) == 2 and all(
            type(item) in (int, float) and math.isfinite(float(item)) and float(item) == 0.0
            for item in pair
        )

    @staticmethod
    def _mapping_ranges_are_exact_zero(mapping) -> bool:
        return isinstance(mapping, dict) and all(
            MotionCommand._range_is_exact_zero_pair(value) for value in mapping.values()
        )

    @staticmethod
    def _canonical_registry_module():
        """Load the repository registry only for the explicitly enabled formal path."""

        import importlib.util
        import sys

        global _CANONICAL_REGISTRY_RUNTIME_MODULE
        module_name = "_hope_canonical_motion_registry_runtime"
        script = (
            Path(__file__).resolve().parents[6]
            / "scripts"
            / "canonical_motion_registry.py"
        )
        if _CANONICAL_REGISTRY_RUNTIME_MODULE is not None:
            if (
                Path(_CANONICAL_REGISTRY_RUNTIME_MODULE.__file__).resolve()
                != script
            ):
                raise ValueError(
                    "cached canonical registry module resolved to a different file"
                )
            return _CANONICAL_REGISTRY_RUNTIME_MODULE
        if not script.is_file():
            raise ValueError(f"canonical registry loader is missing: {script}")
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot create canonical registry loader spec for {script}")
        module = importlib.util.module_from_spec(spec)
        # Do not reuse a caller-preloaded sys.modules object, even when it
        # spoofs __file__.  The first trusted load in this process always
        # executes the exact repository bytes through the standard file loader.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        if Path(module.__file__).resolve() != script:
            sys.modules.pop(module_name, None)
            raise ValueError("canonical registry executed from a wrong file")
        _CANONICAL_REGISTRY_RUNTIME_MODULE = module
        return module

    @staticmethod
    def _exact_config_sha256(value, label: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(
                f"{label} must be one exact 64-character lowercase SHA-256"
            )
        return value

    @staticmethod
    def _configured_motion_files(value) -> tuple[str, ...]:
        if isinstance(value, str):
            files = (value,)
        else:
            try:
                files = tuple(value)
            except TypeError as exc:
                raise ValueError(
                    "canonical motion_file must be one path or an ordered path sequence"
                ) from exc
        if not files or any(type(path) is not str or not path for path in files):
            raise ValueError(
                "canonical motion_file entries must be non-empty path strings"
            )
        try:
            return tuple(str(Path(path).expanduser().resolve(strict=True)) for path in files)
        except OSError as exc:
            raise ValueError(f"cannot resolve canonical motion_file: {exc}") from exc

    @staticmethod
    def _exact_numeric_tuple(value, label: str) -> tuple[float, ...]:
        try:
            raw = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{label} must be an explicit ordered sequence") from exc
        if any(
            isinstance(item, bool)
            or type(item) not in (int, float)
            or not math.isfinite(float(item))
            for item in raw
        ):
            raise ValueError(f"{label} must contain only finite real numbers")
        return tuple(float(item) for item in raw)

    def _load_and_validate_canonical_registry(self, env):
        """Bind all five runtime columns to one pinned, training-authorized registry."""

        registry_path = getattr(self.cfg, "canonical_registry_path", "")
        if type(registry_path) is not str or not registry_path.strip():
            raise ValueError(
                "canonical_ready_mode requires canonical_registry_path"
            )
        expected_registry = self._exact_config_sha256(
            getattr(self.cfg, "canonical_registry_sha256", ""),
            "canonical_registry_sha256",
        )
        expected_alignment = self._exact_config_sha256(
            getattr(self.cfg, "canonical_registry_alignment_sha256", ""),
            "canonical_registry_alignment_sha256",
        )
        expected_ready = self._exact_config_sha256(
            getattr(self.cfg, "canonical_ready_sha256", ""),
            "canonical_ready_sha256",
        )
        expected_ready_fk = self._exact_config_sha256(
            getattr(self.cfg, "canonical_ready_fk_sha256", ""),
            "canonical_ready_fk_sha256",
        )
        promotion_certificate_path = getattr(
            self.cfg, "canonical_promotion_certificate_path", ""
        )
        if (
            type(promotion_certificate_path) is not str
            or not promotion_certificate_path.strip()
        ):
            raise ValueError(
                "canonical_ready_mode requires "
                "canonical_promotion_certificate_path"
            )
        repo_root_value = getattr(self.cfg, "canonical_registry_repo_root", "")
        if type(repo_root_value) is not str:
            raise ValueError("canonical_registry_repo_root must be a path string")
        repo_root = repo_root_value.strip() or None

        registry_module = self._canonical_registry_module()
        try:
            registry = registry_module.load_canonical_motion_bank_registry(
                registry_path,
                repo_root=repo_root,
                expected_registry_sha256=expected_registry,
            )
            admission = registry_module.verify_registry_promotion_certificate(
                registry,
                promotion_certificate_path,
                authorization_purpose="training",
            )
            promotion_binding = registry_module.bank_promotion_binding(
                registry,
                authorization_purpose="training",
            )
            registry_module.motion_admission.require_matching_admission(
                admission, promotion_binding
            )
            tables = registry_module.adapt_registry_for_runtime(
                registry,
                expected_alignment_sha256=expected_alignment,
                expected_canonical_ready_sha256=expected_ready,
                expected_canonical_ready_fk_sha256=expected_ready_fk,
                authorization_purpose="training",
                admission=admission,
            )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ValueError(f"invalid canonical motion registry: {exc}") from exc

        if tables.canonical_ready_sha256 != expected_ready:
            raise ValueError(
                "canonical_ready_sha256 differs from the pinned registry ready: "
                f"config={expected_ready} registry={tables.canonical_ready_sha256}"
            )
        if tables.canonical_ready_fk_sha256 != expected_ready_fk:
            raise ValueError(
                "canonical_ready_fk_sha256 differs from the pinned registry FK truth: "
                f"config={expected_ready_fk} "
                f"registry={tables.canonical_ready_fk_sha256}"
            )
        actual_files = self._motion_files
        if actual_files != tuple(tables.motion_file):
            raise ValueError(
                "canonical motion_file order differs from registry motion_ids: "
                f"ids={tables.motion_ids} config={actual_files} "
                f"registry={tables.motion_file}"
            )
        families = getattr(self.cfg, "clip_family_per_clip", None)
        if (
            not isinstance(families, (tuple, list))
            or tuple(families) != tuple(tables.clip_family_per_clip)
        ):
            raise ValueError(
                "canonical clip_family_per_clip must exactly equal the registry table"
            )

        commands_cfg = getattr(getattr(env, "cfg", None), "commands", None)
        racket_cfg = getattr(commands_cfg, "racket_target", None)
        if racket_cfg is None:
            raise ValueError(
                "canonical_ready_mode requires env.cfg.commands.racket_target "
                "for an atomic phase/face table check"
            )
        phases = self._exact_numeric_tuple(
            getattr(racket_cfg, "strike_phase_per_clip", ()),
            "racket_target.strike_phase_per_clip",
        )
        if phases != tuple(tables.strike_phase_per_clip):
            raise ValueError(
                "racket_target.strike_phase_per_clip differs from the registry table"
            )
        signs = self._exact_numeric_tuple(
            getattr(racket_cfg, "mount_normal_sign_per_clip", ()),
            "racket_target.mount_normal_sign_per_clip",
        )
        if signs != tuple(tables.mount_normal_sign_per_clip):
            raise ValueError(
                "racket_target.mount_normal_sign_per_clip differs from the registry table"
            )

        self.canonical_motion_ids = tuple(tables.motion_ids)
        self.canonical_registry_sha256 = tables.registry_sha256
        self.canonical_registry_alignment_sha256 = tables.alignment_sha256
        self.canonical_ready_sha256 = tables.canonical_ready_sha256
        self.canonical_ready_fk_sha256 = tables.canonical_ready_fk_sha256
        self.canonical_contact_opportunity_frames = tuple(
            tables.contact_opportunity_frames_per_clip
        )
        self.canonical_source_manifest_sha256_per_clip = tuple(
            tables.source_manifest_sha256_per_clip
        )
        self.canonical_build_manifest_sha256_per_clip = tuple(
            tables.build_manifest_sha256_per_clip
        )
        self.canonical_applicability_manifest_sha256_per_clip = tuple(
            tables.applicability_manifest_sha256_per_clip
        )
        self.canonical_evidence_level_per_clip = tuple(
            tables.evidence_level_per_clip
        )
        self.canonical_evidence_manifest_sha256_per_clip = tuple(
            tables.evidence_manifest_sha256_per_clip
        )
        self.canonical_question_bank_sha256_per_clip = tuple(
            tables.question_bank_sha256_per_clip
        )
        self.canonical_training_config_sha256_per_clip = tuple(
            tables.training_config_sha256_per_clip
        )
        self.canonical_onnx_model_sha256_per_clip = tuple(
            tables.onnx_model_sha256_per_clip
        )
        self.canonical_adoption_manifest_sha256_per_clip = tuple(
            tables.adoption_manifest_sha256_per_clip
        )
        # Keep the actual opaque capability and the exact object it authorizes.  The action-ball
        # manifest may repeat these hashes for identity, but it can never mint or replace this
        # code-rooted training admission.
        self._canonical_motion_registry = registry
        self._canonical_motion_admission = admission
        self._canonical_motion_promotion_binding = promotion_binding
        self._canonical_motion_registry_module = registry_module
        return tables

    def _snapshot_canonical_motion_bytes(self) -> tuple[bytes, ...]:
        """Bind MotionLoader to the exact registry-authorized NPZ bytes."""

        payloads: list[bytes] = []
        digests: list[str] = []
        for index, path in enumerate(self._motion_files):
            try:
                payload = Path(path).read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"cannot snapshot canonical motion_file[{index}]: {exc}"
                ) from exc
            payloads.append(payload)
            digests.append(hashlib.sha256(payload).hexdigest())
        expected = tuple(self._canonical_registry_tables.npz_sha256_per_clip)
        if tuple(digests) != expected:
            raise ValueError(
                "canonical motion bytes changed after trusted registry admission"
            )
        return tuple(payloads)

    def _snapshot_diagnostic_motion_bytes(self) -> tuple[bytes, ...]:
        """Bind an unauthorized diagnostic to the exact bytes its loader adopts."""

        payloads: list[bytes] = []
        digests: list[str] = []
        for index, path in enumerate(self._motion_files):
            try:
                payload = Path(path).read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"cannot snapshot diagnostic motion_file[{index}]: {exc}"
                ) from exc
            payloads.append(payload)
            digests.append(hashlib.sha256(payload).hexdigest())
        if tuple(digests) != tuple(self._motion_file_sha256):
            raise ValueError(
                "diagnostic motion bytes changed between initial hashing and "
                "MotionLoader adoption"
            )
        return tuple(payloads)

    def _validate_canonical_registry_motion_bytes(self) -> None:
        """Check schema and the immutable snapshots used by MotionLoader."""

        if not self.motion.kinematics_contract_exact:
            raise ValueError(
                "canonical_ready_mode requires every clip to use exact schema-2 kinematics "
                "with the runtime body order bound"
            )
        tables = self._canonical_registry_tables
        if int(self.motion.num_segments) != len(tables.motion_ids):
            raise ValueError(
                "canonical MotionLoader segment count differs from the five registry rows"
            )
        actual_hashes = tuple(
            hashlib.sha256(payload).hexdigest()
            for payload in self._motion_payloads
        )
        if actual_hashes != tuple(tables.npz_sha256_per_clip):
            raise ValueError(
                "canonical motion bytes changed between registry validation and MotionLoader adoption"
            )

    def _validate_canonical_ready_config(self) -> None:
        """Reject reset curricula that would silently bypass the formal ready contract."""

        conflicts: list[str] = []
        if float(self.cfg.stand_start_prob) != 1.0:
            conflicts.append("stand_start_prob must be 1.0")
        if float(self.cfg.post_swing_start_prob) != 0.0:
            conflicts.append("post_swing_start_prob must be 0.0")
        if str(getattr(self.cfg, "post_swing_teacher_receipt", "") or "").strip():
            conflicts.append("post_swing_teacher_receipt must be empty")
        if any(
            bool(getattr(self.cfg, name, False))
            for name in (
                "post_swing_require_ready_at_init",
                "post_swing_fail_fast_first_reset",
                "post_swing_first_reset_require_readback",
            )
        ):
            conflicts.append("post-swing first-reset/replay gates must be disabled")
        if bool(self.cfg.wrap_teleport):
            conflicts.append("wrap_teleport must be false")
        if float(self.cfg.clip_switch_prob) != 0.0:
            conflicts.append("clip_switch_prob must be 0 (switch only at shared-ready wrap)")
        if (
            str(getattr(self.cfg, "event_timing_mode", EVENT_TIMING_MODE_DISABLED))
            != EVENT_TIMING_MODE_DISABLED
        ):
            conflicts.append(
                "event_timing_mode must be disabled (no mid-stroke ready-reference jump)"
            )
        if int(getattr(self.cfg, "rsi_skip_settle_frames", 0)) != 0:
            conflicts.append("rsi_skip_settle_frames must be 0")
        if not self._range_is_exact_zero_pair(self.cfg.joint_position_range):
            conflicts.append("joint_position_range must be (0, 0)")
        if not self._range_is_exact_zero_pair(self.cfg.stand_start_yaw_range):
            conflicts.append("stand_start_yaw_range must be (0, 0)")
        if not self._mapping_ranges_are_exact_zero(self.cfg.pose_range):
            conflicts.append("all pose_range entries must be (0, 0)")
        if not self._mapping_ranges_are_exact_zero(self.cfg.velocity_range):
            conflicts.append("all velocity_range entries must be (0, 0)")
        if conflicts:
            raise ValueError(
                "canonical_ready_mode is the formal all-true-reset ready-entry path and is "
                "incompatible with RSI/post-swing/noised reset curricula: "
                + "; ".join(conflicts)
            )

    @staticmethod
    def _first_tensor_mismatch(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[int, float]:
        """Return mismatch count/max error for an exact runtime-float32 comparison."""

        unequal = reference != candidate
        count = int(torch.count_nonzero(unequal).item())
        if count == 0:
            return 0, 0.0
        ref64 = reference.to(dtype=torch.float64)
        cand64 = candidate.to(dtype=torch.float64)
        max_abs = float(torch.max(torch.abs(ref64 - cand64)).item())
        return count, max_abs

    def _validate_canonical_ready_clips(self) -> None:
        """Require one literal runtime-ready pose at each clip's own boundaries.

        ``MotionLoader`` intentionally converts references to the consumer's float32 dtype.
        The gate is exact in that runtime dtype (no hidden tolerance): every clip must start
        and end on one identical joint/body pose, including the same quaternion hemisphere;
        all six endpoint velocity channels must be literal zero; every boundary value
        (including the ready root Z) must be finite; and the frame-0 root quaternion must be
        unit length (1e-6, the hope_commands convention).  Yaw-only roots are deliberately
        NOT required: real compiled ready stances carry roll/pitch (fivebind shared ready
        pitch ~-11.2 deg, ChingMu73 ~+8..12 deg measured 2026-07-28), and the per-slot
        action-ball birth frame is the yaw PROJECTION of this root, not the root itself.

        Scope is deliberately PER CLIP (coordinator ruling, 2026-07-28): the runtime's
        per-slot ready machinery (``_action_ball_ready_yaw/quat/z`` captured from each
        clip's own frame 0, B_yaw ball offsets anchored to that per-slot yaw, per-slot
        ready-Z contract in the profile adapter) is the design; aim-rotated canonical
        clips legitimately differ across clips in world orientation.  The former
        cross-clip clause ("all clip starts/ends share one exact world-frame ready
        pose") was a leftover of the single-shared-ready ideal, contradicted that
        per-slot machinery, and is deliberately removed.  Raw capture segments
        (ChingMu73-style units) are still rejected by the per-clip clauses: their own
        endpoints match neither in pose nor in velocity.
        """

        runtime_joint_count = int(self.robot.data.default_joint_pos.shape[-1])
        motion_joint_count = int(self.motion.joint_pos.shape[-1])
        if (
            runtime_joint_count != _A3_CANONICAL_READY_JOINT_COUNT
            or motion_joint_count != _A3_CANONICAL_READY_JOINT_COUNT
        ):
            raise ValueError(
                "canonical_ready_mode is bound to the Agibot A3 31-joint articulation: "
                f"runtime={runtime_joint_count}, motion={motion_joint_count}"
            )
        if int(self.body_indexes[0].item()) != 0:
            raise ValueError(
                "canonical_ready_mode requires body_names[0] to be the articulation root body "
                "so one clip frame can atomically seed root and joint state"
            )

        starts = self.motion.seg_start
        ends = starts + self.motion.seg_len - 1
        pose_channels = (
            ("joint_pos", self.motion.joint_pos),
            ("body_pos_w", self.motion._body_pos_w),
            ("body_quat_w", self.motion._body_quat_w),
        )
        velocity_channels = (
            ("joint_vel", self.motion.joint_vel),
            ("body_lin_vel_w", self.motion._body_lin_vel_w),
            ("body_ang_vel_w", self.motion._body_ang_vel_w),
        )
        for channel_name, channel in (*pose_channels, *velocity_channels):
            endpoint_values = channel[torch.cat((starts, ends))]
            if not bool(torch.isfinite(endpoint_values).all()):
                raise ValueError(
                    f"canonical_ready_mode found non-finite {channel_name} at a clip boundary"
                )

        for clip_index in range(int(self.motion.num_segments)):
            start_index = int(starts[clip_index].item())
            end_index = int(ends[clip_index].item())
            for channel_name, channel in pose_channels:
                mismatch_count, max_abs = self._first_tensor_mismatch(
                    channel[start_index], channel[end_index]
                )
                if mismatch_count:
                    raise ValueError(
                        "canonical_ready_mode requires each clip to start and end on one "
                        "exact runtime-float32 ready pose: "
                        f"clip={clip_index} channel={channel_name} "
                        f"mismatches={mismatch_count} max_abs={max_abs:.9g}"
                    )
            for boundary_name, boundary_index in (
                ("start", start_index),
                ("end", end_index),
            ):
                for channel_name, channel in velocity_channels:
                    value = channel[boundary_index]
                    if int(torch.count_nonzero(value).item()) != 0:
                        max_abs = float(torch.max(torch.abs(value)).item())
                        raise ValueError(
                            "canonical_ready_mode requires literal zero endpoint velocities: "
                            f"clip={clip_index} boundary={boundary_name} channel={channel_name} "
                            f"max_abs={max_abs:.9g}"
                        )
            root_quat = self.motion._body_quat_w[start_index, 0]
            norm = math.sqrt(
                sum(float(value) ** 2 for value in root_quat.tolist())
            )
            if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-6:
                raise ValueError(
                    "canonical_ready_mode requires a unit frame-0 root quaternion (the "
                    "per-slot birth frame is its yaw projection): "
                    f"clip={clip_index} norm={norm:.9g}"
                )

    @staticmethod
    def _action_ball_dynamic_ready_sha256(value: object) -> str:
        """Hash one in-memory runtime binding without filesystem/path ambiguity."""

        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _action_ball_dynamic_ready_exact_sha256(
        value: object, *, name: str
    ) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be one lowercase SHA-256 digest")
        return value

    @staticmethod
    def _action_ball_dynamic_ready_vector(
        value: object, *, name: str, length: int
    ) -> tuple[float, ...]:
        if not isinstance(value, (tuple, list)) or len(value) != length:
            raise ValueError(f"{name} must contain exactly {length} values")
        parsed: list[float] = []
        for index, raw in enumerate(value):
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
            ):
                raise ValueError(f"{name}[{index}] must be finite numeric")
            parsed.append(float(raw))
        return tuple(parsed)

    @classmethod
    def _validate_action_ball_dynamic_ready_plant_v2(
        cls, value: object, *, name: str
    ) -> None:
        """Validate the schema-v2 plant identity retained in the sealed binding."""

        expected_keys = {
            "joint_names",
            "articulation_joint_names",
            "action_joint_ids",
            "joint_stiffness",
            "joint_damping",
            "joint_effort_limits",
            "joint_velocity_limits",
            "joint_armature",
            "default_joint_pos_rad",
            "action_scale_rad",
            "qdes_joint_pos_limits",
            "physics_step_dt_s",
            "policy_step_dt_s",
            "control_decimation",
            "control_step_action_delay",
        }
        if type(value) is not dict or set(value) != expected_keys:
            raise ValueError(f"{name} must contain the exact schema-v2 fields")
        joint_names = value["joint_names"]
        if (
            not isinstance(joint_names, list)
            or len(joint_names) != _A3_CANONICAL_READY_JOINT_COUNT
            or len(set(joint_names)) != _A3_CANONICAL_READY_JOINT_COUNT
            or any(type(joint) is not str or not joint for joint in joint_names)
            or value["articulation_joint_names"] != joint_names
            or value["action_joint_ids"]
            != list(range(_A3_CANONICAL_READY_JOINT_COUNT))
        ):
            raise ValueError(f"{name} must bind one exact 31-joint action order")

        vectors = {
            key: cls._action_ball_dynamic_ready_vector(
                value[key],
                name=f"{name}.{key}",
                length=_A3_CANONICAL_READY_JOINT_COUNT,
            )
            for key in (
                "joint_stiffness",
                "joint_damping",
                "joint_effort_limits",
                "joint_velocity_limits",
                "joint_armature",
                "default_joint_pos_rad",
                "action_scale_rad",
            )
        }
        if (
            any(item <= 0.0 for item in vectors["joint_stiffness"])
            or any(item < 0.0 for item in vectors["joint_damping"])
            or any(item <= 0.0 for item in vectors["joint_effort_limits"])
            or any(item <= 0.0 for item in vectors["joint_velocity_limits"])
            or any(item < 0.0 for item in vectors["joint_armature"])
            or any(item <= 0.0 for item in vectors["action_scale_rad"])
        ):
            raise ValueError(f"{name} contains an invalid actuator value")

        raw_limits = value["qdes_joint_pos_limits"]
        if (
            not isinstance(raw_limits, list)
            or len(raw_limits) != _A3_CANONICAL_READY_JOINT_COUNT
        ):
            raise ValueError(f"{name}.qdes_joint_pos_limits must have 31 rows")
        limits = tuple(
            cls._action_ball_dynamic_ready_vector(
                row, name=f"{name}.qdes_joint_pos_limits[{index}]", length=2
            )
            for index, row in enumerate(raw_limits)
        )
        if any(lower >= upper for lower, upper in limits):
            raise ValueError(f"{name}.qdes_joint_pos_limits contains an empty row")

        physics_dt = value["physics_step_dt_s"]
        policy_dt = value["policy_step_dt_s"]
        decimation = value["control_decimation"]
        if (
            isinstance(physics_dt, bool)
            or not isinstance(physics_dt, (int, float))
            or not math.isfinite(float(physics_dt))
            or float(physics_dt) <= 0.0
            or isinstance(policy_dt, bool)
            or not isinstance(policy_dt, (int, float))
            or not math.isfinite(float(policy_dt))
            or float(policy_dt) <= 0.0
            or type(decimation) is not int
            or decimation <= 0
            or not math.isclose(
                float(policy_dt),
                float(physics_dt) * decimation,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(f"{name} contains inconsistent control timing")

        delay = value["control_step_action_delay"]
        expected_delay_keys = {
            "schema_version",
            "enabled",
            "semantic_unit",
            "sample_timing",
            "distribution",
            "min_steps",
            "max_steps",
            "shared_across_all_31_joints",
            "history_fill",
        }
        if (
            type(delay) is not dict
            or set(delay) != expected_delay_keys
            or delay["schema_version"] != 1
            or type(delay["enabled"]) is not bool
            or delay["semantic_unit"] != "policy_control_step"
            or delay["sample_timing"] != "once_per_episode_reset"
            or delay["distribution"] != "discrete_uniform_inclusive"
            or type(delay["min_steps"]) is not int
            or type(delay["max_steps"]) is not int
            or delay["min_steps"] < 0
            or delay["max_steps"] < delay["min_steps"]
            or delay["enabled"] != (delay["max_steps"] > 0)
            or delay["shared_across_all_31_joints"] is not True
            or delay["history_fill"]
            != "safe_default_or_action_specific_hold"
        ):
            raise ValueError(f"{name}.control_step_action_delay is invalid")

    def _configure_action_ball_dynamic_ready(self) -> None:
        """Validate and device-materialize the action-specific A3 reset/hold binding.

        The binding is deliberately passed as an already materialized mapping by
        ``train.py``.  Motion owns immutable clip bytes and can therefore close the
        two important identities here: ordered motion SHA-256 values and exact
        runtime-float32 frame-0 physical state.  The associated action term owns
        actor/action/q_des state installation at true reset.
        """

        self._action_ball_dynamic_ready_binding_sha256 = None
        self._action_ball_dynamic_ready_action_order = None
        self._action_ball_dynamic_ready_physical_root_pos_w_m = None
        self._action_ball_dynamic_ready_physical_root_quat_wxyz = None
        self._action_ball_dynamic_ready_physical_joint_pos_rad = None
        self._action_ball_dynamic_ready_physical_joint_vel_radps = None
        self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad = None
        self._action_ball_dynamic_ready_normalized_actor_action = None
        self._action_ball_dynamic_ready_action_term = None

        binding = getattr(self.cfg, "action_ball_dynamic_ready", None)
        if binding is None:
            return
        if not self.canonical_ready_mode:
            raise ValueError(
                "action_ball_dynamic_ready requires canonical_ready_mode"
            )
        expected_top_keys = {
            "schema_version",
            "kind",
            "binding_sha256",
            "action_order",
            "motion_sha256_per_action",
            "rows",
        }
        if type(binding) is not dict or set(binding) != expected_top_keys:
            raise ValueError(
                "action_ball_dynamic_ready must be one exact schema-1/2 runtime binding"
            )
        schema_version = binding["schema_version"]
        expected_kind = {
            1: "action_ball_dynamic_ready_runtime_binding_v1",
            2: "action_ball_dynamic_ready_runtime_binding_v2",
        }.get(schema_version)
        if type(schema_version) is not int or binding["kind"] != expected_kind:
            raise ValueError(
                "action_ball_dynamic_ready schema_version/kind mismatch"
            )
        binding_sha256 = self._action_ball_dynamic_ready_exact_sha256(
            binding["binding_sha256"],
            name="action_ball_dynamic_ready.binding_sha256",
        )
        unsigned = dict(binding)
        del unsigned["binding_sha256"]
        actual_binding_sha256 = self._action_ball_dynamic_ready_sha256(
            unsigned
        )
        if actual_binding_sha256 != binding_sha256:
            raise ValueError(
                "action_ball_dynamic_ready binding SHA-256 mismatch: "
                f"{actual_binding_sha256} != {binding_sha256}"
            )

        action_order_raw = binding["action_order"]
        if (
            not isinstance(action_order_raw, list)
            or not action_order_raw
            or any(
                not isinstance(action_id, str) or not action_id
                for action_id in action_order_raw
            )
            or len(set(action_order_raw)) != len(action_order_raw)
        ):
            raise ValueError(
                "action_ball_dynamic_ready.action_order must contain unique "
                "non-empty action ids"
            )
        action_order = tuple(action_order_raw)
        action_count = int(self.motion.num_segments)
        if len(action_order) != action_count:
            raise ValueError(
                "action_ball_dynamic_ready action count differs from loaded motion: "
                f"{len(action_order)} != {action_count}"
            )
        motion_sha_raw = binding["motion_sha256_per_action"]
        if not isinstance(motion_sha_raw, list):
            raise ValueError(
                "action_ball_dynamic_ready.motion_sha256_per_action must be a list"
            )
        motion_sha256_per_action = tuple(
            self._action_ball_dynamic_ready_exact_sha256(
                value,
                name=(
                    "action_ball_dynamic_ready.motion_sha256_per_action"
                    f"[{index}]"
                ),
            )
            for index, value in enumerate(motion_sha_raw)
        )
        if motion_sha256_per_action != tuple(self._motion_file_sha256):
            raise ValueError(
                "action_ball_dynamic_ready ordered motion SHA-256 values differ "
                "from the immutable MotionLoader inputs"
            )
        canonical_motion_ids = getattr(self, "canonical_motion_ids", None)
        if (
            canonical_motion_ids is not None
            and tuple(canonical_motion_ids) != action_order
        ):
            raise ValueError(
                "action_ball_dynamic_ready.action_order differs from the "
                "canonical registry motion ids"
            )

        rows = binding["rows"]
        if not isinstance(rows, list) or len(rows) != action_count:
            raise ValueError(
                "action_ball_dynamic_ready.rows must have one row per action"
            )
        expected_row_keys = {
            "action_id",
            "physical_ready",
            "hold_qdes_joint_pos_rad",
            "normalized_actor_action",
            "artifact",
            "nominal_hold_receipt",
        }
        if schema_version == 2:
            expected_row_keys.add("runtime_plant_identity")
        expected_physical_keys = {
            "root_pos_w_m",
            "root_quat_wxyz",
            "joint_pos_rad",
            "joint_vel_radps",
        }
        expected_pin_keys = {"path", "sha256", "content_sha256"}
        root_pos_rows: list[tuple[float, ...]] = []
        root_quat_rows: list[tuple[float, ...]] = []
        joint_pos_rows: list[tuple[float, ...]] = []
        joint_vel_rows: list[tuple[float, ...]] = []
        hold_qdes_rows: list[tuple[float, ...]] = []
        normalized_action_rows: list[tuple[float, ...]] = []
        for action_slot, row in enumerate(rows):
            if type(row) is not dict or set(row) != expected_row_keys:
                raise ValueError(
                    f"action_ball_dynamic_ready.rows[{action_slot}] has "
                    "unexpected or missing fields"
                )
            if row["action_id"] != action_order[action_slot]:
                raise ValueError(
                    "action_ball_dynamic_ready row order differs from action_order "
                    f"at slot {action_slot}"
                )
            if schema_version == 2:
                self._validate_action_ball_dynamic_ready_plant_v2(
                    row["runtime_plant_identity"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].runtime_plant_identity"
                    ),
                )
            for pin_name in ("artifact", "nominal_hold_receipt"):
                pin = row[pin_name]
                if type(pin) is not dict or set(pin) != expected_pin_keys:
                    raise ValueError(
                        "action_ball_dynamic_ready "
                        f"rows[{action_slot}].{pin_name} must contain exact "
                        "path/file/content pins"
                    )
                if not isinstance(pin["path"], str) or not pin["path"]:
                    raise ValueError(
                        "action_ball_dynamic_ready "
                        f"rows[{action_slot}].{pin_name}.path must be non-empty"
                    )
                for digest_name in ("sha256", "content_sha256"):
                    self._action_ball_dynamic_ready_exact_sha256(
                        pin[digest_name],
                        name=(
                            "action_ball_dynamic_ready."
                            f"rows[{action_slot}].{pin_name}.{digest_name}"
                        ),
                    )

            physical = row["physical_ready"]
            if (
                type(physical) is not dict
                or set(physical) != expected_physical_keys
            ):
                raise ValueError(
                    "action_ball_dynamic_ready physical_ready must contain "
                    "exact root/joint state fields"
                )
            root_pos_rows.append(
                self._action_ball_dynamic_ready_vector(
                    physical["root_pos_w_m"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].physical_ready.root_pos_w_m"
                    ),
                    length=3,
                )
            )
            root_quat_rows.append(
                self._action_ball_dynamic_ready_vector(
                    physical["root_quat_wxyz"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].physical_ready.root_quat_wxyz"
                    ),
                    length=4,
                )
            )
            joint_pos_rows.append(
                self._action_ball_dynamic_ready_vector(
                    physical["joint_pos_rad"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].physical_ready.joint_pos_rad"
                    ),
                    length=_A3_CANONICAL_READY_JOINT_COUNT,
                )
            )
            joint_vel = self._action_ball_dynamic_ready_vector(
                physical["joint_vel_radps"],
                name=(
                    "action_ball_dynamic_ready."
                    f"rows[{action_slot}].physical_ready.joint_vel_radps"
                ),
                length=_A3_CANONICAL_READY_JOINT_COUNT,
            )
            if any(value != 0.0 for value in joint_vel):
                raise ValueError(
                    "action_ball_dynamic_ready physical joint velocities "
                    "must be literal zero"
                )
            joint_vel_rows.append(joint_vel)
            hold_qdes_rows.append(
                self._action_ball_dynamic_ready_vector(
                    row["hold_qdes_joint_pos_rad"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].hold_qdes_joint_pos_rad"
                    ),
                    length=_A3_CANONICAL_READY_JOINT_COUNT,
                )
            )
            normalized_action_rows.append(
                self._action_ball_dynamic_ready_vector(
                    row["normalized_actor_action"],
                    name=(
                        "action_ball_dynamic_ready."
                        f"rows[{action_slot}].normalized_actor_action"
                    ),
                    length=_A3_CANONICAL_READY_JOINT_COUNT,
                )
            )

        starts = self.motion.seg_start
        physical_root_pos = torch.tensor(
            root_pos_rows,
            dtype=self.motion.body_pos_w.dtype,
            device=self.motion.body_pos_w.device,
        )
        physical_root_quat = torch.tensor(
            root_quat_rows,
            dtype=self.motion.body_quat_w.dtype,
            device=self.motion.body_quat_w.device,
        )
        physical_joint_pos = torch.tensor(
            joint_pos_rows,
            dtype=self.motion.joint_pos.dtype,
            device=self.motion.joint_pos.device,
        )
        physical_joint_vel = torch.tensor(
            joint_vel_rows,
            dtype=self.motion.joint_vel.dtype,
            device=self.motion.joint_vel.device,
        )
        exact_frame0 = (
            (
                "root_pos_w_m",
                physical_root_pos,
                self.motion.body_pos_w[starts, 0],
            ),
            (
                "root_quat_wxyz",
                physical_root_quat,
                self.motion.body_quat_w[starts, 0],
            ),
            (
                "joint_pos_rad",
                physical_joint_pos,
                self.motion.joint_pos[starts],
            ),
            (
                "joint_vel_radps",
                physical_joint_vel,
                self.motion.joint_vel[starts],
            ),
        )
        for name, supplied, motion_value in exact_frame0:
            if not torch.equal(supplied, motion_value):
                mismatch_count, max_abs = self._first_tensor_mismatch(
                    motion_value, supplied
                )
                raise ValueError(
                    "action_ball_dynamic_ready physical frame-0 mismatch: "
                    f"channel={name} mismatches={mismatch_count} "
                    f"max_abs={max_abs:.9g}"
                )

        self._action_ball_dynamic_ready_binding_sha256 = binding_sha256
        self._action_ball_dynamic_ready_action_order = action_order
        self._action_ball_dynamic_ready_physical_root_pos_w_m = (
            physical_root_pos
        )
        self._action_ball_dynamic_ready_physical_root_quat_wxyz = (
            physical_root_quat
        )
        self._action_ball_dynamic_ready_physical_joint_pos_rad = (
            physical_joint_pos
        )
        self._action_ball_dynamic_ready_physical_joint_vel_radps = (
            physical_joint_vel
        )
        self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad = torch.tensor(
            hold_qdes_rows,
            dtype=self.motion.joint_pos.dtype,
            device=self.motion.joint_pos.device,
        )
        self._action_ball_dynamic_ready_normalized_actor_action = torch.tensor(
            normalized_action_rows,
            dtype=self.motion.joint_pos.dtype,
            device=self.motion.joint_pos.device,
        )
        # ActionManager constructs after CommandManager in Isaac Lab.  Keep all
        # pre-scene identity/physical validation above, but resolve the decoder
        # term only at the first true reset, before any simulator state write.
        self._action_ball_dynamic_ready_action_term = None

    def _bind_action_ball_dynamic_ready_action_term(self):
        """Resolve the decoder handshake after ActionManager exists."""

        if self._action_ball_dynamic_ready_binding_sha256 is None:
            return None
        if self._action_ball_dynamic_ready_action_term is not None:
            return self._action_ball_dynamic_ready_action_term
        action_manager = getattr(self._env, "action_manager", None)
        get_term = getattr(action_manager, "get_term", None)
        if not callable(get_term):
            raise RuntimeError(
                "action_ball_dynamic_ready requires ActionManager.get_term "
                "before its first true reset"
            )
        try:
            action_term = get_term("joint_pos")
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                "action_ball_dynamic_ready requires the joint_pos action term"
            ) from exc
        for method_name in (
            "install_action_ball_dynamic_ready_state",
            "restore_action_ball_dynamic_ready_state",
        ):
            if not callable(getattr(action_term, method_name, None)):
                raise RuntimeError(
                    "action_ball_dynamic_ready joint_pos action term lacks "
                    f"{method_name}"
                )
        processed = getattr(action_term, "processed_actions", None)
        if (
            not torch.is_tensor(processed)
            or processed.ndim != 2
            or processed.shape[1] != _A3_CANONICAL_READY_JOINT_COUNT
        ):
            raise RuntimeError(
                "action_ball_dynamic_ready requires an identity-ordered "
                "31-D joint_pos decoder"
            )
        self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad = (
            self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad.to(
                dtype=processed.dtype, device=processed.device
            )
        )
        self._action_ball_dynamic_ready_normalized_actor_action = (
            self._action_ball_dynamic_ready_normalized_actor_action.to(
                dtype=processed.dtype, device=processed.device
            )
        )
        self._action_ball_dynamic_ready_action_term = action_term
        return action_term

    def _canonical_ready_steps(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        clips = self.clip_id if env_ids is None else self.clip_id[env_ids]
        return self.motion.seg_start[clips]

    def _require_canonical_ready_boundary(
        self, env_ids: torch.Tensor, operation: str
    ) -> None:
        """Allow an in-episode clip retarget only at a proven zero-speed ready endpoint."""

        if not self.canonical_ready_mode or len(env_ids) == 0:
            return
        clips = self.clip_id[env_ids]
        starts = self.motion.seg_start[clips]
        ends = starts + self.motion.seg_len[clips] - 1
        steps = self.time_steps[env_ids]
        at_ready_boundary = (steps == starts) | (steps == ends)
        if not bool(torch.all(at_ready_boundary)):
            bad = env_ids[~at_ready_boundary].detach().cpu().tolist()
            raise ValueError(
                f"{operation} cannot change canonical clip mid-stroke; "
                f"envs {bad} are not at a shared zero-speed ready boundary"
            )

    def _pose_reference_steps(self) -> torch.Tensor:
        if not self.canonical_ready_mode:
            return self.time_steps
        return torch.where(self.in_hold, self._canonical_ready_steps(), self.time_steps)

    @classmethod
    def _action_ball_plain_int(
        cls, value, *, name: str, minimum: int = 0
    ) -> int:
        if type(value) is not int or not minimum <= value <= cls._ACTION_BALL_INT64_MAX:
            raise ValueError(
                f"{name} must be a plain integer in "
                f"[{minimum}, {cls._ACTION_BALL_INT64_MAX}]"
            )
        return value

    @staticmethod
    def _action_ball_sha256(value, *, name: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"{name} must be exactly 64 lowercase hexadecimal characters"
            )
        return value

    @classmethod
    def _action_ball_runtime_module(cls):
        """Return the exact repository runtime module that minted the broker classes."""

        import importlib
        import importlib.util
        import sys

        global _ACTION_BALL_RUNTIME_MODULE
        script = Path(__file__).resolve().with_name("action_ball_runtime.py")
        module_name = (
            "whole_body_tracking.tasks.tracking.mdp.action_ball_runtime"
        )
        if _ACTION_BALL_RUNTIME_MODULE is None:
            module = sys.modules.get(module_name)
            if module is None:
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    # CPU unit tests load this package from file under a namespace-only stub.
                    # Execute the same repository bytes under the canonical name so the classes
                    # used by Motion and the test broker still have one exact identity.
                    spec = importlib.util.spec_from_file_location(
                        module_name, script
                    )
                    if spec is None or spec.loader is None:
                        raise ValueError(
                            "cannot create action-ball runtime module spec"
                        )
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    try:
                        spec.loader.exec_module(module)
                    except Exception:
                        sys.modules.pop(module_name, None)
                        raise
            _ACTION_BALL_RUNTIME_MODULE = module
        module = _ACTION_BALL_RUNTIME_MODULE
        try:
            module_file = Path(module.__file__).resolve(strict=True)
        except (AttributeError, OSError) as exc:
            raise ValueError(
                "action-ball runtime module has no exact repository file"
            ) from exc
        if module_file != script:
            raise ValueError(
                "action-ball runtime module resolved to a different file"
            )
        if (
            getattr(module, "BROKER_STATE_SCHEMA_VERSION", None) != 4
            or getattr(module, "SCHEMA_VERSION", None) != 3
            or getattr(module, "SAMPLER_SCHEMA_VERSION", None) != 3
        ):
            raise ValueError(
                "unsupported action-ball runtime/broker/sampler schema"
            )
        cls._action_ball_sha256(
            getattr(module, "ARM_CATALOG_SHA256", None),
            name="action-ball runtime arm catalog SHA",
        )
        for name in (
            "ActionBinding",
            "ActionBirthBroker",
            "ActionBirthReceipt",
            "ActionBallTaskReceipt",
            "ActionTaskReceiptRef",
            "BirthReserveRequest",
            "BirthCommitRequest",
        ):
            if not isinstance(getattr(module, name, None), type):
                raise ValueError(
                    f"action-ball runtime is missing exact {name}"
                )
        return module

    @staticmethod
    def _action_ball_vector(
        value, *, name: str, length: int
    ) -> tuple[float, ...]:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, (tuple, list))
            or len(value) != length
        ):
            raise ValueError(f"{name} must be an exact length-{length} tuple/list")
        result = []
        for index, component in enumerate(value):
            if (
                isinstance(component, bool)
                or type(component) not in (int, float)
                or not math.isfinite(float(component))
            ):
                raise ValueError(f"{name}[{index}] must be a plain finite number")
            result.append(float(component))
        return tuple(result)

    @staticmethod
    def _action_ball_resolve_root(value) -> Path:
        if isinstance(value, bool):
            raise ValueError("trusted_repo_root must be one explicit absolute path")
        try:
            raw = Path(os.fspath(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "trusted_repo_root must be one explicit absolute path"
            ) from exc
        if not raw.is_absolute() or raw.is_symlink():
            raise ValueError(
                "trusted_repo_root must be absolute and must not be a symlink"
            )
        try:
            resolved = raw.resolve(strict=True)
        except OSError as exc:
            raise ValueError("trusted_repo_root cannot be resolved") from exc
        try:
            mode = resolved.stat().st_mode
        except OSError as exc:
            raise ValueError("trusted_repo_root cannot be stat'ed") from exc
        if not stat.S_ISDIR(mode):
            raise ValueError("trusted_repo_root must be a regular directory")
        return resolved

    @staticmethod
    def _action_ball_file_receipt(
        repo_root: Path,
        relative_path: str,
        *,
        name: str,
        expected_sha256: str | None = None,
    ) -> tuple[Path, str]:
        """Resolve one normalized repo-relative regular file without following symlinks."""

        if (
            type(relative_path) is not str
            or not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
        ):
            raise ValueError(f"{name} must be one normalized repo-relative POSIX path")
        parts = tuple(relative_path.split("/"))
        if (
            any(part in ("", ".", "..") for part in parts)
            or Path(relative_path).is_absolute()
        ):
            raise ValueError(f"{name} must not escape the trusted repository root")
        cursor = repo_root
        for part in parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"{name} must not contain a symlink")
        try:
            resolved = cursor.resolve(strict=True)
            resolved.relative_to(repo_root)
            mode = resolved.stat().st_mode
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{name} cannot be resolved inside trusted_repo_root"
            ) from exc
        if not stat.S_ISREG(mode):
            raise ValueError(f"{name} must resolve to a regular file")
        try:
            payload = resolved.read_bytes()
        except OSError as exc:
            raise ValueError(f"{name} cannot be read") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            expected = MotionCommand._action_ball_sha256(
                expected_sha256, name=f"{name}.expected_sha256"
            )
            if digest != expected:
                raise ValueError(f"{name} SHA-256 changed after admission")
        return resolved, digest

    @classmethod
    def _action_ball_repo_file_receipt(
        cls,
        repo_root: Path,
        path,
        *,
        name: str,
        expected_sha256: str | None = None,
    ) -> tuple[str, str]:
        try:
            resolved = Path(path).resolve(strict=True)
            relative = resolved.relative_to(repo_root).as_posix()
        except (OSError, ValueError) as exc:
            raise ValueError(f"{name} escaped trusted_repo_root") from exc
        checked, digest = cls._action_ball_file_receipt(
            repo_root,
            relative,
            name=name,
            expected_sha256=expected_sha256,
        )
        if checked != resolved:
            raise ValueError(f"{name} changed during path admission")
        return relative, digest

    def _require_action_ball_motion_admission(
        self, repo_root: Path
    ) -> None:
        registry = self._canonical_motion_registry
        admission = self._canonical_motion_admission
        binding = self._canonical_motion_promotion_binding
        module = self._canonical_motion_registry_module
        if (
            registry is None
            or admission is None
            or binding is None
            or module is None
        ):
            raise ValueError(
                "action-ball requires the code-rooted opaque canonical motion admission"
            )
        if Path(registry.repo_root).resolve(strict=True) != repo_root:
            raise ValueError(
                "action-ball trusted_repo_root differs from canonical motion admission"
            )
        try:
            module.motion_admission.require_matching_admission(
                admission, binding
            )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ValueError(
                "opaque canonical motion admission failed revalidation"
            ) from exc

    def bind_action_ball_birth_broker(
        self, broker, *, trusted_repo_root
    ) -> None:
        """Bind one exact schema-v4 broker to already-admitted motion bytes.

        ``trusted_repo_root`` is deliberately mandatory.  Runtime manifest paths are relative to
        that explicit root and cannot depend on the process working directory.  The manifest only
        supplies identity rows; authorization is re-proved from MotionCommand's retained opaque
        promotion capability.
        """

        if self._action_ball_birth_broker is not None:
            raise ValueError("action-ball birth broker may be bound exactly once")
        if not self.canonical_ready_mode:
            raise ValueError(
                "action-ball birth requires canonical_ready_mode=true"
            )
        if bool(self.cfg.wrap_teleport):
            raise ValueError("action-ball birth requires wrap_teleport=false")
        runtime = self._action_ball_runtime_module()
        if type(broker) is not runtime.ActionBirthBroker:
            raise ValueError(
                "action-ball birth broker must be the exact repository ActionBirthBroker"
            )
        if (
            broker.diagnostic_fast_path
            != self._canonical_diagnostic_unauthorized
        ):
            raise ValueError(
                "action-ball broker diagnostic mode differs from Motion"
            )
        repo_root = self._action_ball_resolve_root(trusted_repo_root)
        if self._canonical_diagnostic_unauthorized:
            print(
                "[MotionCommand] WARN action-ball birth broker bound WITHOUT "
                "canonical motion admission (diagnostic_unauthorized=true)",
                flush=True,
            )
        else:
            self._require_action_ball_motion_admission(repo_root)
        for method_name in (
            "binding_for_slot",
            "reserve_many_true_reset",
            "pending_receipt",
            "commit_many_true_reset",
            "state_dict",
            "load_state_dict",
        ):
            if not callable(getattr(broker, method_name, None)):
                raise ValueError(
                    f"action-ball schema-v4 broker must implement {method_name}()"
                )
        broker_state = broker.state_dict()
        if (
            type(broker_state) is not dict
            or broker_state.get("schema_version")
            != runtime.BROKER_STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                "action-ball broker/provider/domain authority are not fully bound"
            )

        action_count = self._action_ball_plain_int(
            broker.action_count,
            name="broker.action_count",
            minimum=1,
        )
        if action_count != int(self.motion.num_segments):
            raise ValueError(
                "action-ball action count must equal the loaded motion segment count"
            )
        action_uids = tuple(
            self._action_ball_plain_int(
                uid, name=f"broker.ordered_action_uids[{slot}]", minimum=1
            )
            for slot, uid in enumerate(broker.ordered_action_uids)
        )
        if (
            len(action_uids) != action_count
            or len(set(action_uids)) != action_count
        ):
            raise ValueError(
                "broker.ordered_action_uids must contain one unique UID per slot"
            )

        motion_sha256: list[str] = []
        for slot in range(action_count):
            binding = broker.binding_for_slot(slot)
            if type(binding) is not runtime.ActionBinding:
                raise ValueError(
                    f"action-ball binding[{slot}] has a forged runtime type"
                )
            if (
                binding.action_slot != slot
                or binding.action_uid != action_uids[slot]
            ):
                raise ValueError(
                    f"action-ball binding[{slot}] does not match its ordered slot/UID"
                )
            resolved, digest = self._action_ball_file_receipt(
                repo_root,
                binding.motion_path,
                name=f"action-ball binding[{slot}].motion_path",
                expected_sha256=binding.motion_sha256,
            )
            if (
                str(resolved) != self._motion_files[slot]
                or digest != self._motion_file_sha256[slot]
                or digest
                != hashlib.sha256(self._motion_payloads[slot]).hexdigest()
            ):
                raise ValueError(
                    f"action-ball binding[{slot}] does not match the admitted loaded clip bytes"
                )
            motion_sha256.append(digest)

        ready_steps = self.motion.seg_start.to(
            device=self.motion.body_pos_w.device, dtype=torch.long
        )
        ready_root_z = tuple(
            float(value)
            for value in self.motion.body_pos_w[
                ready_steps, 0, 2
            ].detach().cpu().tolist()
        )
        ready_root_quat = tuple(
            tuple(float(component) for component in row)
            for row in self.motion.body_quat_w[
                ready_steps, 0
            ].detach().cpu().tolist()
        )
        segment_lengths = None
        if broker.diagnostic_fast_path:
            segment_lengths = tuple(
                int(value)
                for value in self.motion.seg_len.detach().cpu().tolist()
            )
            if (
                len(segment_lengths) != action_count
                or any(length < 3 for length in segment_lengths)
            ):
                raise ValueError(
                    "action-ball admitted motions require one interior frame per action"
                )
        # Publish only after every opaque admission and file/broker row has passed.  The hard
        # receipt immediately reopens the capability and all implementation sources once more.
        self._action_ball_birth_broker = broker
        self._action_ball_runtime_module_bound = runtime
        self._action_ball_trusted_repo_root = repo_root
        self._action_ball_action_uids = action_uids
        self._action_ball_motion_sha256 = tuple(motion_sha256)
        self._action_ball_segment_lengths = segment_lengths
        self._action_ball_ready_root_z = ready_root_z
        self._action_ball_ready_root_quat = ready_root_quat
        self._action_ball_reset_generation = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._action_ball_swing_generation = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._action_ball_birth_receipt_sha256 = [None] * self.num_envs
        self._action_ball_seen_birth_receipts = set()
        self._action_ball_active_task_refs = [None] * self.num_envs
        self._action_ball_task_timing_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Diagnostic Racket resolves every reset/wrap selection synchronously in
        # the same command-manager pass.  Keep that host-known batch state here
        # so ordinary active steps do not rediscover an empty pending set with a
        # device-wide ``torch.where`` and host length check.
        self._action_ball_diagnostic_pending_row_count = 0
        timing_shape = (self.num_envs,)
        timing_options = {
            "dtype": torch.float64,
            "device": self.device,
        }
        self._action_ball_task_pending_elapsed_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_task_age_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_time_to_contact_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_teacher_rate = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_scaled_t_hit_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_scaled_t_cycle_s = torch.zeros(
            timing_shape, **timing_options
        )
        self._action_ball_pre_swing_wait_s = torch.zeros(
            timing_shape, **timing_options
        )
        try:
            receipt = self.action_ball_motion_admission_hard_contract()
        except Exception:
            self._action_ball_birth_broker = None
            self._action_ball_runtime_module_bound = None
            self._action_ball_trusted_repo_root = None
            self._action_ball_action_uids = None
            self._action_ball_motion_sha256 = None
            self._action_ball_segment_lengths = None
            self._action_ball_ready_root_z = None
            self._action_ball_ready_root_quat = None
            self._action_ball_reset_generation = None
            self._action_ball_swing_generation = None
            self._action_ball_birth_receipt_sha256 = None
            self._action_ball_seen_birth_receipts = None
            self._action_ball_active_task_refs = None
            self._action_ball_task_timing_active = None
            self._action_ball_diagnostic_pending_row_count = None
            self._action_ball_task_pending_elapsed_s = None
            self._action_ball_task_age_s = None
            self._action_ball_time_to_contact_s = None
            self._action_ball_teacher_rate = None
            self._action_ball_scaled_t_hit_s = None
            self._action_ball_scaled_t_cycle_s = None
            self._action_ball_pre_swing_wait_s = None
            raise
        self._action_ball_motion_admission_receipt_sha256 = receipt[
            "canonical_sha256"
        ]

    def bind_action_ball_task_authority(
        self, *, task_ref_for_env, resolve_task_ref, shared_state_sha256
    ) -> None:
        """Bind Racket's opaque ref/resolve and shared-digest authority seams exactly once."""

        if self._action_ball_birth_broker is None:
            raise RuntimeError(
                "action-ball task authority requires the birth broker first"
            )
        if (
            self._action_ball_task_ref_for_env is not None
            or self._action_ball_task_receipt_resolver is not None
            or self._action_ball_shared_state_sha256_accessor is not None
        ):
            raise ValueError("action-ball task authority may be bound exactly once")
        if (
            not callable(task_ref_for_env)
            or getattr(task_ref_for_env, "__name__", None)
            != "action_ball_task_ref_for_env"
            or not callable(resolve_task_ref)
            or getattr(resolve_task_ref, "__name__", None)
            != "action_ball_resolve_task_ref"
            or not callable(shared_state_sha256)
            or getattr(shared_state_sha256, "__name__", None)
            != "action_ball_shared_state_sha256"
        ):
            raise ValueError(
                "action-ball task authority requires the exact public Racket accessors"
            )
        ref_owner = getattr(task_ref_for_env, "__self__", None)
        resolver_owner = getattr(resolve_task_ref, "__self__", None)
        digest_owner = getattr(shared_state_sha256, "__self__", None)
        if (
            ref_owner is None
            or ref_owner is not resolver_owner
            or ref_owner is not digest_owner
        ):
            raise ValueError(
                "action-ball task ref, resolver and shared digest must have one bound owner"
            )
        if self.planner_revision_enabled:
            raise ValueError(
                "action-ball receipt timing is the sole phase/deadline owner"
            )
        if self._speed_per_clip is not None or tuple(
            float(value) for value in self.cfg.speed_scale_range
        ) != (1.0, 1.0):
            raise ValueError(
                "action-ball teacher_rate requires native generic speed configuration"
            )
        if (
            tuple(int(value) for value in self.cfg.hold_steps_range)
            != (0, 0)
            or int(self.cfg.stand_start_min_hold) != 0
            or int(self.cfg.post_swing_min_hold) != 0
            or bool(self.cfg.stagger_initial_clock)
        ):
            raise ValueError(
                "action-ball task receipt owns preparation wait; legacy hold/stagger must be zero"
            )
        self._action_ball_task_ref_for_env = task_ref_for_env
        self._action_ball_task_receipt_resolver = resolve_task_ref
        self._action_ball_shared_state_sha256_accessor = shared_state_sha256
        # Dynamic teacher rates use the existing audited velocity-scaling lane, but their values
        # come only from the current immutable task receipt (never the generic speed sampler).
        self.retiming_active = True
        self._action_ball_expected_shared_racket_state_sha256 = None

    def validate_action_ball_task_authority_binding(self) -> None:
        """Probe the shared Racket digest after both runtime owners are published."""

        if self._action_ball_shared_state_sha256_accessor is None:
            raise RuntimeError("action-ball task authority is not bound")
        self._action_ball_sha256(
            self._action_ball_shared_state_sha256_accessor(),
            name="Racket.action_ball_shared_state_sha256",
        )

    @property
    def action_ball_enabled(self) -> bool:
        return self._action_ball_birth_broker is not None

    @property
    def action_ball_ordered_action_uids(self) -> tuple[int, ...]:
        if self._action_ball_action_uids is None:
            raise RuntimeError("action-ball birth broker is not bound")
        return self._action_ball_action_uids

    @property
    def action_ball_reset_generation(self) -> torch.Tensor:
        if self._action_ball_reset_generation is None:
            raise RuntimeError("action-ball birth broker is not bound")
        return self._action_ball_reset_generation

    @property
    def action_ball_episode_generation(self) -> torch.Tensor:
        """Alias documenting that a reset generation identifies one physical episode."""

        return self.action_ball_reset_generation

    @property
    def action_ball_swing_generation(self) -> torch.Tensor:
        if self._action_ball_swing_generation is None:
            raise RuntimeError("action-ball birth broker is not bound")
        return self._action_ball_swing_generation

    def action_ball_action_uid_for_envs(self, env_ids) -> torch.Tensor:
        if self._action_ball_action_uids is None:
            raise RuntimeError("action-ball birth broker is not bound")
        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        uid_table = torch.tensor(
            self._action_ball_action_uids,
            dtype=torch.long,
            device=self.device,
        )
        return uid_table[self.clip_id[ids]]

    def action_ball_birth_receipt_sha256(self, env_id: int) -> str:
        env_id = self._action_ball_plain_int(
            env_id, name="env_id", minimum=0
        )
        if env_id >= self.num_envs:
            raise ValueError("env_id is outside the environment batch")
        if self._action_ball_birth_receipt_sha256 is None:
            raise RuntimeError("action-ball birth broker is not bound")
        receipt = self._action_ball_birth_receipt_sha256[env_id]
        if receipt is None:
            raise RuntimeError("environment has no committed action-ball birth")
        return receipt

    def action_ball_motion_admission_hard_contract(self) -> dict:
        """Reopen the opaque training admission and emit a content-addressed receipt."""

        if (
            self._action_ball_birth_broker is None
            or self._action_ball_trusted_repo_root is None
            or self._action_ball_runtime_module_bound is None
        ):
            raise RuntimeError("action-ball motion admission is not bound")
        if self._canonical_diagnostic_unauthorized:
            # No admission exists to reopen.  Emit a content-addressed
            # unauthorized binding receipt so exact-resume and hard-contract
            # identities can still pin the immutable bytes without mistaking
            # them for a training capability.
            payload = {
                "schema_version": 1,
                "kind": (
                    "whole_body_tracking.MotionCommand."
                    "action_ball_motion_diagnostic_binding"
                ),
                "diagnostic_unauthorized": True,
                "motion_file_sha256": list(self._motion_file_sha256),
                "training_authorized": False,
            }
            payload["canonical_sha256"] = hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest()
            return payload
        repo_root = self._action_ball_trusted_repo_root
        self._require_action_ball_motion_admission(repo_root)
        registry = self._canonical_motion_registry
        admission = self._canonical_motion_admission
        promotion_binding = self._canonical_motion_promotion_binding
        registry_module = self._canonical_motion_registry_module
        runtime = self._action_ball_runtime_module_bound

        registry_path, registry_sha = self._action_ball_repo_file_receipt(
            repo_root,
            registry.path,
            name="canonical registry",
            expected_sha256=registry.registry_sha256,
        )
        ready_path, ready_sha = self._action_ball_repo_file_receipt(
            repo_root,
            registry.canonical_ready_path,
            name="canonical ready",
            expected_sha256=registry.canonical_ready_sha256,
        )
        ready_fk_path, ready_fk_sha = self._action_ball_repo_file_receipt(
            repo_root,
            registry.canonical_ready_fk_path,
            name="canonical ready FK",
            expected_sha256=registry.canonical_ready_fk_sha256,
        )
        certificate_path, certificate_sha = (
            self._action_ball_repo_file_receipt(
                repo_root,
                getattr(admission, "_certificate_path", ""),
                name="canonical promotion certificate",
                expected_sha256=admission.certificate_sha256,
            )
        )
        binding_sha = registry_module.motion_admission._binding_sha256(
            promotion_binding
        )

        motion_rows = []
        for slot, motion_id in enumerate(self.canonical_motion_ids):
            binding = self._action_ball_birth_broker.binding_for_slot(slot)
            resolved, digest = self._action_ball_file_receipt(
                repo_root,
                binding.motion_path,
                name=f"canonical motion[{slot}]",
                expected_sha256=binding.motion_sha256,
            )
            if (
                str(resolved) != self._motion_files[slot]
                or digest != self._motion_file_sha256[slot]
            ):
                raise RuntimeError(
                    "action-ball motion bytes changed after opaque admission"
                )
            motion_rows.append(
                {
                    "motion_id": motion_id,
                    "action_uid": binding.action_uid,
                    "action_slot": binding.action_slot,
                    "motion_path": binding.motion_path,
                    "motion_sha256": digest,
                    "profile_sha256": binding.profile_sha256,
                }
            )

        source_paths = {
            "commands": Path(__file__).resolve(strict=True),
            "action_ball_runtime": Path(runtime.__file__).resolve(strict=True),
            "canonical_motion_registry": Path(
                registry_module.__file__
            ).resolve(strict=True),
            "canonical_motion_admission": Path(
                registry_module.motion_admission.__file__
            ).resolve(strict=True),
        }
        implementation_sources = {}
        for name, path in source_paths.items():
            relative, digest = self._action_ball_repo_file_receipt(
                repo_root,
                path,
                name=f"implementation source {name}",
            )
            implementation_sources[name] = {
                "path": relative,
                "sha256": digest,
            }

        payload = {
            "schema_version": 1,
            "kind": (
                "whole_body_tracking.MotionCommand."
                "action_ball_motion_admission"
            ),
            "authorization_purpose": "training",
            "trusted_repo_root": str(repo_root),
            "opaque_capability": {
                "capability_type": type(admission).__name__,
                "purpose": admission.purpose,
                "promotion_binding_sha256": binding_sha,
                "certificate_path": certificate_path,
                "certificate_sha256": certificate_sha,
            },
            "canonical_bank": {
                "bank_id": registry.bank_id,
                "scope": registry.scope,
                "registry_path": registry_path,
                "registry_sha256": registry_sha,
                "alignment_sha256": (
                    self.canonical_registry_alignment_sha256
                ),
                "canonical_ready_path": ready_path,
                "canonical_ready_sha256": ready_sha,
                "canonical_ready_fk_path": ready_fk_path,
                "canonical_ready_fk_sha256": ready_fk_sha,
                "motion_rows": motion_rows,
            },
            "runtime_binding": {
                "runtime_contract_sha256": runtime.RUNTIME_CONTRACT_SHA256,
                "broker_state_schema_version": (
                    runtime.BROKER_STATE_SCHEMA_VERSION
                ),
                "broker_registry_sha256": (
                    self._action_ball_birth_broker.registry_sha256
                ),
                "provider_state_owner_sha256": (
                    self._action_ball_birth_broker.provider_state_owner_sha256
                ),
                "ordered_action_uids": list(
                    self._action_ball_action_uids
                ),
                "manifest_rows_are_identity_only": True,
            },
            "implementation_sources": implementation_sources,
        }
        payload["canonical_sha256"] = hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest()
        return payload

    def _validate_action_ball_birth_receipt(
        self,
        receipt,
        *,
        env_id: int,
        reset_generation: int,
        action_slot: int,
        action_uid: int,
    ) -> tuple[
        str,
        tuple[float, float, float],
        tuple[float, float, float, float],
    ]:
        runtime = self._action_ball_runtime_module_bound
        if type(receipt) is not runtime.ActionBirthReceipt:
            raise ValueError("action-ball birth receipt has a forged runtime type")
        if (
            receipt.env_id != env_id
            or receipt.reset_generation != reset_generation
            or receipt.action_slot != action_slot
            or receipt.action_uid != action_uid
        ):
            raise ValueError(
                "action-ball birth does not match the batched reset request"
            )
        binding = self._action_ball_birth_broker.binding_for_slot(action_slot)
        if (
            receipt.registry_sha256
            != self._action_ball_birth_broker.registry_sha256
            or receipt.motion_sha256
            != self._action_ball_motion_sha256[action_slot]
            or receipt.profile_sha256 != binding.profile_sha256
        ):
            raise ValueError(
                "action-ball birth does not match its broker motion/profile registry"
            )
        receipt_sha = self._action_ball_sha256(
            receipt.canonical_sha256,
            name="birth.canonical_sha256",
        )
        if receipt_sha in self._action_ball_seen_birth_receipts:
            raise ValueError("action-ball birth receipt replay detected")
        spawn = self._action_ball_vector(
            receipt.base_spawn_w_m,
            name="birth.base_spawn_w_m",
            length=3,
        )
        quat = self._action_ball_vector(
            receipt.base_quat_wxyz,
            name="birth.base_quat_wxyz",
            length=4,
        )
        ready_z = self._action_ball_ready_root_z[action_slot]
        if not math.isclose(
            spawn[2], ready_z, rel_tol=0.0, abs_tol=1.0e-7
        ):
            raise ValueError(
                "action-ball birth Z differs from canonical-ready root Z"
            )
        # ``base_quat_wxyz`` is the yaw-only B_yaw frame used by the sampler
        # and solver.  The physical floating-base reset keeps the admitted
        # clip's complete ready quaternion (including real roll/pitch), so
        # compare the receipt with that quaternion's yaw projection here.
        # Conflating these two frames strips the ready pitch and moves the
        # paddle/feet by centimetres even though the joint vector is exact.
        ready_root_quat = self._action_ball_ready_root_quat[action_slot]
        rw, rx, ry, rz = ready_root_quat
        ready_yaw = math.atan2(
            2.0 * (rw * rz + rx * ry),
            1.0 - 2.0 * (ry * ry + rz * rz),
        )
        ready_frame_quat = (
            math.cos(0.5 * ready_yaw),
            0.0,
            0.0,
            math.sin(0.5 * ready_yaw),
        )
        direct = max(abs(a - b) for a, b in zip(quat, ready_frame_quat))
        negated = max(abs(a + b) for a, b in zip(quat, ready_frame_quat))
        if min(direct, negated) > 1.0e-6:
            raise ValueError(
                "action-ball birth yaw frame differs from canonical-ready root yaw"
            )
        return receipt_sha, spawn, quat

    def _rollback_action_ball_broker(
        self, state: dict, *, original_error: BaseException
    ) -> None:
        try:
            self._action_ball_birth_broker.load_state_dict(state)
            if self._action_ball_birth_broker.state_dict() != state:
                raise RuntimeError(
                    "broker rollback did not restore exact state"
                )
        except Exception as rollback_error:
            raise RuntimeError(
                "action-ball batch failed and broker/provider/domain rollback failed"
            ) from rollback_error

    def _reserve_action_ball_true_reset(
        self, env_ids: torch.Tensor
    ) -> dict:
        if self._action_ball_birth_broker is None:
            raise RuntimeError("action-ball birth broker is not bound")
        if env_ids.ndim != 1 or len(env_ids) == 0:
            raise ValueError(
                "action-ball reset batch requires unique non-empty env ids"
            )
        env_rows = tuple(
            int(value) for value in env_ids.detach().cpu().tolist()
        )
        if len(set(env_rows)) != len(env_rows):
            raise ValueError(
                "action-ball reset batch requires unique non-empty env ids"
            )
        current = self._action_ball_reset_generation[env_ids]
        if bool((current >= self._ACTION_BALL_INT64_MAX).any()):
            raise OverflowError("action-ball reset generation exhausted")
        next_generation = current + 1
        runtime = self._action_ball_runtime_module_bound
        action_slot_rows = tuple(
            int(value)
            for value in self.clip_id[env_ids].detach().cpu().tolist()
        )
        generation_rows = tuple(
            int(value)
            for value in next_generation.detach().cpu().tolist()
        )
        requests = []
        request_rows = []
        for env_id, action_slot, generation in zip(
            env_rows, action_slot_rows, generation_rows
        ):
            action_uid = self._action_ball_action_uids[action_slot]
            requests.append(
                runtime.BirthReserveRequest(
                    env_id=env_id,
                    reset_generation=generation,
                    action_uid=action_uid,
                    action_slot=action_slot,
                )
            )
            request_rows.append(
                (env_id, generation, action_slot, action_uid)
            )

        diagnostic_fast_path = (
            self._action_ball_birth_broker.diagnostic_fast_path
        )
        broker_state_before = (
            None
            if diagnostic_fast_path
            else self._action_ball_birth_broker.state_dict()
        )
        try:
            receipts = self._action_ball_birth_broker.reserve_many_true_reset(
                tuple(requests), reset_kind="true_reset"
            )
            if (
                type(receipts) is not tuple
                or len(receipts) != len(requests)
            ):
                raise ValueError(
                    "action-ball broker returned a partial reset batch"
                )
            receipt_sha256 = []
            spawn_rows = []
            quat_rows = []
            for receipt, request_row in zip(receipts, request_rows):
                env_id, generation, action_slot, action_uid = request_row
                receipt_sha, spawn, quat = (
                    self._validate_action_ball_birth_receipt(
                        receipt,
                        env_id=env_id,
                        reset_generation=generation,
                        action_slot=action_slot,
                        action_uid=action_uid,
                    )
                )
                pending = self._action_ball_birth_broker.pending_receipt(
                    env_id=env_id,
                    reset_generation=generation,
                    action_uid=action_uid,
                    action_slot=action_slot,
                    reset_kind="true_reset",
                )
                if pending is not receipt:
                    raise ValueError(
                        "action-ball broker changed a reserved receipt object"
                    )
                receipt_sha256.append(receipt_sha)
                spawn_rows.append(spawn)
                quat_rows.append(quat)
            if len(set(receipt_sha256)) != len(receipt_sha256):
                raise ValueError(
                    "action-ball broker replayed one birth within a reset batch"
                )
            spawn = torch.tensor(
                spawn_rows,
                dtype=self.motion.body_pos_w.dtype,
                device=self.device,
            )
            quat = torch.tensor(
                quat_rows,
                dtype=self.motion.body_quat_w.dtype,
                device=self.device,
            )
            if (
                tuple(spawn.shape) != (len(env_ids), 3)
                or tuple(quat.shape) != (len(env_ids), 4)
                or not bool(torch.isfinite(spawn).all())
                or not bool(torch.isfinite(quat).all())
            ):
                raise ValueError(
                    "action-ball broker returned a malformed root batch"
                )
        except Exception as exc:
            if not diagnostic_fast_path:
                self._rollback_action_ball_broker(
                    broker_state_before, original_error=exc
                )
            raise
        if diagnostic_fast_path:
            # A diagnostic exception poisons the run instead of restoring it.
            # Do not clone formal Motion rollback tensors or whole-run
            # containers on every successful short episode.
            return {
                "receipts": receipts,
                "receipt_sha256": tuple(receipt_sha256),
                "request_rows": tuple(request_rows),
                "next_generation": next_generation,
                "spawn": spawn,
                "quat": quat,
            }
        # Preserve the formal transaction's exact field order and values.
        return {
            "broker_state_before": broker_state_before,
            "receipts": receipts,
            "receipt_sha256": tuple(receipt_sha256),
            "request_rows": tuple(request_rows),
            "next_generation": next_generation,
            "spawn": spawn,
            "quat": quat,
            "motion_reset_generation_before": current.clone(),
            "motion_swing_generation_before": (
                self._action_ball_swing_generation[env_ids].clone()
            ),
            "motion_birth_receipts_before": list(
                self._action_ball_birth_receipt_sha256
            ),
            "motion_seen_receipts_before": set(
                self._action_ball_seen_birth_receipts
            ),
        }

    def _rollback_action_ball_true_reset(
        self,
        env_ids: torch.Tensor,
        transaction: dict,
        *,
        original_error: BaseException,
    ) -> None:
        if self._action_ball_birth_broker.diagnostic_fast_path:
            raise RuntimeError(
                "diagnostic action-ball reset is fail-stop and cannot be "
                "rolled back or retried"
            ) from original_error
        rollback_error = None
        broker_state_before = transaction["broker_state_before"]
        if broker_state_before is not None:
            try:
                self._rollback_action_ball_broker(
                    broker_state_before,
                    original_error=original_error,
                )
            except Exception as exc:
                rollback_error = exc
        # Restore Motion's publication fields even when a broken callback prevents broker
        # rollback, so no prefix of the batch is presented as a committed local episode.
        self._action_ball_reset_generation[env_ids] = transaction[
            "motion_reset_generation_before"
        ]
        self._action_ball_swing_generation[env_ids] = transaction[
            "motion_swing_generation_before"
        ]
        self._action_ball_birth_receipt_sha256 = list(
            transaction["motion_birth_receipts_before"]
        )
        self._action_ball_seen_birth_receipts = set(
            transaction["motion_seen_receipts_before"]
        )
        if rollback_error is not None:
            raise RuntimeError(
                "action-ball reset failed and exact transaction rollback failed"
            ) from rollback_error

    def _commit_action_ball_true_reset(
        self, env_ids: torch.Tensor, transaction: dict
    ) -> None:
        runtime = self._action_ball_runtime_module_bound
        receipt_sha256 = transaction["receipt_sha256"]
        next_generation = transaction["next_generation"]
        request_rows = transaction["request_rows"]
        if (
            len(env_ids) != len(receipt_sha256)
            or len(request_rows) != len(receipt_sha256)
            or tuple(next_generation.shape) != (len(env_ids),)
        ):
            raise RuntimeError(
                "action-ball commit batch is internally inconsistent"
            )
        requests = tuple(
            runtime.BirthCommitRequest(
                env_id=env_id,
                reset_generation=generation,
                receipt_sha256=receipt_sha256[index],
            )
            for index, (
                env_id,
                generation,
                _action_slot,
                _action_uid,
            ) in enumerate(request_rows)
        )
        # Validate every pending identity before the simulator mutation is declared committed.
        for index, (
            env_id,
            generation,
            action_slot,
            action_uid,
        ) in enumerate(request_rows):
            pending = self._action_ball_birth_broker.pending_receipt(
                env_id=env_id,
                reset_generation=generation,
                action_uid=action_uid,
                action_slot=action_slot,
                reset_kind="true_reset",
            )
            if pending.canonical_sha256 != receipt_sha256[index]:
                raise RuntimeError(
                    "action-ball pending receipt drifted before atomic commit"
                )
        self._action_ball_birth_broker.commit_many_true_reset(
            requests, reset_kind="true_reset"
        )

        updated_receipts = list(self._action_ball_birth_receipt_sha256)
        updated_seen = set(self._action_ball_seen_birth_receipts)
        for (env_id, _generation, _slot, _uid), receipt_sha in zip(
            request_rows, receipt_sha256
        ):
            if self._action_ball_birth_broker.diagnostic_fast_path:
                previous = updated_receipts[env_id]
                if previous is not None:
                    updated_seen.discard(previous)
            updated_receipts[env_id] = receipt_sha
            updated_seen.add(receipt_sha)
        self._action_ball_reset_generation[env_ids] = next_generation
        self._action_ball_swing_generation[env_ids] = 0
        self._action_ball_birth_receipt_sha256 = updated_receipts
        self._action_ball_seen_birth_receipts = updated_seen

    def _advance_action_ball_wrap_generation(
        self, env_ids: torch.Tensor
    ) -> None:
        current = self._action_ball_swing_generation[env_ids]
        if bool((current >= self._ACTION_BALL_INT64_MAX).any()):
            raise OverflowError("action-ball swing generation exhausted")
        self._action_ball_swing_generation[env_ids] = current + 1

    def _begin_action_ball_task_pending(
        self, env_ids: torch.Tensor, *, elapsed_s: float
    ) -> None:
        """Invalidate the prior swing locally until Racket publishes the new frozen receipt."""

        if (
            self._action_ball_task_ref_for_env is None
            or self._action_ball_task_receipt_resolver is None
        ):
            raise RuntimeError(
                "action-ball reset reached task timing before Racket authority was bound"
            )
        elapsed = float(elapsed_s)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(
                "action-ball pending task elapsed time must be finite and non-negative"
            )
        diagnostic_fast_path = (
            self._action_ball_birth_broker is not None
            and self._action_ball_birth_broker.diagnostic_fast_path
        )
        if not diagnostic_fast_path:
            # Formal keeps the opaque host ref lifecycle and its existing
            # selected-id readback unchanged.
            env_rows = tuple(
                int(value) for value in env_ids.detach().cpu().tolist()
            )
            for env_id in env_rows:
                self._action_ball_active_task_refs[env_id] = None
        self._action_ball_task_timing_active[env_ids] = False
        if not diagnostic_fast_path:
            self._action_ball_task_pending_elapsed_s[env_ids] = elapsed
            self._action_ball_task_age_s[env_ids] = 0.0
            self._action_ball_time_to_contact_s[env_ids] = 0.0
            self._action_ball_teacher_rate[env_ids] = 0.0
            self._action_ball_scaled_t_hit_s[env_ids] = 0.0
            self._action_ball_scaled_t_cycle_s[env_ids] = 0.0
            self._action_ball_pre_swing_wait_s[env_ids] = 0.0
        # Until the exact task arrives the admitted canonical-ready pose is the only safe target.
        self.time_steps[env_ids] = self.motion.seg_start[
            self.clip_id[env_ids]
        ]
        self.time_steps_f[env_ids] = self.time_steps[env_ids].float()
        self.speed_scale[env_ids] = 0.0
        self.hold_counter[env_ids] = 1
        self.metrics["in_hold"][env_ids] = 1.0
        if diagnostic_fast_path:
            # Racket will reuse its one identity D2H to replace the inactive
            # host refs and install every final timing column.  Until then the
            # false active mask makes the previous numeric rows unreachable.
            self._action_ball_diagnostic_pending_row_count += int(
                env_ids.numel()
            )

    @staticmethod
    def _action_ball_finite_float(
        value, *, name: str, minimum: float | None = None
    ) -> float:
        if (
            isinstance(value, bool)
            or type(value) not in (int, float)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be a plain finite number")
        result = float(value)
        if minimum is not None and result < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return result

    @staticmethod
    def _action_ball_close_float(
        actual: float, expected: float, *, name: str
    ) -> None:
        if not math.isclose(
            actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"{name} is inconsistent: actual={actual}, expected={expected}"
            )

    def _validate_action_ball_task_ref_and_receipt_host(
        self,
        task_ref,
        receipt,
        *,
        env_id: int,
        reset_generation: int,
        swing_generation: int,
        action_slot: int,
        segment_length: int,
        pending_elapsed_s: float,
    ) -> dict:
        """Validate one task entirely from immutable host identity and timing rows."""

        runtime = self._action_ball_runtime_module_bound
        env_id = self._action_ball_plain_int(
            env_id, name="task.env_id"
        )
        if env_id >= self.num_envs:
            raise ValueError("task.env_id is outside Motion's environment range")
        reset_generation = self._action_ball_plain_int(
            reset_generation,
            name="task.reset_generation",
            minimum=1,
        )
        swing_generation = self._action_ball_plain_int(
            swing_generation,
            name="task.swing_generation",
        )
        action_slot = self._action_ball_plain_int(
            action_slot, name="task.action_slot"
        )
        if (
            self._action_ball_action_uids is None
            or action_slot >= len(self._action_ball_action_uids)
        ):
            raise ValueError("task.action_slot is outside Motion's action manifest")
        segment_length = self._action_ball_plain_int(
            segment_length,
            name="task.segment_length",
            minimum=3,
        )
        pending_elapsed = self._action_ball_finite_float(
            pending_elapsed_s,
            name="task.pending_elapsed_s",
            minimum=0.0,
        )
        if type(task_ref) is not runtime.ActionTaskReceiptRef:
            raise ValueError("action-ball task ref has a forged runtime type")
        if type(receipt) is not runtime.ActionBallTaskReceipt:
            raise ValueError("action-ball task receipt has a forged runtime type")
        if self._action_ball_birth_broker.diagnostic_fast_path:
            if (
                task_ref.env_id != receipt.env_id
                or task_ref.reset_generation != receipt.reset_generation
                or task_ref.swing_generation != receipt.swing_generation
                or task_ref.action_uid != receipt.action_uid
                or task_ref.action_slot != receipt.action_slot
                or task_ref.birth_sha256 != receipt.birth_sha256
                or task_ref.sample_sha256 != receipt.sample_sha256
                or task_ref.task_sha256 != receipt.canonical_sha256
            ):
                raise ValueError(
                    "diagnostic action-ball task ref changed receipt identity"
                )
            self._action_ball_sha256(
                task_ref.task_sha256, name="task_ref.task_sha256"
            )
        else:
            canonical_ref = receipt.task_ref()
            if type(canonical_ref) is not runtime.ActionTaskReceiptRef:
                raise ValueError(
                    "action-ball task receipt emitted a forged canonical ref"
                )
            if canonical_ref != task_ref:
                raise ValueError(
                    "action-ball task resolver changed the requested "
                    "immutable ref"
                )
        action_uid = self._action_ball_action_uids[action_slot]
        birth_sha256 = self.action_ball_birth_receipt_sha256(env_id)
        if (
            receipt.env_id != env_id
            or receipt.reset_generation != reset_generation
            or receipt.swing_generation != swing_generation
            or receipt.action_slot != action_slot
            or receipt.action_uid != action_uid
            or receipt.birth_sha256 != birth_sha256
            or receipt.motion_sha256
            != self._action_ball_motion_sha256[action_slot]
        ):
            raise ValueError(
                "action-ball task receipt disagrees with Motion birth/action generation"
            )
        binding = self._action_ball_birth_broker.binding_for_slot(
            action_slot
        )
        if (
            receipt.registry_sha256
            != self._action_ball_birth_broker.registry_sha256
            or receipt.profile_sha256 != binding.profile_sha256
            or receipt.arm_catalog_sha256
            != runtime.ARM_CATALOG_SHA256
        ):
            raise ValueError(
                "action-ball task receipt disagrees with broker registry/profile/arm catalog"
            )
        self._action_ball_sha256(
            receipt.sample_sha256, name="task.sample_sha256"
        )
        self._action_ball_sha256(
            receipt.canonical_sha256, name="task.canonical_sha256"
        )

        time_to_contact = self._action_ball_finite_float(
            receipt.time_to_contact_s,
            name="task.time_to_contact_s",
            minimum=0.0,
        )
        reference_t_hit = self._action_ball_finite_float(
            receipt.reference_t_hit_s,
            name="task.reference_t_hit_s",
            minimum=0.0,
        )
        reference_t_cycle = self._action_ball_finite_float(
            receipt.reference_t_cycle_s,
            name="task.reference_t_cycle_s",
            minimum=0.0,
        )
        reference_speed = self._action_ball_finite_float(
            receipt.reference_racket_site_speed_mps,
            name="task.reference_racket_site_speed_mps",
            minimum=0.0,
        )
        required_speed = self._action_ball_finite_float(
            receipt.required_racket_site_speed_mps,
            name="task.required_racket_site_speed_mps",
            minimum=0.0,
        )
        teacher_rate = self._action_ball_finite_float(
            receipt.teacher_rate,
            name="task.teacher_rate",
            minimum=0.0,
        )
        teacher_rate_min = self._action_ball_finite_float(
            receipt.teacher_rate_min,
            name="task.teacher_rate_min",
            minimum=0.0,
        )
        teacher_rate_max = self._action_ball_finite_float(
            receipt.teacher_rate_max,
            name="task.teacher_rate_max",
            minimum=0.0,
        )
        scaled_t_hit = self._action_ball_finite_float(
            receipt.scaled_t_hit_s,
            name="task.scaled_t_hit_s",
            minimum=0.0,
        )
        scaled_t_cycle = self._action_ball_finite_float(
            receipt.scaled_t_cycle_s,
            name="task.scaled_t_cycle_s",
            minimum=0.0,
        )
        pre_swing_wait = self._action_ball_finite_float(
            receipt.pre_swing_wait_s,
            name="task.pre_swing_wait_s",
            minimum=0.0,
        )
        reaction_margin = self._action_ball_finite_float(
            receipt.reaction_margin_s,
            name="task.reaction_margin_s",
            minimum=0.0,
        )
        if (
            time_to_contact <= 0.0
            or reference_t_hit <= 0.0
            or reference_t_cycle <= reference_t_hit
            or reference_speed <= 0.0
            or required_speed <= 0.0
            or teacher_rate <= 0.0
            or teacher_rate_min <= 0.0
            or teacher_rate_max < teacher_rate_min
            or scaled_t_hit <= 0.0
            or scaled_t_cycle <= scaled_t_hit
        ):
            raise ValueError("action-ball task timing has a non-positive/order violation")
        if not teacher_rate_min <= 1.0 <= teacher_rate_max:
            raise ValueError(
                "action-ball certified teacher-rate bounds must contain native rate 1"
            )
        contact_geometry = runtime._contact_geometry
        try:
            contact_geometry.canonical_teacher_rate_from_site_speed(
                teacher_rate,
                1.0,
                teacher_rate_min,
                teacher_rate_max,
            )
            canonical_teacher_rate = (
                contact_geometry.canonical_teacher_rate_from_site_speed(
                required_speed,
                reference_speed,
                teacher_rate_min,
                teacher_rate_max,
                )
            )
        except contact_geometry.ExactFaceContactGeometryError as exc:
            # Keep the consumer on the producer's one SHA-bound float32 seam.
            # This remains fail-closed outside the canonical 5e-7 boundary
            # tolerance and never clips or retimes the task.
            raise ValueError(
                "action-ball teacher_rate is outside its certified range"
            ) from exc
        self._action_ball_close_float(
            teacher_rate,
            canonical_teacher_rate,
            name="task canonical teacher_rate",
        )
        required_vector = self._action_ball_vector(
            receipt.racket_site_velocity_w_mps,
            name="task.racket_site_velocity_w_mps",
            length=3,
        )
        self._action_ball_close_float(
            required_speed,
            math.sqrt(sum(value * value for value in required_vector)),
            name="task required racket-site speed",
        )
        self._action_ball_close_float(
            teacher_rate,
            required_speed / reference_speed,
            name="task teacher_rate=required/reference",
        )
        self._action_ball_close_float(
            scaled_t_hit,
            reference_t_hit / teacher_rate,
            name="task scaled_t_hit_s",
        )
        self._action_ball_close_float(
            scaled_t_cycle,
            reference_t_cycle / teacher_rate,
            name="task scaled_t_cycle_s",
        )
        self._action_ball_close_float(
            pre_swing_wait,
            time_to_contact - scaled_t_hit,
            name="task pre_swing_wait_s",
        )
        if (
            pre_swing_wait + 1.0e-12 < reaction_margin
            or pre_swing_wait > 1.0 + 1.0e-12
        ):
            raise ValueError(
                "action-ball pre-swing wait violates reaction/one-second bounds"
            )
        runtime_episode_length = (
            int(self._env.max_episode_length) * float(self._env.step_dt)
        )
        if (
            pre_swing_wait
            + scaled_t_cycle
            + float(self._env.step_dt)
            > runtime_episode_length + 1.0e-12
        ):
            raise ValueError(
                "action-ball task cycle plus close tick exceeds runtime episode horizon"
            )
        native_cycle = (segment_length - 1) * float(self._env.step_dt)
        self._action_ball_close_float(
            reference_t_cycle,
            native_cycle,
            name="task reference_t_cycle_s vs admitted motion",
        )
        hit_frame = reference_t_hit / float(self._env.step_dt)
        self._action_ball_close_float(
            hit_frame,
            float(round(hit_frame)),
            name="task reference_t_hit_s policy-frame alignment",
        )
        if not 0 < round(hit_frame) < segment_length - 1:
            raise ValueError(
                "action-ball task hit frame is outside the admitted motion interior"
            )
        if pending_elapsed > pre_swing_wait + 1.0e-12:
            raise RuntimeError(
                "action-ball task arrived after its certified ready-wait ended"
            )
        return {
            "time_to_contact_s": time_to_contact,
            "teacher_rate": teacher_rate,
            "scaled_t_hit_s": scaled_t_hit,
            "scaled_t_cycle_s": scaled_t_cycle,
            "pre_swing_wait_s": pre_swing_wait,
            "pending_elapsed_s": pending_elapsed,
        }

    def _validate_action_ball_task_ref_and_receipt(
        self, task_ref, receipt, *, env_id: int
    ) -> dict:
        """Resolve formal/live device identity before the common host validator."""

        env_id = self._action_ball_plain_int(
            env_id, name="task.env_id"
        )
        if env_id >= self.num_envs:
            raise ValueError("task.env_id is outside Motion's environment range")
        action_slot = int(self.clip_id[env_id].item())
        return self._validate_action_ball_task_ref_and_receipt_host(
            task_ref,
            receipt,
            env_id=env_id,
            reset_generation=int(
                self._action_ball_reset_generation[env_id].item()
            ),
            swing_generation=int(
                self._action_ball_swing_generation[env_id].item()
            ),
            action_slot=action_slot,
            segment_length=int(
                self.motion.seg_len[action_slot].item()
            ),
            pending_elapsed_s=float(
                self._action_ball_task_pending_elapsed_s[env_id].item()
            ),
        )

    def _validate_action_ball_task_ref_and_receipt_diagnostic_prevalidated_host(
        self,
        task_ref,
        receipt,
        *,
        env_id: int,
        reset_generation: int,
        swing_generation: int,
        action_slot: int,
        segment_length: int,
        pending_elapsed_s: float,
    ) -> dict:
        """Validate diagnostic runtime identity with lean consumer-owned algebra.

        The diagnostic pool and Racket have already admitted these exact frozen
        runtime objects.  Motion still owns the current-generation identity,
        timing fields it consumes, admitted-motion, episode-horizon, and
        pending-wait checks.  The formal path continues to use the complete
        validator above.
        """

        runtime = self._action_ball_runtime_module_bound
        if (
            self._action_ball_birth_broker is None
            or not self._action_ball_birth_broker.diagnostic_fast_path
        ):
            raise RuntimeError(
                "prevalidated action-ball task validation is diagnostic-only"
            )
        env_id = self._action_ball_plain_int(
            env_id, name="task.env_id"
        )
        if env_id >= self.num_envs:
            raise ValueError("task.env_id is outside Motion's environment range")
        reset_generation = self._action_ball_plain_int(
            reset_generation,
            name="task.reset_generation",
            minimum=1,
        )
        swing_generation = self._action_ball_plain_int(
            swing_generation,
            name="task.swing_generation",
        )
        action_slot = self._action_ball_plain_int(
            action_slot, name="task.action_slot"
        )
        if (
            self._action_ball_action_uids is None
            or action_slot >= len(self._action_ball_action_uids)
        ):
            raise ValueError("task.action_slot is outside Motion's action manifest")
        segment_length = self._action_ball_plain_int(
            segment_length,
            name="task.segment_length",
            minimum=3,
        )
        pending_elapsed = self._action_ball_finite_float(
            pending_elapsed_s,
            name="task.pending_elapsed_s",
            minimum=0.0,
        )
        if type(task_ref) is not runtime.ActionTaskReceiptRef:
            raise ValueError("action-ball task ref has a forged runtime type")
        if type(receipt) is not runtime.ActionBallTaskReceipt:
            raise ValueError("action-ball task receipt has a forged runtime type")
        if (
            task_ref.env_id != receipt.env_id
            or task_ref.reset_generation != receipt.reset_generation
            or task_ref.swing_generation != receipt.swing_generation
            or task_ref.action_uid != receipt.action_uid
            or task_ref.action_slot != receipt.action_slot
            or task_ref.birth_sha256 != receipt.birth_sha256
            or task_ref.sample_sha256 != receipt.sample_sha256
            or task_ref.task_sha256 != receipt.canonical_sha256
        ):
            raise ValueError(
                "diagnostic action-ball task ref changed receipt identity"
            )
        self._action_ball_sha256(
            task_ref.task_sha256, name="task_ref.task_sha256"
        )

        action_uid = self._action_ball_action_uids[action_slot]
        birth_sha256 = self.action_ball_birth_receipt_sha256(env_id)
        if (
            receipt.env_id != env_id
            or receipt.reset_generation != reset_generation
            or receipt.swing_generation != swing_generation
            or receipt.action_slot != action_slot
            or receipt.action_uid != action_uid
            or receipt.birth_sha256 != birth_sha256
            or receipt.motion_sha256
            != self._action_ball_motion_sha256[action_slot]
        ):
            raise ValueError(
                "action-ball task receipt disagrees with Motion birth/action generation"
            )
        binding = self._action_ball_birth_broker.binding_for_slot(
            action_slot
        )
        if (
            receipt.registry_sha256
            != self._action_ball_birth_broker.registry_sha256
            or receipt.profile_sha256 != binding.profile_sha256
            or receipt.arm_catalog_sha256
            != runtime.ARM_CATALOG_SHA256
        ):
            raise ValueError(
                "action-ball task receipt disagrees with broker registry/profile/arm catalog"
            )
        self._action_ball_sha256(
            receipt.sample_sha256, name="task.sample_sha256"
        )
        self._action_ball_sha256(
            receipt.canonical_sha256, name="task.canonical_sha256"
        )

        time_to_contact = self._action_ball_finite_float(
            receipt.time_to_contact_s,
            name="task.time_to_contact_s",
            minimum=0.0,
        )
        reference_t_hit = self._action_ball_finite_float(
            receipt.reference_t_hit_s,
            name="task.reference_t_hit_s",
            minimum=0.0,
        )
        reference_t_cycle = self._action_ball_finite_float(
            receipt.reference_t_cycle_s,
            name="task.reference_t_cycle_s",
            minimum=0.0,
        )
        reference_speed = self._action_ball_finite_float(
            receipt.reference_racket_site_speed_mps,
            name="task.reference_racket_site_speed_mps",
            minimum=0.0,
        )
        required_speed = self._action_ball_finite_float(
            receipt.required_racket_site_speed_mps,
            name="task.required_racket_site_speed_mps",
            minimum=0.0,
        )
        teacher_rate = self._action_ball_finite_float(
            receipt.teacher_rate,
            name="task.teacher_rate",
            minimum=0.0,
        )
        teacher_rate_min = self._action_ball_finite_float(
            receipt.teacher_rate_min,
            name="task.teacher_rate_min",
            minimum=0.0,
        )
        teacher_rate_max = self._action_ball_finite_float(
            receipt.teacher_rate_max,
            name="task.teacher_rate_max",
            minimum=0.0,
        )
        scaled_t_hit = self._action_ball_finite_float(
            receipt.scaled_t_hit_s,
            name="task.scaled_t_hit_s",
            minimum=0.0,
        )
        scaled_t_cycle = self._action_ball_finite_float(
            receipt.scaled_t_cycle_s,
            name="task.scaled_t_cycle_s",
            minimum=0.0,
        )
        pre_swing_wait = self._action_ball_finite_float(
            receipt.pre_swing_wait_s,
            name="task.pre_swing_wait_s",
            minimum=0.0,
        )
        reaction_margin = self._action_ball_finite_float(
            receipt.reaction_margin_s,
            name="task.reaction_margin_s",
            minimum=0.0,
        )
        if (
            time_to_contact <= 0.0
            or reference_t_hit <= 0.0
            or reference_t_cycle <= reference_t_hit
            or reference_speed <= 0.0
            or required_speed <= 0.0
            or teacher_rate <= 0.0
            or teacher_rate_min <= 0.0
            or teacher_rate_max < teacher_rate_min
            or scaled_t_hit <= 0.0
            or scaled_t_cycle <= scaled_t_hit
        ):
            raise ValueError("action-ball task timing has a non-positive/order violation")
        if not teacher_rate_min <= 1.0 <= teacher_rate_max:
            raise ValueError(
                "action-ball certified teacher-rate bounds must contain native rate 1"
            )
        contact_geometry = runtime._contact_geometry
        try:
            contact_geometry.canonical_teacher_rate_from_site_speed(
                teacher_rate,
                1.0,
                teacher_rate_min,
                teacher_rate_max,
            )
            canonical_teacher_rate = (
                contact_geometry.canonical_teacher_rate_from_site_speed(
                    required_speed,
                    reference_speed,
                    teacher_rate_min,
                    teacher_rate_max,
                )
            )
        except contact_geometry.ExactFaceContactGeometryError as exc:
            raise ValueError(
                "action-ball teacher_rate is outside its certified range"
            ) from exc
        self._action_ball_close_float(
            teacher_rate,
            canonical_teacher_rate,
            name="task canonical teacher_rate",
        )
        required_vector = self._action_ball_vector(
            receipt.racket_site_velocity_w_mps,
            name="task.racket_site_velocity_w_mps",
            length=3,
        )
        self._action_ball_close_float(
            required_speed,
            math.sqrt(sum(value * value for value in required_vector)),
            name="task required racket-site speed",
        )
        # These are the O(1) relations Motion consumes.  Rechecking them is
        # intentionally cheap and prevents a forged frozen receipt from
        # retiming the teacher after pool/Racket admission.  It keeps the
        # vector/rate seams used by the full validator while still avoiding
        # canonical receipt serialization and resolver replay in this path.
        self._action_ball_close_float(
            teacher_rate,
            required_speed / reference_speed,
            name="task teacher_rate=required/reference",
        )
        self._action_ball_close_float(
            scaled_t_hit,
            reference_t_hit / teacher_rate,
            name="task scaled_t_hit_s",
        )
        self._action_ball_close_float(
            scaled_t_cycle,
            reference_t_cycle / teacher_rate,
            name="task scaled_t_cycle_s",
        )
        self._action_ball_close_float(
            pre_swing_wait,
            time_to_contact - scaled_t_hit,
            name="task pre_swing_wait_s",
        )
        if (
            pre_swing_wait + 1.0e-12 < reaction_margin
            or pre_swing_wait > 1.0 + 1.0e-12
        ):
            raise ValueError(
                "action-ball pre-swing wait violates reaction/one-second bounds"
            )

        runtime_episode_length = (
            int(self._env.max_episode_length) * float(self._env.step_dt)
        )
        if (
            pre_swing_wait
            + scaled_t_cycle
            + float(self._env.step_dt)
            > runtime_episode_length + 1.0e-12
        ):
            raise ValueError(
                "action-ball task cycle plus close tick exceeds runtime episode horizon"
            )
        native_cycle = (segment_length - 1) * float(self._env.step_dt)
        self._action_ball_close_float(
            reference_t_cycle,
            native_cycle,
            name="task reference_t_cycle_s vs admitted motion",
        )
        hit_frame = reference_t_hit / float(self._env.step_dt)
        self._action_ball_close_float(
            hit_frame,
            float(round(hit_frame)),
            name="task reference_t_hit_s policy-frame alignment",
        )
        if not 0 < round(hit_frame) < segment_length - 1:
            raise ValueError(
                "action-ball task hit frame is outside the admitted motion interior"
            )
        if pending_elapsed > pre_swing_wait + 1.0e-12:
            raise RuntimeError(
                "action-ball task arrived after its certified ready-wait ended"
            )
        return {
            "time_to_contact_s": time_to_contact,
            "teacher_rate": teacher_rate,
            "scaled_t_hit_s": scaled_t_hit,
            "scaled_t_cycle_s": scaled_t_cycle,
            "pre_swing_wait_s": pre_swing_wait,
            "pending_elapsed_s": pending_elapsed,
        }

    def _resolve_action_ball_task_timing_diagnostic_selected(
        self,
        *,
        host_identity_rows: tuple,
        receipts: tuple,
        task_refs: tuple,
    ) -> None:
        """Install one diagnostic timing batch without per-env device reads.

        Racket calls this only after validating and installing every issued
        receipt.  All identities, Motion-owned timing algebra/runtime
        constraints, buffer shapes, and tensor materialization complete before
        any Motion buffer changes.  Producer-owned vector/contact geometry was
        already validated before the frozen receipt entered this handoff.
        Because the diagnostic pool has already issued, any failure is terminal
        for that run and must never be caught and retried.
        Formal runs retain the opaque ref/resolver path below.
        """

        if (
            self._action_ball_birth_broker is None
            or not self._action_ball_birth_broker.diagnostic_fast_path
        ):
            raise RuntimeError(
                "direct action-ball task timing install is diagnostic-only"
            )
        if (
            type(host_identity_rows) is not tuple
            or type(receipts) is not tuple
            or type(task_refs) is not tuple
            or not host_identity_rows
            or len(receipts) != len(host_identity_rows)
            or len(task_refs) != len(host_identity_rows)
        ):
            raise ValueError(
                "diagnostic action-ball timing requires one non-empty aligned tuple batch"
        )
        if (
            self._action_ball_segment_lengths is None
            or self._action_ball_action_uids is None
            or len(self._action_ball_segment_lengths)
            != len(self._action_ball_action_uids)
        ):
            raise RuntimeError(
                "diagnostic action-ball timing lacks admitted segment lengths"
            )

        env_rows: list[int] = []
        device_rows: list[tuple[float, ...]] = []
        validated_refs: list[object] = []
        seen_envs: set[int] = set()
        step_dt = float(self._env.step_dt)
        for row_index, (identity, receipt, task_ref) in enumerate(
            zip(host_identity_rows, receipts, task_refs)
        ):
            if type(identity) is not tuple or len(identity) != 7:
                raise ValueError(
                    "diagnostic action-ball timing identity row must have seven fields"
                )
            (
                raw_env_id,
                raw_action_slot,
                raw_action_uid,
                raw_reset_generation,
                raw_swing_generation,
                raw_previous_swing_generation,
                active_before_install,
            ) = identity
            env_id = self._action_ball_plain_int(
                raw_env_id,
                name=f"host_identity_rows[{row_index}].env_id",
            )
            if env_id >= self.num_envs or env_id in seen_envs:
                raise ValueError(
                    "diagnostic action-ball timing env ids are out of range or repeated"
                )
            seen_envs.add(env_id)
            action_slot = self._action_ball_plain_int(
                raw_action_slot,
                name=f"host_identity_rows[{row_index}].action_slot",
            )
            if action_slot >= len(self._action_ball_action_uids):
                raise ValueError(
                    "diagnostic action-ball timing action slot is outside the manifest"
                )
            action_uid = self._action_ball_plain_int(
                raw_action_uid,
                name=f"host_identity_rows[{row_index}].action_uid",
                minimum=1,
            )
            reset_generation = self._action_ball_plain_int(
                raw_reset_generation,
                name=f"host_identity_rows[{row_index}].reset_generation",
                minimum=1,
            )
            swing_generation = self._action_ball_plain_int(
                raw_swing_generation,
                name=f"host_identity_rows[{row_index}].swing_generation",
            )
            previous_swing_generation = self._action_ball_plain_int(
                raw_previous_swing_generation,
                name=(
                    f"host_identity_rows[{row_index}]"
                    ".previous_swing_generation"
                ),
                minimum=-1,
            )
            if (
                swing_generation > 0
                and swing_generation != previous_swing_generation + 1
            ):
                raise ValueError(
                    "diagnostic action-ball wrap generation did not advance exactly once"
                )
            # Racket already consumed this flag while closing the prior attempt.
            # A true reset may legitimately replace either an active or an
            # inactive env, so Motion only preserves its exact boolean shape.
            if type(active_before_install) is not bool:
                raise ValueError(
                    "diagnostic action-ball timing active flag must be boolean"
                )
            if action_uid != self._action_ball_action_uids[action_slot]:
                raise ValueError(
                    "diagnostic action-ball timing action UID/slot binding changed"
                )

            pending_elapsed = 0.0 if swing_generation == 0 else step_dt
            timing = (
                self
                ._validate_action_ball_task_ref_and_receipt_diagnostic_prevalidated_host(
                    task_ref,
                    receipt,
                    env_id=env_id,
                    reset_generation=reset_generation,
                    swing_generation=swing_generation,
                    action_slot=action_slot,
                    segment_length=self._action_ball_segment_lengths[
                        action_slot
                    ],
                    pending_elapsed_s=pending_elapsed,
                )
            )
            env_rows.append(env_id)
            validated_refs.append(task_ref)
            device_rows.append(
                (
                    env_id,
                    timing["pending_elapsed_s"],
                    timing["time_to_contact_s"],
                    timing["teacher_rate"],
                    timing["scaled_t_hit_s"],
                    timing["scaled_t_cycle_s"],
                    timing["pre_swing_wait_s"],
                )
            )

        timing_buffers = (
            self._action_ball_task_pending_elapsed_s,
            self._action_ball_task_age_s,
            self._action_ball_time_to_contact_s,
            self._action_ball_teacher_rate,
            self._action_ball_scaled_t_hit_s,
            self._action_ball_scaled_t_cycle_s,
            self._action_ball_pre_swing_wait_s,
            self._action_ball_task_timing_active,
        )
        if (
            type(self._action_ball_active_task_refs) is not list
            or len(self._action_ball_active_task_refs) != self.num_envs
            or any(
                buffer is None
                or tuple(buffer.shape) != (self.num_envs,)
                for buffer in timing_buffers
            )
        ):
            raise RuntimeError(
                "diagnostic action-ball timing buffers are not fully bound"
            )
        pending_row_count = (
            self._action_ball_diagnostic_pending_row_count
        )
        if pending_row_count <= 0:
            raise RuntimeError(
                "diagnostic action-ball timing has no pending selected batch"
            )
        if len(env_rows) != pending_row_count:
            raise RuntimeError(
                "diagnostic action-ball timing selected row count does not "
                "match the pending row count"
            )

        # Tensor construction is still staging: no Motion state changes until
        # the complete host batch and its sole H2D payload exist.  Environment
        # ids are exactly representable in the float64 timing dtype and become
        # the indexed-write column on device.
        staged = torch.tensor(
            device_rows,
            dtype=self._action_ball_task_age_s.dtype,
            device=self.device,
        )
        if tuple(staged.shape) != (len(env_rows), 7):
            raise RuntimeError(
                "diagnostic action-ball timing staging shape changed"
            )
        ids = staged[:, 0].to(dtype=torch.long)
        values = staged[:, 1:]
        # The host packet length protects omissions; this same-device guard
        # protects a forged same-length substitution of an already-active row.
        # Diagnostic failures poison the process, so an async device assertion
        # preserves fail-closed semantics without reintroducing a host barrier.
        torch._assert_async(
            torch.all(~self._action_ball_task_timing_active[ids])
        )

        self._action_ball_task_pending_elapsed_s[ids] = values[:, 0]
        self._action_ball_task_age_s[ids] = values[:, 0]
        self._action_ball_time_to_contact_s[ids] = values[:, 1]
        self._action_ball_teacher_rate[ids] = values[:, 2]
        self._action_ball_scaled_t_hit_s[ids] = values[:, 3]
        self._action_ball_scaled_t_cycle_s[ids] = values[:, 4]
        self._action_ball_pre_swing_wait_s[ids] = values[:, 5]
        self._action_ball_task_timing_active[ids] = True
        for env_id, task_ref in zip(env_rows, validated_refs):
            self._action_ball_active_task_refs[env_id] = task_ref
        self._action_ball_diagnostic_pending_row_count -= len(env_rows)
        torch._assert_async(
            torch.all(
                self._action_ball_task_timing_active[
                    self._action_ball_reset_generation > 0
                ]
            )
        )

    def install_action_ball_task_timing_diagnostic_many(
        self,
        *,
        host_identity_rows: tuple,
        receipts: tuple,
        task_refs: tuple,
    ) -> None:
        """Compatibility wrapper for the diagnostic selected-batch resolver."""

        self._resolve_action_ball_task_timing_diagnostic_selected(
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=task_refs,
        )

    def _resolve_pending_action_ball_tasks(self) -> None:
        if self._action_ball_task_ref_for_env is None:
            raise RuntimeError("action-ball task ref authority is not bound")
        if (
            self._action_ball_birth_broker is not None
            and self._action_ball_birth_broker.diagnostic_fast_path
            and self._action_ball_diagnostic_pending_row_count == 0
        ):
            return
        pending_ids = torch.where(
            (self._action_ball_reset_generation > 0)
            & (~self._action_ball_task_timing_active)
        )[0]
        if len(pending_ids) == 0:
            if self._action_ball_birth_broker.diagnostic_fast_path:
                self._action_ball_diagnostic_pending_row_count = 0
            return
        for env_id in (
            int(value) for value in pending_ids.detach().cpu().tolist()
        ):
            task_ref = self._action_ball_task_ref_for_env(env_id)
            if task_ref is None:
                raise RuntimeError(
                    "action-ball Racket authority did not publish the current task ref"
                )
            receipt = self._action_ball_task_receipt_resolver(task_ref)
            timing = self._validate_action_ball_task_ref_and_receipt(
                task_ref, receipt, env_id=env_id
            )
            self._action_ball_active_task_refs[env_id] = task_ref
            self._action_ball_task_age_s[env_id] = timing[
                "pending_elapsed_s"
            ]
            self._action_ball_time_to_contact_s[env_id] = timing[
                "time_to_contact_s"
            ]
            self._action_ball_teacher_rate[env_id] = timing[
                "teacher_rate"
            ]
            self._action_ball_scaled_t_hit_s[env_id] = timing[
                "scaled_t_hit_s"
            ]
            self._action_ball_scaled_t_cycle_s[env_id] = timing[
                "scaled_t_cycle_s"
            ]
            self._action_ball_pre_swing_wait_s[env_id] = timing[
                "pre_swing_wait_s"
            ]
            self._action_ball_task_timing_active[env_id] = True
        if self._action_ball_birth_broker.diagnostic_fast_path:
            self._action_ball_diagnostic_pending_row_count = 0

    def resolve_action_ball_task_timing_now(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        diagnostic_host_identity_rows: tuple | None = None,
        diagnostic_receipts: tuple | None = None,
        diagnostic_task_refs: tuple | None = None,
    ) -> None:
        """Resolve newly published Racket receipts before reset observation.

        CommandManager resets Motion before Racket.  Without this handoff,
        formal rows remain locally pending until the following policy step and
        the first actor observation would report a false zero teacher-start
        clock.  Formal calls retain the opaque device-id resolver.  Diagnostic
        calls pass Racket's already-host-visible selected batch and install it
        through one H2D without advancing task age or teacher phase.
        """

        diagnostic_requested = any(
            value is not None
            for value in (
                diagnostic_host_identity_rows,
                diagnostic_receipts,
                diagnostic_task_refs,
            )
        )
        if diagnostic_requested:
            if env_ids is not None:
                raise ValueError(
                    "diagnostic action-ball timing selection is owned by host identity rows"
                )
            self._resolve_action_ball_task_timing_diagnostic_selected(
                host_identity_rows=diagnostic_host_identity_rows,
                receipts=diagnostic_receipts,
                task_refs=diagnostic_task_refs,
            )
            return
        if env_ids is None:
            raise ValueError(
                "formal action-ball timing resolution requires selected env ids"
            )
        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        self._resolve_pending_action_ball_tasks()
        if bool((~self._action_ball_task_timing_active[ids]).any()):
            raise RuntimeError(
                "action-ball task timing was not active before reset "
                "observation"
            )

    @property
    def action_ball_task_timing_active(self) -> torch.Tensor:
        if self._action_ball_task_timing_active is None:
            raise RuntimeError("action-ball task timing is not bound")
        return self._action_ball_task_timing_active

    @property
    def action_ball_time_to_contact_remaining_s(self) -> torch.Tensor:
        """Signed task deadline; inactive rows use a large fail-closed positive sentinel."""

        if self._action_ball_task_timing_active is None:
            raise RuntimeError("action-ball task timing is not bound")
        remaining = (
            self._action_ball_time_to_contact_s
            - self._action_ball_task_age_s
        )
        return torch.where(
            self._action_ball_task_timing_active,
            remaining,
            torch.full_like(remaining, 1.0e6),
        )

    @property
    def action_ball_pre_swing_wait_remaining_s(self) -> torch.Tensor:
        """Time until this row's teacher leaves its frozen ready frame.

        This is the exact live phase-governor clock, not a value reconstructed
        by the actor from time-to-contact, action identity and requested site
        speed.  Inactive rows expose zero; a valid ActionBall rollout resolves
        every row before the policy observation is consumed.
        """

        if (
            self._action_ball_task_timing_active is None
            or self._action_ball_pre_swing_wait_s is None
            or self._action_ball_task_age_s is None
        ):
            raise RuntimeError("action-ball task timing is not bound")
        remaining = torch.clamp(
            self._action_ball_pre_swing_wait_s
            - self._action_ball_task_age_s,
            min=0.0,
        )
        return torch.where(
            self._action_ball_task_timing_active,
            remaining,
            torch.zeros_like(remaining),
        )

    def _advance_action_ball_task_timing(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance receipt time analytically; return held and cycle-due-before masks."""

        # A task resolved during this command-manager compute has not driven a
        # physics tick yet.  Only rows that were already active on entry may
        # consume this update's elapsed policy interval.  WRAP carries one dt
        # in ``pending_elapsed_s`` because its replacement task is installed
        # later in the previous compute and does drive the intervening tick.
        active_before_resolve = self._action_ball_task_timing_active.clone()
        self._resolve_pending_action_ball_tasks()
        active = self._action_ball_task_timing_active
        if self._action_ball_birth_broker.diagnostic_fast_path:
            if self._action_ball_diagnostic_pending_row_count != 0:
                raise RuntimeError(
                    "diagnostic action-ball task timing remained unresolved"
                )
        elif bool(
            ((self._action_ball_reset_generation > 0) & ~active).any()
        ):
            raise RuntimeError("action-ball task timing remained unresolved")
        cycle_total = (
            self._action_ball_pre_swing_wait_s
            + self._action_ball_scaled_t_cycle_s
        )
        cycle_due_before = active & (
            self._action_ball_task_age_s + 1.0e-12 >= cycle_total
        )
        advancing = active_before_resolve & ~cycle_due_before
        self._action_ball_task_age_s[advancing] += float(
            self._env.step_dt
        )
        active_motion_s = torch.clamp(
            self._action_ball_task_age_s
            - self._action_ball_pre_swing_wait_s,
            min=0.0,
        )
        active_motion_s = torch.minimum(
            active_motion_s, self._action_ball_scaled_t_cycle_s
        )
        clip_starts = self.motion.seg_start[self.clip_id].to(
            dtype=torch.float64
        )
        phase_frames = (
            active_motion_s
            * self._action_ball_teacher_rate
            / float(self._env.step_dt)
        )
        final_frames = (
            self.motion.seg_len[self.clip_id] - 1
        ).to(dtype=torch.float64)
        phase_frames = torch.minimum(phase_frames, final_frames)
        self.time_steps_f.copy_(
            (clip_starts + phase_frames).to(self.time_steps_f.dtype)
        )
        rounded = self.time_steps_f.round().long()
        final_steps = (
            self.motion.seg_start[self.clip_id]
            + self.motion.seg_len[self.clip_id]
            - 1
        )
        self.time_steps.copy_(torch.minimum(rounded, final_steps))
        self.speed_scale.copy_(
            torch.where(
                active,
                self._action_ball_teacher_rate.to(self.speed_scale.dtype),
                torch.zeros_like(self.speed_scale),
            )
        )
        held = active & (active_motion_s <= 1.0e-12)
        self.hold_counter.copy_(held.to(self.hold_counter.dtype))
        self.metrics["in_hold"] = held.float()
        self.metrics["playback_speed"] = self.speed_scale.clone()
        self.metrics["action_ball_task_age_s"] = (
            self._action_ball_task_age_s.to(self.speed_scale.dtype)
        )
        self.metrics["action_ball_time_to_contact_s"] = (
            self.action_ball_time_to_contact_remaining_s.to(
                self.speed_scale.dtype
            )
        )
        self.metrics["action_ball_teacher_rate"] = (
            self._action_ball_teacher_rate.to(self.speed_scale.dtype)
        )
        self.metrics["action_ball_pre_swing_wait_remaining_s"] = (
            self.action_ball_pre_swing_wait_remaining_s.to(
                self.speed_scale.dtype
            )
        )
        return held, cycle_due_before

    def _write_canonical_ready_state(
        self,
        env_ids: torch.Tensor,
        *,
        action_ball_base_spawn_w_m: torch.Tensor | None = None,
        action_ball_base_quat_wxyz: torch.Tensor | None = None,
    ) -> dict | None:
        """Write one clip-owned ready transaction: root + 31 joints, all velocities zero.

        In action-ball mode the provider-issued birth owns the environment-local
        XYZ and supplies a yaw-only B_yaw frame for validation.  The physical
        root quaternion and joint pose both remain the selected opaque-admitted
        clip's literal ready state; a horizontal solver frame must never erase
        its real roll/pitch.  No task/base goal is accepted here.
        """

        ready_steps = self._canonical_ready_steps(env_ids)
        root_pos = self.motion.body_pos_w[ready_steps, 0] + self._env.scene.env_origins[env_ids]
        root_quat = self.motion.body_quat_w[ready_steps, 0]
        action_ball_write = action_ball_base_spawn_w_m is not None
        if action_ball_write != (action_ball_base_quat_wxyz is not None):
            raise ValueError(
                "action-ball root spawn and quaternion must be supplied together"
            )
        if action_ball_write:
            spawn = action_ball_base_spawn_w_m
            frame_quat = action_ball_base_quat_wxyz
            if (
                not torch.is_tensor(spawn)
                or tuple(spawn.shape) != (len(env_ids), 3)
                or not bool(torch.isfinite(spawn).all())
                or not torch.is_tensor(frame_quat)
                or tuple(frame_quat.shape) != (len(env_ids), 4)
                or not bool(torch.isfinite(frame_quat).all())
            ):
                raise ValueError(
                    "action-ball root must be finite [N,3] spawn + [N,4] yaw-frame tensors"
                )
            spawn = spawn.to(dtype=root_pos.dtype, device=root_pos.device)
            # The yaw-frame tensor was already checked against the selected
            # clip by _validate_action_ball_birth_receipt.  Convert here only
            # to fail on an incompatible device/dtype before mutating PhysX;
            # it is deliberately not the physical root quaternion.
            frame_quat = frame_quat.to(
                dtype=root_quat.dtype, device=root_quat.device
            )
            # ``base_spawn_w_m`` is an environment-local world-frame position, not an offset from
            # the historical clip root.  Replacing all XYZ is what makes the birth receipt the
            # sole physical translation truth.  Orientation stays literal to
            # the admitted per-action ready frame.
            root_pos = (
                self._env.scene.env_origins[env_ids].to(root_pos.dtype)
                + spawn
            )
        root_velocity = torch.zeros(
            len(env_ids), 6, dtype=root_pos.dtype, device=root_pos.device
        )
        root_state = torch.cat((root_pos, root_quat, root_velocity), dim=-1)
        joint_pos = self.motion.joint_pos[ready_steps]
        joint_vel = torch.zeros_like(joint_pos)

        diagnostic_fast_path = (
            action_ball_write
            and bool(
                getattr(
                    getattr(self, "_action_ball_birth_broker", None),
                    "diagnostic_fast_path",
                    False,
                )
            )
        )
        rollback_state = None
        if action_ball_write and not diagnostic_fast_path:
            # Isaac exposes separate setters.  Snapshot only for rollback; these live tensors are
            # never used to derive a birth.  All new payloads above came from admitted clip bytes
            # and the provider-issued receipt before the first simulator mutation.
            rollback_state = {
                "root_state": self.robot.data.root_state_w[env_ids].clone(),
                "joint_pos": self.robot.data.joint_pos[env_ids].clone(),
                "joint_vel": self.robot.data.joint_vel[env_ids].clone(),
            }
        dynamic_ready_enabled = (
            getattr(
                self,
                "_action_ball_dynamic_ready_binding_sha256",
                None,
            )
            is not None
        )
        if dynamic_ready_enabled and not action_ball_write:
            raise RuntimeError(
                "action_ball_dynamic_ready may install only inside an "
                "action-ball true-reset transaction"
            )
        dynamic_ready_action_term = (
            self._bind_action_ball_dynamic_ready_action_term()
            if dynamic_ready_enabled
            else None
        )
        try:
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(
                joint_pos, joint_vel, env_ids=env_ids
            )
            if dynamic_ready_action_term is not None:
                action_slots = self.clip_id[env_ids]
                if diagnostic_fast_path:
                    action_rollback_state = (
                        dynamic_ready_action_term
                        .install_action_ball_dynamic_ready_state(
                            env_ids,
                            self._action_ball_dynamic_ready_normalized_actor_action[
                                action_slots
                            ],
                            self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad[
                                action_slots
                            ],
                            capture_rollback=False,
                        )
                    )
                    if action_rollback_state is not None:
                        raise RuntimeError(
                            "diagnostic dynamic-ready install unexpectedly "
                            "returned rollback state"
                        )
                else:
                    action_rollback_state = (
                        dynamic_ready_action_term
                        .install_action_ball_dynamic_ready_state(
                            env_ids,
                            self._action_ball_dynamic_ready_normalized_actor_action[
                                action_slots
                            ],
                            self._action_ball_dynamic_ready_hold_qdes_joint_pos_rad[
                                action_slots
                            ],
                        )
                    )
                    rollback_state["action_state"] = action_rollback_state
        except Exception as exc:
            if rollback_state is not None:
                try:
                    self._restore_action_ball_sim_state(
                        env_ids, rollback_state
                    )
                except Exception as rollback_error:
                    raise RuntimeError(
                        "action-ball root write failed and simulator rollback failed"
                    ) from rollback_error
            raise
        return rollback_state

    def _restore_action_ball_sim_state(
        self, env_ids: torch.Tensor, rollback_state: dict
    ) -> None:
        legacy_keys = {"root_state", "joint_pos", "joint_vel"}
        if (
            type(rollback_state) is not dict
            or set(rollback_state) not in (
                legacy_keys,
                legacy_keys | {"action_state"},
            )
        ):
            raise RuntimeError("action-ball simulator rollback state is malformed")
        self.robot.write_root_state_to_sim(
            rollback_state["root_state"], env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            rollback_state["joint_pos"],
            rollback_state["joint_vel"],
            env_ids=env_ids,
        )
        if "action_state" in rollback_state:
            action_term = getattr(
                self, "_action_ball_dynamic_ready_action_term", None
            )
            if action_term is None:
                raise RuntimeError(
                    "action-ball rollback contains action state without its "
                    "dynamic-ready action term"
                )
            action_term.restore_action_ball_dynamic_ready_state(
                env_ids, rollback_state["action_state"]
            )

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def clip_family_is_forehand(self) -> torch.Tensor:
        """[num_segments] bool 表:True = 该 clip 属正手家族(False = 反手)。

        人话:配了 clip_family_per_clip 用配置表(__init__ 已整表校验);没配按老规矩懒推导——
        单 clip 当正手、恰好 2 clip = (正手, 反手),和写死的 ``clips == 0`` 判断逐字节同值;
        推不出(≥3 clip 没配表)当场报错,绝不猜(见 resolve_clip_family_is_forehand)。
        """
        if self._clip_family_is_forehand_t is None:
            self._clip_family_is_forehand_t = torch.tensor(
                resolve_clip_family_is_forehand(
                    getattr(self.cfg, "clip_family_per_clip", None),
                    int(self.motion.num_segments),
                ),
                dtype=torch.bool,
                device=self.device,
            )
        return self._clip_family_is_forehand_t

    @property
    def in_hold(self) -> torch.Tensor:
        """Bool mask for the *current control step's* pre-swing hold.

        ``_update_command`` snapshots ``held`` and then decrements ``hold_counter``.  Looking only
        at the post-decrement counter made the final frozen-reference step appear unheld to
        rewards/terminations (an off-by-one reference death at release).  The metric stores that
        snapshot; OR it with the counter so the contract is also correct immediately after a
        reset/wrap resample, before the next update.
        """
        counter_hold = self.hold_counter > 0
        metric_hold = self.metrics.get("in_hold")
        return counter_hold if metric_hold is None else (counter_hold | metric_hold.bool())

    @property
    def event_timing_enabled(self) -> bool:
        return self._event_scheduler is not None

    @property
    def event_just_installed(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._event_scheduler.event_just_installed

    @property
    def event_installed(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._event_scheduler.row_installed

    @property
    def event_exact_strike_allowed(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return self._event_scheduler.exact_strike_allowed

    @property
    def event_deadline_ticks_remaining(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return self._event_scheduler.deadline_ticks_remaining

    @property
    def event_current_clip_id(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return self._event_scheduler.current_clip_id

    @property
    def event_current_bank_row(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return self._event_scheduler.current_bank_row

    @property
    def event_schedule(self):
        return self._event_schedule

    def event_timing_hard_contract(self) -> dict:
        """Stable timing facts embedded in every checkpoint contract."""

        if self._event_schedule is None:
            return {"mode": EVENT_TIMING_MODE_DISABLED}
        return {
            "mode": EVENT_TIMING_MODE_POST_STRIKE_T1,
            "schedule": self._event_schedule.hard_contract(),
            "sequence_assignment": "env_id_mod_sequence_count_v1",
            "repeat_within_episode": False,
            "clock_origin": "accepted_exact_strike_opportunity",
            "install_trigger": "immutable_post_strike_reveal_tick",
            "deadline_origin": "previous_scheduled_deadline_after_first_origin",
            "deadline_shift_allowed": False,
            "miss_consumes_opportunity": True,
            "carry_state": True,
            "reset_robot_or_last_action_on_install": False,
            "reset_history_or_noise_on_install": False,
            "event_playback": "native_clip_start_plus_exact_hold_no_retime",
        }

    def bind_event_native_strike_ticks(
        self, native_strike_ticks_by_clip: Sequence[int] | torch.Tensor
    ) -> None:
        """Bind RacketTargetCommand's audited per-clip strike frames exactly once."""

        if self._event_scheduler is None:
            return
        raw = torch.as_tensor(native_strike_ticks_by_clip, device=self.device)
        if raw.dtype == torch.bool or raw.is_floating_point() or raw.is_complex():
            raise ValueError("event native strike ticks must use an integer dtype")
        values = raw.to(dtype=torch.long).reshape(-1)
        if len(values) != int(self.motion.num_segments) or torch.any(values <= 0):
            raise ValueError(
                "event native strike timing must contain one positive offset per motion clip"
            )
        if self._event_native_strike_ticks is not None:
            if not torch.equal(self._event_native_strike_ticks, values):
                raise RuntimeError("event native strike timing was rebound with different values")
            return
        self._event_native_strike_ticks = values.clone()

    def record_event_exact_strike(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return self._event_scheduler.record_exact_strike(env_ids)

    def finalize_event_deadlines(self) -> torch.Tensor:
        if self._event_scheduler is None:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return self._event_scheduler.finalize_deadlines()

    def planner_revision_hard_contract(self) -> dict | None:
        """Return the complete runtime contract, or ``None`` for the byte-identical OFF path."""

        if not self.planner_revision_enabled:
            return None
        profile = self._planner_revision_profile
        if profile is None:
            raise RuntimeError("planner revision enabled without a validated profile")
        initial_tts = tuple(
            float(value) for value in self.cfg.planner_revision_initial_tts_range_s
        )
        return {
            "enabled": True,
            "revision_schema_version": PLANNER_TASK_REVISION_SCHEMA_VERSION,
            "governor": profile.hard_contract(),
            "initial_tts_range_s": list(initial_tts),
        }

    def planner_revision_training_hard_contract(self) -> dict | None:
        """Return canonical training-only planner facts from the validated runtime objects.

        Hydra may retain mapping-shaped values as ``DictConfig`` instances.  The generic legacy
        hard-contract converter intentionally preserves its historical behavior, so feeding the
        raw config through it would serialize a mapping as a list of keys.  The parsed
        ``InitialTtsMixture`` is already the runtime authority; publishing its canonical document
        here keeps the producer and validator on one representation without changing any legacy
        OFF-path contract bytes.
        """

        if not self.planner_revision_enabled:
            return None
        mixture = self._planner_initial_tts_mixture
        if mixture is None:
            raise RuntimeError(
                "planner revision enabled without a validated initial-TTS mixture"
            )
        return {"initial_tts_mixture": mixture.document()}

    def begin_planner_task(
        self,
        env_ids: torch.Tensor,
        *,
        control_epoch: torch.Tensor,
        task_id: torch.Tensor,
        strike_step: torch.Tensor,
        initial_tts: torch.Tensor,
        target_position: torch.Tensor,
        target_velocity: torch.Tensor,
        target_normal: torch.Tensor,
    ) -> None:
        """Install one new immutable physical-ball identity for each selected environment."""

        if not self.planner_revision_enabled:
            raise RuntimeError("begin_planner_task called while planner revisions are disabled")
        profile = self._planner_revision_profile
        if profile is None:
            raise RuntimeError("planner revision profile is unavailable")
        ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if len(ids) == 0:
            return
        epoch = control_epoch.to(device=self.device, dtype=torch.long).reshape(-1)
        tasks = task_id.to(device=self.device, dtype=torch.long).reshape(-1)
        strike = strike_step.to(device=self.device, dtype=torch.float32).reshape(-1)
        raw_tts = initial_tts.to(device=self.device, dtype=torch.float32).reshape(-1)
        tts = self._planner_canonicalize_tts(raw_tts, profile)
        pos = target_position.to(device=self.device, dtype=torch.float32)
        vel = target_velocity.to(device=self.device, dtype=torch.float32)
        normal = target_normal.to(device=self.device, dtype=torch.float32)
        start = self.time_steps_f[ids]
        normal_norm = torch.linalg.vector_norm(normal, dim=-1)
        minimum_tts = self._planner_minimum_finish_time(
            torch.ones_like(tts),
            torch.zeros_like(tts),
            profile.max_phase_rate_per_s,
            profile.max_phase_acceleration_per_s2,
        )
        valid = (
            (epoch > 0)
            & (tasks > 0)
            & torch.isfinite(strike)
            & (strike > start)
            & torch.isfinite(raw_tts)
            & (raw_tts + profile.early_deadline_tolerance_s >= profile.min_tts_s)
            & (raw_tts - profile.early_deadline_tolerance_s <= profile.max_tts_s)
            & (tts + profile.early_deadline_tolerance_s >= minimum_tts)
            & torch.isfinite(pos).all(dim=-1)
            & torch.isfinite(vel).all(dim=-1)
            & torch.isfinite(normal).all(dim=-1)
            & ((normal_norm - 1.0).abs() <= profile.normal_unit_tolerance)
        )
        if not bool(valid.all()):
            raise ValueError("begin_planner_task received an invalid or partial atomic task tuple")
        self._planner_active[ids] = True
        self._planner_control_epoch[ids] = epoch
        self._planner_task_id[ids] = tasks
        self._planner_task_revision[ids] = 1
        self._planner_start_step[ids] = start
        self._planner_strike_step[ids] = strike
        self._planner_phase_rate[ids] = 0.0
        self._planner_slow_only_next[ids] = False
        self._planner_desired_tts[ids] = tts
        self._planner_begin_tts[ids] = tts
        self._planner_truth_tts[ids] = tts
        self._planner_truth_tts_signed[ids] = tts
        self._planner_begin_target_pos[ids] = pos
        self._planner_begin_target_vel[ids] = vel
        self._planner_begin_target_normal[ids] = normal
        self.speed_scale[ids] = 0.0

    @staticmethod
    def _planner_canonicalize_tts(
        tts: torch.Tensor, profile: PhaseGovernorProfile
    ) -> torch.Tensor:
        """Snap only the profile-bound float32 edge bands to their canonical values."""

        tolerance = profile.early_deadline_tolerance_s
        minimum = torch.full_like(tts, profile.min_tts_s)
        maximum = torch.full_like(tts, profile.max_tts_s)
        snapped = torch.where((tts - minimum).abs() <= tolerance, minimum, tts)
        return torch.where((snapped - maximum).abs() <= tolerance, maximum, snapped)

    @staticmethod
    def _planner_minimum_finish_time(
        distance: torch.Tensor,
        initial_rate: torch.Tensor,
        maximum_rate: float,
        maximum_acceleration: float,
    ) -> torch.Tensor:
        """Vector form of planner_revision._minimum_finish_time."""

        rate = initial_rate.clamp(min=0.0, max=maximum_rate)
        accelerate_time = (maximum_rate - rate).clamp(min=0.0) / maximum_acceleration
        accelerate_distance = (
            rate * accelerate_time + 0.5 * maximum_acceleration * accelerate_time.square()
        )
        triangular = (-rate + torch.sqrt(
            (rate.square() + 2.0 * maximum_acceleration * distance.clamp(min=0.0))
        )) / maximum_acceleration
        trapezoidal = accelerate_time + (
            distance - accelerate_distance
        ).clamp(min=0.0) / maximum_rate
        return torch.where(distance <= accelerate_distance, triangular, trapezoidal)

    def submit_planner_revision(
        self,
        env_ids: torch.Tensor,
        *,
        control_epoch: torch.Tensor,
        task_id: torch.Tensor,
        task_revision: torch.Tensor,
        desired_tts: torch.Tensor,
        target_position: torch.Tensor,
        target_velocity: torch.Tensor,
        target_normal: torch.Tensor,
    ) -> torch.Tensor:
        """Atomically accept/reject same-task revisions; rejected rows preserve the old ledger."""

        if not self.planner_revision_enabled:
            raise RuntimeError("submit_planner_revision called while planner revisions are disabled")
        profile = self._planner_revision_profile
        if profile is None:
            raise RuntimeError("planner revision profile is unavailable")
        ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if len(ids) == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        epoch = control_epoch.to(device=self.device, dtype=torch.long).reshape(-1)
        tasks = task_id.to(device=self.device, dtype=torch.long).reshape(-1)
        revisions = task_revision.to(device=self.device, dtype=torch.long).reshape(-1)
        raw_tts = desired_tts.to(device=self.device, dtype=torch.float32).reshape(-1)
        tts = self._planner_canonicalize_tts(raw_tts, profile)
        pos = target_position.to(device=self.device, dtype=torch.float32)
        vel = target_velocity.to(device=self.device, dtype=torch.float32)
        normal = target_normal.to(device=self.device, dtype=torch.float32)
        normal_norm = torch.linalg.vector_norm(normal, dim=-1)
        # Envelope the proposed *absolute* deadline relative to the immutable task-begin deadline.
        # The visible-to-proposed delta is still needed only for the one-step slow-only rule.  These
        # are deliberately separate: a latest-value mailbox may skip revisions, but every accepted
        # snapshot must remain inside the same begin-bound envelope.
        elapsed_since_begin = (
            self._planner_begin_tts[ids] - self._planner_truth_tts[ids]
        ).clamp(min=0.0)
        deadline_delta_from_begin = (
            elapsed_since_begin + tts - self._planner_begin_tts[ids]
        )
        deadline_delta_from_visible = tts - self._planner_desired_tts[ids]
        span = (self._planner_strike_step[ids] - self._planner_start_step[ids]).clamp(min=1.0e-6)
        phase = ((self.time_steps_f[ids] - self._planner_start_step[ids]) / span).clamp(0.0, 1.0)
        minimum_tts = self._planner_minimum_finish_time(
            1.0 - phase,
            self._planner_phase_rate[ids],
            profile.max_phase_rate_per_s,
            profile.max_phase_acceleration_per_s2,
        )
        valid = (
            self._planner_active[ids]
            & (self.time_steps_f[ids] < self._planner_strike_step[ids])
            & (epoch == self._planner_control_epoch[ids])
            & (tasks == self._planner_task_id[ids])
            & (revisions > self._planner_task_revision[ids])
            & torch.isfinite(raw_tts)
            & (raw_tts + profile.early_deadline_tolerance_s >= profile.min_tts_s)
            & (raw_tts - profile.early_deadline_tolerance_s <= profile.max_tts_s)
            & (
                deadline_delta_from_begin.abs()
                <= profile.max_deadline_revision_delta_s
            )
            & (tts + profile.early_deadline_tolerance_s >= minimum_tts)
            & torch.isfinite(pos).all(dim=-1)
            & torch.isfinite(vel).all(dim=-1)
            & torch.isfinite(normal).all(dim=-1)
            & ((normal_norm - 1.0).abs() <= profile.normal_unit_tolerance)
            & (
                torch.linalg.vector_norm(
                    pos - self._planner_begin_target_pos[ids], dim=-1
                )
                <= profile.max_position_revision_delta_m
            )
            & (
                torch.linalg.vector_norm(
                    vel - self._planner_begin_target_vel[ids], dim=-1
                )
                <= profile.max_velocity_revision_delta_mps
            )
            & (
                torch.acos(
                    (normal * self._planner_begin_target_normal[ids])
                    .sum(dim=-1)
                    .clamp(-1.0, 1.0)
                )
                <= profile.max_normal_revision_delta_rad
            )
        )
        accepted_ids = ids[valid]
        if len(accepted_ids) > 0:
            self._planner_task_revision[accepted_ids] = revisions[valid]
            self._planner_desired_tts[accepted_ids] = tts[valid]
            self._planner_slow_only_next[accepted_ids] = (
                deadline_delta_from_visible[valid]
                > profile.early_deadline_tolerance_s
            )
        # CommandTerm.reset() indexes every metric with GLOBAL environment ids. ``valid`` is
        # intentionally compact (one row per currently eligible environment), so rebinding either
        # metric to ``valid.float()`` corrupts the mandatory [num_envs] shape as soon as the first
        # short-preparation task leaves the pre-contact set. Keep the registered per-env buffers
        # stable and scatter the compact decision back through its original ids.
        self.metrics["planner_revision_accepted"][ids] = valid.float()
        self.metrics["planner_revision_rejected"][ids] = (~valid).float()
        return valid

    def _advance_planner_phase(self, held: torch.Tensor) -> torch.Tensor:
        """Advance active planner-owned clocks and return their exact clip-frame delta."""

        profile = self._planner_revision_profile
        if profile is None:
            raise RuntimeError("planner revision profile is unavailable")
        # These are per-step indicators, not held episode metrics. Clearing the full registered
        # tensors here also covers a step with no eligible revision submission; submit() then
        # scatters only the environments that actually attempted a revision.
        self.metrics["planner_revision_accepted"].zero_()
        self.metrics["planner_revision_rejected"].zero_()
        active = self._planner_active
        dt = profile.policy_dt_s
        self._planner_truth_tts[active] = (
            self._planner_truth_tts[active] - dt
        ).clamp(min=0.0)
        # 孪生时钟不截断:触球后转负,供击球窗掩码在 +window 处如约关闭。
        self._planner_truth_tts_signed[active] = (
            self._planner_truth_tts_signed[active] - dt
        )
        remaining_deadline = (self._planner_desired_tts - dt).clamp(min=0.0)
        span = (self._planner_strike_step - self._planner_start_step).clamp(min=1.0e-6)
        phase = ((self.time_steps_f - self._planner_start_step) / span).clamp(0.0, 1.0)
        prestrike = active & (phase < 1.0)
        requested = torch.where(
            remaining_deadline > profile.early_deadline_tolerance_s,
            (1.0 - phase) / remaining_deadline.clamp(min=dt),
            torch.full_like(phase, profile.max_phase_rate_per_s),
        ).clamp(min=0.0, max=profile.max_phase_rate_per_s)
        # Mirror planner_revision.advance_phase / PpPhaseGovernor::Advance exactly.  Near the
        # reachability boundary, dividing remaining phase by the nominal deadline under-requests
        # the rate because it ignores the acceleration ramp; force the cap before applying a
        # one-step slow-only deadline extension.  Without this branch training and deployment
        # diverge specifically on the short-preparation cases this curriculum is meant to expose.
        earliest = self._planner_minimum_finish_time(
            (1.0 - phase).clamp(min=0.0),
            self._planner_phase_rate,
            profile.max_phase_rate_per_s,
            profile.max_phase_acceleration_per_s2,
        )
        requested = torch.where(
            remaining_deadline <= earliest + dt,
            torch.full_like(requested, profile.max_phase_rate_per_s),
            requested,
        )
        requested = torch.where(
            self._planner_slow_only_next,
            torch.minimum(requested, self._planner_phase_rate),
            requested,
        )
        max_delta = profile.max_phase_acceleration_per_s2 * dt
        rate_delta = (requested - self._planner_phase_rate).clamp(
            min=-max_delta, max=max_delta
        )
        new_rate = (self._planner_phase_rate + rate_delta).clamp(
            min=0.0, max=profile.max_phase_rate_per_s
        )
        # Once contact is reached, smoothly return to the native one-frame clock for follow-through.
        native_rate = (1.0 / (span * dt)).clamp(
            max=profile.max_phase_rate_per_s
        )
        post_delta = (native_rate - self._planner_phase_rate).clamp(
            min=-max_delta, max=max_delta
        )
        new_rate = torch.where(
            prestrike,
            new_rate,
            (self._planner_phase_rate + post_delta).clamp(min=0.0),
        )
        new_rate = torch.where(held & active, torch.zeros_like(new_rate), new_rate)
        frame_delta = 0.5 * (self._planner_phase_rate + new_rate) * dt * span
        remaining_frames = (self._planner_strike_step - self.time_steps_f).clamp(min=0.0)
        frame_delta = torch.where(prestrike, torch.minimum(frame_delta, remaining_frames), frame_delta)
        # Keep one full actor interval in reserve whenever the task still has
        # positive time after this update.  The racket command runs later in
        # the same command-manager step, so this is what leaves the final
        # policy_dt target/TTS revision pre-contact and actor-visible.  The
        # following step has remaining_deadline==0 and may reach contact.
        next_rate = torch.minimum(
            torch.full_like(new_rate, profile.max_phase_rate_per_s),
            new_rate + max_delta,
        )
        reserved_phase_distance = 0.5 * (new_rate + next_rate) * dt
        precontact_delta_cap = (
            remaining_frames - reserved_phase_distance * span
        ).clamp(min=0.0)
        reserve_last_actor_interval = prestrike & (
            remaining_deadline > profile.early_deadline_tolerance_s
        )
        frame_delta = torch.where(
            reserve_last_actor_interval,
            torch.minimum(frame_delta, precontact_delta_cap),
            frame_delta,
        )
        frame_delta = torch.where(active, frame_delta.clamp(min=0.0), torch.zeros_like(frame_delta))
        self._planner_phase_rate = torch.where(active, new_rate, self._planner_phase_rate)
        self._planner_slow_only_next[active] = False
        self._planner_desired_tts[active] = remaining_deadline[active]
        self.metrics["planner_phase_rate_per_s"] = self._planner_phase_rate.clone()
        self.metrics["planner_truth_tts_s"] = self._planner_truth_tts.clone()
        return frame_delta

    def _install_event_motion(self, step) -> None:
        """Install clip/start/hold only; carry all physical and policy state across the event."""

        ids = step.install_env_ids
        if len(ids) == 0:
            return
        clips = step.install_clip_ids
        holds = step.install_hold_steps
        # Deliberately no _resample_command, adaptive sampling, simulator write, action write,
        # history reset, or teleport here.  The current robot state and last action continue.
        self._require_canonical_ready_boundary(ids, "event motion install")
        self.clip_id[ids] = clips
        starts = self.motion.seg_start[clips]
        self.time_steps[ids] = starts
        self.time_steps_f[ids] = starts.float()
        self.speed_scale[ids] = 1.0
        self.hold_counter[ids] = holds
        self.metrics["in_hold"][ids] = (holds > 0).float()
        if hasattr(self, "time_left"):
            self.time_left[ids] = float("inf")

    @property
    def joint_pos(self) -> torch.Tensor:
        # HOLD imitates the READY STAND, not the windup crouch (2026-07-05, pragmatic
        # P2.0): clip frame 0 is an asymmetric mid-crouch (knee 0.62/0.52 vs stand 0.25,
        # left hip_roll +0.14) — imitating it all hold long produced the splayed-feet
        # crouch-stand seen in Gate 2.5/3. During hold the joint reference is the
        # default stand pose; the release (stand -> windup) is exactly the trained
        # stand_start transition. C++ mirrors this (pp_policy: refs.joint_pos =
        # default_q at level 0) — keep them in lockstep.
        if self.canonical_ready_mode:
            return self.motion.joint_pos[self._pose_reference_steps()]
        jp = self.motion.joint_pos[self.time_steps]
        dq = self.robot.data.default_joint_pos
        return torch.where(self.in_hold[:, None], dq, jp)

    @property
    def joint_vel(self) -> torch.Tensor:
        # HOLD = a STATIONARY reference (2026-07-05): clip frame 0 is a mid-crouch
        # TRANSIENT (knee +7.8 rad/s, torso -1.11 m/s DOWN in the hopex clips). Feeding
        # its raw velocities through the whole hold taught the policy to fight a phantom
        # squat at soft gains and made "sink slowly" the velocity-reward optimum — the
        # AGI-sim / hardware bare-hold fall (Gate 2.5 P2, 3-5 s tip). A frozen reference
        # is not moving: zero its velocities on held envs. The C++ runner mirrors this
        # (pp_policy zeroes refs.joint_vel in its hold states) — keep them in lockstep.
        jv = self.motion.joint_vel[self.time_steps]
        # R14: at playback speed s the reference joints traverse the same poses s× as fast.
        if self.retiming_active:
            jv = jv * self.speed_scale[:, None]
        return torch.where(self.in_hold[:, None], torch.zeros_like(jv), jv)

    @property
    def body_pos_w(self) -> torch.Tensor:
        steps = self._pose_reference_steps()
        return self.motion.body_pos_w[steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._pose_reference_steps()]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        # Zeroed during hold — see joint_vel. Un-gated motion_body_lin_vel otherwise
        # pays for tracking frame-0's -1.11 m/s DOWNWARD torso velocity all hold long.
        # R14 retiming composes: scale by playback speed first, then hold-zero wins.
        v = self.motion.body_lin_vel_w[self.time_steps]
        if self.retiming_active:
            v = v * self.speed_scale[:, None, None]
        return torch.where(self.in_hold[:, None, None], torch.zeros_like(v), v)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        v = self.motion.body_ang_vel_w[self.time_steps]
        if self.retiming_active:
            v = v * self.speed_scale[:, None, None]
        return torch.where(self.in_hold[:, None, None], torch.zeros_like(v), v)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        steps = self._pose_reference_steps()
        return self.motion.body_pos_w[steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[
            self._pose_reference_steps(), self.motion_anchor_body_index
        ]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        alv = self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
        if self.retiming_active:
            alv = alv * self.speed_scale[:, None]
        if self.canonical_ready_mode:
            alv = torch.where(self.in_hold[:, None], torch.zeros_like(alv), alv)
        return alv

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        aav = self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
        if self.retiming_active:
            aav = aav * self.speed_scale[:, None]
        if self.canonical_ready_mode:
            aav = torch.where(self.in_hold[:, None], torch.zeros_like(aav), aav)
        return aav

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        anchor_pos_err = self.anchor_pos_w - self.robot_anchor_pos_w
        anchor_rot_err = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        anchor_lin_vel_err = self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w
        anchor_ang_vel_err = self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w

        self.metrics["error_anchor_pos"] = torch.norm(anchor_pos_err, dim=-1)
        self.metrics["error_anchor_rot"] = anchor_rot_err
        self.metrics["error_anchor_lin_vel"] = torch.norm(anchor_lin_vel_err, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(anchor_ang_vel_err, dim=-1)
        self.metrics["error_anchor_rot_deg"] = anchor_rot_err * (180.0 / math.pi)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        joint_pos_err = self.joint_pos - self.robot_joint_pos
        joint_vel_err = self.joint_vel - self.robot_joint_vel
        self.metrics["error_joint_pos"] = torch.norm(joint_pos_err, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(joint_vel_err, dim=-1)
        self.metrics["error_joint_pos_mean_abs"] = torch.mean(torch.abs(joint_pos_err), dim=-1)
        self.metrics["error_joint_pos_max_abs"] = torch.max(torch.abs(joint_pos_err), dim=-1).values
        self.metrics["error_joint_vel_mean_abs"] = torch.mean(torch.abs(joint_vel_err), dim=-1)
        self.metrics["error_joint_vel_max_abs"] = torch.max(torch.abs(joint_vel_err), dim=-1).values

        # Log anchor states in an env-origin-relative frame so cross-env averages remain meaningful.
        anchor_ref_rel = self.anchor_pos_w - self._env.scene.env_origins
        anchor_robot_rel = self.robot_anchor_pos_w - self._env.scene.env_origins
        for axis_idx, axis in enumerate(("x", "y", "z")):
            self.metrics[f"reference_anchor_pos_{axis}"] = anchor_ref_rel[:, axis_idx]
            self.metrics[f"robot_anchor_pos_{axis}"] = anchor_robot_rel[:, axis_idx]
            self.metrics[f"reference_anchor_lin_vel_{axis}"] = self.anchor_lin_vel_w[:, axis_idx]
            self.metrics[f"robot_anchor_lin_vel_{axis}"] = self.robot_anchor_lin_vel_w[:, axis_idx]

        self.metrics["reference_anchor_speed"] = torch.norm(self.anchor_lin_vel_w, dim=-1)
        self.metrics["robot_anchor_speed"] = torch.norm(self.robot_anchor_lin_vel_w, dim=-1)
        if self._multiseg:
            seg_start = self.motion.seg_start[self.clip_id]
            seg_len = self.motion.seg_len[self.clip_id].clamp(min=2)
            self.metrics["motion_phase"] = (self.time_steps - seg_start).float() / (seg_len - 1).float()
        else:
            self.metrics["motion_phase"] = self.time_steps.float() / max(self.motion.time_step_total - 1, 1)

    def _action_ball_select_or_rewind_action(
        self, env_ids: Sequence[int]
    ) -> None:
        """Select one episode action, or rewind the frozen action at a natural wrap."""

        n = len(env_ids)
        if n == 0:
            return
        ids = (
            env_ids
            if torch.is_tensor(env_ids)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        ids = ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if self._resampling_from_wrap:
            # An action-ball episode has exactly one action identity.  A natural clip wrap starts
            # a new ball/swing against the same birth; it is never a selector opportunity.
            selected = self.clip_id[ids].clone()
        elif int(self.motion.num_segments) == 1:
            selected = torch.zeros(n, dtype=torch.long, device=self.device)
            self.clip_id[ids] = selected
        else:
            if self._balanced_clip_sampler is not None:
                selected = self._balanced_clip_sampler.sample(n)
            else:
                selected = torch.randint(
                    0, int(self.motion.num_segments), (n,), device=self.device
                )
            self.clip_id[ids] = selected

        starts = self.motion.seg_start[selected]
        self.time_steps[ids] = starts
        self.time_steps_f[ids] = starts.float()
        if self._action_ball_task_ref_for_env is not None:
            # The selected task receipt will install teacher_rate after Racket solves this swing.
            # Do not consume generic retiming RNG, even when its configured range is [1, 1].
            self.speed_scale[ids] = 0.0
        elif self.retiming_active:
            if self._speed_per_clip is not None:
                self.speed_scale[ids] = self._speed_per_clip[selected]
            else:
                speed_lo, speed_hi = self.cfg.speed_scale_range
                self.speed_scale[ids] = sample_uniform(
                    float(speed_lo), float(speed_hi), (n,), device=self.device
                )

        counts = torch.bincount(
            self.clip_id, minlength=int(self.motion.num_segments)
        ).float()
        probabilities = counts / counts.sum().clamp(min=1.0)
        entropy = -(
            probabilities * (probabilities + 1.0e-12).log()
        ).sum()
        self.metrics["sampling_entropy"][:] = entropy / math.log(
            max(int(self.motion.num_segments), 2)
        )
        top_probability, top_index = probabilities.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = top_probability
        self.metrics["sampling_top1_bin"][:] = (
            top_index.float() / max(int(self.motion.num_segments), 1)
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self._action_ball_birth_broker is not None:
            self._action_ball_select_or_rewind_action(env_ids)
            return
        if self._multiseg:
            # HITTER unified policy: each new swing uniformly samples the swing TYPE (clip) and starts at
            # that clip's first frame (reference-state-init at the swing start). The adaptive failure-bin
            # curriculum is single-clip BeyondMimic machinery and is bypassed here.
            n = len(env_ids)
            if n > 0:
                if self._balanced_clip_sampler is not None:
                    new_clip = self._balanced_clip_sampler.sample(n)
                else:
                    new_clip = torch.randint(0, self.motion.num_segments, (n,), device=self.device)
                self.clip_id[env_ids] = new_clip
                # R-c(i) rsi_skip_settle_frames: enter every swing N frames past the clip start —
                # the v5 clips carry a 3-4 frame IK cold-start transient at frame 0 (7.4-15.9 rad/s
                # phantom joint velocities). Wraps go through this same path, so the reference is
                # live-trimmed for the whole run, not only at RSI births. Clamped to the clip's
                # last frame so a short clip can never index out of its segment. 0 (default) = off.
                _skip = int(getattr(self.cfg, "rsi_skip_settle_frames", 0))
                if _skip > 0:
                    self.time_steps[env_ids] = torch.minimum(
                        self.motion.seg_start[new_clip] + _skip,
                        self.motion.seg_start[new_clip] + self.motion.seg_len[new_clip] - 1,
                    )
                else:
                    self.time_steps[env_ids] = self.motion.seg_start[new_clip]
                if self.retiming_active:
                    # R14: re-base the float clock and draw this swing's playback speed.
                    self.time_steps_f[env_ids] = self.time_steps[env_ids].float()
                    if self._speed_per_clip is not None:
                        self.speed_scale[env_ids] = self._speed_per_clip[new_clip]
                    else:
                        s_lo, s_hi = self.cfg.speed_scale_range
                        self.speed_scale[env_ids] = sample_uniform(float(s_lo), float(s_hi), (n,), device=self.device)
            # Report the REAL clip-sampling distribution (repurpose the bin-sampling metrics for clips):
            # entropy of the per-clip env fraction (1.0 = balanced), and the most-sampled clip + its share.
            counts = torch.bincount(self.clip_id, minlength=self.motion.num_segments).float()
            probs = counts / counts.sum().clamp(min=1.0)
            H = -(probs * (probs + 1e-12).log()).sum()
            self.metrics["sampling_entropy"][:] = H / math.log(max(self.motion.num_segments, 2))
            pmax, imax = probs.max(dim=0)
            self.metrics["sampling_top1_prob"][:] = pmax
            self.metrics["sampling_top1_bin"][:] = imax.float() / max(self.motion.num_segments, 1)
            return
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        # R-c(i) rsi_skip_settle_frames (single-clip path): clamp the sampled entry frame to >= N,
        # so the failure-adaptive sampler can never place a birth on the frame-0 IK transient
        # ("越摔越采"的止血). Guarded against clips shorter than N. 0 (default) = off.
        _skip = int(getattr(self.cfg, "rsi_skip_settle_frames", 0))
        if _skip > 0:
            self.time_steps[env_ids] = self.time_steps[env_ids].clamp(
                min=min(_skip, max(int(self.motion.time_step_total) - 1, 0))
            )
        if self.retiming_active:
            # R14: re-base the float clock and draw this swing's playback speed (single-clip path).
            self.time_steps_f[env_ids] = self.time_steps[env_ids].float()
            if self._speed_per_clip is not None:
                self.speed_scale[env_ids] = self._speed_per_clip[self.clip_id[env_ids]]
            else:
                s_lo, s_hi = self.cfg.speed_scale_range
                self.speed_scale[env_ids] = sample_uniform(
                    float(s_lo), float(s_hi), (len(env_ids),), device=self.device
                )

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def balanced_clip_sampler_state_dict(self) -> dict | None:
        """Return exact-resume state for the optional balanced clip sampler."""
        if self._balanced_clip_sampler is None:
            return None
        return self._balanced_clip_sampler.state_dict()

    def load_balanced_clip_sampler_state_dict(self, state: dict | None):
        """Restore balanced clip allocation, rejecting incompatible clip identity/order."""
        if self._balanced_clip_sampler is None:
            if state is not None:
                raise ValueError(
                    "checkpoint contains balanced clip sampler state but "
                    "balanced_clip_sampling is disabled"
                )
            return
        if state is None:
            raise ValueError(
                "balanced_clip_sampling is enabled but checkpoint sampler state is missing"
            )
        self._balanced_clip_sampler.load_state_dict(state)

    @staticmethod
    def _exact_resume_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu").clone()

    def _exact_resume_identity(self) -> dict:
        teacher_contract = self._post_swing_teacher_hard_contract
        teacher_contract_sha256 = (
            None
            if teacher_contract is None
            else hashlib.sha256(_canonical_json_bytes(teacher_contract)).hexdigest()
        )
        identity = {
            "motion": {
                "num_segments": int(self.motion.num_segments),
                "clip_order": tuple(self._motion_files),
                "clip_sha256": tuple(self._motion_file_sha256),
                "segment_lengths": tuple(
                    int(value) for value in self.motion.seg_len.detach().cpu().tolist()
                ),
                "body_names": tuple(str(value) for value in self.cfg.body_names),
                "joint_names": tuple(str(value) for value in self.robot.data.joint_names),
            },
            "adaptive_sampling_config": {
                "bin_count": int(self.bin_count),
                "adaptive_kernel_size": int(self.cfg.adaptive_kernel_size),
                "adaptive_lambda": float(self.cfg.adaptive_lambda),
                "adaptive_uniform_ratio": float(self.cfg.adaptive_uniform_ratio),
                "adaptive_alpha": float(self.cfg.adaptive_alpha),
            },
            "post_swing_replay_config": {
                "post_swing_start_prob": float(self.cfg.post_swing_start_prob),
                "post_swing_buffer_size": int(self.cfg.post_swing_buffer_size),
                "post_swing_min_fill": int(self.cfg.post_swing_min_fill),
                "post_swing_min_hold": int(self.cfg.post_swing_min_hold),
                "post_swing_teacher_hard_contract_sha256": teacher_contract_sha256,
                "post_swing_fail_fast_first_reset": bool(
                    self._post_swing_fail_fast_first_reset
                ),
                "post_swing_first_reset_min_adopted_count": int(
                    self._post_swing_first_reset_min_adopted_count
                ),
                "post_swing_first_reset_min_adopted_fraction": float(
                    self._post_swing_first_reset_min_adopted_fraction
                ),
                "post_swing_first_reset_selection_tolerance": float(
                    self._post_swing_first_reset_selection_tolerance
                ),
                "post_swing_first_reset_require_readback": bool(
                    self._post_swing_first_reset_require_readback
                ),
            },
        }
        if self._action_ball_birth_broker is not None:
            admission_receipt = (
                self.action_ball_motion_admission_hard_contract()
            )
            identity["action_ball"] = {
                "runtime_contract_sha256": (
                    self._action_ball_runtime_module_bound.RUNTIME_CONTRACT_SHA256
                ),
                "broker_state_schema_version": (
                    self._action_ball_runtime_module_bound.BROKER_STATE_SCHEMA_VERSION
                ),
                "broker_registry_sha256": (
                    self._action_ball_birth_broker.registry_sha256
                ),
                "ordered_action_uids": tuple(
                    self._action_ball_action_uids
                ),
                "trusted_repo_root": str(
                    self._action_ball_trusted_repo_root
                ),
                "motion_admission_receipt_sha256": (
                    admission_receipt["canonical_sha256"]
                ),
                "timing_authority": (
                    self._action_ball_runtime_module_bound
                    .TASK_RECEIPT_TIMING_AUTHORITY
                ),
                "policy_dt_s": float(self._env.step_dt),
                "episode_length_s": (
                    int(self._env.max_episode_length)
                    * float(self._env.step_dt)
                ),
            }
        return identity

    def _action_ball_exact_resume_state_dict(self) -> dict:
        broker_state = self._action_ball_birth_broker.state_dict()
        self._action_ball_sha256(
            broker_state.get("integrity_sha256"),
            name="broker.integrity_sha256",
        )
        pending = broker_state.get("pending")
        if not isinstance(pending, list) or any(
            not isinstance(row, dict) or row.get("status") != "committed"
            for row in pending
        ):
            raise RuntimeError(
                "action-ball exact resume cannot snapshot an in-flight reserve transaction"
            )
        transcript = {}
        for row in broker_state.get("consumed_receipts", ()):
            if type(row) is not dict:
                raise RuntimeError("action-ball broker transcript is malformed")
            transcript[(row["env_id"], row["reset_generation"])] = row[
                "canonical_sha256"
            ]
        for pending_row in pending:
            row = pending_row["receipt"]
            transcript[(row["env_id"], row["reset_generation"])] = row[
                "canonical_sha256"
            ]
        expected_seen = set(transcript.values())
        if expected_seen != self._action_ball_seen_birth_receipts:
            raise RuntimeError(
                "Motion/broker committed birth transcript diverged"
            )
        last = {
            int(env): int(generation)
            for env, generation in broker_state.get("last_generations", ())
        }
        reset_generation = [
            int(value)
            for value in self._action_ball_reset_generation.detach()
            .cpu()
            .tolist()
        ]
        runtime = self._action_ball_runtime_module_bound
        task_ref_rows = []
        for env_id, generation in enumerate(reset_generation):
            receipt_sha = self._action_ball_birth_receipt_sha256[env_id]
            task_ref = self._action_ball_active_task_refs[env_id]
            if generation == 0:
                if (
                    env_id in last
                    or receipt_sha is not None
                    or task_ref is not None
                ):
                    raise RuntimeError(
                        "zero-generation env has broker/birth/task state"
                    )
                task_ref_rows.append(None)
                continue
            if (
                last.get(env_id) != generation
                or transcript.get((env_id, generation)) != receipt_sha
            ):
                raise RuntimeError(
                    "Motion current generation/receipt differs from broker transcript"
                )
            if type(task_ref) is not runtime.ActionTaskReceiptRef:
                raise RuntimeError(
                    "positive-generation env lacks an exact active task ref"
                )
            live_ref = self._action_ball_task_ref_for_env(env_id)
            if live_ref != task_ref:
                raise RuntimeError(
                    "Motion active task ref differs from Racket authority"
                )
            resolved = self._action_ball_task_receipt_resolver(task_ref)
            self._validate_action_ball_task_ref_and_receipt(
                task_ref, resolved, env_id=env_id
            )
            task_ref_rows.append(task_ref.to_dict())
        admission_receipt = (
            self.action_ball_motion_admission_hard_contract()
        )
        return {
            "runtime_contract_sha256": (
                self._action_ball_runtime_module_bound.RUNTIME_CONTRACT_SHA256
            ),
            "broker_registry_sha256": (
                self._action_ball_birth_broker.registry_sha256
            ),
            "motion_admission_receipt_sha256": (
                admission_receipt["canonical_sha256"]
            ),
            # Racket is the sole owner of sampler/provider/domain/broker/pool/task bytes.  Motion
            # stores only its canonical full-state digest plus opaque per-env task references.
            "shared_racket_state_sha256": (
                self.action_ball_shared_racket_state_sha256()
            ),
            "reset_generation": self._exact_resume_cpu_tensor(
                self._action_ball_reset_generation
            ),
            "swing_generation": self._exact_resume_cpu_tensor(
                self._action_ball_swing_generation
            ),
            "birth_receipt_sha256": list(
                self._action_ball_birth_receipt_sha256
            ),
            "seen_birth_receipts": sorted(
                self._action_ball_seen_birth_receipts
            ),
            "active_task_refs": task_ref_rows,
        }

    def action_ball_shared_broker_state_sha256(self) -> str:
        """Return the live Racket-owned broker snapshot digest for runner ordering checks."""

        if self._action_ball_birth_broker is None:
            raise RuntimeError("action-ball birth broker is not bound")
        state = self._action_ball_birth_broker.state_dict()
        return self._action_ball_sha256(
            state.get("integrity_sha256"),
            name="broker.integrity_sha256",
        )

    def action_ball_shared_racket_state_sha256(self) -> str:
        """Return Racket's digest over every shared action-ball authority byte."""

        if self._action_ball_shared_state_sha256_accessor is None:
            raise RuntimeError("action-ball shared Racket digest is not bound")
        return self._action_ball_sha256(
            self._action_ball_shared_state_sha256_accessor(),
            name="Racket.action_ball_shared_state_sha256",
        )

    def finalize_action_ball_exact_resume(self) -> None:
        """Verify Racket-first shared restore against Motion's staged digest and refs."""

        expected = self._action_ball_expected_shared_racket_state_sha256
        if expected is None:
            raise RuntimeError(
                "Motion action-ball exact resume has no staged Racket digest"
            )
        if self.action_ball_shared_racket_state_sha256() != expected:
            raise RuntimeError(
                "live Racket state differs from Motion exact-resume handoff"
            )
        # Re-run broker transcript plus opaque task-ref resolution on the live Racket restore.
        snapshot = self._action_ball_exact_resume_state_dict()
        if snapshot["shared_racket_state_sha256"] != expected:
            raise RuntimeError(
                "live Racket task/birth refs differ after exact resume"
            )

    def exact_resume_state_dict(self) -> dict:
        """Return every persistent MotionCommand state that shapes the next rollout."""
        # Per-env clip/hold/planner/event clocks are deliberately absent: the runner performs one
        # full env reset after loading. The two stagger pending flags are also construction state,
        # not curriculum state—the documented resume path must re-spread that freshly reset cohort.
        # The fields below are the state that survives episode boundaries and changes later draws.
        ring_values = (
            self._post_swing_root,
            self._post_swing_joint_pos,
            self._post_swing_joint_vel,
        )
        if any(value is None for value in ring_values) and not all(
            value is None for value in ring_values
        ):
            raise RuntimeError("post-swing replay ring is only partially allocated")
        state = {
            "state_kind": self._EXACT_RESUME_STATE_KIND,
            "schema_version": (
                self._ACTION_BALL_EXACT_RESUME_STATE_SCHEMA_VERSION
                if self._action_ball_birth_broker is not None
                else self._EXACT_RESUME_STATE_SCHEMA_VERSION
            ),
            "identity": self._exact_resume_identity(),
            "adaptive_sampling": {
                "bin_failed_count": self._exact_resume_cpu_tensor(
                    self.bin_failed_count
                ),
                "current_bin_failed": self._exact_resume_cpu_tensor(
                    self._current_bin_failed
                ),
            },
            "post_swing_replay": {
                # Explicit null is part of the schema: an unallocated/disabled ring is state,
                # not a missing key that a loader may silently reinterpret.
                "root": (
                    None
                    if self._post_swing_root is None
                    else self._exact_resume_cpu_tensor(self._post_swing_root)
                ),
                "joint_pos": (
                    None
                    if self._post_swing_joint_pos is None
                    else self._exact_resume_cpu_tensor(self._post_swing_joint_pos)
                ),
                "joint_vel": (
                    None
                    if self._post_swing_joint_vel is None
                    else self._exact_resume_cpu_tensor(self._post_swing_joint_vel)
                ),
                "ptr": int(self._post_swing_ptr),
                "count": int(self._post_swing_count),
                "first_reset_checked": bool(
                    self._post_swing_first_reset_checked
                ),
            },
            "balanced_clip_sampler": self.balanced_clip_sampler_state_dict(),
        }
        if self._action_ball_birth_broker is not None:
            state["action_ball_birth"] = (
                {
                    "diagnostic_unauthorized": True,
                    "exact_resume_supported": False,
                    "broker_registry_sha256": (
                        self._action_ball_birth_broker.registry_sha256
                    ),
                }
                if self._action_ball_birth_broker.diagnostic_fast_path
                else self._action_ball_exact_resume_state_dict()
            )
        return state

    @staticmethod
    def _validate_exact_resume_tensor(
        value,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        nonnegative: bool = False,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise ValueError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be serialized on the CPU")
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"{name} shape/dtype mismatch: checkpoint={tuple(value.shape)}/"
                f"{value.dtype}, runtime={shape}/{dtype}"
            )
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
        if nonnegative and bool((value < 0).any()):
            raise ValueError(f"{name} contains negative counts")
        return value.detach().clone()

    def _prepare_action_ball_exact_resume_state(
        self, value
    ) -> dict:
        expected = {
            "runtime_contract_sha256",
            "broker_registry_sha256",
            "motion_admission_receipt_sha256",
            "shared_racket_state_sha256",
            "reset_generation",
            "swing_generation",
            "birth_receipt_sha256",
            "seen_birth_receipts",
            "active_task_refs",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError(
                "Motion action-ball exact-resume state keys do not match schema 4"
            )
        runtime = self._action_ball_runtime_module_bound
        admission_receipt = (
            self.action_ball_motion_admission_hard_contract()
        )
        if (
            value["runtime_contract_sha256"]
            != runtime.RUNTIME_CONTRACT_SHA256
            or value["broker_registry_sha256"]
            != self._action_ball_birth_broker.registry_sha256
            or value["motion_admission_receipt_sha256"]
            != admission_receipt["canonical_sha256"]
        ):
            raise ValueError(
                "Motion action-ball exact-resume immutable identity differs"
            )
        shared_racket_state_sha256 = self._action_ball_sha256(
            value["shared_racket_state_sha256"],
            name="action_ball.shared_racket_state_sha256",
        )
        reset_generation = self._validate_exact_resume_tensor(
            value["reset_generation"],
            name="action_ball.reset_generation",
            shape=(self.num_envs,),
            dtype=self._action_ball_reset_generation.dtype,
            nonnegative=True,
        )
        swing_generation = self._validate_exact_resume_tensor(
            value["swing_generation"],
            name="action_ball.swing_generation",
            shape=(self.num_envs,),
            dtype=self._action_ball_swing_generation.dtype,
            nonnegative=True,
        )
        if (
            bool((reset_generation > self._ACTION_BALL_INT64_MAX).any())
            or bool((swing_generation > self._ACTION_BALL_INT64_MAX).any())
        ):
            raise ValueError("Motion action-ball generation exceeds int64")

        current = value["birth_receipt_sha256"]
        seen = value["seen_birth_receipts"]
        task_ref_rows = value["active_task_refs"]
        if (
            type(current) is not list
            or len(current) != self.num_envs
            or type(seen) is not list
            or type(task_ref_rows) is not list
            or len(task_ref_rows) != self.num_envs
        ):
            raise ValueError(
                "Motion action-ball receipt state has invalid container shape"
            )
        current_receipts = []
        for index, digest in enumerate(current):
            if digest is not None:
                digest = self._action_ball_sha256(
                    digest,
                    name=f"action_ball.birth_receipt_sha256[{index}]",
                )
            current_receipts.append(digest)
        seen_receipts = [
            self._action_ball_sha256(
                digest, name=f"action_ball.seen_birth_receipts[{index}]"
            )
            for index, digest in enumerate(seen)
        ]
        if (
            seen_receipts != sorted(seen_receipts)
            or len(set(seen_receipts)) != len(seen_receipts)
        ):
            raise ValueError(
                "Motion action-ball seen receipt list must be sorted and unique"
            )

        reset_rows = [
            int(item) for item in reset_generation.tolist()
        ]
        seen_receipt_set = set(seen_receipts)
        task_refs = []
        for env_id, generation in enumerate(reset_rows):
            digest = current_receipts[env_id]
            ref_row = task_ref_rows[env_id]
            task_ref = (
                None
                if ref_row is None
                else runtime.ActionTaskReceiptRef.from_dict(ref_row)
            )
            if generation == 0:
                if digest is not None or task_ref is not None:
                    raise ValueError(
                        "zero-generation Motion env has a birth/task ref"
                    )
            else:
                if digest is None or digest not in seen_receipt_set:
                    raise ValueError(
                        "positive-generation Motion env lacks a seen birth ref"
                    )
                if (
                    type(task_ref) is not runtime.ActionTaskReceiptRef
                    or task_ref.env_id != env_id
                    or task_ref.reset_generation != generation
                    or task_ref.swing_generation
                    != int(swing_generation[env_id].item())
                    or task_ref.birth_sha256 != digest
                    or not (
                        0
                        <= task_ref.action_slot
                        < len(self._action_ball_action_uids)
                    )
                    or task_ref.action_uid
                    != self._action_ball_action_uids[
                        task_ref.action_slot
                    ]
                ):
                    raise ValueError(
                        "positive-generation Motion env has a mismatched task ref"
                    )
            task_refs.append(task_ref)
        return {
            "shared_racket_state_sha256": shared_racket_state_sha256,
            "reset_generation": reset_generation,
            "swing_generation": swing_generation,
            "birth_receipt_sha256": current_receipts,
            "seen_birth_receipts": set(seen_receipts),
            "active_task_refs": task_refs,
        }

    def load_exact_resume_state_dict(self, state: dict, strict: bool = True):
        """Restore only an exact schema/config/clip identity match."""
        if strict is not True:
            raise ValueError("MotionCommand exact resume supports only strict=True")
        if type(state) is not dict:
            raise ValueError("MotionCommand exact resume state must be a dictionary")
        expected_keys = {
            "state_kind",
            "schema_version",
            "identity",
            "adaptive_sampling",
            "post_swing_replay",
            "balanced_clip_sampler",
        }
        action_ball_bound = self._action_ball_birth_broker is not None
        expected_schema = (
            self._ACTION_BALL_EXACT_RESUME_STATE_SCHEMA_VERSION
            if action_ball_bound
            else self._EXACT_RESUME_STATE_SCHEMA_VERSION
        )
        if action_ball_bound:
            expected_keys.add("action_ball_birth")
        if set(state) != expected_keys:
            raise ValueError(
                "MotionCommand exact resume state keys do not match the strict schema"
            )
        if state["state_kind"] != self._EXACT_RESUME_STATE_KIND:
            raise ValueError("MotionCommand exact resume state_kind does not match")
        if state["schema_version"] != expected_schema:
            raise ValueError(
                "MotionCommand exact resume schema_version is unsupported"
            )
        if state["identity"] != self._exact_resume_identity():
            raise ValueError(
                "MotionCommand exact resume motion/config/clip identity does not match"
            )
        if (
            action_ball_bound
            and self._action_ball_birth_broker.diagnostic_fast_path
        ):
            raise ValueError(
                "diagnostic_unauthorized fast checkpoints contain policy/"
                "optimizer weights but no exact Motion ActionBall resume "
                "state"
            )
        action_ball_state = (
            self._prepare_action_ball_exact_resume_state(
                state["action_ball_birth"]
            )
            if action_ball_bound
            else None
        )

        adaptive = state["adaptive_sampling"]
        if type(adaptive) is not dict or set(adaptive) != {
            "bin_failed_count",
            "current_bin_failed",
        }:
            raise ValueError(
                "MotionCommand adaptive_sampling state does not match the strict schema"
            )
        bin_shape = tuple(self.bin_failed_count.shape)
        bin_failed_count = self._validate_exact_resume_tensor(
            adaptive["bin_failed_count"],
            name="bin_failed_count",
            shape=bin_shape,
            dtype=self.bin_failed_count.dtype,
            nonnegative=True,
        )
        current_bin_failed = self._validate_exact_resume_tensor(
            adaptive["current_bin_failed"],
            name="current_bin_failed",
            shape=tuple(self._current_bin_failed.shape),
            dtype=self._current_bin_failed.dtype,
            nonnegative=True,
        )

        replay = state["post_swing_replay"]
        replay_keys = {
            "root",
            "joint_pos",
            "joint_vel",
            "ptr",
            "count",
            "first_reset_checked",
        }
        if type(replay) is not dict or set(replay) != replay_keys:
            raise ValueError(
                "MotionCommand post_swing_replay state does not match the strict schema"
            )
        ptr = replay["ptr"]
        count = replay["count"]
        first_reset_checked = replay["first_reset_checked"]
        if (
            type(ptr) is not int
            or type(count) is not int
            or type(first_reset_checked) is not bool
        ):
            raise ValueError(
                "post-swing replay ptr/count/first_reset_checked have invalid types"
            )
        size = int(self.cfg.post_swing_buffer_size)
        if not (0 <= ptr < size) or not (0 <= count <= size):
            raise ValueError("post-swing replay ptr/count are outside the configured ring")
        ring_values = (replay["root"], replay["joint_pos"], replay["joint_vel"])
        ring_is_none = tuple(value is None for value in ring_values)
        if any(ring_is_none) and not all(ring_is_none):
            raise ValueError("post-swing replay ring is only partially serialized")
        if all(ring_is_none):
            if ptr != 0 or count != 0:
                raise ValueError(
                    "unallocated post-swing replay ring requires ptr=count=0"
                )
            if self._post_swing_teacher_hard_contract is not None:
                raise ValueError(
                    "configured post-swing teacher cannot restore an unallocated ring"
                )
            root = joint_pos = joint_vel = None
        else:
            joint_count = int(self.robot.data.joint_pos.shape[-1])
            root = self._validate_exact_resume_tensor(
                replay["root"],
                name="post_swing_root",
                shape=(size, 13),
                dtype=self.robot.data.root_state_w.dtype,
            )
            joint_pos = self._validate_exact_resume_tensor(
                replay["joint_pos"],
                name="post_swing_joint_pos",
                shape=(size, joint_count),
                dtype=self.robot.data.joint_pos.dtype,
            )
            joint_vel = self._validate_exact_resume_tensor(
                replay["joint_vel"],
                name="post_swing_joint_vel",
                shape=(size, joint_count),
                dtype=self.robot.data.joint_vel.dtype,
            )

        # Racket owns and restores the shared evaluator/curriculum/provider/domain/broker/pool/task
        # graph before this local load.  Motion never restores those bytes; it stages their full
        # digest plus opaque local refs for the runner's post-load finalize.
        sampler_before = self.balanced_clip_sampler_state_dict()
        bin_before = self.bin_failed_count.clone()
        current_bin_before = self._current_bin_failed.clone()
        replay_before = (
            self._post_swing_root,
            self._post_swing_joint_pos,
            self._post_swing_joint_vel,
            self._post_swing_ptr,
            self._post_swing_count,
            self._post_swing_first_reset_checked,
        )
        if action_ball_bound:
            action_ball_before = (
                self._action_ball_reset_generation.clone(),
                self._action_ball_swing_generation.clone(),
                list(self._action_ball_birth_receipt_sha256),
                set(self._action_ball_seen_birth_receipts),
                list(self._action_ball_active_task_refs),
                self._action_ball_task_timing_active.clone(),
                self._action_ball_task_pending_elapsed_s.clone(),
                self._action_ball_task_age_s.clone(),
                self._action_ball_time_to_contact_s.clone(),
                self._action_ball_teacher_rate.clone(),
                self._action_ball_scaled_t_hit_s.clone(),
                self._action_ball_scaled_t_cycle_s.clone(),
                self._action_ball_pre_swing_wait_s.clone(),
                self._action_ball_expected_shared_racket_state_sha256,
                self.clip_id.clone(),
            )
        else:
            action_ball_before = None
        try:
            self.load_balanced_clip_sampler_state_dict(
                state["balanced_clip_sampler"]
            )
            self.bin_failed_count.copy_(
                bin_failed_count.to(device=self.bin_failed_count.device)
            )
            self._current_bin_failed.copy_(
                current_bin_failed.to(
                    device=self._current_bin_failed.device
                )
            )
            self._post_swing_root = (
                None if root is None else root.to(device=self.device)
            )
            self._post_swing_joint_pos = (
                None if joint_pos is None else joint_pos.to(device=self.device)
            )
            self._post_swing_joint_vel = (
                None if joint_vel is None else joint_vel.to(device=self.device)
            )
            self._post_swing_ptr = ptr
            self._post_swing_count = count
            self._post_swing_first_reset_checked = first_reset_checked
            if action_ball_bound:
                self._action_ball_reset_generation.copy_(
                    action_ball_state["reset_generation"].to(
                        device=self.device
                    )
                )
                self._action_ball_swing_generation.copy_(
                    action_ball_state["swing_generation"].to(
                        device=self.device
                    )
                )
                self._action_ball_birth_receipt_sha256 = list(
                    action_ball_state["birth_receipt_sha256"]
                )
                self._action_ball_seen_birth_receipts = set(
                    action_ball_state["seen_birth_receipts"]
                )
                self._action_ball_active_task_refs = list(
                    action_ball_state["active_task_refs"]
                )
                # Positive-generation refs are Motion's checkpoint-local
                # action authority.  Reconstruct clip_id from them before the
                # post-load finalizer validates each live Racket receipt.
                # This is local tensor state only: no RNG or simulator write.
                for env_id, task_ref in enumerate(
                    self._action_ball_active_task_refs
                ):
                    if task_ref is not None:
                        self.clip_id[env_id] = task_ref.action_slot
                # The documented resume path finalizes Racket's restored authority and then
                # performs one full reset.  No pre-checkpoint task clock is replayed or allowed to
                # touch the simulator between those operations.
                self._action_ball_task_timing_active.zero_()
                self._action_ball_task_pending_elapsed_s.zero_()
                self._action_ball_task_age_s.zero_()
                self._action_ball_time_to_contact_s.zero_()
                self._action_ball_teacher_rate.zero_()
                self._action_ball_scaled_t_hit_s.zero_()
                self._action_ball_scaled_t_cycle_s.zero_()
                self._action_ball_pre_swing_wait_s.zero_()
                self._action_ball_expected_shared_racket_state_sha256 = (
                    action_ball_state["shared_racket_state_sha256"]
                )
        except Exception:
            # Restore live state without invoking reset/resample or any simulator setter.
            self.load_balanced_clip_sampler_state_dict(sampler_before)
            self.bin_failed_count.copy_(bin_before)
            self._current_bin_failed.copy_(current_bin_before)
            (
                self._post_swing_root,
                self._post_swing_joint_pos,
                self._post_swing_joint_vel,
                self._post_swing_ptr,
                self._post_swing_count,
                self._post_swing_first_reset_checked,
            ) = replay_before
            if action_ball_bound:
                (
                    reset_before,
                    swing_before,
                    receipt_before,
                    seen_before,
                    task_refs_before,
                    timing_active_before,
                    pending_elapsed_before,
                    task_age_before,
                    time_to_contact_before,
                    teacher_rate_before,
                    scaled_t_hit_before,
                    scaled_t_cycle_before,
                    pre_swing_wait_before,
                    expected_racket_before,
                    clip_id_before,
                ) = action_ball_before
                self._action_ball_reset_generation.copy_(reset_before)
                self._action_ball_swing_generation.copy_(swing_before)
                self._action_ball_birth_receipt_sha256 = receipt_before
                self._action_ball_seen_birth_receipts = seen_before
                self._action_ball_active_task_refs = task_refs_before
                self._action_ball_task_timing_active.copy_(
                    timing_active_before
                )
                self._action_ball_task_pending_elapsed_s.copy_(
                    pending_elapsed_before
                )
                self._action_ball_task_age_s.copy_(task_age_before)
                self._action_ball_time_to_contact_s.copy_(
                    time_to_contact_before
                )
                self._action_ball_teacher_rate.copy_(teacher_rate_before)
                self._action_ball_scaled_t_hit_s.copy_(scaled_t_hit_before)
                self._action_ball_scaled_t_cycle_s.copy_(
                    scaled_t_cycle_before
                )
                self._action_ball_pre_swing_wait_s.copy_(
                    pre_swing_wait_before
                )
                self._action_ball_expected_shared_racket_state_sha256 = (
                    expected_racket_before
                )
                self.clip_id.copy_(clip_id_before)
            raise

    def _capture_post_swing_states(self, env_ids: torch.Tensor):
        """A8: snapshot end-of-swing robot states (wrap envs only) into the ring buffer.

        Wrapped envs necessarily completed their swing physically (no teleport happened and they
        reached the clip's final frame), so every buffer entry is a genuine follow-through state.
        Root position is stored origin-relative; write pairs root_state_w <->
        write_root_state_to_sim (com-frame velocities) to match the stand/RSI branches.
        """
        # Receipt-backed science pairs keep one identical exogenous reset distribution.  Letting
        # each arm overwrite it with policy-owned wraps would reintroduce the treatment-dependent
        # curriculum that this cold-start path exists to remove.
        if self._post_swing_teacher_hard_contract is not None:
            return
        n = len(env_ids)
        if n == 0:
            return
        root = self.robot.data.root_state_w[env_ids].clone()
        root[:, :3] -= self._env.scene.env_origins[env_ids]
        jp = self.robot.data.joint_pos[env_ids].clone()
        jv = self.robot.data.joint_vel[env_ids].clone()
        if self._post_swing_capture_output_dir is not None and not self._post_swing_capture_complete:
            if (
                self._post_swing_capture_runtime_hard_contract_sha256 is None
                or self._post_swing_capture_claim_sha256 is None
            ):
                raise RuntimeError(
                    "natural-wrap capture cannot step before runtime-contract equality is bound"
                )
            # This is the sole source path that populates the capture accumulator: arrays are read
            # directly from the live articulation tensors above.  No writer/capability API accepts
            # arbitrary caller-owned arrays.  The caller remains ordinary Python, so the artifact
            # records reviewed-source/O_EXCL evidence rather than claiming cryptographic callback
            # provenance.  CPU conversion also synchronizes the CUDA producer before publication.
            root_np = root.detach().to(device="cpu", dtype=torch.float32).numpy()
            joint_pos_np = jp.detach().to(device="cpu", dtype=torch.float32).numpy()
            joint_vel_np = jv.detach().to(device="cpu", dtype=torch.float32).numpy()
            rows = min(
                root_np.shape[0] if root_np.ndim == 2 else 0,
                self._post_swing_capture_target_count - self._post_swing_capture_count,
            )
            if (
                rows <= 0
                or root_np.dtype != np.float32
                or joint_pos_np.dtype != np.float32
                or joint_vel_np.dtype != np.float32
                or root_np.shape[1:] != (13,)
                or joint_pos_np.shape
                != (root_np.shape[0], len(self._post_swing_capture_joint_names))
                or joint_vel_np.shape != joint_pos_np.shape
                or not np.isfinite(root_np).all()
                or not np.isfinite(joint_pos_np).all()
                or not np.isfinite(joint_vel_np).all()
            ):
                raise RuntimeError("natural-wrap source path produced an invalid runtime state batch")
            self._post_swing_capture_roots.append(np.array(root_np[:rows], copy=True))
            self._post_swing_capture_joint_pos.append(np.array(joint_pos_np[:rows], copy=True))
            self._post_swing_capture_joint_vel.append(np.array(joint_vel_np[:rows], copy=True))
            self._post_swing_capture_count += rows
            if self._post_swing_capture_count == self._post_swing_capture_target_count:
                self._publish_post_swing_capture()
        size = int(self.cfg.post_swing_buffer_size)
        if self._post_swing_root is None:
            self._post_swing_root = torch.zeros(size, 13, device=self.device)
            self._post_swing_joint_pos = torch.zeros(size, jp.shape[1], device=self.device)
            self._post_swing_joint_vel = torch.zeros(size, jv.shape[1], device=self.device)
        # ring write (n < size in practice; wrap the slot indices just in case)
        slots = (self._post_swing_ptr + torch.arange(n, device=self.device)) % size
        self._post_swing_root[slots] = root
        self._post_swing_joint_pos[slots] = jp
        self._post_swing_joint_vel[slots] = jv
        self._post_swing_ptr = int((self._post_swing_ptr + n) % size)
        self._post_swing_count = min(self._post_swing_count + n, size)

    def _load_post_swing_teacher_if_configured(self) -> None:
        """Seed the replay ring from one immutable natural-wrap teacher receipt."""

        receipt_path = str(
            getattr(self.cfg, "post_swing_teacher_receipt", "") or ""
        ).strip()
        receipt_sha = str(
            getattr(self.cfg, "post_swing_teacher_receipt_sha256", "") or ""
        ).strip().lower()
        authorization_path = str(
            getattr(self.cfg, "post_swing_teacher_retry_authorization", "") or ""
        ).strip()
        authorization_sha = str(
            getattr(
                self.cfg,
                "post_swing_teacher_retry_authorization_sha256",
                "",
            )
            or ""
        ).strip().lower()
        probability = float(self.cfg.post_swing_start_prob)
        configured_identity = tuple(
            bool(value)
            for value in (
                receipt_path,
                receipt_sha,
                authorization_path,
                authorization_sha,
            )
        )
        if any(configured_identity) and not all(configured_identity):
            raise ValueError(
                "post-swing teacher receipt and retry authorization paths/SHA-256 values "
                "must be provided together"
            )
        if (
            receipt_path
            or self._post_swing_require_ready_at_init
            or self._post_swing_fail_fast_first_reset
        ) and probability <= 0.0:
            raise ValueError(
                "post-swing teacher/activation gates require post_swing_start_prob > 0"
            )
        if (
            self._post_swing_require_ready_at_init
            or self._post_swing_fail_fast_first_reset
        ) and not receipt_path:
            raise ValueError(
                "ready-at-init, frozen teacher, and activation fail-fast modes require an "
                "immutable post_swing_teacher_receipt"
            )
        if not receipt_path:
            return

        motion_files = self.cfg.motion_file
        if isinstance(motion_files, str):
            motion_files = [motion_files]
        else:
            motion_files = list(motion_files)
        try:
            joint_velocity_limits = self.robot.data.joint_vel_limits
            if joint_velocity_limits.ndim == 2:
                joint_velocity_limits = joint_velocity_limits[0]
            if joint_velocity_limits.ndim != 1:
                raise ValueError("runtime joint velocity limits have an unexpected shape")
            teacher = load_post_swing_teacher_states(
                receipt_path,
                receipt_sha,
                retry_authorization_path=authorization_path,
                expected_retry_authorization_sha256=authorization_sha,
                expected_motion_sha256=[sha256_file(path) for path in motion_files],
                expected_joint_names=self.robot.data.joint_names,
                expected_joint_velocity_limits=[
                    float(value) for value in joint_velocity_limits.detach().cpu().tolist()
                ],
                expected_root_linear_velocity_limit_mps=float(
                    self.cfg.post_swing_teacher_root_linear_velocity_limit_mps
                ),
                expected_root_angular_velocity_limit_radps=float(
                    self.cfg.post_swing_teacher_root_angular_velocity_limit_radps
                ),
                min_fill=int(self.cfg.post_swing_min_fill),
                buffer_size=int(self.cfg.post_swing_buffer_size),
            )
        except (OSError, PostSwingTeacherError) as exc:
            raise ValueError(f"invalid post-swing teacher receipt: {exc}") from exc

        joint_pos = torch.as_tensor(teacher.joint_pos, device=self.device)
        limits = self.robot.data.soft_joint_pos_limits
        if limits.ndim != 3 or limits.shape[-1] != 2:
            raise ValueError("runtime soft joint-position limits have an unexpected shape")
        lower = limits[0, :, 0].to(device=self.device)
        upper = limits[0, :, 1].to(device=self.device)
        if joint_pos.shape[1] != lower.numel() or torch.any(joint_pos < lower) or torch.any(
            joint_pos > upper
        ):
            raise ValueError(
                "post-swing teacher joint positions violate runtime articulation limits"
            )

        count = int(teacher.root_state_origin_relative.shape[0])
        size = int(self.cfg.post_swing_buffer_size)
        self._post_swing_root = torch.zeros(size, 13, device=self.device)
        self._post_swing_joint_pos = torch.zeros(
            size, joint_pos.shape[1], device=self.device
        )
        self._post_swing_joint_vel = torch.zeros_like(self._post_swing_joint_pos)
        self._post_swing_root[:count] = torch.as_tensor(
            teacher.root_state_origin_relative, device=self.device
        )
        self._post_swing_joint_pos[:count] = joint_pos
        self._post_swing_joint_vel[:count] = torch.as_tensor(
            teacher.joint_vel, device=self.device
        )
        self._post_swing_count = count
        self._post_swing_ptr = count % size
        self._post_swing_teacher_hard_contract = teacher.hard_contract
        if self._post_swing_count < int(self.cfg.post_swing_min_fill):
            # The pure loader already rejects this; retain a local invariant at the simulator
            # adoption boundary so a future loader refactor cannot weaken ready-at-init.
            raise ValueError("post-swing teacher did not make the replay buffer ready")

    def _configure_post_swing_capture_if_requested(self) -> None:
        """Atomically claim one inference-only natural-wrap capture namespace, default off."""

        output_dir = str(getattr(self.cfg, "post_swing_capture_output_dir", "") or "").strip()
        target_count = getattr(self.cfg, "post_swing_capture_target_count", 0)
        if not output_dir:
            if type(target_count) is not int or target_count != 0:
                raise ValueError(
                    "post_swing_capture_target_count requires post_swing_capture_output_dir"
                )
            return
        if self._post_swing_teacher_hard_contract is not None:
            raise ValueError("teacher consumption and teacher capture are mutually exclusive")
        if type(target_count) is not int or target_count <= 0:
            raise ValueError("post_swing_capture_target_count must be a positive integer")
        if bool(self.cfg.wrap_teleport):
            raise ValueError("natural-wrap teacher capture requires wrap_teleport=false")
        if float(self.cfg.post_swing_start_prob) <= 0.0:
            raise ValueError("natural-wrap teacher capture requires post_swing_start_prob > 0")
        motion_files = self.cfg.motion_file
        motion_files = [motion_files] if isinstance(motion_files, str) else list(motion_files)
        capture_dir = Path(output_dir)
        if capture_dir.is_symlink() or not capture_dir.is_dir():
            raise ValueError("natural-wrap capture output must be an existing regular directory")
        for name in (CAPTURE_STATE_NAME, CAPTURE_RESULT_NAME):
            if os.path.lexists(capture_dir / name):
                raise ValueError(
                    "natural-wrap capture output already exists; one-shot replay is forbidden"
                )
        joint_names = [str(value) for value in self.robot.data.joint_names]
        if (
            not joint_names
            or any(not value for value in joint_names)
            or len(set(joint_names)) != len(joint_names)
        ):
            raise ValueError("capture joint names must be non-empty and unique")
        try:
            motion_clips = [
                {"index": index, "sha256": sha256_file(path)}
                for index, path in enumerate(motion_files)
            ]
            producer_source_sha256 = sha256_file(__file__)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            claim_fd = os.open(capture_dir / CAPTURE_CLAIM_NAME, flags, 0o600)
        except (OSError, PostSwingTeacherError) as exc:
            raise ValueError(f"cannot arm natural-wrap teacher capture: {exc}") from exc
        self._post_swing_capture_output_dir = capture_dir
        self._post_swing_capture_target_count = target_count
        self._post_swing_capture_motion_clips = motion_clips
        self._post_swing_capture_joint_names = joint_names
        self._post_swing_capture_producer_source_sha256 = producer_source_sha256
        self._post_swing_capture_claim_fd = claim_fd

    def post_swing_capture_complete(self) -> bool:
        """Return whether the one-shot source-owned result was durably published."""

        return self._post_swing_capture_complete

    def _bind_post_swing_capture_runtime_contract(self, sha256: str) -> None:
        if self._post_swing_capture_output_dir is None or self._post_swing_capture_claim_fd is None:
            raise RuntimeError("post-swing capture is not armed")
        if (
            self._post_swing_capture_count != 0
            or self._post_swing_capture_runtime_hard_contract_sha256 is not None
            or self._post_swing_capture_claim_sha256 is not None
        ):
            raise RuntimeError("capture runtime contract may be bound exactly once before stepping")
        if (
            type(sha256) is not str
            or len(sha256) != 64
            or any(value not in "0123456789abcdef" for value in sha256)
        ):
            raise RuntimeError("capture runtime hard-contract SHA-256 is invalid")
        claim = {
            "schema_version": 1,
            "artifact_kind": CAPTURE_CLAIM_KIND,
            "producer_source_sha256": self._post_swing_capture_producer_source_sha256,
            "runtime_hard_contract_sha256": sha256,
            "target_count": self._post_swing_capture_target_count,
            "motion_clips": list(self._post_swing_capture_motion_clips),
            "joint_names": list(self._post_swing_capture_joint_names),
            "exclusive_create": True,
        }
        raw = _canonical_json_bytes(claim)
        view = memoryview(raw)
        while view:
            written = os.write(self._post_swing_capture_claim_fd, view)
            if written <= 0:
                raise RuntimeError("cannot write the exclusive natural-wrap capture claim")
            view = view[written:]
        os.fsync(self._post_swing_capture_claim_fd)
        self._post_swing_capture_runtime_hard_contract_sha256 = sha256
        self._post_swing_capture_claim_sha256 = hashlib.sha256(raw).hexdigest()

    def _publish_post_swing_capture(self) -> None:
        """Publish accumulated live articulation snapshots; accepts no caller arrays."""

        if (
            self._post_swing_capture_output_dir is None
            or self._post_swing_capture_producer_source_sha256 is None
            or self._post_swing_capture_runtime_hard_contract_sha256 is None
            or self._post_swing_capture_claim_sha256 is None
            or self._post_swing_capture_claim_fd is None
            or self._post_swing_capture_count != self._post_swing_capture_target_count
        ):
            raise RuntimeError("natural-wrap capture publication invariants are not satisfied")
        root = np.concatenate(self._post_swing_capture_roots, axis=0)
        joint_pos = np.concatenate(self._post_swing_capture_joint_pos, axis=0)
        joint_vel = np.concatenate(self._post_swing_capture_joint_vel, axis=0)
        buffer = io.BytesIO()
        np.savez(
            buffer,
            root_state_origin_relative=root,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )
        state_bytes = buffer.getvalue()
        _publish_bytes_no_clobber(
            self._post_swing_capture_output_dir / CAPTURE_STATE_NAME,
            state_bytes,
            "natural-wrap state payload",
        )
        result = {
            "schema_version": 2,
            "artifact_kind": CAPTURE_RESULT_KIND,
            "capture_contract": dict(CAPTURE_CONTRACT),
            "evidence": {
                "producer_source_sha256": self._post_swing_capture_producer_source_sha256,
                "runtime_hard_contract_sha256": (
                    self._post_swing_capture_runtime_hard_contract_sha256
                ),
                "exclusive_claim_sha256": self._post_swing_capture_claim_sha256,
                "exclusive_claim_relative_path": CAPTURE_CLAIM_NAME,
                "no_clobber": True,
            },
            "motion_clips": list(self._post_swing_capture_motion_clips),
            "states": {
                "relative_path": CAPTURE_STATE_NAME,
                "sha256": hashlib.sha256(state_bytes).hexdigest(),
                "count": self._post_swing_capture_count,
                "root_shape": list(root.shape),
                "joint_pos_shape": list(joint_pos.shape),
                "joint_vel_shape": list(joint_vel.shape),
                "joint_names": list(self._post_swing_capture_joint_names),
            },
        }
        _publish_bytes_no_clobber(
            self._post_swing_capture_output_dir / CAPTURE_RESULT_NAME,
            _canonical_json_bytes(result),
            "natural-wrap capture result",
        )
        os.close(self._post_swing_capture_claim_fd)
        self._post_swing_capture_claim_fd = None
        self._post_swing_capture_complete = True

    def post_swing_replay_hard_contract(self) -> dict:
        """Return exact cold-start semantics for checkpoint lineage binding."""

        return {
            "teacher_receipt": self._post_swing_teacher_hard_contract,
            "teacher_distribution": "immutable",
            "require_ready_at_init": self._post_swing_require_ready_at_init,
            "fail_fast_first_reset": self._post_swing_fail_fast_first_reset,
            "first_reset_acceptance": {
                "min_adopted_count": self._post_swing_first_reset_min_adopted_count,
                "min_adopted_fraction": self._post_swing_first_reset_min_adopted_fraction,
                "selection_probability_abs_tolerance": self._post_swing_first_reset_selection_tolerance,
                "require_state_readback": self._post_swing_first_reset_require_readback,
            },
        }

    def _write_post_swing_states(self, env_ids: torch.Tensor):
        """A8: initialize `env_ids` from random buffered end-of-swing states (origin re-based)."""
        picks = torch.randint(0, self._post_swing_count, (len(env_ids),), device=self.device)
        root = self._post_swing_root[picks].clone()
        root[:, :3] += self._env.scene.env_origins[env_ids]
        joint_pos = self._post_swing_joint_pos[picks].clone()
        joint_vel = self._post_swing_joint_vel[picks].clone()
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        if self._post_swing_first_reset_require_readback:
            try:
                observed = (
                    ("root", self.robot.data.root_state_w[env_ids], root),
                    ("joint position", self.robot.data.joint_pos[env_ids], joint_pos),
                    ("joint velocity", self.robot.data.joint_vel[env_ids], joint_vel),
                )
            except (AttributeError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    "post-swing replay readback is unavailable on this runtime"
                ) from exc
            for label, actual, expected in observed:
                if actual.shape != expected.shape or not torch.allclose(
                    actual, expected, rtol=0.0, atol=1.0e-6
                ):
                    raise RuntimeError(f"post-swing replay {label} readback differs from write")

    def consume_post_swing_activation_counters(self) -> dict[str, torch.Tensor]:
        """Return one PPO update's replay-start counts and atomically reset the window.

        The training runner calls this once after each rollout/update.  Returning cloned device
        scalars keeps the completed window stable while the live counters are zeroed for the next
        update.  With ``post_swing_start_prob == 0`` every counter remains exactly zero and this
        instrumentation performs no sampling or simulator write.
        """

        snapshot = {
            name: value.detach().clone()
            for name, value in self._post_swing_activation_counters.items()
        }
        for value in self._post_swing_activation_counters.values():
            value.zero_()
        return snapshot

    def consume_training_activation_counters(self) -> dict[str, torch.Tensor]:
        """Snapshot and reset every integer activation counter for one PPO update.

        ``MotionOnPolicyRunner`` prefers this aggregate consumer over the legacy post-swing-only
        consumer.  Keeping the latter public preserves the original narrow API for diagnostic
        callers while this method guarantees that post-swing, V1 and V2 share one logger
        transaction and cannot be reset at different update boundaries.
        """

        snapshot = {
            name: value.detach().clone()
            for counters in (
                self._post_swing_activation_counters,
                self._reward_activation_counters,
            )
            for name, value in counters.items()
        }
        for counters in (
            self._post_swing_activation_counters,
            self._reward_activation_counters,
        ):
            for value in counters.values():
                value.zero_()
        return snapshot

    def record_v1_velocity_mimic_activation(
        self, eligible_sample_count: int | torch.Tensor, *, held_wrist_excluded: bool
    ) -> None:
        """Record real V1 linear-velocity imitation evaluations.

        The explicit config activation bit is written only by the V1 training override.  When it
        is disabled this method is a strict no-op.  The denominator is recorded before checking
        the resolved body list, so a miswired V1 run produces a positive denominator and a zero
        exclusion numerator instead of a false green.
        """

        if not bool(self.cfg.v1_free_wrist_vel_mimic_activation):
            return
        counters = self._reward_activation_counters
        counters["v1_velocity_mimic_eligible_sample_count"].add_(
            eligible_sample_count
        )
        if held_wrist_excluded:
            counters["v1_held_wrist_excluded_sample_count"].add_(
                eligible_sample_count
            )

    def record_v2_strike_window_scale_activation(
        self, strike_window: torch.Tensor, *, actual_window_scale: float
    ) -> None:
        """Record real V2 reward applications inside the wide strike window.

        The denominator counts wide-window samples reaching ``torch.where`` in the imitation
        reward path.  The numerator counts the same samples only when both the explicit V2
        activation contract and the actually applied reward parameter are exactly ``0.25``.
        Thus a missing/mismatched scale cannot pass by merely exposing a strike-window mask.
        """

        configured_scale = self.cfg.v2_motion_scale_in_window_activation
        if configured_scale is None:
            return
        eligible_sample_count = strike_window.to(dtype=torch.bool).sum(
            dtype=torch.long
        )
        counters = self._reward_activation_counters
        counters["v2_strike_window_eligible_imitation_sample_count"].add_(
            eligible_sample_count
        )
        if (
            float(configured_scale) == 0.25
            and float(actual_window_scale) == 0.25
        ):
            counters[
                "v2_quarter_scaled_strike_window_imitation_sample_count"
            ].add_(eligible_sample_count)

    def _action_ball_reset_motion_snapshot(
        self, env_ids: torch.Tensor
    ) -> dict:
        device = torch.device(self.device)
        if device.type == "cuda":
            rng_state = torch.cuda.get_rng_state(device)
        else:
            rng_state = torch.random.get_rng_state()
        metric_names = (
            "in_hold",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
        )
        return {
            "clip_id": self.clip_id[env_ids].clone(),
            "time_steps": self.time_steps[env_ids].clone(),
            "time_steps_f": self.time_steps_f[env_ids].clone(),
            "speed_scale": self.speed_scale[env_ids].clone(),
            "hold_counter": self.hold_counter[env_ids].clone(),
            "metrics": {
                name: self.metrics[name].clone()
                for name in metric_names
                if name in self.metrics
            },
            "balanced_sampler": self.balanced_clip_sampler_state_dict(),
            "stagger_pending": (
                None
                if self._stagger_hold_pending is None
                else self._stagger_hold_pending[env_ids].clone()
            ),
            "active_task_refs": list(self._action_ball_active_task_refs),
            "diagnostic_pending_row_count": (
                getattr(
                    self,
                    "_action_ball_diagnostic_pending_row_count",
                    0,
                )
            ),
            "task_timing_active": self._action_ball_task_timing_active[
                env_ids
            ].clone(),
            "task_pending_elapsed_s": (
                self._action_ball_task_pending_elapsed_s[env_ids].clone()
            ),
            "task_age_s": self._action_ball_task_age_s[env_ids].clone(),
            "time_to_contact_s": self._action_ball_time_to_contact_s[
                env_ids
            ].clone(),
            "teacher_rate": self._action_ball_teacher_rate[env_ids].clone(),
            "scaled_t_hit_s": self._action_ball_scaled_t_hit_s[
                env_ids
            ].clone(),
            "scaled_t_cycle_s": self._action_ball_scaled_t_cycle_s[
                env_ids
            ].clone(),
            "pre_swing_wait_s": self._action_ball_pre_swing_wait_s[
                env_ids
            ].clone(),
            "rng_state": rng_state,
        }

    def _restore_action_ball_reset_motion_snapshot(
        self, env_ids: torch.Tensor, state: dict
    ) -> None:
        self.load_balanced_clip_sampler_state_dict(
            state["balanced_sampler"]
        )
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["rng_state"], device)
        else:
            torch.random.set_rng_state(state["rng_state"])
        self.clip_id[env_ids] = state["clip_id"]
        self.time_steps[env_ids] = state["time_steps"]
        self.time_steps_f[env_ids] = state["time_steps_f"]
        self.speed_scale[env_ids] = state["speed_scale"]
        self.hold_counter[env_ids] = state["hold_counter"]
        self._action_ball_active_task_refs = list(
            state["active_task_refs"]
        )
        self._action_ball_diagnostic_pending_row_count = state[
            "diagnostic_pending_row_count"
        ]
        self._action_ball_task_timing_active[env_ids] = state[
            "task_timing_active"
        ]
        self._action_ball_task_pending_elapsed_s[env_ids] = state[
            "task_pending_elapsed_s"
        ]
        self._action_ball_task_age_s[env_ids] = state["task_age_s"]
        self._action_ball_time_to_contact_s[env_ids] = state[
            "time_to_contact_s"
        ]
        self._action_ball_teacher_rate[env_ids] = state["teacher_rate"]
        self._action_ball_scaled_t_hit_s[env_ids] = state[
            "scaled_t_hit_s"
        ]
        self._action_ball_scaled_t_cycle_s[env_ids] = state[
            "scaled_t_cycle_s"
        ]
        self._action_ball_pre_swing_wait_s[env_ids] = state[
            "pre_swing_wait_s"
        ]
        for name, value in state["metrics"].items():
            self.metrics[name].copy_(value)
        if self._stagger_hold_pending is not None:
            if state["stagger_pending"] is None:
                raise RuntimeError(
                    "action-ball reset snapshot lost stagger state"
                )
            self._stagger_hold_pending[env_ids] = state[
                "stagger_pending"
            ]

    def _resample_command(self, env_ids: Sequence[int]):
        """Run one formal atomic or diagnostic fail-stop action-ball true reset."""

        if len(env_ids) == 0:
            return
        if (
            self._action_ball_birth_broker is None
            or self._resampling_from_wrap
        ):
            return self._resample_command_body(env_ids)
        env_ids_t = (
            env_ids
            if torch.is_tensor(env_ids)
            else torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
        )
        env_ids_t = env_ids_t.to(
            device=self.device, dtype=torch.long
        ).reshape(-1)
        if self._action_ball_birth_broker.diagnostic_fast_path:
            # Diagnostic broker/provider/domain state is intentionally not
            # recoverable after a true-reset exception.  Let the one attempt
            # either publish normally or poison the whole run; a formal Motion
            # snapshot here is both unused and a dominant short-episode tax.
            return self._resample_command_body(env_ids_t)
        snapshot = self._action_ball_reset_motion_snapshot(env_ids_t)
        try:
            return self._resample_command_body(env_ids_t)
        except Exception:
            self._restore_action_ball_reset_motion_snapshot(
                env_ids_t, snapshot
            )
            raise

    def _resample_command_body(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        # A true episode boundary starts the same immutable sequence from an unarmed ledger.  An
        # intra-episode wrap before the initial origin is not a sequence boundary and must not
        # rewrite scheduler time.  Once armed, T1 suppresses natural wraps entirely.
        if self._event_scheduler is not None and not self._resampling_from_wrap:
            self._event_scheduler.reset(env_ids)
        self._adaptive_sampling(env_ids)

        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long)
        if (
            self._action_ball_birth_broker is not None
            and self._resampling_from_wrap
        ):
            # WRAP is a new swing against the same physical episode birth.  It freezes both the
            # action and root spawn, performs no broker transaction and writes no simulator root.
            self._advance_action_ball_wrap_generation(env_ids_t)

        if self.planner_revision_enabled:
            # A new ball starts from the action's explicit zero-rate entry frame.  Failure-adaptive
            # RSI may still choose the clip, but it may not start halfway through the swing: that
            # would erase preparation time and can place the phase past contact.  Clearing active
            # before any optional RSI simulator write also makes joint/body reference velocities
            # exactly zero until RacketTargetCommand installs the complete new task.
            if self._multiseg:
                starts = self.motion.seg_start[self.clip_id[env_ids_t]]
            else:
                starts = torch.zeros_like(env_ids_t)
            self.time_steps[env_ids_t] = starts
            self.time_steps_f[env_ids_t] = starts.float()
            self.speed_scale[env_ids_t] = 0.0
            self._planner_active[env_ids_t] = False
            # 孪生时钟回哨兵:新任务安装前窗保持关闭(fail-closed)。
            self._planner_truth_tts_signed[env_ids_t] = 1.0e6

        # A formal canonical-ready clip is self-contained: frame 0 is the shared waiting pose and
        # the following frames are its already-integrated path toward contact.  Always re-base a
        # newly selected clip on that frame; never serialize an extra ready->historical-frame-0
        # bridge and never release from adaptive RSI halfway through the stroke.
        if self.canonical_ready_mode:
            ready_steps = self._canonical_ready_steps(env_ids_t)
            self.time_steps[env_ids_t] = ready_steps
            self.time_steps_f[env_ids_t] = ready_steps.float()

        # Pre-swing HOLD (Phase A): freeze the reference at the swing's first frame for a random
        # number of control steps ("the ball is not reaching yet"). Applies to resets AND wraps.
        if self._action_ball_birth_broker is None:
            lo, hi = self.cfg.hold_steps_range
            self.hold_counter[env_ids_t] = torch.randint(
                int(lo),
                int(hi) + 1,
                (len(env_ids_t),),
                device=self.device,
            )
        else:
            # The per-swing receipt is the sole wait owner.  Keep a one-step fail-closed ready
            # sentinel until Racket's public accessor exposes the exact task later in this reset
            # (or, for WRAP, later in this command-manager update).
            self.hold_counter[env_ids_t] = 1
        # A wrap can resample a new hold late inside _update_command. Publish its state now so
        # downstream rewards/terminations on this same control step do not see the old swing mask.
        self.metrics["in_hold"][env_ids_t] = (self.hold_counter[env_ids_t] > 0).float()

        # stagger (a): each env's FIRST true reset adds a uniform hold bias, spreading the swing/
        # strike phases of a same-instant reset cohort across ~one swing period. One-shot per env;
        # wraps and every later reset draw the plain hold range, so steady-state behavior is
        # unchanged. The stand/post-swing min-hold clamps below are min= clamps — the bias
        # survives them. Default OFF (see cfg.stagger_initial_clock): no RNG draw, byte-identical.
        if (
            self._action_ball_birth_broker is None
            and self._stagger_hold_pending is not None
            and not self._resampling_from_wrap
        ):
            _pend_ids = env_ids_t[self._stagger_hold_pending[env_ids_t]]
            if len(_pend_ids) > 0:
                _mx = int(self.cfg.stagger_hold_max_steps)
                if _mx > 0:
                    self.hold_counter[_pend_ids] += torch.randint(
                        0, _mx + 1, (len(_pend_ids),), device=self.device
                    )
                self._stagger_hold_pending[_pend_ids] = False

        # Intra-episode clip WRAP: no teleport (deploy case) — the policy must physically carry
        # the body from the previous swing's end into the new swing's windup. The imitation
        # targets are anchor-relative, so the new reference re-anchors to the robot where it is.
        # Teleporting at a wrap (legacy RSI behavior) requires wrap_teleport=True.
        if self._resampling_from_wrap and not self.cfg.wrap_teleport:
            if self._action_ball_birth_broker is not None:
                self._begin_action_ball_task_pending(
                    env_ids_t, elapsed_s=float(self._env.step_dt)
                )
            return

        if self.canonical_ready_mode:
            # Every true episode reset belongs to the formal ready-entry distribution.  RSI and
            # post-swing replay are rejected at boot rather than silently surviving as alternate
            # reset routes.  The selected clip is immaterial to pose because startup validation
            # proved every start/end shares the same literal runtime ready.
            if self._action_ball_birth_broker is None:
                self._write_canonical_ready_state(env_ids_t)
            else:
                transaction = self._reserve_action_ball_true_reset(env_ids_t)
                sim_rollback_state = None
                try:
                    sim_rollback_state = self._write_canonical_ready_state(
                        env_ids_t,
                        action_ball_base_spawn_w_m=transaction["spawn"],
                        action_ball_base_quat_wxyz=transaction["quat"],
                    )
                    self._commit_action_ball_true_reset(
                        env_ids_t, transaction
                    )
                    self.hold_counter[env_ids_t] = torch.clamp(
                        self.hold_counter[env_ids_t],
                        min=int(self.cfg.stand_start_min_hold),
                    )
                    self.metrics["in_hold"][env_ids_t] = (
                        self.hold_counter[env_ids_t] > 0
                    ).float()
                    self._begin_action_ball_task_pending(
                        env_ids_t, elapsed_s=0.0
                    )
                except Exception as exc:
                    if self._action_ball_birth_broker.diagnostic_fast_path:
                        # Diagnostic transactions carry no rollback fields.
                        # Any failure after reserve poisons the run and must
                        # escape unchanged; retrying could reuse an advanced
                        # provider/RNG tape under the same logical reset.
                        raise
                    # _write_canonical_ready_state already restores a failed setter.  A later
                    # commit failure needs the same physical rollback before rewinding the exact
                    # broker/provider/domain tape.
                    if sim_rollback_state is not None:
                        try:
                            self._restore_action_ball_sim_state(
                                env_ids_t, sim_rollback_state
                            )
                        except Exception as rollback_error:
                            # Still restore the broker tape before surfacing the simulator failure.
                            self._rollback_action_ball_true_reset(
                                env_ids_t,
                                transaction,
                                original_error=exc,
                            )
                            raise RuntimeError(
                                "action-ball commit failed and simulator rollback failed"
                            ) from rollback_error
                    self._rollback_action_ball_true_reset(
                        env_ids_t,
                        transaction,
                        original_error=exc,
                    )
                    raise
                return
            self.hold_counter[env_ids_t] = torch.clamp(
                self.hold_counter[env_ids_t], min=int(self.cfg.stand_start_min_hold)
            )
            self.metrics["in_hold"][env_ids_t] = (
                self.hold_counter[env_ids_t] > 0
            ).float()
            return

        # TRUE episode reset: three-way split — DEFAULT STAND (deploy entry) / POST-SWING buffer
        # (A8: the policy's own end-of-swing states) / legacy RSI teleport onto the (noised)
        # reference frame. One uniform draw per env: u < stand_p -> stand; stand_p <= u <
        # stand_p + post_p -> post-swing (only once the buffer has post_swing_min_fill entries);
        # else RSI.
        u = torch.rand(len(env_ids_t), device=self.device)
        stand_mask = torch.zeros(len(env_ids_t), dtype=torch.bool, device=self.device)
        post_mask = torch.zeros(len(env_ids_t), dtype=torch.bool, device=self.device)
        post_selected_count: torch.Tensor | None = None
        if not self._resampling_from_wrap:
            stand_p = float(self.cfg.stand_start_prob)
            post_p = float(self.cfg.post_swing_start_prob)
            if stand_p > 0.0:
                stand_mask = u < stand_p
            if post_p > 0.0:
                buffer_ready = self._post_swing_count >= int(self.cfg.post_swing_min_fill)
                if buffer_ready:
                    eligible_count = len(env_ids_t)
                    post_mask = (u >= stand_p) & (u < stand_p + post_p)
                    post_selected_count = post_mask.sum(dtype=torch.long)
                    counters = self._post_swing_activation_counters
                    counters["post_swing_replay_eligible_reset_count"].add_(eligible_count)
                    counters["post_swing_replay_selected_reset_count"].add_(
                        post_selected_count
                    )
                    counters["post_swing_replay_random_not_selected_reset_count"].add_(
                        eligible_count - post_selected_count
                    )
                else:
                    self._post_swing_activation_counters[
                        "post_swing_replay_buffer_not_ready_reset_count"
                    ].add_(len(env_ids_t))
        stand_ids = env_ids_t[stand_mask]
        post_ids = env_ids_t[post_mask]
        rsi_ids = env_ids_t[~(stand_mask | post_mask)]

        if len(stand_ids) > 0:
            default_root = self.robot.data.default_root_state[stand_ids].clone()
            default_root[:, :3] += self._env.scene.env_origins[stand_ids]
            default_root[:, 7:] = 0.0  # zero lin/ang velocity
            # Optional heading-recovery curriculum: deploy follow-throughs can enter the
            # recovery hold yawed, so square-only stand starts leave that state unseen.
            yaw = _stand_start_yaw_samples(
                self.cfg.stand_start_yaw_range, len(stand_ids), self.device
            )
            if yaw is not None:
                zero = torch.zeros_like(yaw)
                yaw_delta = quat_from_euler_xyz(zero, zero, yaw)
                default_root[:, 3:7] = quat_mul(yaw_delta, default_root[:, 3:7])
            self.robot.write_root_state_to_sim(default_root, env_ids=stand_ids)
            self.robot.write_joint_state_to_sim(
                self.robot.data.default_joint_pos[stand_ids],
                torch.zeros_like(self.robot.data.default_joint_vel[stand_ids]),
                env_ids=stand_ids,
            )
            # Give the stand-started envs time to travel stand -> windup before the clip runs.
            self.hold_counter[stand_ids] = torch.clamp(
                self.hold_counter[stand_ids], min=int(self.cfg.stand_start_min_hold)
            )

        if len(post_ids) > 0:
            if post_selected_count is None:
                raise RuntimeError(
                    "post-swing replay ids exist without an activation selected count"
                )
            self._write_post_swing_states(post_ids)
            # Count a replay as started only after both root and joint state writes return.  A
            # selected count without a started count therefore exposes a failed adoption path
            # instead of silently treating the random draw as a real replay start.
            self._post_swing_activation_counters[
                "post_swing_replay_started_reset_count"
            ].add_(post_selected_count)
            # Settle follow-through -> windup before the clip runs.
            self.hold_counter[post_ids] = torch.clamp(
                self.hold_counter[post_ids], min=int(self.cfg.post_swing_min_hold)
            )

        if self._post_swing_fail_fast_first_reset and not self._post_swing_first_reset_checked:
            # CommandManager invokes this true-reset path while constructing/resetting the
            # environment, before PPO can collect or optimize its first rollout.  Requiring one
            # successful adoption here catches a dead/endogenous cold start without burning a
            # +200 checkpoint.  The draw is still the configured Bernoulli draw; scientific
            # queues should use a large enough initial cohort that selected>0 is deterministic in
            # practice (4096 envs at p=0.25 in the registered pair).
            if self._post_swing_count < int(self.cfg.post_swing_min_fill):
                raise RuntimeError(
                    "post-swing first-reset fail-fast: teacher buffer is not ready"
                )
            selected = 0 if post_selected_count is None else int(post_selected_count.item())
            eligible = len(env_ids_t)
            selected_fraction = selected / eligible if eligible > 0 else 0.0
            if selected < self._post_swing_first_reset_min_adopted_count:
                raise RuntimeError(
                    "post-swing first-reset fail-fast: adopted count below the frozen minimum "
                    f"({selected} < {self._post_swing_first_reset_min_adopted_count})"
                )
            if selected_fraction < self._post_swing_first_reset_min_adopted_fraction:
                raise RuntimeError(
                    "post-swing first-reset fail-fast: adopted fraction below the frozen minimum "
                    f"({selected_fraction} < {self._post_swing_first_reset_min_adopted_fraction})"
                )
            if abs(selected_fraction - float(self.cfg.post_swing_start_prob)) > (
                self._post_swing_first_reset_selection_tolerance
            ):
                raise RuntimeError(
                    "post-swing first-reset fail-fast: selected fraction differs from the "
                    "configured Bernoulli probability beyond tolerance"
                )
            # Reaching here means _write_post_swing_states returned after both root and joint
            # state writes, and started was incremented from the same selected scalar.
            self._post_swing_first_reset_checked = True

        # stand/post-start clamps may have promoted an initially zero draw to a real hold.
        self.metrics["in_hold"][env_ids_t] = (self.hold_counter[env_ids_t] > 0).float()

        if len(rsi_ids) == 0:
            return
        env_ids = rsi_ids

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # R-c(ii) rsi_hold_root_stand_z: a HELD RSI birth (hold_counter>0, drawn above — ~100/101
        # of RSI births at hold_steps_range [0,100]) writes STAND joints (the joint_pos property's
        # hold gate) but the reference frame's CROUCH root z (~0.78 m; body_pos_w has NO hold
        # gate) — stand legs at crouch height put the feet ~0.29 m under the floor and PhysX
        # depenetration kicks the robot out at birth. Fix: give held-RSI births the DEFAULT-STAND
        # root height (default_root_state z, 1.0684 m on the A3 — read at runtime, never
        # hardcoded); xy + yaw stay the reference frame's. Velocities are already hold-zeroed by
        # the body_*_vel_w properties. Default False = byte-identical.
        if bool(getattr(self.cfg, "rsi_hold_root_stand_z", False)):
            held_rsi = env_ids[self.hold_counter[env_ids] > 0]
            if len(held_rsi) > 0:
                root_pos[held_rsi, 2] = (
                    self.robot.data.default_root_state[held_rsi, 2]
                    + self._env.scene.env_origins[held_rsi, 2]
                )

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def install_external_exam_timing(
        self,
        env_ids: Sequence[int],
        clip_ids: torch.Tensor,
        hold_steps: torch.Tensor,
    ) -> None:
        """Install one evaluator-owned, immutable BankExam item per environment.

        This is deliberately a runtime seam rather than a config field: training still owns its
        normal random clip/hold sampler, while the formal evaluator may replace the *current*
        command only after it has independently validated an exam-split bank and schedule.  The
        method does not reset robot state; callers must first perform the documented nominal-stand
        reset and then refresh observations after installing both motion timing and racket targets.
        """

        raw_ids = torch.as_tensor(env_ids, device=self.device)
        raw_clips = torch.as_tensor(clip_ids, device=self.device)
        raw_holds = torch.as_tensor(hold_steps, device=self.device)
        for name, value in (("env_ids", raw_ids), ("clip_ids", raw_clips),
                            ("hold_steps", raw_holds)):
            if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                raise ValueError(f"external exam {name} must use an integer dtype")
        ids = raw_ids.to(dtype=torch.long).reshape(-1)
        clips = raw_clips.to(dtype=torch.long).reshape(-1)
        holds = raw_holds.to(dtype=torch.long).reshape(-1)
        if len(ids) == 0 or len(ids) != len(clips) or len(ids) != len(holds):
            raise ValueError(
                "external exam timing requires equal, non-empty env/clip/hold vectors"
            )
        if len(torch.unique(ids)) != len(ids) or torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise ValueError("external exam env ids must be unique and in range")
        if torch.any(clips < 0) or torch.any(clips >= int(self.motion.num_segments)):
            raise ValueError("external exam clip ids are outside the loaded motion segments")
        if torch.any(holds < 0):
            raise ValueError("external exam hold steps must be non-negative")
        if bool(self.cfg.stagger_initial_clock) or float(self.cfg.clip_switch_prob) != 0.0:
            raise ValueError(
                "external BankExam requires stagger_initial_clock=false and clip_switch_prob=0"
            )
        if self._speed_per_clip is not None or tuple(float(v) for v in self.cfg.speed_scale_range) != (1.0, 1.0):
            raise ValueError(
                "external BankExam currently requires native one-frame-per-step playback"
            )

        self._require_canonical_ready_boundary(ids, "external BankExam install")
        self.clip_id[ids] = clips
        starts = self.motion.seg_start[clips]
        self.time_steps[ids] = starts
        self.time_steps_f[ids] = starts.float()
        self.speed_scale[ids] = 1.0
        self.hold_counter[ids] = holds
        self.metrics["in_hold"][ids] = (holds > 0).float()
        self.just_resampled[ids] = False
        if hasattr(self, "time_left"):
            self.time_left[ids] = float("inf")
        if self._stagger_hold_pending is not None:
            self._stagger_hold_pending[ids] = False
        self._stagger_ep_pending = False

    def _update_command(self):
        # stagger (b): ONE-SHOT at the first step after construction (fresh run OR resume — both
        # are the same-instant cohort the metric-sync forensics caught): advance every env's
        # episode clock by U[0, max_episode_length) so the first timeouts, and every episode
        # boundary after them, spread out instead of firing in one synchronized wave. Guarded on
        # the env exposing the clock (defensive: metrics must never crash training).
        if self._stagger_ep_pending:
            self._stagger_ep_pending = False
            _ep_buf = getattr(self._env, "episode_length_buf", None)
            _max_len = int(getattr(self._env, "max_episode_length", 0) or 0)
            if _ep_buf is not None and _max_len > 1:
                _ep_buf.add_(torch.randint(0, _max_len, (self.num_envs,), device=_ep_buf.device))
        # Pre-swing HOLD: action-ball owns a continuous receipt deadline (including a possible
        # fractional first motion tick); legacy paths retain their integer random hold counter.
        action_ball_active = self._action_ball_birth_broker is not None
        if action_ball_active and self._event_scheduler is not None:
            # bind_action_ball_birth_broker requires canonical-ready mode, whose
            # boot contract rejects every non-disabled event timing mode.
            raise RuntimeError(
                "action-ball/event timing mutual exclusion drifted after binding"
            )
        if action_ball_active:
            held, action_ball_cycle_due = (
                self._advance_action_ball_task_timing()
            )
        else:
            held = self.hold_counter > 0
            self.hold_counter = torch.clamp(
                self.hold_counter - 1, min=0
            )
            self.metrics["in_hold"] = held.float()
        if "clip_switch_count" not in self.metrics:
            self.metrics["clip_switch_count"] = torch.zeros(self.num_envs, device=self.device)
        if self.planner_revision_enabled:
            # The same-ball governor owns the sole reference clock.  Non-active rows can only
            # occur during construction/reset ordering and remain frozen until RacketTargetCommand
            # installs their first complete task tuple.
            frame_delta = self._advance_planner_phase(held)
            self.speed_scale = torch.where(
                self._planner_active, frame_delta, torch.zeros_like(frame_delta)
            )
            self.time_steps_f += self.speed_scale
            self.time_steps = self.time_steps_f.round().long()
            self.metrics["playback_speed"] = self.speed_scale.clone()
        elif action_ball_active:
            # _advance_action_ball_task_timing analytically installed the current receipt phase.
            pass
        elif self.retiming_active:
            # R14: fractional clock — advance s frames per unheld control step; the integer index is
            # derived by round(), mirroring the deploy clock's nearest-frame mapping (torch rounds
            # half-to-even vs C++ half-away-from-zero — differs only on exact .5 ties, measure-zero
            # for continuous speed ranges).
            self.time_steps_f += (~held).float() * self.speed_scale
            self.time_steps = self.time_steps_f.round().long()
            self.metrics["playback_speed"] = self.speed_scale.clone()
        else:
            self.time_steps += (~held).long()
        if not action_ball_active:
            event_owned = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if not action_ball_active and self._event_scheduler is not None:
            native = self._event_native_strike_ticks
            if native is None:
                if bool(self._event_scheduler.armed.any()):
                    raise RuntimeError(
                        "post_strike_t1 armed before RacketTargetCommand bound native strike timing"
                    )
                # Before the first exact-strike origin no row can reveal and absolute scheduler
                # time has no meaning.  RacketTargetCommand binds the real vector in the same
                # command-manager step that can accept the initial exact strike.
            else:
                event_step = self._event_scheduler.advance(native)
                self._install_event_motion(event_step)
            event_owned = self._event_scheduler.armed
            self.metrics["event_timing_armed"] = event_owned.float()
            self.metrics["event_question_installed"] = (
                self._event_scheduler.event_just_installed.float()
            )
            self.metrics["event_question_unavailable"] = (
                self._event_scheduler.event_just_unavailable.float()
            )
            self.metrics["event_question_infeasible"] = (
                self._event_scheduler.event_just_infeasible.float()
            )
            self.metrics["event_deadline_due"] = (
                self._event_scheduler.deadline_just_due.float()
            )
            self.metrics["event_opportunities_consumed"] = (
                self._event_scheduler.opportunities_consumed.float()
            )
        if action_ball_active:
            # Receipt timing is the sole ActionBall wrap owner.  The bind-time
            # event exclusion above makes clamp/event reductions both
            # semantically impossible and an avoidable host synchronization.
            env_ids = torch.where(action_ball_cycle_due)[0]
            wrap_ids = env_ids
        elif self._multiseg:
            # Wrap at the END of the env's current clip/segment, not the global concatenated end.
            seg_end = self.motion.seg_start[self.clip_id] + self.motion.seg_len[self.clip_id]
            # Once an exact-strike origin arms T1, natural clip completion is a carry-state wait,
            # not permission to draw or teleport to another question.  Clamp the old reference at
            # its final native frame until the immutable reveal installs the next clip.
            clamp = event_owned & (self.time_steps >= seg_end)
            if bool(clamp.any()):
                self.time_steps[clamp] = seg_end[clamp] - 1
                self.time_steps_f[clamp] = self.time_steps[clamp].float()
            wrap_ids = torch.where(
                (~event_owned) & (self.time_steps >= seg_end)
            )[0]
            # DEPLOY-PARITY CLIP SWITCH (venue falls 2026-07-04): the runner's reference clock flips
            # clip_id whenever the planner re-sides the target — at an ARBITRARY mid-swing moment —
            # and the reference jumps to the new clip's first frame (pp_reference_clock.hpp clamps
            # tts-large to seg_start). Training previously only switched clips at clip END, so the
            # policy never saw that discontinuity and falls at 准备/正手/反手 switches on hardware.
            # With per-step prob clip_switch_prob an env aborts its swing operator-style and routes
            # through the SAME wrap-resample path (uniform new clip, frame 0, hold, fresh target).
            # NOTE: aborted swings count as uncompleted starts (slight completion-rate deflation).
            if (
                float(self.cfg.clip_switch_prob) > 0.0
            ):
                sw = torch.rand(self.num_envs, device=self.device) < float(self.cfg.clip_switch_prob)
                sw[wrap_ids] = False
                self.metrics["clip_switch_count"] = sw.float()
                switch_ids = torch.where(sw)[0]
                env_ids = torch.cat([wrap_ids, switch_ids]) if len(switch_ids) > 0 else wrap_ids
            else:
                env_ids = wrap_ids
        else:
            clamp = event_owned & (self.time_steps >= self.motion.time_step_total)
            if bool(clamp.any()):
                self.time_steps[clamp] = int(self.motion.time_step_total) - 1
                self.time_steps_f[clamp] = self.time_steps[clamp].float()
            env_ids = torch.where(
                (~event_owned)
                & (self.time_steps >= self.motion.time_step_total)
            )[0]
            wrap_ids = env_ids
        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(env_ids) > 0:
            self.just_resampled[env_ids] = True
            # A8: only envs that physically COMPLETED a swing (true wraps — passed the strike alive,
            # not teleported, not aborted-by-switch) feed the post-swing ring buffer.
            if self.cfg.post_swing_start_prob > 0.0 and len(wrap_ids) > 0:
                self._capture_post_swing_states(wrap_ids)
        # Wrap-path resample: skips the RSI teleport (cfg.wrap_teleport=False) so the policy
        # physically transitions swing -> swing. True resets go through reset()/manager instead.
        self._resampling_from_wrap = True
        try:
            self._resample_command(env_ids)
        finally:
            self._resampling_from_wrap = False

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    # Historical diagnostic replay only. Formal paths migrate these untagged finite-difference
    # link-origin velocities to the schema-2 COM-point contract instead of enabling this escape.
    allow_legacy_link_origin_velocity: bool = False
    # Formal canonical-library consumer.  Default OFF preserves every historical hold/reset,
    # reference value and RNG draw.  ON requires exact schema-2 clips whose starts and ends share
    # one literal runtime-float32 ready pose with zero endpoint velocities.  Every hold reference
    # (joint/body/anchor) and every true reset then comes from that same clip frame; RSI,
    # post-swing replay, reset noise, yaw perturbation and wrap teleport are rejected instead of
    # silently creating a second entry distribution.
    canonical_ready_mode: bool = False
    # Optional train.py-materialized action-specific reset/hold binding.  ``None`` is the literal
    # legacy path.  The runtime mapping is validated after immutable motion bytes are loaded, then
    # its normalized actor action and hold q_des are installed atomically with every ActionBall
    # true reset so physical spawn, last-action observation and controller state begin coherently.
    action_ball_dynamic_ready: dict | None = None
    # Formal mode has no raw-file escape hatch: one exact registry must authorize and atomically
    # bind the ordered five motion paths, family/phase/face tables, shared ready, and artifact
    # hashes.  All strings remain inert while canonical_ready_mode is false.
    canonical_registry_path: str = ""
    canonical_registry_repo_root: str = ""
    canonical_registry_sha256: str = ""
    canonical_registry_alignment_sha256: str = ""
    canonical_ready_sha256: str = ""
    canonical_ready_fk_sha256: str = ""
    # Path selection is configurable, authority is not: exact certificate bytes
    # must hash to a digest in canonical_motion_admission.py's code-owned set.
    canonical_promotion_certificate_path: str = ""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    # Execution-ledger activation bits.  ``train.py`` writes these only when the corresponding
    # V1/V2 override is explicitly present.  Defaults keep both ledgers inert; they do not change
    # reward values, simulator state, or random-number consumption.
    v1_free_wrist_vel_mimic_activation: bool = False
    v2_motion_scale_in_window_activation: float | None = None

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # --- Phase A (2026-07-02): swing ENTRY / TRANSITION / WAITING coverage --------------------
    # Deploy enters every swing from a NOMINAL STAND, waits at the windup while the ball is not
    # yet reaching, and must physically transition between swings — none of which the pure-RSI
    # scheme ever produced (teleport at every episode start AND every clip wrap). These knobs
    # close that gap; the imitation targets are anchor-RELATIVE (re-anchored to the robot's
    # current xy+yaw every step), so no-teleport starts/wraps are well-posed.
    # Fraction of TRUE episode resets that start from the robot's DEFAULT STAND pose (zero
    # velocities) instead of teleporting onto the reference clip frame (RSI).
    stand_start_prob: float = 0.25
    # Teleport the robot onto the new clip's start frame at intra-episode wraps (legacy RSI
    # behavior). False = the policy must physically transition swing->swing (the deploy case).
    wrap_teleport: bool = False
    # Pre-swing HOLD: on every swing (re)start, freeze the reference at the clip's first frame
    # for U[lo,hi] control steps (50 Hz). While held, time_to_strike sits at its per-clip
    # maximum — exactly the deploy runner's clamped "waiting for the ball" pairing.
    hold_steps_range: tuple[int, int] = (0, 100)
    # Stand-started envs get at least this much hold (they must travel stand -> windup first).
    stand_start_min_hold: int = 25
    # Uniform world-yaw perturbation (rad) for stand starts. Pair a nonzero range with a
    # hold-only heading-recovery objective; (0, 0) preserves the legacy square start.
    stand_start_yaw_range: tuple[float, float] = (0.0, 0.0)
    # --- A8 (Ace recipe): post-swing initial-state distribution ------------------------------
    # Fraction of TRUE episode resets initialized from a ring buffer of the policy's OWN
    # end-of-swing states (captured at every intra-episode clip wrap — envs that physically
    # completed a swing). Teaches "start the next swing from wherever the last one left you"
    # even for single-swing episodes. Drawn AFTER stand_start_prob from the remaining resets;
    # falls back to RSI while the buffer has fewer than post_swing_min_fill entries.
    post_swing_start_prob: float = 0.0
    post_swing_buffer_size: int = 4096
    post_swing_min_fill: int = 256
    # Post-swing-started envs get at least this much hold (settle follow-through -> windup).
    post_swing_min_hold: int = 25
    # Optional exogenous cold start.  The receipt contains only states captured at natural clip
    # wraps and binds teacher checkpoint/source/contract, exact motion bytes and runtime joint
    # order.  Empty/default preserves the historical policy-owned live buffer exactly.
    post_swing_teacher_receipt: str = ""
    post_swing_teacher_receipt_sha256: str = ""
    post_swing_teacher_retry_authorization: str = ""
    post_swing_teacher_retry_authorization_sha256: str = ""
    # Explicit runtime limits accepted by the attestor and rechecked before simulator adoption.
    # A floating base has no actuator limit in PhysX, so the capture contract must pin both norms.
    post_swing_teacher_root_linear_velocity_limit_mps: float = 0.0
    post_swing_teacher_root_angular_velocity_limit_radps: float = 0.0
    # Explicit scientific pairs can refuse endogenous cold starts at process startup and require
    # the initial true-reset cohort to adopt at least one teacher state before the first policy
    # rollout/update.  Both default off so existing checkpoints/queues keep exact behavior.
    post_swing_require_ready_at_init: bool = False
    post_swing_fail_fast_first_reset: bool = False
    post_swing_first_reset_min_adopted_count: int = 1
    post_swing_first_reset_min_adopted_fraction: float = 0.0
    post_swing_first_reset_selection_tolerance: float = 1.0
    post_swing_first_reset_require_readback: bool = False
    # Inference-only producer seam.  It emits a raw natural-wrap callback result; it cannot mint
    # a teacher receipt or attest a checkpoint.  Defaults preserve historical training exactly.
    post_swing_capture_output_dir: str = ""
    post_swing_capture_target_count: int = 0
    # Per-step per-env probability of an operator-style mid-swing clip switch (deploy parity —
    # see the venue-falls note in _update_command). 0.002 ~ one switch per ~3-4 swings. Default off.
    clip_switch_prob: float = 0.0
    # Deterministic exactly balanced multi-clip allocation. OFF keeps the historical
    # torch.randint call and global-RNG consumption byte-identical. ON cycles through one
    # locally seeded permutation, so across any prefix (and across differently sized resample
    # calls) every clip's cumulative assignment count differs by at most one. The cursor,
    # permutation and resolved clip order are exposed by MotionCommand's explicit state hooks.
    balanced_clip_sampling: bool = False
    balanced_clip_sampling_seed: int = 0
    # T1 post-strike event timing.  Disabled is the byte-identical current scheduler.  The enabled
    # path requires a materialized immutable schedule whose exact UTF-8 JSON bytes match the
    # configured SHA-256; rows are assigned deterministically by env id and never repeat inside an
    # episode.  It is intentionally incompatible with random clip switching, stagger, retiming,
    # wrap teleport, and RSI frame skipping.
    event_timing_mode: str = EVENT_TIMING_MODE_DISABLED
    event_timing_schedule: str = ""
    event_timing_schedule_sha256: str = ""
    event_timing_repeat: bool = False
    # P2.4/R14 retiming: per-swing reference playback speed, uniform-sampled from this range at
    # every swing entry (wrap, mid-swing clip switch, and true reset). At speed s the clip clock
    # advances s frames per control step, reference velocities read ×s, time_to_strike runs ÷s,
    # and the racket velocity target scales ×s (hope_commands) — the (frame, tts, velocity)
    # pairing stays consistent, unlike the deploy runner's swing_speed knob which retimes the
    # clock but NOT the velocities (pp_policy.hpp). Default (1.0, 1.0) = OFF: the integer-clock
    # path below is byte-identical to before this flag existed.
    speed_scale_range: tuple[float, float] = (1.0, 1.0)
    # FIXED per-clip playback speed (2026-07-08 backhand-fix ablation): one entry per clip in
    # motion order, e.g. (1.0, 0.8) = forehand 1.0x, backhand 0.8x. Deterministic (no per-swing
    # randomness); overrides speed_scale_range when set. None = OFF (byte-identical default).
    # Question-bank targets are NOT rescaled (bank overrides target sampling downstream) — the
    # reference swing slows, the physical answer stays the answer.
    speed_scale_per_clip: tuple[float, ...] | None = None
    # 每 clip 的挥拍家族标签("forehand"/"backhand"),顺序 = motion_file 拼接后的 clip 顺序(同
    # strike_phase_per_clip / mount_normal_sign_per_clip)。用途:6-clip 变速烤入列表(正手
    # 0.8/1.0/1.2 + 反手 0.8/1.0/1.1)里,正手 1.0/1.2 变体不再被"clips==0 才是正手"的硬编码误判成
    # 反手(spdmix v2 可行性备忘 2026-07-22 硬绑定一:swing_sign、swing_type 观测、uniform 目标 y 侧
    # 全按这张表取)。None(默认)= 现役行为逐字节不变:内部按"单 clip 正手 / 恰好 2 clip = (正手,
    # 反手)"推导,>2 clip 缺表在查表时 fail-loud——那正是会悄悄训错的场景。显式给出时开机整表校验:
    # 长度必须 == clip 数、值只认这两个字符串、正反手至少各一个,错了当场报错(见
    # resolve_clip_family_is_forehand)。
    clip_family_per_clip: tuple[str, ...] | None = None

    # Same-ball task revision + phase-governor contract.  The disabled path allocates no buffers,
    # draws no RNG and preserves the historical reference clock.  When enabled the complete
    # profile is mandatory (no defaults/partial profiles); train.py installs the same mapping in
    # MotionCommand and RacketTargetCommand from one top-level task.planner_revision block.
    planner_revision_enabled: bool = False
    planner_revision_profile: dict | None = None
    planner_revision_initial_tts_range_s: tuple[float, float] = (0.5, 1.5)
    # Training-only weighted preparation-time distribution.  The complete document is bound in
    # planner_task_revision_training; deployment consumes only the enclosing runtime range above.
    planner_revision_initial_tts_mixture: dict | None = None

    # --- R-c RSI birth fixes (reward_staged_design 2026-07-08 §⑥; defaults OFF = byte-identical) --
    # (i) Skip the first N frames of every swing entry (RSI reset AND wrap — both go through
    # _adaptive_sampling): the v5 GMR clips carry a 3-4 frame IK cold-start transient at frame 0
    # (7.4-15.9 rad/s phantom joint velocities), so births teleported onto frame 0 inherit an
    # instant over-speed reference. N=6 (0.12 s @50 fps) is the design stopgap; once the GMR
    # warm-up source fix lands, N returns to 0 and this flag retires. 人话:出生别传送到 IK 瞬态
    # 帧上,参考从第 N 帧起播。
    rsi_skip_settle_frames: int = 0
    # (ii) Held-RSI births (hold_counter>0) write the DEFAULT-STAND root height instead of the
    # reference frame-0 crouch z: the hold gate already substitutes STAND joints, but the root
    # kept the crouch height (0.78 m vs stand 1.0684 m) -> feet ~0.29 m under the floor -> PhysX
    # depenetration kick at birth. This makes the birth state self-consistent; it is a
    # correctness fix, not an incentive change. 人话:站姿关节配站姿身高,脚不再穿地被弹飞。
    rsi_hold_root_stand_z: bool = False

    # --- 防同步 stagger_initial_clock (metric-sync fix 2026-07-09; default OFF = byte-identical) --
    # 4096 envs resumed at the same instant + low fall rate => synchronized mass timeouts
    # (episode_length sawtooth 52->485) => every EMA metric reads a queue oscillation. ON adds two
    # ONE-SHOT uniform biases (see MotionCommand.__init__ / _resample_command / _update_command):
    # (a) first true reset per env: hold += U[0, stagger_hold_max_steps] (swing phases spread);
    # (b) first step after construction: episode clock += U[0, max_episode_length) (episode
    # boundaries spread, permanently). 人话:把所有 env 的节拍随机错开,治 EMA 指标同步振荡;
    # 默认关=现役可比,新点火臂建议开。
    stagger_initial_clock: bool = False
    # (a) 的偏置上限(控制步): 默认 150 步 = 3 s @ 50 Hz ≈ 一个 hold+挥拍 周期。
    stagger_hold_max_steps: int = 150

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
