from __future__ import annotations

import hashlib
import io
import math
import numpy as np
import os
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
        articulation_body_names: Sequence[str],
        selected_body_names: Sequence[str],
        device: str = "cpu",
        allow_legacy_link_origin_velocity: bool = False,
    ):
        files = [motion_file] if isinstance(motion_file, str) else list(motion_file)
        if not files:
            raise ValueError("MotionLoader needs at least one motion file")
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
        for f in files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"Invalid motion file path: {f}")
            data = np.load(f)
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
    if CLIP_FAMILY_FOREHAND not in families or CLIP_FAMILY_BACKHAND not in families:
        raise ValueError(
            "clip_family_per_clip must contain at least one forehand and one backhand clip, got "
            f"{families} — the unified policy keys swing_sign, the swing-type observation and the "
            "target side off both families"
        )
    return tuple(value == CLIP_FAMILY_FOREHAND for value in families)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            self.cfg.motion_file,
            self.body_indexes,
            articulation_body_names=self.robot.body_names,
            selected_body_names=self.cfg.body_names,
            device=self.device,
            allow_legacy_link_origin_velocity=bool(
                self.cfg.allow_legacy_link_origin_velocity
            ),
        )
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
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

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
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        alv = self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
        return alv * self.speed_scale[:, None] if self.retiming_active else alv

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        aav = self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
        return aav * self.speed_scale[:, None] if self.retiming_active else aav

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

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self._multiseg:
            # HITTER unified policy: each new swing uniformly samples the swing TYPE (clip) and starts at
            # that clip's first frame (reference-state-init at the swing start). The adaptive failure-bin
            # curriculum is single-clip BeyondMimic machinery and is bypassed here.
            n = len(env_ids)
            if n > 0:
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

    def _resample_command(self, env_ids: Sequence[int]):
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

        # Pre-swing HOLD (Phase A): freeze the reference at the swing's first frame for a random
        # number of control steps ("the ball is not reaching yet"). Applies to resets AND wraps.
        lo, hi = self.cfg.hold_steps_range
        self.hold_counter[env_ids_t] = torch.randint(int(lo), int(hi) + 1, (len(env_ids_t),), device=self.device)
        # A wrap can resample a new hold late inside _update_command. Publish its state now so
        # downstream rewards/terminations on this same control step do not see the old swing mask.
        self.metrics["in_hold"][env_ids_t] = (self.hold_counter[env_ids_t] > 0).float()

        # stagger (a): each env's FIRST true reset adds a uniform hold bias, spreading the swing/
        # strike phases of a same-instant reset cohort across ~one swing period. One-shot per env;
        # wraps and every later reset draw the plain hold range, so steady-state behavior is
        # unchanged. The stand/post-swing min-hold clamps below are min= clamps — the bias
        # survives them. Default OFF (see cfg.stagger_initial_clock): no RNG draw, byte-identical.
        if self._stagger_hold_pending is not None and not self._resampling_from_wrap:
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
        # Pre-swing HOLD: held envs keep the reference frozen at the swing's first frame
        # ("waiting for the ball"); everyone else advances the clip clock.
        held = self.hold_counter > 0
        self.hold_counter = torch.clamp(self.hold_counter - 1, min=0)
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
        event_owned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._event_scheduler is not None:
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
        if self._multiseg:
            # Wrap at the END of the env's current clip/segment, not the global concatenated end.
            seg_end = self.motion.seg_start[self.clip_id] + self.motion.seg_len[self.clip_id]
            # Once an exact-strike origin arms T1, natural clip completion is a carry-state wait,
            # not permission to draw or teleport to another question.  Clamp the old reference at
            # its final native frame until the immutable reveal installs the next clip.
            clamp = event_owned & (self.time_steps >= seg_end)
            if bool(clamp.any()):
                self.time_steps[clamp] = seg_end[clamp] - 1
                self.time_steps_f[clamp] = self.time_steps[clamp].float()
            wrap_ids = torch.where((~event_owned) & (self.time_steps >= seg_end))[0]
            # DEPLOY-PARITY CLIP SWITCH (venue falls 2026-07-04): the runner's reference clock flips
            # clip_id whenever the planner re-sides the target — at an ARBITRARY mid-swing moment —
            # and the reference jumps to the new clip's first frame (pp_reference_clock.hpp clamps
            # tts-large to seg_start). Training previously only switched clips at clip END, so the
            # policy never saw that discontinuity and falls at 准备/正手/反手 switches on hardware.
            # With per-step prob clip_switch_prob an env aborts its swing operator-style and routes
            # through the SAME wrap-resample path (uniform new clip, frame 0, hold, fresh target).
            # NOTE: aborted swings count as uncompleted starts (slight completion-rate deflation).
            if float(self.cfg.clip_switch_prob) > 0.0:
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
                (~event_owned) & (self.time_steps >= self.motion.time_step_total)
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
