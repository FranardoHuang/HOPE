"""Dependency-light FullMDP active action rows shared by Isaac and MuJoCo.

The fresh curriculum currently selects one action only.  Keep that executable
N=1 view explicit instead of loading 72 cold rows that no runtime selector can
consume.  This also keeps the physical-ready, policy prior and teacher motion
on the same content-addressed Take061 source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import stat
import sys

try:
    if __package__:
        from . import action_ball_manifest as _manifest
    else:  # Dependency-light direct import from the MuJoCo lane.
        import action_ball_manifest as _manifest
except ImportError:  # Spec-loaded tests may provide a non-package MDP parent.
    _manifest_name = (
        f"{__package__}.action_ball_manifest"
        if __package__
        else "action_ball_manifest"
    )
    _manifest = sys.modules.get(_manifest_name)
    if _manifest is None:
        _manifest_spec = importlib.util.spec_from_file_location(
            _manifest_name, Path(__file__).resolve().with_name("action_ball_manifest.py")
        )
        if _manifest_spec is None or _manifest_spec.loader is None:
            raise ImportError("cannot load the portable action manifest source")
        _manifest = importlib.util.module_from_spec(_manifest_spec)
        sys.modules[_manifest_name] = _manifest
        _manifest_spec.loader.exec_module(_manifest)

try:
    if __package__:
        from . import racket_contact_geometry as _racket_geometry
    else:
        import racket_contact_geometry as _racket_geometry
except ImportError:
    _geometry_name = (
        f"{__package__}.racket_contact_geometry"
        if __package__
        else "racket_contact_geometry"
    )
    _racket_geometry = sys.modules.get(_geometry_name)
    if _racket_geometry is None:
        _geometry_spec = importlib.util.spec_from_file_location(
            _geometry_name,
            Path(__file__).resolve().with_name("racket_contact_geometry.py"),
        )
        if _geometry_spec is None or _geometry_spec.loader is None:
            raise ImportError("cannot load portable racket geometry")
        _racket_geometry = importlib.util.module_from_spec(_geometry_spec)
        sys.modules[_geometry_name] = _racket_geometry
        _geometry_spec.loader.exec_module(_racket_geometry)


ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_KIND = (
    "action_ball_full_mdp_code_owned_diagnostic_catalog_v1"
)
ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_ACTION_COUNT = 1
ACTION_BALL_FULL_MDP_FRESH_ACTION_SLOT = 0
PINNED_DIAGNOSTIC_MANIFEST_RELATIVE_PATH = (
    "configs/action_ball_full_mdp_n1_take061_measured_a3p0807_v5_20260828.json"
)
PINNED_DIAGNOSTIC_MANIFEST_FILE_SHA256 = (
    "a13bc6533cf22a0695a861470152683859f35afa7505320fcbe474f76addf597"
)
PINNED_DIAGNOSTIC_MANIFEST_CANONICAL_SHA256 = (
    "c759f13402e1a97b54431076eb3279b8eb7e8dfffaba94dbb92e842817da4277"
)
FRESH_POLICY_STEP_S = 0.02
# Give every fresh row one complete H48 balance rollout before task exposure.
# The exact dynamic-ready receipt observes 60 stable policy steps; revealing
# at 48 stays inside that measured prefix instead of extrapolating it to the
# old 295-tick wait.  Post-shot R07 recovery remains evidence, not admission.
FRESH_FIRST_REVEAL_TICK = 48
FRESH_RECOVERY_END_OFFSET_TICKS = 77
FRESH_HIDDEN_GAP_TICKS = 2
FRESH_REFERENCE_DUE_COUNT = 6
FRESH_REFERENCE_DUE_TICKS = (48, 233, 418, 603, 788, 973)
FRESH_EPISODE_HORIZON_TICKS = 1500
# Raw actor-clock sentinel shared by both backends.  A negative value is
# outside the domain of every real countdown and makes schedule exhaustion
# distinguishable from the sixth shot's still-valid settlement boundary.
FRESH_SCHEDULE_EXHAUSTED_TIME_TO_NEXT_OPPORTUNITY_S = -1.0


@dataclass(frozen=True)
class ActionBallFullMdpDiagnosticCatalogTable:
    """The existing nine-column cold Motion/Racket construction ABI."""

    manifest_file_sha256: str
    manifest_canonical_sha256: str
    action_order: tuple[str, ...]
    action_uids: tuple[int, ...]
    motion_files: tuple[str, ...]
    motion_sha256: tuple[str, ...]
    clip_family_per_clip: tuple[str, ...]
    strike_phase_per_clip: tuple[float, ...]
    mount_normal_sign_per_clip: tuple[float, ...]


@dataclass(frozen=True)
class PortableActionCenterRow:
    """One action identity, centre question, and cold strike-reference row."""

    action_slot: int
    action_id: str
    action_uid: int
    family: str
    mount_normal_sign: int
    motion_file: str
    motion_sha256: str
    strike_phase: float
    reference_t_hit_s: float
    reference_t_cycle_s: float
    reference_racket_site_speed_mps: float
    reaction_margin_s: float
    teacher_rate_min: float
    teacher_rate_max: float
    contact_offset_center_b_yaw_m: tuple[float, float, float]
    time_to_contact_center_s: float
    incoming_direction_center_b_yaw: tuple[float, float, float]
    incoming_speed_center_mps: float
    spin_direction_center_b_yaw: tuple[float, float, float]
    spin_magnitude_center_radps: float
    base_spawn_center_w_xy_m: tuple[float, float]
    base_travel_center_b_yaw_xy_m: tuple[float, float]
    reference_racket_site_position_w_m: tuple[float, float, float]
    reference_racket_quat_wxyz: tuple[float, float, float, float]
    reference_racket_angular_velocity_w_radps: tuple[float, float, float]
    reference_racket_site_velocity_w_mps: tuple[float, float, float]
    reference_raw_face_normal_w: tuple[float, float, float]
    reference_reach_offset_xy_m: tuple[float, float]
    reference_base_root_quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class PortableActionCenterTable:
    """The exact bank plus the current single-action selector declaration."""

    manifest_file_sha256: str
    manifest_canonical_sha256: str
    landing_aim_center_w_xy_m: tuple[float, float]
    actions: tuple[PortableActionCenterRow, ...]
    fresh_action_slot: int = ACTION_BALL_FULL_MDP_FRESH_ACTION_SLOT

    @property
    def fresh_action(self) -> PortableActionCenterRow:
        return self.actions[self.fresh_action_slot]


@dataclass(frozen=True)
class PortableFreshCadence:
    """Cold recurring due schedule shared by the Isaac and MuJoCo lanes.

    A due tick is only an opportunity.  Its row-wise verdict remains
    state-dependent ACCEPT, DEFER, CENSOR, or REJECT.
    """

    first_reveal_tick: int
    maximum_task_close_ticks: int
    cadence_ticks: int
    reference_due_ticks: tuple[int, ...]
    episode_horizon_ticks: int


def derive_portable_fresh_cadence(
    table: PortableActionCenterTable,
) -> PortableFreshCadence:
    """Derive the 30 s cadence from the sealed active-action timing row."""

    if type(table) is not PortableActionCenterTable or not table.actions:
        raise ValueError("portable fresh cadence requires the exact action bank")
    close_ticks = []
    for slot, action in enumerate(table.actions):
        duration_s = action.time_to_contact_center_s + (
            action.reference_t_cycle_s - action.reference_t_hit_s
        ) / action.teacher_rate_min
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(
                f"portable fresh cadence timing is invalid at slot {slot}"
            )
        close_ticks.append(
            math.ceil(duration_s / FRESH_POLICY_STEP_S - 1.0e-12)
        )
    maximum_close = max(close_ticks)
    cadence = (
        maximum_close
        + FRESH_RECOVERY_END_OFFSET_TICKS
        + FRESH_HIDDEN_GAP_TICKS
    )
    due_ticks = tuple(
        FRESH_FIRST_REVEAL_TICK + cadence * ordinal
        for ordinal in range(FRESH_REFERENCE_DUE_COUNT)
    )
    if (
        maximum_close != 106
        or cadence != 185
        or due_ticks != FRESH_REFERENCE_DUE_TICKS
        # Every advertised opportunity must fit its complete construction
        # window.  The six-shot budget is explicit even though the shorter
        # initial balance prefix leaves unused horizon after shot six.
        or due_ticks[-1] + cadence >= FRESH_EPISODE_HORIZON_TICKS
    ):
        raise ValueError("portable fresh cadence differs from the frozen schedule")
    return PortableFreshCadence(
        first_reveal_tick=FRESH_FIRST_REVEAL_TICK,
        maximum_task_close_ticks=maximum_close,
        cadence_ticks=cadence,
        reference_due_ticks=due_ticks,
        episode_horizon_ticks=FRESH_EPISODE_HORIZON_TICKS,
    )


def _load_catalog_source():
    repo_root = Path(__file__).resolve().parents[8]
    manifest_path = (
        repo_root / PINNED_DIAGNOSTIC_MANIFEST_RELATIVE_PATH
    ).resolve()
    try:
        manifest_path.relative_to(repo_root)
        loaded = _manifest.load_action_ball_manifest(
            manifest_path,
            expected_sha256=PINNED_DIAGNOSTIC_MANIFEST_FILE_SHA256,
            verify_referenced_assets=False,
            require_formal_admission=False,
        )
    except Exception as exc:
        raise ValueError(
            "code-owned full-MDP diagnostic catalog is absent or changed"
        ) from exc
    if (
        loaded.canonical_sha256
        != PINNED_DIAGNOSTIC_MANIFEST_CANONICAL_SHA256
        or loaded.manifest.schema_version != _manifest.SCHEMA_VERSION
    ):
        raise ValueError(
            "code-owned full-MDP diagnostic catalog schema/content changed"
        )
    actions = loaded.manifest.actions
    action_order = tuple(loaded.manifest.action_order)
    if (
        len(actions) != ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_ACTION_COUNT
        or len(action_order)
        != ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_ACTION_COUNT
        or action_order != tuple(action.action_id for action in actions)
    ):
        raise ValueError(
            "code-owned full-MDP diagnostic catalog is not the exact active N=1 order"
        )
    motion_files = []
    motion_sha256 = []
    for slot, action in enumerate(actions):
        candidate = repo_root.joinpath(*Path(action.motion_path).parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(repo_root)
            metadata = resolved.stat()
            payload = resolved.read_bytes()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"diagnostic catalog motion[{slot}] is absent or escapes the repository"
            ) from exc
        digest = hashlib.sha256(payload).hexdigest()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or digest != action.motion_sha256
        ):
            raise ValueError(f"diagnostic catalog motion[{slot}] bytes differ")
        motion_files.append(str(resolved))
        motion_sha256.append(digest)
    return loaded, tuple(motion_files), tuple(motion_sha256)


def load_action_ball_full_mdp_diagnostic_catalog_table(
) -> ActionBallFullMdpDiagnosticCatalogTable:
    """Re-read the code-pinned active N=1 manifest and motion file."""

    loaded, motion_files, motion_sha256 = _load_catalog_source()
    actions = loaded.manifest.actions
    return ActionBallFullMdpDiagnosticCatalogTable(
        manifest_file_sha256=loaded.file_sha256,
        manifest_canonical_sha256=loaded.canonical_sha256,
        action_order=tuple(loaded.manifest.action_order),
        action_uids=tuple(int(action.action_uid) for action in actions),
        motion_files=motion_files,
        motion_sha256=motion_sha256,
        clip_family_per_clip=tuple(action.family for action in actions),
        strike_phase_per_clip=tuple(float(action.strike_phase) for action in actions),
        mount_normal_sign_per_clip=tuple(
            float(action.mount_normal_sign) for action in actions
        ),
    )


def _quat_apply_wxyz_numpy(quaternion, vector):
    import numpy as np

    xyz = quaternion[1:]
    return vector + np.float32(2.0) * (
        quaternion[0] * np.cross(xyz, vector)
        + np.cross(xyz, np.cross(xyz, vector))
    )


def _portable_reference_row(
    *,
    motion_file: str,
    strike_phase: float,
    mount_normal_sign: int,
) -> dict[str, tuple[float, ...]]:
    """Mirror the current cold Racket FK builder from one sealed NPZ.

    This is cold construction only.  The wrist/site constants come from the
    shared racket geometry module, and the velocity uses the same two-frame
    centered finite difference as ``RacketTargetCommand``.
    """

    import numpy as np

    with np.load(motion_file, allow_pickle=False) as data:
        required = {
            "fps",
            "body_names",
            "body_pos_w",
            "body_quat_w",
            "body_ang_vel_w",
            "kinematics_schema_version",
            "measured_racket_schema_version",
            "measured_racket_robot_mount_normal_sign",
        }
        if required.difference(data.files):
            raise ValueError("portable action reference NPZ schema differs")
        names = tuple(str(name) for name in data["body_names"].tolist())
        wrist_name = _racket_geometry.GEOMETRY_SOURCE_PAYLOAD[
            "official_wrist_body_name"
        ]
        if names.count(wrist_name) != 1:
            raise ValueError("portable action reference wrist identity differs")
        position = np.asarray(data["body_pos_w"], dtype=np.float32)
        quaternion = np.asarray(data["body_quat_w"], dtype=np.float32)
        angular = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        if (
            position.ndim != 3
            or quaternion.shape != (*position.shape[:2], 4)
            or angular.shape != position.shape
            or position.shape[1] != len(names)
            or int(np.asarray(data["kinematics_schema_version"]).reshape(-1)[0])
            != 2
            or int(np.asarray(data["measured_racket_schema_version"]).reshape(-1)[0])
            != 4
            or int(
                np.asarray(
                    data["measured_racket_robot_mount_normal_sign"]
                ).reshape(-1)[0]
            )
            != mount_normal_sign
        ):
            raise ValueError("portable action reference array/sign contract differs")
        fps_values = np.asarray(data["fps"]).reshape(-1)
        if fps_values.size != 1 or int(fps_values[0]) <= 0:
            raise ValueError("portable action reference fps differs")
        frame_count = position.shape[0]
        if frame_count < 2:
            raise ValueError("portable action reference has no strike segment")
        strike = round(float(strike_phase) * (frame_count - 1))
        if strike < 0 or strike >= frame_count:
            raise ValueError("portable action reference strike frame differs")
        wrist = names.index(wrist_name)
        offset = np.asarray(
            _racket_geometry.RACKET_SITE_OFFSET_WRIST_M, dtype=np.float32
        )

        def site_at(frame: int):
            frame = max(0, min(frame_count - 1, int(frame)))
            return position[frame, wrist] + _quat_apply_wxyz_numpy(
                quaternion[frame, wrist], offset
            )

        site_position = site_at(strike)
        racket_quat = quaternion[strike, wrist]
        racket_angular = angular[strike, wrist]
        clean_window = 2
        step_s = np.float32(1.0 / float(fps_values[0]))
        site_velocity = (
            site_at(strike + clean_window) - site_at(strike - clean_window)
        ) / np.float32(2 * clean_window) / step_s
        raw_normal = _quat_apply_wxyz_numpy(
            racket_quat, np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        )
        raw_normal = raw_normal / (
            np.linalg.norm(raw_normal).astype(np.float32) + np.float32(1.0e-6)
        )
        base_position = position[strike, 0]
        values = (
            site_position,
            racket_quat,
            racket_angular,
            site_velocity,
            raw_normal,
            site_position[:2] - base_position[:2],
            quaternion[strike, 0],
        )
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError("portable action reference contains non-finite values")
        return {
            "reference_racket_site_position_w_m": tuple(
                float(value) for value in site_position
            ),
            "reference_racket_quat_wxyz": tuple(
                float(value) for value in racket_quat
            ),
            "reference_racket_angular_velocity_w_radps": tuple(
                float(value) for value in racket_angular
            ),
            "reference_racket_site_velocity_w_mps": tuple(
                float(value) for value in site_velocity
            ),
            "reference_raw_face_normal_w": tuple(
                float(value) for value in raw_normal
            ),
            "reference_reach_offset_xy_m": tuple(
                float(value) for value in site_position[:2] - base_position[:2]
            ),
            "reference_base_root_quat_wxyz": tuple(
                float(value) for value in quaternion[strike, 0]
            ),
        }


def load_portable_action_center_table() -> PortableActionCenterTable:
    """Load exact action identities and centre questions without Isaac owners."""

    loaded, motion_files, motion_sha256 = _load_catalog_source()
    actions = loaded.manifest.actions
    if len(actions) != len(motion_files) or len(actions) != len(motion_sha256):
        raise ValueError("portable action catalog columns differ in length")
    rows = []
    for slot, (action, motion_file, motion_digest) in enumerate(
        zip(actions, motion_files, motion_sha256)
    ):
        profile = action.ball_profile
        reference = _portable_reference_row(
            motion_file=motion_file,
            strike_phase=float(action.strike_phase),
            mount_normal_sign=int(action.mount_normal_sign),
        )
        rows.append(
            PortableActionCenterRow(
                action_slot=slot,
                action_id=action.action_id,
                action_uid=int(action.action_uid),
                family=action.family,
                mount_normal_sign=int(action.mount_normal_sign),
                motion_file=motion_file,
                motion_sha256=motion_digest,
                strike_phase=float(action.strike_phase),
                reference_t_hit_s=float(action.reference_t_hit_s),
                reference_t_cycle_s=float(action.reference_t_cycle_s),
                reference_racket_site_speed_mps=float(
                    action.reference_racket_site_speed_mps
                ),
                reaction_margin_s=float(action.reaction_margin_s),
                teacher_rate_min=float(action.teacher_rate_min),
                teacher_rate_max=float(action.teacher_rate_max),
                contact_offset_center_b_yaw_m=tuple(
                    profile.contact_offset_center_b_yaw_m
                ),
                time_to_contact_center_s=float(profile.time_to_contact_center_s),
                incoming_direction_center_b_yaw=tuple(
                    profile.incoming_direction_center_b_yaw
                ),
                incoming_speed_center_mps=float(profile.incoming_speed_center_mps),
                spin_direction_center_b_yaw=tuple(
                    profile.spin_direction_center_b_yaw
                ),
                spin_magnitude_center_radps=float(
                    profile.spin_magnitude_center_radps
                ),
                base_spawn_center_w_xy_m=tuple(profile.base_spawn_center_w_xy_m),
                base_travel_center_b_yaw_xy_m=tuple(
                    profile.base_travel_center_b_yaw_xy_m
                ),
                **reference,
            )
        )
    return PortableActionCenterTable(
        manifest_file_sha256=loaded.file_sha256,
        manifest_canonical_sha256=loaded.canonical_sha256,
        landing_aim_center_w_xy_m=tuple(
            loaded.manifest.landing_aim.center_w_xy_m
        ),
        actions=tuple(rows),
    )


__all__ = [
    "ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_KIND",
    "ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_ACTION_COUNT",
    "ACTION_BALL_FULL_MDP_FRESH_ACTION_SLOT",
    "PINNED_DIAGNOSTIC_MANIFEST_RELATIVE_PATH",
    "PINNED_DIAGNOSTIC_MANIFEST_FILE_SHA256",
    "PINNED_DIAGNOSTIC_MANIFEST_CANONICAL_SHA256",
    "FRESH_POLICY_STEP_S",
    "FRESH_FIRST_REVEAL_TICK",
    "FRESH_RECOVERY_END_OFFSET_TICKS",
    "FRESH_HIDDEN_GAP_TICKS",
    "FRESH_REFERENCE_DUE_COUNT",
    "FRESH_REFERENCE_DUE_TICKS",
    "FRESH_EPISODE_HORIZON_TICKS",
    "FRESH_SCHEDULE_EXHAUSTED_TIME_TO_NEXT_OPPORTUNITY_S",
    "ActionBallFullMdpDiagnosticCatalogTable",
    "PortableActionCenterRow",
    "PortableActionCenterTable",
    "PortableFreshCadence",
    "derive_portable_fresh_cadence",
    "load_action_ball_full_mdp_diagnostic_catalog_table",
    "load_portable_action_center_table",
]
