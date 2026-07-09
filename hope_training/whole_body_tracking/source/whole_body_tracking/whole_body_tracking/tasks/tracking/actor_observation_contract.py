from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ActorObservationTerm:
    name: str
    dim: int
    deploy_source: str
    description: str


@dataclass(frozen=True)
class ActorObservationContract:
    name: str
    obs_mode: str
    total_dim: int
    terms: tuple[ActorObservationTerm, ...]
    legacy_names: tuple[str, ...] = ()

    @property
    def layout(self) -> tuple[tuple[str, int], ...]:
        return tuple((term.name, term.dim) for term in self.terms)


FULL = ActorObservationContract(
    name="full",
    obs_mode="full",
    total_dim=180,
    terms=(
        ActorObservationTerm("command", 62, "reference_clip", "reference joint positions and velocities"),
        ActorObservationTerm(
            "motion_anchor_pos_b",
            3,
            "sim_or_localizer",
            "reference torso position error; requires world base position",
        ),
        ActorObservationTerm(
            "motion_anchor_ori_b",
            6,
            "imu_plus_reference_clip",
            "reference torso orientation error",
        ),
        ActorObservationTerm("base_ang_vel", 3, "imu", "pelvis angular velocity in the body frame"),
        ActorObservationTerm("joint_pos", 31, "encoders", "joint position offset from default"),
        ActorObservationTerm("joint_vel", 31, "encoders", "joint velocities"),
        ActorObservationTerm("actions", 31, "runtime_state", "previous policy action"),
        ActorObservationTerm("projected_gravity", 3, "imu", "gravity direction in the base frame"),
        ActorObservationTerm(
            "base_target_pos_b",
            2,
            "planner_plus_localizer",
            "desired base XY target relative to the current base",
        ),
        ActorObservationTerm(
            "racket_target_pos_b",
            3,
            "planner_plus_localizer",
            "desired racket position relative to the current base",
        ),
        ActorObservationTerm("racket_target_vel_w", 3, "planner", "desired racket velocity in the world frame"),
        ActorObservationTerm("time_to_strike", 1, "reference_clock", "time remaining until strike"),
        ActorObservationTerm("swing_type", 1, "reference_clip", "forehand (+1) or backhand (-1) flag"),
    ),
)

DEPLOY_PARITY = ActorObservationContract(
    name="deploy_parity",
    obs_mode="deploy_parity",
    total_dim=175,
    legacy_names=("real_sensor_only",),
    terms=(
        ActorObservationTerm("command", 62, "reference_clip", "reference joint positions and velocities"),
        ActorObservationTerm(
            "motion_anchor_ori_b",
            6,
            "imu_plus_reference_clip",
            "reference torso orientation error",
        ),
        ActorObservationTerm("base_ang_vel", 3, "imu", "pelvis angular velocity in the body frame"),
        ActorObservationTerm("joint_pos", 31, "encoders", "joint position offset from default"),
        ActorObservationTerm("joint_vel", 31, "encoders", "joint velocities"),
        ActorObservationTerm("actions", 31, "runtime_state", "previous policy action"),
        ActorObservationTerm("projected_gravity", 3, "imu", "gravity direction in the base frame"),
        ActorObservationTerm(
            "racket_target_pos_b",
            3,
            "planner_plus_racket_fk",
            "desired racket position relative to the current racket FK; no world base position",
        ),
        ActorObservationTerm("racket_target_vel_w", 3, "planner", "desired racket velocity in the world frame"),
        ActorObservationTerm("time_to_strike", 1, "reference_clock", "time remaining until strike"),
        ActorObservationTerm("swing_type", 1, "reference_clip", "forehand (+1) or backhand (-1) flag"),
    ),
)

HITTER_FOOTWORK = ActorObservationContract(
    name="hitter_footwork",
    obs_mode="hitter_footwork",
    total_dim=177,
    terms=(
        ActorObservationTerm("command", 62, "reference_clip", "reference joint positions and velocities"),
        ActorObservationTerm(
            "motion_anchor_ori_b",
            6,
            "imu_plus_reference_clip",
            "reference torso orientation error",
        ),
        ActorObservationTerm("base_ang_vel", 3, "imu", "pelvis angular velocity in the body frame"),
        ActorObservationTerm("joint_pos", 31, "encoders", "joint position offset from default"),
        ActorObservationTerm("joint_vel", 31, "encoders", "joint velocities"),
        ActorObservationTerm("actions", 31, "runtime_state", "previous policy action"),
        ActorObservationTerm("projected_gravity", 3, "imu", "gravity direction in the base frame"),
        ActorObservationTerm(
            "base_target_pos_b",
            2,
            "planner_plus_mocap",
            "desired base XY station relative to the current base (yaw-heading frame); mocap base "
            "position at deploy — relative Δ only, Δ=0 graceful fallback on mocap dropout",
        ),
        ActorObservationTerm(
            "racket_target_pos_b",
            3,
            "planner_plus_racket_fk",
            "desired racket position relative to the current racket FK; no world base position",
        ),
        ActorObservationTerm("racket_target_vel_w", 3, "planner", "desired racket velocity in the world frame"),
        ActorObservationTerm("time_to_strike", 1, "reference_clock", "time remaining until strike"),
        ActorObservationTerm("swing_type", 1, "reference_clip", "forehand (+1) or backhand (-1) flag"),
    ),
)

# Stage-1 face-command contract (2026-07-06): deploy_parity + the +4D face-command channel
# appended LAST — racket_target_normal_cmd = demanded face normal (3, world frame, from the
# question bank / planner) + spin-rho placeholder (1, zero-filled until the S3 spin tier).
# Matches train.py's face_command_obs wiring and the MuJoCo evaluator's 179-D support (bec7673).
# Contract-day note: the 181-D single cut (this + Hitter's +2 station dims) supersedes it later.
DEPLOY_PARITY_FACE179 = ActorObservationContract(
    name="deploy_parity_face179",
    obs_mode="deploy_parity",
    total_dim=179,
    terms=DEPLOY_PARITY.terms
    + (
        ActorObservationTerm(
            "racket_target_normal_cmd",
            4,
            "planner",
            "demanded racket face normal (world, 3) + spin-rho placeholder (zero until S3)",
        ),
    ),
)

# R10c station-anchor contract (2026-07-09, franco:"planner 的 p_base 应该加进去——就算不需要
# 移动,它也是一个锚"): face179 + station_anchor_err_b(2) appended LAST = 181.
# 人话:在 179(拍面通道)之后再追加 2 维世界系站位锚误差(出生点常数 − 当前 base XY,base 系),
# 躯干漂了这两个数自己变大 = R9a 删缰绳后丢世界系基准的任务通道解法。
# ⚠ 布局取舍(有意为之,和契约日蓝图不同):契约日 181 蓝图 = 175+站位2+拍面3+ρ1(站位在拍面
# 前,即 Hitter 177 的第 167 列位置)。这里为了现役 179 存档能走 pad_obs_cols 纯尾部扩列热启
# (第 0 步逐位相同),站位排在拍面后。契约日统一布局时再做一次插列手术或重训对齐——到时以
# 蓝图为准,这个契约名保留作训练侧过渡。
DEPLOY_PARITY_STATION181 = ActorObservationContract(
    name="deploy_parity_station181",
    obs_mode="deploy_parity",
    total_dim=181,
    terms=DEPLOY_PARITY_FACE179.terms
    + (
        ActorObservationTerm(
            "station_anchor_err_b",
            2,
            "planner_plus_mocap",
            "world-frame station anchor (spawn-point constant) minus current base XY, rotated into "
            "the yaw-heading base frame; mocap base position at deploy — relative Δ only, Δ=0 "
            "graceful fallback on mocap dropout (same recipe as Hitter's base_target_pos_b)",
        ),
    ),
)

CONTRACTS = {
    FULL.name: FULL,
    DEPLOY_PARITY.name: DEPLOY_PARITY,
    DEPLOY_PARITY_FACE179.name: DEPLOY_PARITY_FACE179,
    DEPLOY_PARITY_STATION181.name: DEPLOY_PARITY_STATION181,
    HITTER_FOOTWORK.name: HITTER_FOOTWORK,
    FULL.obs_mode: FULL,
    DEPLOY_PARITY.obs_mode: DEPLOY_PARITY,
    HITTER_FOOTWORK.obs_mode: HITTER_FOOTWORK,
    **{alias: DEPLOY_PARITY for alias in DEPLOY_PARITY.legacy_names},
}


def resolve_actor_observation_contract(name: str | None) -> ActorObservationContract | None:
    if name is None:
        return None
    key = str(name).strip()
    if not key:
        return None
    if key not in CONTRACTS:
        known = ", ".join(sorted(CONTRACTS))
        raise ValueError(f"Unknown actor observation contract '{name}'. Known values: {known}")
    return CONTRACTS[key]


def policy_layout_from_env(env) -> tuple[tuple[str, int], ...]:
    observation_manager = env.observation_manager
    names = list(observation_manager.active_terms["policy"])
    dims = observation_manager.group_obs_term_dim["policy"]
    flat_dims = []
    for dim in dims:
        if isinstance(dim, (tuple, list)):
            flat_dims.append(int(dim[0]))
        else:
            flat_dims.append(int(dim))
    return tuple(zip(names, flat_dims))


def total_policy_dim_from_env(env) -> int:
    total = env.observation_manager.group_obs_dim["policy"]
    if isinstance(total, (tuple, list)):
        return int(total[0])
    return int(total)


def infer_actor_observation_contract(env) -> ActorObservationContract | None:
    layout = policy_layout_from_env(env)
    total_dim = total_policy_dim_from_env(env)
    for contract in (FULL, DEPLOY_PARITY, HITTER_FOOTWORK):
        if layout == contract.layout and total_dim == contract.total_dim:
            return contract
    return None


def validate_actor_observation_contract(env, expected: str | ActorObservationContract) -> ActorObservationContract:
    contract = (
        expected
        if isinstance(expected, ActorObservationContract)
        else resolve_actor_observation_contract(expected)
    )
    if contract is None:
        raise ValueError("Expected actor observation contract must not be empty.")
    layout = policy_layout_from_env(env)
    total_dim = total_policy_dim_from_env(env)
    if layout != contract.layout or total_dim != contract.total_dim:
        raise ValueError(
            "Actor observation contract mismatch.\n"
            f"Expected {contract.name} ({contract.total_dim}D): {format_layout(contract.layout)}\n"
            f"Actual   ({total_dim}D): {format_layout(layout)}"
        )
    return contract


def removed_terms_vs_full(contract: str | ActorObservationContract) -> tuple[ActorObservationTerm, ...]:
    resolved = contract if isinstance(contract, ActorObservationContract) else resolve_actor_observation_contract(contract)
    if resolved is None:
        return ()
    removed = []
    active = {term.name for term in resolved.terms}
    for term in FULL.terms:
        if term.name not in active:
            removed.append(term)
    return tuple(removed)


def format_layout(layout: Iterable[tuple[str, int]]) -> str:
    return ", ".join(f"{name}:{dim}" for name, dim in layout)
