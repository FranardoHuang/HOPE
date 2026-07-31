"""F 轴间隔触发的持续水平力推事件(与速度推 push_by_setting_velocity 同冲量可比)。

人话:每隔几秒(interval 事件触发)朝水平随机方向,对 pelvis_link 施加一个幅度 ``force_n``
牛的恒力,持续 ``duration_steps`` 个控制步(默认 15 步 = 0.30 s @ 50 Hz)后清零。与速度推档位
按同冲量对表:Δv_equiv = force_n × duration_s / m_robot(换算在训练合同里做,这里只施力)。

机制(为什么是"两个事件"):

* ``push_by_applying_wrench`` —— interval 触发的施力事件。触发时对被抽中的 env 采样水平随机
  方向(世界系 z 分量 0、无力矩),把恒力写进 Isaac 的外力缓冲并登记逐 env 到期步。它每次被
  调用时还会先清一遍全体过期力(顺手兜底)。
* ``sweep_expired_force_pushes`` —— 高频小间隔的清扫事件。Isaac 的 interval 事件 **不是逐步
  调用** 的:施力事件下一次被调可能在十几秒后,单靠它自己力永远清不掉。清扫事件以
  interval=(control_dt, control_dt) 挂进 EventManager = 每个控制步都跑,到期立刻清零。
* ``push_combined_exclusive`` —— 合并互斥推(Franco 2026-07-25):速度踢与力推合并成一个
  interval 事件,每次触发按 force_prob 逐 env 抽签二选一,防两种随机推同帧叠加;力分支写
  同一本账本,所以合并模式同样必须挂清扫事件。

坐标系诚实说明(Isaac Lab 2.1):``Articulation.set_external_force_and_torque`` 的力在 **BODY
系** 生效(见 shadow_ball.py / table_tennis_env.py 的同款注释),所以这里在触发瞬间用当前
pelvis 姿态把世界系水平力旋进 body 系。持续期内力在 body 系保持不变——机器人被推歪多少,
世界系方向就漂多少(≤0.3 s,漂移很小);合同里如实写这个语义,不假装它是世界系恒力。

施力点诚实说明(Yikang V9 教训):PhysX ``apply_forces_and_torques_at_position`` 在
positions=None 时把力施加在 **link 原点**(link frame origin),不是 COM。合同字段
``application_point="pelvis_link_origin"`` 必须照实写,不许标 COM。

Reset 语义:env 中途 reset 不会立即清力(残余力最多再活到自己的到期步,≤duration_steps 个
控制步);到期清扫独立于 reset,兜底始终成立。

Fail-closed 纪律:body 找不到/重名、参数非法、state 被别的 body 写过,一律 raise,不静默。
本模块只依赖 torch,可在无 Isaac 的机器上按文件路径单测(mock asset/env)。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# 逐 env 力推账本挂在 env 上的属性名(schema v1:见 _force_push_state)。
FORCE_PUSH_STATE_ATTR = "_hope_force_push_state_v1"
# Diagnostic-only velocity-push receipt.  The tensors stay on the articulation
# device throughout an update; the runner consumes them once at the PPO
# boundary.  Keeping this ledger on ``env`` (rather than an EventTerm) also
# means ordinary per-env resets cannot erase evidence from the current update.
PUSH_VELOCITY_DIAGNOSTIC_STATE_ATTR = "_hope_push_velocity_diagnostic_v1"
PUSH_VELOCITY_DIAGNOSTIC_EVENT = "hope_push_velocity_diagnostic_update"
PUSH_VELOCITY_DIAGNOSTIC_SCHEMA_VERSION = 1
PUSH_VELOCITY_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
# A3 底座 body(robots/agibot_a3.py A3_ROOT_BODY;唯一小写 "_link" 的 body)。
FORCE_PUSH_BODY_NAME_DEFAULT = "pelvis_link"


def _quat_rotate_inverse_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the INVERSE of quaternion ``q`` (w, x, y, z) — world -> body frame.

    人话:把世界系向量转到 body 系。纯 torch 本地实现(与 isaaclab.utils.math 的
    quat_rotate_inverse 逐位一致,shadow_ball.py 同款),模块无 Isaac 也能单测。
    """
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by quaternion ``q`` (w, x, y, z) — body -> world frame(测试用逆变换)。"""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _resolve_single_body_id(asset, body_name: str) -> int:
    """Resolve ``body_name`` to exactly ONE body index — fail-closed on missing/duplicate.

    人话:名字解析不到、或匹配出不止一个 body(正则重名),都直接 raise;绝不悄悄挑第一个。
    """
    try:
        ids, names = asset.find_bodies(body_name, preserve_order=True)
    except Exception as exc:
        raise RuntimeError(
            f"force push body {body_name!r} not found on the articulation (fail-closed)"
        ) from exc
    if len(ids) != 1:
        raise RuntimeError(
            f"force push body {body_name!r} must resolve to exactly one body, "
            f"got {list(names)!r} (fail-closed on missing/duplicate)"
        )
    return int(ids[0])


def _step_counter(env) -> int:
    """Read the integer control-step clock the expiry ledger is keyed to (fail-closed)."""
    step = getattr(env, "common_step_counter", None)
    if isinstance(step, bool) or not isinstance(step, int):
        raise RuntimeError(
            "force push expiry ledger requires integer env.common_step_counter, "
            f"got {step!r}"
        )
    return int(step)


def _force_push_state(env, asset, body_id: int, body_name: str) -> dict:
    """Get or lazily create the per-env force ledger hanging on ``env`` (schema v1).

    人话:账本记三样——每个 env 当前施加的 body 系力(force_b)、零力矩(torque_b)、到期步
    (expiry_step,-1 = 没在推)。整个缓冲一次性整体写给 Isaac,避免"清一部分 env 把
    has_external_wrench 全局翻掉、别的 env 还没到期的力被吞"的坑。账本已存在但 body 不同
    = 有别的写者在抢同一个账本,fail-closed。
    """
    state = getattr(env, FORCE_PUSH_STATE_ATTR, None)
    if state is not None:
        if state.get("schema_version") != 1 or state.get("body_id") != body_id:
            raise RuntimeError(
                "force push state exists with a different schema/body — refusing a "
                f"competing writer (state body_id={state.get('body_id')!r}, "
                f"requested {body_id!r} {body_name!r})"
            )
        return state
    num_envs = int(env.num_envs)
    device = asset.device
    state = {
        "schema_version": 1,
        "body_id": body_id,
        "body_name": str(body_name),
        "force_b": torch.zeros((num_envs, 1, 3), device=device),
        "torque_b": torch.zeros((num_envs, 1, 3), device=device),
        "expiry_step": torch.full((num_envs,), -1, dtype=torch.long, device=device),
        "applied_any": False,
    }
    setattr(env, FORCE_PUSH_STATE_ATTR, state)
    return state


def _clear_expired_rows(state: dict, step: int) -> None:
    """Zero every env whose push has expired (expiry_step <= current step)."""
    expiry = state["expiry_step"]
    expired = (expiry >= 0) & (expiry <= step)
    if bool(expired.any()):
        state["force_b"][expired] = 0.0
        expiry[expired] = -1


def _write_full_buffer(state: dict, asset) -> None:
    """One whole-tensor external-wrench write for ALL envs on the ledger's body.

    人话:永远整体写(env_ids 缺省 = 全部 env)。Isaac 的 set_external_force_and_torque 用
    "força 全零与否"翻转全局 has_external_wrench 开关,子集清零会把别的 env 还活着的力一并
    停掉;整体写让"还有 env 在推 -> 继续施力 / 全部到期 -> 干净归零"两种情形都正确。
    """
    asset.set_external_force_and_torque(
        state["force_b"], state["torque_b"], body_ids=[state["body_id"]]
    )
    if not bool((state["expiry_step"] >= 0).any()):
        state["applied_any"] = False


def push_by_applying_wrench(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    force_n: float,
    duration_steps: int,
    body_name: str = FORCE_PUSH_BODY_NAME_DEFAULT,
):
    """Interval event: horizontal constant-force push on ``body_name`` for ``duration_steps`` steps.

    人话:对被抽中的 env,朝水平随机方向(世界系 z 分量 0)推一把恒力,力矩为零;登记逐 env
    到期步(当前步 + duration_steps),到期由 sweep_expired_force_pushes(每控制步)或下一次
    本函数被调时清零。触发瞬间用当前 pelvis 姿态把世界系力旋进 body 系(Isaac Lab 2.1 外力
    是 body 系语义),持续期内 body 系力不变。参数非法 / body 缺失重名一律 fail-closed raise。
    """
    if (
        isinstance(force_n, bool)
        or not isinstance(force_n, (int, float))
        or not math.isfinite(float(force_n))
        or float(force_n) <= 0.0
    ):
        raise ValueError(
            f"force push force_n must be a finite number > 0 N, got {force_n!r}"
        )
    if isinstance(duration_steps, bool) or not isinstance(duration_steps, int) or duration_steps < 1:
        raise ValueError(
            f"force push duration_steps must be an integer >= 1, got {duration_steps!r}"
        )
    if not isinstance(body_name, str) or not body_name:
        raise ValueError(
            f"force push body_name must be a non-empty string, got {body_name!r}"
        )

    asset = env.scene["robot"]
    body_id = _resolve_single_body_id(asset, body_name)
    state = _force_push_state(env, asset, body_id, body_name)
    step = _step_counter(env)

    # 先清全体过期力(本函数就是推荐的顺手兜底;高频清扫事件另有一份)。
    _clear_expired_rows(state, step)

    if env_ids is None:
        env_ids = torch.arange(int(env.num_envs), device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device)

    if env_ids.numel() > 0:
        # 世界系水平随机方向恒力:z 分量恒 0,模长 = force_n。
        theta = torch.rand(env_ids.numel(), device=asset.device) * (2.0 * math.pi)
        force_w = torch.stack(
            [
                float(force_n) * torch.cos(theta),
                float(force_n) * torch.sin(theta),
                torch.zeros_like(theta),
            ],
            dim=-1,
        )
        # Isaac Lab 2.1 body-frame semantics: rotate the world-horizontal force into the
        # CURRENT body frame at trigger time (world direction drifts with the body afterwards).
        quat_w = asset.data.body_quat_w[env_ids, body_id]
        if quat_w.shape != (env_ids.numel(), 4):
            raise RuntimeError(
                "force push requires per-env wxyz body quaternions, got shape "
                f"{tuple(quat_w.shape)!r}"
            )
        state["force_b"][env_ids, 0, :] = _quat_rotate_inverse_wxyz(quat_w, force_w)
        state["expiry_step"][env_ids] = step + int(duration_steps)
        state["applied_any"] = True

    _write_full_buffer(state, asset)


def _sample_force_branch_mask(num: int, force_prob: float, device) -> torch.Tensor:
    """合并互斥推的抽签器:True = 力推分支,False = 速度踢分支(Bernoulli(force_prob))。

    人话:每个被触发的 env 独立掷一次硬币,决定这次挨哪种推。单独拆成函数是给单测打桩用
    (mock 固定 mask 验证"同一次触发绝不两种都来")。
    """
    return torch.rand(num, device=device) < force_prob


def _velocity_push_delegate():
    """速度踢分支的唯一实现:isaaclab 的 ``push_by_setting_velocity``(惰性 import)。

    人话:合并模式的速度分支不重写第二份采样代码,直接调 legacy 独立事件用的同一个 isaaclab
    函数——分支的物理写者、RNG 和结果语义与 legacy 同源,永不漂移。惰性 import 让本模块在无 Isaac 的机器上
    仍可按文件路径单测(单测在此打桩)。
    """
    from isaaclab.envs.mdp import push_by_setting_velocity

    return push_by_setting_velocity


def _push_velocity_diagnostic_enabled(env) -> bool:
    """Return true only for the explicitly non-promotable ActionBall screen.

    This predicate is intentionally permissive on every other environment:
    missing/intermediate config objects and non-boolean legacy values all take
    the physical-writer/RNG/result-preserving delegate-only path.  The formal
    runner independently fail-closes malformed ActionBall diagnostics before
    learning starts.
    """

    cfg = getattr(env, "cfg", None)
    commands = None if cfg is None else getattr(cfg, "commands", None)
    racket = None if commands is None else getattr(commands, "racket_target", None)
    return (
        getattr(racket, "target_mode", None) == "action_ball"
        and getattr(racket, "action_ball_diagnostic_unauthorized", False) is True
    )


def _push_velocity_diagnostic_state(
    env, *, device, dtype, lower_values: tuple, upper_values: tuple
) -> dict:
    """Get/create the device-resident, reset-proof per-update receipt ledger."""

    state = getattr(env, PUSH_VELOCITY_DIAGNOSTIC_STATE_ATTR, None)
    if state is not None:
        if (
            state.get("schema_version") != PUSH_VELOCITY_DIAGNOSTIC_SCHEMA_VERSION
            or tuple(state.get("axes", ())) != PUSH_VELOCITY_AXES
            or state.get("bounds") != (lower_values, upper_values)
        ):
            raise RuntimeError(
                "velocity-push diagnostic state has an incompatible schema"
            )
        return state
    state = {
        "schema_version": PUSH_VELOCITY_DIAGNOSTIC_SCHEMA_VERSION,
        "axes": PUSH_VELOCITY_AXES,
        "bounds": (lower_values, upper_values),
        "lower": torch.tensor(lower_values, dtype=dtype, device=device),
        "upper": torch.tensor(upper_values, dtype=dtype, device=device),
        "event_call_count": torch.zeros((), dtype=torch.int64, device=device),
        "env_application_count": torch.zeros((), dtype=torch.int64, device=device),
        "delta_nonfinite_element_count": torch.zeros(
            (), dtype=torch.int64, device=device
        ),
        "observed_delta_min": torch.full(
            (6,), math.inf, dtype=dtype, device=device
        ),
        "observed_delta_max": torch.full(
            (6,), -math.inf, dtype=dtype, device=device
        ),
        "below_range_count": torch.zeros((6,), dtype=torch.int64, device=device),
        "above_range_count": torch.zeros((6,), dtype=torch.int64, device=device),
    }
    setattr(env, PUSH_VELOCITY_DIAGNOSTIC_STATE_ATTR, state)
    return state


def push_by_setting_velocity(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    velocity_range: dict,
    asset_cfg=None,
):
    """Delegate one IsaacLab velocity push and optionally book diagnostic evidence.

    Production/default behavior is a strict pass-through: no scene access, no
    tensor allocation, and exactly one call to IsaacLab's implementation.  The
    explicitly unauthorized ActionBall diagnostic additionally snapshots the
    selected rows of ``root_vel_w`` before and after that one call.  It never
    writes either snapshot back, so the physical velocity is still authored
    solely by IsaacLab.

    The observed delta is also the sampled kick under IsaacLab's additive
    ``push_by_setting_velocity`` contract.  Six-axis min/max and range breaches
    are accumulated on-device and synchronized only by
    :func:`consume_push_velocity_diagnostic_counters` at a PPO boundary.
    """

    delegate = _velocity_push_delegate()
    if not _push_velocity_diagnostic_enabled(env):
        if asset_cfg is None:
            return delegate(env, env_ids, velocity_range=velocity_range)
        return delegate(
            env, env_ids, velocity_range=velocity_range, asset_cfg=asset_cfg
        )

    asset_name = "robot" if asset_cfg is None else getattr(asset_cfg, "name", None)
    if not isinstance(asset_name, str) or not asset_name:
        raise RuntimeError(
            "velocity-push diagnostic requires asset_cfg.name or the default robot"
        )
    asset = env.scene[asset_name]
    root_vel_w = asset.data.root_vel_w
    if not torch.is_tensor(root_vel_w) or root_vel_w.ndim != 2 or root_vel_w.shape[1] != 6:
        raise RuntimeError(
            "velocity-push diagnostic requires robot.data.root_vel_w shaped [N, 6]"
        )
    if env_ids is None:
        observed_ids = torch.arange(int(env.num_envs), device=root_vel_w.device)
    else:
        observed_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=root_vel_w.device
        )
    before = root_vel_w[observed_ids].detach().clone()

    try:
        lower_values = tuple(
            float(velocity_range[axis][0]) for axis in PUSH_VELOCITY_AXES
        )
        upper_values = tuple(
            float(velocity_range[axis][1]) for axis in PUSH_VELOCITY_AXES
        )
        if not all(math.isfinite(value) for value in lower_values + upper_values):
            raise ValueError("non-finite velocity-push bound")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "velocity-push diagnostic requires finite bounds for all six axes"
        ) from exc
    state = _push_velocity_diagnostic_state(
        env,
        device=root_vel_w.device,
        dtype=root_vel_w.dtype,
        lower_values=lower_values,
        upper_values=upper_values,
    )
    lower = state["lower"]
    upper = state["upper"]

    # The one and only physical writer.  Never emulate or second-guess the
    # upstream event, because doing so would make the receipt change behavior.
    if asset_cfg is None:
        result = delegate(env, env_ids, velocity_range=velocity_range)
    else:
        result = delegate(
            env, env_ids, velocity_range=velocity_range, asset_cfg=asset_cfg
        )

    after = root_vel_w[observed_ids].detach().clone()
    delta = after - before
    with torch.no_grad():
        state["event_call_count"].add_(1)
        state["env_application_count"].add_(int(observed_ids.numel()))
        finite = torch.isfinite(delta)
        state["delta_nonfinite_element_count"].add_((~finite).sum())
        if delta.shape[0] > 0:
            finite_min = torch.where(
                finite, delta, torch.full_like(delta, math.inf)
            ).amin(dim=0)
            finite_max = torch.where(
                finite, delta, torch.full_like(delta, -math.inf)
            ).amax(dim=0)
            state["observed_delta_min"].copy_(
                torch.minimum(state["observed_delta_min"], finite_min)
            )
            state["observed_delta_max"].copy_(
                torch.maximum(state["observed_delta_max"], finite_max)
            )
            # A tiny numerical tolerance prevents a representable endpoint
            # from becoming a false breach after one subtraction.
            tolerance = torch.maximum(
                torch.full_like(lower, 1.0e-6),
                1.0e-6 * torch.maximum(lower.abs(), upper.abs()),
            )
            state["below_range_count"].add_(
                (finite & (delta < (lower - tolerance))).sum(dim=0)
            )
            state["above_range_count"].add_(
                (finite & (delta > (upper + tolerance))).sum(dim=0)
            )
    return result


def consume_push_velocity_diagnostic_counters(env) -> dict:
    """Synchronize/clear one complete velocity-push update window.

    All device values are packed into one tensor and transferred once.  Empty
    windows use ``None`` for min/max rather than fake zeros; counters are exact
    integers.  Clearing happens only here, never during an environment reset.
    """

    state = getattr(env, PUSH_VELOCITY_DIAGNOSTIC_STATE_ATTR, None)
    if state is None:
        return {
            "event_call_count": 0,
            "env_application_count": 0,
            "delta_nonfinite_element_count": 0,
            "axes": {
                axis: {
                    "observed_delta_min": None,
                    "observed_delta_max": None,
                    "below_range_count": 0,
                    "above_range_count": 0,
                }
                for axis in PUSH_VELOCITY_AXES
            },
        }
    if (
        state.get("schema_version") != PUSH_VELOCITY_DIAGNOSTIC_SCHEMA_VERSION
        or tuple(state.get("axes", ())) != PUSH_VELOCITY_AXES
    ):
        raise RuntimeError("velocity-push diagnostic state has an incompatible schema")
    packed = torch.cat(
        (
            torch.stack(
                (
                    state["event_call_count"],
                    state["env_application_count"],
                    state["delta_nonfinite_element_count"],
                )
            ).to(dtype=torch.float64),
            state["observed_delta_min"].to(dtype=torch.float64),
            state["observed_delta_max"].to(dtype=torch.float64),
            state["below_range_count"].to(dtype=torch.float64),
            state["above_range_count"].to(dtype=torch.float64),
        )
    ).tolist()
    event_calls, env_applications, nonfinite = (int(value) for value in packed[:3])
    mins = packed[3:9]
    maxs = packed[9:15]
    below = [int(value) for value in packed[15:21]]
    above = [int(value) for value in packed[21:27]]
    axes = {}
    for index, axis in enumerate(PUSH_VELOCITY_AXES):
        axes[axis] = {
            "observed_delta_min": mins[index] if math.isfinite(mins[index]) else None,
            "observed_delta_max": maxs[index] if math.isfinite(maxs[index]) else None,
            "below_range_count": below[index],
            "above_range_count": above[index],
        }
    with torch.no_grad():
        state["event_call_count"].zero_()
        state["env_application_count"].zero_()
        state["delta_nonfinite_element_count"].zero_()
        state["observed_delta_min"].fill_(math.inf)
        state["observed_delta_max"].fill_(-math.inf)
        state["below_range_count"].zero_()
        state["above_range_count"].zero_()
    return {
        "event_call_count": event_calls,
        "env_application_count": env_applications,
        "delta_nonfinite_element_count": nonfinite,
        "axes": axes,
    }


def push_combined_exclusive(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    velocity_range: dict,
    force_n: float,
    duration_steps: int,
    force_prob: float,
    body_name: str = FORCE_PUSH_BODY_NAME_DEFAULT,
):
    """Merged EXCLUSIVE push: per fire each chosen env gets EITHER a velocity kick OR a force push.

    人话(Franco 2026-07-25 两种随机推合并抽签):legacy 配方里速度推(push_robot)与力推
    (force_push)是两个独立时钟的 interval 事件,可能同一瞬间一起砸下来叠加;本事件把两种推
    合并成【一个】interval 事件——每次触发,被抽中的每个 env 按 ``force_prob`` 掷硬币二选一:
    力分支 = 调 :func:`push_by_applying_wrench`(水平随机方向恒力 ``force_n`` 牛,持续
    ``duration_steps`` 步,登记同一本到期账本,仍需 ``sweep_expired_force_pushes`` 每控制步
    兜底清扫);速度分支 = 调 isaaclab 的 ``push_by_setting_velocity``(±velocity_range 随机
    改底座速度)。两个分支的 env 集合是同一张 mask 的正反两半,结构上互斥——同一次触发同一个
    env 绝不可能两种都挨。``force_prob`` 必须严格落在 (0, 1)(0/1 等价单类型推,应直接用
    legacy 单事件);参数非法一律 fail-closed raise,不静默。
    """
    if (
        isinstance(force_prob, bool)
        or not isinstance(force_prob, (int, float))
        or not math.isfinite(float(force_prob))
        or not (0.0 < float(force_prob) < 1.0)
    ):
        raise ValueError(
            "combined push force_prob must be a finite number strictly inside (0, 1), "
            f"got {force_prob!r}"
        )
    if not hasattr(velocity_range, "items") or not {"x", "y"} <= set(velocity_range):
        raise ValueError(
            "combined push velocity_range must be a mapping with at least the x and y "
            f"axes, got {velocity_range!r}"
        )

    asset = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(int(env.num_envs), device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device)
    if env_ids.numel() == 0:
        return

    mask = _sample_force_branch_mask(int(env_ids.numel()), float(force_prob), asset.device)
    if (
        not torch.is_tensor(mask)
        or mask.dtype != torch.bool
        or tuple(mask.shape) != (int(env_ids.numel()),)
    ):
        raise RuntimeError(
            "combined push sampler must return a bool mask aligned with env_ids, got "
            f"{mask!r}"
        )
    force_ids = env_ids[mask]
    velocity_ids = env_ids[~mask]
    # 互斥保证:force_ids / velocity_ids 是同一张 mask 的正反两半,交集恒空、并集恒全。
    if force_ids.numel() > 0:
        push_by_applying_wrench(
            env, force_ids,
            force_n=force_n, duration_steps=duration_steps, body_name=body_name,
        )
    if velocity_ids.numel() > 0:
        _velocity_push_delegate()(env, velocity_ids, velocity_range=velocity_range)


def sweep_expired_force_pushes(env: "ManagerBasedEnv", env_ids: torch.Tensor | None = None):
    """Per-control-step sweeper: zero every expired force push (the expiry guarantee).

    人话:挂成 interval=(control_dt, control_dt) 的事件 = 每个控制步跑一遍,凡到期(当前步 >=
    到期步)的 env 力清零并整体重写外力缓冲。``env_ids`` 故意忽略——到期与"这步谁被抽中"无关,
    清扫永远扫全体。没有账本(默认全关 / 从未触发)或账本已经全零时,是严格 no-op:不碰
    asset、不写缓冲,不跟其他外力写者(lateral 扰动适配器、球的气动力)抢。
    """
    del env_ids  # 到期语义是全体的,与触发抽样无关(见 docstring)。
    state = getattr(env, FORCE_PUSH_STATE_ATTR, None)
    if state is None or not state.get("applied_any", False):
        return
    asset = env.scene["robot"]
    _clear_expired_rows(state, _step_counter(env))
    _write_full_buffer(state, asset)
