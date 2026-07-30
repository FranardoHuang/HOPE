"""spdmix-v2 硬绑定四(per-clip 框表/缓存越界)unit tests (CPU, isaaclab STUBBED).

人话:六 clip 变速臂 boot 死于 env.reset 的 CUDA device-side assert,traceback 指向
`_strike_frame_for_clip → motion.seg_start[clip_id]`——但那只是尸体倒下的地方(第一处
host 同步点 .item()),不是案发现场。真凶是 `_sample_targets_uniform` 更早的 GPU gather:
racket_pos/vel_range_per_clip 框表只有正/反手两行(DeployParity YAML 的家族框,train.py
的 _resolve_pos_range_per_clip 也只认 forehand/backhand 两个名字),六 clip 下
`_pos_range_per_clip_t[clip]` 拿 clip_id 0..5 直接越界,CUDA 异步 assert 直到下一个同步点
才爆。修法:所有"每 clip 一行"的表在 gather 前经 `_per_clip_range_rows` 按行数核对——
行数==段数直接用(legacy 2-clip 原张量原样返回,逐字节不变);两行家族框按
clip_family_per_clip 族表展开成每 clip 一行;其他行数当场 ValueError 报人话。配套把
HER 缓存/per-族指标的 `clips == c` 分桶全部换成 `_clip_family_rows()` 族行号(legacy
2-clip 族行号==clip_id,逐字节不变;6-clip 下正手 1.0/1.2 档不再被记进 backhand 桶)。

Reuses the isaaclab stub + real-module loader from test_reward_flags_mdp — everything exercised
is the REAL shipped commands.py / hope_commands.py.

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_spdmix_per_clip_binding.py -q
"""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_reward_flags_mdp import (  # noqa: E402  (installs the isaaclab stub, loads REAL modules)
    _make_motion_command,
    _write_motion_npz,
    hope_commands_mod,
)
from test_clip_family_per_clip import FAM6, _make_rt  # noqa: E402

RT = hope_commands_mod.RacketTargetCommand

# 六段合成 clip 帧数(段起点 cumsum: 0,12,26,42,60,80;总长 102)。
SIX_FRAMES = (12, 14, 16, 18, 20, 22)
PHASES6 = (0.471, 0.471, 0.471, 0.338, 0.338, 0.338)
SIGNS6 = (1.0, 1.0, 1.0, -1.0, -1.0, -1.0)

# 家族框表:每族一行((2, 3, 2) = [族][x/y/z][lo/hi]),y 带符号、两族不重叠。
POS2 = torch.tensor(
    [
        [[0.50, 0.60], [-0.60, -0.20], [0.70, 0.90]],  # forehand 族
        [[0.50, 0.60], [0.20, 0.60], [0.90, 1.10]],    # backhand 族
    ]
)
VEL2 = torch.tensor(
    [
        [[1.00, 2.00], [0.90, 1.90], [0.30, 1.10]],    # forehand 族
        [[1.60, 2.60], [-1.20, -0.20], [0.00, 0.70]],  # backhand 族
    ]
)


@pytest.fixture(scope="module")
def six_clip_motion():
    """真 MotionCommand + 真 6 段 MotionLoader(motion_file 3 正手 + motion_file_2 3 反手的
    合并列表语义:train.py 只是把两个列表按序拼成一个 files 列表)。"""
    tmp = tempfile.mkdtemp(prefix="spdmix_six_clips_")
    files = [
        _write_motion_npz(os.path.join(tmp, f"clip{i}.npz"), frames=SIX_FRAMES[i])
        for i in range(6)
    ]
    mcmd, _ = _make_motion_command(files, clip_family_per_clip=FAM6)
    return mcmd


def _make_strike_rt(mcmd):
    """_ensure_reference_strike_state / _strike_frame_for_clip 需要的最小真实状态。"""
    rt = RT.__new__(RT)
    rt.device = "cpu"
    rt.cfg = types.SimpleNamespace(
        strike_phase=0.5,
        strike_phase_per_clip=PHASES6,
        mount_normal_sign_per_clip=SIGNS6,
        mount_normal_sign=1.0,
        mount_normal_axis=1,
        clean_strike_vel_window=2,
        clean_reference_strike_velocity=True,
        # 参考击球点的"桌面以下不合法"闸门要读虚拟球桌常数(合成 clip 的 x=0,在近桌边之后,闸门空转)。
        vb_table_near_x=0.5,
        vb_table_surface_z=0.76,
    )
    rt._vb_ball_r = 0.02
    rt._motion = lambda: mcmd
    rt._env = types.SimpleNamespace(step_dt=0.02)
    rt._racket_mode = "body"
    rt._racket_body_index = 0
    rt._clip_names = {0: "forehand", 1: "backhand"}
    rt._family_is_forehand_t = None
    rt._clip_family_rows_t = None
    rt._ref_strike_cached = False
    rt._ref_normal_per_clip = None
    return rt


# --------------------------------------------------------------------------------------------- #
# 六段 loader:seg_start/段结构 + _strike_frame_for_clip / _ensure_ref_normal_per_clip 走通
# --------------------------------------------------------------------------------------------- #
def test_six_clip_loader_segments_and_strike_frames_stay_in_segment(six_clip_motion):
    """六段 seg_start 是逐段 cumsum;每个 clip 的击球帧都落在自己段内、tick 按该 clip 的
    strike_phase 算——不再有任何隐含"只有 2 段"的假设。"""
    ml = six_clip_motion.motion
    assert int(ml.num_segments) == 6
    assert ml.seg_len.tolist() == list(SIX_FRAMES)
    assert ml.seg_start.tolist() == [0, 12, 26, 42, 60, 80]
    rt = _make_strike_rt(six_clip_motion)
    for clip_id in range(6):
        strike, phase, seg_start, seg_len = rt._strike_frame_for_clip(ml, clip_id)
        assert phase == PHASES6[clip_id]
        assert seg_start == int(ml.seg_start[clip_id])
        assert seg_len == SIX_FRAMES[clip_id]
        assert seg_start <= strike <= seg_start + seg_len - 1
        assert strike - seg_start == round(phase * (seg_len - 1))


def test_strike_frame_oob_clip_fails_with_human_readable_error(six_clip_motion):
    """越界 clip_id 报人话 IndexError(段数写明),不是 CUDA assert。"""
    rt = _make_strike_rt(six_clip_motion)
    for bad in (6, 7, -1):
        with pytest.raises(IndexError, match="out of range for 6 segments"):
            rt._strike_frame_for_clip(six_clip_motion.motion, bad)


def test_reference_strike_state_walks_all_six_segments(six_clip_motion):
    """_ensure_ref_normal_per_clip 全链(含 _ensure_reference_strike_state 六段循环)走通:
    每 clip 一行缓存,行数 == 段数,击球面符号按六位表逐 clip 取。"""
    rt = _make_strike_rt(six_clip_motion)
    rt._ensure_ref_normal_per_clip()
    assert rt._ref_normal_per_clip.shape == (6, 3)
    assert rt._ref_racket_pos_rel_per_clip.shape == (6, 3)
    assert rt._ref_reach_offset_xy_per_clip.shape == (6, 2)
    # 合成 clip 拍面朝 +Y(单位阵姿态):正手三档 +Y、反手三档(符号 -1)-Y。
    normals = rt._ref_normal_per_clip
    assert torch.all(normals[:3, 1] > 0.9)
    assert torch.all(normals[3:, 1] < -0.9)


def test_stale_ref_normal_cache_rows_fail_loud(six_clip_motion):
    """缓存行数 != 段数(两行旧缓存撞上六段 motion)当场 RuntimeError 报人话,不进 gather。"""
    rt = _make_strike_rt(six_clip_motion)
    rt._ref_normal_per_clip = torch.randn(2, 3)
    with pytest.raises(RuntimeError, match="2 row.*6 clip"):
        rt._ensure_ref_normal_per_clip()


# --------------------------------------------------------------------------------------------- #
# _per_clip_range_rows:legacy 原张量不动、家族框展开、行数错/缺族表当场报人话
# --------------------------------------------------------------------------------------------- #
def test_range_rows_legacy_two_clip_returns_the_same_tensor_object():
    """legacy 2-clip:行数==段数,原张量原样返回(同一对象 = 逐字节不变的最强证据)。"""
    rt = _make_rt(4, [0, 1, 0, 1], num_segments=2)
    out = rt._per_clip_range_rows(POS2, "racket_pos_range_per_clip")
    assert out is POS2
    # 缓存命中也还是同一对象
    assert rt._per_clip_range_rows(POS2, "racket_pos_range_per_clip") is POS2


def test_range_rows_expand_family_table_to_six_clips():
    """六 clip + 两行家族框:按族表展开成每 clip 一行(正手三档共用 0 行、反手三档共用 1 行)。"""
    rt = _make_rt(6, list(range(6)), num_segments=6, families=FAM6)
    out = rt._per_clip_range_rows(POS2, "racket_pos_range_per_clip")
    assert out.shape == (6, 3, 2)
    assert torch.equal(out, POS2[torch.tensor([0, 0, 0, 1, 1, 1])])


def test_range_rows_wrong_rowcount_fails_loud():
    """行数既不等于段数也不是 2:当场 ValueError,报 cfg 键名 + 行数 + 段数。"""
    rt = _make_rt(6, list(range(6)), num_segments=6, families=FAM6)
    bad = torch.zeros(3, 3, 2)
    with pytest.raises(ValueError, match="racket_vel_range_per_clip has 3 row.*6"):
        rt._per_clip_range_rows(bad, "racket_vel_range_per_clip")


def test_range_rows_six_clips_without_family_table_fail_loud():
    """两行家族框 + 六 clip + 没配族表:展开走 _clip_family_is_forehand,当场报
    clip_family_per_clip 人话错误——绝不猜、绝不越界。"""
    rt = _make_rt(6, list(range(6)), num_segments=6, families=None)
    with pytest.raises(ValueError, match="clip_family_per_clip"):
        rt._per_clip_range_rows(POS2, "racket_pos_range_per_clip")


# --------------------------------------------------------------------------------------------- #
# 真 _sample_targets_uniform:六 clip 家族框(修前 = 越界)+ legacy 逐字节不变
# --------------------------------------------------------------------------------------------- #
def _make_box_rt(num_envs, clip_ids, num_segments, families=None, mix_prob=0.0):
    """带 per-clip 框表的 uniform 采样 fake(HER 默认关;开 HER 的测试再补缓存状态)。"""
    rt = _make_rt(num_envs, clip_ids, num_segments, families=families)
    # The real initializer sets both runtime-mode flags unconditionally.  This
    # hand-built fixture exercises the legacy non-TaskFirst/non-ActionBall path.
    rt._task_first_enabled = False
    rt._action_ball_enabled = False
    rt.cfg = types.SimpleNamespace(
        racket_pos_x_range=(0.5, 0.7),
        racket_pos_y_abs_range=(0.1, 0.4),
        racket_pos_z_range=(0.9, 1.2),
        racket_vel_x_range=(1.0, 2.0),
        racket_vel_y_range=(-0.5, 0.5),
        racket_vel_z_range=(0.0, 1.0),
        forehand_on_negative_y=True,
        achieved_target_mix_prob=mix_prob,
        achieved_min_fill=1,
        achieved_jitter_pos=0.0,
        achieved_jitter_vel=0.0,
        achieved_clamp_inflate=0.0,
    )
    rt._pos_range_per_clip_t = POS2.clone()
    rt._vel_range_per_clip_t = VEL2.clone()
    rt._question_bank = None
    rt._ref_normal_per_clip = torch.nn.functional.normalize(
        torch.randn(num_segments, 3), dim=-1
    )
    rt.racket_target_pos_w = torch.zeros(num_envs, 3)
    rt.racket_target_vel_w = torch.zeros(num_envs, 3)
    rt.racket_target_normal_w = torch.zeros(num_envs, 3)
    rt._clip_names = {0: "forehand", 1: "backhand"}
    rt._resample_n_acc = 0.0
    rt._replay_n_acc = 0.0
    rt._env = types.SimpleNamespace(
        scene=types.SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
    )
    return rt


def test_uniform_sampler_six_clip_family_boxes_walk_and_land_in_family_box():
    """崩溃现场回归:六 clip + 两行家族框走真 _sample_targets_uniform——修前这里就是
    `_pos_range_per_clip_t[clip]` 的越界(CPU 上 IndexError,GPU 上异步 CUDA assert);
    修后正手三档采进正手框、反手三档采进反手框。"""
    n = 12
    clips = torch.tensor(list(range(6)) * 2)
    rt = _make_box_rt(n, clips, num_segments=6, families=FAM6)
    torch.manual_seed(3)
    rt._sample_targets_uniform(torch.arange(n), torch.zeros(n, 3), n)
    fh = torch.tensor([True, True, True, False, False, False] * 2)
    pos, vel = rt.racket_target_pos_w, rt.racket_target_vel_w
    for axis in range(3):
        assert torch.all(pos[fh, axis] >= POS2[0, axis, 0]) and torch.all(
            pos[fh, axis] <= POS2[0, axis, 1]
        )
        assert torch.all(pos[~fh, axis] >= POS2[1, axis, 0]) and torch.all(
            pos[~fh, axis] <= POS2[1, axis, 1]
        )
        assert torch.all(vel[fh, axis] >= VEL2[0, axis, 0]) and torch.all(
            vel[fh, axis] <= VEL2[0, axis, 1]
        )
        assert torch.all(vel[~fh, axis] >= VEL2[1, axis, 0]) and torch.all(
            vel[~fh, axis] <= VEL2[1, axis, 1]
        )
    assert torch.equal(rt.racket_target_normal_w, rt._ref_normal_per_clip[clips])


def test_uniform_sampler_legacy_per_clip_boxes_bitwise_identical():
    """legacy 2-clip + per-clip 框:相同种子下修后输出与旧公式逐行复刻逐字节 torch.equal
    (含 RNG 消耗次序;_per_clip_range_rows 直通原张量,零随机数消耗)。"""
    n = 16
    clips = torch.tensor([0, 1] * 8)
    origins = torch.randn(n, 3)
    rt = _make_box_rt(n, clips, num_segments=2)
    torch.manual_seed(20260722)
    rt._sample_targets_uniform(torch.arange(n), origins.clone(), n)

    torch.manual_seed(20260722)
    pos = origins.clone()
    rng_e = POS2[clips]
    lo, hi = rng_e[..., 0], rng_e[..., 1]
    pos[:, :3] += lo + (hi - lo) * torch.rand(n, 3)
    rng_v = VEL2[clips]
    lo_v, hi_v = rng_v[..., 0], rng_v[..., 1]
    vel = lo_v + (hi_v - lo_v) * torch.rand(n, 3)

    assert torch.equal(rt.racket_target_pos_w, pos)
    assert torch.equal(rt.racket_target_vel_w, vel)


# --------------------------------------------------------------------------------------------- #
# HER 回放:按族分桶 + 逐 env 按 clip 框夹回(修前 clip 2..5 永远选不中桶)
# --------------------------------------------------------------------------------------------- #
def test_her_replay_buckets_by_family_and_clamps_into_family_box():
    n = 12
    clips = torch.tensor(list(range(6)) * 2)
    rt = _make_box_rt(n, clips, num_segments=6, families=FAM6, mix_prob=1.0)
    far = 100.0  # 缓存值远在框外,回放后必须被夹回各族框
    rt._ach_pos = {0: torch.full((4, 3), far), 1: torch.full((4, 3), -far)}
    rt._ach_vel = {0: torch.full((4, 3), far), 1: torch.full((4, 3), -far)}
    rt._ach_spd = {0: torch.ones(4), 1: torch.ones(4)}
    rt._ach_fill = {0: 4, 1: 4}
    rt._ach_ptr = {0: 0, 1: 0}
    torch.manual_seed(11)
    rt._sample_targets_uniform(torch.arange(n), torch.zeros(n, 3), n)
    # mix_prob=1.0:全部 12 个 env(含 clip 2..5)都算回放——修前只有 clip 0/1 能命中桶。
    assert rt._replay_n_acc == float(n)
    fh = torch.tensor([True, True, True, False, False, False] * 2)
    pos = rt.racket_target_pos_w
    # 夹回框内(inflate=0):正手 env 夹到正手框上界,反手 env 夹到反手框下界。
    assert torch.allclose(pos[fh], POS2[0, :, 1].expand_as(pos[fh]))
    assert torch.allclose(pos[~fh], POS2[1, :, 0].expand_as(pos[~fh]))


def test_her_replay_legacy_two_clip_buckets_unchanged():
    """legacy 2-clip:族行号 == clip_id,回放分桶/夹框与旧行为同值(每桶只吃自己 clip 的 env)。"""
    n = 8
    clips = torch.tensor([0, 1] * 4)
    rt = _make_box_rt(n, clips, num_segments=2, mix_prob=1.0)
    rt._ach_pos = {0: torch.full((4, 3), 100.0), 1: torch.full((4, 3), -100.0)}
    rt._ach_vel = {0: torch.zeros(4, 3), 1: torch.zeros(4, 3)}
    rt._ach_spd = {0: torch.ones(4), 1: torch.ones(4)}
    rt._ach_fill = {0: 4, 1: 4}
    rt._ach_ptr = {0: 0, 1: 0}
    torch.manual_seed(5)
    rt._sample_targets_uniform(torch.arange(n), torch.zeros(n, 3), n)
    assert rt._replay_n_acc == float(n)
    is0 = clips == 0
    assert torch.allclose(rt.racket_target_pos_w[is0], POS2[0, :, 1].expand(int(is0.sum()), 3))
    assert torch.allclose(rt.racket_target_pos_w[~is0], POS2[1, :, 0].expand(int((~is0).sum()), 3))


# --------------------------------------------------------------------------------------------- #
# 题库锚点行数守卫 + sparse 账本按族分桶 + 源码守卫
# --------------------------------------------------------------------------------------------- #
def test_qb_base_anchor_rows_guard_fails_loud():
    """题库两族锚点(2 行)撞上六段 motion 且没配族表:当场 ValueError 报人话,
    不让调用端拿 clip_id 0..5 越界 gather。"""
    rt = _make_rt(6, list(range(6)), num_segments=6, families=None)
    rt.cfg = types.SimpleNamespace(
        target_mode="uniform", base_couple_blend=0.0, base_couple_max_offset=0.2
    )
    rt._qb_base_anchor = None
    rt._question_bank = types.SimpleNamespace(contact_pos=torch.zeros(2, 3))
    with pytest.raises(ValueError, match="2 row.*6 clip.*clip_family_per_clip"):
        rt._qb_base_anchor_off_xy()


def test_sparse_ledger_buckets_forehand_variant_by_family_not_clip_id():
    """clip 1(正手 1.0 档)的击球必须记进 *_forehand 账本——修前 `clip_id == 1` 把它记进
    backhand;clip 4(反手 1.0 档)记进 *_backhand。"""
    cmd = RT.__new__(RT)
    cmd.device = "cpu"
    cmd.num_envs = 2
    cmd._clip_names = {0: "forehand", 1: "backhand"}
    cmd._family_is_forehand_t = None
    cmd._clip_family_rows_t = None
    motion = types.SimpleNamespace(
        _multiseg=True,
        clip_id=torch.tensor([1, 4]),
        cfg=types.SimpleNamespace(clip_family_per_clip=FAM6),
        motion=types.SimpleNamespace(num_segments=6),
    )
    cmd._motion = lambda: motion
    on = torch.tensor([True, True])
    cmd._book_sparse_reward_eligibility(
        exact_strike=on, capture=on, net_clear=on, landing_valid=on, legal_return=on
    )
    ledger = cmd._sparse_reward_eligibility_counters
    assert ledger["strike_opportunity_count_forehand"].item() == 1
    assert ledger["strike_opportunity_count_backhand"].item() == 1
    assert ledger["virtual_capture_count_forehand"].item() == 1
    assert ledger["virtual_capture_count_backhand"].item() == 1


def test_source_guard_family_bucketing_everywhere():
    """源码守卫:所有按族分桶/按 clip 建表的点位都走 _clip_family_rows/_per_clip_range_rows,
    裸 `== c` clip 分桶和裸表 gather 已死透。"""
    bucketed = (
        RT._count_swing_starts,
        RT._vb_evaluate,
        RT._update_metrics,
        RT._book_sparse_reward_eligibility,
    )
    for func in bucketed:
        src = inspect.getsource(func)
        # ``_metric_bucket_rows`` is the N-stroke successor: it RETURNS ``_clip_family_rows()``
        # unless racket_target.clip_names_per_clip declares one name per clip, in which case the
        # bucket is the clip id (so fh_loop and fh_block_syn stop sharing a bucket). Either name
        # satisfies this guard's actual intent: no bare ``== c`` clip bucketing.
        assert ("_clip_family_rows" in src or "_metric_bucket_rows" in src), func.__qualname__
        for stale in ("clips == c", "clip_all == c", "(_clip == _c)", "clip_id == clip"):
            assert stale not in src, (func.__qualname__, stale)
    for func in (RT._sample_targets_uniform, RT._sample_targets_hitter_pure):
        src = inspect.getsource(func)
        assert "_per_clip_range_rows" in src, func.__qualname__
        assert "self._pos_range_per_clip_t[clip]" not in src, func.__qualname__
        assert "self._vel_range_per_clip_t[clip]" not in src, func.__qualname__


# --------------------------------------------------------------------------------------------- #
# hitter_pure 采样:六 clip 家族框走通(同一根裸 gather 的另一个调用点)
# --------------------------------------------------------------------------------------------- #
def test_hitter_pure_six_clip_family_boxes_walk():
    n = 6
    clips = torch.arange(6)
    rt = _make_rt(n, clips, num_segments=6, families=FAM6)
    rt.cfg = types.SimpleNamespace(
        base_target_x_range=(0.0, 0.0),
        base_target_y_range=(0.0, 0.0),
        normal_mode="velocity",
    )
    rt._pos_range_per_clip_t = POS2.clone()
    rt._vel_range_per_clip_t = VEL2.clone()
    rt.base_target_pos_w = torch.zeros(n, 2)
    rt.racket_target_pos_w = torch.zeros(n, 3)
    rt.racket_target_vel_w = torch.zeros(n, 3)
    rt.racket_target_normal_w = torch.zeros(n, 3)
    torch.manual_seed(9)
    rt._sample_targets_hitter_pure(torch.arange(n), torch.zeros(n, 3), n)
    fh = torch.tensor([True, True, True, False, False, False])
    y = rt.racket_target_pos_w[:, 1]  # station y = 0 -> y 偏移即族框 y
    assert torch.all(y[fh] >= POS2[0, 1, 0]) and torch.all(y[fh] <= POS2[0, 1, 1])
    assert torch.all(y[~fh] >= POS2[1, 1, 0]) and torch.all(y[~fh] <= POS2[1, 1, 1])
    vy = rt.racket_target_vel_w[:, 1]
    assert torch.all(vy[fh] >= VEL2[0, 1, 0]) and torch.all(vy[fh] <= VEL2[0, 1, 1])
    assert torch.all(vy[~fh] >= VEL2[1, 1, 0]) and torch.all(vy[~fh] <= VEL2[1, 1, 1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
