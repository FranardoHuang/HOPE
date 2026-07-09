"""metric-sync-fix-0709 — rally 记账修复 + stagger 防同步 unit tests (CPU, isaaclab STUBBED).

人话:上台率曲线 virtual_return_rate_rally 以前分子分母各记各的账、衰减节拍还不一样,4096 个 env
同时超时排成同步大队列时比值能冲破 1(实测 0.31→1.48 振荡,2026-07-08 取证定案)。修复后改成
"每拍打完,起拍数和有没有回球同时记进同一本账",比值恒 ≤1 且等于真实回球率;旧算法曲线以 _legacy
后缀并行保留一个过渡期。防同步旗标 task.motion.stagger_initial_clock(默认关)把所有 env 的
超时/挥拍节拍随机错开,治所有 EMA 指标的同步读数病。

Reuses the isaaclab stub + real-module loader from test_reward_flags_mdp — everything exercised
below is the REAL shipped hope_commands.py / commands.py, not a re-derivation. Covers:

* synchronized-queue synthetic scenario (mass same-instant reset, strike +116 steps, wrap every
  150, mass timeout): the OLD ledger (_rally_legacy_values -> the *_legacy curves) reproduces
  readings > 1 at its real write moments; the NEW per-swing same-ledger metric
  (virtual_return_rate_rally, via _count_swing_starts + _decay_swing_accounting + _rally_report)
  stays <= 1 at EVERY step and equals the true per-swing return rate.
* timeout-cut swings (episode ends before the strike frame) count as failed attempts.
* pairing details: an env's first-ever resample books nothing; the returned latch clears at the
  swing end (no cross-swing leakage); per-clip attribution via _prev_clip_id.
* wrap-boundary strike parking (防御修 2026-07-09): a strike firing on the SAME step a clip wraps
  (strike phase ~0 / settle-skip on the strike frame) books to the NEW attempt, never the old one.
* END-TO-END multiseg wrap attribution: a REAL 2-clip MotionCommand drives real wraps through the
  real racket _update_command (only target sampling shimmed) — per-clip starts/returns must match
  the actual wrap history, including the phase-0 boundary case above. 人话:换拍记账全程走真代码,
  不再手工塞 _prev_clip_id。
* _vb_book_strike_step == the old inline EMA formulas value-for-value (refactor guard).
* stagger_initial_clock: OFF (default) = no RNG consumed, hold/episode clocks byte-identical;
  ON = one-shot hold bias at each env's FIRST true reset only (wraps and later resets clean),
  one-shot episode-clock bias at the first _update_command only; missing episode clock degrades
  to a no-op instead of crashing.

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_metric_sync_fix.py -q
"""

from __future__ import annotations

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

DECAY = 0.99
MIN_COUNT = 50.0


# --------------------------------------------------------------------------------------------- #
# rally-ledger harness: a RacketTargetCommand with exactly the state the accounting methods touch
# --------------------------------------------------------------------------------------------- #
def _make_rally_cmd(n, clip_ids=None, multiseg=True):
    RT = hope_commands_mod.RacketTargetCommand
    cmd = RT.__new__(RT)
    cmd.num_envs = n
    cmd.device = "cpu"
    cmd.cfg = types.SimpleNamespace(
        exact_success_decay=DECAY, exact_success_min_count=MIN_COUNT,
        rally_legacy_metrics=True, virtual_ball=True, vb_metrics_only=False)
    cmd._clip_names = {0: "forehand", 1: "backhand"}
    # vb outcome ledgers (old / mixed-schedule)
    cmd._vb_exact_acc = 0.0
    cmd._vb_hit_acc = 0.0
    cmd._vb_net_acc = 0.0
    cmd._vb_land_valid_acc = 0.0
    cmd._vb_inb_acc = 0.0
    cmd._vb_exact_acc_c = {0: 0.0, 1: 0.0}
    cmd._vb_inb_acc_c = {0: 0.0, 1: 0.0}
    cmd._vb_hit_acc_c = {0: 0.0, 1: 0.0}
    # swing-start / fall ledgers (old)
    cmd._swing_starts_acc = 0.0
    cmd._swing_starts_acc_c = {0: 0.0, 1: 0.0}
    cmd._prestrike_fall_acc = 0.0
    cmd._poststrike_fall_acc = 0.0
    cmd._prestrike_fall_acc_c = {0: 0.0, 1: 0.0}
    cmd._poststrike_fall_acc_c = {0: 0.0, 1: 0.0}
    cmd._resample_n_acc = 0.0
    cmd._replay_n_acc = 0.0
    # NEW per-swing same-ledger rally state
    cmd._rally_active = torch.zeros(n, dtype=torch.bool)
    cmd._rally_returned = torch.zeros(n, dtype=torch.bool)
    cmd._rally_pending_return = torch.zeros(n, dtype=torch.bool)
    cmd._rally_starts_acc = 0.0
    cmd._rally_returns_acc = 0.0
    cmd._rally_starts_acc_c = {0: 0.0, 1: 0.0}
    cmd._rally_returns_acc_c = {0: 0.0, 1: 0.0}
    clip_ids = torch.zeros(n, dtype=torch.long) if clip_ids is None else clip_ids
    cmd._prev_clip_id = clip_ids.clone()
    cmd._recover_from_clip = torch.full((n,), -1, dtype=torch.long)
    cmd.pre_strike = torch.ones(n, dtype=torch.bool)
    fake_motion = types.SimpleNamespace(_multiseg=multiseg, clip_id=clip_ids.clone())
    cmd._motion = lambda: fake_motion
    cmd._env = types.SimpleNamespace(
        termination_manager=types.SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool)))
    cmd.metrics = {}
    for k in ("virtual_return_rate_rally", "virtual_return_rate_rally_forehand",
              "virtual_return_rate_rally_backhand"):
        cmd.metrics[k] = torch.zeros(n)
    return cmd


def _sim_sync_queue(n, steps, episode_len, wrap_period, strike_offset, legal, clip_ids=None):
    """Drive the REAL accounting methods through a fully synchronized 4096-style queue.

    Per-step order mirrors training exactly: true-reset booking happens BEFORE the step's
    metrics pass (manager resets precede command compute); the strike booking + legacy write
    happen inside the metrics pass (_vb_evaluate) BEFORE _decay_swing_accounting/_rally_report;
    wrap bookings happen AFTER (_update_command runs after _update_metrics).

    Every env is on the same clock (the disease under test): swings start at episode_start +
    k*wrap_period, strike strike_offset steps into each swing (skipped if the episode times out
    first), mass timeout every episode_len steps.

    Returns (cmd, legacy_writes, new_curve): legacy readings at their REAL write moments
    (strike-carrying steps), and the new metric at every step.
    """
    cmd = _make_rally_cmd(n, clip_ids=clip_ids)
    all_ids = torch.arange(n)
    ones = torch.ones(n, dtype=torch.bool)
    legacy_writes, new_curve = [], []
    for t in range(steps):
        ep_t = t % episode_len
        if ep_t == 0:
            # mass timeout -> true reset (terminated stays False: timeouts are not falls)
            cmd._count_swing_starts(all_ids, count_prestrike_falls=True)
        swing_elapsed = ep_t % wrap_period
        swing_start = ep_t - swing_elapsed
        if swing_elapsed == strike_offset and swing_start + strike_offset < episode_len:
            cmd._vb_book_strike_step(DECAY, ones, legal, legal, legal, legal)
            legacy_writes.append(cmd._rally_legacy_values()[0])
        cmd._decay_swing_accounting(DECAY)
        cmd._rally_report()
        new_curve.append(float(cmd.metrics["virtual_return_rate_rally"][0]))
        if ep_t != 0 and ep_t % wrap_period == 0:
            cmd._count_swing_starts(all_ids, count_prestrike_falls=False)  # intra-episode wrap
    return cmd, legacy_writes, new_curve


N = 4000  # divisible by 200 -> exact 72% legal per clip below
CLIPS = torch.arange(N) % 2
LEGAL_72 = (torch.arange(N) // 2) % 100 < 72  # exactly 72% of each clip


def test_sync_queue_old_ledger_breaks_one_new_ledger_never_does():
    """Clean periodic queue (every attempt strikes before timeout): true rate = 0.72 exactly."""
    cmd, legacy_writes, new_curve = _sim_sync_queue(
        N, steps=1200, episode_len=450, wrap_period=150, strike_offset=116,
        legal=LEGAL_72, clip_ids=CLIPS)
    # OLD ledger: the very first strike after the mass reset reads 0.72/0.99^116 ~ 2.31 > 1
    assert max(legacy_writes) > 1.4, f"legacy max {max(legacy_writes):.3f} did not break 1"
    assert min(legacy_writes) < max(legacy_writes) - 0.5  # and it oscillates, not just offsets
    # NEW ledger: <= 1 at EVERY step, and == the true per-swing return rate once booked
    assert max(new_curve) <= 1.0 + 1e-7
    warm = new_curve[160:]  # first paired booking lands at the t=150 wrap
    assert all(abs(v - 0.72) < 1e-5 for v in warm), (min(warm), max(warm))
    # per-clip: same-ledger pairing holds per side too
    for side in ("forehand", "backhand"):
        v = float(cmd.metrics[f"virtual_return_rate_rally_{side}"][0])
        assert v <= 1.0 + 1e-7
        assert abs(v - 0.72) < 1e-5, (side, v)


def test_sync_queue_timeout_cut_swings_count_as_failures():
    """episode_len=485 (the forensic sawtooth): the 4th swing of each episode starts at +450 and
    the mass timeout cuts it before its strike -> a failed attempt. True per-episode rate =
    3*0.72/4 = 0.54; the new curve must stay <= 0.72 (never counting the cut swing as a return)
    and <= 1 always, while the legacy curve still breaks 1."""
    cmd, legacy_writes, new_curve = _sim_sync_queue(
        N, steps=3 * 485, episode_len=485, wrap_period=150, strike_offset=116,
        legal=LEGAL_72, clip_ids=CLIPS)
    assert max(legacy_writes) > 1.4
    assert max(new_curve) <= 1.0 + 1e-7
    warm = new_curve[160:]
    assert max(warm) <= 0.72 + 1e-5
    last_period = new_curve[-485:]
    mean = sum(last_period) / len(last_period)
    assert 0.45 < mean < 0.65, mean  # true rate 0.54; EMA weighting wiggles around it


def test_first_resample_books_no_phantom_attempt():
    cmd = _make_rally_cmd(8)
    cmd._count_swing_starts(torch.arange(8), count_prestrike_falls=True)
    assert cmd._rally_starts_acc == 0.0 and cmd._rally_returns_acc == 0.0
    assert cmd._swing_starts_acc == 8.0  # the OLD ledger books starts at swing START as before
    assert bool(cmd._rally_active.all())


def test_paired_booking_and_latch_clears_at_swing_end():
    cmd = _make_rally_cmd(8)
    ids = torch.arange(8)
    cmd._count_swing_starts(ids, count_prestrike_falls=True)  # activate attempts
    latch = torch.zeros(8, dtype=torch.bool)
    latch[:4] = True
    cmd._vb_book_strike_step(DECAY, latch, latch, latch, latch, latch)
    cmd._count_swing_starts(ids, count_prestrike_falls=False)  # wrap: ends the attempts
    assert cmd._rally_starts_acc == 8.0 and cmd._rally_returns_acc == 4.0
    assert not bool(cmd._rally_returned.any())  # latch consumed
    # next swing ends without any strike -> books 8 starts, 0 returns (no cross-swing leakage)
    cmd._count_swing_starts(ids, count_prestrike_falls=False)
    assert cmd._rally_starts_acc == 16.0 and cmd._rally_returns_acc == 4.0


def test_per_clip_attribution_uses_prev_clip_id():
    clip = torch.tensor([0, 0, 1, 1])
    cmd = _make_rally_cmd(4, clip_ids=clip)
    ids = torch.arange(4)
    cmd._count_swing_starts(ids, count_prestrike_falls=True)
    latch = torch.tensor([True, True, True, False])  # forehand 2/2, backhand 1/2
    cmd._vb_book_strike_step(DECAY, latch, latch, latch, latch, latch)
    # the ended attempt books to _prev_clip_id even after motion resampled clip_id for the NEW one
    cmd._motion().clip_id[:] = torch.tensor([1, 1, 0, 0])
    cmd._count_swing_starts(ids, count_prestrike_falls=False)
    assert cmd._rally_starts_acc_c == {0: 2.0, 1: 2.0}
    assert cmd._rally_returns_acc_c == {0: 2.0, 1: 1.0}


def test_wrap_step_strike_books_to_new_attempt_not_old():
    """Wrap-boundary guard (防御修 2026-07-09): a strike that fires on the SAME step a clip wraps
    (strike phase ~0, or rsi_skip_settle_frames landing on the strike offset) belongs to the swing
    STARTING at that wrap. Without the guard it latched into _rally_returned before the old attempt
    booked — the OLD rally got the credit (cross-rally leakage) and the new one lost it."""
    cmd = _make_rally_cmd(4)
    ids = torch.arange(4)
    cmd._count_swing_starts(ids, count_prestrike_falls=True)  # activate attempt #1 (books nothing)
    # attempt #1 never returns; on its wrap step the NEW clip's strike fires immediately for
    # envs 0/1 (just_resampled True), env 2 is a normal mid-swing strike, env 3 no strike.
    cmd._motion().just_resampled = torch.tensor([True, True, False, False])
    legal = torch.tensor([True, False, True, False])
    cmd._vb_book_strike_step(DECAY, legal, legal, legal, legal, legal)
    assert cmd._rally_returned.tolist() == [False, False, True, False]  # env0 parked, not latched
    assert cmd._rally_pending_return.tolist() == [True, False, False, False]
    cmd._count_swing_starts(ids[:2], count_prestrike_falls=False)  # the wrap books attempt #1
    assert cmd._rally_starts_acc == 2.0 and cmd._rally_returns_acc == 0.0  # no leak to the old rally
    assert bool(cmd._rally_returned[0])  # parked latch handed to the NEW attempt...
    assert not bool(cmd._rally_pending_return.any())  # ...and consumed (one-shot)
    cmd._count_swing_starts(ids[:1], count_prestrike_falls=False)  # env0's attempt #2 ends
    assert cmd._rally_starts_acc == 3.0 and cmd._rally_returns_acc == 1.0  # the return lands HERE


def test_vb_book_strike_step_without_wrap_signal_keeps_plain_or_latch():
    """A motion command without just_resampled (e.g. plain single-clip stubs) degrades to the
    original OR latch — no parking, no crash."""
    cmd = _make_rally_cmd(3)
    assert not hasattr(cmd._motion(), "just_resampled")
    legal = torch.tensor([True, False, True])
    cmd._vb_book_strike_step(DECAY, legal, legal, legal, legal, legal)
    assert torch.equal(cmd._rally_returned, legal)
    assert not bool(cmd._rally_pending_return.any())


def test_rally_report_min_count_gate_and_values():
    clip = (torch.arange(200) >= 100).long()
    cmd = _make_rally_cmd(200, clip_ids=clip)
    ids = torch.arange(200)
    cmd._count_swing_starts(ids, count_prestrike_falls=True)
    cmd._rally_report()
    assert float(cmd.metrics["virtual_return_rate_rally"][0]) == 0.0  # not enough samples yet
    latch = torch.zeros(200, dtype=torch.bool)
    latch[:90] = True     # forehand 90/100
    latch[100:150] = True  # backhand 50/100
    cmd._vb_book_strike_step(DECAY, latch, latch, latch, latch, latch)
    cmd._count_swing_starts(ids, count_prestrike_falls=False)
    cmd._rally_report()
    assert abs(float(cmd.metrics["virtual_return_rate_rally"][0]) - 0.70) < 1e-6
    assert abs(float(cmd.metrics["virtual_return_rate_rally_forehand"][0]) - 0.90) < 1e-6
    assert abs(float(cmd.metrics["virtual_return_rate_rally_backhand"][0]) - 0.50) < 1e-6


def test_vb_book_strike_step_matches_old_inline_formulas():
    """Refactor guard: the extracted booking == the old inline decay*acc+count lines, value for
    value, and the rally latch is a pure OR (idempotent under R14 double-fire)."""
    torch.manual_seed(7)
    cmd = _make_rally_cmd(64)
    cmd._vb_exact_acc, cmd._vb_hit_acc = 10.0, 8.0
    cmd._vb_net_acc, cmd._vb_land_valid_acc, cmd._vb_inb_acc = 6.0, 5.0, 4.0
    exact = torch.rand(64) < 0.5
    gate = exact & (torch.rand(64) < 0.8)
    net = torch.rand(64) < 0.7
    land = torch.rand(64) < 0.9
    legal = gate & net & (torch.rand(64) < 0.6)
    expect = (
        DECAY * 10.0 + float(exact.sum()),
        DECAY * 8.0 + float(gate.sum()),
        DECAY * 6.0 + float((gate & net).sum()),
        DECAY * 5.0 + float((gate & land).sum()),
        DECAY * 4.0 + float(legal.sum()),
    )
    cmd._rally_returned[0] = True  # pre-set latch must survive (OR semantics)
    cmd._vb_book_strike_step(DECAY, exact, gate, net, land, legal)
    got = (cmd._vb_exact_acc, cmd._vb_hit_acc, cmd._vb_net_acc, cmd._vb_land_valid_acc, cmd._vb_inb_acc)
    assert got == pytest.approx(expect, abs=0.0)
    assert bool(cmd._rally_returned[0])
    assert torch.equal(cmd._rally_returned[1:], legal[1:])


def test_legacy_values_min_count_gate():
    cmd = _make_rally_cmd(4)
    cmd._vb_inb_acc, cmd._swing_starts_acc = 10.0, 5.0
    cmd._vb_exact_acc = MIN_COUNT - 1.0
    assert cmd._rally_legacy_values()[0] == 0.0
    cmd._vb_exact_acc = MIN_COUNT
    g, per = cmd._rally_legacy_values()
    assert g == pytest.approx(2.0)  # the legacy ratio CAN exceed 1 — that is the kept-for-对照 disease
    assert set(per) == {"forehand", "backhand"}


# --------------------------------------------------------------------------------------------- #
# REAL multiseg wrap -> per-clip attribution, end to end (no hand-stubbed _prev_clip_id)
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def two_clips():
    tmp = tempfile.mkdtemp(prefix="metric_sync_two_clips_")
    return (
        _write_motion_npz(os.path.join(tmp, "forehand.npz"), frames=20),
        _write_motion_npz(os.path.join(tmp, "backhand.npz"), frames=15),
    )


def _make_wrap_rig(two_clips, strike_phase_per_clip, num_envs=8, seed=11):
    """REAL MotionCommand (2 concatenated clips) + a rally harness whose _update_command /
    _compute_strike_timing / _count_swing_starts are the REAL shipped racket methods — so the whole
    wrap protocol (just_resampled detection, _recover_from_clip latch, _prev_clip_id snapshot
    ordering, wrap-boundary return parking) runs as in training. Only _resample_command is shimmed
    down to its accounting head (re-arm the exact latch + _count_swing_starts, hope_commands
    L1170-1175): target/base sampling needs the full robot scene and is orthogonal to attribution.
    """
    torch.manual_seed(seed)
    mcmd, robot = _make_motion_command(list(two_clips), num_envs=num_envs)
    n_bodies = len(mcmd.cfg.body_names)
    quat = torch.zeros(num_envs, n_bodies, 4)
    quat[..., 0] = 1.0
    robot.data.body_pos_w = torch.zeros(num_envs, n_bodies, 3)
    robot.data.body_quat_w = quat
    cmd = _make_rally_cmd(num_envs)
    cmd._motion = lambda: mcmd
    cmd.cfg.strike_phase = 0.5
    cmd.cfg.strike_phase_per_clip = strike_phase_per_clip
    cmd.cfg.strike_window_s = 0.1
    cmd.cfg.strike_window_pos_s = None
    cmd.cfg.strike_window_wide_s = None
    cmd.cfg.midswing_resample_prob = 0.0
    cmd._strike_phase_per_clip_t = None
    cmd._env.step_dt = 0.02
    cmd._prev_motion_steps = mcmd.time_steps.clone()
    cmd._exact_fired = torch.zeros(num_envs, dtype=torch.bool)
    cmd._resample_is_wrap = False
    cmd._actor_view_active = False

    def _resample_accounting_only(env_ids):
        cmd._exact_fired[env_ids] = False
        cmd._count_swing_starts(env_ids, count_prestrike_falls=not cmd._resample_is_wrap)

    cmd._resample_command = _resample_accounting_only
    ids = torch.arange(num_envs)
    mcmd._resample_command(ids)  # true-reset birth: random clip, frame at seg_start, hold armed
    cmd._resample_command(ids)   # manager reset: activates attempt #1 (books nothing)
    return mcmd, cmd


def _drive_wrap_rig(mcmd, cmd, legal_clip, steps):
    """Per-step order mirrors training: motion term computes first (advance/wrap/resample), then the
    racket metrics pass (REAL timing + REAL strike booking), then the racket _update_command (REAL
    wrap booking). Ground truth is read off the REAL motion command — the clip an env was on BEFORE
    the wrap is the clip its ended attempt must book to. decay=1.0 -> accumulators are exact counts.
    Returns (expected starts per clip, expected returns per clip, forehand->backhand wrap count)."""
    exp_starts = {0: 0.0, 1: 0.0}
    exp_returns = {0: 0.0, 1: 0.0}
    fh_to_bh = 0
    for _ in range(steps):
        clip_before = mcmd.clip_id.clone()
        mcmd._update_command()
        cmd._compute_strike_timing()
        exact = cmd.time_to_strike.abs() <= (0.5 * cmd._env.step_dt + 1e-6)  # hope_commands exact_strike
        legal = exact & (mcmd.clip_id == legal_clip)
        if bool(exact.any()):  # _vb_evaluate returns early on strike-free steps
            cmd._vb_book_strike_step(1.0, exact, legal, legal, legal, legal)
        cmd._update_command()
        for i in torch.where(mcmd.just_resampled)[0].tolist():
            c = int(clip_before[i])
            exp_starts[c] += 1.0
            if c == legal_clip:  # every completed swing passes its own strike frame exactly once
                exp_returns[c] += 1.0
            if c == 0 and int(mcmd.clip_id[i]) == 1:
                fh_to_bh += 1
    return exp_starts, exp_returns, fh_to_bh


def test_e2e_multiseg_wrap_attributes_ended_attempts_to_true_clip(two_clips):
    """人话:真跑双 clip 动作库,让 env 自然换拍(wrap),验证"这拍记到哪个 clip 头上"全程走真代码
    (_prev_clip_id 由真 _update_command 快照,不再手工塞)。正手拍必回球、反手拍必不回,账本上
    每个 clip 的起拍数/回球数必须和真实换拍历史一根一根对得上。"""
    mcmd, cmd = _make_wrap_rig(two_clips, strike_phase_per_clip=(0.47, 0.333))
    exp_starts, exp_returns, _ = _drive_wrap_rig(mcmd, cmd, legal_clip=0, steps=160)
    assert exp_starts[0] > 0 and exp_starts[1] > 0  # both clips actually exercised
    assert cmd._rally_starts_acc == exp_starts[0] + exp_starts[1]
    assert cmd._rally_returns_acc == exp_returns[0]
    assert cmd._rally_starts_acc_c == exp_starts
    assert cmd._rally_returns_acc_c == exp_returns
    assert cmd._rally_returns_acc_c[1] == 0.0  # nothing ever leaks onto the backhand ledger


def test_e2e_wrap_boundary_strike_no_cross_rally_leak(two_clips):
    """人话:把反手击球点放在 clip 第 0 帧——换拍那一步新拍立刻击球(②防御修针对的边界)。回球必须
    记给刚开始的新拍;修复前它会记给还没入账的上一拍(正手账本凭空多出回球)。"""
    mcmd, cmd = _make_wrap_rig(two_clips, strike_phase_per_clip=(0.5, 0.0), seed=13)
    exp_starts, exp_returns, fh_to_bh = _drive_wrap_rig(mcmd, cmd, legal_clip=1, steps=160)
    assert fh_to_bh > 0  # the leaky transition (forehand ends, backhand strikes at entry) occurred
    assert cmd._rally_returns_acc_c[0] == 0.0  # forehand never returns — no cross-rally credit
    assert cmd._rally_starts_acc_c == exp_starts
    assert cmd._rally_returns_acc_c == exp_returns  # every ended backhand attempt kept its return
    assert cmd._rally_returns_acc == exp_starts[1]


# --------------------------------------------------------------------------------------------- #
# stagger_initial_clock (real MotionCommand on a synthetic clip)
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def clip():
    tmp = tempfile.mkdtemp(prefix="metric_sync_clip_")
    return _write_motion_npz(os.path.join(tmp, "single.npz"), frames=30)


def _make_stagger_command(clip_path, **cfg_overrides):
    """_make_motion_command + the live-robot body pose buffers _update_command's tail reads."""
    cmd, robot = _make_motion_command([clip_path], **cfg_overrides)
    n, n_bodies = cmd.num_envs, len(cmd.cfg.body_names)
    quat = torch.zeros(n, n_bodies, 4)
    quat[..., 0] = 1.0
    robot.data.body_pos_w = torch.zeros(n, n_bodies, 3)
    robot.data.body_quat_w = quat
    return cmd, robot


def test_stagger_off_is_inert_and_consumes_no_rng(clip):
    torch.manual_seed(0)
    cmd, _ = _make_stagger_command(clip)  # default: stagger_initial_clock=False
    assert cmd._stagger_hold_pending is None and cmd._stagger_ep_pending is False
    cmd._env.episode_length_buf = torch.zeros(cmd.num_envs, dtype=torch.long)
    cmd._env.max_episode_length = 485
    state = torch.random.get_rng_state()
    cmd._update_command()
    assert torch.equal(state, torch.random.get_rng_state())  # no stagger RNG draw on the default path
    assert torch.equal(cmd._env.episode_length_buf, torch.zeros(cmd.num_envs, dtype=torch.long))
    cmd._resample_command(torch.arange(cmd.num_envs))
    assert torch.all(cmd.hold_counter == 5)  # hold_steps_range=(5,5), no bias


def test_stagger_hold_bias_first_true_reset_only(clip):
    torch.manual_seed(1)
    cmd, _ = _make_stagger_command(clip, stagger_initial_clock=True)
    ids = torch.arange(cmd.num_envs)
    cmd._resample_command(ids)  # first TRUE reset: one-shot uniform bias on top of the base hold
    assert torch.all(cmd.hold_counter >= 5)
    assert int(cmd.hold_counter.max()) > 5
    assert int(cmd.hold_counter.max()) <= 5 + cmd.cfg.stagger_hold_max_steps
    assert not bool(cmd._stagger_hold_pending.any())
    cmd._resample_command(ids)  # SECOND true reset: bias never re-applied
    assert torch.all(cmd.hold_counter == 5)


def test_stagger_hold_bias_skips_wraps(clip):
    torch.manual_seed(2)
    cmd, _ = _make_stagger_command(clip, stagger_initial_clock=True)
    ids = torch.arange(cmd.num_envs)
    cmd._resampling_from_wrap = True
    try:
        cmd._resample_command(ids)  # wrap path: plain hold draw, pending stays armed
    finally:
        cmd._resampling_from_wrap = False
    assert torch.all(cmd.hold_counter == 5)
    assert bool(cmd._stagger_hold_pending.all())


def test_stagger_episode_clock_bias_one_shot(clip):
    torch.manual_seed(3)
    cmd, _ = _make_stagger_command(clip, stagger_initial_clock=True)
    cmd._env.episode_length_buf = torch.zeros(cmd.num_envs, dtype=torch.long)
    cmd._env.max_episode_length = 485
    cmd._update_command()
    buf = cmd._env.episode_length_buf.clone()
    assert int(buf.max()) > 0  # biased (seeded; U[0,485) over 8 envs)
    assert int(buf.max()) < 485 and int(buf.min()) >= 0
    cmd._update_command()  # one-shot: second step adds nothing
    assert torch.equal(cmd._env.episode_length_buf, buf)


def test_stagger_missing_episode_clock_degrades_to_noop(clip):
    torch.manual_seed(4)
    cmd, _ = _make_stagger_command(clip, stagger_initial_clock=True)
    # fake env has no episode_length_buf/max_episode_length: must not crash, just skip (b)
    cmd._update_command()
    assert cmd._stagger_ep_pending is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
