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
