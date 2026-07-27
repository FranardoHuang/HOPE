"""Torch side of ball -> planner task -> target motion (CPU torch, isaaclab STUBBED).

人话:三块。
1. 动作模板张量版和 numpy 版读同一份文件、同一个 sha256,字段逐个对得上——"一份真相"是可核对的,
   不是口号;
2. 选择器 batch 版和规划器 numpy 版对同一批球给同一个动作,判据一模一样;
3. 适配器 batch 版四条约束逐条断言:方向不变(逐位 v_r == s·d̂)、触球点就是球的、速度在本动作
   的上下限内且越界一律拒绝、拍面显式解出且朝对面;过网是**验收条件**不是注释。

Plus the trainer-side gates: per-clip metric buckets keyed by clip index, the per-clip incoming
ball box, and the "landing/net rewards with no bank" refusal.

HOST NOTE: this file needs torch, so it does NOT run on the py3.8 host. Run it on a pod checkout
(which is a COPY of this repo):
    python -m pytest hope_training/whole_body_tracking/tests/test_stroke_chain_torch.py -q
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = pathlib.Path(HERE).resolve().parents[2]
PROTO_PATH = REPO / "configs" / "stroke_prototypes_v1_20260727.json"
PLANNER_PKG = REPO / "hope_ws" / "src" / "hope_planner"

from test_reward_flags_mdp import _PKG, _load, hope_commands_mod  # noqa: E402 (isaaclab stub)
from test_clip_family_per_clip import FAM6, _make_rt  # noqa: E402


MDP_DIR = str(REPO / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
              / "whole_body_tracking" / "tasks" / "tracking" / "mdp")
# The stub registers ``...tracking.mdp`` as a plain module, so a relative import inside a freshly
# loaded sibling ("from .strike_spec_torch import ...") has nowhere to look. Give it a __path__.
sys.modules[_PKG].__path__ = [MDP_DIR]


def _mdp(name):
    """Load a sibling mdp module through the SAME by-path loader the isaaclab stub uses."""
    return _load(f"{_PKG}.{name}", f"{name}.py")


@pytest.fixture(scope="module")
def sp_torch():
    return _mdp("stroke_prototypes_torch")


@pytest.fixture(scope="module")
def sa_torch():
    return _mdp("stroke_adapt_torch")


@pytest.fixture(scope="module")
def vb():
    return _mdp("virtual_ball")


@pytest.fixture(scope="module")
def prm(vb):
    """The VENUE-fit ball parameters — the same yaml the trainer and the scorer read."""
    return vb.load_venue_params(str(REPO / "configs" / "ball_physics_venue.yaml"))


@pytest.fixture(scope="module")
def protos_t(sp_torch):
    if not PROTO_PATH.exists():
        pytest.skip(f"{PROTO_PATH} not built; run scripts/build_stroke_prototypes.py")
    return sp_torch.load_stroke_prototype_tensors(PROTO_PATH, scope="upper")


@pytest.fixture(scope="module")
def protos_np():
    if str(PLANNER_PKG) not in sys.path:
        sys.path.insert(0, str(PLANNER_PKG))
    try:
        from hope_planner.stroke_prototypes import load_stroke_prototypes
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f"hope_planner not importable here ({exc})")
    return load_stroke_prototypes(PROTO_PATH, scope="upper")


# ------------------------------------------------------- one source of truth --- #
def test_torch_and_numpy_readers_agree_field_by_field(protos_t, protos_np):
    """SAME file, same sha256, same values — the checkable form of 'one source of truth'."""
    assert protos_t.file_sha256 == protos_np.file_sha256
    assert protos_t.derived_sha256 == protos_np.derived_sha256
    assert protos_t.motion_ids == protos_np.motion_ids
    for i, p in enumerate(protos_np):
        assert protos_t.families[i] == p.family
        assert torch.allclose(protos_t.v_hat_b[i].double(),
                              torch.as_tensor(p.v_hat_b, dtype=torch.float64), atol=1e-6)
        for tk, npv in (("speed_min", p.speed_min_mps), ("speed_max", p.speed_max_mps),
                        ("v_star_cap", p.v_star_cap_mps), ("t_prepare_min", p.t_prepare_min_s),
                        ("t_prepare_max", p.t_prepare_max_s), ("slack_z_w", p.slack_z_w_m),
                        ("slack_b_xy", p.slack_b_xy_m), ("face_sign", p.face_sign)):
            assert math.isclose(float(getattr(protos_t, tk)[i]), float(npv), rel_tol=1e-6), tk
        assert int(protos_t.priority[i]) == p.priority
        assert bool(protos_t.enabled[i]) == p.enabled


def test_torch_reader_fails_closed_on_a_hand_edit(sp_torch, tmp_path):
    doc = json.loads(PROTO_PATH.read_text(encoding="utf-8"))
    doc["scopes"]["upper"][0]["speed_max_mps"] = 99.0
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="derived_sha256"):
        sp_torch.load_stroke_prototype_tensors(bad, scope="upper")


def test_torch_reader_pins_the_clip_order(sp_torch):
    with pytest.raises(ValueError, match="clip order"):
        sp_torch.load_stroke_prototype_tensors(
            PROTO_PATH, scope="upper", expected_motion_ids=("bh_block", "fh_loop"))


# ------------------------------------------------------------------ selector --- #
def _scene(protos_t, motion_id, v_in, t_avail=0.6):
    """One env whose ball sits inside `motion_id`'s own measured contact region (W_floor)."""
    i = protos_t.motion_ids.index(motion_id)
    base = torch.tensor([[0.0, 0.0, float(protos_t.band_z_w[i].mean()) - 0.42]])
    p = base + torch.cat([protos_t.p_contact_b[i:i + 1, :2],
                          torch.zeros(1, 1)], dim=-1)
    p[0, 2] = float(protos_t.band_z_w[i].mean())
    v = torch.tensor([[-float(v_in), 0.0, -0.3]])
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    aim = torch.tensor([[2.555, 0.0]])
    return p, v, torch.tensor([float(t_avail)]), base, q, aim


def test_block_cannot_answer_a_slow_ball_torch(sa_torch, protos_t, prm):
    i = protos_t.motion_ids.index("bh_block")
    for v_in, want_reject in ((0.5, True), (1.0, True), (1.5, True), (4.0, False)):
        p, v, t, b, q, aim = _scene(protos_t, "bh_block", v_in)
        out = sa_torch.select_stroke_batch(p, v, t, b, q, aim, protos_t, prm, surface_z=0.76)
        code = int(out["reject_code"][0, i])
        name = sa_torch.PREDICATES[code] if code >= 0 else ""
        if want_reject:
            assert name == "P5a_energy_insufficient", (v_in, name)
        else:
            assert code < 0, (v_in, name)


def test_closing_speed_demand_falls_with_incoming_speed(sa_torch, protos_t, prm):
    demands = []
    for v_in in (1.0, 2.0, 3.0, 4.0, 5.0):
        p, v, t, b, q, aim = _scene(protos_t, "bh_block", v_in)
        demands.append(float(sa_torch.closing_speed_demand(p, v, aim, prm, 0.76)[0]))
    assert all(y < x for x, y in zip(demands, demands[1:])), demands


def test_disabled_stroke_is_never_selected_torch(sa_torch, protos_t, prm):
    i = protos_t.motion_ids.index("fh_block_syn")
    assert not bool(protos_t.enabled[i])
    torch.manual_seed(0)
    p = torch.rand(400, 3) * torch.tensor([0.8, 1.4, 0.8]) + torch.tensor([0.3, -0.7, 0.7])
    v = torch.rand(400, 3) * torch.tensor([-4.5, 2.0, 2.0]) + torch.tensor([-0.5, -1.0, -1.0])
    t = torch.rand(400) * 1.2 + 0.3
    b = torch.zeros(400, 3)
    q = torch.zeros(400, 4)
    q[:, 0] = 1.0
    aim = torch.tensor([[2.555, 0.0]]).expand(400, 2)
    out = sa_torch.select_stroke_batch(p, v, t, b, q, aim, protos_t, prm, surface_z=0.76)
    assert not bool((out["clip_index"] == i).any())


def test_selector_returns_minus_one_when_nothing_is_admissible(sa_torch, protos_t, prm):
    p = torch.tensor([[-3.0, 0.0, 0.9]])
    v = torch.tensor([[-3.0, 0.0, 0.0]])
    out = sa_torch.select_stroke_batch(
        p, v, torch.tensor([0.6]), torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]),
        torch.tensor([[2.555, 0.0]]), protos_t, prm, surface_z=0.76)
    assert int(out["clip_index"][0]) == -1
    assert bool((out["reject_code"][0] >= 0).all())


def test_selector_parity_numpy_vs_torch(sa_torch, protos_t, protos_np, prm):
    """Row-by-row: the two implementations must pick the same stroke."""
    from hope_planner.stroke_select import SelectorCfg, select_stroke
    cfg = SelectorCfg(surface_z=0.76, paddle_e_g1=float(prm.paddle_e_g1),
                      paddle_e_g2=float(prm.paddle_e_g2), gravity=float(prm.g))
    g = torch.Generator().manual_seed(5)
    N = 300
    p = torch.rand(N, 3, generator=g) * torch.tensor([0.8, 1.4, 0.8]) \
        + torch.tensor([0.3, -0.7, 0.7])
    v = torch.rand(N, 3, generator=g) * torch.tensor([-5.0, 1.6, 2.0]) \
        + torch.tensor([-0.5, -0.8, -1.2])
    t = torch.rand(N, generator=g) * 1.3 + 0.3
    b = torch.zeros(N, 3)
    q = torch.zeros(N, 4)
    q[:, 0] = 1.0
    aim = torch.tensor([[2.555, 0.0]]).expand(N, 2).contiguous()
    out = sa_torch.select_stroke_batch(p, v, t, b, q, aim, protos_t, prm, surface_z=0.76)
    agree = 0
    for i in range(N):
        ch = select_stroke(p[i].numpy(), v[i].numpy(), None, float(t[i]), b[i].numpy(),
                           q[i].numpy(), aim[i].numpy(), protos_np, cfg=cfg)
        assert ch.clip_index == int(out["clip_index"][i]), (
            i, ch.clip_index, int(out["clip_index"][i]), ch.reject_by_stroke)
        agree += 1
    assert agree == N


# ------------------------------------------------------------------- adapter --- #
def _fit(sa_torch, protos_t, prm, motion_id, v_in, aim_x=2.555, n_iters=12):
    i = protos_t.motion_ids.index(motion_id)
    p, v, t, b, q, aim = _scene(protos_t, motion_id, v_in)
    aim = torch.tensor([[aim_x, 0.0]])
    d = sa_torch.direction_world(protos_t.v_hat_b[i:i + 1], torch.zeros(1))[:, 0, :]
    out = sa_torch.solve_strike_specs_fixed_dir(
        p, v, torch.zeros(1, 3), aim, d,
        protos_t.speed_min[i:i + 1], protos_t.speed_max[i:i + 1],
        prm, surface_z=0.76, net_x=0.5 + 1.37, n_iters=n_iters)
    return out, d, p, i


def test_c1_direction_is_exact_by_construction(sa_torch, protos_t, prm):
    for mid in ("bh_block", "bh_loop_c", "s0_highpress"):
        out, d, p, i = _fit(sa_torch, protos_t, prm, mid, 4.0)
        assert torch.allclose(out["v_r"], out["speed"][:, None] * d, rtol=0, atol=1e-6)
        # arccos is ill-conditioned at 1: in float32 a 6e-8 dot-product error already reads as
        # ~0.02 deg. The EXACT identity is the assertion above; this is the reported-angle sanity.
        assert float(sa_torch.dir_deviation_deg(out["v_r"], d)[0]) < 0.05


def test_c2_contact_position_is_the_ball(sa_torch, protos_t, prm):
    out, d, p, i = _fit(sa_torch, protos_t, prm, "bh_block", 4.0)
    assert torch.equal(out["p_contact"], p)


def test_c3_speed_inside_the_cap_and_never_clamped_into_acceptance(sa_torch, protos_t, prm):
    out, d, p, i = _fit(sa_torch, protos_t, prm, "bh_block", 4.0)
    if bool(out["ok"][0]):
        assert float(protos_t.speed_min[i]) - 1e-6 <= float(out["speed"][0]) \
            <= float(protos_t.speed_max[i]) + 1e-6
    # a landing the block cannot reach: refused with a speed reason, not a clamped ok=True
    far, d2, p2, j = _fit(sa_torch, protos_t, prm, "bh_block", 0.5, aim_x=3.1)
    assert not bool(far["ok"][0])
    assert sa_torch.REASONS[int(far["reason"][0])] in ("speed_over_cap", "resid_gt_tol",
                                                       "no_landing")


def test_c4_face_is_explicit_unit_and_opponent_facing(sa_torch, protos_t, prm):
    tilts = []
    for v_in in (3.0, 4.0, 5.0, 6.0):
        out, d, p, i = _fit(sa_torch, protos_t, prm, "bh_block", v_in)
        n = out["n"][0]
        assert abs(float(torch.linalg.norm(n)) - 1.0) < 1e-6
        if bool(out["ok"][0]):
            assert float(n[0]) > 1e-6
            tilts.append(math.degrees(math.asin(float(n[2].clamp(-1, 1)))))
    assert tilts, "no accepted solve to inspect"


def test_net_clearance_is_an_acceptance_condition_torch(sa_torch, protos_t, prm):
    """Aim just past the net: any flight landing there passes under the net top."""
    out, d, p, i = _fit(sa_torch, protos_t, prm, "bh_block", 3.5, aim_x=1.92)
    assert not bool(out["ok"][0])
    assert sa_torch.REASONS[int(out["reason"][0])] in (
        "net_not_cleared", "speed_under_min", "speed_over_cap", "resid_gt_tol")


def test_adapter_does_not_maximise_speed_torch(sa_torch, protos_t, prm):
    speeds = []
    for v_in in (3.5, 4.5, 5.5):
        out, d, p, i = _fit(sa_torch, protos_t, prm, "bh_block", v_in)
        if bool(out["ok"][0]):
            speeds.append(float(out["speed"][0]))
    assert len(speeds) >= 2
    assert all(y < x + 1e-6 for x, y in zip(speeds, speeds[1:])), speeds


# -------------------------------------------------- trainer: per-clip buckets --- #
def _families_for(nseg):
    """A legal family table of length nseg (needs >= 1 of each family, commands.py:425)."""
    if nseg <= 2:
        return None                       # absent -> the legacy (forehand, backhand) derivation
    half = nseg // 2
    return tuple(["forehand"] * half + ["backhand"] * (nseg - half))


def _rt_with_names(num_envs, clip_ids, num_segments, names):
    rt = _make_rt(num_envs, clip_ids, num_segments, families=_families_for(num_segments))
    rt._family_names = {0: "forehand", 1: "backhand"}
    rt._metric_buckets_per_clip = bool(names)
    rt._clip_names = {i: n for i, n in enumerate(names)} if names else dict(rt._family_names)
    rt._metric_bucket_rows_t = None
    rt._clip_family_rows_t = None
    return rt


FIVE = ["fh_loop", "bh_loop_c", "s0_highpress", "bh_block", "fh_block_syn"]


def test_metric_buckets_default_to_family_rows_bitwise():
    """铁律:没配名字表 = 现役行为逐字节不变(桶就是家族行号)。"""
    clips = torch.tensor([0, 1, 1, 0])
    rt = _rt_with_names(4, clips, 2, names=None)
    assert torch.equal(rt._metric_bucket_rows(), rt._clip_family_rows())


def test_metric_buckets_key_on_clip_index_when_names_are_given():
    """五个动作 = 五个桶;同族的 fh_loop / fh_block_syn 不再共用一个桶。"""
    clips = torch.arange(5)
    rt = _rt_with_names(5, clips, 5, names=FIVE)
    rows = rt._metric_bucket_rows()
    assert rows.tolist() == [0, 1, 2, 3, 4]
    fam = rt._clip_family_rows()
    same_family = [i for i in range(5) if fam[i] == fam[0]]
    assert len(same_family) >= 2, "need two clips in one family to make the point"
    a, b = same_family[0], same_family[1]
    assert fam[a] == fam[b]                    # SAME family row (that was the old bucket)
    assert rows[a] != rows[b]                  # DIFFERENT metric bucket now


def test_metric_bucket_name_table_length_is_fail_closed():
    rt = _rt_with_names(3, torch.arange(3), 3, names=FIVE)      # 5 names, 3 clips
    with pytest.raises(ValueError) as exc:
        rt._metric_bucket_rows()
    assert "5 name(s)" in str(exc.value) and "3 clip(s)" in str(exc.value)


def test_family_scoped_consumers_are_untouched():
    """回放 / 框表按族展开 / 题库按族寻址仍走 _clip_family_rows,不受名字表影响。"""
    src = (REPO / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
           / "whole_body_tracking" / "tasks" / "tracking" / "mdp" / "hope_commands.py").read_text()
    assert "expanded = table[self._clip_family_rows()]" in src        # box family expansion
    assert "self._qb_family_table = (~is_forehand).to(dtype=torch.long)" in src  # bank addressing
    assert '_family_names' in src   # the bank family row keeps FAMILY names


# ------------------------------------------------ trainer: incoming-ball gates --- #
def _cfg(**kw):
    """A cfg stand-in carrying the SHIPPED class-level defaults (the stub's configclass is not a
    dataclass, so read the class attributes directly)."""
    C = hope_commands_mod.RacketTargetCommandCfg
    fields = [n for n in dir(C) if not n.startswith("__") and not callable(getattr(C, n, None))]
    obj = types.SimpleNamespace(**{n: getattr(C, n) for n in fields})
    for k, v in kw.items():
        setattr(obj, k, v)
    return obj


def test_incoming_ball_box_gate_rejects_a_ball_that_flies_away():
    rt = _rt_with_names(2, torch.tensor([0, 1]), 2, names=None)
    rt.cfg = _cfg(vb_vel_range_per_clip=(
        ((-4.5, -2.0), (-0.6, 0.6), (-1.0, 0.5)),
        ((0.5, 2.0), (-0.6, 0.6), (-1.0, 0.5)),        # <- flies AWAY from the robot
    ))
    with pytest.raises(ValueError) as exc:
        rt._assert_incoming_ball_boxes_are_sane()
    msg = str(exc.value)
    assert "backhand" in msg and "x_hi=2.0000" in msg


def test_incoming_ball_box_gate_rejects_an_empty_axis():
    rt = _rt_with_names(1, torch.tensor([0]), 1, names=None)
    rt.cfg = _cfg(vb_vel_range_per_clip=(((-2.0, -4.5), (-0.6, 0.6), (-1.0, 0.5)),))
    with pytest.raises(ValueError, match="an empty box"):
        rt._assert_incoming_ball_boxes_are_sane()


def test_new_cfg_fields_exist_with_backward_compatible_defaults():
    c = _cfg()
    assert c.vb_vel_range_per_clip is None
    assert c.vb_spin_abs_max_per_clip is None
    assert c.clip_names_per_clip == ()
    assert c.reference_return_gate_min_rate == 0.0
    assert c.allow_unbanked_landing_rewards is False


def test_landing_rewards_without_a_bank_are_refused():
    """The construction that made obeying the velocity command anti-correlated with returning."""
    src_cls = hope_commands_mod.RacketTargetCommand
    import inspect
    src = inspect.getsource(src_cls.__init__)
    assert "allow_unbanked_landing_rewards" in src
    assert "anti-correlated with returning the ball" in src


# ------------------------------------------------ continuous vs discrete questions --- #
@pytest.fixture(scope="module")
def cq():
    _mdp("strike_spec_torch")
    _mdp("stroke_prototypes_torch")
    _mdp("stroke_adapt_torch")
    return _mdp("continuous_questions")


def test_continuous_draw_is_continuous_not_a_case_list(cq, prm):
    """连续 = 每次画的球都不一样,不是回放一张定长题表。"""
    n = 64
    clips = torch.zeros(n, dtype=torch.long)
    cfg = cq.ContinuousQuestionCfg(n_iters=4)
    a = cq.generate(clips, prm, surface_z=0.76, net_x=1.87, cfg=cfg)
    b = cq.generate(clips, prm, surface_z=0.76, net_x=1.87, cfg=cfg)
    assert not torch.equal(a.v_ball_in, b.v_ball_in)         # a fresh continuous draw each call
    assert len(torch.unique(a.v_ball_in[:, 0])) > n // 2     # not a small case list


def test_continuous_and_discrete_paths_are_the_same_physics(cq, prm):
    """同一颗球走两条路必须得到同一个 demanded state。

    离散题库是把 ``solve_strike_specs`` 的答案先算好存成 npz;连续路径是同一颗球、同一个求解器、
    同一份 VirtualBallParams 当场算。所以"同物理"的可核对形式是:把生成器自己画出来的球再单独喂
    给求解器,输出必须逐位相同——生成器没有偷偷加自己的物理。
    """
    strike_spec_torch = _mdp("strike_spec_torch")
    n = 48
    clips = torch.zeros(n, dtype=torch.long)
    cfg = cq.ContinuousQuestionCfg(n_iters=6)
    out = cq.generate(clips, prm, surface_z=0.76, net_x=1.87, cfg=cfg)
    # unsolved rows are NaN by construction, so replay only the solved ones
    keep = out.ok
    replay = strike_spec_torch.solve_strike_specs(
        out.p_contact[keep], out.v_ball_in[keep], out.w_ball_in[keep], out.aim_xy[keep], prm,
        surface_z=0.76, net_x=1.87, speed_budget=float(cfg.speed_budget),
        n_iters=int(cfg.n_iters), tol_m=float(cfg.tol_m),
    )
    same = replay["ok"]
    assert int(same.sum()) > n // 2, "too few rows solved on both paths to compare"
    assert torch.allclose(out.v_racket[keep][same], replay["v_r"][same], rtol=0, atol=1e-6)
    assert torch.allclose(out.n_racket[keep][same], replay["n"][same], rtol=0, atol=1e-6)
    assert torch.allclose(out.resid_m[keep][same], replay["resid_m"][same], rtol=0, atol=1e-6)


def test_unsolvable_ball_is_counted_not_substituted(cq, prm):
    """解不出来是响亮且被计数的事件,不许拿别的目标顶上。"""
    n = 32
    clips = torch.zeros(n, dtype=torch.long)
    # aim BEHIND the robot: no legal answer exists for any drawn ball
    cfg = cq.ContinuousQuestionCfg(n_iters=4, max_redraw_rounds=2,
                                   aim_x_range=(-3.0, -2.5), aim_y_range=(-0.1, 0.1))
    out = cq.generate(clips, prm, surface_z=0.76, net_x=1.87, cfg=cfg)
    assert out.exhausted > 0
    # 结构性失败策略:无解行整行 NaN,任何调用方都装不进去(不是靠记得 mask)
    assert bool(torch.isnan(out.p_contact[~out.ok]).all())
    assert bool(torch.isnan(out.v_racket[~out.ok]).all())
    assert int((~out.ok).sum()) == out.exhausted
    assert out.reason_counts, "a failure must carry a named reason, never a bare count"
    # nothing was substituted, and nothing INSTALLABLE came back either: a failed row used to be
    # returned carrying its last failed attempt, i.e. indistinguishable from a solved row except
    # via `ok`. It is NaN now, and resid_m survives so the failure is still diagnosable.
    assert torch.isnan(out.v_ball_in[~out.ok]).all()
    assert not torch.isnan(out.resid_m[~out.ok]).any()      # resid survives for diagnosis


def test_recipe_gate_refuses_a_bank_run_without_face_command():
    with pytest.raises(ValueError, match="face_command"):
        hope_commands_mod._assert_solved_target_recipe_is_coherent(
            _cfg(question_bank="bank.npz", face_command=False))


def test_recipe_gate_refuses_a_missing_face_sign_table():
    with pytest.raises(ValueError, match="mount_normal_sign_per_clip"):
        hope_commands_mod._assert_solved_target_recipe_is_coherent(
            _cfg(question_bank="bank.npz", face_command=True, mount_normal_sign_per_clip=()))


def test_recipe_gate_refuses_a_bad_face_sign_value():
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        hope_commands_mod._assert_solved_target_recipe_is_coherent(
            _cfg(question_bank="bank.npz", face_command=True,
                 mount_normal_sign_per_clip=(1.0, 0.0)))


def test_recipe_gate_names_the_dead_knobs():
    with pytest.raises(ValueError) as exc:
        hope_commands_mod._assert_solved_target_recipe_is_coherent(
            _cfg(question_bank="bank.npz", face_command=True, target_mode="uniform",
                 mount_normal_sign_per_clip=(1.0, -1.0),
                 racket_vel_range_per_clip=(((1.0, 2.0), (-1.0, 1.0), (0.0, 1.0)),),
                 ref_vel_scale=0.6))
    msg = str(exc.value)
    assert "racket_vel_range_per_clip" in msg and "ref_vel_scale" in msg
    assert "DEAD" in msg


def test_recipe_gate_is_silent_for_a_coherent_recipe():
    hope_commands_mod._assert_solved_target_recipe_is_coherent(
        _cfg(question_bank="bank.npz", face_command=True, target_mode="uniform",
             mount_normal_sign_per_clip=(1.0, -1.0)))


def test_recipe_gate_refuses_a_live_perturbation_curriculum():
    """The shipped default target_mode IS reference_perturbed with a non-zero curriculum, so a bank
    run that forgets to zero it is exactly the silent-dead-knob case this gate exists for."""
    with pytest.raises(ValueError) as exc:
        hope_commands_mod._assert_solved_target_recipe_is_coherent(
            _cfg(question_bank="bank.npz", face_command=True,
                 target_mode="reference_perturbed",
                 mount_normal_sign_per_clip=(1.0, -1.0)))
    assert "ref_perturb_pos" in str(exc.value)


def test_recipe_gate_does_not_touch_a_plain_box_run():
    hope_commands_mod._assert_solved_target_recipe_is_coherent(_cfg())   # must not raise


def test_contact_point_box_stays_live_under_a_solved_target():
    """它不是死旋钮:solved 模式下它就是触球点(=球的到达点)的连续分布。"""
    hope_commands_mod._assert_solved_target_recipe_is_coherent(
        _cfg(question_bank="bank.npz", face_command=True, target_mode="uniform",
             mount_normal_sign_per_clip=(1.0, -1.0),
             racket_pos_range_per_clip=(((0.5, 0.6), (-0.4, 0.4), (0.8, 1.2)),)))
