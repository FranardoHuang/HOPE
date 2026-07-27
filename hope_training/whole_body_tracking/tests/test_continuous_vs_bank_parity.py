"""CONTINUOUS vs DISCRETE: same ball, same physics — on REAL bank rows (CPU torch, isaaclab STUBBED).

这就是 ``continuous_questions.py`` 模块 docstring 一直在承诺、但从来不存在的那个文件。

它断言的是**物理后果**,不是 demanded state 逐位相等——后者即使写出来也是假的:5 自由度
(theta, phi, v_n, v_t1, v_t2) 对 5 维残差(2 落点 + 3 个 w_speed 正则)的解流形不是逐点唯一的,
拍面转几度、速度补一点,落点一样(实测 |Δv_r| 最大 0.37 m/s、拍面夹角最大 4.31 deg)。所以断言的
是真正不变的东西:同一颗球、同一份物理、同一个落点、同一侧拍面、同一个速度模长。

七条(阈值全部是在仓库自带的两份真题库上量出来的,余量 4-5 倍):

A. 覆盖率  —— torch 求解器解出 numpy planner 解出的行的 >= 90%   (实测 95.2% / 99.3%)
B. 符号    —— 双方 dot(n, clip_face) > 0,**并带负对照**:去掉 ref_normal 必须产出 dot < 0
              (实测 763/763、746/746 行全翻)。这条负断言是防复发卫兵:谁把符号对齐删掉,测试
              当场红,不用起 env、不用等 3000 迭代。
C. 落点    —— 两条路落点分歧 <= tol_m,各自离瞄点 <= planner TOL + 1 mm
              (实测 p50 1.6/1.7 mm,max 3.8/3.1 mm;离瞄点 max 4.98/4.80 mm)
D. 过网    —— 两个答案都得 net_valid 且过网高度 > 网顶 + 球半径(题库一直拒过网失败并计数,
              连续路径的自由解分支以前一条都不判 —— 这条同时是新加判据的验收)
E. 难度    —— 运行时现算的 angle(n, clip_face) 复现题库存的 difficulty_deg,<= 5 deg
              (实测 max 3.54/3.47 deg)。保住 metrics["question_difficulty_deg"] 量的是同一个量。
F. 速度    —— 只断言**模长** <= 0.1 m/s(实测 max 46.5/31.1 mm/s);方向刻意不断言,理由见上。
G. 前提    —— 题库自己的 incoming_spin_mode == 'zero' 且 |spin|.max() == 0。这是给
              cq_spin_abs_max=0.0 这个默认值的**可执行锚**:哪天题库开始带旋转,这条会红。

还有 H:seam 平价 —— 题库路和连续缓冲路必须走同一段安装代码,且都**不**写
racket_target_normal_w(那是溯源道;_install_event_training_questions / install_external_exam_
questions 会覆盖它,连续路径必须对齐前者)。

HOST NOTE: 需要 torch,所以**不**在 py3.8 host 上跑。在 pod 的 checkout 上跑:
    python -m pytest hope_training/whole_body_tracking/tests/test_continuous_vs_bank_parity.py -q
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = pathlib.Path(HERE).resolve().parents[2]

from test_reward_flags_mdp import _PKG, _load  # noqa: E402 (installs the isaaclab stub)

MDP_DIR = str(REPO / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
              / "whole_body_tracking" / "tasks" / "tracking" / "mdp")
sys.modules[_PKG].__path__ = [MDP_DIR]

CFG_DIR = REPO / "hope_training" / "whole_body_tracking" / "cfg"
BANKS = [
    CFG_DIR / "bh_loop_c_upper_wave1_s2_5_train.npz",
    CFG_DIR / "bh_loop_c_upper_wave1_s1_2p5_train.npz",
]
NET_X_OFF = 1.37        # geometry.NET_X
NET_HEIGHT = 0.1525     # geometry.NET_HEIGHT
PLANNER_TOL_M = 0.005   # hope_planner.strike_spec_planner TOL_M


def _mdp(name):
    return _load(f"{_PKG}.{name}", f"{name}.py")


@pytest.fixture(scope="module")
def cq():
    _mdp("virtual_ball")
    _mdp("strike_spec_torch")
    _mdp("stroke_prototypes_torch")
    _mdp("stroke_adapt_torch")
    return _mdp("continuous_questions")


@pytest.fixture(scope="module")
def prm():
    vb = _mdp("virtual_ball")
    return vb.load_venue_params(str(REPO / "configs" / "ball_physics_venue.yaml"))


def _bank(path):
    """Raw npz rows + meta. Deliberately NOT through load_question_bank: this test is about the
    stored answers, and the loader's own physics-contract check is a different assertion."""
    data = np.load(str(path))
    meta = json.loads(bytes(np.asarray(data["meta_json"], dtype=np.uint8)).decode("utf-8"))
    clip = str(meta["clip_order"][0])
    t = lambda k: torch.as_tensor(np.asarray(data[f"{clip}/{k}"]), dtype=torch.float32)  # noqa: E731
    return meta, clip, {
        "contact": t("contact_pos_env").reshape(3),
        "clip_normal": t("clip_normal").reshape(3),
        "v_in": t("incoming_vel"),
        "w_in": t("incoming_spin"),
        "n": t("demanded_normal"),
        "v_r": t("demanded_vel"),
        "difficulty_deg": t("difficulty_deg"),
    }


def _planes(meta, prm):
    """The three numbers the TRAINER solves and scores against — surface + ball radius, not bare
    0.76 (the bank generator uses the same plane; a bare surface solves 20 mm off)."""
    surf = float(meta["table_surface_z"]) + float(prm.ball_radius)
    net_x = float(meta["near_x"]) + NET_X_OFF
    net_top = float(meta["table_surface_z"]) + NET_HEIGHT + float(prm.ball_radius)
    return surf, net_x, net_top


@pytest.mark.parametrize("path", BANKS, ids=lambda p: pathlib.Path(p).stem)
def test_continuous_and_bank_agree_on_physics(cq, prm, path):
    """A/B/C/D/F: same ball through both paths -> same landing, same net verdict, same face side."""
    if not pathlib.Path(path).is_file():
        pytest.skip(f"bank not present: {path}")
    meta, _clip, rows = _bank(path)
    surf, net_x, net_top = _planes(meta, prm)
    n = int(rows["v_in"].shape[0])
    aim = torch.tensor(meta["landing_env"], dtype=torch.float32).unsqueeze(0).expand(n, 2)
    ref = rows["clip_normal"].unsqueeze(0).expand(n, 3)
    p_c = rows["contact"].unsqueeze(0).expand(n, 3)

    rep = cq.parity_report(
        p_c, rows["v_in"], rows["w_in"], aim, ref, rows["v_r"], rows["n"], prm,
        surface_z=surf, net_x=net_x, net_top_z=net_top,
        speed_budget=float(meta["speed_budget"]), tol_m=PLANNER_TOL_M, n_iters=12,
        land_tol_m=0.02, aim_tol_m=PLANNER_TOL_M + 1e-3, speed_tol_mps=0.10,
        min_coverage=0.90,
    )
    assert not rep["failures"], "\n  - ".join([""] + rep["failures"]) + f"\nstats={rep['stats']}"
    # Reported, never asserted: the demanded states DIFFER, and that is the correct answer.
    assert rep["stats"]["face_angle_max_deg"] >= 0.0


@pytest.mark.parametrize("path", BANKS, ids=lambda p: pathlib.Path(p).stem)
def test_face_sign_alignment_is_load_bearing_negative_control(cq, prm, path):
    """B negative arm: WITHOUT ref_normal this backhand clip comes back 100% flipped.

    每一条防复发断言都必须先在"没修的代码"上被证明是红的,否则它证明不了任何东西。这一条就是
    那个红:``_seed`` 硬把种子法向翻到 "+x opponent-facing"(strike_spec_torch.py:91-92),而这个
    反手 clip 的 clip_normal x 分量是负的,所以不传 ref_normal = 每一颗反手球被 180 度翻过来。
    """
    if not pathlib.Path(path).is_file():
        pytest.skip(f"bank not present: {path}")
    meta, _clip, rows = _bank(path)
    if float(rows["clip_normal"][0]) >= 0.0:
        pytest.skip("this clip's face already agrees with the solver's +x seed convention")
    surf, net_x, _net_top = _planes(meta, prm)
    n = int(rows["v_in"].shape[0])
    aim = torch.tensor(meta["landing_env"], dtype=torch.float32).unsqueeze(0).expand(n, 2)
    ref = rows["clip_normal"].unsqueeze(0).expand(n, 3)
    p_c = rows["contact"].unsqueeze(0).expand(n, 3)

    flipped, solved = cq.face_sign_negative_control(
        p_c, rows["v_in"], rows["w_in"], aim, ref, prm, surface_z=surf, net_x=net_x,
        speed_budget=float(meta["speed_budget"]), tol_m=PLANNER_TOL_M, n_iters=12,
    )
    assert solved > 0, "nothing solved — the negative control proves nothing"
    assert flipped == solved, (
        f"expected EVERY unaligned answer to point away from the clip face on this backhand clip, "
        f"got {flipped}/{solved}. If this ever stops being 100%, the seed convention changed and "
        f"the positive assertion above may be passing for the wrong reason."
    )


@pytest.mark.parametrize("path", BANKS, ids=lambda p: pathlib.Path(p).stem)
def test_difficulty_deg_is_the_same_quantity(cq, prm, path):
    """E: angle(demanded face, clip face) recomputed at runtime reproduces the bank's own column."""
    if not pathlib.Path(path).is_file():
        pytest.skip(f"bank not present: {path}")
    _meta, _clip, rows = _bank(path)
    ang = torch.rad2deg(torch.arccos(
        (rows["n"] @ rows["clip_normal"]).clamp(-1.0, 1.0)))
    err = (ang - rows["difficulty_deg"]).abs()
    assert float(err.max()) <= 5.0, f"difficulty_deg reproduction off by {float(err.max()):.2f} deg"


@pytest.mark.parametrize("path", BANKS, ids=lambda p: pathlib.Path(p).stem)
def test_bank_incoming_spin_is_zero(path):
    """G: the executable anchor for the cq_spin_abs_max = 0.0 default.

    每一份 schema-v3 题库都硬断言 incoming_spin_mode == 'zero'。哪天题库开始带旋转,这条会红,
    提醒连续侧的旋转默认值要重新决定 —— 而不是让它悄悄过期。
    """
    if not pathlib.Path(path).is_file():
        pytest.skip(f"bank not present: {path}")
    meta, _clip, rows = _bank(path)
    assert meta.get("incoming_spin_mode") == "zero"
    assert float(rows["w_in"].abs().max()) == 0.0


def test_generate_is_frame_correct_by_construction(cq, prm):
    """The origins parameter is GONE, so the frame bug cannot be re-introduced by a caller.

    以前 p_contact 是世界系(origins + box)、aim_x 故意不加 origin 而 aim_y 加了、surface_z/net_x
    又是 env-local 标量。两个调用方都传 zeros 所以从没显形;origin=(3,0,0) 时它会**自信地**收敛
    到一个短 3 米、球网在机器人背后的答案(99.22% "解出")。现在签名里没有这个参数了。
    """
    import inspect

    sig = inspect.signature(cq.generate)
    assert "origins" not in sig.parameters, (
        "generate() must stay env-local: the caller adds the env origin at install time"
    )
    n = 32
    clips = torch.zeros(n, dtype=torch.long)
    cfg = cq.ContinuousQuestionCfg(n_iters=4)
    out = cq.generate(clips, prm, surface_z=0.78, net_x=1.87, cfg=cfg)
    ok = out.ok
    assert int(ok.sum()) > 0
    # env-local contact points stay inside the declared box, whatever any env origin might be
    box = torch.tensor(cfg.pos_range, dtype=torch.float32)
    assert bool((out.p_contact[ok] >= box[:, 0] - 1e-5).all())
    assert bool((out.p_contact[ok] <= box[:, 1] + 1e-5).all())


def test_net_clearance_is_actually_enforced(cq, prm):
    """D on the producer: net_top_z=None keeps the historical behaviour; a high net rejects."""
    n = 64
    clips = torch.zeros(n, dtype=torch.long)
    cfg = cq.ContinuousQuestionCfg(n_iters=4, max_redraw_rounds=1)
    loose = cq.generate(clips, prm, surface_z=0.78, net_x=1.87, cfg=cfg, net_top_z=None)
    tight = cq.generate(clips, prm, surface_z=0.78, net_x=1.87, cfg=cfg, net_top_z=3.0)
    assert int(loose.ok.sum()) > 0
    assert int(tight.ok.sum()) < int(loose.ok.sum()), "an absurd net must reject something"
    assert "net_not_cleared" in tight.reason_counts


def test_seam_installs_both_producers_through_one_shared_write():
    """H (structural): the bank branch and the continuous branch must MERGE before the writes.

    人话:设计信号写在这里 —— 这个断言哪天变得难写,就是两条路已经分叉了。特别是
    ``racket_target_normal_w``:``_apply_question_bank_targets`` 故意**不**写它(那是 clip 参考面
    的溯源道),而 ``_install_event_training_questions`` / ``install_external_exam_questions``
    **会**覆盖它。连续路径必须对齐前者;"顺手统一一下"会静默改掉 critic 的参考道。
    """
    import inspect
    import re

    hc = _load(f"{_PKG}.hope_commands", "hope_commands.py")
    src = inspect.getsource(hc.RacketTargetCommand._apply_question_bank_targets)
    head, _, tail = src.partition("BELOW THIS LINE THE TWO PRODUCERS SHARE ONE INSTALL")
    assert tail, "the shared-install marker is gone: the two producers may have forked"
    # every buffer write lives in the shared tail, none in either branch
    for field in ("racket_target_pos_w", "racket_target_vel_w", "target_normal_cmd",
                  "vb_vel_in_w", "vb_spin_in_w"):
        assert re.search(rf"self\.{field}\[env_ids\]\s*=", tail), f"{field} not written in common"
        assert not re.search(rf"self\.{field}\[env_ids\]\s*=", head), f"{field} written per-branch"
    assert not re.search(r"self\.racket_target_normal_w\s*\[", src), (
        "the seam must NOT write racket_target_normal_w — it is the clip-reference provenance lane "
        "(_install_event_training_questions / install_external_exam_questions DO overwrite it, and "
        "continuous must match _apply_question_bank_targets, not those two)"
    )


@pytest.mark.parametrize("path", BANKS, ids=lambda p: pathlib.Path(p).stem)
def test_generate_itself_is_covered_not_only_the_solver(cq, prm, path):
    """THE PRODUCTION CALL PATH. 上面几条走的是 parity_report -> solve_strike_specs;训练走的是
    generate() -> solve_strike_specs,而 generate() 自己那一次调用**以前一条断言都没有**。

    人话:实测过三种改法,把 generate() 里那一行改坏,上面全部 12 条照样绿——
      M1 不传 ref_normal      -> 每颗反手球拍面翻 180 度
      M3 传错的桌面/网平面    -> 解出来的答案落在别处(闭环量得到 167 mm)
      M4 传错的 h/n_steps     -> 通过率 0.975 塌到 0.285(落点还是对的,所以闭环量不到)
    所以这条把 generate() 的输出直接拿去闭环:自己解的答案,用**调用方给的同一套平面和同一套
    rollout 参数**打回去,必须落在瞄点上;顺带钉一个通过率地板,M4 那种"答案对、通过率塌"的坏法
    才有人看得见。
    """
    if not pathlib.Path(path).is_file():
        pytest.skip(f"bank not present: {path}")
    vb = _mdp("virtual_ball")
    meta, _clip, rows = _bank(path)
    surf, net_x, net_top = _planes(meta, prm)
    n = 384
    clips = torch.zeros(n, dtype=torch.long)
    ref = rows["clip_normal"].unsqueeze(0).expand(n, 3).contiguous()
    aim_x, aim_y = (float(v) for v in meta["landing_env"])
    contact = rows["contact"]
    cfg = cq.ContinuousQuestionCfg(
        # 用题库自己的几何:接触点框收成它那一点,瞄点收成它那一点,来球箱覆盖它的来球分布。
        pos_range=tuple((float(contact[i]) - 0.02, float(contact[i]) + 0.02) for i in range(3)),
        aim_x_range=(aim_x, aim_x), aim_y_range=(aim_y, aim_y),
        vel_range=tuple((float(rows["v_in"][:, i].min()), float(rows["v_in"][:, i].max()))
                        for i in range(3)),
        spin_abs_max=0.0, speed_budget=float(meta["speed_budget"]),
        tol_m=PLANNER_TOL_M, n_iters=12, max_redraw_rounds=1,
    )
    gen = torch.Generator(device="cpu").manual_seed(20260727)
    out = cq.generate(clips, prm, surface_z=surf, net_x=net_x, cfg=cfg, ref_normal=ref,
                      net_top_z=net_top, generator=gen, h=0.01, n_steps=100)
    ok = out.ok
    k = int(ok.sum())

    # M4 arm: 通过率地板。答案本身仍然合格,唯一的症状就是通过率塌,所以只有这条看得见。
    assert k / n >= 0.80, (
        f"generate() accept rate {k}/{n} on the bank's own geometry. A silently wrong rollout "
        f"discretisation (h / n_steps) shows up ONLY here — the admitted answers still land fine."
    )
    # M1 arm: generate() 自己那一次调用必须把符号对齐传下去。
    dot = (out.n_racket[ok] * ref[ok]).sum(-1)
    assert float(dot.min()) > 0.0, (
        f"generate() returned {int((dot <= 0).sum())}/{k} faces pointing AWAY from the clip face — "
        f"its own call into solve_strike_specs lost ref_normal (单翻病 by construction)."
    )
    # M3 arm: 闭环。用调用方给的同一套平面把答案打回去,必须落在瞄点上。
    v_plus, w_plus = vb.predict_paddle_contact(
        out.v_ball_in[ok], out.v_racket[ok], out.n_racket[ok], out.w_ball_in[ok], prm)
    land = vb.coarse_landing(out.p_contact[ok], v_plus, w_plus, prm,
                             surface_z=surf, net_x=net_x, h=0.01, n_steps=100)
    assert bool(land["land_valid"].all()), "a solved answer that never lands is not solved"
    err = torch.linalg.norm(land["land_xy"] - out.aim_xy[ok], dim=-1)
    assert float(err.max()) <= 0.05, (
        f"generate()'s own answers close the loop {float(err.max()) * 1e3:.1f} mm from their aim "
        f"(p50 {float(err.median()) * 1e3:.1f} mm) — generate() is solving against different "
        f"planes than the ones its caller handed it."
    )


def test_reject_reasons_are_not_all_one_code(cq, prm):
    """The reason histogram used to hard-code 'resid_gt_tol' for every free-solve failure, so a
    no-landing and a speed-cap rejection were indistinguishable and every printed histogram was
    fiction. A speed budget of ~0 must now surface as speed_over_cap, not as resid_gt_tol."""
    n = 64
    clips = torch.zeros(n, dtype=torch.long)
    cfg = cq.ContinuousQuestionCfg(n_iters=4, max_redraw_rounds=1, speed_budget=0.02)
    out = cq.generate(clips, prm, surface_z=0.78, net_x=1.87, cfg=cfg)
    assert out.reason_counts.get("speed_over_cap", 0) > 0, out.reason_counts
