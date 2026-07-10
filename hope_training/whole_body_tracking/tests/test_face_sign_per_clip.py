"""face-sign-per-clip-0709 — 击球面符号机制 unit tests (CPU, isaaclab STUBBED / numpy only).

人话:统一正反手策略的两个挥拍用拍子**相反的两面**击球(正手=红面/+Y,反手=黑面/−Y),但训练
奖励、训练内拍面误差指标和 MuJoCo 判卷都从 mount 约定的单一符号出发,把反手按"不击球的那一面"
记分——反手拍面目标 ~180° 不可达,误差钉在 115-137°,综合成功清零(M3b 判死;CF 换拍面=1.000)。
franco 拍板语义:"哪面拍子超前就是哪面"——符号是**离线固定的每 clip 常量**(参考 clip 触球帧算
出、登记进配置),运行时只读常量表,绝不用策略当前拍速动态定符号。

Reuses the isaaclab stub + real-module loader from test_reward_flags_mdp — everything exercised is
the REAL shipped hope_commands.py / hope_rewards.py / suggest_face_sign.py / mujoco_eval_onnx.py.
Covers:

* cfg 默认 mount_normal_sign_per_clip=() -> _compute_racket_state 走标量符号,连 _motion() 都不碰
  (现役行为逐位不变的最强证据)。
* 开表(multiseg):racket_normal_w 按每 env 的 clip_id 乘 ±1;单 clip 表长 1 也生效。
* fail-loud:表长和加载 clip 数不匹配当场 ValueError(照 _strike_phases_cfg 先例);±1 以外的
  符号值同样报错。
* 奖励路径:真 hope_rewards.racket_normal_tracking_exp 消费翻面后的 racket_normal_w(反手面对齐
  时奖励 1.0,不开表时同一姿态奖励 ~0)。
* 指标路径:racket_normal_error_deg 的原式(acos(n·n_target))读同一个缓冲,翻面后 180°->0°;
  另有源码守卫断言 _update_metrics 确实从 racket_normal_w 计算该误差(防止未来把指标改读别处,
  让"一处修两处好"悄悄失效)。
* 参考法向路径:_ensure_reference_strike_state 的每 clip 参考拍面法向(诊断表 + 参考锁定拍面
  目标的来源)同样吃到符号。
* 离线建议工具 suggest_face_sign:合成 npz(已知拍面朝向 + 已知拍速方向)上 sign(n·v) 逐位命中;
  贴边(法向⊥拍速)与"干净/存储速度符号打架"都升 exit code 2;未登记触球帧当场 FATAL;
  运行时语义守卫——工具只在离线路径存在,hope_commands 运行时不含任何 n·v 动态符号。
* 判卷器 mujoco_eval_onnx:face_sign_for_clip 表没开 = 标量(逐位不变);开表按 clip 取;
  racket_normal_w(sign=None) 默认参数 = 现役行为;CLI 旗标默认 None。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_face_sign_per_clip.py -q
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import sys
import tempfile
import types

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SCRIPTS_DIR = os.path.abspath(os.path.join(HERE, "..", "scripts"))

from test_reward_flags_mdp import (  # noqa: E402  (installs the isaaclab stub, loads REAL modules)
    _fake_env,
    _fake_racket_cmd,
    hope_commands_mod,
    hope_observations_mod,
    hope_rewards_mod,
)

EY = torch.tensor([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------------------------- #
# harness: a RacketTargetCommand with exactly the state _compute_racket_state touches (body mode)
# --------------------------------------------------------------------------------------------- #
def _make_state_cmd(n, signs, clip_ids=None, multiseg=True, num_segments=2):
    RT = hope_commands_mod.RacketTargetCommand
    rt = RT.__new__(RT)
    rt.device = "cpu"
    rt.num_envs = n
    rt.cfg = types.SimpleNamespace(
        mount_normal_axis=1, mount_normal_sign=1.0, mount_normal_sign_per_clip=signs)
    rt._racket_mode = "body"
    rt._racket_body_index = 0
    quat = torch.zeros(n, 1, 4)
    quat[..., 0] = 1.0  # identity — the stub matrix_from_quat returns eye(3), so axis 1 == +Y
    rt.robot = types.SimpleNamespace(data=types.SimpleNamespace(
        body_pos_w=torch.zeros(n, 1, 3), body_quat_w=quat,
        body_lin_vel_w=torch.zeros(n, 1, 3), body_ang_vel_w=torch.zeros(n, 1, 3),
        body_link_lin_vel_w=torch.zeros(n, 1, 3),
        body_link_ang_vel_w=torch.zeros(n, 1, 3)))
    fake_motion = types.SimpleNamespace(
        _multiseg=multiseg,
        clip_id=torch.tensor(clip_ids if clip_ids is not None else [0] * n),
        motion=types.SimpleNamespace(num_segments=num_segments))
    rt._motion_term = None
    rt._motion = lambda: fake_motion
    rt._mount_sign_per_clip_t = None
    return rt


def test_racket_velocity_reads_link_point_not_legacy_com_velocity():
    """IsaacLab 2.1 body_lin_vel_w 是 COM 点速度；body_link_lin_vel_w 才和 body_pos_w 同点。"""
    rt = _make_state_cmd(2, signs=())
    rt.robot.data.body_lin_vel_w[:] = torch.tensor([90.0, 91.0, 92.0])  # deliberately wrong point
    rt.robot.data.body_link_lin_vel_w = torch.tensor(
        [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]
    )
    rt.robot.data.body_link_ang_vel_w = torch.zeros(2, 1, 3)
    rt._compute_racket_state()
    assert torch.equal(rt.racket_lin_vel_w, rt.robot.data.body_link_lin_vel_w[:, 0])


def test_wrist_fallback_adds_omega_cross_r_to_link_origin_velocity():
    rt = _make_state_cmd(1, signs=())
    rt._racket_mode = "wrist_offset"
    rt._wrist_body_index = 0
    rt._mount_offset = torch.tensor([[0.2, 0.0, 0.0]])
    rt._mount_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rt.robot.data.body_lin_vel_w[:] = torch.tensor([90.0, 91.0, 92.0])
    rt.robot.data.body_link_lin_vel_w = torch.tensor([[[1.0, 2.0, 3.0]]])
    rt.robot.data.body_link_ang_vel_w = torch.tensor([[[0.0, 0.0, 10.0]]])
    rt._compute_racket_state()
    assert torch.allclose(rt.racket_pos_w, torch.tensor([[0.2, 0.0, 0.0]]))
    assert torch.allclose(rt.racket_lin_vel_w, torch.tensor([[1.0, 4.0, 3.0]]))


def test_default_empty_table_is_scalar_and_never_touches_motion():
    """空表(默认)= 标量符号,且 _compute_racket_state 全程不碰 _motion() —— 现役行为逐位不变。"""
    rt = _make_state_cmd(3, signs=())
    rt._motion = lambda: (_ for _ in ()).throw(AssertionError("默认路径不该碰 motion term"))
    rt._compute_racket_state()
    assert torch.allclose(rt.racket_normal_w, EY.expand(3, 3))
    # 标量符号本身也照常生效
    rt2 = _make_state_cmd(2, signs=())
    rt2.cfg.mount_normal_sign = -1.0
    rt2._compute_racket_state()
    assert torch.allclose(rt2.racket_normal_w, -EY.expand(2, 3))


def test_per_clip_sign_flips_by_clip_id():
    rt = _make_state_cmd(4, signs=(1.0, -1.0), clip_ids=[0, 1, 0, 1])
    rt._compute_racket_state()
    expect = torch.stack([EY, -EY, EY, -EY])
    assert torch.allclose(rt.racket_normal_w, expect)
    # 符号张量只建一次(懒构建),第二次调用复用
    t = rt._mount_sign_per_clip_t
    rt._compute_racket_state()
    assert rt._mount_sign_per_clip_t is t


def test_single_clip_table_applies_without_multiseg():
    rt = _make_state_cmd(2, signs=(-1.0,), multiseg=False, num_segments=1)
    rt._compute_racket_state()
    assert torch.allclose(rt.racket_normal_w, -EY.expand(2, 3))


def test_sign_table_length_mismatch_fails_loud():
    """人话:符号表和加载的 clip 数对不上必须当场报错(照 _strike_phases_cfg 先例),不悄悄回退
    标量——那样反手又会按错误的一面被判分还不吭声。"""
    rt = _make_state_cmd(2, signs=(1.0, -1.0, 1.0), clip_ids=[0, 1])  # 3 entries, 2 segments
    with pytest.raises(ValueError, match="mount_normal_sign_per_clip"):
        rt._compute_racket_state()
    with pytest.raises(ValueError, match="mount_normal_sign_per_clip"):
        rt._mount_signs_cfg(2)


def test_sign_values_must_be_plus_minus_one():
    rt = _make_state_cmd(2, signs=(1.0, 0.5), clip_ids=[0, 1])
    with pytest.raises(ValueError, match="must be"):
        rt._compute_racket_state()
    rt0 = _make_state_cmd(2, signs=(0.0, 1.0), clip_ids=[0, 1])
    with pytest.raises(ValueError, match="must be"):
        rt0._compute_racket_state()


# --------------------------------------------------------------------------------------------- #
# 奖励路径 + 指标路径:两者都读 racket_normal_w,一处修两处好
# --------------------------------------------------------------------------------------------- #
def test_reward_path_consumes_per_clip_sign():
    """真 racket_normal_tracking_exp:反手 env(clip 1)拍面物理上用 −Y 面对准目标法向时,
    开表 = 满奖励;不开表(现役单面)同一姿态 = 奖励 ~0(这就是反手学不动的钱景)。"""
    n = 2
    tgt = -EY.expand(n, 3).clone()   # 反手挥拍的可达拍面方向(实际击球面)
    win = torch.ones(n, dtype=torch.bool)

    def _reward(signs, clip_ids):
        rt = _make_state_cmd(n, signs=signs, clip_ids=clip_ids)
        rt._compute_racket_state()
        cmd = _fake_racket_cmd(n, window=win)
        cmd.racket_normal_w = rt.racket_normal_w
        cmd.racket_target_normal_w = tgt
        return hope_rewards_mod.racket_normal_tracking_exp(
            _fake_env(racket_target=cmd), "racket_target", std=0.262)

    r_on = _reward((1.0, -1.0), [1, 1])   # 开表:反手按 −Y 面记分 -> cos=+1 -> 满奖励
    assert torch.allclose(r_on, torch.ones(n))
    r_off = _reward((), [1, 1])           # 现役单面:同一姿态 cos=−1 -> exp(-(pi/0.262)^2) ~ 0
    assert float(r_off.max()) < 1e-30


def test_metric_formula_reads_the_same_buffer():
    """racket_normal_error_deg 的原式(acos(n·n_target),_update_metrics 同一行算法)读同一个
    racket_normal_w 缓冲:不开表 180°,开表 0°。附源码守卫:指标确实从该缓冲计算。"""
    tgt = -EY.expand(2, 3)

    def _err_deg(signs):
        rt = _make_state_cmd(2, signs=signs, clip_ids=[1, 1])
        rt._compute_racket_state()
        cos = torch.sum(rt.racket_normal_w * tgt, dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cos) * (180.0 / math.pi)

    assert torch.allclose(_err_deg(()), torch.full((2,), 180.0))
    assert torch.allclose(_err_deg((1.0, -1.0)), torch.zeros(2), atol=1e-4)
    src = inspect.getsource(hope_commands_mod.RacketTargetCommand._update_metrics)
    assert "face_tracking_pair(self)" in src, (
        "_update_metrics 不再通过共享 face pair 计算拍面误差 —— reward/metric 语义会分裂")


# --------------------------------------------------------------------------------------------- #
# 参考法向路径:_ensure_reference_strike_state 每 clip 的参考拍面法向也吃符号
# --------------------------------------------------------------------------------------------- #
def _make_ref_cmd(signs):
    RT = hope_commands_mod.RacketTargetCommand
    rt = RT.__new__(RT)
    rt.device = "cpu"
    rt.num_envs = 1
    rt.cfg = types.SimpleNamespace(
        mount_normal_axis=1, mount_normal_sign=1.0, mount_normal_sign_per_clip=signs,
        strike_phase=0.5, strike_phase_per_clip=(), clean_strike_vel_window=2,
        clean_reference_strike_velocity=True)
    rt._env = types.SimpleNamespace(step_dt=0.02)
    rt._racket_mode = "body"
    rt._racket_body_index = 1
    rt._clip_names = {0: "forehand", 1: "backhand"}
    rt._ref_strike_cached = False
    T, B = 20, 2
    quat = torch.zeros(T, B, 4)
    quat[..., 0] = 1.0
    ml = types.SimpleNamespace(
        num_segments=2, seg_start=torch.tensor([0, 10]), seg_len=torch.tensor([10, 10]),
        time_step_total=T,
        _body_pos_w=torch.zeros(T, B, 3), _body_quat_w=quat,
        _body_lin_vel_w=torch.zeros(T, B, 3), _body_ang_vel_w=torch.zeros(T, B, 3))
    rt._motion = lambda: types.SimpleNamespace(motion=ml)
    return rt


def test_reference_strike_normal_gets_per_clip_sign(capsys):
    rt = _make_ref_cmd((1.0, -1.0))
    rt._ensure_reference_strike_state()
    nrm = rt._ref_racket_normal_w_per_clip
    assert torch.allclose(nrm[0], EY, atol=1e-5)      # 正手:+Y 面
    assert torch.allclose(nrm[1], -EY, atol=1e-5)     # 反手:−Y 面(参考锁定拍面目标从这里出)
    rt_off = _make_ref_cmd(())
    rt_off._ensure_reference_strike_state()
    assert torch.allclose(rt_off._ref_racket_normal_w_per_clip[1], EY, atol=1e-5)  # 现役行为
    capsys.readouterr()  # 吞掉诊断打印


# --------------------------------------------------------------------------------------------- #
# 离线建议工具 suggest_face_sign(合成用例)
# --------------------------------------------------------------------------------------------- #
def _load_by_path(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sfs = _load_by_path("suggest_face_sign_under_test", os.path.join(SCRIPTS_DIR, "suggest_face_sign.py"))


def _write_clip(path, vel_y, T=30, n_bodies=32, wrist=31, fps=50, stored_vel_y=None,
                body_names=None):
    """合成 clip:腕姿态恒为单位四元数(拍面 = +Y),腕沿 y 轴匀速 vel_y 平移。
    stored_vel_y 缺省 = 真速度(叉验一致);传相反值可制造"干净/存储打架"。"""
    t = np.arange(T, dtype=np.float64) / fps
    pos = np.zeros((T, n_bodies, 3))
    pos[:, wrist, 1] = vel_y * t
    quat = np.zeros((T, n_bodies, 4))
    quat[..., 0] = 1.0
    lin = np.zeros((T, n_bodies, 3))
    lin[:, wrist, 1] = vel_y if stored_vel_y is None else stored_vel_y
    arrays = dict(body_pos_w=pos, body_quat_w=quat, body_lin_vel_w=lin,
                  body_ang_vel_w=np.zeros((T, n_bodies, 3)), fps=np.array([fps]),
                  joint_pos=np.zeros((T, 31)))
    if body_names is not None:
        arrays["body_names"] = np.array(body_names)
    np.savez(path, **arrays)
    return path


def _write_ann(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("clips:\n")
        for stem, phase in entries.items():
            f.write(f"  {stem}:\n    phase: {phase}\n")
    return path


def test_suggest_sign_forehand_plus_backhand_minus(tmp_path):
    fh = _write_clip(str(tmp_path / "hope_forehand_syn.npz"), vel_y=+1.5)
    bh = _write_clip(str(tmp_path / "hope_backhand_syn.npz"), vel_y=-1.5)
    r_fh = sfs.compute_face_sign(fh, 15)
    r_bh = sfs.compute_face_sign(bh, 15)
    assert r_fh["suggested_sign"] == +1.0 and not r_fh["ambiguous"] and r_fh["raw_agrees"]
    assert r_bh["suggested_sign"] == -1.0 and not r_bh["ambiguous"] and r_bh["raw_agrees"]
    # +Y 面 vs ±y 拍速:cos 应为 ±1(合成场景无噪声)
    assert r_fh["cos_clean"] == pytest.approx(1.0, abs=1e-9)
    assert r_bh["cos_clean"] == pytest.approx(-1.0, abs=1e-9)


def test_suggest_sign_uses_embedded_body_names(tmp_path):
    names = ["pelvis_link", "some_link", "right_wrist_yaw_Link"]
    clip = _write_clip(str(tmp_path / "hope_backhand_named.npz"), vel_y=-2.0,
                       n_bodies=3, wrist=2, body_names=names)
    r = sfs.compute_face_sign(clip, 10)
    assert r["wrist_index"] == 2
    assert r["suggested_sign"] == -1.0


def test_suggest_sign_non32_bodies_without_order_fails_loud(tmp_path):
    clip = _write_clip(str(tmp_path / "hope_backhand_odd.npz"), vel_y=-2.0, n_bodies=5, wrist=4)
    with pytest.raises(ValueError, match="body order"):
        sfs.compute_face_sign(clip, 10)


def test_suggest_cli_table_and_exit_codes(tmp_path, capsys):
    fh = _write_clip(str(tmp_path / "hope_forehand_syn.npz"), vel_y=+1.5)
    bh = _write_clip(str(tmp_path / "hope_backhand_syn.npz"), vel_y=-1.5)
    ann = _write_ann(str(tmp_path / "ann.yaml"),
                     {"hope_forehand_syn": 0.5, "hope_backhand_syn": 0.5})
    out_json = str(tmp_path / "signs.json")
    code = sfs.main([fh, bh, "--annotations", ann, "--json", out_json])
    out = capsys.readouterr().out
    assert code == 0
    assert "mount_normal_sign_per_clip=[+1,-1]" in out
    import json
    rows = json.load(open(out_json))
    assert [r["suggested_sign"] for r in rows] == [1.0, -1.0]
    assert [r["expected_sign"] for r in rows] == [1.0, -1.0]  # 命名先验对照列(不参与计算)

    # 贴边:拍速 ⊥ 拍面(沿 x 平移,法向 +Y)-> exit 2,如实报
    amb = _write_clip(str(tmp_path / "hope_forehand_amb.npz"), vel_y=0.0)
    # 沿 x 给速度
    d = dict(np.load(amb))
    d["body_pos_w"][:, 31, 0] = np.arange(30) / 50.0 * 1.5
    d["body_lin_vel_w"][:, 31, 0] = 1.5
    np.savez(amb, **d)
    ann2 = _write_ann(str(tmp_path / "ann2.yaml"), {"hope_forehand_amb": 0.5})
    assert sfs.main([amb, "--annotations", ann2]) == 2
    assert "贴边" in capsys.readouterr().out

    # 干净/存储速度符号打架 -> exit 2
    fight = _write_clip(str(tmp_path / "hope_backhand_fight.npz"), vel_y=-1.5, stored_vel_y=+1.5)
    ann3 = _write_ann(str(tmp_path / "ann3.yaml"), {"hope_backhand_fight": 0.5})
    assert sfs.main([fight, "--annotations", ann3]) == 2
    assert "打架" in capsys.readouterr().out


def test_suggest_cli_unregistered_clip_is_fatal(tmp_path):
    clip = _write_clip(str(tmp_path / "hope_forehand_new.npz"), vel_y=1.0)
    ann = _write_ann(str(tmp_path / "ann.yaml"), {"some_other_clip": 0.5})
    with pytest.raises(SystemExit, match="annotation"):
        sfs.main([clip, "--annotations", ann])


def test_runtime_path_has_no_dynamic_sign():
    """franco 语义修正守卫:运行时(hope_commands)绝不能用当前拍速方向动态定符号——训练早期拍面
    可能整个反着,动态符号会把'反面'合法化。符号只能来自 cfg 常量表;n·v 只准出现在离线工具里。"""
    src = inspect.getsource(hope_commands_mod.RacketTargetCommand._compute_racket_state)
    assert "racket_lin_vel_w" not in src.split("axis_w")[-1], (
        "_compute_racket_state 的符号选择读到了拍速 —— 违反'离线固定常量'拍板语义")
    # 符号张量只从 cfg 表构建
    assert "mount_normal_sign_per_clip" in src and "_mount_signs_cfg" in src


# --------------------------------------------------------------------------------------------- #
# 判卷器 mujoco_eval_onnx —— 旗标默认关,开表按 clip 取
# --------------------------------------------------------------------------------------------- #
M = _load_by_path("mj_eval_face_sign_under_test", os.path.join(SCRIPTS_DIR, "mujoco_eval_onnx.py"))


def test_eval_face_sign_default_off_is_scalar():
    assert M.MOUNT_NORMAL_SIGN_PER_CLIP is None
    assert M.face_sign_for_clip(0) == M.MOUNT_NORMAL_SIGN
    assert M.face_sign_for_clip(1) == M.MOUNT_NORMAL_SIGN


def test_eval_face_sign_table_and_default_param(monkeypatch):
    monkeypatch.setattr(M, "MOUNT_NORMAL_SIGN_PER_CLIP", (1.0, -1.0))
    assert M.face_sign_for_clip(0) == 1.0
    assert M.face_sign_for_clip(1) == -1.0
    # racket_normal_w(sign=None) 默认参数 = 标量符号(现役判卷行为逐位不变)
    sig = inspect.signature(M.MujocoRobot.racket_normal_w)
    assert sig.parameters["sign"].default is None
    # CLI 旗标默认 None(不开表);记分点确实按 clip 翻面
    src = open(os.path.join(SCRIPTS_DIR, "mujoco_eval_onnx.py"), encoding="utf-8").read()
    assert '"--mount-normal-sign-per-clip", nargs="+", type=float, default=None' in src
    assert "robot.racket_normal_w(sign=face_sign_for_clip(clip))" in src


def test_train_py_plumbs_the_key():
    src = open(os.path.join(SCRIPTS_DIR, "train.py"), encoding="utf-8").read()
    assert '"mount_normal_sign_per_clip",' in src           # _RACKET_KEYS 白名单
    assert 'mount_normal_sign_per_clip=' in src              # override 落地 + applied 记录


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------------------------- #
# S1 face 约定修复(2026-07-09 单翻病定案):face_command 通道全程 +Y(A)约定
# 病根:符号表只翻了实测拍面,题库/obs/奖励还在 +Y 旧约定 —— 反手奖励最优点在错误平面
# (M3c/M2f 反手 ~34° 系统偏差)。修复 = face 通道的实测侧读 racket_normal_raw_w(_face_pair)。
# --------------------------------------------------------------------------------------------- #
def _fc_cmd(n, signs, clip_ids, target_cmd, window=None, pre_strike=None):
    """face_command=True 的 fake cmd:实测法向来自真 _compute_racket_state(identity quat →
    raw = +Y,signed = ±Y 按 clip),目标 = A 约定题目法向。"""
    rt = _make_state_cmd(n, signs=signs, clip_ids=clip_ids)
    rt._compute_racket_state()
    cmd = _fake_racket_cmd(
        n,
        window=torch.ones(n, dtype=torch.bool) if window is None else window,
        pre_strike=pre_strike,
    )
    cmd.cfg.face_command = True
    cmd.racket_normal_w = rt.racket_normal_w
    cmd.racket_normal_raw_w = rt.racket_normal_raw_w
    cmd.target_normal_cmd = target_cmd
    return cmd


def test_face_pair_selects_raw_under_face_command():
    """_face_pair 是 face 通道约定的单一来源:face_command=True → (raw, 题目法向);
    False → (signed, clip 参考面)= 现状字节等价路径。"""
    cmd = _fc_cmd(2, signs=(1.0, -1.0), clip_ids=[0, 1], target_cmd=EY.expand(2, 3).clone())
    m, t = hope_rewards_mod._face_pair(cmd)
    assert m is cmd.racket_normal_raw_w and t is cmd.target_normal_cmd
    cmd.cfg.face_command = False
    m2, t2 = hope_rewards_mod._face_pair(cmd)
    assert m2 is cmd.racket_normal_w and t2 is cmd.racket_target_normal_w


def test_face_command_backhand_regression_kernel_alive():
    """病灶回归陷阱:反手(sign=−1)完美执行 A 约定题目(raw +Y ≡ demanded)时 face 奖励必须 =1。
    修前(实测侧误用翻面 racket_normal_w)同一状态 cos=−1 → kernel <1e-30 = M3c/M2f 病根本尊;
    有人把实测侧改回 signed,这个断言当场红。"""
    n = 2
    cmd = _fc_cmd(n, signs=(1.0, -1.0), clip_ids=[1, 1], target_cmd=EY.expand(n, 3).clone())
    env = _fake_env(racket_target=cmd)
    r = hope_rewards_mod.racket_normal_tracking_exp(env, "racket_target", std=0.262)
    assert torch.allclose(r, torch.ones(n)), r
    # 修前语义的数值对照(独立复算,不走 _face_pair):signed vs 题目 = 反面
    old_cos = torch.sum(cmd.racket_normal_w * cmd.target_normal_cmd, dim=-1).clamp(-1.0, 1.0)
    old_kernel = torch.exp(-(torch.acos(old_cos) ** 2) / 0.262**2)
    assert float(old_kernel.max()) < 1e-30


def test_face_reward_invariant_to_sign_table():
    """S1 不变量:face_command 奖励对符号表严格不变(开表 vs 关表逐位 torch.equal)——
    face 通道与符号表彻底解耦,符号表只属于 metric/参考通道。"""
    tgt = torch.tensor([[0.3, 0.9, 0.1]]).expand(3, 3).clone()
    tgt = tgt / tgt.norm(dim=-1, keepdim=True)

    def _r(signs):
        cmd = _fc_cmd(3, signs=signs, clip_ids=[0, 1, 1], target_cmd=tgt.clone())
        return hope_rewards_mod.racket_normal_tracking_exp(
            _fake_env(racket_target=cmd), "racket_target", std=0.262)

    assert torch.equal(_r((1.0, -1.0)), _r(()))


def test_strike_success_and_face_guidance_share_face_pair():
    """strike_success 的法向因子与 racket_face_guidance 线性罚必须走同一 _face_pair。
    反手完美执行:法向因子=1(乘积=位置×速度)、线性罚=0;题目偏 45°:线性罚=45°(修前
    读 signed 会算成 135° 被 theta_max 截成 90° ——线性罚往反方向拉,比死区更糟)。"""
    n = 1
    tgt = EY.expand(n, 3).clone()
    cmd = _fc_cmd(n, signs=(1.0, -1.0), clip_ids=[1], target_cmd=tgt,
                  pre_strike=torch.ones(n, dtype=torch.bool))
    env = _fake_env(racket_target=cmd)
    rs = hope_rewards_mod.racket_strike_success(env, "racket_target", 0.2, 1.0, 0.262)
    assert torch.allclose(rs, torch.ones(n)), rs  # pos/vel 误差全零 → 乘积 = 法向因子 = 1
    g0 = hope_rewards_mod.racket_face_guidance(env, "racket_target")
    assert torch.allclose(g0, torch.zeros(n), atol=1e-6), g0
    tgt45 = torch.tensor([[math.sin(math.pi / 4), math.cos(math.pi / 4), 0.0]])
    cmd45 = _fc_cmd(n, signs=(1.0, -1.0), clip_ids=[1], target_cmd=tgt45,
                    pre_strike=torch.ones(n, dtype=torch.bool))
    g45 = hope_rewards_mod.racket_face_guidance(_fake_env(racket_target=cmd45), "racket_target")
    assert abs(float(g45[0]) - math.pi / 4) < 1e-5, g45
    # 修前语义对照:signed(−Y) vs 45° 题目 = 135° → 截到 theta_max=π/2,方向整个反了
    old_cos = torch.sum(cmd45.racket_normal_w * tgt45, dim=-1).clamp(-1.0, 1.0)
    assert abs(float(torch.acos(old_cos)[0]) - 3 * math.pi / 4) < 1e-5


def test_raw_buffer_bitwise_equal_when_table_empty():
    """字节等价卫兵:符号表空(现役默认)时 raw ≡ signed(torch.equal);标量 sign=−1 只动
    signed,raw 永远是未翻 +Y 轴。"""
    rt = _make_state_cmd(3, signs=())
    rt._compute_racket_state()
    assert torch.equal(rt.racket_normal_w, rt.racket_normal_raw_w)
    rt2 = _make_state_cmd(2, signs=())
    rt2.cfg.mount_normal_sign = -1.0
    rt2._compute_racket_state()
    assert torch.equal(rt2.racket_normal_raw_w, EY.expand(2, 3))
    assert torch.equal(rt2.racket_normal_w, -EY.expand(2, 3))


def test_reference_raw_normals_unsigned(capsys):
    """参考面的 raw 孪生缓冲(_ref_racket_normal_raw_w_per_clip)不乘符号:A 约定卫兵的比对基准。"""
    rt = _make_ref_cmd((1.0, -1.0))
    rt._ensure_reference_strike_state()
    raw = rt._ref_racket_normal_raw_w_per_clip
    assert torch.allclose(raw[0], EY, atol=1e-5)
    assert torch.allclose(raw[1], EY, atol=1e-5)   # 反手 raw 仍 +Y(signed 是 −Y)
    assert torch.allclose(rt._ref_racket_normal_w_per_clip[1], -EY, atol=1e-5)
    capsys.readouterr()


def test_bank_a_frame_guard_fails_loud_on_flipped_bank():
    """防复发卫兵:B 约定题库(需求法向背对 +Y 参考面)喂进来当场 ValueError——
    "题库按翻面重出"旧欠账已被 S1 决议关闭,谁完成它谁触雷。同侧题库正常通过。"""
    RT = hope_commands_mod.RacketTargetCommand
    rt = RT.__new__(RT)
    rt.cfg = types.SimpleNamespace(question_bank="/tmp/fake_bank.npz")
    rt._clip_names = {0: "forehand", 1: "backhand"}
    rt._ref_strike_cached = True  # 短路 _ensure_reference_strike_state(缓存已就位)
    rt._ref_racket_normal_raw_w_per_clip = torch.stack([EY, EY])
    good = torch.zeros(2, 4, 3)
    good[..., 1] = 1.0
    bad = good.clone()
    bad[1] = -bad[1]  # 反手行翻面 = B 约定题库
    counts = torch.tensor([4, 4])
    rt._question_bank = types.SimpleNamespace(demanded_normal=bad, counts=counts, metadata={})
    rt._qb_face_frame_checked = False
    with pytest.raises(ValueError, match="OPPOSITE"):
        rt._check_question_bank_face_frame()
    rt._question_bank = types.SimpleNamespace(demanded_normal=good, counts=counts, metadata={})
    rt._qb_face_frame_checked = False
    rt._check_question_bank_face_frame()
    assert rt._qb_face_frame_checked


def test_face_cmd_metric_reads_raw_frame():
    """新训练指标 face_cmd_normal_error_deg 在奖励自己的坐标系(raw vs 题目)量误差:
    完美执行 0°、翻面 180°;老指标 racket_normal_error_deg 是翻转不变量,看不见这个病。"""
    raw = EY.expand(2, 3)
    cmdn = torch.stack([EY, -EY])
    err = torch.acos(torch.sum(raw * cmdn, dim=-1).clamp(-1.0, 1.0)) * (180.0 / math.pi)
    assert abs(float(err[0])) < 1e-4 and abs(float(err[1]) - 180.0) < 1e-4
    src_pair = inspect.getsource(hope_commands_mod.face_tracking_pair)
    src_metrics = inspect.getsource(hope_commands_mod.RacketTargetCommand._update_metrics)
    assert "racket_normal_raw_w" in src_pair and "target_normal_cmd" in src_pair
    assert "face_tracking_pair(self)" in src_metrics, (
        "exact/composite 指标不再与拍面奖励共用 A-frame 配对")


def test_face_tracking_pair_is_shared_reward_metric_contract():
    cmd = types.SimpleNamespace(
        cfg=types.SimpleNamespace(face_command=True),
        racket_normal_raw_w=torch.stack([EY, EY]),
        target_normal_cmd=torch.stack([EY, -EY]),
        racket_normal_w=torch.stack([EY, -EY]),
        racket_target_normal_w=torch.stack([EY, -EY]),
    )
    measured, target = hope_commands_mod.face_tracking_pair(cmd)
    assert measured is cmd.racket_normal_raw_w and target is cmd.target_normal_cmd
    cmd.cfg.face_command = False
    measured, target = hope_commands_mod.face_tracking_pair(cmd)
    assert measured is cmd.racket_normal_w and target is cmd.racket_target_normal_w


def test_source_guards_face_frame_wiring():
    """源码守卫:防"单翻"复发的全部关键接线。任何一条红 = 有人把约定改散了,先读
    hope_rewards._face_pair 的 docstring 再动手。"""
    assert "_face_pair" in inspect.getsource(hope_rewards_mod._normal_kernel_raw)
    assert "_face_pair" in inspect.getsource(hope_rewards_mod.racket_face_guidance)
    src_p = inspect.getsource(hope_commands_mod.face_tracking_pair)
    assert "racket_normal_raw_w" in src_p and "target_normal_cmd" in src_p
    assert "face_tracking_pair(cmd)" in inspect.getsource(hope_rewards_mod._face_pair)


def test_critic_face_pair_sees_the_same_random_command_without_resizing():
    cmd = _fake_racket_cmd(2)
    cmd.cfg.face_command = True
    cmd.racket_normal_raw_w = torch.stack([EY, -EY])
    cmd.racket_normal_w = -cmd.racket_normal_raw_w  # deliberately different signed/reference view
    cmd.target_normal_cmd = torch.stack([-EY, EY])
    cmd.racket_target_normal_w = cmd.racket_normal_w.clone()
    env = _fake_env(racket_target=cmd)

    assert torch.equal(
        hope_observations_mod.racket_normal_w(env, "racket_target"),
        cmd.racket_normal_raw_w,
    )
    assert torch.equal(
        hope_observations_mod.racket_target_normal_w(env, "racket_target"),
        cmd.target_normal_cmd,
    )

    cmd.cfg.face_command = False
    assert torch.equal(
        hope_observations_mod.racket_normal_w(env, "racket_target"),
        cmd.racket_normal_w,
    )
    assert torch.equal(
        hope_observations_mod.racket_target_normal_w(env, "racket_target"),
        cmd.racket_target_normal_w,
    )
    import re
    mj = open(os.path.join(SCRIPTS_DIR, "mujoco_eval_onnx.py"), encoding="utf-8").read()
    # 守卫必须是 raise(不是降级成 print 的警告)——正则绑定 raise SystemExit 本体
    assert re.search(r'raise SystemExit\("\[FATAL\] --mount-normal-sign-per-clip is incompatible', mj)  # 守卫①
    assert re.search(r'raise SystemExit\("\[FATAL\] --mount-normal-sign-per-clip with a face-obs model', mj)  # 守卫②
    assert "bank scoring/obs are +Y(A)-frame" in mj
    assert "the face lane is +Y(A)-frame in training" in mj
    vb = open(os.path.join(SCRIPTS_DIR, "venue_ball_sampler.py"), encoding="utf-8").read()
    assert re.search(r'raise SystemExit\(\s*f"\[FATAL\] exam bank', vb)  # B 卷卫兵同样必须 raise
    assert "OPPOSITE the +Y reference face" in vb
    # 训练侧卫兵的"接线"(调用点)必须存在——方法本体在而调用点被删 = 卫兵静默脱钩
    src_apply = inspect.getsource(
        hope_commands_mod.RacketTargetCommand._apply_question_bank_targets)
    assert "_check_question_bank_face_frame" in src_apply, (
        "_apply_question_bank_targets 不再调用 A 约定卫兵 —— B 卷会静默进训练,单翻病复发")
    assert "self.vb_vel_in_w[env_ids] = incoming_vel" in src_apply
    assert "self.vb_spin_in_w[env_ids] = incoming_spin" in src_apply
    src_resample = inspect.getsource(hope_commands_mod.RacketTargetCommand._resample_command)
    assert "self._question_bank is None" in src_resample, (
        "bank 来球在同一 resample 末尾又被随机 virtual-ball 采样覆盖")
    ex_path = os.path.join(
        HERE, "..", "source", "whole_body_tracking", "whole_body_tracking", "utils", "exporter.py")
    ex = open(ex_path, encoding="utf-8").read()
    assert "mount_normal_sign_per_clip" in ex and "face_obs_convention" in ex  # 元数据保真
    js = open(os.path.join(SCRIPTS_DIR, "judge.sh"), encoding="utf-8").read()
    assert '("mount_normal_sign_per_clip", rt.get("mount_normal_sign_per_clip"))' in js  # 导出搬运
    tr = open(os.path.join(SCRIPTS_DIR, "train.py"), encoding="utf-8").read()
    assert "face_command kernel frame=+Y(A/bank)" in tr      # 冒烟 grep 的 applied 审计行
