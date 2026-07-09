"""station-anchor-obs-0709 — R10c 站位锚观测通道单元测试(NO Isaac imports)。

franco 2026-07-09 拍板:"planner 的 p_base 应该加进去:就算不需要移动,它也是一个锚;
不小心移动时世界坐标的 p_base 能提醒"。四块覆盖:

* 契约:deploy_parity_station181 注册且 = face179 布局 + 尾部 station_anchor_err_b(2)
  (纯尾部追加 = pad_obs_cols 热启手术的前提;站位在拍面后 = 与契约日蓝图的有意差异)。
* train.py 翻译层(racket.station_obs / station_anchor_offset_xy):默认关逐位不变、
  开启挂 ObsTerm 在尾部、无拍面通道单开 loud error(177 撞 Hitter 布局)、
  cfg 先挂站位再开拍面的错序 loud error、偏移覆盖落到 racket_target cfg。
* MuJoCo 评估器:RacketCommand.station_anchor_err_b 数值(手算 yaw 旋转钉死约定)、
  resample 不动锚、build_obs 181 装配 = 179 前缀逐位相同 + 尾部 2 维站位误差、
  station 无 face 直接 assert 炸。
* pad_obs_cols.py 手术往返:179->181 后首层前 179 列逐位不变/新列全零、第 0 步前向
  逐位等价、归一化器 mean补0/var补1/count 不动、优化器动量补零、critic 侧原样、
  iter 保留;错误旧维数 loud fail;二次手术(181 上再当 179 切)loud fail。

Run:  pytest hope_training/whole_body_tracking/tests/test_station_anchor_obs.py -q
  or: python3 hope_training/whole_body_tracking/tests/test_station_anchor_obs.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
CONTRACT_PATH = os.path.abspath(os.path.join(
    HERE, "..", "source", "whole_body_tracking", "whole_body_tracking",
    "tasks", "tracking", "actor_observation_contract.py"))
MJ_EVAL_PATH = os.path.join(SCRIPTS, "mujoco_eval_onnx.py")
PAD_TOOL = os.path.join(SCRIPTS, "pad_obs_cols.py")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import train as train_mod  # noqa: E402  (hydra/omegaconf only at import time)


def _load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses 解析字符串注解要能在 sys.modules 里找到模块
    spec.loader.exec_module(mod)
    return mod


# ============================================================================================= #
# 1) contract registration
# ============================================================================================= #
@pytest.fixture(scope="module")
def contract_mod():
    return _load_by_path(CONTRACT_PATH, "aoc_under_test")


def test_station181_contract_registered(contract_mod):
    c = contract_mod.resolve_actor_observation_contract("deploy_parity_station181")
    assert c is not None and c.total_dim == 181 and c.obs_mode == "deploy_parity"
    assert sum(d for _, d in c.layout) == 181


def test_station181_is_pure_tail_append_on_face179(contract_mod):
    """手术前提:181 = face179 前缀逐位相同 + 尾部 station(2)。站位排在拍面之后
    (与契约日蓝图 175+站位+拍面 的顺序不同,是为纯尾部扩列热启的有意取舍)。"""
    face179 = contract_mod.DEPLOY_PARITY_FACE179
    st181 = contract_mod.DEPLOY_PARITY_STATION181
    assert st181.terms[: len(face179.terms)] == face179.terms, "179 前缀被改动 — 热启手术失效"
    assert st181.layout[-1] == ("station_anchor_err_b", 2)
    # 站位必须在拍面(racket_target_normal_cmd)之后
    names = [n for n, _ in st181.layout]
    assert names.index("station_anchor_err_b") == names.index("racket_target_normal_cmd") + 1
    # face179 契约本身没被动(179 臂读数可比性)
    assert face179.total_dim == 179 and face179.layout[-1] == ("racket_target_normal_cmd", 4)


def test_unknown_contract_still_raises(contract_mod):
    with pytest.raises(ValueError, match="deploy_parity_station181"):
        # 报错信息必须把新契约列进 known values(可发现性)
        contract_mod.resolve_actor_observation_contract("no_such_contract")


# ============================================================================================= #
# 2) train.py override translation (racket.station_obs)
# ============================================================================================= #
class _Term:
    def __init__(self, weight=1.0, params=None, func="orig_func"):
        self.weight = weight
        self.params = dict(params) if params is not None else {}
        self.func = func


class _NS(types.SimpleNamespace):
    pass


def _make_env_cfg():
    """最小 DeployParity 形状的假 env cfg:只搭 station/face 覆盖路径摸到的面。"""
    racket_target = _NS(
        question_bank="", face_command=False,
        station_anchor_offset_xy=(0.0, 0.0),
        track_envelope_violation=False,
    )
    motion = _NS(speed_scale_range=(1.0, 1.0))
    observations = _NS(policy=_NS(), critic=_NS())
    return _NS(
        rewards=_NS(),
        commands=_NS(motion=motion, racket_target=racket_target),
        observations=observations,
        terminations=_NS(),
        face_command_obs=False,
        station_obs=False,
        scene=_NS(env_spacing=2.5),
    )


class _FakeObsTerm:
    def __init__(self, func=None, params=None):
        self.func = func
        self.params = dict(params or {})


def _stub_modules(monkeypatch):
    """station/face 覆盖分支里的惰性 import:isaaclab.managers.ObservationTermCfg 和
    whole_body_tracking...mdp(station_anchor_err_b / racket_target_normal_cmd)。"""
    fake_managers = types.ModuleType("isaaclab.managers")
    fake_managers.ObservationTermCfg = _FakeObsTerm
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab.managers = fake_managers

    fake_mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    fake_mdp.station_anchor_err_b = "STATION_ANCHOR_FUNC"
    fake_mdp.racket_target_normal_cmd = "FACE_CMD_FUNC"
    fake_mdp.generated_commands_actor_leg_masked = "LEG_MASKED_FUNC"
    fake_tracking = types.ModuleType("whole_body_tracking.tasks.tracking")
    fake_tracking.mdp = fake_mdp
    fake_tasks = types.ModuleType("whole_body_tracking.tasks")
    fake_tasks.tracking = fake_tracking
    fake_root = types.ModuleType("whole_body_tracking")
    fake_root.tasks = fake_tasks
    for name, m in (
        ("isaaclab", fake_isaaclab),
        ("isaaclab.managers", fake_managers),
        ("whole_body_tracking", fake_root),
        ("whole_body_tracking.tasks", fake_tasks),
        ("whole_body_tracking.tasks.tracking", fake_tracking),
        ("whole_body_tracking.tasks.tracking.mdp", fake_mdp),
    ):
        monkeypatch.setitem(sys.modules, name, m)
    return fake_mdp


def _apply(task, env_cfg=None):
    env_cfg = env_cfg if env_cfg is not None else _make_env_cfg()
    applied = train_mod._apply_task_overrides(env_cfg, task, clip_name=None)
    return env_cfg, applied


def test_whitelist_has_station_keys():
    assert "station_obs" in train_mod._RACKET_KEYS
    assert "station_anchor_offset_xy" in train_mod._RACKET_KEYS


def test_station_obs_attaches_tail_term_with_face(monkeypatch):
    fake_mdp = _stub_modules(monkeypatch)
    env_cfg, applied = _apply(
        {"racket": {"face_command_obs": True, "station_obs": True}})
    pol = env_cfg.observations.policy
    assert getattr(pol, "racket_target_normal_cmd").func == fake_mdp.racket_target_normal_cmd
    st = getattr(pol, "station_anchor_err_b")
    assert st.func == fake_mdp.station_anchor_err_b
    assert st.params == {"command_name": "racket_target"}
    # critic 组不动(surgery 前提:critic 归一化器/首层原样)
    assert getattr(env_cfg.observations.critic, "station_anchor_err_b", None) is None
    # 描述性 cfg 字段同步 + 发射日志行
    assert env_cfg.station_obs is True and env_cfg.face_command_obs is True
    assert any("179->181" in a for a in applied)


def test_station_obs_without_face_fails_loud(monkeypatch):
    _stub_modules(monkeypatch)
    with pytest.raises(train_mod._OverrideError, match="177"):
        _apply({"racket": {"station_obs": True}})


def test_station_obs_with_cfg_preattached_face_ok(monkeypatch):
    """face 走 cfg 旗标(__post_init__ 已挂)、station 走 YAML 覆盖的合法组合。"""
    fake_mdp = _stub_modules(monkeypatch)
    env_cfg = _make_env_cfg()
    env_cfg.observations.policy.racket_target_normal_cmd = _FakeObsTerm(func="FACE_CMD_FUNC")
    env_cfg.face_command_obs = True
    env_cfg, applied = _apply({"racket": {"station_obs": True}}, env_cfg)
    assert env_cfg.observations.policy.station_anchor_err_b.func == fake_mdp.station_anchor_err_b
    assert any("station_anchor_err_b" in a for a in applied)


def test_face_after_station_order_guard(monkeypatch):
    """错序守卫:站位已在(cfg 路径),再用 YAML 开拍面 → 布局会变成 175+站位+拍面,必须炸。"""
    _stub_modules(monkeypatch)
    env_cfg = _make_env_cfg()
    env_cfg.observations.policy.station_anchor_err_b = _FakeObsTerm(func="STATION_ANCHOR_FUNC")
    with pytest.raises(train_mod._OverrideError, match="station_anchor_err_b"):
        _apply({"racket": {"face_command_obs": True}}, env_cfg)


def test_station_obs_default_off_is_noop(monkeypatch):
    _stub_modules(monkeypatch)
    for task in ({}, {"racket": {}}, {"racket": {"station_obs": False}}):
        env_cfg, applied = _apply(task)
        assert getattr(env_cfg.observations.policy, "station_anchor_err_b", None) is None
        assert env_cfg.station_obs is False
        assert not any("station" in a for a in applied)


def test_station_anchor_offset_override(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg, applied = _apply(
        {"racket": {"face_command_obs": True, "station_obs": True,
                    "station_anchor_offset_xy": [0.1, -0.2]}})
    assert env_cfg.commands.racket_target.station_anchor_offset_xy == (0.1, -0.2)
    assert any("station_anchor_offset_xy" in a for a in applied)


# ============================================================================================= #
# 3) MuJoCo evaluator: station math + 181 obs assembly
# ============================================================================================= #
@pytest.fixture(scope="module")
def mj():
    return _load_by_path(MJ_EVAL_PATH, "mj_eval_station_under_test")


def _racket(mj, origin=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(0)
    normals = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])]
    return mj.RacketCommand(seg_start=np.array([0, 50]), seg_len=np.array([50, 50]),
                            step_dt=0.02, rng=rng, target_normal_per_clip=normals,
                            origin=np.asarray(origin, np.float64))


def test_station_anchor_is_origin_constant_and_resample_inert(mj):
    r = _racket(mj, origin=(1.0, 2.0, 0.0))
    assert np.allclose(r.station_anchor_pos_w, [1.0, 2.0])
    before = r.station_anchor_pos_w.copy()
    for clip in (0, 1, 0):
        r.resample(clip)
    # base_target 每摆重采,station 锚一动不动 —— 这才叫锚
    assert np.array_equal(r.station_anchor_pos_w, before)


def test_station_anchor_err_b_numeric(mj):
    r = _racket(mj, origin=(1.0, 2.0, 0.0))
    # 在锚上、零 yaw:误差 = 0
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(r.station_anchor_err_b(np.array([1.0, 2.0, 1.0]), q_id), [0.0, 0.0])
    # 漂移 (+0.3, -0.4)、零 yaw:误差 = 锚 − base = (-0.3, +0.4)(世界系直读)
    assert np.allclose(
        r.station_anchor_err_b(np.array([1.3, 1.6, 1.0]), q_id), [-0.3, 0.4])
    # yaw=+90°(w=cos45, z=sin45):世界 Δ=(-0.3, +0.4) → base 系 = (Δ·x̂_b, Δ·ŷ_b) = (0.4, 0.3)
    s2 = np.sqrt(0.5)
    q_yaw90 = np.array([s2, 0.0, 0.0, s2])
    assert np.allclose(
        r.station_anchor_err_b(np.array([1.3, 1.6, 1.0]), q_yaw90), [0.4, 0.3], atol=1e-12)
    # 与同数学的 base_target_pos_b 交叉验证:把 base_target 摆到锚上,两者逐位相等
    r.base_target_pos_w = r.station_anchor_pos_w.copy()
    assert np.allclose(
        r.base_target_pos_b(np.array([1.3, 1.6, 1.0]), q_yaw90),
        r.station_anchor_err_b(np.array([1.3, 1.6, 1.0]), q_yaw90))
    # roll/pitch 不进 yaw-heading 旋转:绕 X 的 90° 分量不改变结果(yaw_quat 提取)
    q_rollyaw = np.array([0.5, 0.5, 0.5, 0.5])  # yaw 90° + 其它分量
    assert np.allclose(
        r.station_anchor_err_b(np.array([1.3, 1.6, 1.0]), q_rollyaw), [0.4, 0.3], atol=1e-12)


class _FakeRobot:
    """build_obs 摸到的最小机器人面:pelvis/torso 位姿 + 关节/IMU 读数。"""

    pelvis_bid, torso_bid = 0, 1

    def __init__(self, base_pos, base_quat):
        self._pos = {0: np.asarray(base_pos, np.float64),
                     1: np.asarray(base_pos, np.float64) + np.array([0.0, 0.0, 0.3])}
        self._quat = {0: np.asarray(base_quat, np.float64), 1: np.asarray(base_quat, np.float64)}

    def body_pos(self, bid):
        return self._pos[bid]

    def body_quat(self, bid):
        return self._quat[bid]

    def pelvis_ang_vel_body(self):
        return np.array([0.01, -0.02, 0.03])

    def q_artic(self):
        return np.linspace(-0.5, 0.5, 31)

    def qd_artic(self):
        return np.linspace(-1.0, 1.0, 31)

    def projected_gravity_body(self):
        return np.array([0.0, 0.0, -1.0])


def _fake_refs(mj):
    nb = len(mj.TRACKED_BODIES)
    quat = np.zeros((nb, 4)); quat[:, 0] = 1.0
    return {
        "joint_pos": np.linspace(0.1, 0.9, 31),
        "joint_vel": np.linspace(-0.2, 0.2, 31),
        "body_pos_w": np.tile(np.array([0.0, 0.0, 1.0]), (nb, 1)),
        "body_quat_w": quat,
    }


def test_build_obs_181_is_179_prefix_plus_station_tail(mj, monkeypatch):
    monkeypatch.setattr(mj, "racket_pos_pelvis", lambda q: np.array([0.35, -0.25, 0.15]))
    r = _racket(mj, origin=(0.0, 0.0, 0.0))
    r.resample(0)
    base_pos = np.array([0.3, -0.4, 1.05])
    s2 = np.sqrt(0.5)
    base_quat = np.array([s2, 0.0, 0.0, s2])
    robot = _FakeRobot(base_pos, base_quat)
    refs = _fake_refs(mj)
    last_action = np.linspace(-0.3, 0.3, 31)

    obs179, *_ = mj.build_obs(refs, robot, r, last_action, np.zeros(31),
                              deploy_parity=True, face_command=True)
    obs181, *_ = mj.build_obs(refs, robot, r, last_action, np.zeros(31),
                              deploy_parity=True, face_command=True, station=True)
    assert obs179.shape == (179,) and obs181.shape == (181,)
    # 179 前缀逐位相同(纯尾部追加 = 热启手术前提在评估器侧同样成立)
    assert np.array_equal(obs181[:179], obs179)
    # 尾部 2 维 = station_anchor_err_b
    assert np.allclose(obs181[179:], r.station_anchor_err_b(base_pos, base_quat))
    # 数值:锚(0,0) − base(0.3,-0.4) = (-0.3, 0.4),yaw 90° → base 系 (0.4, 0.3)
    assert np.allclose(obs181[179:], [0.4, 0.3], atol=1e-12)


def test_build_obs_station_without_face_fails(mj, monkeypatch):
    monkeypatch.setattr(mj, "racket_pos_pelvis", lambda q: np.zeros(3))
    r = _racket(mj)
    r.resample(0)
    robot = _FakeRobot([0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(AssertionError, match="181 = 179 \\+ 2"):
        mj.build_obs(_fake_refs(mj), robot, r, np.zeros(31), np.zeros(31),
                     deploy_parity=True, face_command=False, station=True)


# ============================================================================================= #
# 4) pad_obs_cols.py 179->181 surgery round-trip
# ============================================================================================= #
torch = pytest.importorskip("torch")

_H = 64  # actor 首层宽度(测试用小网络)


def _synth_ckpt(path, obs_dim=179, critic_dim=300):
    g = torch.Generator().manual_seed(7)
    ckpt = {
        "model_state_dict": {
            "actor.0.weight": torch.randn(_H, obs_dim, generator=g),
            "actor.0.bias": torch.randn(_H, generator=g),
            "actor.2.weight": torch.randn(31, _H, generator=g),
            "critic.0.weight": torch.randn(_H, critic_dim, generator=g),
            "std": torch.rand(31, generator=g),
        },
        "optimizer_state_dict": {
            "state": {
                0: {"step": torch.tensor(13000.0),
                    "exp_avg": torch.randn(_H, obs_dim, generator=g),
                    "exp_avg_sq": torch.rand(_H, obs_dim, generator=g)},
                1: {"step": torch.tensor(13000.0),
                    "exp_avg": torch.randn(_H, generator=g),
                    "exp_avg_sq": torch.rand(_H, generator=g)},
            },
            "param_groups": [{"lr": 1e-3, "params": [0, 1]}],
        },
        "obs_norm_state_dict": {
            "mean": torch.randn(obs_dim, generator=g),
            "var": torch.rand(obs_dim, generator=g) + 0.5,
            "count": torch.tensor(1.0e6),
        },
        "privileged_obs_norm_state_dict": {
            "mean": torch.randn(critic_dim, generator=g),
            "var": torch.rand(critic_dim, generator=g) + 0.5,
            "count": torch.tensor(1.0e6),
        },
        "iter": 13000,
    }
    torch.save(ckpt, path)
    return ckpt


def _run_pad(src, dst, old, new):
    return subprocess.run([sys.executable, PAD_TOOL, src, dst, str(old), str(new)],
                          capture_output=True, text=True)


def test_pad_179_to_181_roundtrip(tmp_path):
    src, dst = str(tmp_path / "model_179.pt"), str(tmp_path / "out" / "model_181.pt")
    orig = _synth_ckpt(src)
    res = _run_pad(src, dst, 179, 181)
    assert res.returncode == 0, res.stderr
    out = torch.load(dst, map_location="cpu", weights_only=False)

    # actor 首层:前 179 列逐位不变,新 2 列全零
    w0, w1 = orig["model_state_dict"]["actor.0.weight"], out["model_state_dict"]["actor.0.weight"]
    assert w1.shape == (_H, 181)
    assert torch.equal(w1[:, :179], w0) and torch.all(w1[:, 179:] == 0)
    # 第 0 步前向等价:任意 179 观测 + 任意站位尾巴,输出与原模型一致(新列贡献恰为 0;
    # matmul 对 181 vs 179 长度的归约顺序可能不同,所以用紧 allclose 而非逐位 equal——
    # 数学贡献为零,浮点只剩重排误差)
    obs179 = torch.randn(5, 179)
    tail = torch.randn(5, 2) * 10.0
    assert torch.allclose(torch.cat([obs179, tail], dim=1) @ w1.T, obs179 @ w0.T,
                          atol=1e-5, rtol=1e-6)
    # 归一化器:mean 补 0 / var 补 1 / count 不动(尾巴直通,再被零权重吃掉)
    ons = out["obs_norm_state_dict"]
    assert torch.equal(ons["mean"][:179], orig["obs_norm_state_dict"]["mean"])
    assert torch.all(ons["mean"][179:] == 0.0) and torch.all(ons["var"][179:] == 1.0)
    assert torch.equal(ons["count"], orig["obs_norm_state_dict"]["count"])
    # 优化器动量同步扩列,新列 0(冷启动)
    st = out["optimizer_state_dict"]["state"][0]
    assert st["exp_avg"].shape == (_H, 181) and torch.all(st["exp_avg"][:, 179:] == 0)
    assert st["exp_avg_sq"].shape == (_H, 181) and torch.all(st["exp_avg_sq"][:, 179:] == 0)
    # 其它张量原样;critic/privileged 侧必须没被摸;iter 保留(热启不清零)
    assert torch.equal(out["model_state_dict"]["critic.0.weight"],
                       orig["model_state_dict"]["critic.0.weight"])
    assert torch.equal(out["privileged_obs_norm_state_dict"]["mean"],
                       orig["privileged_obs_norm_state_dict"]["mean"])
    assert torch.equal(out["model_state_dict"]["actor.2.weight"],
                       orig["model_state_dict"]["actor.2.weight"])
    assert out["iter"] == 13000


def test_pad_wrong_old_dim_fails_loud(tmp_path):
    src = str(tmp_path / "model_179.pt")
    _synth_ckpt(src)
    res = _run_pad(src, str(tmp_path / "bad.pt"), 175, 181)  # 拿 175 刀切 179 存档
    assert res.returncode != 0
    assert not os.path.exists(str(tmp_path / "bad.pt"))


def test_pad_double_surgery_fails_loud(tmp_path):
    """往返守卫:已是 181 的存档再按 179->181 切一刀必须炸(而不是再垫 2 列)。"""
    src, dst = str(tmp_path / "model_179.pt"), str(tmp_path / "model_181.pt")
    _synth_ckpt(src)
    assert _run_pad(src, dst, 179, 181).returncode == 0
    res = _run_pad(dst, str(tmp_path / "model_183.pt"), 179, 181)
    assert res.returncode != 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
