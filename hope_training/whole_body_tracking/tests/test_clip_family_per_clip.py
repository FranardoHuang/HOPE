"""clip-family-per-clip-20260722 — spdmix v2 硬绑定一 unit tests (CPU, isaaclab STUBBED).

人话:原来 hope_commands 里四处写死"clips == 0 才是正手"(swing_sign 三处 + uniform 目标 y 侧
一处),6-clip 变速烤入列表(正手 0.8/1.0/1.2 + 反手 0.8/1.0/1.1)里正手 1.0/1.2 变体会被当成
反手——obs 挥拍类型、目标侧全错,不崩但训错。修法:新配置 task.motion.clip_family_per_clip
(每 clip 一个 "forehand"/"backhand" 标签),四处判断全部换成家族查表。铁律:配置缺席时(现役
所有在跑臂)行为逐字节不变——单 clip 当正手、恰好 2 clip = (正手, 反手),和写死判断同值;
≥3 clip 缺表当场报错,不猜。

Reuses the isaaclab stub + real-module loader from test_reward_flags_mdp — everything exercised is
the REAL shipped commands.py / hope_commands.py. Covers:

* resolve_clip_family_is_forehand:缺席推导(1/2 clip 与现状全等、≥3 clip fail-loud)、显式表
  (顺序照抄、长度/取值/两族齐全校验 fail-loud)。
* MotionCommand 开机:显式表整表校验(错表 boot 炸)、合法表落成张量并打 ACTIVE、缺席不建表
  不打印(懒推导),3-clip 缺表能开机但查表当场炸。
* RacketTargetCommand._clip_family_is_forehand:缺席 2-clip 与旧 clips==0 逐字节同值;6-clip
  家族表下正手变速变体 swing_sign 与 clip0 相同;懒缓存;老测试 fake(无 cfg)不炸。
* 真 _sample_targets_uniform:相同随机种子下,缺席配置的新路径输出与旧公式逐字节 torch.equal
  (含 RNG 消耗次序);6-clip 家族表下正手变体落正手 y 侧。
* 源码守卫:四个函数不再含 "clips == 0"/"clip == 0" 硬编码,全部走 _clip_family_is_forehand。
* train.py 接线:_MOTION_KEYS 白名单、translation、合同块条件落键(缺席不写键 = 合同字节不变,
  老 checkpoint resume 对账不受影响)。
* training_contract 合同校验 roundtrip:合法表通过;类型错/缺一族/长度和 clip 数不一致 fail-loud;
  缺席键 = 结构校验照旧通过。
* face179 拍面符号(spdmix 硬绑定三拆除):家族表缺席 = legacy 判法逐字 [+1,-1] 不变(含报错
  文本逐字节一致);家族表在场 = 按族核对(正手全 +1、反手全 -1、长度==clip 数),六 clip 合法
  向量通过,族符号错/长度错/非数值 fail-loud;非 face179 合同不碰符号表(判定只挂 face179)。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_clip_family_per_clip.py -q
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
SCRIPTS_DIR = os.path.abspath(os.path.join(HERE, "..", "scripts"))

from test_reward_flags_mdp import (  # noqa: E402  (installs the isaaclab stub, loads REAL modules)
    _make_motion_command,
    _write_motion_npz,
    commands_mod,
    hope_commands_mod,
)
from test_training_contract_schema3 import TC, _schema3_contract  # noqa: E402

resolve = commands_mod.resolve_clip_family_is_forehand

FAM6 = ("forehand", "forehand", "forehand", "backhand", "backhand", "backhand")


# --------------------------------------------------------------------------------------------- #
# resolver: 缺席推导 = 现状全等;显式表 fail-loud 校验
# --------------------------------------------------------------------------------------------- #
def test_absent_single_clip_derives_forehand():
    """缺席 + 单 clip:当正手 —— 和 clips==0 -> +1 的写死行为同值。"""
    assert resolve(None, 1) == (True,)


def test_absent_two_clips_derives_forehand_backhand():
    """缺席 + 恰 2 clip:(正手, 反手) —— 现役所有多 clip 在跑臂的既有语义,逐字节同值。"""
    assert resolve(None, 2) == (True, False)


def test_absent_many_clips_fails_loud():
    """缺席 + ≥3 clip 当场报错:那正是变速正手会被悄悄当反手训错的场景,不猜。"""
    for nseg in (3, 6):
        with pytest.raises(ValueError, match="clip_family_per_clip"):
            resolve(None, nseg)


def test_explicit_six_clip_table_maps_families():
    assert resolve(FAM6, 6) == (True, True, True, False, False, False)


def test_explicit_order_is_honored_not_assumed():
    """表顺序照抄,不重排:反手在前就是反手在前(clip 0 不再天然是正手)。"""
    assert resolve(("backhand", "forehand"), 2) == (False, True)
    # 显式 (正手, 反手) 和缺席推导等价——同一张表,两条路一致
    assert resolve(("forehand", "backhand"), 2) == resolve(None, 2)


def test_explicit_wrong_length_fails_loud():
    with pytest.raises(ValueError, match="2 entries.*6 clip"):
        resolve(("forehand", "backhand"), 6)
    with pytest.raises(ValueError, match="0 entries"):
        resolve((), 2)  # 显式空表不是"缺席",同样 fail-loud


def test_explicit_unknown_value_fails_loud():
    with pytest.raises(ValueError, match="must be one of"):
        resolve(("forehand", "fore-hand"), 2)
    with pytest.raises(ValueError, match="must be one of"):
        resolve(("Forehand", "backhand"), 2)  # 大小写也不认,拒绝悄悄归一化


def test_explicit_missing_one_family_fails_loud():
    with pytest.raises(ValueError, match="at least one forehand and one backhand"):
        resolve(("forehand", "forehand"), 2)
    with pytest.raises(ValueError, match="at least one forehand and one backhand"):
        resolve(("backhand", "backhand", "backhand"), 3)


def test_single_clip_may_declare_either_family():
    """一条只有一个 clip 的臂必须说得出自己是哪一手。

    both-families 规则保护的是统一策略的两条通道;单 clip 根本没有通道之分(swing_sign 对每个
    env 都是同一个常数),规则在这里没有东西可保护,却会让反手独臂无法自述——只能落进 None 默认,
    而那条默认把单 clip 硬编码成正手,于是逐侧指标恒为 0.0000 而总量在动。
    """
    assert resolve(("backhand",), 1) == (False,)
    assert resolve(("forehand",), 1) == (True,)
    # 缺席仍走老默认(现役在跑臂逐字节不变),显式声明才纠正它
    assert resolve(None, 1) == (True,)
    # ≥2 clip 的保护一字未动
    with pytest.raises(ValueError, match="at least one forehand and one backhand"):
        resolve(("backhand", "backhand"), 2)


# --------------------------------------------------------------------------------------------- #
# MotionCommand 开机校验 + 懒推导(真 MotionCommand,合成 npz clip)
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def clip_files():
    tmp = tempfile.mkdtemp(prefix="clip_family_clips_")
    return tuple(
        _write_motion_npz(os.path.join(tmp, f"clip{i}.npz"), frames=12 + i) for i in range(3)
    )


def test_boot_rejects_bad_explicit_tables(clip_files):
    """显式表在 MotionCommand 构造时整表校验(boot fail-loud),不等到第一次查表。"""
    with pytest.raises(ValueError, match="3 entries.*2 clip"):
        _make_motion_command(
            [clip_files[0], clip_files[1]],
            clip_family_per_clip=("forehand", "backhand", "backhand"),
        )
    with pytest.raises(ValueError, match="at least one forehand and one backhand"):
        _make_motion_command(
            [clip_files[0], clip_files[1]], clip_family_per_clip=("forehand", "forehand")
        )
    with pytest.raises(ValueError, match="must be one of"):
        _make_motion_command(
            [clip_files[0], clip_files[1]], clip_family_per_clip=("forehand", "left_hand")
        )


def test_boot_accepts_valid_table_and_builds_tensor(clip_files, capsys):
    cmd, _ = _make_motion_command(
        [clip_files[0], clip_files[1]], clip_family_per_clip=("backhand", "forehand")
    )
    table = cmd.clip_family_is_forehand()
    assert table.dtype == torch.bool
    assert table.tolist() == [False, True]
    assert "clip_family_per_clip ACTIVE" in capsys.readouterr().out


def test_absent_config_builds_nothing_at_boot_and_derives_lazily(clip_files, capsys):
    """缺席 = 开机零动作(不建表、不打印,现役行为逐字节不变的最强证据),查表才懒推导。"""
    cmd, _ = _make_motion_command([clip_files[0], clip_files[1]])
    assert cmd.cfg.clip_family_per_clip is None
    assert cmd._clip_family_is_forehand_t is None
    assert "clip_family_per_clip" not in capsys.readouterr().out
    assert cmd.clip_family_is_forehand().tolist() == [True, False]
    single, _ = _make_motion_command([clip_files[2]])
    assert single.clip_family_is_forehand().tolist() == [True]


def test_absent_three_clips_boots_but_family_lookup_fails_loud(clip_files):
    """3-clip 缺表:纯动作模仿臂照常开机(legacy 路径不变),家族查表当场炸(fail-loud)。"""
    cmd, _ = _make_motion_command(list(clip_files))  # 构造成功 = 不影响无 racket 命令的任务
    with pytest.raises(ValueError, match="3 clips.*clip_family_per_clip"):
        cmd.clip_family_is_forehand()


# --------------------------------------------------------------------------------------------- #
# RacketTargetCommand._clip_family_is_forehand + 四处查表点
# --------------------------------------------------------------------------------------------- #
def _make_rt(num_envs, clip_ids, num_segments, families=None, with_cfg=True):
    RT = hope_commands_mod.RacketTargetCommand
    rt = RT.__new__(RT)
    rt.device = "cpu"
    rt.num_envs = num_envs
    motion_cfg = types.SimpleNamespace(clip_family_per_clip=families)
    motion = types.SimpleNamespace(
        _multiseg=num_segments > 1,
        clip_id=torch.as_tensor(clip_ids, dtype=torch.long),
        retiming_active=False,
        motion=types.SimpleNamespace(num_segments=num_segments),
    )
    if with_cfg:
        motion.cfg = motion_cfg
    rt._motion = lambda: motion
    rt._family_is_forehand_t = None
    return rt


def test_helper_absent_two_clip_is_bitwise_equal_to_legacy_hardcode():
    clips = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    rt = _make_rt(8, clips, num_segments=2)
    new_sign = torch.where(rt._clip_family_is_forehand()[clips], 1.0, -1.0)
    old_sign = torch.where(clips == 0, 1.0, -1.0)
    assert torch.equal(new_sign, old_sign)


def test_helper_six_clip_forehand_variants_match_clip0():
    """正手 1.0/1.2 变体(clip 1、2)swing_sign 必须和 clip 0 相同 = +1;反手三档 = −1。"""
    clips = torch.arange(6)
    rt = _make_rt(6, clips, num_segments=6, families=FAM6)
    sign = torch.where(rt._clip_family_is_forehand()[clips], 1.0, -1.0)
    assert sign.tolist() == [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    assert float(sign[1]) == float(sign[0]) and float(sign[2]) == float(sign[0])


def test_helper_is_cached_and_tolerates_cfgless_fakes():
    """懒缓存(第二次查表复用同一张量);老测试 fake 的 motion 没有 cfg 属性也按缺席推导,不炸。"""
    rt = _make_rt(2, [0, 1], num_segments=2, with_cfg=False)
    table = rt._clip_family_is_forehand()
    assert table.tolist() == [True, False]
    assert rt._clip_family_is_forehand() is table


def test_helper_absent_many_clips_fails_loud_at_lookup():
    rt = _make_rt(3, [0, 1, 2], num_segments=3)
    with pytest.raises(ValueError, match="clip_family_per_clip"):
        rt._clip_family_is_forehand()


def _make_uniform_rt(num_envs, clip_ids, num_segments, families=None):
    """真 _sample_targets_uniform 需要的最小状态(multiseg、共享盒、无 HER、无题库)。"""
    rt = _make_rt(num_envs, clip_ids, num_segments, families=families)
    rt.cfg = types.SimpleNamespace(
        racket_pos_x_range=(0.5, 0.7),
        racket_pos_y_abs_range=(0.1, 0.4),
        racket_pos_z_range=(0.9, 1.2),
        racket_vel_x_range=(1.0, 2.0),
        racket_vel_y_range=(-0.5, 0.5),
        racket_vel_z_range=(0.0, 1.0),
        forehand_on_negative_y=True,
        achieved_target_mix_prob=0.0,
    )
    rt._pos_range_per_clip_t = None
    rt._vel_range_per_clip_t = None
    rt._question_bank = None
    rt._ref_normal_per_clip = torch.nn.functional.normalize(
        torch.randn(num_segments, 3), dim=-1
    )
    rt.racket_target_pos_w = torch.zeros(num_envs, 3)
    rt.racket_target_vel_w = torch.zeros(num_envs, 3)
    rt.racket_target_normal_w = torch.zeros(num_envs, 3)
    return rt


def test_uniform_sampler_absent_config_is_bitwise_identical_to_old_formula():
    """相同随机种子:缺席配置的新路径(家族查表)输出与旧公式(clips==0)逐字节 torch.equal,
    包括 RNG 消耗次序——查表本身零随机数消耗。"""
    n = 16
    clips = torch.tensor([0, 1] * 8)
    env_ids = torch.arange(n)
    origins = torch.randn(n, 3)
    su = hope_commands_mod.sample_uniform

    rt = _make_uniform_rt(n, clips, num_segments=2)
    torch.manual_seed(20260722)
    rt._sample_targets_uniform(env_ids, origins.clone(), n)

    # 旧公式逐行复刻(共享盒 multiseg 分支,修改前源码)
    torch.manual_seed(20260722)
    pos = origins.clone()
    pos[:, 0] += su(*rt.cfg.racket_pos_x_range, (n,), "cpu")
    ymag = su(*rt.cfg.racket_pos_y_abs_range, (n,), "cpu")
    fh_sign = -1.0
    sign = torch.where(clips == 0, fh_sign, -fh_sign)
    pos[:, 1] = origins[:, 1] + sign * ymag
    pos[:, 2] += su(*rt.cfg.racket_pos_z_range, (n,), "cpu")
    vel = torch.empty(n, 3)
    vel[:, 0] = su(*rt.cfg.racket_vel_x_range, (n,), "cpu")
    vel[:, 1] = su(*rt.cfg.racket_vel_y_range, (n,), "cpu")
    vel[:, 2] = su(*rt.cfg.racket_vel_z_range, (n,), "cpu")

    assert torch.equal(rt.racket_target_pos_w, pos)
    assert torch.equal(rt.racket_target_vel_w, vel)
    assert torch.equal(rt.racket_target_normal_w, rt._ref_normal_per_clip[clips])


def test_uniform_sampler_six_clip_forehand_variants_share_the_forehand_y_side():
    """6-clip 家族表:正手三档(clip 0/1/2)目标 y 全在正手侧(forehand_on_negative_y -> y<origin),
    反手三档在对侧——变速变体不再被当成反手采到错误半场。"""
    n = 12
    clips = torch.tensor(list(range(6)) * 2)
    env_ids = torch.arange(n)
    origins = torch.zeros(n, 3)
    rt = _make_uniform_rt(n, clips, num_segments=6, families=FAM6)
    torch.manual_seed(7)
    rt._sample_targets_uniform(env_ids, origins.clone(), n)
    y = rt.racket_target_pos_w[:, 1]
    fh = torch.tensor([True, True, True, False, False, False] * 2)
    assert torch.all(y[fh] < 0.0), y
    assert torch.all(y[~fh] > 0.0), y


def test_source_guard_all_four_sites_use_family_lookup():
    """源码守卫:四处写死判断已死透——swing_sign/目标侧不再含 clips==0/clip==0,全走家族表。"""
    RT = hope_commands_mod.RacketTargetCommand
    for func in (
        RT._sample_targets_uniform,
        RT._install_event_training_questions,
        RT.install_external_exam_questions,
        RT._resample_command,
    ):
        src = inspect.getsource(func)
        assert "clips == 0" not in src and "clip == 0" not in src, func.__qualname__
        assert "_clip_family_is_forehand" in src, func.__qualname__
    # 解析/校验规则单一来源:helper 用的就是 commands.resolve_clip_family_is_forehand
    helper_src = inspect.getsource(RT._clip_family_is_forehand)
    assert "resolve_clip_family_is_forehand" in helper_src


# --------------------------------------------------------------------------------------------- #
# train.py 接线 + 合同块
# --------------------------------------------------------------------------------------------- #
def test_train_py_plumbs_whitelist_translation_and_conditional_contract_key():
    src = open(os.path.join(SCRIPTS_DIR, "train.py"), encoding="utf-8").read()
    assert '"clip_family_per_clip",' in src                       # _MOTION_KEYS 白名单
    assert '_set_attr(M, "clip_family_per_clip"' in src           # translation 落地 + applied 记录
    assert '"motion_clip_family_per_clip"' in src                 # 合同块字段
    # 合同键必须是条件落键:缺席不写键 => 合同字节与历史一致,老 checkpoint resume 对账不炸
    assert 'if getattr(motion, "clip_family_per_clip", None) is None' in src


def test_contract_roundtrip_valid_table_passes_structure_validation():
    contract = _schema3_contract()  # motion_segment_lengths == [11, 13](2 段)
    TC.validate_schema3_contract_structure(contract)  # 缺席键:legacy 合同照旧通过
    contract["motion_clip_family_per_clip"] = ["forehand", "backhand"]
    TC.validate_schema3_contract_structure(contract)
    contract["motion_clip_family_per_clip"] = ["backhand", "forehand"]
    TC.validate_schema3_contract_structure(contract)


def test_contract_rejects_malformed_family_arrays():
    contract = _schema3_contract()
    for bad in ([], ["forehand", 1], "forehand", ["forehand", "overhead"]):
        contract["motion_clip_family_per_clip"] = bad
        with pytest.raises(ValueError, match="motion_clip_family_per_clip"):
            TC.validate_schema3_contract_structure(contract)


def test_contract_rejects_missing_family_and_length_mismatch():
    contract = _schema3_contract()
    contract["motion_clip_family_per_clip"] = ["forehand", "forehand"]
    with pytest.raises(ValueError, match="at least one forehand and one"):
        TC.validate_schema3_contract_structure(contract)
    contract["motion_clip_family_per_clip"] = ["forehand", "backhand", "backhand"]
    with pytest.raises(ValueError, match="one family per loaded"):
        TC.validate_schema3_contract_structure(contract)


# --------------------------------------------------------------------------------------------- #
# face179 拍面符号合同(spdmix 硬绑定三拆除):legacy 逐字判法不变,家族表在场按族核对
# --------------------------------------------------------------------------------------------- #
_LEGACY_FACE_SIGN_MSG = (
    "formal face179 schema-3 contract requires mount_normal_sign_per_clip=[+1,-1]"
)


def _face179_contract(*, families=None, signs=(1.0, -1.0), clips=2):
    """最小 face179 合同:基线 2 段,clips=6 时把段长/fps/运动学合同表复制到 6 段。"""
    contract = _schema3_contract()
    if clips != 2:
        reps = clips // 2
        contract["motion_segment_lengths"] = list(contract["motion_segment_lengths"]) * reps
        contract["motion_clip_fps"] = list(contract["motion_clip_fps"]) * reps
        contract["motion_kinematics_contracts"] = (
            list(contract["motion_kinematics_contracts"]) * reps
        )
    contract["actor_obs_contract"] = "deploy_parity_face179"
    contract["face_command_enabled"] = True
    contract["face_command_pairing"] = "shared_plus_y"
    contract["mount_normal_sign_per_clip"] = list(signs)
    if families is not None:
        contract["motion_clip_family_per_clip"] = list(families)
    return contract


def test_face179_legacy_exact_signs_still_pass():
    """家族表缺席(现役 2-clip 臂):逐字 [+1,-1] 照旧通过,int 拼法也照旧被转成 float。"""
    TC.validate_schema3_contract_structure(_face179_contract())
    TC.validate_schema3_contract_structure(_face179_contract(signs=(1, -1)))


def test_face179_legacy_wrong_signs_reject_with_bytewise_identical_message():
    """家族表缺席 + 符号不是逐字 [+1,-1]:照旧拒绝,报错文本和改前逐字节一致。"""
    for signs in ((-1.0, 1.0), (1.0, 1.0), (1.0, -1.0, -1.0), (1.0,)):
        with pytest.raises(ValueError) as exc:
            TC.validate_schema3_contract_structure(_face179_contract(signs=signs))
        assert str(exc.value) == _LEGACY_FACE_SIGN_MSG, signs


def test_face179_legacy_malformed_signs_reject_with_bytewise_identical_message():
    """家族表缺席 + 缺键/布尔/非数值:照旧拒绝,报错文本逐字节一致。"""
    missing = _face179_contract()
    del missing["mount_normal_sign_per_clip"]
    boolean = _face179_contract(signs=(True, False))
    textual = _face179_contract(signs=("x", "-1"))
    for contract in (missing, boolean, textual):
        with pytest.raises(ValueError) as exc:
            TC.validate_schema3_contract_structure(contract)
        assert str(exc.value) == _LEGACY_FACE_SIGN_MSG


def test_face179_six_clip_family_signs_pass():
    """六 clip 变速烤入:正手三档全 +1、反手三档全 -1 —— 合法向量整体通过结构校验。"""
    TC.validate_schema3_contract_structure(
        _face179_contract(families=FAM6, signs=(1.0, 1.0, 1.0, -1.0, -1.0, -1.0), clips=6)
    )


def test_face179_family_order_is_honored_not_assumed():
    """家族表顺序说了算:反手在前就要 [-1,+1];仍抄 legacy 的 [+1,-1] 被按族拒绝。"""
    TC.validate_schema3_contract_structure(
        _face179_contract(families=("backhand", "forehand"), signs=(-1.0, 1.0))
    )
    with pytest.raises(ValueError, match="forehand.*backhand"):
        TC.validate_schema3_contract_structure(
            _face179_contract(families=("backhand", "forehand"), signs=(1.0, -1.0))
        )


def test_face179_forehand_clip_given_minus_one_rejects():
    """族符号错(正手变体给了 -1 = 用黑面打正手):fail-loud,不许悄悄换拍面。"""
    with pytest.raises(ValueError, match="forehand.*backhand"):
        TC.validate_schema3_contract_structure(
            _face179_contract(families=FAM6, signs=(1.0, 1.0, -1.0, -1.0, -1.0, -1.0), clips=6)
        )


def test_face179_backhand_clip_given_plus_one_or_zero_rejects():
    """反手族给 +1 或 0(非法拍面):同样按族拒绝。"""
    for signs in (
        (1.0, 1.0, 1.0, -1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0, -1.0, 0.0, -1.0),
    ):
        with pytest.raises(ValueError, match="forehand.*backhand"):
            TC.validate_schema3_contract_structure(
                _face179_contract(families=FAM6, signs=signs, clips=6)
            )


def test_face179_family_sign_length_mismatch_rejects():
    """符号数 != clip 数:fail-loud,报错说清各有几个。"""
    with pytest.raises(ValueError, match="2 signs for 6 clips"):
        TC.validate_schema3_contract_structure(
            _face179_contract(families=FAM6, signs=(1.0, -1.0), clips=6)
        )


def test_face179_family_malformed_signs_reject_with_family_message():
    """家族表在场 + 缺键/布尔:走家族版报错(不再误报 [+1,-1]),同样 fail-loud。"""
    missing = _face179_contract(families=FAM6, clips=6)
    del missing["mount_normal_sign_per_clip"]
    boolean = _face179_contract(
        families=FAM6, signs=(True, True, True, False, False, False), clips=6
    )
    for contract in (missing, boolean):
        with pytest.raises(ValueError, match="numeric.*entry per clip"):
            TC.validate_schema3_contract_structure(contract)


def test_non_face179_contract_leaves_sign_table_unjudged():
    """判定只挂 face179:普通合同带家族表 + 任意符号向量,结构校验不碰符号表照旧通过。"""
    contract = _face179_contract(families=FAM6, signs=(0.5, 2.0, -3.0, 0.0, 1.0, 1.0), clips=6)
    contract["actor_obs_contract"] = "wide"
    TC.validate_schema3_contract_structure(contract)


def test_train_py_records_whole_sign_vector_verbatim_in_contract():
    """合同块照实记录整个符号向量(train.py 不再有另一道 [+1,-1] 写死):
    源码守卫——合同字段逐字段照抄 racket cfg,train.py 全文无 (1.0, -1.0) 拍面钉死。"""
    src = open(os.path.join(SCRIPTS_DIR, "train.py"), encoding="utf-8").read()
    assert '"mount_normal_sign_per_clip": attr(racket, "mount_normal_sign_per_clip")' in src
    assert "(1.0, -1.0)" not in src and "[1.0, -1.0]" not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
