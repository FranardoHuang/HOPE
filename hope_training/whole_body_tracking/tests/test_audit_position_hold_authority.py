"""「位置指令撑不撑得住这个姿态」这条判据本身的测试。

这条判据的全部力量来自一个运动学事实:**脚不在腰/臂/头这些关节带动的那截身体里**,
所以脚底的支撑力在它们身上产生的力矩恒为零 —— 于是"要多大力矩"没有解算自由度,
它就等于 `qfrc_bias`。判据只对这一类关节说话;对腿上的关节它必须**闭嘴**,
因为那里地面真能帮忙,同一个数字什么也不证明。

这里的测试全是 dependency-light 的:MuJoCo 只负责给出"哪些行是 contact-free"
和 `qfrc_bias`,那两件在 pod 上跑;判据本身是纯 numpy,在 host 上就能钉死。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "materialize_a3_dynamic_ready_contract",
    _SCRIPTS / "materialize_a3_dynamic_ready_contract.py",
)
_DYNAMIC = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DYNAMIC
_SPEC.loader.exec_module(_DYNAMIC)

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_position_hold_authority",
    _SCRIPTS / "audit_position_hold_authority.py",
)
_AUDIT = importlib.util.module_from_spec(_AUDIT_SPEC)
sys.modules[_AUDIT_SPEC.name] = _AUDIT
_AUDIT_SPEC.loader.exec_module(_AUDIT)


# 现役 A3 的真实数字,take_061_unit04_BH 接地后的 frame 0(pod 实测):
# waist_pitch 需要 -49.155 N*m,位置指令最多只能到 -21.704 N*m。
_WAIST_PITCH_REQUIRED_NM = -49.15464109801158
_WAIST_PITCH_REACHABLE_NM = (-21.70368380844593, 15.052949115633965)
_WAIST_PITCH_KP = 50.0
_WAIST_PITCH_EFFORT_NM = 118.2
_WAIST_PITCH_Q_RAD = 0.03160078078508377
_WAIST_PITCH_QDES = (-0.40247289538383485, 0.3326597630977631)


def _case(**overrides):
    """Three joints: one contact-free and short, one contact-free and fine, one leg."""

    base = dict(
        joint_names=["waist_pitch_joint", "left_elbow_joint", "left_knee_joint"],
        contact_free=np.asarray([True, True, False]),
        required_nm=np.asarray([_WAIST_PITCH_REQUIRED_NM, -1.839, 900.0]),
        tau_lower_nm=np.asarray([_WAIST_PITCH_REACHABLE_NM[0], -24.0, -226.945]),
        tau_upper_nm=np.asarray([_WAIST_PITCH_REACHABLE_NM[1], 24.0, 303.199]),
        kp=np.asarray([_WAIST_PITCH_KP, 30.0, 250.0]),
        ready_q_rad=np.asarray([_WAIST_PITCH_Q_RAD, 0.2658, 1.0343]),
        executed_qdes_lower_rad=np.asarray([_WAIST_PITCH_QDES[0], -0.7029, 0.1265]),
        executed_qdes_upper_rad=np.asarray([_WAIST_PITCH_QDES[1], 1.4883, 2.2471]),
        motor_effort_nm=np.asarray([_WAIST_PITCH_EFFORT_NM, 24.0, 320.0]),
    )
    base.update(overrides)
    return base


def test_it_names_the_contact_free_joint_that_cannot_be_held():
    records = _DYNAMIC.contact_free_hold_torque_shortfall(**_case())
    assert [r["joint"] for r in records] == ["waist_pitch_joint"]
    only = records[0]
    assert only["required_hold_torque_nm"] == pytest.approx(_WAIST_PITCH_REQUIRED_NM)
    assert only["binding_side"] == "lower"
    assert only["shortfall_nm"] == pytest.approx(
        _WAIST_PITCH_REQUIRED_NM - _WAIST_PITCH_REACHABLE_NM[0]
    )
    assert only["binding_authority"] == "kp_times_available_qdes_travel"


def test_the_needed_qdes_is_exactly_q_plus_tau_over_kp():
    """报出来的"要多大 q_des"必须是可复算的,不是形容词。"""

    records = _DYNAMIC.contact_free_hold_torque_shortfall(**_case())
    assert records[0]["qdes_that_would_be_needed_rad"] == pytest.approx(
        _WAIST_PITCH_Q_RAD + _WAIST_PITCH_REQUIRED_NM / _WAIST_PITCH_KP
    )
    # 而且它确实落在执行包络之外 —— 这才是"撑不住"的人话版本。
    assert records[0]["qdes_that_would_be_needed_rad"] < _WAIST_PITCH_QDES[0]


def test_mutation_a_ground_loaded_joint_is_never_named():
    """变异测试(误拦方向):腿上的关节即使 `qfrc_bias` 离谱地大也不许被点名。

    把 `left_knee` 的 `qfrc_bias` 设成 `900 N*m`(远超它 `[-226.9, +303.2]` 的区间)。
    地面在膝盖上真能使上力,所以这个数字**什么也不证明**。一条把 `contact_free`
    忘掉的实现会在这里点名膝盖 —— 那就是把"重定向缺陷"读成"机器人站不住"的老毛病。
    """

    records = _DYNAMIC.contact_free_hold_torque_shortfall(**_case())
    assert "left_knee_joint" not in [r["joint"] for r in records]
    # 反过来:同一个数字挂在一个 contact-free 的关节上,就必须被点名。
    mutated = _case(contact_free=np.asarray([True, True, True]))
    assert "left_knee_joint" in [
        r["joint"]
        for r in _DYNAMIC.contact_free_hold_torque_shortfall(**mutated)
    ]


def test_mutation_only_checking_the_upper_side_would_miss_this_entire_finding():
    """变异测试(该拦仍拦):真实的短缺是在**下**边界,不是上边界。

    只查 `need > tau_upper` 的实现会让 take061 全部 57 帧静默通过。
    这里把同一个短缺镜像到上边界,两侧都必须被抓到。
    """

    lower_side = _DYNAMIC.contact_free_hold_torque_shortfall(**_case())
    assert lower_side[0]["binding_side"] == "lower"
    upper_side = _DYNAMIC.contact_free_hold_torque_shortfall(
        **_case(required_nm=np.asarray([-_WAIST_PITCH_REQUIRED_NM, -1.839, 900.0]))
    )
    assert [r["joint"] for r in upper_side] == ["waist_pitch_joint"]
    assert upper_side[0]["binding_side"] == "upper"


def test_mutation_a_ten_times_wider_envelope_still_refuses_this_pose():
    """"粗一档就过不了":把可达力矩区间放宽 `10` 倍,waist_pitch 仍然撑不住?

    不 —— `10` 倍会吞掉它(`-217 N*m` 够用了),所以这条测的是**另一半**:
    放宽到 `2.0` 倍(`-43.4 N*m`)时它必须**仍然**被拒,因为真实需求是 `-49.155`。
    也就是说这条判据不是靠一个松垮的容差活着的,它离边界还有 `5.7 N*m`。
    """

    doubled = _case(
        tau_lower_nm=np.asarray([2.0 * _WAIST_PITCH_REACHABLE_NM[0], -24.0, -226.945]),
    )
    still = _DYNAMIC.contact_free_hold_torque_shortfall(**doubled)
    assert [r["joint"] for r in still] == ["waist_pitch_joint"]
    # 放到 3 倍(-65.1 N*m)才够 —— 这就是那个缺口的价钱。
    tripled = _case(
        tau_lower_nm=np.asarray([3.0 * _WAIST_PITCH_REACHABLE_NM[0], -24.0, -226.945]),
    )
    assert _DYNAMIC.contact_free_hold_torque_shortfall(**tripled) == []


def test_the_binding_authority_separates_the_motor_from_the_gain():
    """卡在电机限幅上,和卡在 `kp × 行程` 上,是两件事,收据必须分清。

    分不清就会有人跑去换电机 —— 而现役这条卡的是增益,电机只用了 `41.6%`。
    """

    motor = _case(
        required_nm=np.asarray([-130.0, -1.839, 900.0]),
        tau_lower_nm=np.asarray([-_WAIST_PITCH_EFFORT_NM, -24.0, -226.945]),
    )
    records = _DYNAMIC.contact_free_hold_torque_shortfall(**motor)
    assert records[0]["binding_authority"] == "motor_effort_limit"
    assert _DYNAMIC.contact_free_hold_torque_shortfall(**_case())[0][
        "binding_authority"
    ] == "kp_times_available_qdes_travel"


def test_a_holdable_pose_produces_no_records_at_all():
    """现役出生姿态在 waist_pitch 上只要 `-18.746 N*m`,在区间内 —— 必须一条都不报。

    这一条是"误拦的不再拦"的正面证据:同一个判据既拒接地后的 frame 0,
    又放行现役出生姿态,所以它不是一个"什么都拒"的判据。
    """

    fine = _case(required_nm=np.asarray([-18.746346180035147, -1.839, 900.0]))
    assert _DYNAMIC.contact_free_hold_torque_shortfall(**fine) == []


def test_refusal_text_carries_the_joint_and_both_numbers():
    records = _DYNAMIC.contact_free_hold_torque_shortfall(**_case())
    text = _DYNAMIC._contact_free_hold_refusal_text(records)
    assert "waist_pitch_joint" in text
    assert "-49.155" in text
    assert "-21.704" in text
    assert "kp_times_available_qdes_travel" in text
    assert "q_des=-0.9515" in text


def test_the_refusal_still_refuses_when_the_attribution_itself_blows_up():
    """归因算不出来时,**不许**变成放行 —— 只许退回原来那句话。

    这是"改门要连证据一起改"的另一半:新增的自陈能力是叠上去的,
    不是把拒绝条件换掉的。归因用的是同一台 MuJoCo,它出问题时门必须原样还在。
    """

    class _Boom:
        @property
        def model(self):
            raise RuntimeError("mujoco is unavailable in this environment")

    text = _DYNAMIC._static_hold_refusal_message(
        backend=_Boom(),
        qpos=np.zeros(38),
        actuated=np.arange(6, 37),
        model_row_for_runtime=np.arange(31),
        plant={"joint_names": ["j%d" % i for i in range(31)],
               "kp": np.full(31, 50.0), "effort": np.full(31, 100.0)},
        ready_q=np.zeros(31),
        executed_qdes_lower=np.full(31, -1.0),
        executed_qdes_upper=np.full(31, 1.0),
        hold_tau_lower_model=np.full(31, -1.0),
        hold_tau_upper_model=np.full(31, 1.0),
    )
    assert text.startswith(_DYNAMIC.STATIC_HOLD_REFUSAL_PREFIX)
    assert "could not be computed" in text
    assert "RuntimeError" in text


def test_no_records_means_no_invented_culprit():
    """腰/臂/头都在区间里时,不许硬凑一个关节出来顶罪 —— 一条也不报。

    `_materialize` 里对应的分支会改口说"卡的是地面那一侧或摩擦锥",
    而不是把某个没问题的关节写进拒绝理由。
    """

    assert _DYNAMIC._contact_free_hold_refusal_text([]) == ""


def test_a_wrong_length_vector_fails_closed():
    with pytest.raises(_DYNAMIC.DynamicReadyMaterializationError):
        _DYNAMIC.contact_free_hold_torque_shortfall(
            **_case(kp=np.asarray([50.0, 30.0]))
        )


def test_position_command_torque_interval_is_kp_times_travel_capped_by_the_motor():
    lower, upper = _AUDIT.position_command_torque_interval(
        kp=np.asarray([50.0, 20.0]),
        effort=np.asarray([118.2, 6.0]),
        q_rad=np.asarray([0.0, 0.0]),
        qdes_lower=np.asarray([-0.4, -1.0]),
        qdes_upper=np.asarray([0.4, 1.0]),
    )
    # kp * travel is the binding side for the waist-like row ...
    assert lower[0] == pytest.approx(-20.0)
    assert upper[0] == pytest.approx(20.0)
    # ... and the motor limit for the wrist-like row (20 * 1.0 = 20 > 6.0).
    assert lower[1] == pytest.approx(-6.0)
    assert upper[1] == pytest.approx(6.0)


def test_executed_envelope_applies_both_the_inset_and_the_hard_inner_guard():
    mechanical = np.tile(np.asarray([[-1.0, 1.0]]), (31, 1))
    mechanical[0] = [-0.5, 0.5]
    plant = {
        "qdes_limits": np.tile(np.asarray([[-1.0, 1.0]]), (31, 1)),
        "projection_inset": 0.05,
        "physx_control_position_limits": {
            "mechanical_joint_pos_limits": mechanical
        },
    }
    lower, upper = _AUDIT.executed_qdes_envelope(plant)
    # row 0: the 2% hard-inner guard on [-0.5, 0.5] wins over the 5% inset on [-1, 1]
    assert lower[0] == pytest.approx(-0.48)
    assert upper[0] == pytest.approx(0.48)
    # row 1: no tighter mechanical limit, so the 5% projection inset governs
    assert lower[1] == pytest.approx(-0.9)
    assert upper[1] == pytest.approx(0.9)


def test_frame_selection_refuses_a_frame_outside_the_clip():
    assert _AUDIT._parse_frames("0,2-4", 10) == (0, 2, 3, 4)
    with pytest.raises(_AUDIT.PositionHoldAuditError):
        _AUDIT._parse_frames("0-12", 10)
    with pytest.raises(_AUDIT.PositionHoldAuditError):
        _AUDIT._parse_frames("", 10)


def test_summary_counts_poses_not_joints_and_keeps_the_two_causes_apart():
    rows = [
        {"holdable_by_position_command": False, "worst_shortfall_nm": 27.45,
         "gain_short_joints": ["waist_pitch_joint", "waist_roll_joint"],
         "pose_outside_envelope_joints": []},
        {"holdable_by_position_command": False, "worst_shortfall_nm": 0.0,
         "gain_short_joints": [],
         "pose_outside_envelope_joints": ["right_wrist_yaw_joint"]},
        {"holdable_by_position_command": True, "worst_shortfall_nm": 0.0,
         "gain_short_joints": [], "pose_outside_envelope_joints": []},
    ]
    assert _AUDIT._summarize(rows) == {
        "poses": 3,
        "holdable": 1,
        "not_holdable": 2,
        "poses_with_a_gain_shortfall": 1,
        "poses_with_a_joint_outside_the_qdes_envelope": 1,
        "gain_short_joint_counts": {"waist_pitch_joint": 1, "waist_roll_joint": 1},
        "outside_envelope_joint_counts": {"right_wrist_yaw_joint": 1},
        "worst_shortfall_nm": 27.45,
    }


def test_a_joint_parked_outside_the_qdes_envelope_gets_its_own_name():
    """变异测试(别把两种病混成一种):关节已经站在可发指令区间之外时,
    "差多少 N·m"是个假问题 —— 那时连零力矩都发不出来,区间本身是空的。

    现役库里真有这种帧(`Take_063_unit04_BH` 的 `right_wrist_yaw`:
    可达区间算出来是 `[-6.000, -6.168]`,下界比上界还大)。把它当成
    "增益不够"报出去,会让人跑去调 `kp`,而真正要动的是限位/姿态。
    """

    parked = _case(
        ready_q_rad=np.asarray([_WAIST_PITCH_QDES[1] + 0.25, 0.2658, 1.0343]),
    )
    records = _DYNAMIC.contact_free_hold_torque_shortfall(**parked)
    assert [r["joint"] for r in records] == ["waist_pitch_joint"]
    only = records[0]
    assert only["binding_side"] == "pose_outside_executed_qdes_envelope"
    assert only["binding_authority"] == "pose_outside_executed_qdes_envelope"
    assert only["pose_outside_envelope_by_rad"] == pytest.approx(0.25)
    # 没有"差多少 N·m"这个数,而且必须是 None 不是 NaN —— 报告用 allow_nan=False。
    assert only["shortfall_nm"] is None
    json.dumps(records, allow_nan=False)
    text = _DYNAMIC._contact_free_hold_refusal_text(records)
    assert "outside the executed q_des envelope" in text
    assert "short" not in text
