"""MuJoCo 评估器 phase-governor 三方对拍 + 老代际不变(2026-07-22 判卷协议修复 A).

人话:task-revision 代际训练时,参考不是 1x 原生播放,而是被 governor 奴役到
"initial_tts 倒数到 0 恰好走到击球帧"。评估器补的 numpy governor 必须和三份现役实现
一个字都不差:
  1. C++ PpPhaseGovernor(test_pp_phase_governor.cpp 的 golden trace 向量,1e-12);
  2. python 参考 planner_revision.advance_phase(逐 tick,1e-12;击球后语义有意分叉,
     参考版减速到 0,训练/评估版接原生跟随播放,故只对拍到到达 tick);
  3. torch 版 commands.py::_advance_planner_phase(从源文件原文抽出、float64 跑,
     帧域逐位 == 断言 —— 不是重抄公式,是真源码).
另外锁死:无 planner_task_revision metadata 的老代际 ONNX 走原路径逐字节不变。

无 GPU、无 Isaac、无 MuJoCo/onnxruntime 依赖;torch 仅第 3 组用,缺了自动 skip。
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "scripts" / "mujoco_eval_onnx.py"
PLANNER_PATH = (
    ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking"
    / "mdp" / "planner_revision.py"
)
COMMANDS_PATH = (
    ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking"
    / "mdp" / "commands.py"
)
CPP_TEST_PATH = (
    ROOT.parents[1] / "agi" / "a3_deploy_example" / "src" / "a3" / "a3_deploy_onnx_ref"
    / "unit_tests" / "test_pp_phase_governor.cpp"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load("mj_eval_governor_under_test", EVAL_PATH)
P = _load("planner_revision_for_governor_parity", PLANNER_PATH)


# 与 C++ Profile() / test_planner_revision._profile 同一套数(min_tts 0.1 便于短 tts 向量).
CPP_PROFILE = {
    "policy_dt_s": 0.02,
    "min_tts_s": 0.1,
    "max_tts_s": 2.0,
    "max_phase_rate_per_s": 4.0,
    "max_phase_acceleration_per_s2": 20.0,
    "max_deadline_revision_delta_s": 0.25,
    "max_position_revision_delta_m": 0.1,
    "max_velocity_revision_delta_mps": 0.5,
    "max_normal_revision_delta_rad": 0.2,
    "normal_unit_tolerance": 1e-4,
    "early_deadline_tolerance_s": 1e-9,
}

# C++ test_pp_phase_governor.cpp UrgentDeadlineGoldenTrace{PointFour,PointFive}Seconds 原向量.
CPP_GOLDEN = {
    0.4: {
        1: (0.004, 0.4),
        10: (0.3895454545454533, 3.3909090909090445),
        13: (0.5970261994949448, 3.791445707070643),
        19: (0.92, 4.0),
        20: (1.0, None),
    },
    0.5: {
        1: (0.004, 0.4),
        20: (0.809682744022909, 2.378965699713386),
        21: (0.8612620580171767, 2.778965699713386),
        24: (0.9244206860057322, 3.9789656997133855),
        25: (1.0, None),
    },
}


def _gov(initial_tts, seg_start=(0,), seg_len=(2,), strike_phases=(1.0,), profile=None):
    """EvalPhaseGovernor with span=1 by default: frame domain == C++/reference phase domain."""
    cfg = {
        "profile": dict(profile or CPP_PROFILE),
        "profile_sha256": "0" * 64,
        "initial_tts_range_s": (0.11, 1.9),
        "initial_tts_s": float(initial_tts),
        "strike_phases": tuple(strike_phases),
    }
    return M.EvalPhaseGovernor(cfg, list(seg_start), list(seg_len), step_dt=0.02)


def test_cpp_golden_trace_vectors_match_numpy_eval_governor():
    for tts, checkpoints in CPP_GOLDEN.items():
        gov = _gov(tts)
        gov.begin(0)
        for tick in range(1, max(checkpoints) + 1):
            gov.advance()
            if tick in checkpoints:
                phase, rate = checkpoints[tick]
                assert gov.time_step_f == pytest.approx(phase, abs=1e-12, rel=0.0), (tts, tick)
                if rate is not None:
                    assert gov.phase_rate == pytest.approx(rate, abs=1e-12, rel=0.0), (tts, tick)
        assert gov.time_step_f == 1.0
        # 到达帧 = deadline 到点帧:truth tts 同一 tick 打到 0.
        assert gov.tts_obs == 0.0


def test_cpp_golden_vectors_in_this_file_are_verbatim_from_the_cpp_test():
    source = CPP_TEST_PATH.read_text(encoding="utf-8")
    for checkpoints in CPP_GOLDEN.values():
        for phase, rate in checkpoints.values():
            if phase not in (0.004, 1.0):
                assert repr(phase) in source, phase
            if rate is not None and rate not in (0.4, 4.0):
                assert repr(rate) in source, rate


def test_rate_and_acceleration_bounds_and_monotonic_phase():
    """C++ 边界测试的击球前段;击球后训练/评估版转原生跟随(C++ 参考钉 1,有意分叉)."""
    profile = CPP_PROFILE
    gov = _gov(0.8)
    gov.begin(0)
    prior_phase, prior_rate = 0.0, 0.0
    for tick in range(1, 151):
        gov.advance()
        assert gov.time_step_f >= prior_phase
        assert 0.0 <= gov.phase_rate <= profile["max_phase_rate_per_s"]
        assert abs(gov.phase_rate - prior_rate) <= (
            profile["max_phase_acceleration_per_s2"] * profile["policy_dt_s"] + 1e-12
        )
        prior_phase, prior_rate = gov.time_step_f, gov.phase_rate
        if gov.time_step_f >= 1.0:
            break
    assert gov.time_step_f == 1.0
    assert tick == 40  # 0.8s / 0.02 —— 到达 tick 恰为 deadline tick


def _reference_ledger(profile_map, tts):
    profile = P.PhaseGovernorProfile(**profile_map)
    truth = P.LatentTaskTruth(control_epoch=7, task_id=10, truth_sha256="a" * 64)
    revision = P.PlannerTaskRevision(
        control_epoch=7,
        task_id=10,
        task_revision=1,
        command_sequence=100,
        source_monotonic_s=5.1,
        truth_sha256="a" * 64,
        target_position_m=(0.2, -0.1, 0.9),
        target_velocity_mps=(-2.0, 0.0, -0.2),
        target_normal=(1.0, 0.0, 0.0),
        desired_tts_s=tts,
    )
    decision = P.begin_task(
        profile, P.PhaseGovernorLedger(), truth, revision, local_monotonic_s=20.0
    )
    assert decision.accepted, decision.reason
    return profile, decision.ledger


@pytest.mark.parametrize("tts", [0.36, 0.4, 0.5, 0.7, 0.9, 1.3, 1.7])
def test_full_prestrike_trace_matches_python_reference(tts):
    """逐 tick 对拍到到达 tick(含);击球后参考版减速到 0、训练/评估版接原生跟随,有意分叉."""
    profile, ledger = _reference_ledger(CPP_PROFILE, tts)
    gov = _gov(tts)
    gov.begin(0)
    for tick in range(1, 200):
        ledger = P.advance_phase(profile, ledger)
        gov.advance()
        assert ledger.active is not None
        assert gov.time_step_f == pytest.approx(ledger.active.phase, abs=1e-12, rel=0.0), tick
        assert gov.phase_rate == pytest.approx(
            ledger.active.phase_rate_per_s, abs=1e-12, rel=0.0
        ), tick
        if ledger.active.phase >= 1.0:
            break
    assert ledger.active is not None and ledger.active.phase == 1.0
    assert gov.time_step_f == 1.0
    # 到达 tick == deadline tick(governor 的设计承诺:不早到、不迟到).
    assert tick == int(round(tts / 0.02))
    assert gov.tts_obs == 0.0


# ---------------------------------------------------------------------------------------------
# torch 版 commands.py::_advance_planner_phase 真源码逐位对拍(float64)
# ---------------------------------------------------------------------------------------------

def _extract_method(source, header):
    start = source.index(header)
    nxt = source.index("\n    def ", start + len(header))
    return source[start:nxt]


def _build_torch_shim():
    torch = pytest.importorskip("torch")
    source = COMMANDS_PATH.read_text(encoding="utf-8")
    minimum = _extract_method(
        source, "    @staticmethod\n    def _planner_minimum_finish_time("
    )
    advance = _extract_method(source, "    def _advance_planner_phase(")
    class_src = "class TorchGovernorShim:\n" + minimum + "\n" + advance + "\n"
    namespace = {"torch": torch}
    exec(compile(class_src, str(COMMANDS_PATH), "exec"), namespace)  # noqa: S102 — 真源码对拍
    return torch, namespace["TorchGovernorShim"]


def _torch_state(torch, shim_cls, profile_map, *, start, strike, tts):
    shim = shim_cls()
    one = lambda v: torch.tensor([float(v)], dtype=torch.float64)  # noqa: E731
    shim._planner_revision_profile = SimpleNamespace(**profile_map)
    shim.metrics = {
        "planner_revision_accepted": one(0.0),
        "planner_revision_rejected": one(0.0),
    }
    shim._planner_active = torch.tensor([True])
    shim._planner_start_step = one(start)
    shim._planner_strike_step = one(strike)
    shim._planner_phase_rate = one(0.0)
    shim._planner_slow_only_next = torch.tensor([False])
    shim._planner_desired_tts = one(tts)
    shim._planner_truth_tts = one(tts)
    shim.time_steps_f = one(start)
    return shim


@pytest.mark.parametrize("tts", [0.36, 0.4, 0.5, 0.9, 1.7])
@pytest.mark.parametrize(
    ("seg_len", "strike_phase"),
    [(140, 0.471), (135, 0.338), (2, 1.0)],
)
def test_frame_domain_trace_bitwise_matches_torch_commands_source(tts, seg_len, strike_phase):
    torch, shim_cls = _build_torch_shim()
    gov = _gov(tts, seg_start=(0,), seg_len=(seg_len,), strike_phases=(strike_phase,))
    gov.begin(0)
    shim = _torch_state(
        torch, shim_cls, CPP_PROFILE,
        start=gov.start_step, strike=gov.strike_step, tts=gov.initial_tts,
    )
    held = torch.tensor([False])
    for tick in range(1, int(round(tts / 0.02)) + 2 * seg_len + 80):
        frame_delta = shim._advance_planner_phase(held)
        shim.time_steps_f += torch.where(
            shim._planner_active, frame_delta, torch.zeros_like(frame_delta)
        )
        gov.advance()
        assert float(shim.time_steps_f) == gov.time_step_f, tick
        assert float(shim._planner_phase_rate) == gov.phase_rate, tick
        assert float(shim._planner_truth_tts) == gov.truth_tts, tick
        assert int(shim.time_steps_f.round().long()) == gov.time_step, tick
        if gov.time_step_f >= seg_len - 1:
            break
    # 击球后原生跟随把帧推到剪辑末尾(wrap 条件可达),不是停在击球帧.
    assert gov.time_step_f >= seg_len - 1


def test_governor_arrival_frame_and_followthrough_on_realistic_clip():
    seg_len, phase = 140, 0.471
    gov = _gov(0.5, seg_start=(0,), seg_len=(seg_len,), strike_phases=(phase,))
    gov.begin(0)
    strike_frame = int(round(phase * (seg_len - 1)))
    for _ in range(25):
        gov.advance()
    # 25 步(0.5s)后:deadline 打到 0,参考恰好站在击球帧 —— 判分帧 = governor 到达帧.
    assert gov.tts_obs == 0.0
    assert gov.time_step == strike_frame
    # 跟随段:帧单调、最终越过剪辑末尾触发 wrap;tts 全程钉 0(训练 schema-4 语义).
    prev = gov.time_step_f
    for _ in range(2 * seg_len):
        gov.advance()
        assert gov.time_step_f >= prev
        assert gov.tts_obs == 0.0
        prev = gov.time_step_f
        if gov.time_step >= seg_len:
            break
    assert gov.time_step >= seg_len


def test_begin_rearms_exact_latch_and_multi_clip_frames():
    gov = _gov(
        0.5, seg_start=(0, 140), seg_len=(140, 135), strike_phases=(0.471, 0.338)
    )
    gov.begin(1)
    assert gov.time_step == 140
    assert gov.strike_step == 140 + int(round(0.338 * (135 - 1)))
    gov.exact_fired = True
    gov.begin(0)
    assert gov.exact_fired is False
    assert gov.time_step == 0
    assert gov.tts_obs == gov.initial_tts == 0.5


def test_infeasible_or_out_of_support_initial_tts_fails_loud():
    with pytest.raises(SystemExit):
        _gov(0.2)  # < minimum_finish_time(1, 0, 4, 20) = 0.35 —— begin 不可达
    with pytest.raises(SystemExit):
        _gov(1.95)  # 在 governor 包络内但在 initial_tts_range_s (0.11, 1.9) 支持外


# ---------------------------------------------------------------------------------------------
# metadata 门控:解析、拒收、老代际不变
# ---------------------------------------------------------------------------------------------

def _metadata_doc(profile=None):
    profile = dict(profile or CPP_PROFILE)
    profile["contract_version"] = "phase_governor_v1"
    profile["schema_version"] = 1
    canonical = (
        json.dumps(profile, allow_nan=False, ensure_ascii=False,
                   separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    return {
        "enabled": True,
        "revision_schema_version": 1,
        "governor": {
            "contract_version": "phase_governor_v1",
            "schema_version": 1,
            "profile_sha256": hashlib.sha256(canonical).hexdigest(),
            "profile": profile,
        },
        "initial_tts_range_s": [0.36, 1.7],
    }


def test_metadata_absent_returns_none_and_rollout_default_is_legacy():
    assert M.planner_governor_config_from_metadata({}) is None
    assert M.planner_governor_config_from_metadata({"planner_task_revision": ""}) is None
    # 老代际逐字节不变的结构性钥匙:不传 cfg 时 run_rollout 走原路径.
    assert inspect.signature(M.run_rollout).parameters["planner_governor_cfg"].default is None


def test_metadata_valid_roundtrip_binds_profile_and_range():
    doc = _metadata_doc()
    cfg = M.planner_governor_config_from_metadata(
        {"planner_task_revision": json.dumps(doc, separators=(",", ":"), sort_keys=True)}
    )
    assert cfg is not None
    assert cfg["profile"]["max_phase_rate_per_s"] == 4.0
    assert cfg["profile"]["max_phase_acceleration_per_s2"] == 20.0
    assert cfg["profile"]["policy_dt_s"] == 0.02
    assert cfg["initial_tts_range_s"] == (0.36, 1.7)
    assert cfg["profile_sha256"] == doc["governor"]["profile_sha256"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(enabled=False),
        lambda d: d.update(revision_schema_version=2),
        lambda d: d["governor"].update(profile_sha256="f" * 64),
        lambda d: d["governor"]["profile"].pop("max_phase_rate_per_s"),
        lambda d: d["governor"]["profile"].update(max_phase_rate_per_s=float("nan")),
        lambda d: d.update(initial_tts_range_s=[1.7, 0.36]),
        lambda d: d.update(initial_tts_range_s=[0.36, 2.5]),
        lambda d: d["governor"].update(contract_version="phase_governor_v2"),
    ],
)
def test_metadata_malformed_fails_closed_never_falls_back_to_legacy_clock(mutate):
    doc = _metadata_doc()
    mutate(doc)
    with pytest.raises(SystemExit):
        M.planner_governor_config_from_metadata(
            {"planner_task_revision": json.dumps(doc, separators=(",", ":"), sort_keys=True)}
        )


def test_export_and_eval_agree_on_the_metadata_key_spelling():
    training_contract = (
        ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils"
        / "training_contract.py"
    ).read_text(encoding="utf-8")
    assert 'PLANNER_TASK_REVISION_KEY = "planner_task_revision"' in training_contract
    eval_source = EVAL_PATH.read_text(encoding="utf-8")
    assert 'metadata.get("planner_task_revision"' in eval_source


def test_reference_postcontact_deliberately_diverges_from_followthrough():
    """钉死语义注释:参考版击球后 rate 减速到 0(部署 tts 语义),训练/评估版接原生播放."""
    profile, ledger = _reference_ledger(CPP_PROFILE, 0.5)
    gov = _gov(0.5, seg_start=(0,), seg_len=(140,), strike_phases=(0.471,))
    gov.begin(0)
    for _ in range(25):
        ledger = P.advance_phase(profile, ledger)
        gov.advance()
    assert ledger.active is not None and ledger.active.phase == 1.0
    for _ in range(30):
        ledger = P.advance_phase(profile, ledger)
        gov.advance()
    assert ledger.active.phase == 1.0                       # 参考:phase 钉死在击球
    assert ledger.active.phase_rate_per_s == 0.0            # 参考:rate 减速归零
    assert gov.time_step_f > gov.strike_step                # 评估:跟随段继续走帧
