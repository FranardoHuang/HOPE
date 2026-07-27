"""离线端不再产出"减速档"独立资产 —— bake 工具 × 慢放门的接线测试。

人话:r < 1 的减速片段由运行时慢放交付(R14 分数时钟),bake 工具默认拒绝生产它们;
r ≥ 1 照旧必须烤(时钟只往下走)。--envelope 的 manifest 里写清楚每档怎么交付、
省下几次烤入。历史对照仍可用 --force-slowdown-bake 强出,但资产上永久标记。

纯 CPU,NO mujoco/torch:沿用 test_bake_topp_strike_speed 的合成 clip 口径。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_bake_slowdown_delivery.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _name in ("synthesize_timing", "synthesize_timing_v2", "topp_mintime",
              "canonical_playback_speed_gate", "bake_topp_strike_speed"):
    _spec = importlib.util.spec_from_file_location(_name, _SCRIPTS / (_name + ".py"))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bk = sys.modules["bake_topp_strike_speed"]
pbg = sys.modules["canonical_playback_speed_gate"]
v1 = sys.modules["synthesize_timing"]

FPS = 50.0
J = 31
NB = 32
T = 81
CONTACT = 48
PHASE = CONTACT / (T - 1)


def make_clip(T=T, blade_step=0.02, joint_amp=3.0):
    q = np.zeros((T, J), dtype=np.float32)
    q[:, 24] = np.linspace(0.0, joint_amp, T, dtype=np.float32)
    dq = np.gradient(q.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    bp = np.zeros((T, NB, 3), dtype=np.float32)
    bp[:, v1.RACKET_BODY, 0] = np.arange(T, dtype=np.float32) * blade_step
    bq = np.zeros((T, NB, 4), dtype=np.float32)
    bq[..., 0] = 1.0
    bl = np.gradient(bp.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    ba = np.zeros_like(bp)
    data = {"fps": np.array([int(FPS)], dtype=np.int64), "joint_pos": q, "joint_vel": dq,
            "body_pos_w": bp, "body_quat_w": bq, "body_lin_vel_w": bl,
            "body_ang_vel_w": ba}
    data.update(v1.metadata_arrays(
        body_names=["body_%d" % index for index in range(NB)],
        body_lin_vel_point="link_origin",
    ))
    return data


def _write_clip(path: Path, data: dict) -> None:
    np.savez(path, **data)


def _loose_budget_npz(path: Path) -> None:
    """最小 budget clip:每关节 |Δq̇|·fps = 1000(宽松包络,全为正,无 floor)。"""
    jv = np.zeros((2, J), dtype=np.float32)
    jv[1] = 1000.0 / FPS
    np.savez(path, fps=np.array([int(FPS)], dtype=np.int64), joint_vel=jv)


@pytest.fixture()
def workspace(tmp_path):
    clip = tmp_path / "clip.npz"
    _write_clip(clip, make_clip())
    budget = tmp_path / "budget.npz"
    _loose_budget_npz(budget)
    return tmp_path, clip, budget


def _base_args(clip, budget):
    return ["--motion", str(clip), "--phase", str(PHASE),
            "--budget-clips", str(budget), "--body-mode", "interp"]


# --------------------------------------------------------------------------------- #
# 减速档:默认拒绝出货                                                                #
# --------------------------------------------------------------------------------- #
def test_slowdown_bake_is_refused_by_default(workspace):
    tmp_path, clip, budget = workspace
    out = tmp_path / "slow.npz"
    manifest = tmp_path / "slow.json"
    with pytest.raises(SystemExit) as excinfo:
        bk.main([*_base_args(clip, budget), "--speed-ratio", "0.8",
                 "--out", str(out), "--manifest", str(manifest)])
    message = str(excinfo.value)
    assert "refusing to bake a slowdown variant" in message
    assert "canonical_playback_speed_gate.py" in message
    assert not out.exists(), "拒绝时不许落任何资产"
    assert not manifest.exists()


def test_refusal_says_a_bake_would_not_rescue_an_infeasible_ratio(workspace):
    """拒绝信必须讲清楚:重定时不改位形 ⇒ 不随 s 缩放的重力项一模一样。"""
    tmp_path, clip, budget = workspace
    with pytest.raises(SystemExit) as excinfo:
        bk.main([*_base_args(clip, budget), "--speed-ratio", "0.65",
                 "--out", str(tmp_path / "a.npz"), "--manifest", str(tmp_path / "a.json")])
    assert "gravity term c0(q) is" in str(excinfo.value)


def test_native_and_speedup_bakes_still_work(workspace):
    tmp_path, clip, budget = workspace
    for ratio, kind in ((1.0, "bake"), (1.1, "bake")):
        out = tmp_path / ("r%s.npz" % ratio)
        manifest_path = tmp_path / ("r%s.json" % ratio)
        assert bk.main([*_base_args(clip, budget), "--speed-ratio", str(ratio),
                        "--out", str(out), "--manifest", str(manifest_path)]) == 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["delivery"]["kind"] == kind
        assert manifest["feasibility"]["verdict"] == "feasible"


def test_forced_slowdown_bake_is_permanently_marked(workspace):
    tmp_path, clip, budget = workspace
    out = tmp_path / "forced.npz"
    manifest_path = tmp_path / "forced.json"
    assert bk.main([*_base_args(clip, budget), "--speed-ratio", "0.8",
                    "--out", str(out), "--manifest", str(manifest_path),
                    "--force-slowdown-bake"]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["delivery"]["kind"] == "forced_slowdown_bake"
    assert manifest["delivery"]["runtime_playback_max_ratio"] == 1.0
    assert out.exists()


# --------------------------------------------------------------------------------- #
# envelope manifest 的交付计划                                                        #
# --------------------------------------------------------------------------------- #
def test_envelope_manifest_carries_a_fail_closed_delivery_plan(workspace):
    tmp_path, clip, budget = workspace
    manifest_path = tmp_path / "envelope.json"
    assert bk.main([*_base_args(clip, budget), "--envelope",
                    "--manifest", str(manifest_path)]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = manifest["delivery_plan"]
    assert plan["certified"] is False
    assert plan["bakes_saved"] == 0           # 没证书 = 一个都不许省
    assert plan["bakes_required"] == 6


def test_envelope_with_certificate_reports_the_saved_bakes(workspace):
    tmp_path, clip, budget = workspace
    certificate = tmp_path / "cert.json"
    certificate.write_text(json.dumps({
        "verdict": "feasible", "max_playback_ratio": 1.25, "requested_ratio": 1.0,
    }), encoding="utf-8")
    manifest_path = tmp_path / "envelope_cert.json"
    assert bk.main([*_base_args(clip, budget), "--envelope",
                    "--manifest", str(manifest_path),
                    "--playback-certificate", str(certificate)]) == 0
    plan = json.loads(manifest_path.read_text(encoding="utf-8"))["delivery_plan"]
    assert plan["certified"] is True
    assert plan["bakes_saved"] == 3           # 0.65 / 0.80 / 0.90 由运行时慢放交付
    assert plan["bakes_required"] == 3        # 1.10 / 1.20 / 1.35 仍必须烤
    deliveries = {row["ratio"]: row["delivery"] for row in plan["ladder"]}
    assert deliveries[1.0] == "native_source"


# --------------------------------------------------------------------------------- #
# MUTATION CHECK:把"减速档默认拒绝"这条规则突变掉,断言测试抓得住                     #
# --------------------------------------------------------------------------------- #
def test_mutation_removing_the_slowdown_refusal_is_caught(workspace, monkeypatch):
    """突变体:把阈值从 1.0 挪到 0.0(等于关掉这条规则)。

    突变后 r=0.8 会安安静静产出一份重复资产;真实实现拒绝。这条测试保证规则被删/被放宽
    时红灯,而不是悄悄回到"每个速度烤一份"。
    """
    tmp_path, clip, budget = workspace
    monkeypatch.setattr(bk, "SLOWDOWN_BAKE_EPS", 1.0 + 1e-9)   # 1.0 - EPS <= 0
    out = tmp_path / "mutant.npz"
    manifest_path = tmp_path / "mutant.json"
    assert bk.main([*_base_args(clip, budget), "--speed-ratio", "0.8",
                    "--out", str(out), "--manifest", str(manifest_path)]) == 0
    assert out.exists(), "mutant must silently produce the duplicate asset"

    monkeypatch.undo()
    out2 = tmp_path / "real.npz"
    with pytest.raises(SystemExit, match="refusing to bake a slowdown variant"):
        bk.main([*_base_args(clip, budget), "--speed-ratio", "0.8",
                 "--out", str(out2), "--manifest", str(tmp_path / "real.json")])
    assert not out2.exists()
