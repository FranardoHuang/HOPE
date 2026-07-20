"""Unit tests for scripts/bake_topp_strike_speed.py (轴 A:击球速度 ×r 重定时 bake).

Pure CPU, NO mujoco/torch:模块按文件路径加载,body_mode="interp"(link_origin 测试
clip),运动学限值直接喂数组。合成 clip 与 test_topp_mintime 同款:单活动关节 +
拍心匀速直线(源干净拍速 = 1.0 m/s),所以 r 与实测拍速的换算是解析可验的。

Covered(任务 2026-07-20 轴 A 验收 ①-⑤):
  ① 恒等 r=1.0:joint_pos/body_* 逐字节不变,joint_vel 数值近似不变,帧数/触球帧/相位不动
  ② 实测拍速随 r 单调,且每个可行 r 的误差 ≤2%(manifest 断言口径)
  ③ 不可行 fail-closed:超包络 / 目标速度不可达 → physical-infeasible,拒发资产,
     且不许静默降速(拒发时目标拍速仍是解到位的,不是打折货)
  ④ SHA 记录:manifest 源/输出内容 SHA 与磁盘一致;分桶 train/interp/OOD 写死
  ⑤ --envelope:max 可行 r 有限、限位收紧则包络收缩;r=1 自身不可行 = fail loud

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_bake_topp_strike_speed.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _name in ("synthesize_timing", "synthesize_timing_v2", "topp_mintime",
              "bake_topp_strike_speed"):
    _spec = importlib.util.spec_from_file_location(_name, _SCRIPTS / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bk = sys.modules["bake_topp_strike_speed"]
v1 = sys.modules["synthesize_timing"]

FPS = 50.0
J = 31
NB = 32
T = 81
CONTACT = 48
PHASE = CONTACT / (T - 1)

LOOSE_VLIM = np.full(J, 100.0)
LOOSE_ACC = np.full(J, 1000.0)


def make_clip(T=T, blade_step=0.02, joint_amp=3.0):
    """单活动关节(col 24)线性 + 拍心 +x 匀速 blade_step m/帧 → 干净拍速 1.0 m/s。"""
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
        body_names=[f"body_{index}" for index in range(NB)],
        body_lin_vel_point="link_origin",
    ))
    return data


def _bake(data, ratio, **kw):
    kw.setdefault("body_mode", "interp")
    return bk.bake(data, PHASE, ratio, kw.pop("vlim", LOOSE_VLIM),
                   kw.pop("acc_env", LOOSE_ACC), **kw)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_clip(path: Path, data: dict) -> None:
    np.savez(path, **data)


def _loose_budget_npz(path: Path) -> None:
    """最小 budget clip:每关节 |Δq̇|·fps = 1000(宽松包络,全为正,无 floor)。"""
    jv = np.zeros((2, J), dtype=np.float32)
    jv[1] = 1000.0 / FPS
    np.savez(path, fps=np.array([int(FPS)], dtype=np.int64), joint_vel=jv)


# --------------------------------------------- ① 恒等 r=1.0 --------------------------- #
def test_identity_ratio_is_bitwise_noop():
    data = make_clip()
    res = _bake(data, 1.0)
    assert res.feasible and res.reasons == []
    assert res.rho == 1.0                                    # 解析种子:恰好 1.0
    out = res.out
    assert np.array_equal(out["joint_pos"], data["joint_pos"])       # 逐字节
    assert np.array_equal(out["body_pos_w"], data["body_pos_w"])
    assert np.array_equal(out["body_quat_w"], data["body_quat_w"])
    assert np.allclose(out["joint_vel"], data["joint_vel"], atol=1e-5)
    assert out["joint_pos"].shape[0] == T                    # 帧数不变
    assert res.error_frac == 0.0                             # 实测拍速恰 = v0


def test_identity_keeps_contact_registration():
    data = make_clip()
    res = _bake(data, 1.0)
    assert res.c == CONTACT
    assert res.contact["frame"] == CONTACT
    assert abs(res.contact["phase"] - PHASE) < 1e-12


# --------------------------------------------- ② 拍速单调 + 误差 ≤2% ------------------- #
def test_strike_speed_scales_monotonically():
    data = make_clip()
    actuals = []
    for r in (0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35):
        res = _bake(data, r)
        assert res.feasible, (r, res.reasons)
        assert abs(res.actual_mps - r * res.v0_mps) / (r * res.v0_mps) <= 0.02
        assert res.error_frac <= 0.02
        actuals.append(res.actual_mps)
    assert all(b > a for a, b in zip(actuals, actuals[1:]))  # 严格单调

def test_contact_row_face_and_duration_preserved():
    data = make_clip()
    for r in (0.8, 1.2):
        res = _bake(data, r)
        out = res.out
        assert np.array_equal(out["joint_pos"][CONTACT], data["joint_pos"][CONTACT])
        assert res.contact["row_bitwise"] is True
        assert res.contact["face_normal_dev_deg"] < 1e-6
        assert res.contact["face_min_dot_over_clip"] > 0.999
        assert out["joint_pos"].shape[0] == T                # 总时长 = 帧数/fps 不变
        assert float(np.asarray(out["fps"]).reshape(-1)[0]) == FPS
        assert res.contact["frame"] == CONTACT               # 触球帧号不变
        assert abs(res.contact["phase"] - PHASE) < 1e-12     # 相位不变


def test_speed_tolerance_is_asserted_fail_closed():
    data = make_clip()
    with pytest.raises(SystemExit, match="deviates"):
        _bake(data, 1.17, speed_tol=1e-12)   # float32 资产量化误差必然 > 1e-12


# --------------------------------------------- ③ 不可行 fail-closed ------------------- #
def test_kinematic_infeasible_refuses_without_silent_slowdown():
    data = make_clip()
    cruise = 3.0 / (T - 1) * FPS                             # 巡航 |q̇| = 1.875 rad/s
    tight_vlim = np.full(J, cruise * 1.02 / 0.85)            # r=1 可行,r=1.2 必超
    ok = _bake(data, 1.0, vlim=tight_vlim)
    assert ok.feasible
    bad = _bake(data, 1.2, vlim=tight_vlim)
    assert not bad.feasible
    assert any("physical-infeasible" in reason for reason in bad.reasons)
    assert bad.kin["offending"], "manifest 必须点名越界关节"
    # 禁止静默降速:拒发时目标拍速仍解到位(误差 ≤2%),没有打折资产
    assert bad.error_frac is not None and bad.error_frac <= 0.02
    assert abs(bad.actual_mps - 1.2 * bad.v0_mps) / (1.2 * bad.v0_mps) <= 0.02


def test_unreachable_target_speed_is_infeasible():
    data = make_clip()
    res = _bake(data, 50.0)                                  # 单调 warp 内绝无可能
    assert not res.feasible
    assert any("unreachable" in reason for reason in res.reasons)
    assert res.out is None                                   # 连候选资产都没有


def test_unknown_keys_and_nonfinite_refused():
    data = make_clip()
    data["mystery"] = np.zeros((3, 2))
    with pytest.raises(SystemExit, match="unknown npz keys"):
        _bake(data, 1.0)
    data2 = make_clip()
    data2["joint_pos"][3, 5] = np.nan
    with pytest.raises(SystemExit, match="non-finite"):
        _bake(data2, 1.0)


def test_contact_too_close_to_edge_refused():
    data = make_clip()
    with pytest.raises(SystemExit, match="lock window"):
        bk.bake(data, 0.05, 1.0, LOOSE_VLIM, LOOSE_ACC, body_mode="interp")


def test_interp_mode_on_com_point_asset_refused():
    data = make_clip()
    data["body_lin_vel_point"] = np.asarray("center_of_mass")
    with pytest.raises(SystemExit, match="body_lin_vel"):
        _bake(data, 1.0)


# --------------------------------------------- ④ 分桶 + SHA(CLI) --------------------- #
def test_bucket_constants_pinned_to_franco_spec():
    assert bk.SPEED_RATIO_BUCKETS == {
        "train": (0.80, 1.00, 1.20),
        "interpolation": (0.90, 1.10),
        "OOD": (0.65, 1.35),
    }
    assert bk.bucket_of(0.9) == "interpolation"
    assert bk.bucket_of(1.35) == "OOD"
    assert bk.bucket_of(1.0) == "train"
    assert bk.bucket_of(0.77) == "custom"


def test_cli_bake_records_content_shas_and_buckets(tmp_path: Path):
    clip = tmp_path / "clip.npz"
    _write_clip(clip, make_clip())
    budget = tmp_path / "budget.npz"
    _loose_budget_npz(budget)
    out = tmp_path / "baked.npz"
    manifest_path = tmp_path / "baked.json"
    rc = bk.main(["--motion", str(clip), "--phase", str(PHASE), "--speed-ratio", "1.1",
                  "--out", str(out), "--manifest", str(manifest_path),
                  "--budget-clips", str(budget), "--body-mode", "interp"])
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source"]["sha256"] == _sha(clip)        # 源内容 SHA 一致
    assert manifest["output"]["sha256"] == _sha(out)         # 输出内容 SHA 一致
    assert manifest["speed_ratio_buckets"] == {
        "train": [0.80, 1.00, 1.20], "interpolation": [0.90, 1.10], "OOD": [0.65, 1.35]}
    assert manifest["speed"]["bucket"] == "interpolation"
    assert manifest["speed"]["error_frac"] <= 0.02
    assert manifest["feasibility"]["verdict"] == "feasible"
    assert len(manifest["feasibility"]["per_joint"]) == J    # 逐关节峰值 vs 限值
    assert manifest["output"]["contact_frame"] == CONTACT
    baked = dict(np.load(out, allow_pickle=False))
    assert np.array_equal(baked["joint_pos"][CONTACT], make_clip()["joint_pos"][CONTACT])


def test_cli_infeasible_writes_manifest_but_no_asset(tmp_path: Path):
    from audit_motion_npz import parse_urdf_limits
    urdf = _SCRIPTS.parents[2] / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"
    cap = parse_urdf_limits(str(urdf))[v1.ISAAC_JOINT_NAMES[24]].velocity * 0.85
    amp = 0.7 * cap * (T - 1) / FPS                          # 巡航=0.7cap:1.5×0.7>1 必超
    clip = tmp_path / "clip.npz"
    _write_clip(clip, make_clip(joint_amp=amp))
    budget = tmp_path / "budget.npz"
    _loose_budget_npz(budget)
    out = tmp_path / "baked.npz"
    manifest_path = tmp_path / "baked.json"
    with pytest.raises(SystemExit) as excinfo:
        bk.main(["--motion", str(clip), "--phase", str(PHASE), "--speed-ratio", "1.5",
                 "--out", str(out), "--manifest", str(manifest_path),
                 "--budget-clips", str(budget), "--body-mode", "interp"])
    assert excinfo.value.code == bk.EXIT_INFEASIBLE
    assert not out.exists(), "physical-infeasible 时禁止落任何资产"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["feasibility"]["verdict"] == "physical-infeasible"
    assert manifest["feasibility"]["offending"]
    assert manifest["output"] is None


def test_cli_refuses_to_overwrite(tmp_path: Path):
    clip = tmp_path / "clip.npz"
    _write_clip(clip, make_clip())
    budget = tmp_path / "budget.npz"
    _loose_budget_npz(budget)
    out = tmp_path / "baked.npz"
    out.write_bytes(b"occupied")
    with pytest.raises(SystemExit, match="拒绝覆盖"):
        bk.main(["--motion", str(clip), "--phase", str(PHASE), "--speed-ratio", "1.0",
                 "--out", str(out), "--manifest", str(tmp_path / "m.json"),
                 "--budget-clips", str(budget), "--body-mode", "interp"])
    assert out.read_bytes() == b"occupied"


# --------------------------------------------- ⑤ --envelope --------------------------- #
def test_envelope_scan_reports_max_and_shrinks_with_tighter_limits():
    data = make_clip()
    loose = bk.envelope_scan(data, PHASE, LOOSE_VLIM, LOOSE_ACC, body_mode="interp")
    cruise = 3.0 / (T - 1) * FPS
    tight_vlim = np.full(J, cruise * 1.02 / 0.85)
    tight = bk.envelope_scan(data, PHASE, tight_vlim, LOOSE_ACC, body_mode="interp")
    assert loose["max_feasible_ratio"] > 1.35                # 宽松:整条梯子都进
    assert 1.0 <= tight["max_feasible_ratio"] < 1.15         # 收紧:包络跟着缩
    assert tight["max_feasible_ratio"] < loose["max_feasible_ratio"]
    by_ratio = {row["ratio"]: row for row in tight["ladder"]}
    assert by_ratio[1.0]["feasible"] is True
    assert by_ratio[1.2]["feasible"] is False                # 梯子行如实打标
    assert by_ratio[1.2]["reasons"]
    assert {row["bucket"] for row in loose["ladder"]} == {"train", "interpolation", "OOD"}


def test_envelope_native_infeasible_fails_loud():
    data = make_clip()
    cruise = 3.0 / (T - 1) * FPS
    broken_vlim = np.full(J, cruise * 0.5 / 0.85)            # r=1 自己就超限
    with pytest.raises(SystemExit, match="native ratio"):
        bk.envelope_scan(data, PHASE, broken_vlim, LOOSE_ACC, body_mode="interp")


def test_cli_envelope_manifest_has_full_ladder(tmp_path: Path):
    clip = tmp_path / "clip.npz"
    _write_clip(clip, make_clip())
    budget = tmp_path / "budget.npz"
    _loose_budget_npz(budget)
    manifest_path = tmp_path / "envelope.json"
    rc = bk.main(["--motion", str(clip), "--phase", str(PHASE), "--envelope",
                  "--manifest", str(manifest_path), "--budget-clips", str(budget),
                  "--body-mode", "interp"])
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    ladder = manifest["envelope"]["ladder"]
    assert [row["ratio"] for row in ladder] == [0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35]
    for row in ladder:
        assert set(row) >= {"ratio", "bucket", "feasible", "target_mps", "actual_mps",
                            "vel_util_max", "acc_util_max"}
    assert manifest["envelope"]["max_feasible_ratio"] > 1.0
    assert manifest["source"]["sha256"] == _sha(clip)
    assert "output" not in manifest                          # 包络模式不出资产
