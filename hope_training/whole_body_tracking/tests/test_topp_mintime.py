"""Unit tests for scripts/topp_mintime.py (统一预算 min-time 双向重定时, TOPP v3).

Pure CPU, NO mujoco/torch:模块按文件路径加载,oracle 用确定性的桩(v2 测试同手法:
按关节速度打分——局部放慢 ⇒ |q̇|∝ṡ 下降 ⇒ 桩的"不可行"消退,和真剂量同向)。
body_mode="interp",不需要 FK。

Covered (task 2026-07-09 v3 spec ①-⑤ + 对抗复核 0709 补①②):
  ① 健康 clip 被压缩(总时长 < γ=1 基线,方向=accelerated)且不破预算
  ② 病 clip 被放慢到预算内(剂量过闸,ρ>1,时长 > 健康同款)
  ③ 触球行逐位保真 + 拍速不降(压缩/放慢两个方向都查)
  ④ 运动学硬边界(URDF 速度限位×余量)拦住过度压缩
  ⑤ 预算收紧 → min-time 时长单调不减
  ⑥ 外层梯子扫到 γ 下界含下界收尾点,best=全部可行探点的全局最短(对抗复核:
     平台期早停曾错过更短可行解 fh_v5hLs 1.66→1.50s / 守卫尾巴 fh_v4rg 2.44→2.38s)
  ⑦ 上探梯子越过 γ 上界时补探上界本身(fail-loud 文案里的 --scale-max 名副其实)

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_topp_mintime.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _name in ("synthesize_timing", "synthesize_timing_v2", "topp_mintime"):
    _spec = importlib.util.spec_from_file_location(_name, _SCRIPTS / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
v3 = sys.modules["topp_mintime"]
v2 = sys.modules["synthesize_timing_v2"]
v1 = sys.modules["synthesize_timing"]

FPS = 50.0
J = 31
NB = 32


def make_clip(T=81, contact=48, blade_step=0.02, joint_amp=3.0):
    """v2 测试同款合成 clip:一个活动关节(右肘 col 24)对 s 线性;手腕 body 沿 +x 每帧走
    blade_step m,姿态恒等 -> 源干净拍速 = blade_step*FPS = 1.0 m/s;巡航 |q̇| = 1.875 rad/s。"""
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
    return data, contact / (T - 1)


LOOSE_VLIM = np.full(J, 100.0)
LOOSE_BUDGET = np.full(J, 1000.0)


def _write_canonical_sidecar(path: Path, **overrides) -> Path:
    payload = {
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "tool_id": "canonical_schema2_builder_v1",
        "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
        "hashes": {"output_npz_sha256": v3._sha256_file(path)},
    }
    payload.update(overrides)
    sidecar = Path(str(path) + ".manifest.json")
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


def stub_oracle(vel_thresh: float):
    """桩 oracle(v2 测试同手法):max|q̇| 超过 vel_thresh 的帧记为不可行,CoP 超出量随
    超速量增长;放慢 ⇒ |q̇| 降 ⇒ flag 消退,与真剂量同向。压缩 ⇒ |q̇| 升 + 分母(帧数)缩
    ⇒ 剂量抬 —— 预算闸门天然是压缩的地板。"""
    def judge(out, stem, phase_out):
        jv = np.abs(np.asarray(out["joint_vel"], float)).max(axis=1)
        T = jv.shape[0]
        cop = np.where(jv > vel_thresh, (jv - vel_thresh) * 0.05, -0.01)
        cop[0] = cop[-1] = np.nan
        fric = np.full(T, np.nan)
        util = np.zeros(T)
        n_eval = max(T - 2, 1)
        cop_dose = float(np.sum(np.nan_to_num(cop, nan=-np.inf) > 0.0)) / n_eval
        verdict = "PASS" if cop_dose < 0.05 else ("WARN" if cop_dose < 0.20 else "FAIL")
        return v2.OracleReading(cop_excess=cop, fric_ratio=fric, util_max=util,
                                fz=np.full(T, 500.0),
                                doses={"cop": cop_dose, "friction": 0.0, "torque": 0.0},
                                verdict=verdict, contact_frame=None)
    return judge


def _run(data, phase, judge, **kw):
    kw.setdefault("body_mode", "interp")
    return v3.mintime(data, phase, kw.pop("vlim", LOOSE_VLIM),
                      kw.pop("acc_budget", LOOSE_BUDGET), judge, "clip", **kw)


# --------------------------------------------- ① 健康 clip:被压缩且不破预算 ---------- #
def test_healthy_clip_compressed_within_budget():
    data, phase = make_clip()
    out, res, law, meta = _run(data, phase, stub_oracle(vel_thresh=1e9))  # 永不 flag
    assert res.best.feasible
    assert res.best.gamma < 0.7                                  # 真的往下压了
    assert res.best.duration_s < res.gamma1.duration_s - 1.0 / FPS   # 比 γ=1 基线短
    src_dur = (data["joint_pos"].shape[0] - 1) / FPS
    assert res.best.duration_s < src_dur                         # 也比源 clip 短
    # 不破预算:oracle 剂量过闸,窗外运动学零越界
    assert res.best.reading.doses["cop"] <= v3.DEFAULT_COP_GATE
    assert res.best.breakdown["kin_bad_out_window"] == 0
    rep = v3.build_report(data, res, law, meta, "interp")
    assert rep["direction"] == "accelerated"
    assert rep["stretch"]["rho_global_min"] < 1.0                # 压缩段可见 (ρ<1)


# --------------------------------------------- ② 病 clip:被放慢进预算 ---------------- #
def test_sick_clip_slowed_into_budget():
    data, phase = make_clip()
    judge = stub_oracle(vel_thresh=1.2)          # 巡航 1.875 rad/s > 1.2 -> 基线判病
    out, res, law, meta = _run(data, phase, judge, cop_gate=0.10)
    assert res.best.feasible
    assert res.reading0.doses["cop"] > 0.10      # 基线(未修复)确实超预算
    assert res.best.reading.doses["cop"] <= 0.10  # 修进预算
    assert res.best.field.rho.max() > 1.0        # 病段确实被放慢(ρ>1)
    # 同一预算下,病动作的 min-time 比健康同款花更多时间
    out_h, res_h, _, _ = _run(data, phase, stub_oracle(vel_thresh=1e9))
    assert res.best.duration_s > res_h.best.duration_s
    rep = v3.build_report(data, res, law, meta, "interp")
    assert rep["direction"] == "slowed"


# ------------------------------- ③ 触球行逐位 + 拍速不降(两个方向都查) --------------- #
def test_contact_row_bitwise_and_blade_speed_kept():
    data, phase = make_clip()
    for judge in (stub_oracle(1e9), stub_oracle(1.2)):   # 压缩方向 & 放慢方向
        out, res, law, meta = _run(data, phase, judge)
        k = res.best.warp.k_star
        assert np.array_equal(out["joint_pos"][k], data["joint_pos"][meta["c"]])
        blade_out = v1.blade_positions(out)
        v_out = v1.clean_speed_at(blade_out, k, 1.0 / FPS)
        assert abs(v_out - meta["v_star"]) / meta["v_star"] < 0.02
        assert v_out >= meta["v_star"] * 0.98            # 拍速不降
        rep = v3.build_report(data, res, law, meta, "interp")
        assert rep["fidelity"]["contact_row_bitwise"] is True
        assert rep["fidelity"]["face_normal_diff_deg"] < 1e-3


# --------------------------------- ④ 运动学硬边界拦住过度压缩 ------------------------- #
def test_kinematic_cap_blocks_overcompression():
    data, phase = make_clip()
    judge = stub_oracle(vel_thresh=1e9)                  # oracle 永远干净,只剩运动学管着
    cruise_qdot = 3.0 / 80.0 * FPS                       # 巡航 |q̇| = 1.875 rad/s
    tight_vlim = np.full(J, cruise_qdot * 1.02 / 0.85)   # cap=URDF×0.85 只留 2% 余量
    out_t, res_t, law_t, meta_t = _run(data, phase, judge, vlim=tight_vlim)
    out_l, res_l, _, _ = _run(data, phase, judge)        # 宽松限位对照
    assert res_t.best.feasible and res_l.best.feasible
    # 限位紧 -> min-time 明显更长(压缩被硬边界拦住)
    assert res_t.best.duration_s > res_l.best.duration_s * 1.15
    # 硬边界守住:窗外没有任何一帧越过 URDF×0.85
    lw = res_t.best.field.lock_weight(res_t.best.warp.s_out)
    assert (res_t.best.vel_util[lw >= 0.5] <= 1.0 + 1e-6).all()
    assert res_t.best.breakdown["kin_bad_out_window"] == 0


def test_lock_window_hard_limit_violation_is_infeasible(monkeypatch):
    """A contact-locked violation cannot be repaired, but it can never be accepted."""

    data, phase = make_clip()
    contact_row = np.asarray(data["joint_pos"])[int(round(phase * 80))]

    def contact_only_violation(out, vel_cap, acc_budget, fps_out):
        del vel_cap, acc_budget, fps_out
        frames = len(out["joint_pos"])
        vel = np.zeros(frames)
        acc = np.zeros(frames)
        matches = np.where(np.all(np.asarray(out["joint_pos"]) == contact_row, axis=1))[0]
        assert len(matches) == 1
        vel[int(matches[0])] = 1.01
        return vel, acc

    monkeypatch.setattr(v3, "kin_utils", contact_only_violation)
    with pytest.raises(SystemExit, match="仍不可行"):
        _run(
            data,
            phase,
            stub_oracle(1e9),
            scale_min=1.0,
            scale_max=1.0,
        )


# --------------------------------- ⑤ 预算收紧 -> 时长单调不减 -------------------------- #
def test_tighter_budget_duration_monotone():
    data, phase = make_clip()
    durs = []
    for gate in (0.5, 0.15, 0.06):                       # 从松到紧
        out, res, law, meta = _run(data, phase, stub_oracle(vel_thresh=1.2),
                                   cop_gate=gate)
        assert res.best.feasible
        durs.append(res.best.duration_s)
    assert durs[1] >= durs[0] - 1e-6                     # 收紧只会更慢,绝不更快
    assert durs[2] >= durs[1] - 1e-6


def test_oracle_pass_cannot_bypass_tighter_cli_dose_gate():
    """The oracle's broad PASS band must not override this run's declared budget."""

    data, phase = make_clip()

    def broad_pass(out, stem, phase_out):
        del stem, phase_out
        frames = len(out["joint_pos"])
        return v2.OracleReading(
            cop_excess=np.full(frames, -0.01),
            fric_ratio=np.full(frames, np.nan),
            util_max=np.zeros(frames),
            fz=np.full(frames, 500.0),
            doses={"cop": 0.15, "friction": 0.0, "torque": 0.0},
            verdict="PASS",
            contact_frame=None,
        )

    # No per-frame bump can repair this intentionally adversarial aggregate reading.  The only
    # correct result is fail-closed; the old `PASS or dose<=gate` bug returned a green asset.
    with pytest.raises(SystemExit):
        _run(data, phase, broad_pass, cop_gate=0.10, scale_max=1.0)


# ---------------- ⑥ 外层全程扫描:到下界收尾点 + best=可行探点全局最短 ------------------ #
def test_outer_ladder_full_scan_global_min():
    data, phase = make_clip()
    for judge in (stub_oracle(1e9), stub_oracle(1.2)):    # 健康 & 病 两种
        out, res, law, meta = _run(data, phase, judge)
        gammas = [t["gamma"] for t in res.outer_trace]
        assert any(abs(g - v3.DEFAULT_SCALE_MIN) < 1e-9 for g in gammas), \
            "梯子越过下界后必须补探 γ=scale_min 收尾点"
        feas_durs = [t["duration_s"] for t in res.outer_trace if t["feasible"]]
        assert abs(res.best.duration_s - min(feas_durs)) < 1e-9, \
            "best 必须是全部可行探点的全局最短(平台期早停禁用)"


def test_runup_objective_selects_shortest_feasible_start_to_contact_candidate():
    data, phase = make_clip()
    out, res, law, meta = _run(
        data,
        phase,
        stub_oracle(1e9),
        objective="runup",
        scale_min=0.1,
    )
    feasible_runups = [row["runup_s"] for row in res.outer_trace if row["feasible"]]
    assert round(res.best.runup_s, 4) == min(feasible_runups)
    report = v3.build_report(data, res, law, meta, "interp")
    assert report["search_objective"] == "runup"
    assert report["timing_bound"]["strict_global_minimum_proven"] is False
    assert "upper bound" in report["timing_bound"]["bound_semantics"]
    assert np.array_equal(out["joint_pos"][0], out["joint_pos"][1])
    assert np.array_equal(out["joint_vel"][0], np.zeros_like(out["joint_vel"][0]))
    assert report["fidelity"]["first_frame_max_joint_vel"] == 0.0


# ---------------- ⑦ 上探梯子补探 scale_max 本身(fail-loud 名副其实) ------------------- #
def test_expand_ladder_probes_scale_max():
    data, phase = make_clip()
    # vel_thresh 低到 RHO_MAX 也修不干净(1.875/40 > 0.01)→ 每个 γ 都不可行 → fail loud
    with pytest.raises(SystemExit) as ei:
        _run(data, phase, stub_oracle(vel_thresh=0.01), scale_max=2.0)
    assert "γ=2.00" in str(ei.value)      # 报错声称的 --scale-max 必须真的探过


# ------------------------------------------------------------------ fail loud --------- #
def test_unknown_keys_refused():
    data, phase = make_clip()
    data["mystery"] = np.zeros((3, 2))
    with pytest.raises(SystemExit):
        _run(data, phase, stub_oracle(1e9))


def test_bad_search_params_refused():
    data, phase = make_clip()
    with pytest.raises(SystemExit):
        _run(data, phase, stub_oracle(1e9), compress_step=1.2)   # 步长必须 <1
    with pytest.raises(SystemExit):
        _run(data, phase, stub_oracle(1e9), scale_min=1.5)       # 下界必须 ≤1
    with pytest.raises(SystemExit):
        _run(data, phase, stub_oracle(1e9), objective="not-an-objective")


def test_production_cli_refuses_interp_before_reading_inputs(tmp_path: Path):
    with pytest.raises(SystemExit, match="必须 --body-mode fk"):
        v3.main(
            [
                "--input",
                str(tmp_path / "missing.npz"),
                "--output",
                str(tmp_path / "out.npz"),
                "--phase",
                "0.5",
                "--budget-clips",
                str(tmp_path / "missing-budget.npz"),
                "--body-mode",
                "interp",
                "--report",
                str(tmp_path / "report.json"),
                "--md",
                str(tmp_path / "report.md"),
            ]
        )


def test_canonical_v2_cli_requires_explicit_legacy_comparator_flag(tmp_path: Path):
    data, phase = make_clip()
    source = tmp_path / "source.npz"
    budget = tmp_path / "budget.npz"
    np.savez(source, **data)
    np.savez(budget, **data)
    _write_canonical_sidecar(source)
    common = [
        "--input",
        str(source),
        "--output",
        str(tmp_path / "out.npz"),
        "--phase",
        str(phase),
        "--budget-clips",
        str(budget),
        "--report",
        str(tmp_path / "report.json"),
        "--md",
        str(tmp_path / "report.md"),
    ]
    with pytest.raises(SystemExit, match="canonical-v2 candidate/adopted/comparator"):
        v3.main(common)
    # The explicit escape hatch passes the lineage gate; this focused CLI test
    # intentionally stops at the next production-only prerequisite.
    with pytest.raises(SystemExit, match="--mjcf and --body-order"):
        v3.main([*common, "--allow-canonical-v2-legacy-comparator"])
    receipt = v3._legacy_comparator_guard(
        [("source", source)],
        allow_canonical_v2_legacy_comparator=True,
    )
    marked = v3._mark_legacy_comparator({}, receipt)
    assert marked["publication_class"] == v3.LEGACY_COMPARATOR_CLASS
    assert marked["legacy_semantics"] == v3.LEGACY_COMPARATOR_CLASS
    assert marked["training_authorized"] is False
    assert marked["hardware_authorized"] is False
    _, result, law, meta = _run(data, phase, stub_oracle(1e9))
    report = v3.build_report(data, result, law, meta, "interp")
    v3._mark_legacy_comparator(report, receipt)
    assert v3.LEGACY_COMPARATOR_CLASS in v3.report_md(report)


def test_guard_is_strict_not_filename_based(tmp_path: Path):
    filename_only = tmp_path / "looks_canonical_v2.npz"
    filename_only.write_bytes(b"legacy")
    assert (
        v3._legacy_comparator_guard(
            [("filename-only legacy", filename_only)],
            allow_canonical_v2_legacy_comparator=False,
        )
        is None
    )

    duplicate = Path(str(filename_only) + ".manifest.json")
    duplicate.write_text(
        '{"publication_class":"compiler_candidate",'
        '"publication_class":"compiler_candidate",'
        '"tool_id":"canonical_schema2_builder_v1",'
        '"build_verdict":"PASS_COMPILER_CANDIDATE_ONLY"}',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="strict JSON"):
        v3._legacy_comparator_guard(
            [("duplicate sidecar", filename_only)],
            allow_canonical_v2_legacy_comparator=True,
        )

    registry_bound = tmp_path / "registry-bound.npz"
    registry_bound.write_bytes(b"canonical")
    registry_payload = {
        "publication_class": "training_adopted",
        "training_authorized": True,
        "deployment_authorized": False,
        "hardware_authorized": False,
        "tool_id": "canonical_schema2_builder_v1",
        "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
        "npz_sha256": v3._sha256_file(registry_bound),
    }
    Path(str(registry_bound) + ".registry.json").write_text(
        json.dumps(registry_payload),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="canonical-v2 candidate/adopted/comparator"):
        v3._legacy_comparator_guard(
            [("registry-bound canonical", registry_bound)],
            allow_canonical_v2_legacy_comparator=False,
        )


def test_guard_rejects_symlink_and_contradictory_canonical_state(tmp_path: Path):
    target = tmp_path / "target.npz"
    target.write_bytes(b"legacy")
    link = tmp_path / "link.npz"
    link.symlink_to(target)
    with pytest.raises(SystemExit, match="禁止 symlink"):
        v3._legacy_comparator_guard(
            [("symlink input", link)],
            allow_canonical_v2_legacy_comparator=True,
        )

    sidecar_link_input = tmp_path / "sidecar-link.npz"
    sidecar_link_input.write_bytes(b"canonical")
    sidecar_target = tmp_path / "sidecar-target.json"
    sidecar_target.write_text("{}", encoding="utf-8")
    Path(str(sidecar_link_input) + ".manifest.json").symlink_to(sidecar_target)
    with pytest.raises(SystemExit, match="禁止 symlink"):
        v3._legacy_comparator_guard(
            [("symlink sidecar", sidecar_link_input)],
            allow_canonical_v2_legacy_comparator=True,
        )

    conflict = tmp_path / "conflict.npz"
    conflict.write_bytes(b"canonical")
    _write_canonical_sidecar(conflict, training_authorized=True)
    with pytest.raises(SystemExit, match="状态机矛盾"):
        v3._legacy_comparator_guard(
            [("conflicting canonical input", conflict)],
            allow_canonical_v2_legacy_comparator=True,
        )


def test_certificate_bundle_publish_is_no_replace_and_rolls_back_partial(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    sources = []
    for name, payload in (("a", b"one"), ("b", b"two"), ("c", b"three")):
        path = stage / name
        path.write_bytes(payload)
        sources.append(path)
        evidence = v3._file_evidence(path, name)
        assert evidence["bytes"] == len(payload)
        assert len(evidence["sha256"]) == 64

    destinations = [tmp_path / f"out-{index}" for index in range(3)]
    v3._publish_staged_bundle(list(zip(sources, destinations)), parent=tmp_path)
    assert [path.read_bytes() for path in destinations] == [b"one", b"two", b"three"]

    rollback_first = tmp_path / "rollback-first"
    occupied = tmp_path / "occupied"
    occupied.write_bytes(b"immutable")
    with pytest.raises(FileExistsError):
        v3._publish_staged_bundle(
            [(sources[0], rollback_first), (sources[1], occupied)], parent=tmp_path
        )
    assert not rollback_first.exists()
    assert occupied.read_bytes() == b"immutable"
