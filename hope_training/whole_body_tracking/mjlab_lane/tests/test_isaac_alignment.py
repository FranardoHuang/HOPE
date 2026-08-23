#!/usr/bin/env python3
"""Mutation tests for the mjlab-vs-Isaac alignment ledger.

人话
----
这组测试要证明的不是"台账能跑",而是**台账真的在比活值**。

所以每个变异都刻意做成"**粗一个档次的检查会照样通过**"的形状:
交换两个关节的 Kp(31 个数的和、排序后的多重集、个数全都不变)、
交换两条 ``<motor>`` 的 ctrlrange(同上)、
把 Isaac 的值藏在 ``__post_init__`` 里(只扫模块级赋值的读法看不见它)。
每个变异测试都**先断言粗检查确实过得去**,再断言台账当场红。

变异全部在子进程里跑,树是临时拷贝出来的:本仓一个字节不动,
而且被拷出来的那棵树同时复现了 pod 上那种"车道旁边自带一份 geometry.py"的部署形状。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

LANE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LANE_DIR))

import isaac_alignment as align  # noqa: E402


REPO_ROOT = align.resolve_repo_root()
needs_repo = pytest.mark.skipif(
    REPO_ROOT is None,
    reason="no repo checkout above this lane; the Isaac side is unreachable")

#: Lane modules a temp copy needs in order to import.  `geometry.py` is added
#: separately -- putting it next to the lane is exactly the pod deployment
#: shape the provenance axis exists to catch.
LANE_FILES = (
    "isaac_alignment.py",
    "a3_train_ppo.py",
    "a3_court_env.py",
    "a3_plant_env.py",
    "calibrate_restitution.py",
    "mujoco_gpu_ac_full_mdp_initial_wait_env.py",
    "mujoco_gpu_ac_full_mdp_wait_rsl3.py",
)


# ---------------------------------------------------------------------------
# Temp-tree harness.
# ---------------------------------------------------------------------------


def _build_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the lane and every live Isaac source into a throwaway tree."""

    lane = tmp_path / "lane"
    repo = tmp_path / "repo"
    lane.mkdir(parents=True)
    for name in LANE_FILES:
        shutil.copy2(LANE_DIR / name, lane / name)
    shutil.copy2(
        REPO_ROOT / align.ISAAC_SOURCE_RELPATHS["qdes_guard"],
        lane / "action_ball_qdes_guard.py",
    )
    for name, rel in align.ISAAC_SOURCE_RELPATHS.items():
        src = REPO_ROOT / rel
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # The lane resolves geometry next to itself first; give it that copy so the
    # temp tree behaves like the deployed one.
    shutil.copy2(REPO_ROOT / align.ISAAC_SOURCE_RELPATHS["geometry"],
                 lane / "geometry.py")
    return lane, repo


_RUNNER = textwrap.dedent("""
    import json, sys
    import isaac_alignment as align
    try:
        ledger = align.build_ledger()
    except align.AlignmentError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(0)
    print(json.dumps({"ok": True,
                      "counts": ledger["verdict_counts"],
                      "blocking": ledger["blocking_axes"],
                      "observed": {k: r["observed"]
                                   for k, r in ledger["rows"].items()}}))
""")


def _run_ledger(lane: Path, repo: Path) -> dict:
    env = dict(os.environ)
    env["HOPE_REPO_ROOT"] = str(repo)
    env.pop("HOPE_GEOMETRY_PY", None)
    env["PYTHONPATH"] = str(lane) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _RUNNER], cwd=str(lane),
                          env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"runner crashed: {proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


_ACTION_RUNNER = textwrap.dedent("""
    import json, sys
    import isaac_alignment as align
    try:
        sources = align.isaac_sources()
        if sources["missing"]:
            raise align.AlignmentError("missing action-axis sources")
        observed = {}
        for axis in align.AXES:
            if axis.key not in ("raw_action_affine", "executable_qdes_guard"):
                continue
            row = axis.probe(sources["paths"], {})
            if row["observed"] != axis.declared:
                raise align.AlignmentError(
                    f"{axis.key} declared {axis.declared} but observed "
                    f"{row['observed']}")
            observed[axis.key] = row
    except align.AlignmentError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(0)
    print(json.dumps({"ok": True, "observed": observed}))
""")


def _run_action_axes(lane: Path, repo: Path) -> dict:
    env = dict(os.environ)
    env["HOPE_REPO_ROOT"] = str(repo)
    env["OMP_NUM_THREADS"] = "1"
    env["KMP_USE_SHM"] = "0"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env.pop("HOPE_GEOMETRY_PY", None)
    env["PYTHONPATH"] = str(lane) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _ACTION_RUNNER], cwd=str(lane),
                          env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"runner crashed: {proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _patch(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, (
        f"mutation anchor is not unique in {path.name}: {text.count(old)} hits")
    path.write_text(text.replace(old, new), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. The un-mutated tree: every declared verdict matches the live one.
# ---------------------------------------------------------------------------


@needs_repo
def test_ledger_builds_and_every_declared_verdict_matches_live():
    ledger = align.build_ledger()
    assert ledger["n_axes"] == len(align.AXES)
    for key, row in ledger["rows"].items():
        assert row["observed"] == row["declared"], (
            f"{key}: declared {row['declared']} but live says {row['observed']}")
    assert ledger["verdict_counts"][align.UNVERIFIABLE] == 0
    # This lane is NOT asking the Isaac question today.  If that ever becomes
    # false, this assertion is the thing that makes somebody notice and come
    # rewrite the section instead of quietly inheriting a stale claim.
    assert ledger["blocking_axes"], "no blocking axes -- re-read EXP 9.2.9"
    assert ledger["cross_engine_comparable"] is False


@needs_repo
def test_temp_tree_reproduces_the_real_verdicts(tmp_path):
    lane, repo = _build_tree(tmp_path)
    result = _run_ledger(lane, repo)
    assert result["ok"] is True, result.get("error")
    real = align.build_ledger()
    assert result["observed"] == {k: r["observed"] for k, r in real["rows"].items()}


# ---------------------------------------------------------------------------
# 2. Mutation: swap two joints' Kp.  Sum, multiset and count are all preserved.
# ---------------------------------------------------------------------------


@needs_repo
def test_swapping_two_kp_values_is_invisible_to_every_coarser_check():
    """Prove the mutation is genuinely a hard one before using it."""

    import a3_plant_env as plant

    joints = align.mjcf_actuated_joints(
        {"vendor_mjcf": REPO_ROOT / align.ISAAC_SOURCE_RELPATHS["vendor_mjcf"]})
    before, _ = plant.vendor_pd_for_joint_names(joints)
    original = dict(plant.VENDOR_KP)
    try:
        plant.VENDOR_KP["shoulder_yaw"], plant.VENDOR_KP["wrist_pitch"] = (
            original["wrist_pitch"], original["shoulder_yaw"])
        after, _ = plant.vendor_pd_for_joint_names(joints)
    finally:
        plant.VENDOR_KP.clear()
        plant.VENDOR_KP.update(original)

    assert sum(before) == sum(after), "sum would have caught it -- pick a harder swap"
    assert sorted(before) == sorted(after), "the sorted multiset would have caught it"
    assert len(before) == len(after)
    assert list(before) != list(after), "the swap did nothing at all"


@needs_repo
def test_mutation_swapped_kp_flips_the_pd_gains_axis(tmp_path):
    lane, repo = _build_tree(tmp_path)
    clean = _run_ledger(lane, repo)
    assert clean["observed"]["pd_gains"] == align.ALIGNED

    plant_py = lane / "a3_plant_env.py"
    _patch(plant_py, '"shoulder_yaw": 30.0', '"shoulder_yaw": 20.0')
    _patch(plant_py, '"wrist_pitch": 20.0, "wrist_yaw": 20.0,\n    "hip_yaw"',
           '"wrist_pitch": 30.0, "wrist_yaw": 20.0,\n    "hip_yaw"')

    mutated = _run_ledger(lane, repo)
    assert mutated["ok"] is False, "the swapped Kp table was accepted"
    assert "pd_gains" in mutated["error"]


# ---------------------------------------------------------------------------
# 3. Mutation: swap two <motor> ctrlrange values in the vendor MJCF.
#    Same trick -- sum and multiset over the 31 motors are unchanged.
# ---------------------------------------------------------------------------


@needs_repo
def test_mutation_swapped_ctrlrange_flips_the_effort_limits_axis(tmp_path):
    lane, repo = _build_tree(tmp_path)
    mjcf = repo / align.ISAAC_SOURCE_RELPATHS["vendor_mjcf"]
    paths = {"vendor_mjcf": mjcf}
    before = align.mjcf_ctrlrange(paths)

    _patch(mjcf,
           'name="left_shoulder_yaw_joint_motor" joint="left_shoulder_yaw_joint" ctrlrange="-24 24"',
           'name="left_shoulder_yaw_joint_motor" joint="left_shoulder_yaw_joint" ctrlrange="-6 6"')
    _patch(mjcf,
           'name="left_wrist_pitch_joint_motor" joint="left_wrist_pitch_joint" ctrlrange="-6 6"',
           'name="left_wrist_pitch_joint_motor" joint="left_wrist_pitch_joint" ctrlrange="-24 24"')

    after = align.mjcf_ctrlrange(paths)
    assert sum(before.values()) == sum(after.values()), "sum would have caught it"
    assert sorted(before.values()) == sorted(after.values()), "multiset would have"
    assert before != after

    mutated = _run_ledger(lane, repo)
    assert mutated["ok"] is False
    assert "effort_limits" in mutated["error"]


# ---------------------------------------------------------------------------
# 3b. Raw action reachability: deterministic artifact math, never rollout.
# ---------------------------------------------------------------------------


@needs_repo
def test_legacy_raw_clip_makes_five_live_slot0_frame0_joints_unreachable():
    ready = json.loads((REPO_ROOT / (
        "configs/action_ball_n1_measured_20260803/"
        "evidence_holdpass_robust20n_20260803/"
        "take061.measured_teacher.yaw_aligned_full_seed."
        "robust20n.dynamic_ready.v2.json")).read_text())
    manifest = json.loads((REPO_ROOT /
        "configs/action_ball_chingmu73_measured_a3p0807_f10_20260819.json").read_text())
    slot0 = manifest["actions"][0]
    with np.load(REPO_ROOT / slot0["motion_path"], allow_pickle=False) as motion:
        joint_pos = np.asarray(motion["joint_pos"])
        frame0 = joint_pos[0]
    plant = ready["runtime_plant"]
    action_offset = np.asarray(plant["default_joint_pos_rad"])
    scale = np.asarray(plant["action_scale_rad"])
    required = (frame0 - action_offset) / scale
    required_all = (joint_pos - action_offset) / scale
    indices = np.flatnonzero(np.abs(required) > 4.0)
    assert slot0["action_id"] == "take_058_unit02_fh"
    assert np.all(frame0 >= plant["executed_qdes_lower_rad"])
    assert np.all(frame0 <= plant["executed_qdes_upper_rad"])
    assert [plant["joint_names"][index] for index in indices] == [
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]
    assert required[26] > 10.0 and required[30] < -15.0
    assert np.all(np.any(np.abs(required_all) > 4.0, axis=1))
    assert float(np.max(np.abs(required_all))) > 21.0


@needs_repo
def test_affine_offset_is_runtime_default_not_physical_reset_pose():
    ready = json.loads((REPO_ROOT / align.ISAAC_SOURCE_RELPATHS[
        "ready_pose"]).read_text())
    plant, hold = ready["runtime_plant"], ready["hold_candidate"]
    offset = np.asarray(plant["default_joint_pos_rad"])
    physical = np.asarray(ready["physical_ready"]["joint_pos_rad"])
    scale = np.asarray(plant["action_scale_rad"])
    expected_raw = (np.asarray(hold["hold_qdes_joint_pos_rad"]) - offset) / scale
    assert np.count_nonzero(offset != physical) == 14
    assert np.max(np.abs(offset - physical)) == pytest.approx(1.5199244618415833)
    np.testing.assert_array_equal(
        np.asarray(hold["normalized_actor_action"]), expected_raw)


@needs_repo
def test_mutation_reintroducing_raw_clip_flips_the_action_contract_axis(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(lane / "a3_train_ppo.py",
           "  action_clip: object = None",
           "  action_clip: object = 4.0")
    mutated = _run_action_axes(lane, repo)
    assert mutated["ok"] is False
    assert "raw_action_affine" in mutated["error"]


@needs_repo
def test_mutation_hardcoded_raw_clip_in_advance_path_is_refused(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(lane / "a3_train_ppo.py",
           "    incoming = actions.to(self.device)",
           "    incoming = torch.clamp(actions.to(self.device), -4.0, 4.0)")
    mutated = _run_action_axes(lane, repo)
    assert mutated["ok"] is False
    assert "raw_action_affine" in mutated["error"]


@needs_repo
def test_mutation_hardcoded_raw_clip_in_full_a_wrapper_is_refused(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(
        lane / "mujoco_gpu_ac_full_mdp_initial_wait_env.py",
        "        scheduled_due_event, launch_event, missed_launch_event = (\n"
        "            self._full_a_prepare_step()\n"
        "        )\n"
        "        _st_before_forward, _tau_sq, requested_qdes = "
        "self._advance_plant(actions)",
        "        scheduled_due_event, launch_event, missed_launch_event = (\n"
        "            self._full_a_prepare_step()\n"
        "        )\n"
        "        _st_before_forward, _tau_sq, requested_qdes = "
        "self._advance_plant(torch.clamp(actions, -4.0, 4.0))",
    )
    mutated = _run_action_axes(lane, repo)
    assert mutated["ok"] is False
    assert "raw_action_affine" in mutated["error"]


@needs_repo
def test_mutation_active_isaac_wrapper_clip_is_refused(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(
        repo / align.ISAAC_SOURCE_RELPATHS["train_entry"],
        "RslRlVecEnvWrapper(env)",
        "RslRlVecEnvWrapper(env, 4.0)",
    )
    mutated = _run_action_axes(lane, repo)
    assert mutated["ok"] is False
    assert "raw_action_affine" in mutated["error"]


@needs_repo
def test_focused_action_axes_match_their_honest_verdicts(tmp_path):
    lane, repo = _build_tree(tmp_path)
    result = _run_action_axes(lane, repo)
    assert result["ok"] is True, result.get("error")
    raw = result["observed"]["raw_action_affine"]
    guard = result["observed"]["executable_qdes_guard"]
    assert raw["observed"] == align.ALIGNED
    assert set(raw["components"]) == {
        "raw_policy_action", "runtime_joint_order", "scale", "offset"}
    assert guard["observed"] == align.ALIGNED
    assert guard["isaac"] == guard["mjlab"]


@needs_repo
def test_mutation_final_ctrl_wrong_order_is_refused(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(lane / "a3_train_ppo.py",
           "      d.ctrl[:] = tau[:, self.actuator_from_runtime]",
           "      d.ctrl[:] = tau")
    result = _run_action_axes(lane, repo)
    assert result["ok"] is False
    assert "raw_action_affine" in result["error"]


@needs_repo
def test_mutation_vendor_scale_factor_is_refused(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(lane / "a3_train_ppo.py",
           "self.act_scale = 0.25 * self.tau_hi / self.kp",
           "self.act_scale = 0.5 * self.tau_hi / self.kp")
    result = _run_action_axes(lane, repo)
    assert result["ok"] is False
    assert "raw_action_affine" in result["error"]


@needs_repo
def test_mutation_physical_ready_as_affine_offset_is_refused(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(lane / "a3_train_ppo.py",
           "pre_clamp_qdes = self.action_offset.unsqueeze(0) + self.act_scale * incoming",
           "pre_clamp_qdes = self.q_ready.unsqueeze(0) + self.act_scale * incoming")
    result = _run_action_axes(lane, repo)
    assert result["ok"] is False
    assert "raw_action_affine" in result["error"]


@needs_repo
def test_mutation_deleted_brake_hidden_in_string_is_refused(tmp_path):
    lane, repo = _build_tree(tmp_path)
    actions = repo / align.ISAAC_SOURCE_RELPATHS["hope_actions"]
    _patch(
        actions,
        "                crossing_violation = guard.crossing_violation",
        "                crossing_violation = torch.zeros_like(\n"
        "                    guard.crossing_violation\n"
        "                )",
    )
    result = _run_action_axes(lane, repo)
    assert result["ok"] is False
    assert "executable_qdes_guard" in result["error"]


@needs_repo
def test_mutation_mujoco_qdes_guard_dt_is_refused(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(
        lane / "a3_train_ppo.py",
        "        policy_dt_s=0.02,",
        "        policy_dt_s=0.04,",
    )
    result = _run_action_axes(lane, repo)
    assert result["ok"] is False
    assert "executable_qdes_guard" in result["error"]


@needs_repo
def test_mutation_declaring_the_live_guard_divergent_is_refused(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(
        lane / "isaac_alignment.py",
        "    return _row(shared, dict(shared), ALIGNED)",
        "    return _row(shared, dict(shared), DIVERGENT_DECLARED)",
    )
    result = _run_action_axes(lane, repo)
    assert result["ok"] is False
    assert "declared aligned but observed divergent_declared" in result["error"]


# ---------------------------------------------------------------------------
# 4. Mutation: the Isaac control rate, which lives inside `__post_init__`.
#    A module-level-only reader is blind to it -- assert that, then assert the
#    ledger is not.
# ---------------------------------------------------------------------------


@needs_repo
def test_isaac_control_rate_is_invisible_to_a_module_level_reader():
    path = REPO_ROOT / align.ISAAC_SOURCE_RELPATHS["tracking_env_cfg"]
    tree = align._parse(path)
    consts = align._module_consts(tree)
    assert "decimation" not in consts and "sim" not in consts
    attrs = align._post_init_self_attrs(tree, "TrackingEnvCfg")
    assert attrs["decimation"] == 4 and attrs["sim.dt"] == 0.005


@needs_repo
def test_mutation_isaac_physics_dt_flips_the_control_rate_axis(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(repo / align.ISAAC_SOURCE_RELPATHS["tracking_env_cfg"],
           "self.sim.dt = 0.005", "self.sim.dt = 0.004")
    mutated = _run_ledger(lane, repo)
    assert mutated["ok"] is False
    assert "control_rate" in mutated["error"]


# ---------------------------------------------------------------------------
# 5. Mutation: the vendor MJCF <option> timestep -- Franco's inheritance rule.
# ---------------------------------------------------------------------------


@needs_repo
def test_mutation_vendor_timestep_flips_the_plant_inheritance_axis(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(repo / align.ISAAC_SOURCE_RELPATHS["vendor_mjcf"],
           'timestep="0.001"', 'timestep="0.002"')
    mutated = _run_ledger(lane, repo)
    assert mutated["ok"] is False
    assert "vendor_plant_inheritance" in mutated["error"]


# ---------------------------------------------------------------------------
# 6. Mutation: the lane's own geometry.py copy drifts from the repo's.
#    This is the pod deployment shape: /workspace/mjlab_lane carries a byte
#    copy that the resolver prefers, with no digest check anywhere.
# ---------------------------------------------------------------------------


@needs_repo
def test_mutation_stale_geometry_copy_flips_the_provenance_axis(tmp_path):
    lane, repo = _build_tree(tmp_path)
    clean = _run_ledger(lane, repo)
    assert clean["observed"]["geometry_provenance"] == align.ALIGNED

    # Byte-level drift with IDENTICAL semantics: one appended comment.  Even
    # this must flip the row -- "the lane loaded a copy" is the defect, not
    # "the copy happened to say something different".  A check that compared
    # only the values the lane happens to read today would sail past it.
    with open(lane / "geometry.py", "a", encoding="utf-8") as handle:
        handle.write("\n# mutation: this copy is no longer the repo's file\n")
    assert (lane / "geometry.py").read_text(encoding="utf-8").count(
        "TABLE_WIDTH") == (
        (REPO_ROOT / align.ISAAC_SOURCE_RELPATHS["geometry"])
        .read_text(encoding="utf-8").count("TABLE_WIDTH")), (
        "the mutation was supposed to be semantics-preserving")

    mutated = _run_ledger(lane, repo)
    assert mutated["ok"] is False
    assert "geometry_provenance" in mutated["error"]


# ---------------------------------------------------------------------------
# 7. Enumeration guards: a new upstream termination / a renamed reward anchor /
#    an unclassified lane knob must all fail closed.
# ---------------------------------------------------------------------------


@needs_repo
def test_mutation_new_isaac_termination_is_refused_until_classified(tmp_path):
    lane, repo = _build_tree(tmp_path)
    _patch(repo / align.ISAAC_SOURCE_RELPATHS["hope_env_cfg"],
           "    base_fell_tilt = DoneTerm(func=mdp.bad_orientation, "
           'params={"limit_angle": 0.7})',
           "    base_fell_tilt = DoneTerm(func=mdp.bad_orientation, "
           'params={"limit_angle": 0.7})\n'
           "    invented_guard = DoneTerm(func=mdp.bad_orientation, "
           'params={"limit_angle": 0.9})')
    mutated = _run_ledger(lane, repo)
    assert mutated["ok"] is False
    assert "invented_guard" in mutated["error"]
    assert "ISAAC_TO_MJLAB_TERMINATION" in mutated["error"]


@needs_repo
def test_mutation_removed_reward_anchor_is_refused(tmp_path):
    """The strike-guidance anchor is declared exactly once in the whole chain.

    Renaming a term that a PARENT class also declares would legitimately leave
    the anchor resolvable, so the mutation has to target a term with no shadow
    -- otherwise the test would be asserting the guard is broken.
    """

    lane, repo = _build_tree(tmp_path)
    hope_cfg = repo / align.ISAAC_SOURCE_RELPATHS["hope_env_cfg"]
    anchor = "    c225_strike_ball_paddle_center_proximity = RewTerm("
    assert hope_cfg.read_text(encoding="utf-8").count(anchor) == 1
    _patch(hope_cfg, anchor,
           "    c225_strike_ball_paddle_center_proximity_renamed = RewTerm(")
    mutated = _run_ledger(lane, repo)
    assert mutated["ok"] is False
    assert "c225_strike_ball_paddle_center_proximity" in mutated["error"]


@needs_repo
def test_mutation_unwiring_the_table_refusal_is_caught_by_the_ledger(tmp_path):
    """The ledger claims the lane blocks on robot-table contact.  Prove it checks.

    Deleting the refusal leaves the counter, the receipt field and the docstring
    all intact -- the classic "a counter nobody reads" shape.  Only a check that
    looks for the BLOCK, not for the measurement, notices.
    """

    lane, repo = _build_tree(tmp_path)
    clean = _run_ledger(lane, repo)
    assert clean["ok"] is True

    _patch(lane / "a3_train_ppo.py",
           '        "ROBOT_LEANED_ON_THE_TABLE",',
           '        "ROBOT_MERELY_NOTED_AS_A_STATISTIC",')
    mutated = _run_ledger(lane, repo)
    assert mutated["ok"] is False
    assert "robot-vs-table" in mutated["error"]
    assert "refusal_wired" in mutated["error"]


@needs_repo
def test_an_unclassified_lane_knob_fails_closed(monkeypatch):
    lane = align.mjlab_side()
    object.__setattr__(lane["task"], "invented_knob", 1.0)
    assert "invented_knob" in align.unclassified_lane_fields(lane)
    with pytest.raises(align.AlignmentError, match="invented_knob"):
        align.build_ledger(lane=lane)


@needs_repo
def test_a_removed_lane_knob_is_reported_as_stale():
    lane = align.mjlab_side()
    delattr(lane["task"], "action_scale_mode")
    assert "action_scale_mode" in align.stale_classifications(lane)
    with pytest.raises(align.AlignmentError, match="no longer exist"):
        align.build_ledger(lane=lane)


# ---------------------------------------------------------------------------
# 8. The claim gate.
# ---------------------------------------------------------------------------


def test_bitwise_parity_is_refused_even_with_a_perfectly_clean_ledger():
    """Item (e): a bit-exact cross-engine acceptance is wrong, not strict."""

    spotless = {"blocking_axes": [], "cross_engine_comparable": True}
    with pytest.raises(align.AlignmentClaimRefused, match="no CPU fallback"):
        align.assert_cross_engine_claim(spotless, align.CLAIM_BITWISE_PARITY)


def test_comparability_is_refused_while_any_axis_blocks():
    blocked = {"blocking_axes": ["actor_observation_abi"],
               "cross_engine_comparable": False}
    with pytest.raises(align.AlignmentClaimRefused, match="actor_observation_abi"):
        align.assert_cross_engine_claim(blocked,
                                        align.CLAIM_CROSS_ENGINE_COMPARABLE)
    align.assert_cross_engine_claim({"blocking_axes": []},
                                    align.CLAIM_CROSS_ENGINE_COMPARABLE)


def test_aligned_guard_adds_no_special_transfer_restriction():
    ledger = {
        "blocking_axes": [],
        "rows": {"executable_qdes_guard": {"observed": align.ALIGNED}},
        "mujoco_only_diagnostic_25k_blocked_by_alignment": False,
    }
    assert ledger["mujoco_only_diagnostic_25k_blocked_by_alignment"] is False
    for claim in (align.CLAIM_POLICY_TRANSFER,
                  align.CLAIM_MATCHED_CAUSAL_COMPARISON):
        align.assert_cross_engine_claim(ledger, claim)


def test_unknown_claims_are_refused_rather_than_ignored():
    with pytest.raises(align.AlignmentClaimRefused):
        align.assert_cross_engine_claim({"blocking_axes": []}, "vibes")


# ---------------------------------------------------------------------------
# 9. The lane's own declared ABI has to match the code that builds it.
# ---------------------------------------------------------------------------


def _lane_module(name: str):
    spec = importlib.util.spec_from_file_location(name, LANE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_obs_layout_names_match_the_rows_compute_obs_actually_builds():
    """Parsed from source, so this holds on a host with no torch and no GPU."""

    train = _lane_module("a3_train_ppo")
    tree = ast.parse((LANE_DIR / "a3_train_ppo.py").read_text(encoding="utf-8"))
    built = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_obs":
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Assign) and len(inner.targets) == 1
                        and getattr(inner.targets[0], "id", None) == "rows"
                        and isinstance(inner.value, ast.Dict)):
                    built = [k.value for k in inner.value.keys]
    assert built is not None, "_compute_obs no longer builds a named `rows` dict"
    assert built == [name for name, _w in train.OBS_LAYOUT]
    assert train.OBS_WIDTH == sum(w for _n, w in train.OBS_LAYOUT) == 114


def test_reward_group_registry_covers_exactly_the_priced_terms():
    train = _lane_module("a3_train_ppo")
    assert set(train.REWARD_TERM_GROUP) == set(
        train.reward_term_ceilings(train.TaskCfg()))
    assert set(train.REWARD_TERMS) == set(train.REWARD_TERM_GROUP)


def test_receipt_refusal_fires_on_a_self_contradicting_receipt():
    train = _lane_module("a3_train_ppo")
    receipt = {
        "status": "completed",
        "capacity": {"verdict": "PASS_NO_OVERFLOW"},
        "learning": {"binary_contact_rate": {"measured": True}},
        "isaac_alignment": {"cross_engine_comparable": True,
                            "blocking_axes": ["reward_surface"]},
    }
    codes = [c for c, _w in train.report_refusals(
        [("r0", receipt), ("r1", receipt)], None)]
    assert "CLAIMS_ISAAC_COMPARABILITY_WITHOUT_EARNING_IT" in codes


def test_receipt_without_a_ledger_is_scoped_not_silently_blessed():
    train = _lane_module("a3_train_ppo")
    old = {"status": "completed"}
    scope = train._report_alignment_scope([("legacy", old)])
    assert scope["per_receipt"]["legacy"]["ledger"] is None
    assert scope["every_receipt_is_cross_engine_comparable"] is False
    assert "UNRECORDED" in scope["per_receipt"]["legacy"]["note"]
