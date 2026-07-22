"""集成升级波三件套测试（phase1_integrated_upgrade_wave_20260723）。

Pinned here（结构照 test_run_phase1_chatter_ground_foot_wave_queue.py，覆盖本波特有纪律）：

* 入库 YAML 通过全量校验；治理旗标冻结（simulation_only 等）；预算冻结
  （science 13301 -> 绝对里程碑到 20000；combo_fresh fresh 独立 20001）。
* 3 条 combo 臂冻结（Franco 07-23 变更已并入）：action_rate 统一 -0.2；全程高摩擦
  静[1.0,1.6]/动[0.8,1.2]；落地罚 -3e-3 @300 N；抬脚罚 -0.01 @0.15 m 只进 combo_fresh；
  二阶平滑 -0.05（源码未接线，action_acc 闸门锁全波）；qbar barrier -0.65/0.08 与
  action_rate 并存；速度推 w_p035 + 力推 w_f035 两组事件并存、键面逐字。
* reward 比值守卫：击球组 17/7/5/5/10 一动不动（strike_success/progress 禁 override）；
  软惩罚组全额不叠加（蹭滑/拖脚/挥拍前脚滑键一个不带）。
* combo_fresh fresh 铁律：命令绝不带 checkpoint 键；预算走 science_fresh。
* 闸门：groundfoot/push/force_push/qbar/action_acc 任一 false 锁全部三臂；
  franco_contract（false 或资产占位符）只锁 combo_franco——combo_franco 换
  motion_file_2 = franco_bh_loop_b，另两臂必须用 v4rg 反手。
* 渲染注入 pod/gpu 校验、shell 引号往返、kit_boot_lock 而非 locked launcher、
  no-clobber、launch_order = franco->resume->fresh 冻结。

Run:  python -m pytest tests/test_run_phase1_integrated_upgrade_wave_queue.py -q
"""

from __future__ import annotations

import copy
import importlib.util
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts/run_phase1_integrated_upgrade_wave_queue.py"
QUEUE_PATH = REPO / "configs/phase1_integrated_upgrade_wave_20260723.yaml"

PINNED_COMMIT = "ad0110e88cf9c481247b0d554430d5585f13bcd2"
DELIVERED_FRANCO_NPZ = (
    "/workspace/codexschema/franco_pipeline_20260722/assets/"
    "franco_bh_loop_b_schema2_cal.npz"
)


def _module():
    spec = importlib.util.spec_from_file_location("iu_wave_queue", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _module()


def _raw() -> dict:
    return yaml.safe_load(QUEUE_PATH.read_text(encoding="utf-8"))


def _write_yaml(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _load(tmp_path: Path, value: dict) -> dict:
    return M.load_queue(_write_yaml(tmp_path, value))


def _rendered() -> dict:
    """渲染态队列：全部闸门开 + franco 资产已交付（只在内存里改）。"""
    value = _raw()
    value["action_acc_contract"]["wiring_confirmed"] = True
    value["franco_contract"]["wiring_confirmed"] = True
    value["assets"]["motion_backhand_franco"] = DELIVERED_FRANCO_NPZ
    return value


def _load_rendered() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        return _load(Path(tmp), _rendered())


def _job(queue: dict, job_id: str) -> dict:
    return M._job_by_id(queue, job_id)


def _compiled(queue: dict, job_id: str, stage: str) -> dict[str, str]:
    argv = M._training_argv(queue, _job(queue, job_id), stage)
    return M._override_map(argv[2:], "test")


# --------------------------------------------------------------------------------------------- #
# queue identity / governance
# --------------------------------------------------------------------------------------------- #
def test_checked_in_queue_validates():
    queue = M.load_queue(QUEUE_PATH)
    assert queue["queue_id"] == "phase1_integrated_upgrade_wave_20260723"


def test_safety_flags_frozen(tmp_path):
    for key, bad in (
        ("simulation_only", False),
        ("real_robot_authorized", True),
        ("launch_authorized_by_default", True),
        ("formal_exact_eligible", True),
    ):
        value = _raw()
        value[key] = bad
        with pytest.raises(M.QueueError):
            _load(tmp_path, value)


def test_missing_top_level_key_rejected(tmp_path):
    for missing in ("franco_contract", "reward_budget_contract", "action_acc_contract"):
        value = _raw()
        del value[missing]
        with pytest.raises(M.QueueError, match="keys differ"):
            _load(tmp_path, value)


def test_budgets_frozen_13301_and_fresh_20001():
    queue = M.load_queue(QUEUE_PATH)
    science = queue["budgets"]["science"]
    assert science["max_iterations"] == 13301
    assert science["absolute_milestones"] == [6900, 7200, 7700, 8700, 10700, 13300, 16700, 20000]
    assert [6700 + o for o in science["milestone_offsets_from_parent"]] == (
        science["absolute_milestones"]
    )
    fresh = queue["budgets"]["science_fresh"]
    assert fresh["max_iterations"] == 20001
    assert fresh["absolute_milestones"] == [200, 500, 1000, 2000, 4000, 10700, 20000]


def test_budget_drift_rejected(tmp_path):
    value = _raw()
    value["budgets"]["science"]["max_iterations"] = 4001
    with pytest.raises(M.QueueError, match="budgets.science"):
        _load(tmp_path, value)


def test_controls_point_at_matrix_w_c_s0_only(tmp_path):
    queue = M.load_queue(QUEUE_PATH)
    assert queue["controls"]["baseline_jobs"] == ["w_c_s0"]
    value = _raw()
    value["controls"]["baseline_jobs"] = ["w_c_s0", "v_c_s0"]
    with pytest.raises(M.QueueError, match="controls"):
        _load(tmp_path, value)


def test_single_parent_w_only(tmp_path):
    value = _raw()
    value["parents"]["V"] = copy.deepcopy(value["parents"]["W"])
    with pytest.raises(M.QueueError, match="exactly W"):
        _load(tmp_path, value)


def test_parent_recipe_verbatim_frozen(tmp_path):
    value = _raw()
    value["parents"]["W"]["recipe_overrides"][0] = (
        "task.rewards.racket_position_weight=18.0"
    )
    with pytest.raises(M.QueueError, match="recipe_overrides"):
        _load(tmp_path, value)


def test_commit_pinned_to_prereg_day_main_head():
    queue = M.load_queue(QUEUE_PATH)
    assert queue["source"]["commit"] == PINNED_COMMIT


def test_bad_commit_rejected(tmp_path):
    for bad in ("deadbeef", "0" * 40, "PENDING_40HEX_AFTER_WIRING_MERGE"):
        value = _raw()
        value["source"]["commit"] = bad
        with pytest.raises(M.QueueError, match="source.commit"):
            _load(tmp_path, value)


# --------------------------------------------------------------------------------------------- #
# 3 combo arms: frozen deltas (Franco 07-23 revisions folded in)
# --------------------------------------------------------------------------------------------- #
def test_exactly_3_unique_jobs_in_frozen_order():
    queue = M.load_queue(QUEUE_PATH)
    assert [job["id"] for job in queue["jobs"]] == list(M.ARM_ORDER)
    assert len({job["run_name"] for job in queue["jobs"]}) == 3
    assert all(job["run_name"].startswith("p1iu_") for job in queue["jobs"])


def test_launch_order_franco_first_frozen(tmp_path):
    queue = M.load_queue(QUEUE_PATH)
    assert queue["launch_order"] == ["combo_franco", "combo_resume", "combo_fresh"]
    value = _raw()
    value["launch_order"] = ["combo_resume", "combo_fresh", "combo_franco"]
    with pytest.raises(M.QueueError, match="launch_order"):
        _load(tmp_path, value)


def test_action_rate_minus_0p2_in_every_arm():
    queue = _load_rendered()
    for job_id in ("combo_fresh", "combo_resume", "combo_franco"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.rewards.action_rate_weight"] == "-0.2"


def test_temporal_drift_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["combo_resume"]["temporal_overrides"] = [
        "task.rewards.action_rate_weight=-0.1",
        *value["mechanisms"]["arms"]["combo_resume"]["temporal_overrides"][1:],
    ]
    with pytest.raises(M.QueueError, match="temporal_overrides drifted"):
        _load(tmp_path, value)


def test_stability_must_stay_s0_everywhere(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["combo_fresh"]["stability_overrides"] = [
        line.replace(
            "lower_body_pose_imitation_weight=0.0",
            "lower_body_pose_imitation_weight=2.0",
        )
        for line in value["mechanisms"]["arms"]["combo_fresh"]["stability_overrides"]
    ]
    with pytest.raises(M.QueueError, match="stability_overrides drifted"):
        _load(tmp_path, value)


def test_high_friction_ranges_in_every_arm():
    # Franco 07-23 变更①：全程高摩擦（0.6 太低）。
    queue = _load_rendered()
    for job_id in ("combo_fresh", "combo_resume", "combo_franco"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.plant.robot_material_static_friction_range"] == "[1.0,1.6]"
        assert compiled["task.plant.robot_material_dynamic_friction_range"] == "[0.8,1.2]"
        assert compiled["task.plant.zero_joint_friction"] == "true"  # 现役 plant 控制不动


def test_old_low_friction_floor_rejected(tmp_path):
    value = _raw()
    extras = value["mechanisms"]["arms"]["combo_resume"]["extra_overrides"]
    value["mechanisms"]["arms"]["combo_resume"]["extra_overrides"] = [
        line.replace("[1.0,1.6]", "[0.6,1.6]") for line in extras
    ]
    with pytest.raises(M.QueueError, match="extra_overrides drifted"):
        _load(tmp_path, value)


def test_rough_terrain_and_clearance_only_in_combo_fresh():
    queue = _load_rendered()
    fresh = _compiled(queue, "combo_fresh", "science")
    assert fresh["task.plant.terrain_rough_height_range"] == "[0.02,0.06]"
    assert fresh["task.rewards.foot_clearance_weight"] == "-0.01"
    assert fresh["task.rewards.foot_clearance_target_m"] == "0.15"
    for job_id in ("combo_resume", "combo_franco"):
        compiled = _compiled(queue, job_id, "science")
        assert "task.plant.terrain_rough_height_range" not in compiled
        assert "task.rewards.foot_clearance_weight" not in compiled
        assert "task.rewards.foot_clearance_target_m" not in compiled


def test_mjlab_tier1_trio_in_combo_fresh():
    # mjlab 档①三项全收：落地罚 + 抬脚罚 + 二阶平滑（Franco 追加②）。
    queue = _load_rendered()
    compiled = _compiled(queue, "combo_fresh", "science")
    assert compiled["task.rewards.foot_soft_landing_weight"] == "-0.003"
    assert compiled["task.rewards.foot_soft_landing_force_threshold_n"] == "300.0"
    assert compiled["task.rewards.foot_clearance_weight"] == "-0.01"
    assert compiled["task.rewards.action_acc_weight"] == "-0.05"


def test_footrw_dose_drift_rejected(tmp_path):
    value = _raw()
    extras = value["mechanisms"]["arms"]["combo_fresh"]["extra_overrides"]
    value["mechanisms"]["arms"]["combo_fresh"]["extra_overrides"] = [
        line.replace("foot_soft_landing_weight=-0.003", "foot_soft_landing_weight=-0.1")
        for line in extras
    ]
    with pytest.raises(M.QueueError, match="extra_overrides drifted"):
        _load(tmp_path, value)


def test_qbar_barrier_coexists_with_action_rate():
    # Franco 追加①：barrier 与 action_rate=-0.2 并存（intel 单变量互斥不适用）。
    queue = _load_rendered()
    for job_id in ("combo_fresh", "combo_resume", "combo_franco"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.rewards.qdes_limit_barrier_weight"] == "-0.65"
        assert compiled["task.rewards.qdes_limit_barrier_margin_frac"] == "0.08"
        assert compiled["task.rewards.action_rate_weight"] == "-0.2"


def test_both_push_events_in_every_arm_verbatim():
    # Franco 07-23 变更②：速度推 w_p035 + 力推 w_f035 两组事件并存，键面逐字。
    queue = _load_rendered()
    for job_id in ("combo_fresh", "combo_resume", "combo_franco"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.push.enable"] == "true"
        assert compiled["task.push.interval_range_s"] == "[5.0,15.0]"
        assert compiled["task.push.vel_xy_mps"] == "0.35"
        assert compiled["task.push.ang_vel_radps"] == "0.0"
        assert compiled["task.push.ang_axes"] == "none"
        assert compiled["task.force_push.enable"] == "true"
        assert compiled["task.force_push.interval_range_s"] == "[5.0,15.0]"
        assert compiled["task.force_push.force_n"] == "68.0"
        assert compiled["task.force_push.duration_s"] == "0.3"


def test_push_face_drift_rejected(tmp_path):
    value = _raw()
    extras = value["mechanisms"]["arms"]["combo_resume"]["extra_overrides"]
    value["mechanisms"]["arms"]["combo_resume"]["extra_overrides"] = [
        line.replace("vel_xy_mps=0.35", "vel_xy_mps=0.8") for line in extras
    ]
    with pytest.raises(M.QueueError, match="extra_overrides drifted"):
        _load(tmp_path, value)


def test_force_push_face_drift_rejected(tmp_path):
    value = _raw()
    extras = value["mechanisms"]["arms"]["combo_franco"]["extra_overrides"]
    value["mechanisms"]["arms"]["combo_franco"]["extra_overrides"] = [
        line.replace("force_n=68.0", "force_n=155.4") for line in extras
    ]
    with pytest.raises(M.QueueError, match="extra_overrides drifted"):
        _load(tmp_path, value)


def test_push_keys_in_base_rejected(tmp_path):
    value = _raw()
    value["common"]["base_overrides"].append("++task.push.enable=true")
    with pytest.raises(M.QueueError, match="never in base"):
        _load(tmp_path, value)


def test_new_surface_key_smuggled_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["combo_resume"]["extra_overrides"] = [
        *value["mechanisms"]["arms"]["combo_resume"]["extra_overrides"],
        "++task.plant.passive_damping_fold=true",
    ]
    with pytest.raises(M.QueueError, match="extra_overrides drifted"):
        _load(tmp_path, value)


# --------------------------------------------------------------------------------------------- #
# reward-ratio guard: hitting group frozen + soft penalties not stacked
# --------------------------------------------------------------------------------------------- #
def test_hitting_group_17_7_5_untouched_everywhere():
    queue = _load_rendered()
    for job_id in ("combo_fresh", "combo_resume", "combo_franco"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.rewards.racket_position_weight"] == "17.0"
        assert compiled["task.rewards.racket_velocity_weight"] == "7.0"
        assert compiled["task.rewards.racket_normal_weight"] == "5.0"
        # strike_success/progress 禁止 override（源码默认 5/10 = 击球组后两位）
        assert not [k for k in compiled if k.startswith("task.rewards.racket_strike_success")]
        assert not [k for k in compiled if k.startswith("task.rewards.racket_progress")]


def test_strike_success_override_rejected_in_base(tmp_path):
    value = _raw()
    value["common"]["base_overrides"].append(
        "++task.rewards.racket_progress_weight=20.0"
    )
    with pytest.raises(M.QueueError, match="17/7/5/5/10"):
        _load(tmp_path, value)


def test_soft_penalties_full_dose_not_stacked():
    queue = _load_rendered()
    for job_id in ("combo_fresh", "combo_resume", "combo_franco"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.rewards.racket_face_conditional_guidance_weight"] == "-0.4"
        assert compiled["task.rewards.foot_orientation_weight"] == "-0.3"
        assert compiled["task.rewards.prestrike_upright_weight"] == "-1.0"
        # penlight 教训：不再叠加软惩罚——蹭滑/拖脚/挥拍前脚滑键一个不带
        assert "task.rewards.pre_strike_foot_slip_weight" not in compiled
        assert "task.rewards.foot_slip_sq_weight" not in compiled
        assert "task.rewards.foot_drag_weight" not in compiled
        assert compiled["task.actions.qdes_clamp"] == "true"


def test_reward_budget_contract_values_frozen(tmp_path):
    queue = M.load_queue(QUEUE_PATH)
    hitting = queue["reward_budget_contract"]["hitting_group_frozen"]
    assert hitting["racket_position_weight"] == 17.0
    assert hitting["racket_strike_success_weight_default"] == 5.0
    assert hitting["racket_progress_weight_default"] == 10.0
    negative = queue["reward_budget_contract"]["new_negative_terms"]
    assert negative == {
        "action_rate_weight": -0.2,
        "action_acc_weight": -0.05,
        "foot_soft_landing_weight": -0.003,
        "foot_clearance_weight_fresh_only": -0.01,
        "qdes_limit_barrier_weight": -0.65,
    }
    value = _raw()
    value["reward_budget_contract"]["hitting_group_frozen"]["racket_position_weight"] = 18.0
    with pytest.raises(M.QueueError, match="does not move"):
        _load(tmp_path, value)
    value = _raw()
    value["reward_budget_contract"]["new_negative_terms"]["action_rate_weight"] = -0.5
    with pytest.raises(M.QueueError, match="new_negative_terms"):
        _load(tmp_path, value)


# --------------------------------------------------------------------------------------------- #
# combo_fresh: fresh-from-random iron rule
# --------------------------------------------------------------------------------------------- #
def test_fresh_carries_no_checkpoint_keys_and_fresh_budget():
    queue = _load_rendered()
    for stage in ("probe", "science"):
        compiled = _compiled(queue, "combo_fresh", stage)
        for key in M.CHECKPOINT_KEYS:
            assert key not in compiled
    assert _compiled(queue, "combo_fresh", "science")["max_iterations"] == "20001"
    assert _compiled(queue, "combo_fresh", "probe")["max_iterations"] == "2"


def test_resume_arms_carry_full_checkpoint_quadruple():
    queue = _load_rendered()
    for job_id in ("combo_resume", "combo_franco"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["checkpoint_path"].endswith("/model_6700.pt")
        assert compiled["checkpoint_tolerant"] == "false"
        assert compiled["checkpoint_allow_missing_contract"] == "false"
        assert compiled["checkpoint_allow_contract_mismatch"] == "true"
        assert compiled["max_iterations"] == "13301"


def test_fresh_flag_frozen_to_combo_fresh_only(tmp_path):
    value = _raw()
    for job in value["jobs"]:
        if job["id"] == "combo_resume":
            job["fresh_from_random"] = True
    with pytest.raises(M.QueueError, match="fresh_from_random"):
        _load(tmp_path, value)
    value = _raw()
    for job in value["jobs"]:
        if job["id"] == "combo_fresh":
            job["fresh_from_random"] = False
    with pytest.raises(M.QueueError, match="fresh_from_random"):
        _load(tmp_path, value)


def test_fresh_remote_body_omits_parent_files(tmp_path):
    queue = _load(tmp_path, _rendered())
    body_fresh = M._remote_body(queue, _job(queue, "combo_fresh"), "science", 0)
    assert "model_6700.pt" not in body_fresh
    assert "training_contract.json" not in body_fresh
    body_resume = M._remote_body(queue, _job(queue, "combo_resume"), "science", 0)
    assert "model_6700.pt" in body_resume
    assert "training_contract.json" in body_resume


# --------------------------------------------------------------------------------------------- #
# gates: action_acc locks all arms; franco locks combo_franco; motion swap
# --------------------------------------------------------------------------------------------- #
def test_checked_in_gate_states():
    queue = M.load_queue(QUEUE_PATH)
    assert queue["groundfoot_contract"]["wiring_confirmed"] is True
    assert queue["push_contract"]["wiring_confirmed"] is True
    assert queue["force_push_contract"]["wiring_confirmed"] is True
    assert queue["qbar_contract"]["wiring_confirmed"] is True
    # mjlab 档①第三项源码未接线：checked-in 状态必须锁死
    assert queue["action_acc_contract"]["wiring_confirmed"] is False
    assert queue["franco_contract"]["wiring_confirmed"] is False
    assert queue["assets"]["motion_backhand_franco"] == "PENDING_FRANCO_PIPELINE_DELIVERY"


def test_action_acc_gate_locks_all_three_arms():
    queue = M.load_queue(QUEUE_PATH)  # checked-in: action_acc false
    for job_id in ("combo_fresh", "combo_resume", "combo_franco"):
        with pytest.raises(M.QueueError, match="action_acc_contract"):
            M.render_command(queue, _job(queue, job_id), "science", "pod1", 0)


def test_groundfoot_gate_locks_all_three_arms(tmp_path):
    value = _rendered()
    value["groundfoot_contract"]["wiring_confirmed"] = False
    queue = _load(tmp_path, value)
    for job_id in ("combo_fresh", "combo_resume", "combo_franco"):
        with pytest.raises(M.QueueError, match="groundfoot_contract"):
            M.render_command(queue, _job(queue, job_id), "science", "pod1", 0)


def test_push_and_force_push_gates_lock_all_arms(tmp_path):
    for gate in ("push_contract", "force_push_contract", "qbar_contract"):
        value = _rendered()
        value[gate]["wiring_confirmed"] = False
        queue = _load(tmp_path, value)
        with pytest.raises(M.QueueError, match=gate):
            M.render_command(queue, _job(queue, "combo_resume"), "science", "pod1", 0)


def test_franco_gate_locks_only_combo_franco(tmp_path):
    value = _rendered()
    value["franco_contract"]["wiring_confirmed"] = False
    queue = _load(tmp_path, value)
    with pytest.raises(M.QueueError, match="franco_pipeline_20260722"):
        M.render_command(queue, _job(queue, "combo_franco"), "science", "pod1", 0)
    for job_id in ("combo_fresh", "combo_resume"):
        assert M.render_command(queue, _job(queue, job_id), "science", "pod1", 0)


def test_franco_placeholder_asset_refuses_even_with_gate_open(tmp_path):
    value = _rendered()
    value["assets"]["motion_backhand_franco"] = "PENDING_FRANCO_PIPELINE_DELIVERY"
    queue = _load(tmp_path, value)
    with pytest.raises(M.QueueError, match="placeholder"):
        M.render_command(queue, _job(queue, "combo_franco"), "science", "pod1", 0)
    # 另两臂不受影响
    assert M.render_command(queue, _job(queue, "combo_resume"), "science", "pod2", 1)


def test_franco_arm_swaps_backhand_motion_only(tmp_path):
    queue = _load(tmp_path, _rendered())
    franco = _compiled(queue, "combo_franco", "science")
    assert franco["motion_file_2"] == DELIVERED_FRANCO_NPZ
    assert franco["motion_file"].endswith("hope_forehand_v4rg_cal.npz")
    for job_id in ("combo_fresh", "combo_resume"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["motion_file_2"].endswith("hope_backhand_v4rg_cal.npz")
    body = M._remote_body(queue, _job(queue, "combo_franco"), "science", 1)
    assert DELIVERED_FRANCO_NPZ in body


def test_motion_asset_flag_frozen(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["combo_resume"]["motion_backhand_asset"] = "franco"
    with pytest.raises(M.QueueError, match="motion_backhand_asset"):
        _load(tmp_path, value)


def test_franco_prerequisites_must_be_three(tmp_path):
    value = _raw()
    value["franco_contract"]["unlock_prerequisites"] = (
        value["franco_contract"]["unlock_prerequisites"][:2]
    )
    with pytest.raises(M.QueueError, match="three"):
        _load(tmp_path, value)


def test_gate_contract_key_list_frozen(tmp_path):
    value = _raw()
    value["qbar_contract"]["expected_cli_keys"] = ["task.rewards.qdes_limit_barrier_weight"]
    with pytest.raises(M.QueueError, match="frozen key list"):
        _load(tmp_path, value)
    value = _raw()
    value["action_acc_contract"]["expected_cli_keys"] = ["task.rewards.action_acc"]
    with pytest.raises(M.QueueError, match="frozen key list"):
        _load(tmp_path, value)


def test_gate_boolean_must_be_bool(tmp_path):
    value = _raw()
    value["action_acc_contract"]["wiring_confirmed"] = "yes"
    with pytest.raises(M.QueueError, match="must be a bool"):
        _load(tmp_path, value)


# --------------------------------------------------------------------------------------------- #
# render mechanics（继承 CGF/intel wave 的护栏）
# --------------------------------------------------------------------------------------------- #
def test_pod_gpu_injected_at_render_time(tmp_path):
    queue = _load(tmp_path, _rendered())
    for pod, gpu in (("pod1", 0), ("pod2", 2)):
        command = M.render_command(queue, _job(queue, "combo_resume"), "science", pod, gpu)
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in command
        host, port = M.EXPECTED_PODS[pod]
        assert f"root@{host}" in command and str(port) in command


def test_invalid_pod_or_gpu_rejected(tmp_path):
    queue = _load(tmp_path, _rendered())
    with pytest.raises(M.QueueError, match="--pod"):
        M.render_command(queue, _job(queue, "combo_resume"), "science", "pod3", 0)
    with pytest.raises(M.QueueError, match="--gpu"):
        M.render_command(queue, _job(queue, "combo_resume"), "science", "pod1", 7)


def test_probe_and_science_namespaces_disjoint(tmp_path):
    queue = _load(tmp_path, _rendered())
    for job in queue["jobs"]:
        probe_dir = M._stage_run_dir(queue, job, "probe")
        science_dir = M._stage_run_dir(queue, job, "science")
        assert probe_dir != science_dir
        assert probe_dir.startswith(M.EXPECTED_NAMESPACE + "/probes/")
        assert science_dir.startswith(M.EXPECTED_NAMESPACE + "/runs/")
        assert M._stage_run_name(job, "probe").startswith("p1iu_probe_")


def test_remote_body_uses_kit_boot_lock_and_no_clobber(tmp_path):
    queue = _load(tmp_path, _rendered())
    body = M._remote_body(queue, _job(queue, "combo_resume"), "science", 1)
    assert M.KIT_BOOT_LOCK in body
    assert "launch_kit_training_locked.sh" not in body
    assert "test ! -e" in body  # no-clobber
    assert "rev-parse HEAD" in body  # exact commit check
    assert "status --porcelain" in body  # clean checkout check
    assert "KIT_BOOT_TIMEOUT_S=1800" in body
    assert "KIT_BOOT_STALE_TIMEOUT_S=900" in body


def test_ssh_command_shell_quote_roundtrip(tmp_path):
    queue = _load(tmp_path, _rendered())
    command = M.render_command(queue, _job(queue, "combo_fresh"), "science", "pod2", 1)
    argv = shlex.split(command)
    assert argv[0] == "ssh"
    inner = argv[-1]
    assert inner.startswith("bash -lc ")
    remote = shlex.split(inner)[2]
    assert "set -euo pipefail" in remote


def test_all_arms_compile_both_stages_without_duplicate_keys(tmp_path):
    queue = _load(tmp_path, _rendered())
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            argv = M._training_argv(queue, job, stage)
            M._override_map(argv[2:], "no-dup")  # 重复键会 raise
            assert not set(M._override_map(argv[2:], "no-deploy")) & {
                "ros", "deploy", "real_robot",
            }


def test_cli_plan_and_checklist_succeed():
    for flag in ([], ["--checklist"]):
        result = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--queue", str(QUEUE_PATH), *flag],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr


def test_cli_render_refused_while_action_acc_unwired():
    # checked-in 状态：action_acc 闸门锁全部三臂——CLI 渲染必须 rc2 REFUSED。
    refused = subprocess.run(
        [
            sys.executable, str(MODULE_PATH), "--queue", str(QUEUE_PATH),
            "--render-stage", "science", "--render-job", "combo_resume",
            "--pod", "pod1", "--gpu", "0",
        ],
        capture_output=True, text=True, check=False,
    )
    assert refused.returncode == 2
    assert "REFUSED" in refused.stderr
    assert "action_acc" in refused.stderr


def test_checklist_contains_wave_specific_guards():
    queue = M.load_queue(QUEUE_PATH)
    checklist = M.cmd_checklist(queue)
    assert "0. [阻塞]" in checklist and "action_acc" in checklist
    assert "1/3" in checklist  # 比值守卫
    assert "17/7/5/5/10" in checklist
    assert "franco_pipeline_20260722" in checklist
    assert "fresh-from-random" in checklist
    assert "ground_plant" in checklist
    assert "w_c_s0" in checklist
    assert "w_f035" in checklist and "w_p035" in checklist
    assert "margin_frac" in checklist
    assert "WARN" in checklist and "CLAMP ACTIVE" in checklist
    assert "sha256sum" in checklist


def test_plan_shows_gate_states_and_fresh_tag():
    queue = M.load_queue(QUEUE_PATH)
    plan = M.cmd_plan(queue)
    assert "action_acc_contract: 锁定" in plan
    assert "franco_contract: 锁定" in plan
    assert "groundfoot_contract: 已确认" in plan
    assert "fresh-from-random" in plan
    assert plan.count("p1iu_") >= 3
    assert "比值守卫" in plan


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
