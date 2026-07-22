"""抖动-地面-脚部消融波三件套测试（phase1_chatter_ground_foot_wave_20260722）。

Pinned here（结构照 test_run_phase1_intel_wave_queue.py，覆盖本波特有纪律）：

* 入库 YAML 通过全量校验；治理旗标冻结（simulation_only 等）；预算冻结
  （science 13301 -> 绝对里程碑到 20000；rough fresh 独立 20001）。
* 8 档臂存在且互斥改动：ar 三臂只动 action_rate 一行；grip/rough/footrw/penlight/
  kdpassive 的新键面逐臂冻结、别的臂出现即拒（单变量纪律）。
* penlight 双落点替换：base 原位替换拍面条件引导行 + parent 配方原位替换脚朝向/
  挥拍前直立两行 + extra 显式挥拍前脚滑——四个软惩罚 -0.13/-0.1/-0.33/-0.13,
  其余七臂保持全额 -0.4/-0.3/-1.0。
* rough fresh 铁律：命令绝不带 checkpoint 键；预算走 science_fresh；probe 同样无 resume。
* 三道闸：占位符 commit 拒渲染（--plan/--checklist 不受影响）；groundfoot 闸门锁
  grip/rough/footrw；kdpassive 闸门锁 kdpassive——各自 false 时渲染拒绝、其余臂照常。
* 渲染注入 pod/gpu 校验、shell 引号往返、kit_boot_lock 而非 locked launcher、
  no-clobber、push/force 键面全禁。

Run:  python -m pytest tests/test_run_phase1_chatter_ground_foot_wave_queue.py -q
"""

from __future__ import annotations

import copy
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts/run_phase1_chatter_ground_foot_wave_queue.py"
QUEUE_PATH = REPO / "configs/phase1_chatter_ground_foot_wave_20260722.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("cgf_wave_queue", MODULE_PATH)
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
    """渲染态队列：真实 40-hex commit + 两道闸门全开（只在内存里改）。"""
    value = _raw()
    value["source"]["commit"] = "a" * 40
    value["groundfoot_contract"]["wiring_confirmed"] = True
    value["kdpassive_contract"]["wiring_confirmed"] = True
    return value


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
    assert queue["queue_id"] == "phase1_chatter_ground_foot_wave_20260722"


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
    value = _raw()
    del value["groundfoot_contract"]
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
    value = _raw()
    value["budgets"]["science_fresh"]["max_iterations"] = 13301
    with pytest.raises(M.QueueError, match="science_fresh"):
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


# --------------------------------------------------------------------------------------------- #
# 8 arms: existence + mutually-exclusive deltas
# --------------------------------------------------------------------------------------------- #
def test_exactly_8_unique_jobs_in_frozen_order():
    queue = M.load_queue(QUEUE_PATH)
    assert [job["id"] for job in queue["jobs"]] == list(M.ARM_ORDER)
    assert len({job["run_name"] for job in queue["jobs"]}) == 8
    assert len({job["run_dir"] for job in queue["jobs"]}) == 8


def test_launch_order_frozen_value_ranking(tmp_path):
    queue = M.load_queue(QUEUE_PATH)
    assert queue["launch_order"] == [
        "grip", "footrw", "ar05", "penlight", "rough", "ar02", "ar10", "kdpassive",
    ]
    value = _raw()
    value["launch_order"] = list(reversed(value["launch_order"]))
    with pytest.raises(M.QueueError, match="launch_order"):
        _load(tmp_path, value)


def test_action_rate_dose_curve_compiled():
    queue = _load_rendered()
    for job_id, dose in (("ar02", "-0.2"), ("ar05", "-0.5"), ("ar10", "-1.0")):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.rewards.action_rate_weight"] == dose
        # 单变量：ar 臂无任何新键面
        assert not [k for k in compiled if k.startswith("task.plant.robot_material")]
        assert not [k for k in compiled if k.startswith("task.rewards.foot_soft")]
    for job_id in ("grip", "rough", "footrw", "penlight", "kdpassive"):
        assert _compiled(queue, job_id, "science")[
            "task.rewards.action_rate_weight"
        ] == "-0.1"


def _load_rendered(tmp_path=None) -> dict:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        return _load(Path(tmp), _rendered())


def test_temporal_drift_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["ar05"]["temporal_overrides"][0] = (
        "task.rewards.action_rate_weight=-0.4"
    )
    with pytest.raises(M.QueueError, match="temporal_overrides drifted"):
        _load(tmp_path, value)


def test_stability_must_stay_s0_everywhere(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["footrw"]["stability_overrides"] = [
        line.replace(
            "lower_body_pose_imitation_weight=0.0",
            "lower_body_pose_imitation_weight=2.0",
        )
        for line in value["mechanisms"]["arms"]["footrw"]["stability_overrides"]
    ]
    with pytest.raises(M.QueueError, match="stability_overrides drifted"):
        _load(tmp_path, value)


def test_grip_arm_carries_exactly_two_material_ranges():
    queue = _load_rendered()
    compiled = _compiled(queue, "grip", "science")
    assert compiled["task.plant.robot_material_static_friction_range"] == "[0.6,1.6]"
    assert compiled["task.plant.robot_material_dynamic_friction_range"] == "[0.5,1.2]"
    assert "task.plant.terrain_rough_height_range" not in compiled
    assert compiled["task.plant.zero_joint_friction"] == "true"  # 现役 plant 控制不动


def test_rough_arm_only_terrain_key():
    queue = _load_rendered()
    compiled = _compiled(queue, "rough", "science")
    assert compiled["task.plant.terrain_rough_height_range"] == "[0.02,0.06]"
    assert "task.plant.robot_material_static_friction_range" not in compiled


def test_footrw_arm_mjlab_equivalent_dose():
    queue = _load_rendered()
    compiled = _compiled(queue, "footrw", "science")
    # 量纲铁律：无量纲超阈倍数 -> mjlab -1e-5/N x 300 N = -3e-3；别抄 -0.1/-1e-4。
    assert compiled["task.rewards.foot_soft_landing_weight"] == "-0.003"
    assert compiled["task.rewards.foot_soft_landing_force_threshold_n"] == "300.0"
    assert "task.rewards.foot_clearance_weight" not in compiled  # 站立击球不开抬脚罚


def test_footrw_dose_drift_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["footrw"]["extra_overrides"] = [
        "++task.rewards.foot_soft_landing_weight=-0.1",
        "++task.rewards.foot_soft_landing_force_threshold_n=300.0",
    ]
    with pytest.raises(M.QueueError, match="extra_overrides drifted"):
        _load(tmp_path, value)


def test_new_surface_key_smuggled_into_other_arm_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["ar02"]["extra_overrides"] = [
        "++task.rewards.foot_soft_landing_weight=-0.003",
    ]
    with pytest.raises(M.QueueError, match="extra_overrides drifted"):
        _load(tmp_path, value)


def test_push_key_smuggled_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["grip"]["extra_overrides"] = [
        "++task.plant.robot_material_static_friction_range=[0.6,1.6]",
        "++task.plant.robot_material_dynamic_friction_range=[0.5,1.2]",
        "++task.push.enable=true",
    ]
    with pytest.raises(M.QueueError, match="extra_overrides drifted"):
        _load(tmp_path, value)


def test_no_push_force_or_deploy_keys_anywhere():
    queue = _load_rendered()
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            compiled = _compiled(queue, job["id"], stage)
            assert not [
                k for k in compiled
                if k.startswith("task.push.") or k.startswith("task.force_push.")
            ]
            assert not set(compiled) & {"ros", "deploy", "real_robot"}


# --------------------------------------------------------------------------------------------- #
# penlight: dual-site replacement + full-dose everywhere else
# --------------------------------------------------------------------------------------------- #
def test_penlight_reduces_six_soft_penalties_to_one_third():
    queue = _load_rendered()
    compiled = _compiled(queue, "penlight", "science")
    assert compiled["task.rewards.racket_face_conditional_guidance_weight"] == "-0.13"
    assert compiled["task.rewards.foot_orientation_weight"] == "-0.1"
    assert compiled["task.rewards.prestrike_upright_weight"] == "-0.33"
    assert compiled["task.rewards.pre_strike_foot_slip_weight"] == "-0.13"
    # 蹭滑/拖脚(源码常开 -1.0/-0.5,07-22 新接 CLI)也进减负组
    assert compiled["task.rewards.foot_slip_sq_weight"] == "-0.33"
    assert compiled["task.rewards.foot_drag_weight"] == "-0.17"
    # 硬保护/击球组不动
    assert compiled["task.rewards.racket_position_weight"] == "17.0"
    assert compiled["task.actions.qdes_clamp"] == "true"


def test_all_other_arms_keep_full_dose_soft_penalties():
    queue = _load_rendered()
    for job_id in ("ar02", "ar05", "ar10", "grip", "rough", "footrw", "kdpassive"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.rewards.racket_face_conditional_guidance_weight"] == "-0.4"
        assert compiled["task.rewards.foot_orientation_weight"] == "-0.3"
        assert compiled["task.rewards.prestrike_upright_weight"] == "-1.0"
        assert "task.rewards.pre_strike_foot_slip_weight" not in compiled
        assert "task.rewards.foot_slip_sq_weight" not in compiled
        assert "task.rewards.foot_drag_weight" not in compiled


def test_penlight_replacements_are_in_place_not_appended():
    queue = _load_rendered()
    argv = M._training_argv(queue, _job(queue, "penlight"), "science")
    # 原行消失、新行只出现一次、无重复键（_override_map 编译已保证后者）
    assert M.FACE_GUIDANCE_BASE not in argv
    assert argv.count(M.FACE_GUIDANCE_PENLIGHT) == 1
    assert M.FOOT_ORI_BASE not in argv
    assert argv.count(M.FOOT_ORI_PENLIGHT) == 1
    assert M.UPRIGHT_BASE not in argv
    assert argv.count(M.UPRIGHT_PENLIGHT) == 1


def test_penlight_replacement_drift_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["penlight"]["parent_recipe_replacements"][0]["new"] = (
        "task.rewards.foot_orientation_weight=-0.2"
    )
    with pytest.raises(M.QueueError, match="parent_recipe_replacements"):
        _load(tmp_path, value)


def test_replacement_must_keep_same_hydra_key(tmp_path):
    value = _raw()
    value["mechanisms"]["arms"]["penlight"]["base_replacements"] = [{
        "old": "++task.rewards.racket_face_conditional_guidance_weight=-0.4",
        "new": "++task.rewards.racket_face_guidance_weight=-0.13",
    }]
    with pytest.raises(M.QueueError, match="frozen design"):
        _load(tmp_path, value)


# --------------------------------------------------------------------------------------------- #
# rough: fresh-from-random iron rule
# --------------------------------------------------------------------------------------------- #
def test_rough_science_carries_no_checkpoint_keys_and_fresh_budget():
    queue = _load_rendered()
    for stage in ("probe", "science"):
        compiled = _compiled(queue, "rough", stage)
        for key in M.CHECKPOINT_KEYS:
            assert key not in compiled
    assert _compiled(queue, "rough", "science")["max_iterations"] == "20001"
    assert _compiled(queue, "rough", "probe")["max_iterations"] == "2"


def test_resume_arms_carry_full_checkpoint_quadruple():
    queue = _load_rendered()
    for job_id in ("ar02", "grip", "footrw", "penlight", "kdpassive"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["checkpoint_path"].endswith("/model_6700.pt")
        assert compiled["checkpoint_tolerant"] == "false"
        assert compiled["checkpoint_allow_missing_contract"] == "false"
        assert compiled["checkpoint_allow_contract_mismatch"] == "true"
        assert compiled["max_iterations"] == "13301"


def test_fresh_flag_frozen_to_rough_only(tmp_path):
    value = _raw()
    for job in value["jobs"]:
        if job["id"] == "grip":
            job["fresh_from_random"] = True
    with pytest.raises(M.QueueError, match="fresh_from_random"):
        _load(tmp_path, value)
    value = _raw()
    for job in value["jobs"]:
        if job["id"] == "rough":
            job["fresh_from_random"] = False
    with pytest.raises(M.QueueError, match="fresh_from_random"):
        _load(tmp_path, value)


def test_rough_remote_body_omits_parent_files(tmp_path):
    queue = _load(tmp_path, _rendered())
    body_rough = M._remote_body(queue, _job(queue, "rough"), "science", 0)
    assert "model_6700.pt" not in body_rough
    assert "training_contract.json" not in body_rough
    body_grip = M._remote_body(queue, _job(queue, "grip"), "science", 0)
    assert "model_6700.pt" in body_grip
    assert "training_contract.json" in body_grip


# --------------------------------------------------------------------------------------------- #
# gates: placeholder commit + groundfoot + kdpassive
# --------------------------------------------------------------------------------------------- #
def test_checked_in_commit_is_the_wiring_placeholder():
    queue = M.load_queue(QUEUE_PATH)
    assert queue["source"]["commit"] == "PENDING_40HEX_AFTER_WIRING_MERGE"
    assert queue["groundfoot_contract"]["wiring_confirmed"] is False
    assert queue["kdpassive_contract"]["wiring_confirmed"] is False


def test_placeholder_commit_accepted_for_plan_and_checklist():
    queue = M.load_queue(QUEUE_PATH)
    assert "PENDING_40HEX_AFTER_WIRING_MERGE" in M.cmd_plan(queue)
    assert "0. [阻塞]" in M.cmd_checklist(queue)


def test_placeholder_commit_refuses_render(tmp_path):
    queue = M.load_queue(QUEUE_PATH)
    with pytest.raises(M.QueueError, match="placeholder"):
        M.render_command(queue, _job(queue, "ar02"), "science", "pod1", 0)


def test_non_hex_and_zero_commit_rejected(tmp_path):
    for bad in ("deadbeef", "0" * 40, "PENDING_EXACT_COMMIT"):
        value = _raw()
        value["source"]["commit"] = bad
        with pytest.raises(M.QueueError, match="source.commit"):
            _load(tmp_path, value)


def test_groundfoot_gate_locks_exactly_four_arms(tmp_path):
    value = _rendered()
    value["groundfoot_contract"]["wiring_confirmed"] = False
    queue = _load(tmp_path, value)
    for job_id in ("grip", "rough", "footrw", "penlight"):
        with pytest.raises(M.QueueError, match="groundfoot"):
            M.render_command(queue, _job(queue, job_id), "science", "pod1", 0)
    # 其余四臂不受影响（kdpassive 闸门在本例已开）
    for job_id in ("ar02", "ar05", "ar10", "kdpassive"):
        assert M.render_command(queue, _job(queue, job_id), "science", "pod1", 0)


def test_kdpassive_gate_locks_only_kdpassive(tmp_path):
    value = _rendered()
    value["kdpassive_contract"]["wiring_confirmed"] = False
    queue = _load(tmp_path, value)
    with pytest.raises(M.QueueError, match="NO source wiring"):
        M.render_command(queue, _job(queue, "kdpassive"), "science", "pod1", 0)
    assert M.render_command(queue, _job(queue, "grip"), "science", "pod1", 0)


def test_gate_contract_key_list_frozen(tmp_path):
    value = _raw()
    value["groundfoot_contract"]["expected_cli_keys"] = [
        "task.plant.robot_material_static_friction_range",
    ]
    with pytest.raises(M.QueueError, match="frozen key list"):
        _load(tmp_path, value)
    value = _raw()
    value["kdpassive_contract"]["expected_cli_keys"] = ["task.plant.kd_fold"]
    with pytest.raises(M.QueueError, match="frozen key list"):
        _load(tmp_path, value)


def test_gate_boolean_must_be_bool(tmp_path):
    value = _raw()
    value["groundfoot_contract"]["wiring_confirmed"] = "yes"
    with pytest.raises(M.QueueError, match="must be a bool"):
        _load(tmp_path, value)


# --------------------------------------------------------------------------------------------- #
# render mechanics（继承 intel wave 的护栏）
# --------------------------------------------------------------------------------------------- #
def test_pod_gpu_injected_at_render_time(tmp_path):
    queue = _load(tmp_path, _rendered())
    for pod, gpu in (("pod1", 0), ("pod2", 2)):
        command = M.render_command(queue, _job(queue, "ar05"), "science", pod, gpu)
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in command
        host, port = M.EXPECTED_PODS[pod]
        assert f"root@{host}" in command and str(port) in command


def test_invalid_pod_or_gpu_rejected(tmp_path):
    queue = _load(tmp_path, _rendered())
    with pytest.raises(M.QueueError, match="--pod"):
        M.render_command(queue, _job(queue, "ar05"), "science", "pod3", 0)
    with pytest.raises(M.QueueError, match="--gpu"):
        M.render_command(queue, _job(queue, "ar05"), "science", "pod1", 7)


def test_probe_and_science_namespaces_disjoint(tmp_path):
    queue = _load(tmp_path, _rendered())
    for job in queue["jobs"]:
        probe_dir = M._stage_run_dir(queue, job, "probe")
        science_dir = M._stage_run_dir(queue, job, "science")
        assert probe_dir != science_dir
        assert probe_dir.startswith(M.EXPECTED_NAMESPACE + "/probes/")
        assert science_dir.startswith(M.EXPECTED_NAMESPACE + "/runs/")
        assert M._stage_run_name(job, "probe").startswith("p1cgf_probe_")


def test_remote_body_uses_kit_boot_lock_and_no_clobber(tmp_path):
    queue = _load(tmp_path, _rendered())
    body = M._remote_body(queue, _job(queue, "grip"), "science", 1)
    assert M.KIT_BOOT_LOCK in body
    assert "launch_kit_training_locked.sh" not in body
    assert "test ! -e" in body  # no-clobber
    assert "rev-parse HEAD" in body  # exact commit check
    assert "status --porcelain" in body  # clean checkout check
    assert "KIT_BOOT_TIMEOUT_S=1800" in body
    assert "KIT_BOOT_STALE_TIMEOUT_S=900" in body


def test_ssh_command_shell_quote_roundtrip(tmp_path):
    queue = _load(tmp_path, _rendered())
    command = M.render_command(queue, _job(queue, "penlight"), "science", "pod2", 1)
    argv = shlex.split(command)
    assert argv[0] == "ssh"
    inner = argv[-1]
    assert inner.startswith("bash -lc ")
    # bash -lc 后的整体必须能安全反解析出 remote body
    remote = shlex.split(inner)[2]
    assert "set -euo pipefail" in remote


def test_all_8_arms_compile_both_stages_without_duplicate_keys(tmp_path):
    queue = _load(tmp_path, _rendered())
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            argv = M._training_argv(queue, job, stage)
            M._override_map(argv[2:], "no-dup")  # 重复键会 raise


def test_cli_plan_and_checklist_succeed_with_placeholder():
    for flag in ([], ["--checklist"]):
        result = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--queue", str(QUEUE_PATH), *flag],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr


def test_cli_render_refused_with_placeholder():
    result = subprocess.run(
        [
            sys.executable, str(MODULE_PATH), "--queue", str(QUEUE_PATH),
            "--render-stage", "science", "--render-job", "ar02",
            "--pod", "pod1", "--gpu", "0",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "REFUSED" in result.stderr


def test_checklist_contains_wave_specific_guards():
    queue = M.load_queue(QUEUE_PATH)
    checklist = M.cmd_checklist(queue)
    assert "groundfoot" in checklist
    assert "passive_damping_fold" in checklist
    assert "fresh-from-random" in checklist
    assert "ground_plant" in checklist
    assert "w_c_s0" in checklist
    assert "WARN" in checklist and "CLAMP ACTIVE" in checklist
    assert "sha256sum" in checklist


def test_plan_shows_gate_states_and_fresh_tag():
    queue = M.load_queue(QUEUE_PATH)
    plan = M.cmd_plan(queue)
    assert "groundfoot 渲染: 锁定" in plan
    assert "kdpassive 渲染: 锁定" in plan
    assert "fresh-from-random" in plan
    assert plan.count("p1cgf_") >= 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
