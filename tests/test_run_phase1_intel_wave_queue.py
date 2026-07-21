from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import shlex
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_intel_wave_queue.py"
QUEUE = ROOT / "configs" / "phase1_intel_wave_20260721.yaml"
MATRIX_QUEUE = ROOT / "configs" / "phase1_balance_temporal_matrix_20260720.yaml"
TRAIN_PY = ROOT / "hope_training" / "whole_body_tracking" / "scripts" / "train.py"
REAL_COMMIT = "b" * 40
LEVELS = ("spdmix", "hstrong", "fullbody", "qbar")


def _module():
    spec = importlib.util.spec_from_file_location("intel_queue_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = _module()


def _raw() -> dict:
    value = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _matrix_raw() -> dict:
    value = yaml.safe_load(MATRIX_QUEUE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _load(tmp_path: Path, value: dict) -> dict:
    return Q.load_queue(_write_yaml(tmp_path, value))


def _rendered() -> dict:
    """Checked-in queue with the placeholder commit replaced by a real 40-hex one."""

    value = _raw()
    value["source"]["commit"] = REAL_COMMIT
    return value


def _rendered_qbar_open() -> dict:
    """渲染 fixture：真 commit + qbar 闸门翻开（主控合并 wiring 后的状态）。"""

    value = _rendered()
    value["qbar_contract"]["qbar_wiring_confirmed"] = True
    return value


def _job(queue: dict, job_id: str) -> dict:
    return Q._job_by_id(queue, job_id)


def _compiled(queue: dict, job_id: str, stage: str) -> dict[str, str]:
    argv = Q._training_argv(queue, _job(queue, job_id), stage)
    return Q._override_map(argv[2:], "test")


# ---------------------------------------------------------------- structure


def test_checked_in_queue_validates():
    queue = Q.load_queue(QUEUE)
    assert queue["queue_id"] == "phase1_intel_wave_20260721"


def test_safety_flags_frozen():
    queue = Q.load_queue(QUEUE)
    assert queue["simulation_only"] is True
    assert queue["real_robot_authorized"] is False
    assert queue["launch_authorized_by_default"] is False
    assert queue["formal_exact_eligible"] is False
    assert queue["evidence_class"] == (
        "diagnostic_only_intentional_parent_contract_mismatch"
    )


def test_missing_top_level_key_rejected(tmp_path):
    value = _raw()
    del value["qbar_contract"]
    with pytest.raises(Q.QueueError, match="keys differ"):
        _load(tmp_path, value)


def test_watchdog_and_namespace_inherited_verbatim(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["watchdog"]["boot_stall_timeout_s"] == 1800
    assert queue["watchdog"]["post_first_iteration_stall_timeout_s"] == 900
    assert queue["watchdog"]["retry_policy"] == "one_verbatim_retry_suffix_r2"
    assert queue["namespace"]["root"] == (
        "/workspace/codexschema/phase1_intel_wave_20260721"
    )
    assert queue["namespace"]["no_clobber"] is True
    value = _raw()
    value["namespace"]["no_clobber"] = False
    with pytest.raises(Q.QueueError, match="no-clobber"):
        _load(tmp_path, value)


def test_budget_is_4001_updates_save_100():
    queue = Q.load_queue(QUEUE)
    science = queue["budgets"]["science"]
    assert science["max_iterations"] == 4001
    assert science["save_interval"] == 100
    assert science["milestone_offsets_from_parent"] == [200, 500, 1000, 2000, 4000]
    assert science["absolute_milestones"] == [6900, 7200, 7700, 8700, 10700]
    assert [6700 + offset for offset in science["milestone_offsets_from_parent"]] == (
        science["absolute_milestones"]
    )


def test_budget_drift_rejected(tmp_path):
    value = _raw()
    value["budgets"]["science"]["max_iterations"] = 10001
    with pytest.raises(Q.QueueError, match="max_iterations"):
        _load(tmp_path, value)


# ---------------------------------------------------------------- cross-check vs matrix yaml


def test_base_overrides_verbatim_from_matrix_yaml():
    # 防漂移：base 配方必须与矩阵 yaml 逐字一致（w_c_s0/v_c_s0 才够格当对照），
    # 含 speed_scale_range=[1.0,1.0] 行（spdmix 的替换靶）。
    queue = Q.load_queue(QUEUE)
    matrix = _matrix_raw()
    assert queue["common"]["base_overrides"] == matrix["common"]["base_overrides"]
    assert "task.motion.speed_scale_range=[1.0,1.0]" in queue["common"]["base_overrides"]
    assert queue["common"]["seed"] == matrix["common"]["seed"] == 3
    assert (
        queue["common"]["planner_revision_override"]
        == matrix["common"]["planner_revision_override"]
    )


def test_parents_and_assets_verbatim_from_matrix_yaml():
    queue = Q.load_queue(QUEUE)
    matrix = _matrix_raw()
    assert queue["assets"] == matrix["assets"]
    for name in ("W", "V"):
        assert queue["parents"][name] == matrix["parents"][name]


def test_temporal_c_verbatim_from_matrix_yaml_in_spdmix_and_fullbody():
    queue = Q.load_queue(QUEUE)
    matrix_c = _matrix_raw()["mechanisms"]["temporal"]["C"]["overrides"]
    for level in ("spdmix", "fullbody"):
        assert queue["mechanisms"]["intel"][level]["temporal_overrides"] == matrix_c


def test_qbar_temporal_is_c_with_action_rate_zeroed_only():
    # Franco："把别的去掉"——qbar 相对 C 档唯一时序差异是 action_rate -0.1→0，
    # barrier 作为该臂唯一 qdes 惩罚；slew hinge/margin/窗四行逐字 = C 档。
    queue = Q.load_queue(QUEUE)
    matrix_c = _matrix_raw()["mechanisms"]["temporal"]["C"]["overrides"]
    qbar = queue["mechanisms"]["intel"]["qbar"]["temporal_overrides"]
    assert qbar == [
        "task.rewards.action_rate_weight=0.0",
        "++task.rewards.processed_qdes_slew_hinge_weight=0.0",
        "++task.rewards.processed_qdes_slew_hinge_margin=0.85",
        "++task.rewards.processed_qdes_slew_hinge_recovery_start_s=0.2",
        "++task.rewards.processed_qdes_slew_hinge_recovery_end_s=1.55",
    ]
    diffs = [(old, new) for old, new in zip(matrix_c, qbar) if old != new]
    assert diffs == [(
        "task.rewards.action_rate_weight=-0.1",
        "task.rewards.action_rate_weight=0.0",
    )]


def test_stability_s0_verbatim_from_matrix_yaml_in_non_fullbody_arms():
    queue = Q.load_queue(QUEUE)
    matrix_s0 = _matrix_raw()["mechanisms"]["stability"]["S0"]["overrides"]
    for level in ("spdmix", "hstrong", "qbar"):
        assert queue["mechanisms"]["intel"][level]["stability_overrides"] == matrix_s0


def test_hstrong_temporal_matches_matrix_h_window_with_minus_one_weight():
    # hstrong = 矩阵 H 档同款窗/margin，只把 hinge weight 从 -0.25 提到 -1.0。
    queue = Q.load_queue(QUEUE)
    matrix_h = _matrix_raw()["mechanisms"]["temporal"]["H"]["overrides"]
    hstrong = queue["mechanisms"]["intel"]["hstrong"]["temporal_overrides"]
    assert hstrong == [
        "task.rewards.action_rate_weight=0.0",
        "++task.rewards.processed_qdes_slew_hinge_weight=-1.0",
        "++task.rewards.processed_qdes_slew_hinge_margin=0.85",
        "++task.rewards.processed_qdes_slew_hinge_recovery_start_s=0.2",
        "++task.rewards.processed_qdes_slew_hinge_recovery_end_s=1.55",
    ]
    assert hstrong[0] == matrix_h[0]  # action_rate=0.0 与 H 档同
    assert hstrong[2:] == matrix_h[2:]  # margin/窗三行逐字 = H 档
    assert matrix_h[1] == "++task.rewards.processed_qdes_slew_hinge_weight=-0.25"


def test_fullbody_stability_is_s0_with_three_line_swap():
    # fullbody = S0 三行替换：模仿 weight 0→2.0 + 支持窗 0.3/0.4→10.0/10.0
    # （≈ 全程；两阶段下肢方案第一阶段：静止击球下肢全程软模仿）。
    queue = Q.load_queue(QUEUE)
    matrix_s0 = _matrix_raw()["mechanisms"]["stability"]["S0"]["overrides"]
    fullbody = queue["mechanisms"]["intel"]["fullbody"]["stability_overrides"]
    swaps = {
        "++task.rewards.lower_body_pose_imitation_weight=0.0":
            "++task.rewards.lower_body_pose_imitation_weight=2.0",
        "++task.rewards.lower_body_pose_imitation_support_pre_s=0.3":
            "++task.rewards.lower_body_pose_imitation_support_pre_s=10.0",
        "++task.rewards.lower_body_pose_imitation_support_post_s=0.4":
            "++task.rewards.lower_body_pose_imitation_support_post_s=10.0",
    }
    assert fullbody == [swaps.get(line, line) for line in matrix_s0]
    diffs = [
        (old, new) for old, new in zip(matrix_s0, fullbody) if old != new
    ]
    assert diffs == sorted(swaps.items(), key=lambda pair: matrix_s0.index(pair[0]))


def test_temporal_drift_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["intel"]["qbar"]["temporal_overrides"][0] = (
        "task.rewards.action_rate_weight=-0.2"
    )
    with pytest.raises(Q.QueueError, match="temporal"):
        _load(tmp_path, value)


def test_stability_drift_rejected(tmp_path):
    value = _raw()
    overrides = value["mechanisms"]["intel"]["spdmix"]["stability_overrides"]
    index = overrides.index("++task.rewards.post_swing_settle_debt_weight=0.0")
    overrides[index] = "++task.rewards.post_swing_settle_debt_weight=-0.25"
    with pytest.raises(Q.QueueError, match="stability"):
        _load(tmp_path, value)


def test_controls_point_at_matrix_baseline(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["controls"]["baseline_queue_id"] == (
        "phase1_balance_temporal_matrix_20260720"
    )
    assert queue["controls"]["baseline_jobs"] == ["w_c_s0", "v_c_s0"]
    assert queue["controls"]["baseline_run_names"] == [
        "p1btm_w_c_s0_seed3_20260720", "p1btm_v_c_s0_seed3_20260720",
    ]
    matrix = _matrix_raw()
    matrix_names = {job["run_name"] for job in matrix["jobs"]}
    for name in queue["controls"]["baseline_run_names"]:
        assert name in matrix_names
    value = _raw()
    value["controls"]["baseline_jobs"] = ["w_n_s0", "v_n_s0"]
    with pytest.raises(Q.QueueError, match="controls"):
        _load(tmp_path, value)


# ---------------------------------------------------------------- jobs / arms


def test_exactly_8_unique_jobs():
    queue = Q.load_queue(QUEUE)
    jobs = queue["jobs"]
    assert len(jobs) == 8
    assert len({job["id"] for job in jobs}) == 8
    assert len({job["run_name"] for job in jobs}) == 8
    assert len({job["run_dir"] for job in jobs}) == 8


def test_parent_intel_coverage_complete():
    queue = Q.load_queue(QUEUE)
    cells = {(job["parent"], job["intel"]) for job in queue["jobs"]}
    assert len(cells) == 8
    for parent in ("W", "V"):
        for level in LEVELS:
            assert (parent, level) in cells


def test_jobs_carry_no_pod_or_gpu():
    # 槽位策略：本波空槽即填，yaml 不写死 pod/gpu，渲染时注入。
    queue = Q.load_queue(QUEUE)
    for job in queue["jobs"]:
        assert "pod" not in job
        assert "gpu" not in job


def test_run_name_formula_enforced(tmp_path):
    queue = Q.load_queue(QUEUE)
    for job in queue["jobs"]:
        assert job["run_name"] == f"p1iq_{job['id']}_seed3_20260721"
    value = _raw()
    value["jobs"][0]["run_name"] = "p1iq_wrong_seed3_20260721"
    with pytest.raises(Q.QueueError, match="run_name must be"):
        _load(tmp_path, value)


def test_duplicate_run_name_rejected(tmp_path):
    value = _raw()
    value["jobs"][1]["run_name"] = value["jobs"][0]["run_name"]
    with pytest.raises(Q.QueueError, match="run_name"):
        _load(tmp_path, value)


def test_job_axes_must_match_id(tmp_path):
    value = _raw()
    value["jobs"][0]["intel"] = "qbar"
    with pytest.raises(Q.QueueError, match="axes do not match"):
        _load(tmp_path, value)


def test_launch_order_frozen_exact_sequence(tmp_path):
    # 任务书冻结：spdmix -> fullbody -> hstrong -> qbar，W 先 V 后交错。
    queue = Q.load_queue(QUEUE)
    assert queue["launch_order"] == [
        "w_spdmix", "v_spdmix", "w_fullbody", "v_fullbody",
        "w_hstrong", "v_hstrong", "w_qbar", "v_qbar",
    ]
    assert queue["launch_order"] == list(Q.LAUNCH_ORDER)
    assert sorted(queue["launch_order"]) == sorted(job["id"] for job in queue["jobs"])
    value = _raw()
    value["launch_order"][0], value["launch_order"][1] = (
        value["launch_order"][1], value["launch_order"][0]
    )
    with pytest.raises(Q.QueueError, match="launch_order"):
        _load(tmp_path, value)


# ---------------------------------------------------------------- spdmix（逐字断言两种情况）


def test_spdmix_base_replacement_frozen(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["mechanisms"]["intel"]["spdmix"]["base_replacements"] == [
        {
            "old": "task.motion.speed_scale_range=[1.0,1.0]",
            "new": "task.motion.speed_scale_range=[0.8,1.2]",
        }
    ]
    for level in ("hstrong", "fullbody", "qbar"):
        assert queue["mechanisms"]["intel"][level]["base_replacements"] == []
    value = _raw()
    value["mechanisms"]["intel"]["spdmix"]["base_replacements"][0]["new"] = (
        "task.motion.speed_scale_range=[0.5,1.5]"
    )
    with pytest.raises(Q.QueueError, match="base_replacements"):
        _load(tmp_path, value)


def test_spdmix_arms_get_verbatim_replacement_others_keep_base(tmp_path):
    # 任务书逐字断言：spdmix 两臂 argv 含 [0.8,1.2] 不含 [1.0,1.0]；其余六臂反向。
    queue = _load(tmp_path, _rendered_qbar_open())
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            argv = Q._training_argv(queue, job, stage)
            if job["intel"] == "spdmix":
                assert "task.motion.speed_scale_range=[0.8,1.2]" in argv
                assert "task.motion.speed_scale_range=[1.0,1.0]" not in argv
            else:
                assert "task.motion.speed_scale_range=[1.0,1.0]" in argv
                assert "task.motion.speed_scale_range=[0.8,1.2]" not in argv


def test_spdmix_replacement_is_in_place_not_appended(tmp_path):
    # 原位替换：spdmix argv 的 speed_scale 键只出现一次，位置与 base 行一致。
    queue = _load(tmp_path, _rendered())
    argv = Q._training_argv(queue, _job(queue, "w_spdmix"), "science")
    hits = [item for item in argv if item.startswith("task.motion.speed_scale_range=")]
    assert hits == ["task.motion.speed_scale_range=[0.8,1.2]"]
    base = queue["common"]["base_overrides"]
    base_index = base.index("task.motion.speed_scale_range=[1.0,1.0]")
    assert argv[2:].index("task.motion.speed_scale_range=[0.8,1.2]") == base_index


def test_base_missing_replacement_target_rejected(tmp_path):
    value = _raw()
    base = value["common"]["base_overrides"]
    base[base.index("task.motion.speed_scale_range=[1.0,1.0]")] = (
        "task.motion.speed_scale_range=[0.9,1.1]"
    )
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


# ---------------------------------------------------------------- hstrong / fullbody 语义


def test_hstrong_compiled_weights(tmp_path):
    queue = _load(tmp_path, _rendered())
    for job_id in ("w_hstrong", "v_hstrong"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.rewards.action_rate_weight"] == "0.0"
        assert compiled["task.rewards.processed_qdes_slew_hinge_weight"] == "-1.0"
        assert compiled["task.rewards.processed_qdes_slew_hinge_margin"] == "0.85"
        assert compiled["task.rewards.processed_qdes_slew_hinge_recovery_start_s"] == "0.2"
        assert compiled["task.rewards.processed_qdes_slew_hinge_recovery_end_s"] == "1.55"


def test_spdmix_and_fullbody_arms_keep_matrix_c_weights(tmp_path):
    queue = _load(tmp_path, _rendered_qbar_open())
    for job in queue["jobs"]:
        if job["intel"] in ("hstrong", "qbar"):
            continue
        compiled = _compiled(queue, job["id"], "science")
        assert compiled["task.rewards.action_rate_weight"] == "-0.1"
        assert compiled["task.rewards.processed_qdes_slew_hinge_weight"] == "0.0"


def test_qbar_arms_zero_action_rate_and_keep_slew_zero(tmp_path):
    # qbar 两臂去掉 raw action_rate（barrier 是唯一 qdes 惩罚），slew hinge 保持 0。
    queue = _load(tmp_path, _rendered_qbar_open())
    for job_id in ("w_qbar", "v_qbar"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.rewards.action_rate_weight"] == "0.0"
        assert compiled["task.rewards.processed_qdes_slew_hinge_weight"] == "0.0"


def test_fullbody_compiled_weights(tmp_path):
    queue = _load(tmp_path, _rendered())
    for job_id in ("w_fullbody", "v_fullbody"):
        compiled = _compiled(queue, job_id, "science")
        assert compiled["task.rewards.lower_body_pose_imitation_weight"] == "2.0"
        assert compiled["task.rewards.lower_body_pose_imitation_std"] == "0.35"
        # 支持窗放开到全程（pre/post 10 s ≈ 覆盖整个 16 s episode 的击球相位）。
        assert compiled["task.rewards.lower_body_pose_imitation_support_pre_s"] == "10.0"
        assert compiled["task.rewards.lower_body_pose_imitation_support_post_s"] == "10.0"
        # 其余两个稳定机制保持 0（单变量）。
        assert compiled["task.rewards.post_swing_settle_debt_weight"] == "0.0"
        assert compiled["task.rewards.lower_body_stability_bundle_weight"] == "0.0"


def test_non_fullbody_arms_keep_pose_imitation_zero_and_narrow_window(tmp_path):
    queue = _load(tmp_path, _rendered_qbar_open())
    for job in queue["jobs"]:
        if job["intel"] == "fullbody":
            continue
        compiled = _compiled(queue, job["id"], "science")
        assert compiled["task.rewards.lower_body_pose_imitation_weight"] == "0.0"
        assert compiled["task.rewards.lower_body_pose_imitation_support_pre_s"] == "0.3"
        assert compiled["task.rewards.lower_body_pose_imitation_support_post_s"] == "0.4"


def test_fullbody_window_drift_rejected(tmp_path):
    # 全程窗是设计的一部分：fullbody 改回窄窗（或其他值）必须被拒绝。
    value = _raw()
    overrides = value["mechanisms"]["intel"]["fullbody"]["stability_overrides"]
    index = overrides.index(
        "++task.rewards.lower_body_pose_imitation_support_pre_s=10.0"
    )
    overrides[index] = "++task.rewards.lower_body_pose_imitation_support_pre_s=0.3"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_fullbody_weight_drift_rejected(tmp_path):
    value = _raw()
    overrides = value["mechanisms"]["intel"]["fullbody"]["stability_overrides"]
    index = overrides.index("++task.rewards.lower_body_pose_imitation_weight=2.0")
    overrides[index] = "++task.rewards.lower_body_pose_imitation_weight=0.5"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


# ---------------------------------------------------------------- qbar（键面 + 渲染闸门）


def test_qbar_extra_overrides_exact_verbatim():
    queue = Q.load_queue(QUEUE)
    assert queue["mechanisms"]["intel"]["qbar"]["extra_overrides"] == [
        "++task.rewards.qdes_limit_barrier_weight=-0.65",
        "++task.rewards.qdes_limit_barrier_margin=0.08",
    ]
    for level in ("spdmix", "hstrong", "fullbody"):
        assert queue["mechanisms"]["intel"][level]["extra_overrides"] == []


def test_qbar_weight_drift_rejected(tmp_path):
    value = _raw()
    overrides = value["mechanisms"]["intel"]["qbar"]["extra_overrides"]
    overrides[0] = "++task.rewards.qdes_limit_barrier_weight=-1.0"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_qbar_arms_single_variable_barrier_keys(tmp_path):
    # qbar 两臂必带两个 barrier 键；其余六臂绝无 barrier 键（单变量）。
    queue = _load(tmp_path, _rendered_qbar_open())
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            compiled = _compiled(queue, job["id"], stage)
            barrier = sorted(
                key for key in compiled
                if key.startswith("task.rewards.qdes_limit_barrier")
            )
            if job["intel"] == "qbar":
                assert barrier == [
                    "task.rewards.qdes_limit_barrier_margin",
                    "task.rewards.qdes_limit_barrier_weight",
                ]
                assert compiled["task.rewards.qdes_limit_barrier_weight"] == "-0.65"
                assert compiled["task.rewards.qdes_limit_barrier_margin"] == "0.08"
            else:
                assert barrier == []


def test_qbar_gate_closed_because_wiring_not_landed_in_worktree():
    # fail-closed 交叉断言：本地工作树 train.py 没有 qdes_limit_barrier wiring 时，
    # 检入的 yaml 闸门必须还关着（wiring 落盘后本测试自动放行 true）。
    text = TRAIN_PY.read_text(encoding="utf-8")
    queue = Q.load_queue(QUEUE)
    if "qdes_limit_barrier" not in text:
        assert queue["qbar_contract"]["qbar_wiring_confirmed"] is not True, (
            "train.py has no qdes_limit_barrier wiring yet, so the qbar render gate "
            "must stay closed"
        )


def test_qbar_contract_frozen(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["qbar_contract"]["expected_cli_keys"] == [
        "task.rewards.qdes_limit_barrier_weight",
        "task.rewards.qdes_limit_barrier_margin",
    ]
    assert isinstance(queue["qbar_contract"]["qbar_wiring_confirmed"], bool)
    value = _raw()
    value["qbar_contract"]["expected_cli_keys"] = ["task.rewards.other_key"]
    with pytest.raises(Q.QueueError, match="expected_cli_keys"):
        _load(tmp_path, value)
    value = _raw()
    value["qbar_contract"]["qbar_wiring_confirmed"] = "yes"
    with pytest.raises(Q.QueueError, match="bool"):
        _load(tmp_path, value)


def test_qbar_render_locked_until_wiring_confirmed(tmp_path):
    # 渲染闸门：qbar_wiring_confirmed=false 时 qbar 两臂渲染被拒，其余六臂不受
    # 影响；翻 true 后 qbar 渲染出带两个 barrier 键的完整命令。
    value = _rendered()
    value["qbar_contract"]["qbar_wiring_confirmed"] = False
    queue = _load(tmp_path, value)
    with pytest.raises(Q.QueueError, match="wiring"):
        Q.render_command(queue, _job(queue, "w_qbar"), "probe", "pod1", 0)
    with pytest.raises(Q.QueueError, match="wiring"):
        Q.render_command(queue, _job(queue, "v_qbar"), "science", "pod2", 1)
    for job_id in (
        "w_spdmix", "v_spdmix", "w_hstrong", "v_hstrong", "w_fullbody", "v_fullbody",
    ):
        command = Q.render_command(queue, _job(queue, job_id), "science", "pod1", 0)
        assert command.startswith("ssh ")

    queue = _load(tmp_path, _rendered_qbar_open())
    command = Q.render_command(queue, _job(queue, "w_qbar"), "science", "pod1", 0)
    assert "task.rewards.qdes_limit_barrier_weight=-0.65" in command
    assert "task.rewards.qdes_limit_barrier_margin=0.08" in command


# ---------------------------------------------------------------- commit gate


def _placeholder_path(tmp_path):
    value = _raw()
    value["source"]["commit"] = "PENDING_EXACT_COMMIT"
    return _write_yaml(tmp_path, value)


def test_checked_in_commit_is_exact_40_hex():
    # 已冻结：qbar wiring 已合并、闸门已开；占位语义由 tmp fixture 继续覆盖。
    import re as _re
    queue = Q.load_queue(QUEUE)
    assert _re.fullmatch(r"[0-9a-f]{40}", queue["source"]["commit"])
    assert queue["source"]["checkout"] == "/workspace/codexschema/nohope_push_20260721"


def test_placeholder_commit_accepted_for_plan_and_checklist(tmp_path):
    queue = Q.load_queue(_placeholder_path(tmp_path))
    plan = Q.cmd_plan(queue)
    assert "PENDING_EXACT_COMMIT" in plan
    checklist = Q.cmd_checklist(queue)
    assert "PENDING_EXACT_COMMIT" in checklist
    assert "[阻塞]" in checklist


def test_placeholder_commit_refuses_render(tmp_path):
    queue = Q.load_queue(_placeholder_path(tmp_path))
    job = queue["jobs"][0]
    with pytest.raises(Q.QueueError, match="placeholder"):
        Q.render_command(queue, job, "probe", "pod1", 0)
    with pytest.raises(Q.QueueError, match="placeholder"):
        Q.render_command(queue, job, "science", "pod2", 2)


def test_placeholder_commit_refuses_render_via_cli(tmp_path):
    path = _placeholder_path(tmp_path)
    result = Q.main([
        "--queue", str(path), "--render-stage", "probe",
        "--render-job", "w_spdmix", "--pod", "pod1", "--gpu", "0",
    ])
    assert result == 2


def test_non_hex_commit_rejected(tmp_path):
    value = _raw()
    value["source"]["commit"] = "not-a-commit"
    with pytest.raises(Q.QueueError, match="commit"):
        _load(tmp_path, value)


def test_zero_commit_rejected(tmp_path):
    value = _raw()
    value["source"]["commit"] = "0" * 40
    with pytest.raises(Q.QueueError, match="commit"):
        _load(tmp_path, value)


def test_real_commit_renders(tmp_path):
    queue = _load(tmp_path, _rendered())
    command = Q.render_command(queue, _job(queue, "w_spdmix"), "science", "pod1", 0)
    assert command.startswith("ssh ")
    assert REAL_COMMIT in command


# ---------------------------------------------------------------- pod/gpu injection


def test_pod_gpu_injected_at_render_time(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "v_fullbody")
    body = Q._remote_body(queue, job, "science", 1)
    assert "CUDA_VISIBLE_DEVICES=1" in body
    assert "nvidia-smi -i 1" in body
    pod1_argv = Q._ssh_argv(queue, job, "science", "pod1", 1)
    assert "root@162.43.172.171" in pod1_argv
    assert "18333" in pod1_argv
    pod2_argv = Q._ssh_argv(queue, job, "science", "pod2", 2)
    assert "root@162.43.172.181" in pod2_argv
    assert "13146" in pod2_argv


def test_same_job_renders_on_any_slot(tmp_path):
    # 空槽即填：同一臂必须能渲染到任意 pod/gpu 组合。
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "w_hstrong")
    for pod in ("pod1", "pod2"):
        for gpu in (0, 1, 2):
            command = Q.render_command(queue, job, "probe", pod, gpu)
            assert f"CUDA_VISIBLE_DEVICES={gpu}" in command


def test_invalid_pod_rejected(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "w_spdmix")
    with pytest.raises(Q.QueueError, match="--pod"):
        Q.render_command(queue, job, "science", "pod3", 0)


def test_invalid_gpu_rejected(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "w_spdmix")
    with pytest.raises(Q.QueueError, match="--gpu"):
        Q.render_command(queue, job, "science", "pod1", 3)
    with pytest.raises(Q.QueueError, match="--gpu"):
        Q.render_command(queue, job, "science", "pod1", "junk")
    with pytest.raises(Q.QueueError, match="--gpu"):
        Q.render_command(queue, job, "science", "pod1", -1)


def test_cli_render_requires_pod_and_gpu():
    result = Q.main([
        "--queue", str(QUEUE), "--render-stage", "probe", "--render-job", "w_spdmix",
    ])
    assert result == 2


def test_render_unknown_job_rejected(tmp_path):
    queue = _load(tmp_path, _rendered())
    with pytest.raises(Q.QueueError, match="unknown job id"):
        Q.cmd_render(queue, "science", "w_nope", "pod1", 0)


def test_gpu_precheck_under_four_procs_inherited(tmp_path):
    # 逐字继承矩阵的"同卡 <4 个 compute 进程"预检。
    queue = _load(tmp_path, _rendered())
    body = Q._remote_body(queue, _job(queue, "w_spdmix"), "science", 2)
    assert "--query-compute-apps=pid" in body
    assert "-lt 4" in body


# ---------------------------------------------------------------- rendered argv


def test_science_command_contains_all_required_keys(tmp_path):
    queue = _load(tmp_path, _rendered())
    compiled = _compiled(queue, "w_spdmix", "science")
    parent = queue["parents"]["W"]
    assert compiled["checkpoint_path"] == parent["checkpoint_path"]
    assert compiled["checkpoint_tolerant"] == "false"
    assert compiled["checkpoint_allow_missing_contract"] == "false"
    assert compiled["checkpoint_allow_contract_mismatch"] == "true"
    assert compiled["seed"] == "3"
    assert compiled["num_envs"] == "4096"
    assert compiled["algo.runner.num_steps_per_env"] == "24"
    assert compiled["max_iterations"] == "4001"
    assert compiled["algo.runner.save_interval"] == "100"
    assert compiled["run_name"] == "p1iq_w_spdmix_seed3_20260721"
    assert compiled["logger"] == "tensorboard"
    assert compiled["device"] == "cuda:0"


def test_probe_budget_and_disjoint_namespaces(tmp_path):
    queue = _load(tmp_path, _rendered())
    compiled = _compiled(queue, "v_hstrong", "probe")
    assert compiled["max_iterations"] == "2"
    assert compiled["algo.runner.save_interval"] == "1"
    assert compiled["run_name"] == "p1iq_probe_v_hstrong_seed3_20260721"
    job = _job(queue, "v_hstrong")
    probe_dir = Q._stage_run_dir(queue, job, "probe")
    science_dir = Q._stage_run_dir(queue, job, "science")
    assert probe_dir != science_dir
    assert probe_dir.endswith("/probes/v_hstrong")
    assert science_dir.endswith("/runs/v_hstrong")


def test_all_8_arms_compile_without_duplicate_keys(tmp_path):
    queue = _load(tmp_path, _rendered_qbar_open())
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            argv = Q._training_argv(queue, job, stage)
            Q._override_map(argv[2:], job["id"])  # raises on duplicates


def test_no_push_or_force_keys_anywhere(tmp_path):
    # 剔除 push/force 键面：本波任何臂的 argv 都不得出现 task.push.* /
    # task.force_push.* 键（缺席 == 逐字节 no-op）。
    queue = _load(tmp_path, _rendered_qbar_open())
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            compiled = _compiled(queue, job["id"], stage)
            assert not any(
                key.startswith("task.push.") or key.startswith("task.force_push.")
                for key in compiled
            )


def test_push_key_smuggled_into_extra_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["intel"]["qbar"]["extra_overrides"] = [
        "++task.rewards.qdes_limit_barrier_weight=-0.65",
        "++task.rewards.qdes_limit_barrier_margin=0.08",
        "++task.push.enable=true",
    ]
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_no_deploy_arguments_anywhere(tmp_path):
    queue = _load(tmp_path, _rendered_qbar_open())
    for job in queue["jobs"]:
        compiled = _compiled(queue, job["id"], "science")
        assert not (set(compiled) & {"ros", "deploy", "real_robot", "motion_command"})


# ---------------------------------------------------------------- remote command


def test_ssh_command_shell_quote_roundtrip(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "v_spdmix")
    argv = Q._ssh_argv(queue, job, "science", "pod2", 1)
    rendered = Q.render_command(queue, job, "science", "pod2", 1)
    assert shlex.split(rendered) == argv


def test_remote_body_quotes_survive_bash_lc_extraction(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "w_fullbody")
    argv = Q._ssh_argv(queue, job, "probe", "pod1", 2)
    payload = argv[-1]
    assert payload.startswith("bash -lc ")
    inner = shlex.split(payload)
    assert inner[:2] == ["bash", "-lc"]
    assert Q._remote_body(queue, job, "probe", 2) == inner[2]


def test_remote_body_uses_kit_boot_lock_not_locked_launcher(tmp_path):
    queue = _load(tmp_path, _rendered())
    body = Q._remote_body(queue, _job(queue, "w_spdmix"), "science", 0)
    assert "/workspace/bin/kit_boot_lock.sh" in body
    assert "launch_kit_training_locked.sh" not in body
    assert "setsid nohup" in body
    assert "KIT_BOOT_TIMEOUT_S=1800" in body
    assert "KIT_BOOT_STALE_TIMEOUT_S=900" in body


def test_remote_body_verifies_commit_and_no_clobber(tmp_path):
    queue = _load(tmp_path, _rendered())
    # qbar 闸门默认锁着，用不受闸门影响的臂验证 no-clobber 语义。
    job = _job(queue, "v_fullbody")
    body = Q._remote_body(queue, job, "science", 0)
    assert (
        'test "$(git -C /workspace/codexschema/nohope_push_20260721 rev-parse HEAD)" = '
        + REAL_COMMIT
    ) in body
    assert "status --porcelain" in body
    assert f"test ! -e {shlex.quote(job['run_dir'])}" in body


# ---------------------------------------------------------------- plan / checklist


def test_plan_lists_8_rows_and_fill_order():
    queue = Q.load_queue(QUEUE)
    plan = Q.cmd_plan(queue)
    for job in queue["jobs"]:
        assert job["id"] in plan
        assert job["run_name"] in plan
    assert "人话" in plan
    assert "推荐填充顺序" in plan
    assert "w_c_s0/v_c_s0" in plan
    assert "speed_scale uniform[0.8,1.2]" in plan
    assert "slew hinge -1.0" in plan
    assert "lower_body_pose_imitation +2.0" in plan
    assert "qdes_limit_barrier weight -0.65 margin 0.08" in plan
    for index, job_id in enumerate(queue["launch_order"], start=1):
        assert f"{index:2d}. {job_id}" in plan


def test_plan_shows_qbar_gate_state():
    queue = Q.load_queue(QUEUE)
    plan = Q.cmd_plan(queue)
    if queue["qbar_contract"]["qbar_wiring_confirmed"]:
        assert "qbar 渲染: 已解锁" in plan
    else:
        assert "qbar 渲染: 锁定" in plan
        assert "qdes_limit_barrier" in plan


def test_checklist_contains_inherited_guards_and_intel_checks():
    queue = Q.load_queue(QUEUE)
    checklist = Q.cmd_checklist(queue)
    assert "grep -n 'WARN' run.log" in checklist
    assert "q_des CLAMP ACTIVE" in checklist
    assert "sha256sum" in checklist
    assert queue["parents"]["W"]["checkpoint_sha256"] in checklist
    assert queue["parents"]["V"]["checkpoint_sha256"] in checklist
    assert ">= 60 s" in checklist
    assert "kit_boot_lock" in checklist
    assert "180 s" in checklist
    assert "_r2" in checklist
    assert "p1btm_w_c_s0_seed3_20260720" in checklist
    # 本波加检条目：qbar 白名单/闸门、spdmix 替换、对照跨 commit diff、Jiayi caveat。
    assert "qdes_limit_barrier" in checklist
    assert "qbar_wiring_confirmed" in checklist
    assert "task.motion.speed_scale_range=[0.8,1.2]" in checklist
    assert "6900/7200/7700/8700/10700" in checklist
    assert "Jiayi" in checklist


def test_cli_plan_and_checklist_succeed_with_placeholder(capsys):
    assert Q.main(["--queue", str(QUEUE)]) == 0
    assert Q.main(["--queue", str(QUEUE), "--checklist"]) == 0
    output = capsys.readouterr().out
    assert "q_des CLAMP ACTIVE" in output
