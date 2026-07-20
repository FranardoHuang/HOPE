from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import shlex
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_push_robustness_queue.py"
QUEUE = ROOT / "configs" / "phase1_push_robustness_20260721.yaml"
MATRIX_QUEUE = ROOT / "configs" / "phase1_balance_temporal_matrix_20260720.yaml"
REAL_COMMIT = "b" * 40


def _module():
    spec = importlib.util.spec_from_file_location("push_queue_under_test", SCRIPT)
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


def _job(queue: dict, job_id: str) -> dict:
    return Q._job_by_id(queue, job_id)


def _compiled(queue: dict, job_id: str, stage: str) -> dict[str, str]:
    argv = Q._training_argv(queue, _job(queue, job_id), stage)
    return Q._override_map(argv[2:], "test")


# ---------------------------------------------------------------- structure


def test_checked_in_queue_validates():
    queue = Q.load_queue(QUEUE)
    assert queue["queue_id"] == "phase1_push_robustness_20260721"


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
    del value["controls"]
    with pytest.raises(Q.QueueError, match="keys differ"):
        _load(tmp_path, value)


def test_watchdog_and_namespace_inherited_verbatim(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["watchdog"]["boot_stall_timeout_s"] == 1800
    assert queue["watchdog"]["post_first_iteration_stall_timeout_s"] == 900
    assert queue["watchdog"]["retry_policy"] == "one_verbatim_retry_suffix_r2"
    assert queue["namespace"]["root"] == (
        "/workspace/codexschema/phase1_push_robustness_20260721"
    )
    assert queue["namespace"]["no_clobber"] is True
    value = _raw()
    value["namespace"]["no_clobber"] = False
    with pytest.raises(Q.QueueError, match="no-clobber"):
        _load(tmp_path, value)


# ---------------------------------------------------------------- cross-check vs matrix yaml


def test_base_overrides_verbatim_from_matrix_yaml():
    # 防漂移：base 配方必须与正在跑的矩阵 yaml 逐字一致（w_c_s0/v_c_s0 才够格当对照）。
    queue = Q.load_queue(QUEUE)
    matrix = _matrix_raw()
    assert queue["common"]["base_overrides"] == matrix["common"]["base_overrides"]
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


def test_temporal_c_recipe_verbatim_from_matrix_yaml():
    queue = Q.load_queue(QUEUE)
    matrix = _matrix_raw()
    assert queue["recipe"]["temporal_c"]["overrides"] == (
        matrix["mechanisms"]["temporal"]["C"]["overrides"]
    )


def test_stability_s0_recipe_verbatim_from_matrix_yaml():
    queue = Q.load_queue(QUEUE)
    matrix = _matrix_raw()
    assert queue["recipe"]["stability_s0"]["overrides"] == (
        matrix["mechanisms"]["stability"]["S0"]["overrides"]
    )


def test_temporal_c_drift_rejected(tmp_path):
    value = _raw()
    value["recipe"]["temporal_c"]["overrides"][0] = "task.rewards.action_rate_weight=-0.2"
    with pytest.raises(Q.QueueError, match="temporal_c"):
        _load(tmp_path, value)


def test_stability_s0_drift_rejected(tmp_path):
    value = _raw()
    overrides = value["recipe"]["stability_s0"]["overrides"]
    index = overrides.index("++task.rewards.post_swing_settle_debt_weight=0.0")
    overrides[index] = "++task.rewards.post_swing_settle_debt_weight=-0.25"
    with pytest.raises(Q.QueueError, match="stability_s0"):
        _load(tmp_path, value)


def test_controls_point_at_matrix_no_push_baseline(tmp_path):
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


def test_exactly_12_unique_jobs():
    queue = Q.load_queue(QUEUE)
    jobs = queue["jobs"]
    assert len(jobs) == 12
    assert len({job["id"] for job in jobs}) == 12
    assert len({job["run_name"] for job in jobs}) == 12
    assert len({job["run_dir"] for job in jobs}) == 12


def test_parent_push_coverage_complete():
    queue = Q.load_queue(QUEUE)
    cells = {(job["parent"], job["push"]) for job in queue["jobs"]}
    assert len(cells) == 12
    for parent in ("W", "V"):
        for push in ("p02", "p035", "p05", "yaw", "ang", "fast"):
            assert (parent, push) in cells


def test_jobs_carry_no_pod_or_gpu():
    # 槽位策略：本波空槽即填，yaml 不写死 pod/gpu，渲染时注入。
    queue = Q.load_queue(QUEUE)
    for job in queue["jobs"]:
        assert "pod" not in job
        assert "gpu" not in job


def test_run_name_formula_enforced(tmp_path):
    queue = Q.load_queue(QUEUE)
    for job in queue["jobs"]:
        assert job["run_name"] == f"p1push_{job['id']}_seed3_20260721"
    value = _raw()
    value["jobs"][0]["run_name"] = "p1push_wrong_seed3_20260721"
    with pytest.raises(Q.QueueError, match="run_name must be"):
        _load(tmp_path, value)


def test_run_dir_formula_enforced(tmp_path):
    value = _raw()
    value["jobs"][0]["run_dir"] = "/workspace/somewhere/else"
    with pytest.raises(Q.QueueError, match="run_dir"):
        _load(tmp_path, value)


def test_duplicate_run_name_rejected(tmp_path):
    value = _raw()
    value["jobs"][1]["run_name"] = value["jobs"][0]["run_name"]
    with pytest.raises(Q.QueueError, match="run_name"):
        _load(tmp_path, value)


def test_job_axes_must_match_id(tmp_path):
    value = _raw()
    value["jobs"][0]["push"] = "p05"
    with pytest.raises(Q.QueueError, match="axes do not match"):
        _load(tmp_path, value)


def test_launch_order_frozen_and_interleaved(tmp_path):
    queue = Q.load_queue(QUEUE)
    order = queue["launch_order"]
    assert order == list(Q.LAUNCH_ORDER)
    assert sorted(order) == sorted(job["id"] for job in queue["jobs"])
    # parent 全程交错，前 6 位覆盖全部 6 档。
    parents = [job_id.split("_", 1)[0] for job_id in order]
    for left, right in zip(parents, parents[1:]):
        assert left != right
    assert {job_id.split("_", 1)[1] for job_id in order[:6]} == {
        "p02", "p035", "p05", "yaw", "ang", "fast",
    }
    value = _raw()
    value["launch_order"][0], value["launch_order"][1] = (
        value["launch_order"][1], value["launch_order"][0]
    )
    with pytest.raises(Q.QueueError, match="launch_order"):
        _load(tmp_path, value)


# ---------------------------------------------------------------- push overrides


def test_push_p05_overrides_exact_verbatim():
    queue = Q.load_queue(QUEUE)
    assert queue["mechanisms"]["push"]["p05"]["overrides"] == [
        "++task.push.enable=true",
        "++task.push.interval_range_s=[5.0,15.0]",
        "++task.push.vel_xy_mps=0.5",
        "++task.push.ang_vel_radps=0.0",
        "++task.push.ang_axes=none",
    ]


def test_push_ang_overrides_exact_verbatim():
    queue = Q.load_queue(QUEUE)
    assert queue["mechanisms"]["push"]["ang"]["overrides"] == [
        "++task.push.enable=true",
        "++task.push.interval_range_s=[5.0,15.0]",
        "++task.push.vel_xy_mps=0.35",
        "++task.push.ang_vel_radps=0.5",
        "++task.push.ang_axes=rpy",
    ]


def test_push_levels_match_frozen_design():
    queue = Q.load_queue(QUEUE)
    compiled = _compiled(queue, "w_p02", "science")
    assert compiled["task.push.vel_xy_mps"] == "0.2"
    assert compiled["task.push.interval_range_s"] == "[5.0,15.0]"
    assert compiled["task.push.ang_axes"] == "none"
    compiled = _compiled(queue, "v_p035", "science")
    assert compiled["task.push.vel_xy_mps"] == "0.35"
    assert compiled["task.push.ang_vel_radps"] == "0.0"
    compiled = _compiled(queue, "w_yaw", "science")
    assert compiled["task.push.ang_axes"] == "yaw"
    assert compiled["task.push.ang_vel_radps"] == "0.5"
    compiled = _compiled(queue, "v_fast", "science")
    assert compiled["task.push.interval_range_s"] == "[1.0,3.0]"
    assert compiled["task.push.vel_xy_mps"] == "0.35"
    assert compiled["task.push.ang_axes"] == "none"


def test_push_keys_match_train_py_whitelist():
    # 键名真源交叉断言：yaml 用到的 task.push.* 键必须逐字等于工作树 train.py
    # 落盘的 _PUSH_KEYS 白名单（wiring 是并行作业，防两边漂移）。
    train_py = ROOT / "hope_training" / "whole_body_tracking" / "scripts" / "train.py"
    text = train_py.read_text(encoding="utf-8")
    import re

    match = re.search(r"_PUSH_KEYS\s*=\s*\(([^)]*)\)", text)
    assert match is not None, "train.py no longer defines _PUSH_KEYS"
    whitelist = {item.strip().strip('"\'') for item in match.group(1).split(",") if item.strip()}
    assert whitelist == {
        "enable", "interval_range_s", "vel_xy_mps", "ang_vel_radps", "ang_axes",
    }
    queue = Q.load_queue(QUEUE)
    for level in ("p02", "p035", "p05", "yaw", "ang", "fast"):
        used = {
            raw.split("=", 1)[0].lstrip("+").removeprefix("task.push.")
            for raw in queue["mechanisms"]["push"][level]["overrides"]
        }
        assert used == whitelist


def test_push_enabled_in_every_arm_both_stages():
    queue = Q.load_queue(QUEUE)
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            compiled = _compiled(queue, job["id"], stage)
            assert compiled["task.push.enable"] == "true"
            for key in (
                "task.push.interval_range_s", "task.push.vel_xy_mps",
                "task.push.ang_vel_radps", "task.push.ang_axes",
            ):
                assert key in compiled


def test_push_negative_amplitude_rejected(tmp_path):
    value = _raw()
    overrides = value["mechanisms"]["push"]["p02"]["overrides"]
    index = overrides.index("++task.push.vel_xy_mps=0.2")
    overrides[index] = "++task.push.vel_xy_mps=-0.2"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_push_axes_amplitude_mismatch_rejected(tmp_path):
    # ang_axes=yaw 配 ang_vel_radps=0（train.py 合同同款 fail-closed 组合）。
    value = _raw()
    overrides = value["mechanisms"]["push"]["yaw"]["overrides"]
    index = overrides.index("++task.push.ang_vel_radps=0.5")
    overrides[index] = "++task.push.ang_vel_radps=0.0"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_push_disable_rejected(tmp_path):
    value = _raw()
    overrides = value["mechanisms"]["push"]["fast"]["overrides"]
    overrides[0] = "++task.push.enable=false"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_push_level_drift_rejected(tmp_path):
    value = _raw()
    overrides = value["mechanisms"]["push"]["p05"]["overrides"]
    index = overrides.index("++task.push.vel_xy_mps=0.5")
    overrides[index] = "++task.push.vel_xy_mps=0.6"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


# ---------------------------------------------------------------- commit gate


def _placeholder_path(tmp_path):
    value = _raw()
    value["source"]["commit"] = "PENDING_EXACT_COMMIT"
    return _write_yaml(tmp_path, value)


def test_checked_in_commit_is_exact_40_hex():
    # 冻结后必须是 40-hex；占位语义由 tmp fixture 继续覆盖。
    queue = Q.load_queue(QUEUE)
    assert re.fullmatch(r"[0-9a-f]{40}", queue["source"]["commit"])


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
        "--render-job", "w_p02", "--pod", "pod1", "--gpu", "0",
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
    command = Q.render_command(queue, _job(queue, "w_p02"), "science", "pod1", 0)
    assert command.startswith("ssh ")
    assert REAL_COMMIT in command


# ---------------------------------------------------------------- pod/gpu injection


def test_pod_gpu_injected_at_render_time(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "v_ang")
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
    job = _job(queue, "w_fast")
    for pod in ("pod1", "pod2"):
        for gpu in (0, 1, 2):
            command = Q.render_command(queue, job, "probe", pod, gpu)
            assert f"CUDA_VISIBLE_DEVICES={gpu}" in command


def test_invalid_pod_rejected(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "w_p02")
    with pytest.raises(Q.QueueError, match="--pod"):
        Q.render_command(queue, job, "science", "pod3", 0)


def test_invalid_gpu_rejected(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "w_p02")
    with pytest.raises(Q.QueueError, match="--gpu"):
        Q.render_command(queue, job, "science", "pod1", 3)
    with pytest.raises(Q.QueueError, match="--gpu"):
        Q.render_command(queue, job, "science", "pod1", "junk")
    with pytest.raises(Q.QueueError, match="--gpu"):
        Q.render_command(queue, job, "science", "pod1", -1)


def test_cli_render_requires_pod_and_gpu():
    result = Q.main([
        "--queue", str(QUEUE), "--render-stage", "probe", "--render-job", "w_p02",
    ])
    assert result == 2


def test_render_unknown_job_rejected(tmp_path):
    queue = _load(tmp_path, _rendered())
    with pytest.raises(Q.QueueError, match="unknown job id"):
        Q.cmd_render(queue, "science", "w_p99", "pod1", 0)


def test_gpu_precheck_under_four_procs_inherited(tmp_path):
    # 逐字继承矩阵的"同卡 <4 个 compute 进程"预检。
    queue = _load(tmp_path, _rendered())
    body = Q._remote_body(queue, _job(queue, "w_p02"), "science", 2)
    assert "--query-compute-apps=pid" in body
    assert '-lt 4' in body


# ---------------------------------------------------------------- rendered argv


def test_science_command_contains_all_required_keys(tmp_path):
    queue = _load(tmp_path, _rendered())
    compiled = _compiled(queue, "w_p02", "science")
    parent = queue["parents"]["W"]
    assert compiled["checkpoint_path"] == parent["checkpoint_path"]
    assert compiled["checkpoint_tolerant"] == "false"
    assert compiled["checkpoint_allow_missing_contract"] == "false"
    assert compiled["checkpoint_allow_contract_mismatch"] == "true"
    assert compiled["seed"] == "3"
    assert compiled["num_envs"] == "4096"
    assert compiled["algo.runner.num_steps_per_env"] == "24"
    assert compiled["max_iterations"] == "10001"
    assert compiled["algo.runner.save_interval"] == "100"
    assert compiled["run_name"] == "p1push_w_p02_seed3_20260721"
    assert compiled["logger"] == "tensorboard"
    assert compiled["device"] == "cuda:0"


def test_c_s0_recipe_present_in_every_science_command(tmp_path):
    queue = _load(tmp_path, _rendered())
    for job in queue["jobs"]:
        compiled = _compiled(queue, job["id"], "science")
        assert compiled["task.rewards.action_rate_weight"] == "-0.1"
        assert compiled["task.rewards.processed_qdes_slew_hinge_weight"] == "0.0"
        assert compiled["task.rewards.post_swing_settle_debt_weight"] == "0.0"
        assert compiled["task.rewards.lower_body_stability_bundle_weight"] == "0.0"
        assert compiled["task.rewards.lower_body_pose_imitation_weight"] == "0.0"


def test_probe_budget_and_disjoint_namespaces(tmp_path):
    queue = _load(tmp_path, _rendered())
    compiled = _compiled(queue, "v_p05", "probe")
    assert compiled["max_iterations"] == "2"
    assert compiled["algo.runner.save_interval"] == "1"
    assert compiled["run_name"] == "p1push_probe_v_p05_seed3_20260721"
    job = _job(queue, "v_p05")
    probe_dir = Q._stage_run_dir(queue, job, "probe")
    science_dir = Q._stage_run_dir(queue, job, "science")
    assert probe_dir != science_dir
    assert probe_dir.endswith("/probes/v_p05")
    assert science_dir.endswith("/runs/v_p05")


def test_all_12_arms_compile_without_duplicate_keys(tmp_path):
    queue = _load(tmp_path, _rendered())
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            argv = Q._training_argv(queue, job, stage)
            Q._override_map(argv[2:], job["id"])  # raises on duplicates


def test_no_deploy_arguments_anywhere(tmp_path):
    queue = _load(tmp_path, _rendered())
    for job in queue["jobs"]:
        compiled = _compiled(queue, job["id"], "science")
        assert not (set(compiled) & {"ros", "deploy", "real_robot", "motion_command"})


# ---------------------------------------------------------------- remote command


def test_ssh_command_shell_quote_roundtrip(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "v_yaw")
    argv = Q._ssh_argv(queue, job, "science", "pod2", 1)
    rendered = Q.render_command(queue, job, "science", "pod2", 1)
    assert shlex.split(rendered) == argv


def test_remote_body_quotes_survive_bash_lc_extraction(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "w_ang")
    argv = Q._ssh_argv(queue, job, "probe", "pod1", 2)
    payload = argv[-1]
    assert payload.startswith("bash -lc ")
    inner = shlex.split(payload)
    assert inner[:2] == ["bash", "-lc"]
    assert Q._remote_body(queue, job, "probe", 2) == inner[2]


def test_remote_body_uses_kit_boot_lock_not_locked_launcher(tmp_path):
    queue = _load(tmp_path, _rendered())
    body = Q._remote_body(queue, _job(queue, "w_p02"), "science", 0)
    assert "/workspace/bin/kit_boot_lock.sh" in body
    assert "launch_kit_training_locked.sh" not in body
    assert "setsid nohup" in body
    assert "KIT_BOOT_TIMEOUT_S=1800" in body
    assert "KIT_BOOT_STALE_TIMEOUT_S=900" in body


def test_remote_body_verifies_commit_and_no_clobber(tmp_path):
    queue = _load(tmp_path, _rendered())
    job = _job(queue, "v_p02")
    body = Q._remote_body(queue, job, "science", 0)
    assert (
        'test "$(git -C /workspace/codexschema/nohope_push_20260721 rev-parse HEAD)" = '
        + REAL_COMMIT
    ) in body
    assert "status --porcelain" in body
    assert f"test ! -e {shlex.quote(job['run_dir'])}" in body


# ---------------------------------------------------------------- plan / checklist


def test_plan_lists_12_rows_and_fill_order():
    queue = Q.load_queue(QUEUE)
    plan = Q.cmd_plan(queue)
    for job in queue["jobs"]:
        assert job["id"] in plan
        assert job["run_name"] in plan
    assert "人话" in plan
    assert "推荐填充顺序" in plan
    assert "w_c_s0/v_c_s0" in plan
    assert "axes=rpy" in plan
    assert "axes=yaw" in plan
    assert "interval=1.0-3.0s" in plan
    for index, job_id in enumerate(queue["launch_order"], start=1):
        assert f"{index:2d}. {job_id}" in plan


def test_checklist_contains_inherited_guards_and_push_wiring_check():
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
    assert "_PUSH_KEYS" in checklist
    assert "ang_axes" in checklist
    assert "task.push" in checklist
    assert "p1btm_w_c_s0_seed3_20260720" in checklist
    assert "_r2" in checklist


def test_cli_plan_and_checklist_succeed_with_placeholder(capsys):
    assert Q.main(["--queue", str(QUEUE)]) == 0
    assert Q.main(["--queue", str(QUEUE), "--checklist"]) == 0
    output = capsys.readouterr().out
    assert "q_des CLAMP ACTIVE" in output
