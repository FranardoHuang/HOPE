from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import shlex
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_balance_temporal_matrix_queue.py"
QUEUE = ROOT / "configs" / "phase1_balance_temporal_matrix_20260720.yaml"
REAL_COMMIT = "a" * 40


def _module():
    spec = importlib.util.spec_from_file_location("btm_queue_under_test", SCRIPT)
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


def _write_yaml(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _load(tmp_path: Path, value: dict) -> dict:
    return Q.load_queue(_write_yaml(tmp_path, value))


def _rendered(tmp_path: Path) -> dict:
    """Checked-in queue with the placeholder commit replaced by a real 40-hex one."""

    value = _raw()
    value["source"]["commit"] = REAL_COMMIT
    return value


def _job(queue: dict, job_id: str) -> dict:
    return Q._job_by_id(queue, job_id)


# ---------------------------------------------------------------- structure


def test_checked_in_queue_validates():
    queue = Q.load_queue(QUEUE)
    assert queue["queue_id"] == "phase1_balance_temporal_matrix_20260720"


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
    del value["watchdog"]
    with pytest.raises(Q.QueueError, match="keys differ"):
        _load(tmp_path, value)


def test_namespace_no_clobber_required(tmp_path):
    value = _raw()
    value["namespace"]["no_clobber"] = False
    with pytest.raises(Q.QueueError, match="no-clobber"):
        _load(tmp_path, value)


def test_watchdog_values_frozen(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["watchdog"]["boot_stall_timeout_s"] == 1800
    assert queue["watchdog"]["post_first_iteration_stall_timeout_s"] == 900
    assert queue["watchdog"]["retry_policy"] == "one_verbatim_retry_suffix_r2"
    value = _raw()
    value["watchdog"]["boot_stall_timeout_s"] = 180
    with pytest.raises(Q.QueueError, match="watchdog"):
        _load(tmp_path, value)


def test_assets_verbatim_from_wave_a(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["assets"] == Q.EXPECTED_ASSETS
    value = _raw()
    value["assets"]["motion_forehand"] = "/workspace/other.npz"
    with pytest.raises(Q.QueueError, match="assets.motion_forehand"):
        _load(tmp_path, value)


def test_parent_recipes_verbatim(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["parents"]["W"]["recipe_overrides"] == Q.EXPECTED_PARENT_RECIPES["W"]
    assert queue["parents"]["V"]["recipe_overrides"] == Q.EXPECTED_PARENT_RECIPES["V"]
    value = _raw()
    value["parents"]["V"]["recipe_overrides"][0] = "task.rewards.racket_position_weight=8.0"
    with pytest.raises(Q.QueueError, match="recipe_overrides"):
        _load(tmp_path, value)


def test_parent_checkpoint_sha256_full_values(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["parents"]["W"]["checkpoint_sha256"] == (
        "2caab3dde3a0ac6c051ff8ac65385a641cac152aa3f84b640126b5ed7b96fcce"
    )
    assert queue["parents"]["V"]["checkpoint_sha256"] == (
        "ad9019100f199f23669829b0fbc4f8c2ad45c8073f930348f177da9487332716"
    )
    value = _raw()
    value["parents"]["W"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(Q.QueueError, match="checkpoint_sha256"):
        _load(tmp_path, value)


def test_parent_null_sha_allowed_but_checklist_keeps_verification(tmp_path):
    value = _raw()
    value["parents"]["W"]["checkpoint_sha256"] = None
    queue = _load(tmp_path, value)
    assert queue["parents"]["W"]["checkpoint_sha256"] is None


def test_seed_must_be_three(tmp_path):
    value = _raw()
    value["common"]["seed"] = 4
    with pytest.raises(Q.QueueError, match="seed"):
        _load(tmp_path, value)


def test_base_overrides_exclude_per_cell_keys():
    queue = Q.load_queue(QUEUE)
    base_keys = {
        raw.split("=", 1)[0].lstrip("+") for raw in queue["common"]["base_overrides"]
    }
    for forbidden in (
        "task.rewards.action_rate_weight",
        "task.rewards.processed_qdes_slew_hinge_weight",
        "task.rewards.post_swing_settle_debt_weight",
        "task.rewards.lower_body_stability_bundle_weight",
        "task.rewards.lower_body_pose_imitation_weight",
    ):
        assert forbidden not in base_keys


def test_base_overrides_owning_factor_key_rejected(tmp_path):
    value = _raw()
    value["common"]["base_overrides"].append("task.rewards.action_rate_weight=-0.1")
    with pytest.raises(Q.QueueError, match="per-cell keys"):
        _load(tmp_path, value)


def test_logger_tensorboard_pinned(tmp_path):
    value = _raw()
    value["common"]["base_overrides"] = [
        raw for raw in value["common"]["base_overrides"] if raw != "logger=tensorboard"
    ]
    with pytest.raises(Q.QueueError, match="logger=tensorboard"):
        _load(tmp_path, value)


# ---------------------------------------------------------------- jobs / matrix


def test_exactly_24_unique_jobs():
    queue = Q.load_queue(QUEUE)
    jobs = queue["jobs"]
    assert len(jobs) == 24
    assert len({job["id"] for job in jobs}) == 24
    assert len({job["run_name"] for job in jobs}) == 24
    assert len({job["run_dir"] for job in jobs}) == 24


def test_t_s_parent_coverage_complete():
    queue = Q.load_queue(QUEUE)
    cells = {(job["parent"], job["temporal"], job["stability"]) for job in queue["jobs"]}
    assert len(cells) == 24
    for parent in ("W", "V"):
        for temporal in ("N", "C", "H"):
            for stability in ("S0", "S1", "S2", "S3"):
                assert (parent, temporal, stability) in cells


def test_every_gpu_holds_exactly_four_jobs():
    queue = Q.load_queue(QUEUE)
    per_slot: dict[tuple[str, int], int] = {}
    for job in queue["jobs"]:
        per_slot[(job["pod"], job["gpu"])] = per_slot.get((job["pod"], job["gpu"]), 0) + 1
    assert per_slot == {slot: 4 for slot in Q.GPU_SLOTS}


def test_frozen_round_robin_assignment(tmp_path):
    queue = Q.load_queue(QUEUE)
    assert queue["jobs"][0]["id"] == "w_n_s0"
    assert (queue["jobs"][0]["pod"], queue["jobs"][0]["gpu"]) == ("pod1", 0)
    assert queue["jobs"][5]["id"] == "v_h_s0"
    assert (queue["jobs"][5]["pod"], queue["jobs"][5]["gpu"]) == ("pod2", 2)
    value = _raw()
    value["jobs"][0], value["jobs"][1] = value["jobs"][1], value["jobs"][0]
    with pytest.raises(Q.QueueError, match="round-robin"):
        _load(tmp_path, value)


def test_gpu_overload_rejected(tmp_path):
    value = _raw()
    value["jobs"][1]["pod"] = "pod1"
    value["jobs"][1]["gpu"] = 0
    with pytest.raises(Q.QueueError, match="more than 4|round-robin|exactly four"):
        _load(tmp_path, value)


def test_duplicate_run_name_rejected(tmp_path):
    value = _raw()
    value["jobs"][1]["run_name"] = value["jobs"][0]["run_name"]
    with pytest.raises(Q.QueueError, match="run_name"):
        _load(tmp_path, value)


def test_run_name_formula_enforced(tmp_path):
    value = _raw()
    value["jobs"][0]["run_name"] = "p1btm_wrong_seed3_20260720"
    with pytest.raises(Q.QueueError, match="run_name must be"):
        _load(tmp_path, value)


def test_run_dir_formula_enforced(tmp_path):
    value = _raw()
    value["jobs"][0]["run_dir"] = "/workspace/somewhere/else"
    with pytest.raises(Q.QueueError, match="run_dir"):
        _load(tmp_path, value)


def test_job_axes_must_match_id(tmp_path):
    value = _raw()
    value["jobs"][0]["stability"] = "S1"
    with pytest.raises(Q.QueueError, match="axes do not match"):
        _load(tmp_path, value)


# ---------------------------------------------------------------- mechanisms


def test_temporal_overrides_exact():
    queue = Q.load_queue(QUEUE)
    temporal = queue["mechanisms"]["temporal"]
    assert temporal["N"]["overrides"] == [
        "task.rewards.action_rate_weight=0.0",
        "++task.rewards.processed_qdes_slew_hinge_weight=0.0",
        "++task.rewards.processed_qdes_slew_hinge_margin=0.85",
        "++task.rewards.processed_qdes_slew_hinge_recovery_start_s=0.2",
        "++task.rewards.processed_qdes_slew_hinge_recovery_end_s=1.55",
    ]
    assert "task.rewards.action_rate_weight=-0.1" in temporal["C"]["overrides"]
    assert (
        "++task.rewards.processed_qdes_slew_hinge_weight=-0.25"
        in temporal["H"]["overrides"]
    )


def test_temporal_weight_sign_error_rejected(tmp_path):
    value = _raw()
    overrides = value["mechanisms"]["temporal"]["C"]["overrides"]
    overrides[0] = "task.rewards.action_rate_weight=0.1"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_stability_overrides_exact_all_levels():
    queue = Q.load_queue(QUEUE)
    stability = queue["mechanisms"]["stability"]
    for name in ("S0", "S1", "S2", "S3"):
        assert stability[name]["overrides"] == Q.EXPECTED_STABILITY[name]


def test_stability_s1_settle_debt_weight():
    queue = Q.load_queue(QUEUE)
    overrides = queue["mechanisms"]["stability"]["S1"]["overrides"]
    assert "++task.rewards.post_swing_settle_debt_weight=-0.25" in overrides
    assert "++task.rewards.lower_body_stability_bundle_weight=0.0" in overrides
    assert "++task.rewards.lower_body_pose_imitation_weight=0.0" in overrides


def test_stability_s1_full_settle_default_params():
    queue = Q.load_queue(QUEUE)
    overrides = queue["mechanisms"]["stability"]["S1"]["overrides"]
    assert "++task.rewards.post_swing_settle_base_lin_margin_mps=0.3" in overrides
    assert "++task.rewards.post_swing_settle_base_lin_scale_mps=0.2" in overrides
    assert "++task.rewards.post_swing_settle_base_ang_margin_radps=0.5" in overrides
    assert "++task.rewards.post_swing_settle_base_ang_scale_radps=0.3" in overrides
    assert "++task.rewards.post_swing_settle_tilt_margin_rad=0.1" in overrides
    assert "++task.rewards.post_swing_settle_tilt_scale_rad=0.1" in overrides
    assert "++task.rewards.post_swing_settle_nominal_root_z_m=1.0684" in overrides
    assert "++task.rewards.post_swing_settle_root_height_deadband_m=0.05" in overrides
    assert "++task.rewards.post_swing_settle_root_height_scale_m=0.05" in overrides
    assert "++task.rewards.post_swing_settle_foot_slip_margin_mps=0.05" in overrides
    assert "++task.rewards.post_swing_settle_foot_slip_scale_mps=0.1" in overrides
    assert "++task.rewards.post_swing_settle_recovery_start_s=0.2" in overrides
    assert "++task.rewards.post_swing_settle_recovery_end_s=1.55" in overrides


def test_stability_weight_sign_error_rejected(tmp_path):
    value = _raw()
    overrides = value["mechanisms"]["stability"]["S1"]["overrides"]
    overrides[0] = "++task.rewards.post_swing_settle_debt_weight=0.25"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_stability_pose_negative_rejected(tmp_path):
    value = _raw()
    overrides = value["mechanisms"]["stability"]["S3"]["overrides"]
    index = overrides.index("++task.rewards.lower_body_pose_imitation_weight=0.5")
    overrides[index] = "++task.rewards.lower_body_pose_imitation_weight=-0.5"
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_stability_mutual_exclusion_rejected(tmp_path):
    value = _raw()
    overrides = copy.deepcopy(Q.EXPECTED_STABILITY["S2"])
    overrides[0] = "++task.rewards.post_swing_settle_debt_weight=-0.25"
    value["mechanisms"]["stability"]["S2"]["overrides"] = overrides
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_no_probe_weight_cli_key_anywhere():
    # train.py 没有 *_probe_weight CLI 键；探针由显式机制 weight 键自动强制为 1.0。
    # 写了这种键会在 boot 时被 _check_unknown_keys(task.rewards) fail-loud 杀掉。
    queue = Q.load_queue(QUEUE)
    for name in ("S0", "S1", "S2", "S3"):
        for raw in queue["mechanisms"]["stability"][name]["overrides"]:
            assert not raw.lstrip("+").split("=", 1)[0].endswith("_probe_weight")


def test_probe_weight_cli_key_rejected(tmp_path):
    value = _raw()
    value["mechanisms"]["stability"]["S0"]["overrides"] = [
        *Q.EXPECTED_STABILITY["S0"],
        "++task.rewards.post_swing_settle_debt_probe_weight=1.0",
    ]
    with pytest.raises(Q.QueueError):
        _load(tmp_path, value)


def test_all_three_mechanism_weight_keys_explicit_in_every_level():
    # 显式 weight 键是三个测量探针开启的开关，每档必须齐全。
    queue = Q.load_queue(QUEUE)
    for name in ("S0", "S1", "S2", "S3"):
        keys = {
            raw.lstrip("+").split("=", 1)[0]
            for raw in queue["mechanisms"]["stability"][name]["overrides"]
        }
        assert "task.rewards.post_swing_settle_debt_weight" in keys
        assert "task.rewards.lower_body_stability_bundle_weight" in keys
        assert "task.rewards.lower_body_pose_imitation_weight" in keys


# ---------------------------------------------------------------- commit gate


def test_placeholder_commit_accepted_for_plan_and_checklist():
    queue = Q.load_queue(QUEUE)
    assert queue["source"]["commit"] == "PENDING_EXACT_COMMIT"
    plan = Q.cmd_plan(queue)
    assert "PENDING_EXACT_COMMIT" in plan
    checklist = Q.cmd_checklist(queue)
    assert "PENDING_EXACT_COMMIT" in checklist


def test_placeholder_commit_refuses_render():
    queue = Q.load_queue(QUEUE)
    job = queue["jobs"][0]
    with pytest.raises(Q.QueueError, match="placeholder"):
        Q.render_command(queue, job, "probe")
    with pytest.raises(Q.QueueError, match="placeholder"):
        Q.render_command(queue, job, "science")


def test_placeholder_commit_refuses_render_via_cli():
    result = Q.main(["--queue", str(QUEUE), "--render-stage", "probe", "--job", "all"])
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
    queue = _load(tmp_path, _rendered(tmp_path))
    command = Q.render_command(queue, _job(queue, "w_n_s0"), "science")
    assert command.startswith("ssh ")
    assert REAL_COMMIT in command


# ---------------------------------------------------------------- rendered argv


def _compiled(queue: dict, job_id: str, stage: str) -> dict[str, str]:
    argv = Q._training_argv(queue, _job(queue, job_id), stage)
    return Q._override_map(argv[2:], "test")


def test_science_command_contains_all_required_keys(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    compiled = _compiled(queue, "w_n_s0", "science")
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
    assert compiled["run_name"] == "p1btm_w_n_s0_seed3_20260720"
    assert compiled["logger"] == "tensorboard"
    assert compiled["device"] == "cuda:0"


def test_probe_budget_in_probe_command(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    compiled = _compiled(queue, "v_c_s2", "probe")
    assert compiled["max_iterations"] == "2"
    assert compiled["algo.runner.save_interval"] == "1"
    assert compiled["num_envs"] == "4096"
    assert compiled["run_name"] == "p1btm_probe_v_c_s2_seed3_20260720"


def test_probe_and_science_namespaces_disjoint(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    job = _job(queue, "w_h_s1")
    probe_dir = Q._stage_run_dir(queue, job, "probe")
    science_dir = Q._stage_run_dir(queue, job, "science")
    assert probe_dir != science_dir
    assert probe_dir.endswith("/probes/w_h_s1")
    assert science_dir.endswith("/runs/w_h_s1")


def test_representative_cell_w_n_s0_overrides_exact(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    compiled = _compiled(queue, "w_n_s0", "science")
    assert compiled["task.rewards.action_rate_weight"] == "0.0"
    assert compiled["task.rewards.processed_qdes_slew_hinge_weight"] == "0.0"
    assert compiled["task.rewards.post_swing_settle_debt_weight"] == "0.0"
    assert compiled["task.rewards.post_swing_settle_recovery_start_s"] == "0.2"
    assert compiled["task.rewards.post_swing_settle_recovery_end_s"] == "1.55"
    assert compiled["task.rewards.lower_body_stability_bundle_weight"] == "0.0"
    assert compiled["task.rewards.lower_body_pose_imitation_weight"] == "0.0"
    assert compiled["task.rewards.racket_position_weight"] == "17.0"
    assert compiled["task.rewards.free_non_striking_arm_mimic"] == "true"


def test_representative_cell_v_h_s3_overrides_exact(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    compiled = _compiled(queue, "v_h_s3", "science")
    assert compiled["task.rewards.action_rate_weight"] == "0.0"
    assert compiled["task.rewards.processed_qdes_slew_hinge_weight"] == "-0.25"
    assert compiled["task.rewards.processed_qdes_slew_hinge_margin"] == "0.85"
    assert compiled["task.rewards.processed_qdes_slew_hinge_recovery_start_s"] == "0.2"
    assert compiled["task.rewards.processed_qdes_slew_hinge_recovery_end_s"] == "1.55"
    assert compiled["task.rewards.post_swing_settle_debt_weight"] == "0.0"
    assert compiled["task.rewards.lower_body_stability_bundle_weight"] == "0.0"
    assert compiled["task.rewards.lower_body_pose_imitation_weight"] == "0.5"
    assert compiled["task.rewards.lower_body_pose_imitation_std"] == "0.35"
    assert compiled["task.rewards.lower_body_pose_imitation_support_pre_s"] == "0.3"
    assert compiled["task.rewards.lower_body_pose_imitation_support_post_s"] == "0.4"
    assert compiled["task.rewards.racket_velocity_weight"] == "17.0"
    assert compiled["task.rewards.free_non_striking_arm_mimic"] == "false"


def test_representative_cell_w_c_s1_overrides_exact(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    compiled = _compiled(queue, "w_c_s1", "science")
    assert compiled["task.rewards.action_rate_weight"] == "-0.1"
    assert compiled["task.rewards.processed_qdes_slew_hinge_weight"] == "0.0"
    assert compiled["task.rewards.post_swing_settle_debt_weight"] == "-0.25"
    assert compiled["task.rewards.lower_body_stability_bundle_weight"] == "0.0"
    assert compiled["task.rewards.lower_body_pose_imitation_weight"] == "0.0"
    assert compiled["task.rewards.lower_body_stability_min_stance_width_m"] == "0.22"
    assert compiled["task.rewards.lower_body_stability_stance_scale_m"] == "0.05"
    assert compiled["task.rewards.lower_body_stability_leg_velocity_margin_radps"] == "1.0"
    assert compiled["task.rewards.lower_body_stability_leg_velocity_scale_radps"] == "0.5"


def test_all_24_cells_compile_without_duplicate_keys(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            argv = Q._training_argv(queue, job, stage)
            Q._override_map(argv[2:], job["id"])  # raises on duplicates


def test_no_deploy_arguments_anywhere(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    for job in queue["jobs"]:
        compiled = _compiled(queue, job["id"], "science")
        assert not (set(compiled) & {"ros", "deploy", "real_robot", "motion_command"})


# ---------------------------------------------------------------- remote command


def test_ssh_command_shell_quote_roundtrip(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    job = _job(queue, "v_c_s3")
    argv = Q._ssh_argv(queue, job, "science")
    rendered = Q.render_command(queue, job, "science")
    assert shlex.split(rendered) == argv


def test_remote_body_quotes_survive_bash_lc_extraction(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    job = _job(queue, "w_n_s2")
    argv = Q._ssh_argv(queue, job, "probe")
    payload = argv[-1]
    assert payload.startswith("bash -lc ")
    inner = shlex.split(payload)
    assert inner[:2] == ["bash", "-lc"]
    body = inner[2]
    assert Q._remote_body(queue, job, "probe") == body


def test_remote_body_uses_kit_boot_lock_not_locked_launcher(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    body = Q._remote_body(queue, _job(queue, "w_n_s0"), "science")
    assert "/workspace/bin/kit_boot_lock.sh" in body
    assert "launch_kit_training_locked.sh" not in body
    assert "setsid nohup" in body
    assert "KIT_BOOT_TIMEOUT_S=1800" in body
    assert "KIT_BOOT_STALE_TIMEOUT_S=900" in body


def test_remote_body_prepares_log_dir_and_log_path(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    job = _job(queue, "v_h_s0")
    body = Q._remote_body(queue, job, "science")
    run_dir = job["run_dir"]
    assert f"mkdir -p {shlex.quote(str(Path(run_dir).parent))}" in body
    assert f"mkdir {shlex.quote(run_dir)}" in body
    assert f"{run_dir}/run.log" in body
    assert f"test ! -e {shlex.quote(run_dir)}" in body


def test_remote_body_pins_gpu_and_pythonpath(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    job = _job(queue, "v_h_s2")
    assert (job["pod"], job["gpu"]) == ("pod2", 2)
    body = Q._remote_body(queue, job, "science")
    assert "CUDA_VISIBLE_DEVICES=2" in body
    assert 'PYTHONPATH="${HOPE_WBT_PYTHONPATH}"' in body
    assert "PYTHONUNBUFFERED=1" in body
    assert "source " in body
    assert "cd " in body
    assert "nvidia-smi -i 2" in body


def test_remote_body_verifies_commit_and_clean_checkout(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    body = Q._remote_body(queue, _job(queue, "w_c_s0"), "probe")
    assert f'test "$(git -C /workspace/codexschema/nohope_btm_20260720 rev-parse HEAD)" = {REAL_COMMIT}' in body
    assert "status --porcelain" in body


def test_ssh_targets_correct_pod(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    pod1_argv = Q._ssh_argv(queue, _job(queue, "w_n_s0"), "science")
    pod2_argv = Q._ssh_argv(queue, _job(queue, "v_c_s0"), "science")
    assert "root@162.43.172.171" in pod1_argv
    assert "18333" in pod1_argv
    assert "root@162.43.172.181" in pod2_argv
    assert "13146" in pod2_argv


def test_render_all_outputs_24_blocks(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    output = Q.cmd_render(queue, "science", "all")
    assert output.count("ssh -i") == 24
    for job in queue["jobs"]:
        assert job["run_name"] in output


def test_render_unknown_job_rejected(tmp_path):
    queue = _load(tmp_path, _rendered(tmp_path))
    with pytest.raises(Q.QueueError, match="unknown job id"):
        Q.cmd_render(queue, "science", "w_x_s9")


# ---------------------------------------------------------------- plan / checklist


def test_plan_lists_all_24_rows_with_human_language():
    queue = Q.load_queue(QUEUE)
    plan = Q.cmd_plan(queue)
    for job in queue["jobs"]:
        assert job["id"] in plan
        assert job["run_name"] in plan
        assert f"{job['pod']}/gpu{job['gpu']}" in plan
    assert "人话" in plan
    assert "action_rate" in plan
    assert "settle_debt" in plan


def test_checklist_contains_warn_and_clamp_capture_lines():
    checklist = Q.cmd_checklist(Q.load_queue(QUEUE))
    assert "WARN" in checklist
    assert "grep -n 'WARN' run.log" in checklist
    assert "q_des CLAMP ACTIVE" in checklist


def test_checklist_contains_parent_sha_verification_and_stagger():
    queue = Q.load_queue(QUEUE)
    checklist = Q.cmd_checklist(queue)
    assert "sha256sum" in checklist
    assert queue["parents"]["W"]["checkpoint_sha256"] in checklist
    assert queue["parents"]["V"]["checkpoint_sha256"] in checklist
    assert ">= 60 s" in checklist
    assert "kit_boot_lock" in checklist
    assert "180 s" in checklist


def test_checklist_flags_placeholder_commit_as_blocking():
    checklist = Q.cmd_checklist(Q.load_queue(QUEUE))
    assert "[阻塞]" in checklist
    assert "PENDING_EXACT_COMMIT" in checklist


def test_checklist_requires_settle_debt_cli_keys_regrep():
    checklist = Q.cmd_checklist(Q.load_queue(QUEUE))
    assert "post_swing_settle_debt_weight" in checklist
    assert "_REWARD_KEYS" in checklist
    assert "probe.weight=1.0" in checklist


def test_cli_plan_and_checklist_succeed_with_placeholder(capsys):
    assert Q.main(["--queue", str(QUEUE)]) == 0
    assert Q.main(["--queue", str(QUEUE), "--checklist"]) == 0
    output = capsys.readouterr().out
    assert "q_des CLAMP ACTIVE" in output
