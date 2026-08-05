"""Cross-launcher contract tests for the exact A211/C211 Isaac four-grid."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = _load("a211_four_grid_contract", "launch_action_ball_a211_four_arm_diagnostic.py")
C = _load("c211_four_grid_contract", "launch_action_ball_c211_diagnostic.py")
F = _load("shared_action_ball_211_four_grid", "action_ball_211_four_grid_contract.py")

TRAINING_CONTRACT_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "training_contract.py"
)


def _load_training_contract():
    spec = importlib.util.spec_from_file_location(
        "four_grid_training_contract_literals", TRAINING_CONTRACT_FILE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a211_c211_share_one_exact_sealed_four_grid_manifest():
    manifest = A._isaac_four_grid_manifest()
    assert manifest == C._isaac_four_grid_manifest()
    unsigned = dict(manifest)
    seal = unsigned.pop("content_sha256")
    assert seal == A.canonical_sha256(unsigned) == C.canonical_sha256(unsigned)
    assert seal == F.CONTENT_SHA256
    assert manifest["cell_order"] == [
        A.A_OBS_NOISE_OFF_CELL_ID,
        A.A_OBS_NOISE_ON_CELL_ID,
        C.C_OBS_NOISE_OFF_CELL_ID,
        C.C_OBS_NOISE_ON_CELL_ID,
    ]
    assert A.ARM_IDS == tuple(manifest["cell_order"][:2])
    assert C.RECIPE_IDS == tuple(manifest["cell_order"][2:])
    # 2026-08-05(第二次改版):第二轴由探索包换成本体感观测噪声开关(exp §5.6.2d)。
    assert manifest["registered_difference_axes"] == [
        "task_semantics_and_reward",
        "policy_observation_corruption_cell",
    ]
    assert manifest["deferred_difference_axes"] == [
        "ppo_learning_rate_schedule_cell",
        "actor_initialization_and_exploration_sigma_cell",
    ]
    assert manifest["adaptive_term_disambiguation"] == {
        "adaptive_means": "ppo_kl_learning_rate_schedule",
        "ppo_kl_learning_rate_schedule": "disabled_fixed_learning_rate_all_cells",
        "contact_kernel_sigma_controller": "disabled_static_all_cells",
        "init_noise_std_is": (
            "static_ppo_action_distribution_initialization_not_a_controller"
        ),
        "policy_observation_corruption_is": (
            "sensor_side_observation_noise_owned_by_the_dr_level_not_the_ppo_recipe"
        ),
    }
    assert manifest["schema_version"] == 4
    assert manifest["kind"].endswith("_v4")
    # cell_id 必须自陈真实变量:四格都写着标准初始化 + sigma1p0,噪声开关一off一on。
    for cell_id in manifest["cell_order"]:
        assert "standard-init-sigma1p0" in cell_id
        assert "zero-weight-bootstrap" not in cell_id
    assert sum("proprio-obs-noise-off" in cell_id for cell_id in manifest["cell_order"]) == 2
    assert sum("proprio-obs-noise-on" in cell_id for cell_id in manifest["cell_order"]) == 2


def test_four_grid_actor_init_mode_literals_match_the_runtime_contract():
    """人话:四格权威里的 actor_init_mode 字面量必须和 train.py 真正认的那两个一字不差。"""

    runtime = _load_training_contract()
    assert (
        F.ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
        == runtime.ACTION_BALL_ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
    )
    assert F.ACTOR_INIT_MODE_DEFAULT == runtime.ACTION_BALL_ACTOR_INIT_MODE_DEFAULT
    assert F.ACTOR_INIT_MODES == tuple(runtime.ACTION_BALL_ACTOR_INIT_MODES)
    assert A.ACTOR_INIT_MODE_DEFAULT == C.ACTOR_INIT_MODE_DEFAULT == "default"
    assert (
        A.ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
        == C.ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
        == "zero_weight_ready_bias"
    )


def test_four_grid_dr_level_literals_match_the_runtime_contract():
    """人话:四格权威里那两个 DR 档身份必须和 training_contract 真正解析出来的一字不差。"""

    runtime = _load_training_contract()
    assert (
        F.DR_LEVEL_IDENTITY_OBS_NOISE_OFF
        == runtime.action_ball_dr_l0_contract_payload()["identity"]
    )
    assert (
        F.DR_LEVEL_IDENTITY_OBS_NOISE_ON
        == runtime.ACTION_BALL_DR_L0N_IDENTITY
        == runtime.action_ball_dr_l0n_contract_payload()["identity"]
    )
    # 通道表也是手抄副本,同样不许漂。
    assert (
        F.PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS
        == runtime.ACTION_BALL_DR_L0N_PROPRIO_NOISE_CHANNELS
    )
    # 这一档只准是"L0 + 传感器":plant 那半边必须逐字节相同。
    l0 = runtime.action_ball_dr_l0_contract_payload()
    l0n = runtime.action_ball_dr_l0n_contract_payload()
    for key in set(l0) | set(l0n):
        if key in runtime.ACTION_BALL_DR_L0N_DECLARED_DIFFERENCES:
            continue
        assert l0[key] == l0n[key], key
    assert l0["policy_observation_corruption"] is False
    assert l0n["policy_observation_corruption"] is True
    assert (
        l0n["proprioceptive_observation_noise"]["task_channel_observation_noise"]
        is False
    )


def test_observation_noise_package_cross_lock_rejects_every_half_set():
    """开关 / 通道表 / DR 身份 / 任务通道无噪必须整包对上,任一处半套都拒。"""

    good = F.manifest()["cells"]
    off = next(row for row in good if not row["policy_observation_corruption"])
    on = next(row for row in good if row["policy_observation_corruption"])
    assert F.validate_observation_noise_package(off)[
        "proprioceptive_observation_noise_channels"
    ] is None
    assert F.validate_observation_noise_package(on)[
        "proprioceptive_observation_noise_channels"
    ] == F.PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS
    off_mutations = (
        # 只把布尔翻成 true 而不给通道表 / 不换 DR 身份 -> 拒
        lambda row: row.__setitem__("policy_observation_corruption", True),
        lambda row: row.__setitem__(
            "dr_level_identity", F.DR_LEVEL_IDENTITY_OBS_NOISE_ON
        ),
        lambda row: row.__setitem__(
            "proprioceptive_observation_noise_channels",
            copy.deepcopy(F.PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS),
        ),
        # 任务通道加噪 -> 拒(改支撑集 = 换题)
        lambda row: row.__setitem__("task_channel_observation_noise", True),
        lambda row: row.__setitem__("observation_noise_axis", "unregistered"),
    )
    for mutate in off_mutations:
        candidate = copy.deepcopy(off)
        mutate(candidate)
        with pytest.raises(F.FourGridContractError):
            F.validate_observation_noise_package(candidate)
    on_mutations = (
        lambda row: row.__setitem__("policy_observation_corruption", False),
        lambda row: row.__setitem__(
            "dr_level_identity", F.DR_LEVEL_IDENTITY_OBS_NOISE_OFF
        ),
        lambda row: row.__setitem__(
            "proprioceptive_observation_noise_channels", None
        ),
        lambda row: row.__setitem__("task_channel_observation_noise", True),
        # 偷偷放宽某一路的幅度 -> 拒
        lambda row: row["proprioceptive_observation_noise_channels"].__setitem__(
            "joint_vel", [-5.0, 5.0]
        ),
        # 偷偷多加一路(比如给任务通道之外的 base 线速度加噪)-> 拒
        lambda row: row["proprioceptive_observation_noise_channels"].__setitem__(
            "actual_base_pose_lin_vel_world", [-0.1, 0.1]
        ),
        # 少一路 -> 拒
        lambda row: row["proprioceptive_observation_noise_channels"].pop("joint_pos"),
    )
    for mutate in on_mutations:
        candidate = copy.deepcopy(on)
        mutate(candidate)
        with pytest.raises(F.FourGridContractError):
            F.validate_observation_noise_package(candidate)


def test_exploration_package_cross_lock_rejects_every_half_set():
    """初始化方式 / sigma / std 参数化 / 4σ 门开关必须整包对上,任一处半套都拒。"""

    zero_weight = copy.deepcopy(
        F.EXPLORATION_PACKAGES[F.ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS]
    )
    standard = copy.deepcopy(
        F.manifest()["matched_contract"]["exploration_package"]
    )
    # 本轮四格全部取标准初始化那一包,零权重路线只是仍然注册、不再被选中。
    assert standard == F.EXPLORATION_PACKAGES[F.ACTOR_INIT_MODE_DEFAULT]
    assert F.validate_exploration_package(zero_weight)["init_noise_std"] == 0.1
    assert F.validate_exploration_package(standard)["init_noise_std"] == 1.0
    mutations = (
        # 只把字面量改成 default 而不改其余三项 -> 拒
        lambda row: row.__setitem__("actor_init_mode", F.ACTOR_INIT_MODE_DEFAULT),
        # 零权重路上把 sigma 抬到 1.0 -> 拒(4σ 门上界 0.1698)
        lambda row: row.__setitem__("init_noise_std", 1.0),
        # 零权重路上声称 4σ 门跳过 -> 拒
        lambda row: row.__setitem__("four_sigma_hard_inner_gate_applies", False),
        lambda row: row.__setitem__("noise_std_type", "scalar"),
    )
    for mutate in mutations:
        candidate = copy.deepcopy(zero_weight)
        mutate(candidate)
        with pytest.raises(F.FourGridContractError):
            F.validate_exploration_package(candidate)
    standard_mutations = (
        # 标准初始化路上声称 4σ 门 applied -> 拒
        lambda row: row.__setitem__("four_sigma_hard_inner_gate_applies", True),
        lambda row: row.__setitem__("noise_std_type", "log"),
        lambda row: row.__setitem__(
            "actor_init_mode", F.ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
        ),
        lambda row: row.__setitem__("init_noise_std", 1.5),
        lambda row: row.__setitem__("exploration_axis", "unregistered"),
    )
    for mutate in standard_mutations:
        candidate = copy.deepcopy(standard)
        mutate(candidate)
        with pytest.raises(F.FourGridContractError):
            F.validate_exploration_package(candidate)


def test_both_launchers_pin_the_same_shared_authority_file_and_content():
    assert A.FOUR_GRID_SOURCE == C.FOUR_GRID_SOURCE
    assert A.FOUR_GRID_FILE.resolve() == C.FOUR_GRID_FILE.resolve()
    assert A.FOUR_GRID_FILE.resolve() == (
        SCRIPTS / "action_ball_211_four_grid_contract.py"
    ).resolve()
    assert sum(path == A.FOUR_GRID_SOURCE for path, _label in A.RUNTIME_SOURCE_PATHS) == 1
    assert sum(path == C.FOUR_GRID_SOURCE for path, _label in C.RUNTIME_SOURCE_PATHS) == 1
    assert hashlib.sha256(A.FOUR_GRID_FILE.read_bytes()).hexdigest() == hashlib.sha256(
        C.FOUR_GRID_FILE.read_bytes()
    ).hexdigest()
    assert A._F.CONTENT_SHA256 == C._F.CONTENT_SHA256 == F.CONTENT_SHA256


def test_shared_authority_rejects_resealed_field_count_order_and_family_drift():
    mutations = (
        lambda value: value["matched_contract"].__setitem__("entropy_coef", 0.02),
        lambda value: value["cells"].pop(),
        lambda value: value["cell_order"].reverse(),
        lambda value: value["cells"][0].__setitem__("task_family", "C211"),
        lambda value: value.__setitem__("unknown_field", True),
    )
    for mutate in mutations:
        candidate = copy.deepcopy(F.manifest())
        mutate(candidate)
        unsigned = dict(candidate)
        unsigned.pop("content_sha256")
        candidate["content_sha256"] = F.canonical_sha256(unsigned)
        with pytest.raises(F.FourGridContractError):
            F.validate_manifest(candidate)


def test_launchers_reject_cross_family_selection_and_local_matched_drift(monkeypatch):
    with pytest.raises(A.LaunchRefused, match="another task family"):
        A._four_grid_cell(C.C_OBS_NOISE_OFF_CELL_ID, task_family="C211")
    with pytest.raises(C.LaunchRefused, match="another task family"):
        C._four_grid_cell(A.A_OBS_NOISE_OFF_CELL_ID, task_family="A211")
    monkeypatch.setattr(A, "TEACHER_ID", "safe-but-wrong-teacher")
    with pytest.raises(A.LaunchRefused, match="four-grid authority differs"):
        A._isaac_four_grid_manifest()


def test_four_grid_cells_match_every_non_registered_setting():
    contracts = [
        A._arm_contract(A.A_OBS_NOISE_OFF_CELL_ID),
        A._arm_contract(A.A_OBS_NOISE_ON_CELL_ID),
        C._recipe_contract(C.C_OBS_NOISE_OFF_CELL_ID),
        C._recipe_contract(C.C_OBS_NOISE_ON_CELL_ID),
    ]
    matched_keys = (
        "soft_weights",
        "actor_hidden_dims",
        "critic_hidden_dims",
        "entropy_coef",
        "reference_guard_mode",
        "contact_sigma_adaptation",
        "ppo",
        "ppo_adaptation_axis",
    )
    for key in matched_keys:
        assert [contract[key] for contract in contracts] == [contracts[0][key]] * 4
    # 第二轴换成探索包之后,四格 PPO 完全相同(保留 A0/C0 原本的保守 fixed lr1e-4)。
    assert contracts[0]["ppo"] == {
        "schedule": "fixed",
        "learning_rate": 1.0e-4,
        "desired_kl": 0.01,
        "clip_param": 0.2,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
    }
    # 探索包本轮**不是**差异轴:四格逐字相同,标准初始化 + sigma 1.0 + scalar。
    exploration = [
        (
            contract["actor_init_mode"],
            contract["init_noise_std"],
            contract["noise_std_type"],
            contract["four_sigma_hard_inner_gate_applies"],
        )
        for contract in contracts
    ]
    assert exploration == [("default", 1.0, "scalar", False)] * 4
    assert len({contract["exploration_axis"] for contract in contracts}) == 1
    # 唯一的注册差异:本体感观测噪声开关(以及随它变的通道表与 DR 档身份)。
    noise = [
        (
            contract["policy_observation_corruption"],
            contract["dr_level_identity"],
            contract["task_channel_observation_noise"],
        )
        for contract in contracts
    ]
    assert noise == [
        (False, F.DR_LEVEL_IDENTITY_OBS_NOISE_OFF, False),
        (True, F.DR_LEVEL_IDENTITY_OBS_NOISE_ON, False),
        (False, F.DR_LEVEL_IDENTITY_OBS_NOISE_OFF, False),
        (True, F.DR_LEVEL_IDENTITY_OBS_NOISE_ON, False),
    ]
    for contract in contracts:
        channels = contract["proprioceptive_observation_noise_channels"]
        if contract["policy_observation_corruption"]:
            assert channels == F.PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS
        else:
            assert channels is None
    # 探索包现在住在 matched_contract 里;cells[i] 上一个探索键都不许留下。
    manifest = A._isaac_four_grid_manifest()
    assert manifest["matched_contract"]["exploration_package"]["init_noise_std"] == 1.0
    assert (
        manifest["matched_contract"]["exploration_axis_is_registered_difference"]
        is False
    )
    for cell in manifest["cells"]:
        for key in F.EXPLORATION_CELL_KEYS:
            assert key not in cell
    matched = manifest["matched_contract"]
    assert matched["wait_contract"] == A._wait_contract() == C._wait_contract()
    assert matched["formal_budgets"] == {
        stage: list(C.BUDGETS[stage]) for stage in C.STAGE_ORDER
    }
    assert all(A.BUDGETS[stage] == C.BUDGETS[stage] for stage in C.STAGE_ORDER)
    assert matched["seed"] == 0
    # 2026-08-05 层级对齐(exp §5.6 第 7 条):death -300.0 -> -10.0。
    assert matched["soft_weights"] == {
        "death_penalty": -10.0,
        "qdes_limit": -5.0,
        "qdes_projection": -5.0,
        "joint_limit": -5.0,
    }
    source = matched["runtime_question_source"]
    assert source["action_id"] == A.ACTION_ID == C.ACTION_ID
    assert source["action_uid"] == A.ACTION_UID == C.ACTION_UID
    assert source["teacher_id"] == A.TEACHER_ID == C.TEACHER_ID
    assert source["source"] == "runtime_curriculum_sampler"
    assert source["cadence"] == "every_episode_reset"
    assert source["sampler_runs_every_reset"] is True
    assert source["zero_physical_rng_draw_claim_permitted"] is False
    assert source["family_target_providers"] == {
        "A211": "online_solver_with_complete_semantic_answer_cache",
        "C211": "direct_ball_no_inverse_no_answer_cache",
    }
    assert "fixed_question" not in matched
    assert "canonical_source_tape" not in repr(manifest)


def test_both_families_keep_exact_scale_long_max_two_gpu_policy():
    assert A.BUDGETS["scale4096"] == C.BUDGETS["scale4096"] == (4096, 5, 1)
    assert A.BUDGETS["long4096"] == C.BUDGETS["long4096"] == (4096, 1000, 100)
    assert A.COLOCATED_STAGES == C.COLOCATED_STAGES == (
        "scale4096",
        "long4096",
    )
    assert A.MAX_COLOCATED_PROCESSES_PER_GPU == 2
    assert C.MAX_COLOCATED_PROCESSES_PER_GPU == 2


def test_formal_grid_samples_each_reset_and_only_a_caches_inverse_answers():
    source = F.manifest()["matched_contract"]["runtime_question_source"]
    assert source["selection"] == "sample_current_domain_levels"
    assert source["curriculum_domain_levels_consulted_every_reset"] is True
    assert source["physical_rng_draw_count_authority"] == (
        "sample_receipt_draw_end_minus_draw_start"
    )
    assert source["shared_ac_question_claim"].endswith("not_one_frozen_question")
    assert A._curriculum_scope_contract()["answer_reuse"] == (
        "complete_semantic_question_sha256_exact_cache"
    )
    assert C._curriculum_scope_contract()["desired_contact_inverse"] is False
    assert C._curriculum_scope_contract()["online_inverse_solve_calls"] == 0


def _canonical_write(path: Path, value) -> None:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _fake_proc_stat(pid: int, starttime: int) -> str:
    fields = ["S"] + ["0"] * 19
    fields[19] = str(starttime)
    return "%d (python worker) %s\n" % (pid, " ".join(fields))


@pytest.mark.parametrize("launcher", (A, C))
def test_both_admissions_scan_one_checkout_local_physical_gpu_root(
    tmp_path: Path, launcher
):
    checkout = tmp_path / launcher.EXPERIMENT_NAME / "checkout"
    lock_path = tmp_path / (launcher.EXPERIMENT_NAME + ".lock")
    coordination_root = Path(
        str(lock_path) + launcher._A.GPU_RESERVATION_REGISTRY_SUFFIX
    )
    namespace = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / launcher.EXPERIMENT_NAME
        / "candidate"
    )
    captured = []

    def reservations(root, **_kwargs):
        captured.append(root)
        return []

    snapshot = launcher._ADMISSION._verify_gpu_admission(
        {
            "source": {"checkout": str(checkout), "commit_sha": "a" * 40},
            "gpu": {
                "index": 2,
                "uuid": "GPU-12345678",
                "lock_path": str(lock_path),
            },
            "namespace": str(namespace),
            launcher.COLOCATION_SPEC_KEY: False,
        },
        phase="pre_launch",
        current_namespace=None,
        query_gpu_processes=lambda *_args: {
            "total_memory_mib": 24576,
            "free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
            "processes": [],
            "nvidia_smi_path": "/usr/bin/nvidia-smi",
            "nvidia_smi_sha256": "b" * 64,
        },
        live_reservations=reservations,
    )
    assert captured == [coordination_root]
    assert snapshot["live_reservation_count"] == 0


@pytest.mark.parametrize("launcher,foreign", ((A, C), (C, A)))
def test_live_same_gpu_foreign_family_reservation_is_never_invisible(
    tmp_path: Path, launcher, foreign
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    coordination_root = tmp_path / "gpu.lock.vendor_v2_reservations"
    coordination_root.mkdir()
    namespace = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / foreign.EXPERIMENT_NAME
        / "live-foreign"
    )
    namespace.mkdir(parents=True)
    pid = 4567
    starttime = 987
    _canonical_write(
        coordination_root / ("b" * 64 + ".json"),
        {
            "schema_version": 1,
            "kind": "measured_vendor_v2_gpu_slot_reservation_v1",
            "owner_pid": pid,
            "owner_proc_starttime_ticks": starttime,
            "gpu_index": 2,
            "gpu_uuid": "GPU-12345678",
            "namespace": str(namespace),
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "launch_claim_sha256": "b" * 64,
            "max_compute_pids": 2,
            "minimum_free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
            "allow_vendor_v2_colocation": True,
        },
    )
    proc_root = tmp_path / "proc"
    pid_root = proc_root / str(pid)
    pid_root.mkdir(parents=True)
    (pid_root / "stat").write_text(
        _fake_proc_stat(pid, starttime), encoding="ascii"
    )
    with pytest.raises(
        launcher.LaunchRefused, match="belongs to another experiment family"
    ):
        launcher._ADMISSION._live_reservations(
            coordination_root,
            checkout=checkout,
            commit="a" * 40,
            gpu_index=2,
            gpu_uuid="GPU-12345678",
            proc_root=proc_root,
        )


@pytest.mark.parametrize(
    "launcher,cells,gpu_index,gpu_uuid",
    (
        (A, A.ARM_IDS, 0, "GPU-A0000000"),
        (C, C.RECIPE_IDS, 1, "GPU-C1111111"),
    ),
)
def test_planned_same_family_two_process_colocation_remains_available(
    tmp_path: Path, launcher, cells, gpu_index: int, gpu_uuid: str
):
    assert len(cells) == 2
    assert launcher.COLOCATED_STAGES == ("scale4096", "long4096")
    assert launcher.BUDGETS["scale4096"] == (4096, 5, 1)
    checkout = tmp_path / "checkout"
    lock_path = tmp_path / ("gpu%d.lock" % gpu_index)
    coordination_root = Path(
        str(lock_path) + launcher._A.GPU_RESERVATION_REGISTRY_SUFFIX
    )
    experiment_root = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / launcher.EXPERIMENT_NAME
    )
    first = experiment_root / cells[0]
    second = experiment_root / cells[1]
    existing = {
        "owner_pid": 1001,
        "owner_proc_starttime_ticks": 2001,
        "reservation_owner_kind": "outer_launcher",
        "namespace": str(first),
        "reservation_receipt": {"path": "/reservation", "sha256": "d" * 64},
        "allow_vendor_v2_colocation": True,
    }
    spec = {
        "source": {"checkout": str(checkout), "commit_sha": "a" * 40},
        "gpu": {
            "index": gpu_index,
            "uuid": gpu_uuid,
            "lock_path": str(lock_path),
        },
        "namespace": str(second),
        launcher.COLOCATION_SPEC_KEY: True,
    }
    query = lambda *_args: {
        "total_memory_mib": 24576,
        "free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
        "processes": [],
        "nvidia_smi_path": "/usr/bin/nvidia-smi",
        "nvidia_smi_sha256": "e" * 64,
    }
    snapshot = launcher._ADMISSION._verify_gpu_admission(
        spec,
        phase="pre_launch",
        current_namespace=None,
        query_gpu_processes=query,
        live_reservations=lambda root, **_kwargs: (
            [existing] if root == coordination_root else pytest.fail("wrong root")
        ),
    )
    assert snapshot["gpu_index"] == gpu_index
    assert snapshot["live_reservation_count"] == 1
    assert snapshot["allow_vendor_v2_colocation"] is True

    third = dict(existing)
    third["namespace"] = str(
        experiment_root / "already-second"
    )
    with pytest.raises(launcher.LaunchRefused, match="no free compute-PID slot"):
        launcher._ADMISSION._verify_gpu_admission(
            spec,
            phase="pre_launch",
            current_namespace=None,
            query_gpu_processes=query,
            live_reservations=lambda *_args, **_kwargs: [existing, third],
        )


@pytest.mark.parametrize("launcher", (A, C))
def test_current_layout_preflight_rejects_live_legacy_vendor_v2_pending(
    tmp_path: Path, launcher, monkeypatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    lock_path = tmp_path / "gpu0.lock"
    old_experiment = (
        "agibot_a3_action_ball_measured_vendor_v2_n1_diagnostic"
    )
    old_namespace = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / old_experiment
        / "live-old"
    )
    old_namespace.mkdir(parents=True)
    pid = 8123
    starttime = 9001
    _canonical_write(
        old_namespace / launcher._A.GPU_RESERVATION_FILENAME,
        {
            "schema_version": 1,
            "kind": "measured_vendor_v2_gpu_slot_reservation_v1",
            "owner_pid": pid,
            "owner_proc_starttime_ticks": starttime,
            "gpu_index": 0,
            "gpu_uuid": "GPU-A0000000",
            "namespace": str(old_namespace),
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "launch_claim_sha256": "f" * 64,
            "max_compute_pids": 2,
            "minimum_free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
            "allow_vendor_v2_colocation": True,
        },
    )
    proc_root = tmp_path / "proc"
    stat_path = proc_root / str(pid) / "stat"
    stat_path.parent.mkdir(parents=True)
    stat_path.write_text(_fake_proc_stat(pid, starttime), encoding="ascii")
    current_namespace = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / launcher.EXPERIMENT_NAME
        / "candidate"
    )
    monkeypatch.setattr(
        launcher,
        "_query_gpu_processes",
        lambda *_args: {
            "total_memory_mib": 24576,
            "free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
            "processes": [],
            "nvidia_smi_path": "/usr/bin/nvidia-smi",
            "nvidia_smi_sha256": "e" * 64,
        },
    )
    with pytest.raises(
        launcher.LaunchRefused, match="belongs to another experiment family"
    ):
        launcher._verify_gpu_admission(
            {
                "source": {
                    "checkout": str(checkout),
                    "commit_sha": "a" * 40,
                },
                "gpu": {
                    "index": 0,
                    "uuid": "GPU-A0000000",
                    "lock_path": str(lock_path),
                },
                "namespace": str(current_namespace),
                launcher.COLOCATION_SPEC_KEY: False,
            },
            phase="pre_launch",
            current_namespace=None,
            proc_root=proc_root,
        )
