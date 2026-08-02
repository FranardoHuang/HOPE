from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_stage1_natural_clip_a3_vendor.py"
)
ROOT = Path(__file__).resolve().parents[1]
OBS_CONTRACT = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/"
    "actor_observation_contract.py"
)
ENV_CFG = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/"
    "agibot_a3/hope_env_cfg.py"
)
SPEC = importlib.util.spec_from_file_location(
    "launch_stage1_natural_clip_a3_vendor", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)


def _argv_map(argv: list[str]) -> dict[str, str]:
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in argv[2:]
        if "=" in item
    }


def _class_literal(path: Path, class_name: str, attribute: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            target = None
            value = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                value = statement.value
            elif isinstance(statement, ast.AnnAssign):
                target = statement.target
                value = statement.value
            if (
                isinstance(target, ast.Name)
                and target.id == attribute
                and value is not None
            ):
                return ast.literal_eval(value)
    raise AssertionError(
        f"{class_name}.{attribute} has no class-level literal in {path}"
    )


def test_launcher_consumes_exact_three_unique_code_owned_lanes() -> None:
    lanes = tuple(L.STAGE1_NATURAL_CLIP_LANES)
    assert len(lanes) == 3
    assert tuple(lane.lane_id for lane in lanes) == (
        "bh_quality_take061_unit15",
        "fh_stable_take058_unit04",
        "bh_diverse_take060_unit09",
    )
    for values in (
        tuple(lane.motion_path for lane in lanes),
        tuple(lane.motion_sha256 for lane in lanes),
        tuple(lane.strike_phase for lane in lanes),
    ):
        assert len(values) == len(set(values))
    assert tuple(L.LANE_SEEDS[lane.lane_id] for lane in lanes) == (0, 1, 2)


@pytest.mark.parametrize(
    ("stage", "expected_budget"),
    (
        ("smoke", (1, 2, 1)),
        ("probe", (4096, 5, 1)),
        ("long", (4096, 20_001, 100)),
    ),
)
def test_exact_stage_budgets_and_save_intervals(
    tmp_path: Path, stage: str, expected_budget: tuple[int, int, int]
) -> None:
    payload = L.build_launch_payload(
        stage=stage,
        lane_id="bh_quality_take061_unit15",
        root=tmp_path,
        gpu=3,
    )
    budget = payload["spec"]["budget"]
    assert (
        budget["num_envs"],
        budget["max_iterations"],
        budget["save_interval"],
    ) == expected_budget
    argv = _argv_map(payload["argv"])
    assert argv["num_envs"] == str(expected_budget[0])
    assert argv["max_iterations"] == str(expected_budget[1])
    assert argv["algo.runner.save_interval"] == str(expected_budget[2])


@pytest.mark.parametrize(
    "lane",
    tuple(lane.lane_id for lane in L.STAGE1_NATURAL_CLIP_LANES),
)
def test_lane_argv_is_local_single_clip_registry_null_and_diagnostic(
    tmp_path: Path, lane: str
) -> None:
    payload = L.build_launch_payload(
        stage="probe", lane_id=lane, root=tmp_path, gpu=2
    )
    spec = payload["spec"]
    argv = _argv_map(payload["argv"])
    expected = L.STAGE1_NATURAL_CLIP_LANES_BY_ID[lane]

    assert argv["task"] == L.TASK_PROFILE_ID
    assert json.loads(argv["motion_file"]) == [
        spec["lane"]["motion_absolute_path"]
    ]
    assert Path(spec["lane"]["motion_absolute_path"]).is_absolute()
    assert spec["lane"]["motion_path"] == expected.motion_path
    assert spec["lane"]["motion_sha256"] == expected.motion_sha256
    assert spec["lane"]["strike_phase"] == expected.strike_phase
    assert json.loads(argv["task.racket.clip_names"]) == [expected.action_id]
    assert argv["motion_file_2"] == "null"
    assert argv["registry_name"] == "null"
    assert argv["registry_name_2"] == "null"
    assert argv["task.registry_name"] == "null"
    assert argv["task.registry_name_2"] == "null"
    assert argv["task.racket.action_ball_diagnostic_unauthorized"] == "true"
    assert spec["diagnostic_unauthorized"] is True
    assert not any(spec["authorization"].values())
    assert argv["device"] == "cuda:0"
    assert argv["algo.policy.init_noise_std"] == "0.02"
    assert argv["algo.policy.noise_std_type"] == "log"
    assert spec["physical_gpu"] == 2
    assert spec["cuda_visible_devices"] == "2"
    assert argv["seed"] == str(L.LANE_SEEDS[lane])
    assert f"_{lane}_seed{L.LANE_SEEDS[lane]}_" in spec["run_name"]


def test_active_profile_binds_v2_actor_and_full_phase_nine_term_paddle_recipe(
) -> None:
    task = yaml.safe_load(L._TASK_FILE.read_text(encoding="utf-8"))
    assert task["name"] == L.TASK_PROFILE_ID
    assert task["actor_obs_contract"] == (
        "stage1_natural_clip_paddle_world_v2"
    )
    assert task["rewards"]["full_body_mimic"] is True

    racket = task["racket"]
    assert racket["target_mode"] == "reference_perturbed"
    assert racket["adaptive_sigma"] is True
    assert racket["adaptive_sigma_monotonic"] is True
    assert racket["adaptive_sigma_normal"] is True
    assert racket["adaptive_sigma_source"] == (
        "stage1_clip_site_full_phase_rms"
    )
    assert (
        racket["sigma_pos_max"],
        racket["sigma_vel_max"],
        racket["sigma_normal_max"],
    ) == pytest.approx((0.50, 3.0, 2.10))

    expected_site_recipe = {
        "racket_position": (0.90, 0.50),
        "racket_position_coarse": (0.30, 0.70),
        "racket_velocity": (0.45, 3.0),
        "racket_velocity_coarse": (0.15, 4.0),
        "racket_normal": (0.90, 2.10),
        "racket_normal_coarse": (0.30, 3.141592653589793),
        "racket_position_precision": (0.50, 0.075),
        "racket_velocity_precision": (0.25, 0.50),
        "racket_normal_precision": (0.50, 0.262),
    }
    rewards = task["rewards"]
    for name, (weight, std) in expected_site_recipe.items():
        assert rewards[f"{name}_weight"] == pytest.approx(weight)
        assert rewards[f"{name}_std"] == pytest.approx(std)

    contract_spec = importlib.util.spec_from_file_location(
        "stage1_actor_observation_contract_under_launcher_test",
        OBS_CONTRACT,
    )
    assert contract_spec is not None and contract_spec.loader is not None
    contract_module = importlib.util.module_from_spec(contract_spec)
    sys.modules[contract_spec.name] = contract_module
    contract_spec.loader.exec_module(contract_module)
    contract = contract_module.resolve_actor_observation_contract(
        task["actor_obs_contract"]
    )
    assert contract.name == "stage1_natural_clip_paddle_world_v2"
    assert contract.obs_mode == "stage1_natural_clip_paddle_world"
    assert contract.total_dim == 225
    assert _class_literal(
        ENV_CFG,
        "HOPEPingPongStage1NaturalClipV2AgibotA3EnvCfg",
        "obs_mode",
    ) == contract.obs_mode


def test_lane_names_seeds_and_namespaces_are_independent(tmp_path: Path) -> None:
    payloads = [
        L.build_launch_payload(
            stage="long", lane_id=lane.lane_id, root=tmp_path, gpu=index
        )
        for index, lane in enumerate(L.STAGE1_NATURAL_CLIP_LANES)
    ]
    assert len({row["spec"]["seed"] for row in payloads}) == 3
    assert len({row["spec"]["run_name"] for row in payloads}) == 3
    assert len({row["spec"]["namespace"] for row in payloads}) == 3
    assert len({row["spec"]["physical_gpu"] for row in payloads}) == 3


def test_payload_json_is_stable_and_dry_run_has_no_side_effect(
    tmp_path: Path,
) -> None:
    arguments = [
        sys.executable,
        str(SCRIPT),
        "--stage",
        "smoke",
        "--lane",
        "fh_stable_take058_unit04",
        "--root",
        str(tmp_path),
        "--gpu",
        "1",
        "--dry-run",
    ]
    first = subprocess.run(
        arguments, check=False, capture_output=True, text=True
    )
    second = subprocess.run(
        arguments, check=False, capture_output=True, text=True
    )
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    document = json.loads(first.stdout)
    assert first.stdout == L.canonical_json(document) + "\n"
    assert document["spec"]["stage"] == "smoke"
    assert document["spec"]["physical_gpu"] == 1
    assert not Path(document["spec"]["namespace"]).exists()


def test_payload_preserves_venv_python_symlink(tmp_path: Path) -> None:
    python_link = tmp_path / "exact_venv" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to(Path(sys.executable))

    payload = L.build_launch_payload(
        stage="smoke",
        lane_id="bh_quality_take061_unit15",
        root=tmp_path,
        gpu=0,
        python_executable=python_link,
    )

    assert payload["argv"][0] == str(python_link.absolute())
    assert payload["argv"][0] != str(python_link.resolve())


def test_namespace_claim_is_exclusive_and_persists_exact_payload(
    tmp_path: Path,
) -> None:
    payload = L.build_launch_payload(
        stage="probe",
        lane_id="bh_diverse_take060_unit09",
        root=tmp_path,
        gpu=0,
    )
    namespace = L.claim_namespace(payload)
    persisted = namespace / "launch_spec_and_argv.v2.json"
    assert persisted.read_text(encoding="ascii") == L.canonical_json(payload) + "\n"
    with pytest.raises(L.LaunchRefused, match="already spent"):
        L.claim_namespace(payload)


def test_launcher_is_host_safe_independent_and_direct_exec() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    from_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    direct_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("isaaclab") for name in from_imports)
    assert not any(name.startswith("isaaclab") for name in direct_imports)
    assert "launch_n1_reward_screen_diagnostic" not in source
    assert "launch_n1_vendor_baseline_diagnostic" not in source
    assert "os.execvpe(argv[0], argv, env)" in source
    assert "subprocess" not in from_imports | direct_imports
