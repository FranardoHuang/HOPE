"""CPU-only guards for the post-swing capture play recipe."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAY_CONFIG = ROOT / "hope_training/whole_body_tracking/cfg/play.yaml"
PLAY_SOURCE = ROOT / "hope_training/whole_body_tracking/scripts/play.py"
V1_RESULT = (
    ROOT / "configs/phase1_post_swing_teacher_capture_attempt_v1_result_20260715.json"
)
TRAIN_ONLY_CHECKPOINT_KEYS = (
    "checkpoint_tolerant",
    "checkpoint_allow_missing_contract",
    "checkpoint_allow_contract_mismatch",
)


def _play_tree() -> ast.Module:
    source = PLAY_SOURCE.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(PLAY_SOURCE))


def _load_seed_validator():
    tree = _play_tree()
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_play_seed"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, object] = {}
    exec(compile(isolated, str(PLAY_SOURCE), "exec"), namespace)
    return namespace["_validate_play_seed"]


def _compose_play(overrides: list[str]):
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(
        config_dir=str(PLAY_CONFIG.parent.resolve()), version_base=None
    ):
        return hydra.compose(config_name="play", overrides=overrides)


def test_play_seed_validator_is_dependency_light_and_fail_closed():
    assert "\nseed: 0\n" in PLAY_CONFIG.read_text(encoding="utf-8")
    validate = _load_seed_validator()
    assert validate(0) == 0
    assert validate(3) == 3
    assert validate(0xFFFFFFFF) == 0xFFFFFFFF
    for invalid in (None, True, False, -1, 0x100000000, 1.0, "3"):
        with pytest.raises(ValueError, match="play seed must be a plain int"):
            validate(invalid)


def test_play_binds_one_seed_to_env_and_runner_before_gym_make():
    tree = _play_tree()
    run_play = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_play"
    )
    statements = list(ast.walk(run_play))
    validate_call = next(
        node
        for node in statements
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_play_seed"
    )
    seed_assignments = {
        ast.unparse(node.targets[0]): node
        for node in statements
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) in {"env_cfg.seed", "agent_cfg.seed"}
    }
    gym_make = next(
        node
        for node in statements
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "gym.make"
    )
    assert set(seed_assignments) == {"env_cfg.seed", "agent_cfg.seed"}
    for assignment in seed_assignments.values():
        assert ast.unparse(assignment.value) == "int(cfg.seed)"
        assert validate_call.lineno < assignment.lineno < gym_make.lineno


def test_v1_evidence_names_all_train_only_checkpoint_keys():
    result = json.loads(V1_RESULT.read_text(encoding="utf-8"))
    assert (
        tuple(result["root_cause"]["retained_train_only_keys"])
        == TRAIN_ONLY_CHECKPOINT_KEYS
    )
    assert result["decision"]["successor_must_remove_all_train_only_checkpoint_keys"] is True


@pytest.mark.parametrize("train_only_key", TRAIN_ONLY_CHECKPOINT_KEYS)
def test_play_hydra_compose_rejects_each_train_only_checkpoint_key(train_only_key):
    hydra_errors = pytest.importorskip("hydra.errors")
    with pytest.raises(hydra_errors.ConfigCompositionException, match=train_only_key):
        _compose_play(["seed=3", f"{train_only_key}=false"])


def test_sanitized_capture_recipe_composes_and_preserves_training_seed():
    training_derived = [
        "seed=3",
        "headless=true",
        "checkpoint=/tmp/model_500.pt",
        "motion_file=/tmp/forehand.npz",
        "motion_file_2=/tmp/backhand.npz",
        "task.motion.wrap_teleport=false",
        "task.motion.post_swing_start_prob=0.25",
        "+task.motion.post_swing_capture_output_dir=/tmp/capture_v2",
        "+task.motion.post_swing_capture_target_count=4096",
        "post_swing_capture_max_steps=20000",
        *(f"{key}=false" for key in TRAIN_ONLY_CHECKPOINT_KEYS),
    ]
    sanitized = [
        override
        for override in training_derived
        if override.split("=", 1)[0] not in TRAIN_ONLY_CHECKPOINT_KEYS
    ]
    cfg = _compose_play(sanitized)
    assert int(cfg.seed) == 3
    assert str(cfg.checkpoint) == "/tmp/model_500.pt"
    assert int(cfg.task.motion.post_swing_capture_target_count) == 4096
    assert all(key not in cfg for key in TRAIN_ONLY_CHECKPOINT_KEYS)
