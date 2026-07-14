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
V2_RESULT = (
    ROOT / "configs/phase1_post_swing_teacher_capture_attempt_v2_result_20260715.json"
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


def _load_owned_environment_runner():
    tree = _play_tree()
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_with_owned_play_environment"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, object] = {}
    exec(compile(isolated, str(PLAY_SOURCE), "exec"), namespace)
    return namespace["_run_with_owned_play_environment"]


class _ClosingEnv:
    def __init__(self, close_error: BaseException | None = None):
        self.close_calls = 0
        self.close_error = close_error

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


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


def test_play_actor_observation_adapter_covers_initial_and_step_observations():
    tree = _play_tree()
    execute = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_created_environment"
    )
    calls = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "policy_observation_tensor"
    ]
    assert len(calls) == 2
    assert any("env.get_observations()" in ast.unparse(node) for node in calls)
    assert any(ast.unparse(node.args[0]) == "obs" for node in calls)


@pytest.mark.parametrize("stage", ("before_wrap", "initial_observation", "step"))
def test_play_closes_exactly_one_owned_environment_on_runtime_failure(stage):
    run_owned = _load_owned_environment_runner()
    base = _ClosingEnv()
    wrapper = _ClosingEnv()
    failure = RuntimeError(stage)

    def body(owner):
        if stage != "before_wrap":
            owner[0] = wrapper
        raise failure

    with pytest.raises(RuntimeError, match=stage) as caught:
        run_owned(base, body)
    assert caught.value is failure
    assert base.close_calls == (1 if stage == "before_wrap" else 0)
    assert wrapper.close_calls == (0 if stage == "before_wrap" else 1)


def test_play_closes_wrapper_exactly_once_on_success_and_preserves_primary_close_error():
    run_owned = _load_owned_environment_runner()
    base = _ClosingEnv()
    wrapper = _ClosingEnv()

    assert run_owned(base, lambda owner: (owner.__setitem__(0, wrapper), "ok")[1]) == "ok"
    assert base.close_calls == 0
    assert wrapper.close_calls == 1

    primary = RuntimeError("initial observation failed")
    bad_close = _ClosingEnv(RuntimeError("teardown failed"))
    with pytest.raises(RuntimeError, match="initial observation failed") as caught:
        run_owned(bad_close, lambda _owner: (_ for _ in ()).throw(primary))
    assert caught.value is primary
    assert bad_close.close_calls == 1
    assert any("teardown failed" in note for note in getattr(primary, "__notes__", ()))


def test_v1_evidence_names_all_train_only_checkpoint_keys():
    result = json.loads(V1_RESULT.read_text(encoding="utf-8"))
    assert (
        tuple(result["root_cause"]["retained_train_only_keys"])
        == TRAIN_ONLY_CHECKPOINT_KEYS
    )
    assert result["decision"]["successor_must_remove_all_train_only_checkpoint_keys"] is True


def test_v2_runtime_failure_is_bound_spent_and_not_silently_promoted():
    result = json.loads(V2_RESULT.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["status"] == "blocked_runtime_observation_api"
    assert result["preregistration"]["file_sha256"] == (
        "fd72fa9ef1305960bd715c087623a21ef4709c7f05f751bb9ab2d6d4fc94b60c"
    )
    assert result["plan_compose"]["returncode"] == 0
    assert result["launch"]["pid"] == result["launch"]["pgid"] == result["launch"]["sid"]
    assert result["runtime_checks_before_failure"]["runtime_hard_contract_match"] is True
    assert result["failure"]["message"] == "'tuple' object has no attribute 'to'"
    assert result["artifacts"] == {
        "natural_wrap_capture_claim_json": "present",
        "natural_wrap_states_npz": "absent",
        "natural_wrap_capture_json": "absent",
        "teacher_receipt_json": "absent",
    }
    assert result["teardown"]["signal_sent_utc"] is None
    assert result["teardown"]["exit_observed_utc"] is None
    assert result["decision"]["v2_retry_forbidden"] is True
    assert result["decision"]["v2_namespace_spent"] is True
    assert result["decision"]["v2_scientific_training_authorized"] is False


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
