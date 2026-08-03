"""Pure producer/config tests for real-task A211 and causal-ball C211."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
OBS_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "hope_observations.py"
)
ENV_CFG_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "config"
    / "agibot_a3"
    / "hope_env_cfg.py"
)


def _matrix_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    q = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _yaw_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    zeros = torch.zeros_like(yaw)
    return torch.stack(
        (torch.cos(0.5 * yaw), zeros, zeros, torch.sin(0.5 * yaw)),
        dim=-1,
    )


def _quat_rotate_inverse_wxyz(
    quat: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    matrix = _matrix_from_quat_wxyz(quat)
    return torch.matmul(
        matrix.transpose(-1, -2), vector.unsqueeze(-1)
    ).squeeze(-1)


def _load_producers():
    source = OBS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OBS_PATH))
    function_names = {
        "_action_ball_225_identity",
        "_action_ball_225_source_values",
        "_action_ball_225_assert",
        "_action_ball_225_is_construction_probe",
        "_action_ball_225_install_token",
        "_action_ball_211_task_valid",
        "_action_ball_211_mask_task_value",
        "_action_ball_225_snapshot_heading",
        "action_ball_a211_task_desired_contact_position_heading",
        "action_ball_a211_task_desired_contact_velocity_heading",
        "action_ball_a211_task_desired_contact_face_heading",
        "action_ball_c211_incoming_ball_contact_position_heading",
        "action_ball_c211_incoming_ball_contact_velocity_heading",
        "action_ball_c211_incoming_ball_contact_spin_heading",
        "action_ball_211_base_target_position_world_xy",
        "action_ball_211_time_to_contact",
        "action_ball_211_time_to_teacher_start",
        "action_ball_task_valid",
    }
    nodes = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_ACTION_BALL_225_SNAPSHOT_SOURCES"
                for target in node.targets
            )
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            nodes.append(node)
    assert {
        node.name for node in nodes if isinstance(node, ast.FunctionDef)
    } == function_names
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "yaw_quat": _yaw_quat_wxyz,
        "quat_rotate_inverse": _quat_rotate_inverse_wxyz,
        "_cmd": lambda env, command_name: env.commands[command_name],
        "stage1_base_target_position_world_xy": (
            lambda env, _command_name: env.base_target_position_world_xy
        ),
        "time_to_strike": lambda env, _command_name: env.time_to_contact,
        "time_to_teacher_start_s": (
            lambda env, _command_name: env.time_to_teacher_start
        ),
    }
    exec(compile(module, str(OBS_PATH), "exec"), namespace)
    return namespace


def _command(*, active: bool = True):
    dtype = torch.float64
    half = math.pi / 4.0
    command = SimpleNamespace(
        _action_ball_reset_generation=torch.tensor([3], dtype=torch.int64),
        _action_ball_swing_generation=torch.tensor([7], dtype=torch.int64),
        _action_ball_action_uid=torch.tensor([5527597793770800], dtype=torch.int64),
        _action_ball_action_slot=torch.tensor([0], dtype=torch.int64),
        _action_ball_attempt_action=torch.tensor([0], dtype=torch.int64),
        _action_ball_attempt_active=torch.tensor([active], dtype=torch.bool),
        _action_ball_task_valid=torch.tensor([active], dtype=torch.bool),
        _action_ball_task_by_env=[
            SimpleNamespace(
                env_id=0,
                reset_generation=3,
                swing_generation=7,
                action_uid=5527597793770800,
                action_slot=0,
            )
        ],
        robot=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[10.0, 20.0, 1.0]], dtype=dtype),
                root_quat_w=torch.tensor(
                    [[math.cos(half), 0.0, 0.0, math.sin(half)]],
                    dtype=dtype,
                ),
            )
        ),
        racket_target_pos_w=torch.tensor([[11.0, 23.0, 2.0]], dtype=dtype),
        racket_target_vel_w=torch.tensor([[2.0, 0.0, 1.0]], dtype=dtype),
        racket_target_normal_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype),
        _action_ball_ball_contact_target_w=torch.tensor(
            [[12.0, 21.0, 2.0]], dtype=dtype
        ),
        vb_vel_in_w=torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype),
        vb_spin_in_w=torch.tensor([[4.0, 5.0, 6.0]], dtype=dtype),
    )
    command._ensure_action_ball_runtime_initialized = lambda: None
    command.action_ball_target_component_valid = lambda component: component in {
        "position",
        "velocity",
        "face",
    }
    return command


def _env(command, *, token: int = 11, construction_probe: bool = False):
    return SimpleNamespace(
        common_step_counter=token,
        observation_manager=None if construction_probe else object(),
        commands={"racket_target": command},
    )


def test_c211_uses_real_ball_contact_p_v_spin_with_exact_heading_math():
    producers = _load_producers()
    command = _command()
    env = _env(command)
    position = producers[
        "action_ball_c211_incoming_ball_contact_position_heading"
    ](env, "racket_target")

    def forbidden_recapture(*_args, **_kwargs):
        raise AssertionError("later producer term tried to rebuild device snapshot")

    producers["_action_ball_225_identity"] = forbidden_recapture
    producers["_action_ball_225_source_values"] = forbidden_recapture
    velocity = producers[
        "action_ball_c211_incoming_ball_contact_velocity_heading"
    ](env, "racket_target")
    spin = producers["action_ball_c211_incoming_ball_contact_spin_heading"](
        env, "racket_target"
    )
    assert torch.allclose(
        torch.cat((position, velocity, spin), dim=-1),
        torch.tensor(
            [[1.0, -2.0, 1.0, 2.0, -1.0, 3.0, 5.0, -4.0, 6.0]],
            dtype=torch.float64,
        ),
        atol=1.0e-12,
        rtol=0.0,
    )
    cached = command._action_ball_225_observation_cache["c211"]
    assert cached["token"] == 11
    assert cached["result"].shape == (1, 9)


def test_a211_uses_real_task_desired_contact_not_teacher_copy():
    producers = _load_producers()
    command = _command()
    env = _env(command)
    actual = torch.cat(
        (
            producers["action_ball_a211_task_desired_contact_position_heading"](
                env, "racket_target"
            ),
            producers["action_ball_a211_task_desired_contact_velocity_heading"](
                env, "racket_target"
            ),
            producers["action_ball_a211_task_desired_contact_face_heading"](
                env, "racket_target"
            ),
        ),
        dim=-1,
    )
    assert torch.allclose(
        actual,
        torch.tensor(
            [[3.0, -1.0, 1.0, 0.0, -2.0, 1.0, 0.0, -1.0, 0.0]],
            dtype=torch.float64,
        ),
        atol=1.0e-12,
        rtol=0.0,
    )

    command = _command()
    command.action_ball_target_component_valid = lambda component: component != "velocity"
    with pytest.raises(RuntimeError, match="complete valid task-derived"):
        producers["action_ball_a211_task_desired_contact_position_heading"](
            _env(command), "racket_target"
        )


def test_interleaved_position_reinstall_velocity_fails_closed_without_rebuild():
    producers = _load_producers()
    command = _command()
    env = _env(command)
    position = producers["action_ball_c211_incoming_ball_contact_position_heading"](
        env, "racket_target"
    )
    frozen_result = command._action_ball_225_observation_cache["c211"][
        "result"
    ]
    command._action_ball_reset_generation[0] += 1
    command._action_ball_swing_generation[0] += 1
    command.vb_vel_in_w[0, 0] += 1.0
    command._action_ball_task_by_env[0] = SimpleNamespace(
        env_id=0,
        reset_generation=4,
        swing_generation=8,
        action_uid=5527597793770800,
        action_slot=0,
    )
    with pytest.raises(RuntimeError, match="install identity changed"):
        producers["action_ball_c211_incoming_ball_contact_velocity_heading"](
            env, "racket_target"
        )
    assert position.data_ptr() == frozen_result[:, 0:3].data_ptr()



def test_same_counter_reinstall_generation_invalidates_cached_snapshot():
    producers = _load_producers()
    command = _command()
    env = _env(command)
    first = producers[
        "action_ball_c211_incoming_ball_contact_position_heading"
    ](env, "racket_target").clone()
    producers["action_ball_c211_incoming_ball_contact_velocity_heading"](
        env, "racket_target"
    )
    producers["action_ball_c211_incoming_ball_contact_spin_heading"](
        env, "racket_target"
    )

    # A reset/install transaction can replace task bytes before the global
    # control counter advances.  Its authoritative identity is the cache key.
    command._action_ball_reset_generation[0] += 1
    command._action_ball_swing_generation[0] += 1
    command._action_ball_action_uid[0] += 1
    command._action_ball_task_by_env[0] = SimpleNamespace(
        env_id=0,
        reset_generation=4,
        swing_generation=8,
        action_uid=5527597793770801,
        action_slot=0,
    )
    command._action_ball_ball_contact_target_w[:] = torch.tensor(
        [[14.0, 21.0, 2.0]], dtype=torch.float64
    )
    second = producers[
        "action_ball_c211_incoming_ball_contact_position_heading"
    ](env, "racket_target")

    assert not torch.equal(second, first)
    assert torch.allclose(
        second,
        torch.tensor([[1.0, -4.0, 1.0]], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    cached = command._action_ball_225_observation_cache["c211"]
    assert torch.equal(
        cached["identity"],
        producers["_action_ball_225_identity"](command),
    )


def test_wait_masks_task_packet_and_active_validity_mismatch_fails_closed():
    producers = _load_producers()
    command = _command(active=False)
    actual = producers[
        "action_ball_c211_incoming_ball_contact_position_heading"
    ](_env(command), "racket_target")
    assert torch.equal(actual, torch.zeros((1, 3), dtype=torch.float64))

    command = _command(active=False)
    command._action_ball_task_valid.fill_(True)
    with pytest.raises(RuntimeError):
        producers["action_ball_c211_incoming_ball_contact_position_heading"](
            _env(command), "racket_target"
        )

    command = _command()
    command.vb_spin_in_w[0, 1] = float("nan")
    with pytest.raises(RuntimeError):
        producers["action_ball_c211_incoming_ball_contact_position_heading"](
            _env(command), "racket_target"
        )

    command = _command()
    command._action_ball_ball_contact_target_w = torch.zeros(
        (1, 2), dtype=torch.float64
    )
    with pytest.raises(RuntimeError, match="wrong shape"):
        producers["action_ball_c211_incoming_ball_contact_position_heading"](
            _env(command), "racket_target"
        )


def test_pristine_pre_reset_shape_probe_is_zero_but_never_cached():
    producers = _load_producers()
    command = _command(active=False)
    command._action_ball_reset_generation.zero_()
    command._action_ball_swing_generation.fill_(-1)
    command._action_ball_action_uid.fill_(-1)
    command._action_ball_action_slot.fill_(-1)
    command._action_ball_attempt_action.fill_(-1)
    result = producers[
        "action_ball_c211_incoming_ball_contact_position_heading"
    ](_env(command, token=0, construction_probe=True), "racket_target")
    assert torch.equal(result, torch.zeros((1, 3), dtype=torch.float64))
    assert not hasattr(command, "_action_ball_225_observation_cache")


def test_real_reset_wait_is_zero_and_transaction_bound():
    producers = _load_producers()
    command = _command(active=False)
    command._action_ball_reset_generation.zero_()
    command._action_ball_swing_generation.fill_(-1)
    command._action_ball_action_uid.fill_(-1)
    command._action_ball_action_slot.fill_(-1)
    command._action_ball_attempt_action.fill_(-1)
    actual = producers[
        "action_ball_c211_incoming_ball_contact_position_heading"
    ](_env(command, token=0), "racket_target")
    assert torch.equal(actual, torch.zeros((1, 3), dtype=torch.float64))
    assert command._action_ball_225_observation_cache["c211"]["token"] == 0


def test_token_zero_installed_task_uses_real_bytes_and_is_cached():
    producers = _load_producers()
    command = _command(active=True)
    env = _env(command, token=0, construction_probe=True)
    actual = producers[
        "action_ball_c211_incoming_ball_contact_position_heading"
    ](env, "racket_target")
    assert torch.allclose(
        actual,
        torch.tensor([[1.0, -2.0, 1.0]], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert command._action_ball_225_observation_cache["c211"]["token"] == 0


def test_action_ball_assertions_use_torch20_single_argument_signature(monkeypatch):
    calls = []

    def torch20_assert_async(condition):
        calls.append(condition)
        if not bool(condition):
            raise RuntimeError("torch20 async assertion failed")

    monkeypatch.setattr(torch, "_assert_async", torch20_assert_async)
    producers = _load_producers()
    actual = producers[
        "action_ball_c211_incoming_ball_contact_position_heading"
    ](_env(_command()), "racket_target")
    assert actual.shape == (1, 3)
    assert calls

    source = OBS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OBS_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_action_ball_225_assert"
    )
    async_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_assert_async"
    ]
    assert len(async_calls) == 1
    assert len(async_calls[0].args) == 1
    assert async_calls[0].keywords == []


def test_normal_three_term_transaction_extracts_no_host_tensor_scalar(monkeypatch):
    producers = _load_producers()
    command = _command()
    env = _env(command)

    def forbidden_host_extraction(*_args, **_kwargs):
        raise AssertionError("normal producer transaction synchronized to host")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_host_extraction)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_host_extraction)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden_host_extraction)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_host_extraction)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden_host_extraction)

    position = producers[
        "action_ball_c211_incoming_ball_contact_position_heading"
    ](env, "racket_target")
    velocity = producers[
        "action_ball_c211_incoming_ball_contact_velocity_heading"
    ](env, "racket_target")
    spin = producers["action_ball_c211_incoming_ball_contact_spin_heading"](
        env, "racket_target"
    )
    assert position.shape == velocity.shape == spin.shape == (1, 3)
    cached = command._action_ball_225_observation_cache["c211"]
    assert cached["next_component"] == 3


def test_final_observation_boundary_masks_wait_task_base_goal_and_two_clocks():
    producers = _load_producers()
    command = _command(active=False)
    env = _env(command)
    env.base_target_position_world_xy = torch.tensor(
        [[1.2, -0.3]], dtype=torch.float64
    )
    env.time_to_contact = torch.tensor([[0.7]], dtype=torch.float64)
    env.time_to_teacher_start = torch.tensor([[0.2]], dtype=torch.float64)

    task_packet = torch.cat(
        (
            producers[
                "action_ball_a211_task_desired_contact_position_heading"
            ](env, "racket_target"),
            producers[
                "action_ball_a211_task_desired_contact_velocity_heading"
            ](env, "racket_target"),
            producers["action_ball_a211_task_desired_contact_face_heading"](
                env, "racket_target"
            ),
        ),
        dim=-1,
    )
    masked_suffix = torch.cat(
        (
            producers["action_ball_211_base_target_position_world_xy"](
                env, "racket_target"
            ),
            producers["action_ball_211_time_to_contact"](
                env, "racket_target"
            ),
            producers["action_ball_211_time_to_teacher_start"](
                env, "racket_target"
            ),
        ),
        dim=-1,
    )
    assert torch.equal(task_packet, torch.zeros((1, 9), dtype=torch.float64))
    assert torch.equal(masked_suffix, torch.zeros((1, 4), dtype=torch.float64))
    assert torch.equal(
        producers["action_ball_task_valid"](env, "racket_target"),
        torch.zeros((1, 1), dtype=torch.float64),
    )

    active_command = _command(active=True)
    active_env = _env(active_command)
    active_env.base_target_position_world_xy = env.base_target_position_world_xy
    active_env.time_to_contact = env.time_to_contact
    active_env.time_to_teacher_start = env.time_to_teacher_start
    assert torch.equal(
        producers["action_ball_211_base_target_position_world_xy"](
            active_env, "racket_target"
        ),
        active_env.base_target_position_world_xy,
    )
    assert torch.equal(
        producers["action_ball_task_valid"](active_env, "racket_target"),
        torch.ones((1, 1), dtype=torch.float64),
    )


def test_public_a211_c211_observation_abi_exposes_no_wait_remaining():
    source = OBS_PATH.read_text(encoding="utf-8")
    env_source = ENV_CFG_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "def action_ball_a225_",
        "def action_ball_c225_",
        "wait_remaining = ObsTerm",
        "wait_remaining_ticks = ObsTerm",
    ):
        assert forbidden not in source
        assert forbidden not in env_source


def _class_assignments(node: ast.ClassDef) -> list[str]:
    return [
        target.id
        for child in node.body
        if isinstance(child, ast.Assign)
        for target in child.targets
        if isinstance(target, ast.Name)
        and isinstance(child.value, ast.Call)
        and target.id not in {"policy", "observations"}
    ]


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    raise AssertionError(type(node).__name__)


def test_a211_c211_policy_configs_match_contract_order_and_are_critic_isolated():
    source = ENV_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENV_CFG_PATH))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    common = [
        "actual_base_now_world",
        "joint_pos",
        "teacher_joint_pos",
        "joint_vel",
        "teacher_joint_vel",
        "actions",
        "racket_site_achieved_now_heading",
        "racket_site_teacher_now_heading",
        "racket_site_teacher_at_reference_hit_heading",
    ]
    suffix = [
        "desired_base_xy_world",
        "time_to_contact",
        "time_to_teacher_start",
        "task_valid",
    ]
    expected = {
        "HOPEActionBallA211ObservationsCfg": common
        + [
            "task_desired_contact_position_heading",
            "task_desired_contact_velocity_heading",
            "task_desired_contact_face_heading",
        ]
        + suffix,
        "HOPEActionBallC211ObservationsCfg": common
        + [
            "incoming_ball_contact_position_heading",
            "incoming_ball_contact_velocity_heading",
            "incoming_ball_contact_spin_heading",
        ]
        + suffix,
    }
    for outer_name, layout in expected.items():
        outer = classes[outer_name]
        policy = next(child for child in outer.body if isinstance(child, ast.ClassDef))
        assert _class_assignments(policy) == layout
        segment = ast.get_source_segment(source, outer)
        assert segment is not None
        assert "critic = None" in segment
        assert "Stage1CriticCfg()" not in segment
        assert "HOPECritic" not in segment

    c_segment = ast.get_source_segment(
        source, classes["HOPEActionBallC211ObservationsCfg"]
    )
    assert c_segment is not None
    for forbidden in (
        "landing",
        "racket_contact_desired",
        "stage1_racket_contact_desired",
        "target_normal",
    ):
        assert forbidden not in c_segment

    for leaf, mode, observations in (
        (
            "HOPEPingPongActionBallA211AgibotA3EnvCfg",
            "action_ball_a211",
            "HOPEActionBallA211ObservationsCfg",
        ),
        (
            "HOPEPingPongActionBallC211AgibotA3EnvCfg",
            "action_ball_c211",
            "HOPEActionBallC211ObservationsCfg",
        ),
    ):
        segment = ast.get_source_segment(source, classes[leaf])
        assert segment is not None
        assert f'obs_mode: str = "{mode}"' in segment
        assert "action_ball_211_construction_only: bool = True" in segment
        assert f"observations: {observations}" in segment
        assert "_validate_action_ball_211_wait_schedule_cfg(self)" in segment


@pytest.mark.parametrize(
    "leaf_name, observations_name, actor_contract",
    (
        (
            "HOPEPingPongActionBallA211AgibotA3EnvCfg",
            "HOPEActionBallA211ObservationsCfg",
            "action_ball_a211",
        ),
        (
            "HOPEPingPongActionBallC211AgibotA3EnvCfg",
            "HOPEActionBallC211ObservationsCfg",
            "action_ball_c211",
        ),
    ),
)
def test_construction_only_leaf_class_itself_remains_constructible(
    leaf_name, observations_name, actor_contract
):
    source = ENV_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENV_CFG_PATH))
    leaf = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == leaf_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            leaf,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    class BaseCfg:
        pass

    class ObservationsCfg:
        pass

    class RewardsCfg:
        pass

    namespace = {
        "configclass": lambda cls: cls,
        "HOPEPingPongActionBallAgibotA3EnvCfg": BaseCfg,
        "HOPEActionBallC211RewardsCfg": RewardsCfg,
        observations_name: ObservationsCfg,
    }
    exec(compile(module, str(ENV_CFG_PATH), "exec"), namespace)
    cfg = namespace[leaf_name]()
    assert cfg.obs_mode == actor_contract
    assert cfg.action_ball_211_construction_only is True
    assert isinstance(cfg.observations, ObservationsCfg)


@pytest.mark.parametrize(
    "entrypoint, actor_contract",
    (
        ("direct_train", "action_ball_a211"),
        ("rsl_rl_runner", "action_ball_c211"),
    ),
)
def test_construction_only_leaf_trainability_guard_rejects_runner_entry(
    entrypoint, actor_contract
):
    source = ENV_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENV_CFG_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_action_ball_211_trainability"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            helper,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(ENV_CFG_PATH), "exec"), namespace)
    env_cfg = SimpleNamespace(
        obs_mode=actor_contract,
        action_ball_211_construction_only=True,
    )
    with pytest.raises(
        RuntimeError,
        match="critic ABI.*normalizer lineage.*checkpoint contract",
    ):
        namespace["validate_action_ball_211_trainability"](env_cfg)
    assert entrypoint in {"direct_train", "rsl_rl_runner"}


def test_trainability_guard_passes_nonprototype_and_rejects_missing_authority():
    source = ENV_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENV_CFG_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_action_ball_211_trainability"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(ENV_CFG_PATH), "exec"), namespace)
    validate = namespace["validate_action_ball_211_trainability"]
    assert validate(SimpleNamespace(obs_mode="action_ball_deploy")) is None
    with pytest.raises(RuntimeError, match="construction-only authority marker"):
        validate(SimpleNamespace(obs_mode="action_ball_a211"))
