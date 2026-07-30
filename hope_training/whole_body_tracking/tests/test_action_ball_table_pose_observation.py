"""Pure regressions for the ActionBall table-pose actor channels.

No Isaac Lab import is required.  The two tensor-only kernels are extracted
from the production observation module, and the train finalizer is inspected
as syntax/source so these tests can also run in the lightweight Pod test lane.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

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
TRAIN_PATH = ROOT / "scripts" / "train.py"


def _matrix_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Small test-only wxyz quaternion-to-matrix implementation."""

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


def _load_tensor_kernels():
    source = OBS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OBS_PATH))
    names = {
        "_base_position_table_from_tensors",
        "_base_orientation_table_6d_from_quat",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
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
        "matrix_from_quat": _matrix_from_quat_wxyz,
    }
    exec(compile(module, str(OBS_PATH), "exec"), namespace)
    return (
        namespace["_base_position_table_from_tensors"],
        namespace["_base_orientation_table_6d_from_quat"],
    )


def _load_base_lin_vel_producer():
    source = OBS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OBS_PATH))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "base_lin_vel_b"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "_cmd": lambda env, command_name: env.commands[command_name],
    }
    exec(compile(module, str(OBS_PATH), "exec"), namespace)
    return namespace["base_lin_vel_b"]


def test_base_position_is_relative_to_each_env_table_surface_center():
    position, _orientation = _load_tensor_kernels()
    env_origins = torch.tensor(
        [[8.0, -2.0, 0.0], [-4.0, 6.0, 0.0]],
        dtype=torch.float64,
    )
    base_local = torch.tensor(
        [[-0.5, 0.0, 1.0684], [0.25, -0.3, 0.96]],
        dtype=torch.float64,
    )
    actual = position(
        base_local + env_origins,
        env_origins,
        table_near_x=0.5,
        table_surface_z=0.76,
        table_length=2.74,
    )
    expected = torch.tensor(
        [[-2.37, 0.0, 0.3084], [-1.62, -0.3, 0.20]],
        dtype=torch.float64,
    )
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=0.0)


def test_base_orientation_6d_keeps_roll_pitch_yaw_without_euler_or_quat_sign():
    _position, orientation = _load_tensor_kernels()
    half = math.sqrt(0.5)
    quat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half, 0.0, 0.0, half],  # yaw +90 deg
            [half, half, 0.0, 0.0],  # roll +90 deg
            [-half, -half, 0.0, 0.0],  # identical roll, opposite quat sign
        ],
        dtype=torch.float64,
    )
    actual = orientation(quat)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert actual.shape == (4, 6)
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=0.0)


def test_base_linear_velocity_producer_uses_root_body_frame_truth():
    velocity = torch.tensor([[1.0, -2.0, 0.5]])
    env = SimpleNamespace(
        commands={
            "racket_target": SimpleNamespace(
                robot=SimpleNamespace(
                    data=SimpleNamespace(root_lin_vel_b=velocity)
                )
            )
        }
    )
    actual = _load_base_lin_vel_producer()(env, "racket_target")
    assert actual is velocity


def test_train_inserts_table_pose_twist_before_face_and_action():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAIN_PATH))
    finalizer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finalize_action_ball_training_cfg"
    )
    segment = ast.get_source_segment(source, finalizer)
    assert segment is not None
    assert 'f"action_ball_n{action_count}"' in segment
    assert 'f"action_ball_table_pose_n{action_count}"' in segment
    assert 'f"action_ball_table_pose_twist_n{action_count}"' in segment
    assert (
        segment.index("policy.base_position_table =")
        < segment.index("policy.base_orientation_table_6d =")
        < segment.index("policy.base_lin_vel_b =")
        < segment.index("policy.racket_target_normal_cmd =")
        < segment.index("policy.action_one_hot =")
    )
    assert "if include_table_pose:" in segment
    assert "if include_base_twist:" in segment
    assert (
        "configured_actor_contract == table_pose_twist_actor_contract"
        in segment
    )
