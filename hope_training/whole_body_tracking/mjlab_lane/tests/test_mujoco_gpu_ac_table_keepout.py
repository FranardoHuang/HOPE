"""Parity and chronology tests for the device exact table keepout."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env  # noqa: E402
import mujoco_gpu_ac_table_keepout as keepout  # noqa: E402


def test_import_rejects_preloaded_authority_from_another_checkout(
    monkeypatch, tmp_path
):
    foreign = ModuleType("mujoco_native.table_termination")
    foreign.__file__ = str(tmp_path / "mujoco_native/table_termination.py")
    foreign.REPO_ROOT = tmp_path
    package = ModuleType("mujoco_native")
    package.table_termination = foreign
    monkeypatch.setitem(sys.modules, "mujoco_native", package)
    spec = importlib.util.spec_from_file_location(
        "_foreign_authority_keepout", LANE / "mujoco_gpu_ac_table_keepout.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="authority resolved outside this tree"):
        spec.loader.exec_module(module)


def _authority_inputs(count: int):
    authority = keepout._authority
    components = authority.load_collision_components()
    scene = keepout._load_table_scene()
    rows = scene.action_ball_policy_obstacle_geometry()
    contract = scene.action_ball_policy_geometry_contract(rows)
    lo, hi = authority._validated_table_aabbs(contract)

    rng = np.random.default_rng(20260818)
    positions = np.full((count, 32, 3), 50.0, dtype=np.float64)
    quats = rng.normal(size=(count, 32, 4))
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    # One construction-derived positive and several random near-table rows.
    owner = int(components.owner_indices[0])
    box_center = 0.5 * (lo[0] + hi[0])
    positions[0, owner] = box_center - components.local_centers_m[0]
    quats[0] = (1.0, 0.0, 0.0, 0.0)
    for env in range(1, count - 1, 2):
        body = int(components.owner_indices[env % len(components.owner_indices)])
        positions[env, body] = rng.uniform(
            (0.35, -0.4, 0.55), (0.75, 0.4, 1.0)
        )
    positions[-1, 0, 0] = np.nan

    w, x, y, z = (quats[..., index] for index in range(4))
    rotations = np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        axis=-2,
    )
    expected = np.asarray(
        [
            authority.geometric_robot_table_hit(
                positions[row],
                rotations[row],
                components,
                lo,
                hi,
                racket_body_index=authority.TABLE_CONTACT_BODY_NAMES.index(
                    authority.RACKET_BODY_NAME
                ),
            )
            for row in range(count)
        ],
        dtype=bool,
    )
    return positions, quats, components, lo, hi, expected


def _device_verdict(device: str, count: int = 24, dtype=torch.float64):
    positions, quats, components, lo, hi, expected = _authority_inputs(count)
    tensor = lambda value, tensor_dtype=dtype: torch.as_tensor(  # noqa: E731
        np.array(value, copy=True), dtype=tensor_dtype, device=device
    )
    actual = keepout.geometric_robot_table_hit_mask(
        tensor(positions),
        tensor(quats),
        torch.zeros((count, 3), dtype=dtype, device=device),
        tensor(components.owner_indices, torch.long),
        tensor(components.local_centers_m),
        tensor(components.local_half_axes_m),
        tensor(lo),
        tensor(hi),
        racket_body_index=keepout._authority.TABLE_CONTACT_BODY_NAMES.index(
            keepout._authority.RACKET_BODY_NAME
        ),
        blade_center_offset=tensor(
            keepout._authority.RACKET_BLADE_CENTER_OFFSET_WRIST_M
        ),
        blade_local_half_axes=tensor(
            keepout._authority.RACKET_BLADE_LOCAL_HALF_AXES_M
        ),
    )
    assert actual.device == torch.device(device)
    return actual.cpu(), torch.as_tensor(expected)


def _empty_corner_discriminator(device: str, dtype):
    positions = torch.full((1, 32, 3), 50.0, dtype=dtype, device=device)
    positions[0, 0] = torch.tensor((0.112, 0.112, 0.0), dtype=dtype, device=device)
    quats = torch.zeros((1, 32, 4), dtype=dtype, device=device)
    quats[..., 0] = 1.0
    quats[0, 0, 0] = np.cos(np.pi / 8.0)
    quats[0, 0, 3] = np.sin(np.pi / 8.0)
    axes = torch.eye(3, dtype=dtype, device=device).mul_(0.01).unsqueeze(0)
    lo = torch.tensor(((-0.1, -0.1, -0.1),), dtype=dtype, device=device)
    hi = -lo
    broad_half = float(np.sqrt(2.0) * 0.01 + 1.0e-6)
    assert 0.112 - broad_half <= 0.1
    actual = keepout.geometric_robot_table_hit_mask(
        positions,
        quats,
        torch.zeros((1, 3), dtype=dtype, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
        torch.zeros((1, 3), dtype=dtype, device=device),
        axes,
        lo,
        hi,
        racket_body_index=31,
        blade_center_offset=torch.zeros(3, dtype=dtype, device=device),
        blade_local_half_axes=axes[0],
    )
    assert actual.device == torch.device(device)
    return actual.cpu()


def test_substep_latch_skips_preintegration_pose_and_includes_final_forward():
    class Guard:
        def __init__(self):
            self.rows = iter(
                (torch.tensor([True, False]), torch.tensor([False, True]))
            )
            self.calls = 0

        def sample(self, _data):
            self.calls += 1
            return next(self.rows)

    guard = Guard()
    env = SimpleNamespace(
        _cur_table_keepout=torch.zeros(2, dtype=torch.bool),
        _table_keepout=guard,
        sim=SimpleNamespace(data=object()),
    )
    wait_env.FullMdpInitialWaitVecEnv._after_physics_substep(env, 0)
    assert guard.calls == 0
    wait_env.FullMdpInitialWaitVecEnv._after_physics_substep(env, 1)
    wait_env.FullMdpInitialWaitVecEnv._latch_post_forward_table_keepout(env)
    assert guard.calls == 2
    assert env._cur_table_keepout.tolist() == [True, True]


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_torch_cpu_guard_matches_authority_in_runtime_and_audit_dtypes(dtype):
    actual, expected = _device_verdict("cpu", dtype=dtype)
    assert torch.equal(actual, expected)
    assert bool(actual[0]) and bool(actual[-1])
    assert bool((~actual).any())
    assert _empty_corner_discriminator("cpu", dtype).tolist() == [False]


@pytest.mark.skipif(
    os.environ.get("ACTIONBALL_RUN_MUJOCO_GPU_DIRECT") != "1",
    reason="requires the exact MuJoCo-Warp GPU environment",
)
def test_torch_gpu_guard_matches_existing_numpy_authority_exactly():
    actual, expected = _device_verdict(
        os.environ.get("ACTIONBALL_MUJOCO_DEVICE", "cuda:0"),
        dtype=torch.float32,
    )
    assert torch.equal(actual, expected)
    assert _empty_corner_discriminator(
        os.environ.get("ACTIONBALL_MUJOCO_DEVICE", "cuda:0"), torch.float32
    ).tolist() == [False]
