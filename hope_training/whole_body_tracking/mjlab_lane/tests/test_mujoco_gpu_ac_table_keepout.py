"""Parity and chronology tests for the device exact table keepout."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import warnings

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
    inputs = {
        "body_pos_w": tensor(positions),
        "body_quat_w": tensor(quats),
        "env_origins": torch.zeros((count, 3), dtype=dtype, device=device),
        "component_owner_indices": tensor(components.owner_indices, torch.long),
        "component_local_centers": tensor(components.local_centers_m),
        "component_local_half_axes": tensor(components.local_half_axes_m),
        "table_lo": tensor(lo),
        "table_hi": tensor(hi),
        "racket_body_index": keepout._authority.TABLE_CONTACT_BODY_NAMES.index(
            keepout._authority.RACKET_BODY_NAME
        ),
        "blade_center_offset": tensor(
            keepout._authority.RACKET_BLADE_CENTER_OFFSET_WRIST_M
        ),
        "blade_local_half_axes": tensor(
            keepout._authority.RACKET_BLADE_LOCAL_HALF_AXES_M
        ),
    }
    if torch.device(device).type == "cpu":
        actual = keepout.geometric_robot_table_hit_mask(
            inputs["body_pos_w"],
            inputs["body_quat_w"],
            inputs["env_origins"],
            inputs["component_owner_indices"],
            inputs["component_local_centers"],
            inputs["component_local_half_axes"],
            inputs["table_lo"],
            inputs["table_hi"],
            racket_body_index=inputs["racket_body_index"],
            blade_center_offset=inputs["blade_center_offset"],
            blade_local_half_axes=inputs["blade_local_half_axes"],
        )
    else:
        actual = _cuda_bound_verdict(inputs, device)
    assert actual.device == torch.device(device)
    return actual.cpu(), torch.as_tensor(expected)


def test_construction_exposes_only_path_free_plant_receipt(monkeypatch):
    digests = {
        "root_mjcf_sha256": "a" * 64,
        "identity_manifest_sha256": "b" * 64,
        "portable_identity_sha256": "c" * 64,
        "verification_receipt_sha256": "d" * 64,
        "owner_local_frame_sha256": "e" * 64,
    }
    def immutable(value):
        value.setflags(write=False)
        return value

    authority = SimpleNamespace(
        identity_receipt={
            **digests,
            "root_mjcf_path": "/must/not/leak/a3_pingpong.xml",
            "identity_manifest_path": "/must/not/leak/manifest.json",
        },
        body_ids=immutable(np.arange(32, dtype=np.int64)),
        components=SimpleNamespace(
            component_ids=("component_00",),
            owner_indices=immutable(np.zeros(1, dtype=np.int64)),
            local_centers_m=immutable(np.zeros((1, 3), dtype=np.float64)),
            local_half_axes_m=immutable(
                np.zeros((1, 3, 3), dtype=np.float64)
            ),
            artifact_sha256="f" * 64,
            content_sha256="0" * 64,
        ),
        aabb_lo=immutable(np.zeros((1, 3), dtype=np.float64)),
        aabb_hi=immutable(np.ones((1, 3), dtype=np.float64)),
        racket_body_index=31,
    )
    captured = {}

    def fake_guard(*_args, **kwargs):
        captured.update(kwargs)
        return authority

    monkeypatch.setattr(keepout._authority, "ExactRobotTableGuard", fake_guard)
    monkeypatch.setattr(
        keepout,
        "_load_table_scene",
        lambda: SimpleNamespace(
            action_ball_policy_obstacle_geometry=lambda: (),
            action_ball_policy_geometry_contract=lambda _rows: {},
        ),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="The given NumPy array is not writable.*"
        )
        bound = keepout.DeviceExactTableKeepout(
            mujoco=object(),
            model=object(),
            mjcf_path=Path("/tmp/a3_pingpong.xml"),
            env_origins=torch.zeros((2, 3)),
            device=torch.device("cpu"),
        )
    assert dict(bound.plant_identity_receipt) == digests
    assert bound._component_ids == ("component_00",)
    assert all("path" not in key for key in bound.plant_identity_receipt)
    registered = captured["registered_plant"]
    assert registered["root_mjcf_sha256"] == (
        "7bbda723f339bdf252a20622afa7a7d53a6fca97464252c66c6e1a45199bcae1"
    )
    assert registered["identity_manifest_path"].name == (
        "a3p_p1_0807_mujoco_identity_v1_20260828.json"
    )


@pytest.mark.parametrize(
    ("reason_code", "owner_index", "message"),
    (
        (1, -1, "invalid_origin"),
        (2, 17, "invalid_body_pose at local body 17"),
        (3, 8, "zero_quaternion at local body 8"),
        (0, -1, "unknown"),
    ),
)
def test_keepout_witness_invalid_precheck_reasons_fail_closed(
    reason_code, owner_index, message
):
    with pytest.raises(RuntimeError, match=message):
        keepout._diagnostic_witness_reason(reason_code, owner_index)
    assert keepout._diagnostic_witness_reason(4, 0) == "sat_overlap"


def _empty_corner_discriminator(device: str, dtype):
    positions = torch.full((1, 32, 3), 50.0, dtype=dtype, device=device)
    positions[0, 0] = torch.tensor((0.112, 0.112, 0.0), dtype=dtype, device=device)
    quats = torch.zeros((1, 32, 4), dtype=dtype, device=device)
    quats[..., 0] = 1.0
    quats[0, 0, 0] = np.cos(np.pi / 8.0)
    quats[0, 0, 3] = np.sin(np.pi / 8.0)
    one_axes = torch.eye(3, dtype=dtype, device=device).mul_(0.01)
    axes = one_axes.unsqueeze(0).repeat(62, 1, 1)
    lo = torch.tensor(
        (
            (-0.1, -0.1, -0.1),
            (10.0, 10.0, 10.0),
            (20.0, 20.0, 20.0),
            (30.0, 30.0, 30.0),
            (40.0, 40.0, 40.0),
        ),
        dtype=dtype,
        device=device,
    )
    hi = lo + torch.tensor((0.2, 0.2, 0.2), dtype=dtype, device=device)
    hi[0] = -lo[0]
    broad_half = float(np.sqrt(2.0) * 0.01 + 1.0e-6)
    assert 0.112 - broad_half <= 0.1
    inputs = {
        "body_pos_w": positions,
        "body_quat_w": quats,
        "env_origins": torch.zeros((1, 3), dtype=dtype, device=device),
        "component_owner_indices": torch.zeros(
            62, dtype=torch.long, device=device
        ),
        "component_local_centers": torch.zeros(
            (62, 3), dtype=dtype, device=device
        ),
        "component_local_half_axes": axes,
        "table_lo": lo,
        "table_hi": hi,
        "racket_body_index": 31,
        "blade_center_offset": torch.zeros(3, dtype=dtype, device=device),
        "blade_local_half_axes": one_axes,
    }
    if torch.device(device).type == "cpu":
        actual = keepout.geometric_robot_table_hit_mask(
            inputs["body_pos_w"],
            inputs["body_quat_w"],
            inputs["env_origins"],
            inputs["component_owner_indices"],
            inputs["component_local_centers"],
            inputs["component_local_half_axes"],
            inputs["table_lo"],
            inputs["table_hi"],
            racket_body_index=inputs["racket_body_index"],
            blade_center_offset=inputs["blade_center_offset"],
            blade_local_half_axes=inputs["blade_local_half_axes"],
        )
    else:
        actual = _cuda_bound_verdict(inputs, device)
    assert actual.device == torch.device(device)
    return actual.cpu()


def _fixed_shape_tape(dtype=torch.float32):
    """One immutable tape spanning every discrete SAT/fail-closed stratum."""

    count = 94
    positions = torch.full((count, 32, 3), 50.0, dtype=dtype)
    quats = torch.zeros((count, 32, 4), dtype=dtype)
    quats[..., 0] = 1.0
    origins = torch.zeros((count, 3), dtype=dtype)
    owners = torch.arange(62, dtype=torch.long).remainder(31)
    centers = torch.zeros((62, 3), dtype=dtype)
    half_axes = torch.eye(3, dtype=dtype).mul(0.125).repeat(62, 1, 1)
    anisotropic = torch.diag(torch.tensor((0.2, 0.05, 0.1), dtype=dtype))
    half_axes[owners == 0] = anisotropic
    lo = torch.tensor(
        (
            (-1.0, -1.0, -1.0),
            (10.0, 10.0, 10.0),
            (20.0, 20.0, 20.0),
            (30.0, 30.0, 30.0),
            (40.0, 40.0, 40.0),
        ),
        dtype=dtype,
    )
    hi = lo + 0.5
    hi[0] = 1.0

    # none=[0:8]; sparse=[8:16] with exactly two positive environments.
    positions[9, 0] = 0.0
    positions[14, 0] = 0.0
    # all=[16:24].
    positions[16:24, 0] = 0.0
    # touching=[24] is inclusive; [25] is separated by a representable gap.
    positions[24, 1] = torch.tensor((1.125, 0.0, 0.0), dtype=dtype)
    positions[25, 1] = torch.tensor((1.126953125, 0.0, 0.0), dtype=dtype)
    # Exercise world->environment subtraction on both known-negative and
    # known-positive rows without changing their local geometry.
    # Keep the one-ULP boundary rows [36:38] at zero origin: adding and then
    # subtracting a large origin would itself consume the intended ULP gap.
    translated = torch.tensor((*range(26), *range(30, 36)), dtype=torch.long)
    origins[translated] = torch.linspace(
        -37.0, 41.0, translated.numel(), dtype=dtype
    )[:, None] * torch.tensor((1.0, -0.5, 0.25), dtype=dtype)
    positions[translated] += origins[translated, None]

    # nonfinite/zero-quaternion=[26:30], all fail closed even though geometry is far.
    positions[26, 7, 0] = float("nan")
    quats[27, 11, 2] = float("inf")
    quats[28, 19] = 0.0
    origins[29, 1] = float("nan")

    # Each nonzero-index table box gets one positive component row [30:34].
    # Row 34 is blade-only because component owners exclude body 31.  Row 35
    # proves quaternion normalization, rather than unit input, owns the pose.
    for row, table in zip(range(30, 34), range(1, 5)):
        positions[row, 0] = 0.5 * (lo[table] + hi[table]) + origins[row]
    positions[34, 31] = origins[34]
    positions[35, 0] = origins[35]
    quats[35] *= 2.0

    # Rows 36/37 are the same rotated anisotropic OBB at exact contact and one
    # representable step outside it.  This catches FMA/operation-order drift at
    # the only place where a float difference can flip the terminal bit.
    theta = torch.tensor(np.pi / 4.0, dtype=dtype)
    rotated_quat = torch.stack(
        (
            torch.cos(0.5 * theta),
            torch.zeros_like(theta),
            torch.zeros_like(theta),
            torch.sin(0.5 * theta),
        )
    )
    quats[36:38, 0] = rotated_quat
    normalized_rotated_quat = rotated_quat / torch.linalg.vector_norm(rotated_quat)
    rotated_axes = keepout._quat_rotate_wxyz(
        normalized_rotated_quat.expand(3, -1), anisotropic
    )
    touching_center_x = 1.0 + torch.sum(torch.abs(rotated_axes[:, 0]))
    separated_center_x = torch.nextafter(
        touching_center_x, torch.tensor(float("inf"), dtype=dtype)
    )
    positions[36, 0] = (
        torch.stack(
            (touching_center_x, torch.zeros_like(theta), torch.zeros_like(theta))
        )
        + origins[36]
    )
    positions[37, 0] = (
        torch.stack(
            (separated_center_x, torch.zeros_like(theta), torch.zeros_like(theta))
        )
        + origins[37]
    )

    # random=[38:94], generated once from a fixed seed and intentionally not
    # concentrated near a boundary.  Its verdict comes from the retained oracle.
    generator = torch.Generator(device="cpu").manual_seed(20260825)
    positions[38:] = torch.empty((56, 32, 3), dtype=dtype).uniform_(
        -3.0, 3.0, generator=generator
    )
    random_quat = torch.randn((56, 32, 4), dtype=dtype, generator=generator)
    quats[38:] = random_quat / torch.linalg.vector_norm(
        random_quat, dim=-1, keepdim=True
    )
    return {
        "body_pos_w": positions,
        "body_quat_w": quats,
        "env_origins": origins,
        "component_owner_indices": owners,
        "component_local_centers": centers,
        "component_local_half_axes": half_axes,
        "table_lo": lo,
        "table_hi": hi,
        "racket_body_index": 31,
        "blade_center_offset": torch.zeros(3, dtype=dtype),
        "blade_local_half_axes": torch.eye(3, dtype=dtype).mul(0.125),
    }


def _cuda_bound_verdict(inputs, device, *, full_body=False):
    """Exercise the only CUDA production path with synthetic bound constants."""

    tensor_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    body_ids = torch.arange(32, dtype=torch.long, device=device)
    positions = tensor_inputs["body_pos_w"]
    quats = tensor_inputs["body_quat_w"]
    if full_body:
        body_ids = torch.tensor((31, *range(31)), dtype=torch.long, device=device)
        count = positions.shape[0]
        full_positions = torch.full(
            (count, 47, 3), float("nan"), dtype=torch.float32, device=device
        )
        full_quats = torch.full(
            (count, 47, 4), float("nan"), dtype=torch.float32, device=device
        )
        full_positions[:, body_ids] = positions
        full_quats[:, body_ids] = quats
        positions = full_positions
        quats = full_quats

    bound = object.__new__(keepout.DeviceExactTableKeepout)
    bound.env_origins = tensor_inputs["env_origins"]
    bound.body_ids = body_ids
    bound.component_owner_indices = tensor_inputs["component_owner_indices"]
    bound.component_local_centers = tensor_inputs["component_local_centers"]
    bound.component_local_half_axes = tensor_inputs["component_local_half_axes"]
    bound.table_lo = tensor_inputs["table_lo"]
    bound.table_hi = tensor_inputs["table_hi"]
    bound.racket_body_index = inputs["racket_body_index"]
    bound.blade_center_offset = tensor_inputs["blade_center_offset"]
    bound.blade_local_half_axes = tensor_inputs["blade_local_half_axes"]
    bound._bind_cuda_kernel(body_count=positions.shape[1])
    wp_data = SimpleNamespace(
        xpos=keepout._wp.from_torch(
            positions, dtype=keepout._wp.vec3, requires_grad=False
        ),
        xquat=keepout._wp.from_torch(
            quats, dtype=keepout._wp.quat, requires_grad=False
        ),
    )
    return bound.sample(SimpleNamespace(struct=wp_data))


def _cuda_compile_first_witness_kernel(inputs, device):
    """Launch the replay-only kernel, not merely its host result decoder."""

    live_rows = {"body_pos_w", "body_quat_w", "env_origins"}
    tensor_inputs = {
        key: value[16:17].to(device) if key in live_rows else value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    bound = object.__new__(keepout.DeviceExactTableKeepout)
    bound.env_origins = tensor_inputs["env_origins"]
    bound.body_ids = torch.arange(32, dtype=torch.long, device=device)
    bound.component_owner_indices = tensor_inputs["component_owner_indices"]
    bound.component_local_centers = tensor_inputs["component_local_centers"]
    bound.component_local_half_axes = tensor_inputs["component_local_half_axes"]
    bound.table_lo = tensor_inputs["table_lo"]
    bound.table_hi = tensor_inputs["table_hi"]
    bound.racket_body_index = inputs["racket_body_index"]
    bound.blade_center_offset = tensor_inputs["blade_center_offset"]
    bound.blade_local_half_axes = tensor_inputs["blade_local_half_axes"]
    bound._bind_cuda_kernel(body_count=32)
    positions = tensor_inputs["body_pos_w"]
    quaternions = tensor_inputs["body_quat_w"]
    wp_pos = keepout._wp.from_torch(
        positions, dtype=keepout._wp.vec3, requires_grad=False
    )
    wp_quat = keepout._wp.from_torch(
        quaternions, dtype=keepout._wp.quat, requires_grad=False
    )
    winner = torch.full((4,), -1, dtype=torch.int64, device=device)
    witness = torch.full((15,), float("nan"), dtype=torch.float32, device=device)
    keepout._wp.launch(
        keepout._warp_table_keepout_first_witness_f32,
        dim=1,
        inputs=(wp_pos, wp_quat, *bound._cuda_warp_static_inputs),
        outputs=(
            keepout._wp.from_torch(winner, requires_grad=False),
            keepout._wp.from_torch(witness, requires_grad=False),
        ),
        device=bound._cuda_warp_device,
        stream=bound._current_warp_stream(),
        record_tape=False,
    )
    torch.cuda.synchronize(torch.device(device))
    return winner.cpu(), witness.cpu()


def _tape_verdict(inputs, device):
    if torch.device(device).type == "cuda":
        return _cuda_bound_verdict(inputs, device)
    tensor_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    return keepout.geometric_robot_table_hit_mask(
        tensor_inputs["body_pos_w"],
        tensor_inputs["body_quat_w"],
        tensor_inputs["env_origins"],
        tensor_inputs["component_owner_indices"],
        tensor_inputs["component_local_centers"],
        tensor_inputs["component_local_half_axes"],
        tensor_inputs["table_lo"],
        tensor_inputs["table_hi"],
        racket_body_index=inputs["racket_body_index"],
        blade_center_offset=tensor_inputs["blade_center_offset"],
        blade_local_half_axes=tensor_inputs["blade_local_half_axes"],
    )


def _full_body_tape_verdict(inputs, device):
    """Exercise the production local-body -> MuJoCo-body gather inside Warp."""

    return _cuda_bound_verdict(inputs, device, full_body=True)


def _assert_fixed_tape_strata(verdict):
    assert verdict[0:8].tolist() == [False] * 8
    assert verdict[8:16].tolist() == [
        False, True, False, False, False, False, True, False
    ]
    assert verdict[16:24].tolist() == [True] * 8
    assert verdict[24:26].tolist() == [True, False]
    assert verdict[26:30].tolist() == [True] * 4
    assert verdict[30:36].tolist() == [True] * 6
    assert verdict[36:38].tolist() == [True, False]


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


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_fixed_shape_tape_covers_none_sparse_all_touching_nonfinite_random(dtype):
    inputs = _fixed_shape_tape(dtype)
    actual = _tape_verdict(inputs, "cpu")
    oracle = keepout._host_test_geometric_robot_table_hit_mask(
        inputs["body_pos_w"],
        inputs["body_quat_w"],
        inputs["env_origins"],
        inputs["component_owner_indices"],
        inputs["component_local_centers"],
        inputs["component_local_half_axes"],
        inputs["table_lo"],
        inputs["table_hi"],
        racket_body_index=inputs["racket_body_index"],
        blade_center_offset=inputs["blade_center_offset"],
        blade_local_half_axes=inputs["blade_local_half_axes"],
    )
    assert torch.equal(actual, oracle)
    _assert_fixed_tape_strata(actual)


def test_missing_warp_backend_fails_loud_instead_of_selecting_eager(monkeypatch):
    import_error = ImportError("deliberately absent")
    monkeypatch.setattr(keepout, "_wp", None)
    monkeypatch.setattr(keepout, "_warp_table_keepout_f32", None)
    monkeypatch.setattr(keepout, "_WARP_IMPORT_ERROR", import_error)
    with pytest.raises(
        RuntimeError, match="eager fallback is host-test-only"
    ) as caught:
        keepout._require_warp_table_keepout_kernel()
    assert caught.value.__cause__ is import_error


def test_free_function_rejects_non_cpu_instead_of_opening_second_cuda_path():
    body_pos = torch.empty((1, 32, 3), device="meta")
    with pytest.raises(RuntimeError, match="free function is a CPU oracle"):
        keepout.geometric_robot_table_hit_mask(
            body_pos,
            torch.empty((1, 32, 4), device="meta"),
            torch.empty((1, 3), device="meta"),
            torch.empty(62, dtype=torch.long, device="meta"),
            torch.empty((62, 3), device="meta"),
            torch.empty((62, 3, 3), device="meta"),
            torch.empty((5, 3), device="meta"),
            torch.empty((5, 3), device="meta"),
            racket_body_index=31,
            blade_center_offset=torch.empty(3, device="meta"),
            blade_local_half_axes=torch.empty((3, 3), device="meta"),
        )


class _FakeWarpArray:
    """Strict metadata double for a native ``warp.array`` instance."""

    def __init__(
        self,
        *,
        shape,
        dtype,
        device,
        ptr=4096,
        is_contiguous=True,
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.ptr = ptr
        self.is_contiguous = is_contiguous


def test_cuda_bound_sampler_uses_native_mjwarp_arrays_without_torch_bridge(
    monkeypatch,
):
    seen = {}

    class FakeWarp:
        array = _FakeWarpArray
        vec3 = object()
        quat = object()

        @staticmethod
        def from_torch(tensor, *, dtype=None, requires_grad=None):
            raise AssertionError("live MJWarp arrays must not be rewrapped from Torch")

        @staticmethod
        def launch(kernel, **kwargs):
            seen["kernel"] = kernel
            seen.update(kwargs)

    monkeypatch.setattr(keepout, "_wp", FakeWarp())
    bound = object.__new__(keepout.DeviceExactTableKeepout)
    bound._cuda_kernel_output = torch.empty(2, dtype=torch.bool)
    bound._cuda_warp_static_inputs = (object(),)
    bound._cuda_warp_output = object()
    warp_device = object()
    bound._cuda_warp_device = warp_device
    bound._cuda_warp_stream = object()
    bound._cuda_torch_stream_handle = 17
    bound._cuda_live_device = torch.device("cpu")
    bound._cuda_live_pos_shape = (2, 47)
    bound._cuda_live_quat_shape = (2, 47)
    bound.env_origins = torch.zeros((2, 3))
    bound.body_ids = torch.arange(32)
    bound.component_owner_indices = torch.arange(62).remainder(32)
    bound.component_local_centers = torch.zeros((62, 3))
    bound.component_local_half_axes = torch.zeros((62, 3, 3))
    bound.table_lo = torch.zeros((5, 3))
    bound.table_hi = torch.ones((5, 3))
    bound.racket_body_index = 31
    bound.blade_center_offset = torch.zeros(3)
    bound.blade_local_half_axes = torch.zeros((3, 3))
    wp_data = SimpleNamespace(
        xpos=_FakeWarpArray(
            shape=(2, 47), dtype=keepout._wp.vec3, device=warp_device
        ),
        xquat=_FakeWarpArray(
            shape=(2, 47), dtype=keepout._wp.quat, device=warp_device, ptr=8192
        ),
    )

    class DataBridgeDouble:
        struct = wp_data

        @property
        def xpos(self):
            raise AssertionError("production must bypass the TorchArray proxy")

        @property
        def xquat(self):
            raise AssertionError("production must bypass the TorchArray proxy")

    data = DataBridgeDouble()
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=17),
    )
    actual = bound.sample(data)
    assert seen["inputs"][0] is wp_data.xpos
    assert seen["inputs"][1] is wp_data.xquat
    assert seen["inputs"][2:] == bound._cuda_warp_static_inputs
    assert seen["outputs"] == (bound._cuda_warp_output,)
    assert seen["device"] is bound._cuda_warp_device
    assert seen["stream"] is bound._cuda_warp_stream
    assert actual is bound._cuda_kernel_output


def test_cuda_bound_sampler_rejects_live_contract_drift_before_launch(
    monkeypatch,
):
    class FakeWarp:
        array = _FakeWarpArray
        vec3 = object()
        quat = object()

        @staticmethod
        def launch(*_args, **_kwargs):
            pytest.fail("invalid live poses reached Warp")

    fake = FakeWarp()
    monkeypatch.setattr(keepout, "_wp", fake)
    bound = object.__new__(keepout.DeviceExactTableKeepout)
    bound._cuda_kernel_output = torch.empty(2, dtype=torch.bool)
    warp_device = object()
    bound._cuda_warp_device = warp_device
    bound._cuda_live_device = torch.device("cpu")
    bound._cuda_live_pos_shape = (2, 47)
    bound._cuda_live_quat_shape = (2, 47)
    valid_pos = _FakeWarpArray(
        shape=(2, 47), dtype=fake.vec3, device=warp_device
    )
    valid_quat = _FakeWarpArray(
        shape=(2, 47), dtype=fake.quat, device=warp_device, ptr=8192
    )
    duck_array = SimpleNamespace(
        shape=(2, 47),
        dtype=fake.vec3,
        device=warp_device,
        ptr=4096,
        is_contiguous=True,
    )
    cases = (
        (duck_array, valid_quat, "requires native MJWarp pose arrays"),
        (
            _FakeWarpArray(
                shape=(2, 47), dtype=object(), device=warp_device
            ),
            valid_quat,
            "live pose dtype differs",
        ),
        (
            _FakeWarpArray(shape=(2, 46), dtype=fake.vec3, device=warp_device),
            valid_quat,
            "live pose shape differs",
        ),
        (
            _FakeWarpArray(shape=(2, 47), dtype=fake.vec3, device=object()),
            valid_quat,
            "live pose device differs",
        ),
        (
            _FakeWarpArray(
                shape=(2, 47), dtype=fake.vec3, device=warp_device, ptr=0
            ),
            valid_quat,
            "live pose pointer is null",
        ),
    )
    for xpos, xquat, message in cases:
        data = SimpleNamespace(struct=SimpleNamespace(xpos=xpos, xquat=xquat))
        with pytest.raises(RuntimeError, match=message):
            bound.sample(data)


def test_cuda_bound_sampler_caches_static_wrappers_and_rebinds_only_on_stream_change(
    monkeypatch,
):
    class FakeWarp:
        array = _FakeWarpArray
        vec3 = object()
        quat = object()
        mat33 = object()
        bool = object()

        def __init__(self):
            self.wraps = []
            self.streams = []
            self.launches = []

        def from_torch(self, tensor, *, dtype=None, requires_grad=None):
            wrapped = object()
            self.wraps.append((tensor, dtype, requires_grad, wrapped))
            return wrapped

        def stream_from_torch(self, stream):
            wrapped = object()
            self.streams.append((stream.cuda_stream, wrapped))
            return wrapped

        def device_from_torch(self, _device):
            raise AssertionError("the bound Warp device must be reused")

        def launch(self, kernel, **kwargs):
            self.launches.append((kernel, kwargs))

    fake = FakeWarp()
    monkeypatch.setattr(keepout, "_wp", fake)

    bound = object.__new__(keepout.DeviceExactTableKeepout)
    bound.env_origins = torch.zeros((2, 3))
    bound.body_ids = torch.arange(32)
    bound.component_owner_indices = torch.arange(62).remainder(32)
    bound.component_local_centers = torch.zeros((62, 3))
    bound.component_local_half_axes = torch.zeros((62, 3, 3))
    bound.table_lo = torch.zeros((5, 3))
    bound.table_hi = torch.ones((5, 3))
    bound.racket_body_index = 31
    bound.blade_center_offset = torch.zeros(3)
    bound.blade_local_half_axes = torch.zeros((3, 3))
    bound._cuda_kernel_output = torch.empty(2, dtype=torch.bool)
    bound._cuda_warp_static_inputs = keepout._warp_static_launch_inputs(
        bound.env_origins,
        bound.body_ids,
        bound.component_owner_indices,
        bound.component_local_centers,
        bound.component_local_half_axes,
        bound.table_lo,
        bound.table_hi,
        racket_body_index=bound.racket_body_index,
        blade_center_offset=bound.blade_center_offset,
        blade_local_half_axes=bound.blade_local_half_axes,
    )
    bound._cuda_warp_output = fake.from_torch(
        bound._cuda_kernel_output, dtype=fake.bool, requires_grad=False
    )
    warp_device = object()
    bound._cuda_warp_device = warp_device
    bound._cuda_warp_stream = None
    bound._cuda_torch_stream_handle = None
    bound._cuda_live_device = torch.device("cpu")
    bound._cuda_live_pos_shape = (2, 47)
    bound._cuda_live_quat_shape = (2, 47)
    monkeypatch.setattr(
        keepout,
        "_validate_cuda_static_inputs",
        lambda *_args, **_kwargs: pytest.fail(
            "sample repeated construction-time static validation"
        ),
    )

    current = SimpleNamespace(cuda_stream=41)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current)
    wp_data = SimpleNamespace(
        xpos=_FakeWarpArray(
            shape=(2, 47), dtype=fake.vec3, device=warp_device
        ),
        xquat=_FakeWarpArray(
            shape=(2, 47), dtype=fake.quat, device=warp_device, ptr=8192
        ),
    )
    data = SimpleNamespace(struct=wp_data)
    construction_wrap_count = len(fake.wraps)

    for _ in range(2):
        assert bound.sample(data) is bound._cuda_kernel_output
    current.cuda_stream = 73
    for _ in range(2):
        assert bound.sample(data) is bound._cuda_kernel_output

    assert [handle for handle, _wrapped in fake.streams] == [41, 73]
    assert [launch[1]["stream"] for launch in fake.launches] == [
        fake.streams[0][1],
        fake.streams[0][1],
        fake.streams[1][1],
        fake.streams[1][1],
    ]
    static_tensors = (
        bound.env_origins,
        bound.body_ids,
        bound.component_owner_indices,
        bound.component_local_centers,
        bound.component_local_half_axes,
        bound.table_lo,
        bound.table_hi,
        bound._cuda_kernel_output,
    )
    for tensor in static_tensors:
        assert sum(wrapped[0] is tensor for wrapped in fake.wraps) == 1
    assert (
        sum(
            wrapped[0]._base is bound.blade_center_offset
            for wrapped in fake.wraps
        )
        == 1
    )
    assert (
        sum(
            wrapped[0]._base is bound.blade_local_half_axes
            for wrapped in fake.wraps
        )
        == 1
    )
    assert len(fake.wraps) == construction_wrap_count
    assert all(wrapped[0] is not wp_data.xpos for wrapped in fake.wraps)
    assert all(wrapped[0] is not wp_data.xquat for wrapped in fake.wraps)
    assert all(wrapped[2] is False for wrapped in fake.wraps)


@pytest.mark.skipif(
    os.environ.get("ACTIONBALL_RUN_MUJOCO_GPU_DIRECT") != "1",
    reason="requires the exact MuJoCo-Warp GPU environment",
)
def test_warp_gpu_guard_matches_existing_numpy_authority_exactly():
    device = os.environ.get("ACTIONBALL_MUJOCO_DEVICE", "cuda:0")
    actual, expected = _device_verdict(device, dtype=torch.float32)
    assert torch.equal(actual, expected)
    assert _empty_corner_discriminator(device, torch.float32).tolist() == [False]
    inputs = _fixed_shape_tape(torch.float32)
    oracle = _tape_verdict(inputs, "cpu")
    kernel = _tape_verdict(inputs, device).cpu()
    assert torch.equal(kernel, oracle)
    _assert_fixed_tape_strata(kernel)
    full_body_kernel = _full_body_tape_verdict(inputs, device).cpu()
    assert torch.equal(full_body_kernel, oracle)
    winner, witness = _cuda_compile_first_witness_kernel(inputs, device)
    assert winner.shape == (4,)
    assert int(winner[3]) == 4
    assert torch.isfinite(witness).all()
