"""Device-resident port of the exact Isaac robot/table pose guard.

Construction binds the tracked plant, 62 collision-component OBBs, and canonical
five-box table assembly. CUDA sampling is one fixed-shape Warp SAT launch with no
host readback; Torch remains only as the dependency-light CPU test oracle.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType

import torch

try:
    import warp as _wp
except ImportError as _warp_import_exc:  # pragma: no cover - host-only test wheels
    _wp = None
    _WARP_IMPORT_ERROR = _warp_import_exc
    _WARP_READY = False
else:
    _WARP_IMPORT_ERROR = None
    _WARP_READY = False


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_WBT_ROOT = _HERE.parent
if str(_WBT_ROOT) not in sys.path:
    sys.path.insert(0, str(_WBT_ROOT))

from mujoco_native import table_termination as _authority  # noqa: E402
if (Path(_authority.__file__).resolve(), _authority.REPO_ROOT.resolve()) != (
    _WBT_ROOT / "mujoco_native/table_termination.py", _REPO
):
    raise RuntimeError("MuJoCo table keepout authority resolved outside this tree")


def _load_table_scene():
    path = _REPO / "scripts/mujoco_table_scene.py"
    spec = importlib.util.spec_from_file_location(
        "_mujoco_gpu_ac_table_scene", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical table scene from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quat_rotate_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    scalar = quat[..., :1]
    xyz = quat[..., 1:]
    return vector + 2.0 * (
        scalar * torch.cross(xyz, vector, dim=-1)
        + torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1)
    )


def _sat_overlap(
    center: torch.Tensor,
    half_axes: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    broad: torch.Tensor,
) -> torch.Tensor:
    """Exact fixed-shape 15-axis OBB/AABB SAT, matching the Isaac kernel."""

    pair_axes = half_axes[:, :, None]
    box_center = 0.5 * (lo + hi)
    box_half = 0.5 * (hi - lo)
    delta = box_center[None, None] - center[:, :, None]
    norm = torch.linalg.vector_norm(pair_axes, dim=-1)
    unit_axes = pair_axes / torch.clamp(
        norm, min=torch.finfo(center.dtype).tiny
    )[..., None]
    overlap = broad.clone()

    def apply(axis: torch.Tensor) -> None:
        separation = torch.abs(torch.sum(delta * axis, dim=-1))
        obb_radius = torch.sum(
            torch.abs(torch.sum(pair_axes * axis[..., None, :], dim=-1)),
            dim=-1,
        )
        box_radius = torch.sum(
            box_half[None, None] * torch.abs(axis), dim=-1
        )
        overlap.logical_and_(separation <= obb_radius + box_radius)

    world = torch.eye(3, dtype=center.dtype, device=center.device)
    for axis in range(3):
        apply(world[axis])
    for obb_axis in range(3):
        axis = unit_axes[..., obb_axis, :]
        apply(axis)
        for world_axis in range(3):
            apply(torch.cross(axis, world[world_axis].expand_as(axis), dim=-1))
    return overlap


def _host_test_geometric_robot_table_hit_mask(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    env_origins: torch.Tensor,
    component_owner_indices: torch.Tensor,
    component_local_centers: torch.Tensor,
    component_local_half_axes: torch.Tensor,
    table_lo: torch.Tensor,
    table_hi: torch.Tensor,
    *,
    racket_body_index: int,
    blade_center_offset: torch.Tensor,
    blade_local_half_axes: torch.Tensor,
) -> torch.Tensor:
    """Torch oracle retained only for dependency-light CPU tests.

    The CUDA production path below never calls this function.  In particular,
    a missing or broken Warp binding must stop construction/launch rather than
    silently restoring the many-kernel eager implementation.
    """

    if body_pos_w.device.type != "cpu":
        raise RuntimeError("the table keepout Torch fallback is host-test-only")

    local_pos = body_pos_w - env_origins[:, None]
    norm_sq = torch.sum(body_quat_w * body_quat_w, dim=-1, keepdim=True)
    quat = body_quat_w / torch.sqrt(
        torch.clamp(norm_sq, min=torch.finfo(body_pos_w.dtype).tiny)
    )
    owner_quat = torch.index_select(quat, 1, component_owner_indices)
    owner_pos = torch.index_select(local_pos, 1, component_owner_indices)
    centers = owner_pos + _quat_rotate_wxyz(
        owner_quat,
        component_local_centers.unsqueeze(0).expand(body_pos_w.shape[0], -1, -1),
    )
    half_axes = torch.stack(
        tuple(
            _quat_rotate_wxyz(
                owner_quat,
                component_local_half_axes[:, axis]
                .unsqueeze(0)
                .expand(body_pos_w.shape[0], -1, -1),
            )
            for axis in range(3)
        ),
        dim=2,
    )
    world_half = torch.sum(torch.abs(half_axes), dim=2) + 1.0e-6
    broad = torch.all(
        (centers[:, :, None] + world_half[:, :, None] >= table_lo[None, None])
        & (centers[:, :, None] - world_half[:, :, None] <= table_hi[None, None]),
        dim=-1,
    )

    racket_quat = quat[:, racket_body_index]
    blade_center = local_pos[:, racket_body_index] + _quat_rotate_wxyz(
        racket_quat, blade_center_offset.expand(body_pos_w.shape[0], -1)
    )
    blade_axes = _quat_rotate_wxyz(
        racket_quat[:, None].expand(-1, 3, -1),
        blade_local_half_axes.unsqueeze(0).expand(body_pos_w.shape[0], -1, -1),
    )
    blade_half = torch.sum(torch.abs(blade_axes), dim=1)
    blade_broad = torch.all(
        (blade_center[:, None] + blade_half[:, None] >= table_lo[None])
        & (blade_center[:, None] - blade_half[:, None] <= table_hi[None]),
        dim=-1,
    )
    fused_center = torch.cat((centers, blade_center[:, None]), dim=1)
    fused_axes = torch.cat((half_axes, blade_axes[:, None]), dim=1)
    fused_broad = torch.cat((broad, blade_broad[:, None]), dim=1)
    exact = _sat_overlap(fused_center, fused_axes, table_lo, table_hi, fused_broad)
    invalid = (
        ~torch.isfinite(body_pos_w).reshape(body_pos_w.shape[0], -1).all(dim=1)
        | ~torch.isfinite(body_quat_w).reshape(body_pos_w.shape[0], -1).all(dim=1)
        | ~torch.isfinite(env_origins).all(dim=1)
        | ~(norm_sq[..., 0] > 0.0).all(dim=1)
    )
    return exact.reshape(exact.shape[0], -1).any(dim=1) | invalid


if _wp is not None:

    @_wp.func
    def _warp_quat_rotate_wxyz_f32(q: _wp.vec4, v: _wp.vec3) -> _wp.vec3:
        xyz = _wp.vec3(q[1], q[2], q[3])
        first = _wp.cross(xyz, v)
        return v + 2.0 * (q[0] * first + _wp.cross(xyz, first))


    @_wp.func
    def _warp_sat_axis_f32(
        delta: _wp.vec3,
        half_axes: _wp.mat33,
        box_half: _wp.vec3,
        axis: _wp.vec3,
    ) -> _wp.bool:
        separation = _wp.abs(_wp.dot(delta, axis))
        obb_radius = 0.0
        for index in range(3):
            obb_radius += _wp.abs(
                _wp.dot(_warp_mat_row_f32(half_axes, index), axis)
            )
        box_radius = (
            box_half[0] * _wp.abs(axis[0])
            + box_half[1] * _wp.abs(axis[1])
            + box_half[2] * _wp.abs(axis[2])
        )
        return separation <= obb_radius + box_radius


    @_wp.func
    def _warp_world_axis_f32(index: int) -> _wp.vec3:
        if index == 0:
            return _wp.vec3(1.0, 0.0, 0.0)
        if index == 1:
            return _wp.vec3(0.0, 1.0, 0.0)
        return _wp.vec3(0.0, 0.0, 1.0)


    @_wp.func
    def _warp_mat_row_f32(matrix: _wp.mat33, row: int) -> _wp.vec3:
        return _wp.vec3(matrix[row, 0], matrix[row, 1], matrix[row, 2])


    @_wp.func
    def _warp_world_half_f32(
        half_axes: _wp.mat33, guard: float
    ) -> _wp.vec3:
        return _wp.vec3(
            _wp.abs(half_axes[0, 0])
            + _wp.abs(half_axes[1, 0])
            + _wp.abs(half_axes[2, 0])
            + guard,
            _wp.abs(half_axes[0, 1])
            + _wp.abs(half_axes[1, 1])
            + _wp.abs(half_axes[2, 1])
            + guard,
            _wp.abs(half_axes[0, 2])
            + _wp.abs(half_axes[1, 2])
            + _wp.abs(half_axes[2, 2])
            + guard,
        )


    @_wp.func
    def _warp_broad_overlap_f32(
        center: _wp.vec3, half: _wp.vec3, lo: _wp.vec3, hi: _wp.vec3
    ) -> _wp.bool:
        return (
            center[0] + half[0] >= lo[0]
            and center[0] - half[0] <= hi[0]
            and center[1] + half[1] >= lo[1]
            and center[1] - half[1] <= hi[1]
            and center[2] + half[2] >= lo[2]
            and center[2] - half[2] <= hi[2]
        )


    @_wp.func
    def _warp_sat_overlap_f32(
        center: _wp.vec3,
        half_axes: _wp.mat33,
        lo: _wp.vec3,
        hi: _wp.vec3,
    ) -> _wp.bool:
        box_center = 0.5 * (lo + hi)
        box_half = 0.5 * (hi - lo)
        delta = box_center - center
        for world_index in range(3):
            if not _warp_sat_axis_f32(
                delta,
                half_axes,
                box_half,
                _warp_world_axis_f32(world_index),
            ):
                return False
        for obb_index in range(3):
            half_axis = _warp_mat_row_f32(half_axes, obb_index)
            norm = _wp.max(
                _wp.sqrt(_wp.dot(half_axis, half_axis)),
                1.1754943508222875e-38,
            )
            unit_axis = half_axis / norm
            if not _warp_sat_axis_f32(
                delta,
                half_axes,
                box_half,
                unit_axis,
            ):
                return False
            for world_index in range(3):
                cross_axis = _wp.cross(
                    unit_axis, _warp_world_axis_f32(world_index)
                )
                if not _warp_sat_axis_f32(
                    delta,
                    half_axes,
                    box_half,
                    cross_axis,
                ):
                    return False
        return True


    @_wp.kernel(enable_backward=False)
    def _warp_table_keepout_f32(
        body_pos_w: _wp.array(dtype=_wp.vec3, ndim=2),
        body_quat_w: _wp.array(dtype=_wp.vec4, ndim=2),
        env_origins: _wp.array(dtype=_wp.vec3, ndim=1),
        body_ids: _wp.array(dtype=_wp.int64, ndim=1),
        component_owner_indices: _wp.array(dtype=_wp.int64, ndim=1),
        component_local_centers: _wp.array(dtype=_wp.vec3, ndim=1),
        component_local_half_axes: _wp.array(dtype=_wp.mat33, ndim=1),
        table_lo: _wp.array(dtype=_wp.vec3, ndim=1),
        table_hi: _wp.array(dtype=_wp.vec3, ndim=1),
        racket_body_index: int,
        blade_center_offset: _wp.array(dtype=_wp.vec3, ndim=1),
        blade_local_half_axes: _wp.array(dtype=_wp.mat33, ndim=1),
        output: _wp.array(dtype=_wp.bool, ndim=1),
    ):
        env = _wp.tid()
        origin = env_origins[env]
        if not _wp.isfinite(origin):
            output[env] = True
            return

        # Fail closed before geometric early-out.  Every one of the 32 selected
        # body poses belongs to the existing oracle's nonfinite/zero-quaternion
        # domain, even when no collision component happens to reference it.
        for local_body in range(32):
            body = int(body_ids[local_body])
            pos = body_pos_w[env, body]
            quat = body_quat_w[env, body]
            norm_sq = _wp.dot(quat, quat)
            if (not _wp.isfinite(pos)) or (not _wp.isfinite(quat)) or norm_sq <= 0.0:
                output[env] = True
                return

        # Components 0..61 and the blade at 62 share one exact SAT path.  This
        # preserves the authority order and permits an environment-local return
        # as soon as the first table box is hit.
        for obb in range(63):
            local_body = racket_body_index
            local_center = blade_center_offset[0]
            local_axes = blade_local_half_axes[0]
            broad_guard = 0.0
            if obb < 62:
                local_body = int(component_owner_indices[obb])
                local_center = component_local_centers[obb]
                local_axes = component_local_half_axes[obb]
                broad_guard = 1.0e-6
            body = int(body_ids[local_body])
            pos = body_pos_w[env, body] - origin
            quat = body_quat_w[env, body]
            quat = quat / _wp.sqrt(_wp.max(_wp.dot(quat, quat), 1.1754943508222875e-38))
            center = pos + _warp_quat_rotate_wxyz_f32(quat, local_center)
            half_axes = _wp.matrix_from_rows(
                _warp_quat_rotate_wxyz_f32(
                    quat, _warp_mat_row_f32(local_axes, 0)
                ),
                _warp_quat_rotate_wxyz_f32(
                    quat, _warp_mat_row_f32(local_axes, 1)
                ),
                _warp_quat_rotate_wxyz_f32(
                    quat, _warp_mat_row_f32(local_axes, 2)
                ),
            )
            world_half = _warp_world_half_f32(half_axes, broad_guard)
            for table in range(5):
                lo = table_lo[table]
                hi = table_hi[table]
                if _warp_broad_overlap_f32(
                    center, world_half, lo, hi
                ) and _warp_sat_overlap_f32(
                    center, half_axes, lo, hi
                ):
                    output[env] = True
                    return
        output[env] = False

else:
    _warp_table_keepout_f32 = None


def _require_warp_table_keepout_kernel() -> None:
    global _WARP_READY
    if _wp is None or _warp_table_keepout_f32 is None:
        raise RuntimeError(
            "CUDA table keepout requires the Warp 1.16 single-kernel backend; "
            "the eager fallback is host-test-only"
        ) from _WARP_IMPORT_ERROR
    if not _WARP_READY:
        try:
            _wp.init()
        except Exception as exc:
            raise RuntimeError(
                "CUDA table keepout could not initialize its required Warp backend"
            ) from exc
        _WARP_READY = True


def _validate_cuda_static_inputs(
    env_origins: torch.Tensor,
    body_ids: torch.Tensor,
    component_owner_indices: torch.Tensor,
    component_local_centers: torch.Tensor,
    component_local_half_axes: torch.Tensor,
    table_lo: torch.Tensor,
    table_hi: torch.Tensor,
    *,
    racket_body_index: int,
    blade_center_offset: torch.Tensor,
    blade_local_half_axes: torch.Tensor,
    body_count: int,
) -> None:
    tensors = (
        env_origins,
        body_ids,
        component_owner_indices,
        component_local_centers,
        component_local_half_axes,
        table_lo,
        table_hi,
        blade_center_offset,
        blade_local_half_axes,
    )
    device = env_origins.device
    if device.type != "cuda" or any(value.device != device for value in tensors):
        raise RuntimeError("CUDA table keepout static inputs must share one CUDA device")
    if any(
        value.dtype != torch.float32
        for value in (
            env_origins,
            component_local_centers,
            component_local_half_axes,
            table_lo,
            table_hi,
            blade_center_offset,
            blade_local_half_axes,
        )
    ):
        raise RuntimeError("CUDA table keepout production contract is float32")
    if component_owner_indices.dtype != torch.int64 or body_ids.dtype != torch.int64:
        raise RuntimeError("CUDA table keepout owner/body indices must be int64")
    env_shape = tuple(env_origins.shape)
    if len(env_shape) != 2 or env_shape[1:] != (3,):
        raise RuntimeError("CUDA table keepout environment origins shape differs")
    env_count = env_shape[0]
    expected = {
        "env_origins": (env_count, 3),
        "body_ids": (32,),
        "component_owner_indices": (62,),
        "component_local_centers": (62, 3),
        "component_local_half_axes": (62, 3, 3),
        "table_lo": (5, 3),
        "table_hi": (5, 3),
        "blade_center_offset": (3,),
        "blade_local_half_axes": (3, 3),
    }
    actual = {
        "env_origins": tuple(env_origins.shape),
        "body_ids": tuple(body_ids.shape),
        "component_owner_indices": tuple(component_owner_indices.shape),
        "component_local_centers": tuple(component_local_centers.shape),
        "component_local_half_axes": tuple(component_local_half_axes.shape),
        "table_lo": tuple(table_lo.shape),
        "table_hi": tuple(table_hi.shape),
        "blade_center_offset": tuple(blade_center_offset.shape),
        "blade_local_half_axes": tuple(blade_local_half_axes.shape),
    }
    if actual != expected:
        raise RuntimeError(
            f"CUDA table keepout fixed-shape contract differs: {actual!r}"
        )
    if type(body_count) is not int or body_count <= 0:
        raise RuntimeError("CUDA table keepout live body count differs")
    if type(racket_body_index) is not int or not 0 <= racket_body_index < 32:
        raise RuntimeError("CUDA table keepout racket body index differs")
    if bool(((body_ids < 0) | (body_ids >= body_count)).any().item()):
        raise RuntimeError("CUDA table keepout body-id binding exceeds live pose rows")
    if bool(
        (
            (component_owner_indices < 0)
            | (component_owner_indices >= body_ids.shape[0])
        )
        .any()
        .item()
    ):
        raise RuntimeError("CUDA table keepout component owner binding differs")


def _validate_cuda_live_poses(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    *,
    device: torch.device,
    pos_shape: tuple[int, int, int],
    quat_shape: tuple[int, int, int],
) -> None:
    """Check only the two live MuJoCo views before a bound Warp launch."""

    if not isinstance(body_pos_w, torch.Tensor) or not isinstance(
        body_quat_w, torch.Tensor
    ):
        raise RuntimeError("CUDA table keepout live poses must be Torch tensors")
    if body_pos_w.device != device or body_quat_w.device != device:
        raise RuntimeError("CUDA table keepout live pose device differs")
    if body_pos_w.dtype != torch.float32 or body_quat_w.dtype != torch.float32:
        raise RuntimeError("CUDA table keepout live pose dtype differs")
    if tuple(body_pos_w.shape) != pos_shape or tuple(body_quat_w.shape) != quat_shape:
        raise RuntimeError("CUDA table keepout live pose shape differs")


def _warp_static_launch_inputs(
    env_origins: torch.Tensor,
    body_ids: torch.Tensor,
    component_owner_indices: torch.Tensor,
    component_local_centers: torch.Tensor,
    component_local_half_axes: torch.Tensor,
    table_lo: torch.Tensor,
    table_hi: torch.Tensor,
    *,
    racket_body_index: int,
    blade_center_offset: torch.Tensor,
    blade_local_half_axes: torch.Tensor,
) -> tuple[object, ...]:
    """Bind run-static Torch storage to Warp once without copying it."""

    return (
        _wp.from_torch(env_origins, dtype=_wp.vec3, requires_grad=False),
        _wp.from_torch(body_ids, requires_grad=False),
        _wp.from_torch(component_owner_indices, requires_grad=False),
        _wp.from_torch(
            component_local_centers, dtype=_wp.vec3, requires_grad=False
        ),
        _wp.from_torch(
            component_local_half_axes, dtype=_wp.mat33, requires_grad=False
        ),
        _wp.from_torch(table_lo, dtype=_wp.vec3, requires_grad=False),
        _wp.from_torch(table_hi, dtype=_wp.vec3, requires_grad=False),
        racket_body_index,
        _wp.from_torch(
            blade_center_offset.unsqueeze(0),
            dtype=_wp.vec3,
            requires_grad=False,
        ),
        _wp.from_torch(
            blade_local_half_axes.unsqueeze(0),
            dtype=_wp.mat33,
            requires_grad=False,
        ),
    )


def geometric_robot_table_hit_mask(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    env_origins: torch.Tensor,
    component_owner_indices: torch.Tensor,
    component_local_centers: torch.Tensor,
    component_local_half_axes: torch.Tensor,
    table_lo: torch.Tensor,
    table_hi: torch.Tensor,
    *,
    racket_body_index: int,
    blade_center_offset: torch.Tensor,
    blade_local_half_axes: torch.Tensor,
) -> torch.Tensor:
    """Return the exact Isaac-equivalent keepout bit for every environment.

    This free function is the dependency-light CPU oracle only.  CUDA callers
    must use the construction-bound :class:`DeviceExactTableKeepout` sampler.
    """

    if body_pos_w.device.type != "cpu":
        raise RuntimeError(
            "table keepout free function is a CPU oracle; "
            "CUDA production requires DeviceExactTableKeepout"
        )
    return _host_test_geometric_robot_table_hit_mask(
        body_pos_w,
        body_quat_w,
        env_origins,
        component_owner_indices,
        component_local_centers,
        component_local_half_axes,
        table_lo,
        table_hi,
        racket_body_index=racket_body_index,
        blade_center_offset=blade_center_offset,
        blade_local_half_axes=blade_local_half_axes,
    )


class DeviceExactTableKeepout:
    """Construction-bound constants plus one tensor-only live sampler."""

    def __init__(self, *, mujoco, model, mjcf_path, env_origins, device) -> None:
        scene = _load_table_scene()
        rows = scene.action_ball_policy_obstacle_geometry()
        contract = scene.action_ball_policy_geometry_contract(rows)
        authority = _authority.ExactRobotTableGuard(
            mujoco,
            model,
            contract,
            mjcf_path=mjcf_path,
            body_name_prefix="robot/",
        )
        identity = authority.identity_receipt
        identity_keys = (
            "root_mjcf_sha256",
            "identity_manifest_sha256",
            "portable_identity_sha256",
            "verification_receipt_sha256",
            "owner_local_frame_sha256",
        )
        if any(
            type(identity.get(key)) is not str
            or len(identity[key]) != 64
            or any(char not in "0123456789abcdef" for char in identity[key])
            for key in identity_keys
        ):
            raise RuntimeError("MuJoCo table keepout plant identity receipt differs")
        self.plant_identity_receipt = MappingProxyType(
            {key: identity[key] for key in identity_keys}
        )
        dtype = env_origins.dtype

        def tensor(value, tensor_dtype=dtype):
            return torch.as_tensor(value, dtype=tensor_dtype, device=device)

        self.body_ids = tensor(authority.body_ids, torch.long)
        self.env_origins = env_origins
        self.component_owner_indices = tensor(
            authority.components.owner_indices, torch.long
        )
        self.component_local_centers = tensor(authority.components.local_centers_m)
        self.component_local_half_axes = tensor(
            authority.components.local_half_axes_m
        )
        self.table_lo = tensor(authority.aabb_lo)
        self.table_hi = tensor(authority.aabb_hi)
        self.racket_body_index = int(authority.racket_body_index)
        self.blade_center_offset = tensor(
            _authority.RACKET_BLADE_CENTER_OFFSET_WRIST_M
        )
        self.blade_local_half_axes = tensor(
            _authority.RACKET_BLADE_LOCAL_HALF_AXES_M
        )
        self._cuda_kernel_output = None
        self._cuda_warp_static_inputs = None
        self._cuda_warp_output = None
        self._cuda_warp_device = None
        self._cuda_warp_stream = None
        self._cuda_torch_stream_handle = None
        self._cuda_live_device = None
        self._cuda_live_pos_shape = None
        self._cuda_live_quat_shape = None
        if torch.device(device).type == "cuda":
            try:
                body_count = int(model.nbody)
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "CUDA table keepout could not bind the live model body count"
                ) from exc
            self._bind_cuda_kernel(body_count=body_count)

    def _bind_cuda_kernel(self, *, body_count: int) -> None:
        """Validate and bind every run-static CUDA input exactly once."""

        _require_warp_table_keepout_kernel()
        _validate_cuda_static_inputs(
            self.env_origins,
            self.body_ids,
            self.component_owner_indices,
            self.component_local_centers,
            self.component_local_half_axes,
            self.table_lo,
            self.table_hi,
            racket_body_index=self.racket_body_index,
            blade_center_offset=self.blade_center_offset,
            blade_local_half_axes=self.blade_local_half_axes,
            body_count=body_count,
        )
        env_count = int(self.env_origins.shape[0])
        self._cuda_live_device = self.env_origins.device
        self._cuda_live_pos_shape = (env_count, body_count, 3)
        self._cuda_live_quat_shape = (env_count, body_count, 4)
        self._cuda_kernel_output = torch.empty(
            (env_count,), dtype=torch.bool, device=self._cuda_live_device
        )
        self._cuda_warp_static_inputs = _warp_static_launch_inputs(
            self.env_origins,
            self.body_ids,
            self.component_owner_indices,
            self.component_local_centers,
            self.component_local_half_axes,
            self.table_lo,
            self.table_hi,
            racket_body_index=self.racket_body_index,
            blade_center_offset=self.blade_center_offset,
            blade_local_half_axes=self.blade_local_half_axes,
        )
        self._cuda_warp_output = _wp.from_torch(
            self._cuda_kernel_output,
            dtype=_wp.bool,
            requires_grad=False,
        )
        self._cuda_warp_device = _wp.device_from_torch(self._cuda_live_device)
        self._cuda_warp_stream = None
        self._cuda_torch_stream_handle = None

    def _current_warp_stream(self) -> object:
        torch_stream = torch.cuda.current_stream(self._cuda_live_device)
        handle = int(torch_stream.cuda_stream)
        if self._cuda_warp_stream is None or handle != self._cuda_torch_stream_handle:
            self._cuda_warp_stream = _wp.stream_from_torch(torch_stream)
            self._cuda_torch_stream_handle = handle
        return self._cuda_warp_stream

    def sample(self, data) -> torch.Tensor:
        if self._cuda_kernel_output is not None:
            body_pos_w = data.xpos
            body_quat_w = data.xquat
            _validate_cuda_live_poses(
                body_pos_w,
                body_quat_w,
                device=self._cuda_live_device,
                pos_shape=self._cuda_live_pos_shape,
                quat_shape=self._cuda_live_quat_shape,
            )
            try:
                _wp.launch(
                    _warp_table_keepout_f32,
                    dim=body_pos_w.shape[0],
                    inputs=(
                        _wp.from_torch(
                            body_pos_w, dtype=_wp.vec3, requires_grad=False
                        ),
                        _wp.from_torch(
                            body_quat_w, dtype=_wp.vec4, requires_grad=False
                        ),
                        *self._cuda_warp_static_inputs,
                    ),
                    outputs=(self._cuda_warp_output,),
                    device=self._cuda_warp_device,
                    stream=self._current_warp_stream(),
                    record_tape=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    "CUDA table keepout bound single-kernel launch failed; "
                    "eager fallback forbidden"
                ) from exc
            return self._cuda_kernel_output
        return geometric_robot_table_hit_mask(
            data.xpos[:, self.body_ids],
            data.xquat[:, self.body_ids],
            self.env_origins,
            self.component_owner_indices,
            self.component_local_centers,
            self.component_local_half_axes,
            self.table_lo,
            self.table_hi,
            racket_body_index=self.racket_body_index,
            blade_center_offset=self.blade_center_offset,
            blade_local_half_axes=self.blade_local_half_axes,
        )


__all__ = (
    "DeviceExactTableKeepout",
    "geometric_robot_table_hit_mask",
)
