"""Device-resident port of the exact Isaac robot/table pose guard.

Construction binds the tracked plant, 62 collision-component OBBs, and canonical
five-box table assembly. Sampling is fixed-shape Torch SAT with no host readback.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType

import torch


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
    """Return the exact Isaac-equivalent keepout bit for every environment."""

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

    def sample(self, data) -> torch.Tensor:
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
