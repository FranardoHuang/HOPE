"""Isaac Lab 2.1 runtime candidate for the lateral-balance perturbation source core.

This module is deliberately dependency-light: it uses the public shape of an Isaac Lab 2.1
``Articulation`` but does not import Isaac Lab at module-import time.  That lets the transaction
and lifecycle logic run in ordinary unit tests while the exact full-scene probe still executes in
the pinned Isaac Sim/Isaac Lab runtime.

The important timing fact is that Isaac Lab 2.1's
``Articulation.set_external_force_and_torque`` only updates a BODY-frame buffer.  The buffer is
submitted by ``scene.write_data_to_sim()`` before every physics substep.  A WORLD-Y command must
therefore be transformed again before *every* substep; transforming once per policy tick would
rotate the commanded force with the torso.  :class:`IsaacLateralPerturbationRuntimeHook` wraps only
the explicit probe call path and intercepts those scene writes.  When disabled, it delegates to
``env.step`` directly and does not inspect or mutate the environment.

The candidate synchronizes every CUDA commit and checks the Isaac-side command buffer after the
scene write.  Isaac Lab 2.1 exposes no getter for the wrench consumed by the PhysX solver, so this
is command-buffer/readback evidence, not solver-execution evidence.  The feature must remain
``launch_authorized=false`` until the full-scene probe, throughput/no-host-sync redesign and an
independent dynamics response check close that boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import torch
from dataclasses import dataclass
from typing import Any, Callable

from .lateral_perturbation import (
    LateralApplicationLedgerRow,
    LateralPerturbationConfig,
    LateralPerturbationStep,
    LateralPulseScheduler,
    LateralWrenchPreflightReceipt,
    dispatch_lateral_wrench_fail_closed,
)

_ISAACLAB_TAG = "v2.1.0"
_ISAACLAB_COMMIT = "21f7136325136ca3f6ca4e0a8125edffe5c24f7e"
_BACKEND_CONTRACT = {
    "schema_version": 1,
    "isaaclab_tag": _ISAACLAB_TAG,
    "isaaclab_commit": _ISAACLAB_COMMIT,
    "buffer_api": "Articulation._external_{force,torque}_b",
    "submit_api": "InteractiveScene.write_data_to_sim",
    "physx_submit_api": "ArticulationView.apply_forces_and_torques_at_position",
    "application_point": "body_center_of_mass_position_data_none",
    "full_articulation_buffer_overwrite": True,
    "solver_execution_readback_available": False,
}
_TRANSFORM_CONTRACT = {
    "schema_version": 1,
    "input": "world_wrench_at_torso_com",
    "backend": "body_local_wrench_at_torso_com",
    "quaternion": "isaaclab_wxyz_body_link_quat_w",
    "refresh": "before_every_physics_substep",
    "algorithm": "q_inverse_rotate_world_vector_v1",
}


def _canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def isaac_lateral_backend_contract() -> dict[str, object]:
    """Return a copy of the reviewed Isaac Lab 2.1 command-buffer contract."""

    return dict(_BACKEND_CONTRACT)


def isaac_lateral_backend_identity_sha256() -> str:
    return _canonical_sha256(_BACKEND_CONTRACT)


def isaac_lateral_transform_contract() -> dict[str, object]:
    """Return a copy of the WORLD-to-BODY substep transform contract."""

    return dict(_TRANSFORM_CONTRACT)


def isaac_lateral_transform_identity_sha256() -> str:
    return _canonical_sha256(_TRANSFORM_CONTRACT)


def _require_tensor(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value


def _host_all(condition: torch.Tensor, message: str) -> None:
    """Make a tensor predicate visible before/after a simulator-side command."""

    predicate = torch.all(condition)
    if not torch.equal(predicate, torch.ones_like(predicate, dtype=torch.bool)):
        raise RuntimeError(message)


def _quat_rotate_wxyz(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate BODY vectors into WORLD using scalar-first unit quaternions."""

    q_vec = quat_wxyz[..., 1:]
    uv = torch.cross(q_vec, vector, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (quat_wxyz[..., :1] * uv + uuv)


def _quat_rotate_inverse_wxyz(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate WORLD vectors into BODY using scalar-first unit quaternions."""

    q_vec = quat_wxyz[..., 1:]
    uv = torch.cross(q_vec, vector, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (-quat_wxyz[..., :1] * uv + uuv)


@dataclass(frozen=True)
class IsaacLateralSubstepReceipt:
    """Command-buffer evidence for one physics substep.

    ``solver_execution_readback_available`` is intentionally false.  The receipt proves that the
    exact BODY buffer survived through the synchronous ``scene.write_data_to_sim`` boundary; it
    does not claim that PhysX exposes the solver-consumed wrench.
    """

    policy_step_token: int
    physics_substep_index: int
    commanded_force_w: torch.Tensor
    commanded_torque_w: torch.Tensor
    written_force_b: torch.Tensor
    written_torque_b: torch.Tensor
    scene_write_completed_synchronously: bool
    buffer_readback_exact: bool
    solver_execution_readback_available: bool = False

    def clone(self) -> "IsaacLateralSubstepReceipt":
        return IsaacLateralSubstepReceipt(
            policy_step_token=self.policy_step_token,
            physics_substep_index=self.physics_substep_index,
            commanded_force_w=self.commanded_force_w.clone(),
            commanded_torque_w=self.commanded_torque_w.clone(),
            written_force_b=self.written_force_b.clone(),
            written_torque_b=self.written_torque_b.clone(),
            scene_write_completed_synchronously=(self.scene_write_completed_synchronously),
            buffer_readback_exact=self.buffer_readback_exact,
            solver_execution_readback_available=(self.solver_execution_readback_available),
        )


@dataclass(frozen=True)
class IsaacLateralPolicyStepReceipt:
    """Exact episode/window/application reconciliation for one policy step."""

    step_token: int
    episode_indices: torch.Tensor
    episode_steps: torch.Tensor
    recovery_hold_eligible: torch.Tensor
    strike_window: torch.Tensor
    safe_window_remaining_steps: torch.Tensor
    scheduler_step: LateralPerturbationStep
    application_ledger: LateralApplicationLedgerRow
    physics_substeps: tuple[IsaacLateralSubstepReceipt, ...]
    reset_after_step: torch.Tensor
    reset_scene_write_observed: bool
    reset_torso_buffer_zero_exact: bool
    async_backend_completion_synchronized: bool
    solver_execution_readback_available: bool = False

    def clone(self) -> "IsaacLateralPolicyStepReceipt":
        return IsaacLateralPolicyStepReceipt(
            step_token=self.step_token,
            episode_indices=self.episode_indices.clone(),
            episode_steps=self.episode_steps.clone(),
            recovery_hold_eligible=self.recovery_hold_eligible.clone(),
            strike_window=self.strike_window.clone(),
            safe_window_remaining_steps=self.safe_window_remaining_steps.clone(),
            scheduler_step=self.scheduler_step.clone(),
            application_ledger=self.application_ledger.clone(),
            physics_substeps=tuple(row.clone() for row in self.physics_substeps),
            reset_after_step=self.reset_after_step.clone(),
            reset_scene_write_observed=self.reset_scene_write_observed,
            reset_torso_buffer_zero_exact=self.reset_torso_buffer_zero_exact,
            async_backend_completion_synchronized=(self.async_backend_completion_synchronized),
            solver_execution_readback_available=(self.solver_execution_readback_available),
        )


@dataclass(frozen=True)
class _StagedIsaacWrench:
    source_token: object
    step_token: int
    total_mass_kg: torch.Tensor
    force_w: torch.Tensor
    torque_w: torch.Tensor
    force_b_full: torch.Tensor
    torque_b_full: torch.Tensor
    receipt: LateralWrenchPreflightReceipt


class IsaacLab21LateralWrenchAdapter:
    """Reviewed candidate adapter for one live Isaac Lab 2.1 articulation.

    The adapter owns the complete robot external-wrench buffer while attached.  It rejects a
    non-zero pre-existing buffer and also rejects ``has_external_wrench=True`` with zero bytes;
    the latter may be another owner's temporarily idle command.  Every later preflight compares
    live buffer identity and bytes to this adapter's prior readback.  This is important because
    the v2.1 setter is overwrite-based and has no composable wrench owner identity.
    """

    body_name = "torso_link"
    input_force_frame = "world"
    application_point = "center_of_mass"
    full_batch_overwrite = True
    inactive_zero_overwrite = True
    preflight_side_effect_free = True
    commit_failure_is_terminal = True
    discard_is_noexcept = True
    world_to_backend_transform_identity_sha256 = isaac_lateral_transform_identity_sha256()
    application_backend_identity_sha256 = isaac_lateral_backend_identity_sha256()

    def __init__(
        self,
        robot: object,
        *,
        synchronize: Callable[[torch.device], None] | None = None,
    ) -> None:
        self._robot = robot
        body_names = tuple(str(name) for name in getattr(robot, "body_names", ()))
        if body_names.count(self.body_name) != 1:
            raise RuntimeError(f"Isaac lateral adapter requires exactly one {self.body_name!r}, got {body_names}")
        self._body_index = body_names.index(self.body_name)
        self._num_envs = int(getattr(robot, "num_instances", -1))
        self._num_bodies = int(getattr(robot, "num_bodies", -1))
        if self._num_envs <= 0 or self._num_bodies != len(body_names):
            raise RuntimeError("Isaac articulation instance/body count is inconsistent")

        force_buffer = getattr(robot, "_external_force_b", None)
        torque_buffer = getattr(robot, "_external_torque_b", None)
        if not isinstance(force_buffer, torch.Tensor) or not isinstance(torque_buffer, torch.Tensor):
            raise RuntimeError("Isaac Lab 2.1 external-wrench buffers are unavailable; adapter cannot prove readback")
        expected = (self._num_envs, self._num_bodies, 3)
        _require_tensor("robot._external_force_b", force_buffer, shape=expected)
        _require_tensor(
            "robot._external_torque_b",
            torque_buffer,
            shape=expected,
            dtype=force_buffer.dtype,
            device=force_buffer.device,
        )
        if not torch.is_floating_point(force_buffer):
            raise RuntimeError("Isaac external-wrench buffer must use a floating dtype")
        if getattr(robot, "has_external_wrench", None) is not False:
            raise RuntimeError(
                "lateral adapter refuses an articulation with an existing external-wrench owner"
            )
        _host_all(
            force_buffer == 0.0,
            "lateral adapter refuses to steal a non-zero robot external-force buffer",
        )
        _host_all(
            torque_buffer == 0.0,
            "lateral adapter refuses to steal a non-zero robot external-torque buffer",
        )
        self._device = force_buffer.device
        self._dtype = force_buffer.dtype
        self.application_backend_token = force_buffer
        self._torque_backend_token = torque_buffer
        self._uses_default_synchronize = synchronize is None
        self._synchronize_fn = synchronize or self._default_synchronize
        self._pending: _StagedIsaacWrench | None = None
        self._committed_step_token: int | None = None
        self._commanded_force_w = torch.zeros((self._num_envs, 1, 3), dtype=self._dtype, device=self._device)
        self._commanded_torque_w = torch.zeros_like(self._commanded_force_w)
        self._last_force_b_full = torch.zeros_like(force_buffer)
        self._last_torque_b_full = torch.zeros_like(torque_buffer)
        self._dirty_unknown = False

    @staticmethod
    def _default_synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dirty_unknown(self) -> bool:
        return self._dirty_unknown

    @property
    def async_completion_synchronization_trusted(self) -> bool:
        """Whether receipts may call the enqueue boundary synchronously complete."""

        return self._device.type != "cuda" or self._uses_default_synchronize

    def _synchronize(self) -> None:
        self._synchronize_fn(self._device)

    def _assert_live_owned_state_unchanged(self) -> None:
        """Reject any wrench producer that changed the exclusively owned live buffers."""

        try:
            force = getattr(self._robot, "_external_force_b", None)
            torque = getattr(self._robot, "_external_torque_b", None)
            if force is not self.application_backend_token or torque is not self._torque_backend_token:
                raise RuntimeError("Isaac live wrench-buffer identity changed")
            if self._committed_step_token is None:
                if getattr(self._robot, "has_external_wrench", None) is not False:
                    raise RuntimeError("another external-wrench owner activated before first commit")
            elif getattr(self._robot, "has_external_wrench", None) is not True:
                raise RuntimeError("Isaac external-wrench submission flag changed after ownership")
            self._synchronize()
            _host_all(
                force == self._last_force_b_full,
                "another producer changed the owned external-force buffer between policy steps",
            )
            _host_all(
                torque == self._last_torque_b_full,
                "another producer changed the owned external-torque buffer between policy steps",
            )
        except BaseException:
            self._dirty_unknown = True
            raise

    def read_actual_total_mass_kg(self) -> torch.Tensor:
        """Read and sum the current post-randomization PhysX body masses."""

        root_view = getattr(self._robot, "root_physx_view", None)
        getter = getattr(root_view, "get_masses", None)
        if not callable(getter):
            raise RuntimeError("Isaac articulation exposes no current-mass getter")
        masses = getter()
        if not isinstance(masses, torch.Tensor):
            raise RuntimeError("Isaac current-mass getter returned no tensor")
        masses = masses.to(device=self._device, dtype=self._dtype)
        _require_tensor(
            "root_physx_view.get_masses()",
            masses,
            shape=(self._num_envs, self._num_bodies),
            dtype=self._dtype,
            device=self._device,
        )
        _host_all(
            torch.isfinite(masses) & masses.gt(0.0),
            "Isaac post-randomization body masses must be finite and positive",
        )
        total = masses.sum(dim=-1)
        _host_all(
            torch.isfinite(total) & total.gt(0.0),
            "Isaac post-randomization total mass must be finite and positive",
        )
        return total

    def _body_quat_w(self) -> torch.Tensor:
        data = getattr(self._robot, "data", None)
        body_quat_w = getattr(data, "body_quat_w", None)
        if not isinstance(body_quat_w, torch.Tensor):
            raise RuntimeError("Isaac articulation exposes no body_quat_w tensor")
        expected = (self._num_envs, self._num_bodies, 4)
        _require_tensor(
            "robot.data.body_quat_w",
            body_quat_w,
            shape=expected,
            dtype=self._dtype,
            device=self._device,
        )
        quat = body_quat_w[:, self._body_index, :]
        _host_all(torch.isfinite(quat), "torso quaternion must be finite")
        norm_sq = torch.sum(quat * quat, dim=-1)
        _host_all(
            torch.abs(norm_sq - 1.0).le(2.0e-4),
            "torso quaternion must be unit length before WORLD-to-BODY transform",
        )
        return quat

    def _stage_body_buffers(self, force_w: torch.Tensor, torque_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quat = self._body_quat_w()
        force_b = _quat_rotate_inverse_wxyz(quat, force_w[:, 0, :])
        torque_b = _quat_rotate_inverse_wxyz(quat, torque_w[:, 0, :])
        full_force = torch.zeros(
            (self._num_envs, self._num_bodies, 3),
            dtype=self._dtype,
            device=self._device,
        )
        full_torque = torch.zeros_like(full_force)
        full_force[:, self._body_index, :] = force_b
        full_torque[:, self._body_index, :] = torque_b
        return full_force, full_torque

    def _copy_and_readback(self, force_b_full: torch.Tensor, torque_b_full: torch.Tensor) -> None:
        if self._dirty_unknown:
            raise RuntimeError("Isaac lateral backend is DIRTY/UNKNOWN")
        live_force = getattr(self._robot, "_external_force_b", None)
        live_torque = getattr(self._robot, "_external_torque_b", None)
        if live_force is not self.application_backend_token:
            self._dirty_unknown = True
            raise RuntimeError("Isaac live force-buffer identity changed")
        if live_torque is not self._torque_backend_token:
            self._dirty_unknown = True
            raise RuntimeError("Isaac live torque-buffer identity changed")
        try:
            live_force.copy_(force_b_full)
            live_torque.copy_(torque_b_full)
            # Keep the v2.1 submit path active even for an all-zero clear.  A zero buffer must be
            # explicitly submitted on pulse completion/interruption, not optimized away.
            setattr(self._robot, "has_external_wrench", True)
            if getattr(self._robot, "has_external_wrench", None) is not True:
                raise RuntimeError("Isaac articulation did not enable external-wrench submission")
            self._synchronize()
            _host_all(live_force == force_b_full, "Isaac force-buffer readback mismatch")
            _host_all(live_torque == torque_b_full, "Isaac torque-buffer readback mismatch")
        except BaseException:
            self._dirty_unknown = True
            raise
        self._last_force_b_full.copy_(force_b_full)
        self._last_torque_b_full.copy_(torque_b_full)

    def preflight_world_wrench_at_body_com(
        self,
        *,
        step_token: int,
        total_mass_kg: torch.Tensor,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
        preflight_token: object,
    ) -> LateralWrenchPreflightReceipt:
        if self._dirty_unknown:
            raise RuntimeError("Isaac lateral backend is DIRTY/UNKNOWN")
        if self._pending is not None:
            raise RuntimeError("Isaac lateral adapter already has a pending preflight")
        if type(step_token) is not int or step_token < 0:
            raise ValueError("step_token must be a non-negative plain int")
        self._assert_live_owned_state_unchanged()
        _require_tensor(
            "total_mass_kg",
            total_mass_kg,
            shape=(self._num_envs,),
            dtype=self._dtype,
            device=self._device,
        )
        _require_tensor(
            "force_w",
            force_w,
            shape=(self._num_envs, 1, 3),
            dtype=self._dtype,
            device=self._device,
        )
        _require_tensor(
            "torque_w",
            torque_w,
            shape=(self._num_envs, 1, 3),
            dtype=self._dtype,
            device=self._device,
        )
        _host_all(torch.isfinite(force_w), "commanded WORLD force must be finite")
        _host_all(torch.isfinite(torque_w), "commanded WORLD torque must be finite")
        _host_all(force_w[:, :, 0] == 0.0, "WORLD-X force is forbidden")
        _host_all(force_w[:, :, 2] == 0.0, "WORLD-Z force is forbidden")
        _host_all(torque_w == 0.0, "explicit torque is forbidden")
        actual_mass = self.read_actual_total_mass_kg()
        _host_all(
            actual_mass == total_mass_kg,
            "dispatch total mass does not match current post-randomization PhysX mass",
        )

        force_b_full, torque_b_full = self._stage_body_buffers(force_w, torque_w)
        applied_mask = force_w[:, 0, 1].ne(0.0)
        receipt = LateralWrenchPreflightReceipt(
            step_token=step_token,
            body_name=self.body_name,
            input_force_frame=self.input_force_frame,
            application_point=self.application_point,
            full_batch_overwrite=True,
            inactive_zero_overwrite=True,
            zero_torque=True,
            world_to_backend_transform_identity_sha256=(self.world_to_backend_transform_identity_sha256),
            application_backend_identity_sha256=(self.application_backend_identity_sha256),
            actual_total_mass_kg=actual_mass.clone(),
            commanded_force_w=force_w.clone(),
            commanded_torque_w=torque_w.clone(),
            applied_force_mask=applied_mask.clone(),
            preflight_token=preflight_token,
        )
        self._pending = _StagedIsaacWrench(
            source_token=preflight_token,
            step_token=step_token,
            total_mass_kg=actual_mass.clone(),
            force_w=force_w.clone(),
            torque_w=torque_w.clone(),
            force_b_full=force_b_full,
            torque_b_full=torque_b_full,
            receipt=receipt,
        )
        return receipt

    def commit_preflighted_world_wrench_at_body_com(self, *, preflight_token: object) -> None:
        staged = self._pending
        if staged is None or staged.source_token is not preflight_token:
            raise RuntimeError("Isaac lateral commit received a stale/foreign preflight token")
        try:
            self._copy_and_readback(staged.force_b_full, staged.torque_b_full)
        except BaseException:
            # dispatch_lateral_wrench_fail_closed also marks scheduler state dirty.  Keep the
            # adapter independently fail-closed if somebody catches the outer exception.
            self._dirty_unknown = True
            self._pending = None
            raise
        self._commanded_force_w.copy_(staged.force_w)
        self._commanded_torque_w.copy_(staged.torque_w)
        self._committed_step_token = staged.step_token
        self._pending = None
        return None

    def discard_preflighted_world_wrench_at_body_com(self, *, preflight_token: object) -> None:
        # No live buffer was touched by preflight.  Discard is deliberately idempotent/no-throw.
        if self._pending is not None and self._pending.source_token is preflight_token:
            self._pending = None
        return None

    def refresh_before_sim_substep(
        self, *, policy_step_token: int, physics_substep_index: int
    ) -> IsaacLateralSubstepReceipt:
        """Refresh BODY buffer from the frozen WORLD command for one physics substep."""

        if self._committed_step_token != policy_step_token:
            raise RuntimeError("substep refresh has no matching committed policy-step command")
        if type(physics_substep_index) is not int or physics_substep_index < 0:
            raise ValueError("physics_substep_index must be a non-negative plain int")
        force_b_full, torque_b_full = self._stage_body_buffers(self._commanded_force_w, self._commanded_torque_w)
        self._copy_and_readback(force_b_full, torque_b_full)
        return IsaacLateralSubstepReceipt(
            policy_step_token=policy_step_token,
            physics_substep_index=physics_substep_index,
            commanded_force_w=self._commanded_force_w.clone(),
            commanded_torque_w=self._commanded_torque_w.clone(),
            written_force_b=force_b_full[:, self._body_index : self._body_index + 1].clone(),
            written_torque_b=torque_b_full[:, self._body_index : self._body_index + 1].clone(),
            scene_write_completed_synchronously=False,
            buffer_readback_exact=True,
        )

    def confirm_scene_write_completed(self, receipt: IsaacLateralSubstepReceipt) -> IsaacLateralSubstepReceipt:
        """Synchronize the PhysX enqueue boundary and re-read the exact command buffer."""

        if receipt.policy_step_token != self._committed_step_token:
            raise RuntimeError("scene-write receipt belongs to another policy step")
        try:
            self._synchronize()
            live_force = getattr(self._robot, "_external_force_b")
            live_torque = getattr(self._robot, "_external_torque_b")
            if live_force is not self.application_backend_token or live_torque is not self._torque_backend_token:
                raise RuntimeError("Isaac live wrench-buffer identity changed at scene-write readback")
            expected_force = self._last_force_b_full
            expected_torque = self._last_torque_b_full
            _host_all(
                live_force == expected_force,
                "scene write changed the staged full-articulation force buffer",
            )
            _host_all(
                live_torque == expected_torque,
                "scene write changed the staged full-articulation torque buffer",
            )
        except BaseException:
            self._dirty_unknown = True
            raise
        return IsaacLateralSubstepReceipt(
            policy_step_token=receipt.policy_step_token,
            physics_substep_index=receipt.physics_substep_index,
            commanded_force_w=receipt.commanded_force_w.clone(),
            commanded_torque_w=receipt.commanded_torque_w.clone(),
            written_force_b=receipt.written_force_b.clone(),
            written_torque_b=receipt.written_torque_b.clone(),
            scene_write_completed_synchronously=(self.async_completion_synchronization_trusted),
            buffer_readback_exact=True,
            solver_execution_readback_available=False,
        )

    def clear_full_batch_before_reset_scene_write(self) -> None:
        """Write all-environment zero before a reset-only scene submission.

        ``ManagerBasedRLEnv`` writes the entire scene once more after resetting any subset of
        environments.  Leaving non-reset rows non-zero at that boundary could enqueue one extra
        wrench outside the configured decimation loop.  The reset write must therefore be an
        all-batch zero write; the scheduler re-applies any continuing pulse on the next policy
        step.
        """

        zeros = torch.zeros(
            (self._num_envs, self._num_bodies, 3),
            dtype=self._dtype,
            device=self._device,
        )
        self._copy_and_readback(zeros, zeros)

    def assert_full_batch_zero(self) -> None:
        """Synchronously prove that no robot external wrench survives in any environment."""

        try:
            self._synchronize()
            force = getattr(self._robot, "_external_force_b")
            torque = getattr(self._robot, "_external_torque_b")
            if force is not self.application_backend_token or torque is not self._torque_backend_token:
                raise RuntimeError("Isaac live wrench-buffer identity changed at reset readback")
            _host_all(force == 0.0, "reset scene write left an external force alive")
            _host_all(torque == 0.0, "reset scene write left an external torque alive")
        except BaseException:
            self._dirty_unknown = True
            raise

    def assert_reset_envs_zero(self, reset_mask: torch.Tensor) -> None:
        """Prove that Isaac's articulation reset cleared every reset row."""

        _require_tensor(
            "reset_mask",
            reset_mask,
            shape=(self._num_envs,),
            dtype=torch.bool,
            device=self._device,
        )
        self._synchronize()
        force = getattr(self._robot, "_external_force_b")[:, self._body_index : self._body_index + 1]
        torque = getattr(self._robot, "_external_torque_b")[:, self._body_index : self._body_index + 1]
        if torch.any(reset_mask):
            _host_all(force[reset_mask] == 0.0, "reset left a torso force alive")
            _host_all(torque[reset_mask] == 0.0, "reset left a torso torque alive")


class IsaacLateralPerturbationRuntimeHook:
    """Probe-only policy/substep lifecycle wrapper around a ManagerBasedRLEnv.

    The hook is not installed into existing task registrations.  A caller must explicitly create
    it and call :meth:`step`.  ``enabled=False`` is a direct delegation path and intentionally does
    not read any environment field.
    """

    def __init__(
        self,
        env: object,
        cfg: LateralPerturbationConfig,
        *,
        enabled: bool = False,
        synchronize: Callable[[torch.device], None] | None = None,
    ) -> None:
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        self.enabled = enabled
        self._env = env
        self._cfg = cfg
        self._dirty_unknown = False
        self._receipts: list[IsaacLateralPolicyStepReceipt] = []
        if not enabled:
            self._adapter = None
            self._scheduler = None
            return

        num_envs = int(getattr(env, "num_envs", -1))
        device = torch.device(getattr(env, "device", "cpu"))
        if num_envs <= 0:
            raise RuntimeError("runtime hook requires a vectorized environment")
        step_dt = float(getattr(env, "step_dt", float("nan")))
        if not math.isfinite(step_dt) or not math.isclose(step_dt, cfg.policy_dt_s, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("environment step_dt does not match perturbation policy_dt_s")
        robot = env.scene["robot"]
        self._adapter = IsaacLab21LateralWrenchAdapter(robot, synchronize=synchronize)
        if self._adapter.num_envs != num_envs or self._adapter.device != device:
            raise RuntimeError("environment and articulation batch/device disagree")
        self._scheduler = LateralPulseScheduler(
            num_envs,
            cfg,
            device=device,
            require_application_ack=True,
        )
        self._num_envs = num_envs
        self._device = device
        self._episode_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        env_episode_steps = getattr(env, "episode_length_buf", None)
        _require_tensor(
            "env.episode_length_buf",
            env_episode_steps,
            shape=(num_envs,),
            dtype=torch.long,
            device=device,
        )
        _host_all(
            env_episode_steps == 0,
            "lateral runtime probe must attach immediately after a full environment reset",
        )
        self._step_token = 0

    @property
    def dirty_unknown(self) -> bool:
        return self._dirty_unknown or bool(self._adapter is not None and self._adapter.dirty_unknown)

    def _command_terms(self) -> tuple[object, object]:
        manager = getattr(self._env, "command_manager", None)
        get_term = getattr(manager, "get_term", None)
        if not callable(get_term):
            raise RuntimeError("environment command manager exposes no get_term")
        motion = get_term("motion")
        target = get_term("racket_target")
        return motion, target

    def _derive_windows(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion, target = self._command_terms()
        if bool(getattr(motion, "event_timing_enabled", False)):
            raise RuntimeError("lateral runtime v1 does not infer safe windows for event-driven T1")
        hold = getattr(motion, "in_hold")
        if callable(hold):
            hold = hold()
        hold = _require_tensor(
            "motion.in_hold",
            hold,
            shape=(self._num_envs,),
            dtype=torch.bool,
            device=self._device,
        ).clone()
        strike = _require_tensor(
            "racket_target.strike_window",
            getattr(target, "strike_window", None),
            shape=(self._num_envs,),
            dtype=torch.bool,
            device=self._device,
        ).clone()
        pre_strike = _require_tensor(
            "racket_target.pre_strike",
            getattr(target, "pre_strike", None),
            shape=(self._num_envs,),
            dtype=torch.bool,
            device=self._device,
        ).clone()
        post_strike = ~pre_strike & ~strike
        eligible = (hold | post_strike) & ~strike

        hold_counter = _require_tensor(
            "motion.hold_counter",
            getattr(motion, "hold_counter", None),
            shape=(self._num_envs,),
            dtype=torch.long,
            device=self._device,
        )
        hold_remaining = torch.maximum(hold_counter, hold.to(dtype=torch.long))
        time_steps = _require_tensor(
            "motion.time_steps",
            getattr(motion, "time_steps", None),
            shape=(self._num_envs,),
            dtype=torch.long,
            device=self._device,
        )
        motion_asset = getattr(motion, "motion", None)
        if bool(getattr(motion, "_multiseg", False)):
            clip_id = _require_tensor(
                "motion.clip_id",
                getattr(motion, "clip_id", None),
                shape=(self._num_envs,),
                dtype=torch.long,
                device=self._device,
            )
            seg_start = getattr(motion_asset, "seg_start", None)
            seg_len = getattr(motion_asset, "seg_len", None)
            if not isinstance(seg_start, torch.Tensor) or not isinstance(seg_len, torch.Tensor):
                raise RuntimeError("multi-clip motion exposes no segment bounds")
            post_remaining = seg_start[clip_id] + seg_len[clip_id] - time_steps
        else:
            total = int(getattr(motion_asset, "time_step_total", -1))
            if total <= 0:
                raise RuntimeError("single-clip motion exposes no positive time_step_total")
            post_remaining = total - time_steps
        post_remaining = post_remaining.clamp_min(0).to(dtype=torch.long)
        safe_remaining = torch.where(
            hold,
            hold_remaining,
            torch.where(post_strike, post_remaining, torch.zeros_like(post_remaining)),
        )
        return eligible, strike, safe_remaining

    def _prepare_policy_step(
        self,
    ) -> tuple[LateralPerturbationStep, LateralApplicationLedgerRow, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._scheduler is not None and self._adapter is not None
        env_steps = getattr(self._env, "episode_length_buf")
        _host_all(
            env_steps == self._episode_steps,
            "environment episode clock diverged from lateral runtime ledger",
        )
        eligible, strike, safe = self._derive_windows()
        result = self._scheduler.step(
            step_token=self._step_token,
            episode_indices=self._episode_indices,
            episode_steps=self._episode_steps,
            recovery_hold_eligible=eligible,
            strike_window=strike,
            safe_window_remaining_steps=safe,
        )
        total_mass = self._adapter.read_actual_total_mass_kg()
        application = dispatch_lateral_wrench_fail_closed(
            scheduler=self._scheduler,
            result=result,
            total_mass_kg=total_mass,
            adapter=self._adapter,
        )
        return result, application, eligible, strike, safe

    def step(self, action: Any) -> Any:
        """Execute exactly one environment step through the explicit probe hook."""

        if not self.enabled:
            return self._env.step(action)
        if self.dirty_unknown:
            raise RuntimeError("lateral runtime hook is DIRTY/UNKNOWN")
        assert self._adapter is not None and self._scheduler is not None
        result, application, eligible, strike, safe = self._prepare_policy_step()
        scene = getattr(self._env, "scene")
        original_write = getattr(scene, "write_data_to_sim", None)
        if not callable(original_write):
            raise RuntimeError("environment scene exposes no write_data_to_sim")
        decimation = int(getattr(getattr(self._env, "cfg", None), "decimation", -1))
        if decimation <= 0:
            raise RuntimeError("environment cfg exposes no positive decimation")

        substeps: list[IsaacLateralSubstepReceipt] = []
        scene_write_count = 0
        reset_scene_write_observed = False
        reset_zero_exact = False

        def intercepted_scene_write(*args: object, **kwargs: object) -> object:
            nonlocal scene_write_count, reset_scene_write_observed, reset_zero_exact
            if scene_write_count < decimation:
                row = self._adapter.refresh_before_sim_substep(
                    policy_step_token=self._step_token,
                    physics_substep_index=scene_write_count,
                )
                output = original_write(*args, **kwargs)
                substeps.append(self._adapter.confirm_scene_write_completed(row))
            else:
                # ManagerBasedRLEnv issues one extra scene write after _reset_idx when any
                # sub-environment terminates.  Clear *all* rows before that write: otherwise a
                # continuing non-reset pulse could be submitted one extra time outside the
                # configured decimation loop.  It is re-applied by the next policy step.
                reset_mask = getattr(self._env, "reset_buf", None)
                _require_tensor(
                    "env.reset_buf",
                    reset_mask,
                    shape=(self._num_envs,),
                    dtype=torch.bool,
                    device=self._device,
                )
                self._adapter.assert_reset_envs_zero(reset_mask)
                self._adapter.clear_full_batch_before_reset_scene_write()
                self._adapter.assert_full_batch_zero()
                output = original_write(*args, **kwargs)
                self._adapter.assert_full_batch_zero()
                reset_scene_write_observed = True
                reset_zero_exact = True
            scene_write_count += 1
            return output

        # Instance-local interception keeps the default task class and every disabled run
        # untouched.  The full-scene probe verifies that the pinned v2.1.0 step calls exactly
        # ``decimation`` physics writes plus at most one reset write.
        had_instance_override = hasattr(scene, "__dict__") and ("write_data_to_sim" in scene.__dict__)
        old_instance_override = scene.__dict__.get("write_data_to_sim") if hasattr(scene, "__dict__") else None
        try:
            setattr(scene, "write_data_to_sim", intercepted_scene_write)
            output = self._env.step(action)
        except BaseException:
            self._dirty_unknown = True
            raise
        finally:
            if had_instance_override:
                setattr(scene, "write_data_to_sim", old_instance_override)
            else:
                try:
                    delattr(scene, "write_data_to_sim")
                except AttributeError:
                    pass

        if scene_write_count < decimation or len(substeps) != decimation:
            self._dirty_unknown = True
            raise RuntimeError("pinned ManagerBasedRLEnv did not expose every expected physics scene write")
        if scene_write_count > decimation + 1:
            self._dirty_unknown = True
            raise RuntimeError("unexpected extra scene writes in one policy step")
        if not isinstance(output, tuple) or len(output) < 4:
            self._dirty_unknown = True
            raise RuntimeError("environment step returned no terminated/truncated tensors")
        terminated = _require_tensor(
            "terminated",
            output[2],
            shape=(self._num_envs,),
            dtype=torch.bool,
            device=self._device,
        )
        truncated = _require_tensor(
            "truncated",
            output[3],
            shape=(self._num_envs,),
            dtype=torch.bool,
            device=self._device,
        )
        reset = terminated | truncated
        reset_any = bool(torch.any(reset).detach().cpu())
        if reset_any != reset_scene_write_observed:
            self._dirty_unknown = True
            raise RuntimeError("reset mask and reset scene-write evidence disagree")
        if reset_any and not reset_zero_exact:
            self._dirty_unknown = True
            raise RuntimeError("reset scene write lacks exact zero-buffer evidence")

        expected_next_steps = self._episode_steps + 1
        expected_next_steps = torch.where(reset, torch.zeros_like(expected_next_steps), expected_next_steps)
        _host_all(
            getattr(self._env, "episode_length_buf") == expected_next_steps,
            "post-step environment episode clock does not reconcile",
        )
        receipt = IsaacLateralPolicyStepReceipt(
            step_token=self._step_token,
            episode_indices=self._episode_indices.clone(),
            episode_steps=self._episode_steps.clone(),
            recovery_hold_eligible=eligible.clone(),
            strike_window=strike.clone(),
            safe_window_remaining_steps=safe.clone(),
            scheduler_step=result.clone(),
            application_ledger=application.clone(),
            physics_substeps=tuple(row.clone() for row in substeps),
            reset_after_step=reset.clone(),
            reset_scene_write_observed=reset_scene_write_observed,
            reset_torso_buffer_zero_exact=(not reset_any or reset_zero_exact),
            async_backend_completion_synchronized=(self._adapter.async_completion_synchronization_trusted),
            solver_execution_readback_available=False,
        )
        self._receipts.append(receipt)
        self._episode_steps.copy_(expected_next_steps)
        self._episode_indices.add_(reset.to(dtype=torch.long))
        self._step_token += 1
        return output

    def receipts(self) -> tuple[IsaacLateralPolicyStepReceipt, ...]:
        return tuple(row.clone() for row in self._receipts)

    def consume_counters(self) -> dict[str, torch.Tensor]:
        if self._scheduler is None:
            return {}
        return self._scheduler.consume_counters()


__all__ = [
    "IsaacLab21LateralWrenchAdapter",
    "IsaacLateralPerturbationRuntimeHook",
    "IsaacLateralPolicyStepReceipt",
    "IsaacLateralSubstepReceipt",
    "isaac_lateral_backend_contract",
    "isaac_lateral_backend_identity_sha256",
    "isaac_lateral_transform_contract",
    "isaac_lateral_transform_identity_sha256",
]
