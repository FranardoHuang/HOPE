#!/usr/bin/env python3
"""Strict full-scene probe for the Isaac lateral-wrench runtime candidate.

This is a simulator-only probe, not a trainer.  It attaches the explicit runtime hook to an
existing HOPE tracking task, runs zero actions, and writes one no-clobber JSON receipt.  Even a
successful run remains blocked for training because Isaac Lab 2.1 exposes command-buffer readback
but no getter for the wrench consumed by the PhysX solver.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import isaaclab
import isaaclab.app as isaaclab_app_module
from isaaclab.app import AppLauncher

_CONFIRM = "SIM_ONLY_PROBE_ONE_LATERAL_WRENCH_RUNTIME"
_ISAACLAB_COMMIT = "21f7136325136ca3f6ca4e0a8125edffe5c24f7e"


def _git(args: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _verify_clean_exact_checkout(root: Path, expected_commit: str, label: str) -> str:
    if not root.is_dir():
        raise RuntimeError(f"{label} root is absent: {root}")
    top_level = Path(_git(["rev-parse", "--show-toplevel"], cwd=root)).resolve()
    if top_level != root.resolve():
        raise RuntimeError(f"{label} root must be the exact Git top-level: {top_level}")
    actual = _git(["rev-parse", "HEAD"], cwd=root)
    if actual != expected_commit:
        raise RuntimeError(f"{label} commit mismatch: expected {expected_commit}, got {actual}")
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"], cwd=root):
        raise RuntimeError(f"{label} checkout must be clean")
    return actual


def _require_module_under(module: object, expected_root: Path, label: str) -> str:
    """Bind an imported module to the reviewed checkout instead of an ambient install."""

    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(f"{label} module exposes no __file__")
    resolved = Path(module_file).resolve(strict=True)
    try:
        resolved.relative_to(expected_root.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"{label} module is not imported from the reviewed checkout: {resolved}") from exc
    return str(resolved)


def _require_regular_file_without_symlink(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{label} path must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} path contains a symlink: {current}")
    if not path.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    return path.resolve(strict=True)


def _validate_output_path(path: Path, forbidden_roots: tuple[Path, ...]) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"output path must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parent.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"output parent contains a symlink: {current}")
    if not path.parent.is_dir():
        raise RuntimeError(f"output parent must already exist: {path.parent}")
    if os.path.lexists(path):
        raise RuntimeError(f"refusing to clobber output: {path}")
    resolved_parent = path.parent.resolve(strict=True)
    resolved = resolved_parent / path.name
    for forbidden in forbidden_roots:
        try:
            resolved.relative_to(forbidden.resolve(strict=True))
        except ValueError:
            continue
        raise RuntimeError(f"output must be outside reviewed checkout: {forbidden}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_receipt_digest(digest: Any, value: Any) -> None:
    """Hash the complete typed receipt tree without lossy JSON float conversion."""

    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        header = {
            "type": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\0")
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        digest.update(f"dataclass:{type(value).__module__}.{type(value).__qualname__}\0".encode("utf-8"))
        for field in dataclasses.fields(value):
            digest.update(f"field:{field.name}\0".encode("utf-8"))
            _update_receipt_digest(digest, getattr(value, field.name))
        return
    if isinstance(value, (tuple, list)):
        digest.update(f"sequence:{type(value).__name__}:{len(value)}\0".encode("ascii"))
        for item in value:
            _update_receipt_digest(digest, item)
        return
    if type(value) in (bool, int, float, str) or value is None:
        digest.update(f"scalar:{type(value).__name__}:".encode("ascii"))
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        return
    raise RuntimeError(f"unsupported receipt field type for exact digest: {type(value)!r}")


def _event_term_manifest_and_reject_interval(event_manager: object) -> list[dict[str, object]]:
    """Bind every active EventManager term and reject mid-episode competing writers."""

    active = getattr(event_manager, "active_terms", None)
    get_term_cfg = getattr(event_manager, "get_term_cfg", None)
    if not isinstance(active, dict) or not callable(get_term_cfg):
        raise RuntimeError("EventManager exposes no auditable active-term contract")
    manifest: list[dict[str, object]] = []
    for mode in sorted(active):
        names = active[mode]
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise RuntimeError("EventManager active-term names have an unexpected shape")
        for name in names:
            cfg = get_term_cfg(name)
            func = getattr(cfg, "func", None)
            if func is None:
                raise RuntimeError(f"EventManager term {name!r} exposes no callable")
            identity_source = func if hasattr(func, "__qualname__") else type(func)
            identity = (
                f"{getattr(identity_source, '__module__', '')}."
                f"{getattr(identity_source, '__qualname__', type(func).__qualname__)}"
            )
            params = getattr(cfg, "params", None)
            if not isinstance(params, dict):
                raise RuntimeError(f"EventManager term {name!r} exposes no parameter mapping")
            manifest.append(
                {
                    "mode": mode,
                    "name": name,
                    "function_identity": identity,
                    "parameter_keys": sorted(str(key) for key in params),
                }
            )
    interval_names = [row["name"] for row in manifest if row["mode"] == "interval"]
    if interval_names:
        raise RuntimeError(
            "runtime probe refuses all interval EventManager terms; disable and bind them explicitly: "
            + ", ".join(str(name) for name in interval_names)
        )
    return manifest


parser = argparse.ArgumentParser(description="Run one strict, no-training Isaac lateral-wrench full-scene probe.")
parser.add_argument(
    "--task",
    default="HOPE-PingPong-VirtualBall-AgibotA3-v0",
    help="Existing HOPE task with motion and racket_target command terms.",
)
parser.add_argument(
    "--motion-file",
    action="append",
    required=True,
    help="Exact local schema motion NPZ; repeat for a multi-clip task.",
)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--cell", choices=("L0", "L1"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--expected-source-commit", required=True)
parser.add_argument("--isaaclab-root", type=Path, required=True)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--confirm", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.execute or args.confirm != _CONFIRM:
    raise SystemExit("refusing simulator probe without --execute --confirm " + _CONFIRM)
if args.num_envs <= 0 or args.steps <= 0:
    raise SystemExit("--num-envs and --steps must be positive")
source_root = args.source_root.resolve()
isaaclab_root = args.isaaclab_root.resolve()
source_commit = _verify_clean_exact_checkout(source_root, args.expected_source_commit, "source")
isaaclab_commit = _verify_clean_exact_checkout(isaaclab_root, _ISAACLAB_COMMIT, "IsaacLab")
output_path = _validate_output_path(args.output, (source_root, isaaclab_root))
isaaclab_module_path = _require_module_under(
    isaaclab,
    isaaclab_root / "source" / "isaaclab" / "isaaclab",
    "isaaclab",
)
isaaclab_app_module_path = _require_module_under(
    isaaclab_app_module,
    isaaclab_root / "source" / "isaaclab" / "isaaclab",
    "isaaclab.app",
)
motion_paths = tuple(_require_regular_file_without_symlink(value, "motion") for value in args.motion_file)
motion_sha256 = tuple(_sha256(path) for path in motion_paths)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _to_scalar(value: Any) -> int | float:
    import torch

    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise RuntimeError("counter must be a scalar tensor")
    raw = value.detach().cpu().item()
    return int(raw) if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) else float(raw)


def main() -> None:
    import gymnasium as gym
    import torch

    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg

    import whole_body_tracking
    import whole_body_tracking.tasks  # noqa: F401 -- register tasks after SimulationApp
    from whole_body_tracking.tasks.tracking.mdp.isaac_lateral_perturbation import (
        IsaacLateralPerturbationRuntimeHook,
        isaac_lateral_backend_contract,
        isaac_lateral_backend_identity_sha256,
        isaac_lateral_transform_contract,
        isaac_lateral_transform_identity_sha256,
    )
    from whole_body_tracking.tasks.tracking.mdp.lateral_perturbation import LateralPerturbationConfig

    source_module_path = _require_module_under(
        whole_body_tracking,
        args.source_root.resolve()
        / "hope_training"
        / "whole_body_tracking"
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking",
        "whole_body_tracking",
    )
    isaaclab_tasks_module_path = _require_module_under(
        isaaclab_tasks,
        args.isaaclab_root.resolve() / "source" / "isaaclab_tasks" / "isaaclab_tasks",
        "isaaclab_tasks",
    )

    env_cfg = parse_env_cfg(args.task, device=str(args.device), num_envs=int(args.num_envs))
    env_cfg.commands.motion.motion_file = (
        str(motion_paths[0]) if len(motion_paths) == 1 else [str(path) for path in motion_paths]
    )
    # The probe isolates this force path from the legacy root-velocity interval event.
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if bool(getattr(env_cfg.commands.motion, "stagger_initial_clock", False)):
        raise RuntimeError("runtime probe requires stagger_initial_clock=false")

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped
    try:
        event_term_manifest = _event_term_manifest_and_reject_interval(base_env.event_manager)
        reset_output = env.reset()
        if not isinstance(reset_output, tuple) or len(reset_output) != 2:
            raise RuntimeError("Gym environment did not return an explicit reset observation/info pair")
        magnitude = (0.0, 0.0) if args.cell == "L0" else (0.04, 0.08)
        cfg = LateralPerturbationConfig(
            policy_dt_s=float(base_env.step_dt),
            opportunity_interval_steps=25,
            pulse_duration_steps=5,
            selection_probability=0.5,
            normalized_impulse_min_mps=magnitude[0],
            normalized_impulse_max_mps=magnitude[1],
            seed=20260715,
        )
        hook = IsaacLateralPerturbationRuntimeHook(base_env, cfg, enabled=True)
        action_dim = int(base_env.action_manager.total_action_dim)
        action = torch.zeros(
            (int(base_env.num_envs), action_dim),
            dtype=torch.float32,
            device=base_env.device,
        )
        with torch.inference_mode():
            for _ in range(int(args.steps)):
                hook.step(action)

        receipts = hook.receipts()
        counters = {name: _to_scalar(value) for name, value in hook.consume_counters().items()}
        all_substeps = [substep for row in receipts for substep in row.physics_substeps]
        expected_substeps = int(args.steps) * int(base_env.cfg.decimation)
        if len(receipts) != int(args.steps) or len(all_substeps) != expected_substeps:
            raise RuntimeError("runtime receipt count does not match requested steps/decimation")
        if not all(
            row.async_backend_completion_synchronized
            and not row.solver_execution_readback_available
            and (not row.reset_scene_write_observed or row.reset_torso_buffer_zero_exact)
            for row in receipts
        ):
            raise RuntimeError("policy-step receipt honesty/zero-clear checks failed")
        if not all(
            substep.scene_write_completed_synchronously
            and substep.buffer_readback_exact
            and not substep.solver_execution_readback_available
            for substep in all_substeps
        ):
            raise RuntimeError("substep command-buffer receipt checks failed")
        applied_pulse_count = int(counters["lateral_perturbation_applied_pulse_count"])
        applied_force_steps = int(counters["lateral_perturbation_applied_force_env_step_count"])
        eligible_count = int(counters["lateral_perturbation_eligible_opportunity_count"])
        selected_count = int(counters["lateral_perturbation_selected_start_count"])
        if eligible_count <= 0 or selected_count <= 0:
            raise RuntimeError("full-scene probe observed no eligible selected opportunity")
        if args.cell == "L0":
            if applied_pulse_count != 0 or applied_force_steps != 0:
                raise RuntimeError("L0 command-buffer probe emitted a non-zero application")
            if any(torch.any(row.commanded_force_w != 0.0) for row in all_substeps):
                raise RuntimeError("L0 substep receipt contains a non-zero WORLD command")
        else:
            if applied_pulse_count <= 0 or applied_force_steps <= 0:
                raise RuntimeError("L1 full-scene probe observed no non-zero application")
            if not any(torch.any(row.commanded_force_w != 0.0) for row in all_substeps):
                raise RuntimeError("L1 substep receipts contain no non-zero WORLD command")
        receipt_digest = hashlib.sha256()
        _update_receipt_digest(receipt_digest, receipts)
        reset_env_steps = int(sum(int(row.reset_after_step.sum().detach().cpu()) for row in receipts))
        strike_window_env_steps = int(sum(int(row.strike_window.sum().detach().cpu()) for row in receipts))
        eligible_env_steps = int(sum(int(row.recovery_hold_eligible.sum().detach().cpu()) for row in receipts))
        reset_scene_write_count = sum(int(row.reset_scene_write_observed) for row in receipts)
        strike_interrupt_count = int(counters["lateral_perturbation_interrupted_for_strike_count"])
        window_interrupt_count = int(counters["lateral_perturbation_interrupted_for_window_count"])
        strike_interrupt_zero_rows = 0
        nonzero_strike_interrupt_zero_rows = 0
        window_interrupt_zero_rows = 0
        for row in receipts:
            for mask, label in (
                (row.scheduler_step.interrupted_for_strike_mask, "strike"),
                (row.scheduler_step.interrupted_for_window_mask, "window"),
            ):
                count = int(mask.sum().detach().cpu())
                if count == 0:
                    continue
                if torch.any(row.scheduler_step.active_force_mask[mask]):
                    raise RuntimeError(f"{label} interruption left scheduler force active")
                if torch.any(row.application_ledger.applied_force_mask[mask]):
                    raise RuntimeError(f"{label} interruption left application force active")
                for substep in row.physics_substeps:
                    if torch.any(substep.commanded_force_w[mask] != 0.0) or torch.any(
                        substep.written_force_b[mask] != 0.0
                    ):
                        raise RuntimeError(f"{label} interruption did not write same-step zero force")
                    if torch.any(substep.commanded_torque_w[mask] != 0.0) or torch.any(
                        substep.written_torque_b[mask] != 0.0
                    ):
                        raise RuntimeError(f"{label} interruption did not write same-step zero torque")
                if label == "strike":
                    strike_interrupt_zero_rows += count
                    nonzero_strike_interrupt_zero_rows += int(
                        (row.scheduler_step.strike_interrupted_sampled_impulse_y_mps[mask].abs() > 0.0)
                        .sum()
                        .detach()
                        .cpu()
                    )
                else:
                    window_interrupt_zero_rows += count
        if strike_interrupt_zero_rows != strike_interrupt_count:
            raise RuntimeError("strike interruption counter and zero-clear receipts disagree")
        if window_interrupt_zero_rows != window_interrupt_count:
            raise RuntimeError("window interruption counter and zero-clear receipts disagree")
        reset_observed = reset_env_steps > 0 and reset_scene_write_count > 0
        strike_observed = strike_window_env_steps > 0
        strike_interrupt_observed = nonzero_strike_interrupt_zero_rows > 0
        full_lifecycle_coverage = args.cell == "L1" and reset_observed and strike_observed and strike_interrupt_observed
        status = (
            "command_buffer_full_lifecycle_probe_pass_solver_readback_unavailable"
            if full_lifecycle_coverage
            else "command_buffer_only_probe_pass_lifecycle_paths_uncovered_solver_readback_unavailable"
        )

        result = {
            "schema_version": 1,
            "status": status,
            "launch_authorized": False,
            "training_authorized": False,
            "task": args.task,
            "cell": args.cell,
            "num_envs": int(args.num_envs),
            "policy_steps": int(args.steps),
            "physics_substeps": len(all_substeps),
            "source_commit": source_commit,
            "isaaclab_commit": isaaclab_commit,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "source_module_path": source_module_path,
            "isaaclab_module_path": isaaclab_module_path,
            "isaaclab_app_module_path": isaaclab_app_module_path,
            "isaaclab_tasks_module_path": isaaclab_tasks_module_path,
            "motion_files": [str(path) for path in motion_paths],
            "motion_sha256": list(motion_sha256),
            "backend_contract": isaac_lateral_backend_contract(),
            "backend_identity_sha256": isaac_lateral_backend_identity_sha256(),
            "transform_contract": isaac_lateral_transform_contract(),
            "transform_identity_sha256": isaac_lateral_transform_identity_sha256(),
            "random_schedule_identity_sha256": cfg.random_schedule_identity_sha256,
            "hard_safety_identity_sha256": cfg.hard_safety_identity_sha256,
            "policy_step_dt_s": float(base_env.step_dt),
            "physics_dt_s": float(base_env.physics_dt),
            "decimation": int(base_env.cfg.decimation),
            "explicit_initial_reset_completed": True,
            "event_term_manifest": event_term_manifest,
            "interval_event_terms_present": False,
            "all_scene_writes_synchronized": True,
            "all_command_buffer_readbacks_exact": True,
            "observed_reset_torso_buffers_zero_exact": (True if reset_observed else None),
            "solver_execution_readback_available": False,
            "receipt_transcript_schema": "typed_dataclass_tensor_bytes_v1",
            "receipt_transcript_sha256": receipt_digest.hexdigest(),
            "reset_env_step_count": reset_env_steps,
            "strike_window_env_step_count": strike_window_env_steps,
            "recovery_hold_eligible_env_step_count": eligible_env_steps,
            "lifecycle_coverage": {
                "full_lifecycle_coverage": full_lifecycle_coverage,
                "reset_env_step_count": reset_env_steps,
                "reset_scene_write_count": reset_scene_write_count,
                "reset_clear_observed": reset_observed,
                "strike_window_env_step_count": strike_window_env_steps,
                "strike_window_observed": strike_observed,
                "active_pulse_strike_interrupt_count": strike_interrupt_count,
                "nonzero_active_pulse_strike_interrupt_zero_clear_count": (nonzero_strike_interrupt_zero_rows),
                "active_pulse_strike_interrupt_zero_clear_observed": (strike_interrupt_observed),
                "active_pulse_window_interrupt_count": window_interrupt_count,
                "active_pulse_window_interrupt_zero_clear_count": (window_interrupt_zero_rows),
            },
            "counters": counters,
            "non_claims": [
                "No PhysX solver-consumed wrench getter exists in Isaac Lab 2.1.",
                "No throughput/no-host-sync gate was run.",
                "No training, checkpoint, behaviour, MuJoCo, deployment, or hardware result was produced.",
            ],
        }
        payload = (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(output_path, flags, 0o600)
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("short write while persisting runtime probe receipt")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if hasattr(os, "O_DIRECTORY"):
            parent_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
