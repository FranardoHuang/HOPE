#!/usr/bin/env python3
"""Launch the host-small fixed-centre N1 diagnostic PPO recipe.

The default mode only prints a content-checked plan.  Execution is no-clobber,
requires an explicit diagnostic-only acknowledgement, and is capped below a
4096 workload.  It proves a reset-boundary save followed by an exact next
update in a fresh Python process; it does not support mid-episode resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

# On macOS, importing NumPy-linked native modules before this conda Torch build
# can load a conflicting OpenMP runtime and abort the interpreter.  Plan mode
# remains dependency-light when Torch is absent.
try:  # pragma: no cover - ordering guard, not application logic
    import torch as _torch_import_order_guard  # noqa: F401
except ImportError:  # plan-only hosts need no Torch installation
    _torch_import_order_guard = None


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hope_training.whole_body_tracking.mujoco_native import checkpoint  # noqa: E402
from hope_training.whole_body_tracking.mujoco_native import (
    fixed_center_recipe,
)  # noqa: E402
from hope_training.whole_body_tracking.mujoco_native import trainer  # noqa: E402
from hope_training.whole_body_tracking.mujoco_native import vec_env  # noqa: E402


PLAN_KIND = "a3_mujoco_fixed_center_diagnostic_plan_v1"
RESULT_KIND = "a3_mujoco_fixed_center_diagnostic_result_v1"
CHILD_REQUEST_KIND = "a3_mujoco_fixed_center_cold_child_request_v1"
CHILD_RESULT_KIND = "a3_mujoco_fixed_center_cold_child_result_v1"
MAX_EXECUTE_ENVS = 64


class LaunchError(RuntimeError):
    """The controlled diagnostic launch failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _authority(path: Path, expected: str, name: str) -> dict[str, Any]:
    if path is None:
        raise LaunchError(f"{name} path is required")
    source = path.expanduser().resolve()
    if not source.is_file():
        raise LaunchError(f"{name} is not a file: {source}")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise LaunchError(f"expected {name} SHA must be lowercase SHA-256")
    actual = _sha256_file(source)
    if actual != expected:
        raise LaunchError(f"{name} SHA differs from explicit authority")
    return {"path": str(source), "sha256": actual, "bytes": source.stat().st_size}


def _finite_tree(value: Any, path: str = "value") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LaunchError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
        return
    raise LaunchError(f"{path} contains unsupported {type(value).__name__}")


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _state_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def add(item: Any) -> None:
        try:
            import numpy as np
            import torch
        except ImportError as exc:  # pragma: no cover - execution dependency
            raise LaunchError("state hashing requires NumPy and torch") from exc
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"torch\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
        elif isinstance(item, tuple):
            digest.update(b"tuple\0")
            for child in item:
                add(child)
        elif isinstance(item, list):
            digest.update(b"list\0")
            for child in item:
                add(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(json.dumps(item, allow_nan=False).encode("utf-8"))
        else:
            raise LaunchError(f"unsupported trainer state {type(item).__name__}")

    add(value)
    return digest.hexdigest()


def _authorities(args: argparse.Namespace) -> dict[str, Any]:
    rows = {
        "plant_contract": _authority(
            args.plant_contract, args.expected_plant_sha256, "plant contract"
        ),
        "robot_tape": _authority(
            args.robot_tape, args.expected_robot_tape_sha256, "robot tape"
        ),
        "question": _authority(
            args.question, args.expected_question_sha256, "question"
        ),
        "selected_rubber_manifest": _authority(
            args.selected_rubber_manifest,
            args.expected_selected_rubber_manifest_sha256,
            "selected-rubber manifest",
        ),
        "mjcf": _authority(args.mjcf, args.expected_mjcf_sha256, "MJCF"),
    }
    if (args.phase_fidelity_reference_tape is None) != (
        args.expected_phase_fidelity_reference_tape_sha256 is None
    ):
        raise LaunchError("phase tape and expected SHA must be supplied together")
    if args.phase_fidelity_reference_tape is not None:
        rows["phase_fidelity_reference_tape"] = _authority(
            args.phase_fidelity_reference_tape,
            args.expected_phase_fidelity_reference_tape_sha256,
            "phase-fidelity reference tape",
        )
    return rows


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_envs < 1 or args.pre_checkpoint_updates < 1 or args.active_steps < 1:
        raise LaunchError(
            "num_envs, active_steps and pre_checkpoint_updates must be positive"
        )
    if args.output_dir is None:
        raise LaunchError("output directory is required")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise LaunchError("output directory already exists (no-clobber)")
    if not output.parent.is_dir():
        raise LaunchError("output directory parent does not exist")
    episode_steps = args.reset_wait_steps + args.active_steps
    spec = fixed_center_recipe.FixedCenterRecipeSpec(
        reset_wait_steps=args.reset_wait_steps
    )
    return {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "mode": "execute" if args.execute else "plan",
        "authorities": _authorities(args),
        "recipe_source_sha256": fixed_center_recipe.RECIPE_SOURCE_SHA256,
        "recipe_spec_sha256": spec.content_sha256,
        "workload": {
            "num_envs": args.num_envs,
            "reset_wait_steps": args.reset_wait_steps,
            "task_active_steps": args.active_steps,
            "rollout_steps_per_update": episode_steps,
            "pre_checkpoint_updates": args.pre_checkpoint_updates,
            "fresh_process_matched_updates": 1,
            "cpu_sequential_vecenv": True,
            "num_envs_4096_plan_shape_supported": True,
            "matched_4096_runtime_measured": False,
            "execute_env_cap": MAX_EXECUTE_ENVS,
        },
        "outputs": {
            "directory": str(output),
            "launch_preparation": str(output / "launch_preparation.json"),
            "checkpoint": str(output / "reset_boundary.pt"),
            "result": str(output / "result.json"),
        },
        "claims": {
            "joint_space_frame0_teacher_only": True,
            "full_body_measured_mimic": False,
            "actor_task_mask_only_wait": True,
            "physical_ball_parked_during_wait": True,
            "physical_ball_trajectory_includes_wait_from_reset": False,
            "ball_only_atomic_sealed_launch_on_reveal": True,
            "fixed_question": True,
            "online_inverse_solve_calls": 0,
            "reset_boundary_resume_only": True,
            "mid_episode_resume": False,
            "formal_training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
        "formal_blockers": list(fixed_center_recipe.FORMAL_BLOCKERS),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }


def _validate_selected_rubber(env: Any, args: argparse.Namespace) -> None:
    manifest = args.selected_rubber_manifest.expanduser().resolve()
    for index, question in enumerate(env.questions):
        lineage = getattr(question, "selected_rubber_action_lineage", None)
        if not isinstance(lineage, Mapping):
            raise LaunchError(f"question {index} has no selected-rubber lineage")
        if lineage.get("action_manifest_sha256") != (
            args.expected_selected_rubber_manifest_sha256
        ):
            raise LaunchError(f"question {index} manifest SHA differs")
        logical = lineage.get("action_manifest_repo_relative_path")
        if not isinstance(logical, str) or (REPO_ROOT / logical).resolve() != manifest:
            raise LaunchError(f"question {index} manifest path differs")


def _launch_preparation_payload(env: Any) -> dict[str, Any]:
    preparation = env.continuous_wait_preparation
    payload = {
        "schema_version": 1,
        "kind": "a3_mujoco_fixed_center_launch_preparation_artifact_v1",
        "recipe_source_sha256": fixed_center_recipe.RECIPE_SOURCE_SHA256,
        "recipe_spec_sha256": preparation.spec_sha256,
        "preparation_content_sha256": preparation.content_sha256,
        "wait_policy_steps": preparation.wait_policy_steps,
        "wait_physics_substeps": preparation.wait_physics_substeps,
        "physics_step_dt_s": preparation.physics_step_dt_s,
        "per_env": [dict(row) for row in preparation.per_env],
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    _finite_tree(payload, "launch_preparation")
    return payload


def _build_env(
    args: argparse.Namespace,
    *,
    expected_launch_preparation: Mapping[str, Any] | None = None,
) -> Any:
    ranged_wait = getattr(args, "reset_wait_min_steps", None) is not None
    if ranged_wait:
        episode_length = int(args.episode_horizon_steps)
        recipe_spec = fixed_center_recipe.FixedCenterRecipeSpec(
            reset_wait_steps=None,
            reset_wait_min_steps=int(args.reset_wait_min_steps),
            reset_wait_max_steps=int(args.reset_wait_max_steps),
            reset_wait_seed=int(args.reset_wait_seed),
            required_active_steps=int(args.required_active_steps),
        )
    else:
        episode_length = args.reset_wait_steps + args.active_steps
        recipe_spec = fixed_center_recipe.FixedCenterRecipeSpec(
            reset_wait_steps=args.reset_wait_steps
        )
    base = vec_env.MujocoN1DiagnosticVecEnv.from_authorities(
        contract_path=args.plant_contract,
        robot_tape_path=args.robot_tape,
        expected_robot_tape_sha256=args.expected_robot_tape_sha256,
        question_path=args.question,
        expected_question_sha256=args.expected_question_sha256,
        num_envs=args.num_envs,
        mjcf_path=args.mjcf,
        phase_fidelity_reference_tape_path=args.phase_fidelity_reference_tape,
        expected_phase_fidelity_reference_tape_sha256=(
            args.expected_phase_fidelity_reference_tape_sha256
        ),
        enable_c_lite_reward=True,
        diagnostic_episode_length=episode_length,
    )
    _validate_selected_rubber(base, args)
    spec = recipe_spec
    base = fixed_center_recipe.prepare_continuous_wait_base(base, spec)
    teacher_reference = fixed_center_recipe.Frame0JointTeacher.from_fixed_tape(
        base.cores[0].binding, base.robot_tape
    )
    env = fixed_center_recipe.FixedCenterDiagnosticVecEnv(
        base_env=base,
        teacher_reference=teacher_reference,
        spec=spec,
    )
    if expected_launch_preparation is not None and (
        _launch_preparation_payload(env) != expected_launch_preparation
    ):
        raise LaunchError("derived launch differs from sealed preparation artifact")
    return env


def _build_trainer(env: Any, args: argparse.Namespace) -> Any:
    identity = trainer.TrainerIdentity(**env.diagnostic_training_identity())
    config = trainer.DiagnosticPPOConfig(
        observation_dim=env.num_observations,
        action_dim=env.num_actions,
        rollout_steps=args.reset_wait_steps + args.active_steps,
        hidden_dims=tuple(args.hidden_dims),
        seed=args.seed,
        learning_rate=args.learning_rate,
        initial_action_std=args.initial_action_std,
    )
    return trainer.MujocoDiagnosticPPOTrainer(env=env, identity=identity, config=config)


def _child_args(args: argparse.Namespace) -> dict[str, Any]:
    fields = (
        "plant_contract",
        "expected_plant_sha256",
        "robot_tape",
        "expected_robot_tape_sha256",
        "question",
        "expected_question_sha256",
        "selected_rubber_manifest",
        "expected_selected_rubber_manifest_sha256",
        "mjcf",
        "expected_mjcf_sha256",
        "phase_fidelity_reference_tape",
        "expected_phase_fidelity_reference_tape_sha256",
        "num_envs",
        "reset_wait_steps",
        "active_steps",
        "seed",
        "learning_rate",
        "initial_action_std",
    )
    values = {name: getattr(args, name) for name in fields}
    for name, value in tuple(values.items()):
        if isinstance(value, Path):
            values[name] = str(value.expanduser().resolve())
    values["hidden_dims"] = list(args.hidden_dims)
    return values


def _args_from_child(values: Mapping[str, Any]) -> argparse.Namespace:
    values = dict(values)
    for name in (
        "plant_contract",
        "robot_tape",
        "question",
        "selected_rubber_manifest",
        "mjcf",
        "phase_fidelity_reference_tape",
    ):
        if values.get(name) is not None:
            values[name] = Path(values[name])
    return argparse.Namespace(**values)


def _cold_child(request_path: Path, result_path: Path) -> int:
    try:
        request = json.loads(request_path.resolve().read_text("utf-8"))
        if (
            not isinstance(request, Mapping)
            or request.get("kind") != CHILD_REQUEST_KIND
            or request.get("runner_sha256") != _sha256_file(Path(__file__).resolve())
            or request.get("recipe_source_sha256")
            != fixed_center_recipe.RECIPE_SOURCE_SHA256
            or request.get("parent_pid") == os.getpid()
            or request.get("diagnostic_unauthorized") is not True
        ):
            raise LaunchError("cold-child request identity differs")
        checkpoint_path = Path(request["checkpoint_path"]).resolve()
        if _sha256_file(checkpoint_path) != request.get("checkpoint_sha256"):
            raise LaunchError("cold-child checkpoint SHA differs")
        args = _args_from_child(request["args"])
        _authorities(args)
        preparation_path = Path(request["launch_preparation_path"]).resolve()
        if _sha256_file(preparation_path) != request.get(
            "launch_preparation_file_sha256"
        ):
            raise LaunchError("cold-child launch preparation file SHA differs")
        sealed_preparation = json.loads(preparation_path.read_text("utf-8"))
        if (
            sealed_preparation.get("preparation_content_sha256")
            != request.get("preparation_content_sha256")
            or sealed_preparation.get("recipe_source_sha256")
            != fixed_center_recipe.RECIPE_SOURCE_SHA256
        ):
            raise LaunchError("cold-child launch preparation content differs")
        cold_env = _build_env(args, expected_launch_preparation=sealed_preparation)
        cold = _build_trainer(cold_env, args)
        load_receipt = checkpoint.ResetBoundaryCheckpoint().load(checkpoint_path, cold)
        update_receipt = cold.run_update()
        result = {
            "schema_version": 1,
            "kind": CHILD_RESULT_KIND,
            "pid": os.getpid(),
            "parent_pid": request["parent_pid"],
            "checkpoint_sha256": request["checkpoint_sha256"],
            "launch_preparation_file_sha256": request["launch_preparation_file_sha256"],
            "preparation_content_sha256": request["preparation_content_sha256"],
            "recipe_source_sha256": fixed_center_recipe.RECIPE_SOURCE_SHA256,
            "checkpoint_load_receipt": load_receipt,
            "next_update_receipt": update_receipt,
            "state_sha256": _state_digest(cold.checkpoint_state()),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        _finite_tree(result)
        _write_new_json(result_path.resolve(), result)
        return 0
    except Exception as exc:  # noqa: BLE001 - process failure boundary
        print(f"[COLD-CHILD-FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _execute(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not args.confirm_diagnostic_unauthorized:
        raise LaunchError("--execute requires --confirm-diagnostic-unauthorized")
    if args.num_envs > MAX_EXECUTE_ENVS:
        raise LaunchError(
            f"execution is capped at {MAX_EXECUTE_ENVS} envs; 4096 is unmeasured"
        )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    source_env = _build_env(args)
    launch_preparation = _launch_preparation_payload(source_env)
    preparation_path = output / "launch_preparation.json"
    _write_new_json(preparation_path, launch_preparation)
    preparation_file_sha = _sha256_file(preparation_path)
    source = _build_trainer(source_env, args)
    pre_checkpoint = []
    for _ in range(args.pre_checkpoint_updates):
        receipt = source.run_update()
        if receipt.get("at_reset_boundary") is not True:
            raise LaunchError("update did not end at a reset boundary")
        pre_checkpoint.append(receipt)
    checkpoint_path = output / "reset_boundary.pt"
    save_receipt = checkpoint.ResetBoundaryCheckpoint().save(checkpoint_path, source)
    reference_update = source.run_update()
    reference_state_sha = _state_digest(source.checkpoint_state())
    request = {
        "schema_version": 1,
        "kind": CHILD_REQUEST_KIND,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "parent_pid": os.getpid(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "launch_preparation_path": str(preparation_path),
        "launch_preparation_file_sha256": preparation_file_sha,
        "preparation_content_sha256": launch_preparation["preparation_content_sha256"],
        "recipe_source_sha256": fixed_center_recipe.RECIPE_SOURCE_SHA256,
        "args": _child_args(args),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    request_path = output / "cold_request.json"
    child_path = output / "cold_result.json"
    _write_new_json(request_path, request)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_cold-child",
            str(request_path),
            str(child_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise LaunchError(
            "cold child failed: " + (completed.stderr.strip() or "no stderr")
        )
    child = json.loads(child_path.read_text("utf-8"))
    if child.get("next_update_receipt") != reference_update:
        raise LaunchError("fresh-process next-update receipt differs")
    if child.get("state_sha256") != reference_state_sha:
        raise LaunchError("fresh-process trainer state differs")
    if (
        child.get("launch_preparation_file_sha256") != preparation_file_sha
        or child.get("preparation_content_sha256")
        != launch_preparation["preparation_content_sha256"]
        or child.get("recipe_source_sha256") != fixed_center_recipe.RECIPE_SOURCE_SHA256
    ):
        raise LaunchError("fresh-process launch/recipe identity differs")
    result = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "status": "DIAGNOSTIC_COMPLETE",
        "plan": plan,
        "pre_checkpoint_update_receipts": pre_checkpoint,
        "checkpoint_save_receipt": save_receipt,
        "checkpoint_sha256": request["checkpoint_sha256"],
        "launch_preparation_path": str(preparation_path),
        "launch_preparation_file_sha256": preparation_file_sha,
        "preparation_content_sha256": launch_preparation["preparation_content_sha256"],
        "recipe_source_sha256": fixed_center_recipe.RECIPE_SOURCE_SHA256,
        "matched_next_update_receipt": reference_update,
        "matched_state_sha256": reference_state_sha,
        "cold_child_pid": child["pid"],
        "fresh_process_cold_load_exact": True,
        "formal_blockers": list(fixed_center_recipe.FORMAL_BLOCKERS),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    _finite_tree(result)
    _write_new_json(output / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plant-contract",
        "robot-tape",
        "question",
        "selected-rubber-manifest",
        "mjcf",
    ):
        parser.add_argument(f"--{name}", type=Path)
    for name in (
        "expected-plant-sha256",
        "expected-robot-tape-sha256",
        "expected-question-sha256",
        "expected-selected-rubber-manifest-sha256",
        "expected-mjcf-sha256",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--phase-fidelity-reference-tape", type=Path)
    parser.add_argument("--expected-phase-fidelity-reference-tape-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--reset-wait-steps", type=int, default=1)
    parser.add_argument("--active-steps", type=int, default=1)
    parser.add_argument("--pre-checkpoint-updates", type=int, default=1)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(64, 64))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--initial-action-std", type=float, default=0.02)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-diagnostic-unauthorized", action="store_true")
    parser.add_argument("--_cold-child", nargs=2, metavar=("REQUEST", "RESULT"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._cold_child is not None:
        return _cold_child(Path(args._cold_child[0]), Path(args._cold_child[1]))
    try:
        plan = _plan(args)
        result = _execute(args, plan) if args.execute else plan
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI failure boundary
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
