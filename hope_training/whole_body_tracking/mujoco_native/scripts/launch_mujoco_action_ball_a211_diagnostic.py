#!/usr/bin/env python3
"""Execute the narrow real-MjData A211 partial-reward checkpoint smoke.

This is the A counterpart of ``launch_mujoco_action_ball_211_diagnostic``.
It reuses that launcher's sealed plant/question/motion authorities and PPO
checkpoint protocol, but consumes ``current_lm`` 111 desired-contact semantics
through the independent A211 adapter.  It remains diagnostic-only and capped
at 64 sequential CPU environments; the intended first Pod proof is one env.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - preserve the validated native import order
    import torch as _torch_import_order_guard  # noqa: F401
except ImportError:
    _torch_import_order_guard = None


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hope_training.whole_body_tracking.mujoco_native import (  # noqa: E402
    action_ball_211_abi as abi,
)
from hope_training.whole_body_tracking.mujoco_native import (  # noqa: E402
    action_ball_a211_env,
)
from hope_training.whole_body_tracking.mujoco_native import checkpoint  # noqa: E402
from hope_training.whole_body_tracking.mujoco_native import trainer  # noqa: E402
from hope_training.whole_body_tracking.mujoco_native.scripts import (  # noqa: E402
    launch_mujoco_action_ball_211_diagnostic as shared_launch,
)
from hope_training.whole_body_tracking.mujoco_native.scripts import (  # noqa: E402
    launch_mujoco_fixed_center_diagnostic as fixed_launch,
)


EXECUTION_PLAN_KIND = "action_ball_a211_mujoco_partial_isaac_reward_plan_v1"
RESULT_KIND = "action_ball_a211_mujoco_partial_isaac_reward_result_v1"
CHILD_REQUEST_KIND = "action_ball_a211_mujoco_cold_child_request_v1"
CHILD_RESULT_KIND = "action_ball_a211_mujoco_cold_child_result_v1"
MAX_EXECUTE_ENVS = 64


class LaunchBlocked(RuntimeError):
    """The A211 diagnostic lacks an exact authority or contract."""


def _runtime_module_sha256s() -> dict[str, str]:
    values = shared_launch._runtime_module_sha256s()
    values.update(
        {
            "a211_runner": shared_launch._sha256_file(Path(__file__).resolve()),
            "a211_env": shared_launch._sha256_file(
                Path(action_ball_a211_env.__file__).resolve()
            ),
        }
    )
    return values


def _execution_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile != "A211":
        raise LaunchBlocked("A211 execution requires --profile A211")
    if (
        args.num_envs < 1
        or args.required_active_steps < 1
        or args.pre_checkpoint_updates != 1
    ):
        raise LaunchBlocked(
            "A211 smoke requires positive env/active counts and one pre-checkpoint update"
        )
    if (
        args.reset_wait_min_steps != 5
        or args.reset_wait_max_steps != 25
        or args.reset_wait_seed != 20260804
        or args.episode_horizon_steps != 500
        or args.required_active_steps != 200
    ):
        raise LaunchBlocked(
            "A211 smoke requires the frozen WAIT schedule 20260804/5..25/500/200"
        )
    if args.output_dir is None:
        raise LaunchBlocked("execution output directory is required")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise LaunchBlocked("output directory already exists (no-clobber)")
    if not output.parent.is_dir():
        raise LaunchBlocked("output directory parent does not exist")
    task = action_ball_a211_env.A211TaskAuthority.load(
        args.immutable_tape,
        expected_file_sha256=args.expected_immutable_tape_sha256,
    )
    nominal_tick_float = (
        task.time_to_contact_s
        + args.reset_wait_max_steps * action_ball_a211_env.A211_POLICY_DT_S
    ) / action_ball_a211_env.A211_POLICY_DT_S
    nominal_tick = int(round(nominal_tick_float))
    if nominal_tick < 1 or abs(nominal_tick_float - nominal_tick) > 1.0e-12:
        raise LaunchBlocked("fixed A211 strike is not on one exact policy tick")
    if args.episode_horizon_steps < nominal_tick + 2:
        raise LaunchBlocked("A211 rollout does not reach strike plus outcome tail")
    return {
        "schema_version": 1,
        "kind": EXECUTION_PLAN_KIND,
        "mode": "execute",
        "profile": "A211",
        "source_lineage": shared_launch._source_lineage(abi.A211_PROFILE),
        "authorities": shared_launch._execution_authorities(args),
        "runtime_module_sha256s": _runtime_module_sha256s(),
        "workload": {
            "num_envs": args.num_envs,
            "minimum_rollout_steps_per_update": args.episode_horizon_steps,
            "maximum_rollout_steps_per_update": 2 * args.episode_horizon_steps,
            "update_boundary_rule": (
                "same_policy_collect_all_rows_until_first_full_reset_boundary_"
                "at_or_after_minimum_no_discarded_tail"
            ),
            "reset_wait_schedule": {
                "seed": args.reset_wait_seed,
                "min_wait_steps": args.reset_wait_min_steps,
                "max_wait_steps": args.reset_wait_max_steps,
                "episode_horizon_steps": args.episode_horizon_steps,
                "required_active_steps": args.required_active_steps,
            },
            "nominal_strike_policy_tick_1based": nominal_tick,
            "pre_checkpoint_updates": 1,
            "fresh_process_matched_updates": 1,
            "total_ppo_updates": 2,
            "actor_width": abi.ACTOR_WIDTH,
            "critic_width": abi.CRITIC_WIDTH,
            "cpu_sequential_vecenv": True,
            "execute_env_cap": MAX_EXECUTE_ENVS,
            "torch_and_mujoco_execution_device": "cpu",
            "cuda_or_gpu_execution_used": False,
            "pod_gpu_assignment_consumed": False,
            "functional_canary_may_colocate_with_isaac_gpu_runs": True,
            "colocated_wall_time_is_speed_evidence": False,
            "matched_4096_runtime_measured": False,
        },
        "outputs": {
            "directory": str(output),
            "launch_preparation": str(output / "launch_preparation.json"),
            "checkpoint": str(output / "reset_boundary.pt"),
            "result": str(output / "result.json"),
            "fresh_wait_bootstrap_canary": str(
                output / "fresh_wait_bootstrap_canary.json"
            ),
        },
        "claims": {
            "live_mjdata_plant_fields": True,
            "phase_aware_measured_v4_teacher": True,
            "hidden_wait_atomic_reveal": True,
            "hidden_wait_ball_parked": True,
            "ball_only_atomic_sealed_launch_on_reveal": True,
            "robot_state_continuous_across_reveal": True,
            "fixed_a211_current_lm_111_task_tuple": True,
            "a211_desired_contact_window_reward_implemented": True,
            "a211_base_position_reward_enabled": False,
            "a211_prestrike_racket_progress_reward_implemented": True,
            "a211_actual_contact_achieved_landing_implemented": True,
            "same_motion_sha_fail_closed": True,
            "isaac_equivalent_fresh_actor_hold_bootstrap_required": True,
            "fresh_actor_max_wait_runtime_canary_required": True,
            "separate_actor_critic_normalizers": True,
            "fresh_process_update_2_exact": True,
            "placeholder_or_zero_padded_columns_used": False,
            "reward_scope": action_ball_a211_env.A211_REWARD_SCOPE,
            "reward_parity_status": "partial_fail_closed",
            "complete_isaac_reward_parity_claimed": False,
            "matched_4096_runtime_measured": False,
            "formal_training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
        "formal_blockers": list(action_ball_a211_env.A211_FORMAL_BLOCKERS),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }


def _build_env(
    args: argparse.Namespace,
    *,
    expected_launch_preparation: Mapping[str, Any] | None = None,
) -> action_ball_a211_env.MujocoA211DiagnosticVecEnv:
    fixed = fixed_launch._build_env(args)
    task = action_ball_a211_env.A211TaskAuthority.load(
        args.immutable_tape,
        expected_file_sha256=args.expected_immutable_tape_sha256,
    )
    mimic = action_ball_a211_env.MeasuredA211MimicAuthority.load(
        args.measured_motion,
        expected_file_sha256=args.expected_measured_motion_sha256,
        task=task,
    )
    env = action_ball_a211_env.MujocoA211DiagnosticVecEnv(
        base_env=fixed,
        task_authority=task,
        mimic_authority=mimic,
    )
    if expected_launch_preparation is not None and (
        _launch_preparation_payload(env) != expected_launch_preparation
    ):
        raise LaunchBlocked("derived A211 launch differs from sealed preparation")
    return env


def _launch_preparation_payload(
    env: action_ball_a211_env.MujocoA211DiagnosticVecEnv,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "action_ball_a211_mujoco_launch_preparation_v1",
        "fixed_center_preparation": fixed_launch._launch_preparation_payload(env.base),
        "task_authority": env.producer.task.receipt,
        "measured_mimic_authority": env.producer.mimic.receipt,
        "a211_reward_contract": copy.deepcopy(env._reward_contract),
        "observation_authorities_sha256": env.producer.authorities.content_sha256,
        "training_identity": env.diagnostic_training_identity(),
        "fresh_actor_bootstrap": env.fresh_actor_bootstrap_contract(),
        "canonical_reset_boundary_sha256": env._canonical_boundary_sha256,
        "safe_ready_authority_status": (
            action_ball_a211_env.shared.SAFE_READY_AUTHORITY_STATUS
        ),
        "safe_ready_formal_pass_claimed": False,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    payload["preparation_content_sha256"] = shared_launch._sha256_json(payload)
    fixed_launch._finite_tree(payload, "a211_launch_preparation")
    return payload


def _build_trainer(
    env: action_ball_a211_env.MujocoA211DiagnosticVecEnv,
    args: argparse.Namespace,
) -> trainer.MujocoDiagnosticPPOTrainer:
    identity = trainer.TrainerIdentity(**env.diagnostic_training_identity())
    bootstrap = env.fresh_actor_bootstrap_contract()
    config = trainer.DiagnosticPPOConfig(
        action_dim=env.num_actions,
        rollout_steps=args.episode_horizon_steps,
        rollout_reset_boundary_extension_steps=args.episode_horizon_steps,
        hidden_dims=tuple(args.hidden_dims),
        seed=args.seed,
        learning_rate=args.learning_rate,
        initial_action_std=args.initial_action_std,
        fresh_actor_output_bias=tuple(
            float(value)
            for value in env.producer.robot_tape.history_fill_action.tolist()
        ),
        fresh_actor_bootstrap_authority_sha256=bootstrap["content_sha256"],
        **abi.A211_PROFILE.trainer_config_kwargs(),
    )
    return trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=identity,
        config=config,
    )


def _run_audited_update(
    env: action_ball_a211_env.MujocoA211DiagnosticVecEnv,
    diagnostic_trainer: trainer.MujocoDiagnosticPPOTrainer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env.reset_reward_audit()
    update = diagnostic_trainer.run_update()
    audit = env.reward_audit_receipt()
    actual_rollout_steps = update.get("rollout_steps")
    if type(actual_rollout_steps) is not int or actual_rollout_steps < 1:
        raise LaunchBlocked("A211 update receipt omits its actual rollout length")
    expected_rows = actual_rollout_steps * env.num_envs
    if (
        audit.get("transition_step_count")
        != actual_rollout_steps
        or audit.get("row_count") != expected_rows
        or any(
            row.get("sample_count") != expected_rows
            for row in audit.get("prior_terms", {}).values()
        )
    ):
        raise LaunchBlocked("A211 raw reward audit did not cover the complete update")
    # A fresh balance/mimic policy is allowed to terminate before its first
    # strike.  The exact 3/11/11 eligibility formulas are unit-tested at the
    # reward boundary; an optimizer update must report, but must not fabricate
    # or require, target-window denominators that the rollout did not reach.
    for name in (
        "desired_contact_position_window_row_count",
        "desired_contact_velocity_window_row_count",
        "desired_contact_face_window_row_count",
        "desired_contact_any_window_row_count",
    ):
        value = audit.get(name)
        if type(value) is not int or value < 0 or value > expected_rows:
            raise LaunchBlocked(f"A211 raw reward audit has invalid {name}")
    return update, audit


def _child_args(args: argparse.Namespace) -> dict[str, Any]:
    values = fixed_launch._child_args(args)
    values.update(
        {
            "profile": "A211",
            "immutable_tape": str(args.immutable_tape.expanduser().resolve()),
            "expected_immutable_tape_sha256": args.expected_immutable_tape_sha256,
            "measured_motion": str(args.measured_motion.expanduser().resolve()),
            "expected_measured_motion_sha256": args.expected_measured_motion_sha256,
            "reset_wait_min_steps": args.reset_wait_min_steps,
            "reset_wait_max_steps": args.reset_wait_max_steps,
            "reset_wait_seed": args.reset_wait_seed,
            "episode_horizon_steps": args.episode_horizon_steps,
            "required_active_steps": args.required_active_steps,
        }
    )
    return values


def _args_from_child(values: Mapping[str, Any]) -> argparse.Namespace:
    args = fixed_launch._args_from_child(values)
    args.profile = "A211"
    args.immutable_tape = Path(args.immutable_tape)
    args.measured_motion = Path(args.measured_motion)
    return args


def _cold_child(request_path: Path, result_path: Path) -> int:
    try:
        request = json.loads(request_path.resolve().read_text("utf-8"))
        if (
            not isinstance(request, Mapping)
            or request.get("kind") != CHILD_REQUEST_KIND
            or request.get("runtime_module_sha256s") != _runtime_module_sha256s()
            or request.get("parent_pid") == os.getpid()
            or request.get("diagnostic_unauthorized") is not True
        ):
            raise LaunchBlocked("cold-child request identity differs")
        checkpoint_path = Path(request["checkpoint_path"]).resolve()
        if shared_launch._sha256_file(checkpoint_path) != request.get(
            "checkpoint_sha256"
        ):
            raise LaunchBlocked("cold-child checkpoint SHA differs")
        preparation_path = Path(request["launch_preparation_path"]).resolve()
        if shared_launch._sha256_file(preparation_path) != request.get(
            "launch_preparation_file_sha256"
        ):
            raise LaunchBlocked("cold-child preparation file SHA differs")
        sealed = json.loads(preparation_path.read_text("utf-8"))
        if sealed.get("preparation_content_sha256") != request.get(
            "preparation_content_sha256"
        ):
            raise LaunchBlocked("cold-child preparation content differs")
        args = _args_from_child(request["args"])
        shared_launch._execution_authorities(args)
        env = _build_env(args, expected_launch_preparation=sealed)
        cold = _build_trainer(env, args)
        load_receipt = checkpoint.ResetBoundaryCheckpoint().load(
            checkpoint_path, cold
        )
        update, audit = _run_audited_update(env, cold)
        result = {
            "schema_version": 1,
            "kind": CHILD_RESULT_KIND,
            "pid": os.getpid(),
            "parent_pid": request["parent_pid"],
            "checkpoint_sha256": request["checkpoint_sha256"],
            "runtime_module_sha256s": copy.deepcopy(
                request["runtime_module_sha256s"]
            ),
            "launch_preparation_file_sha256": request[
                "launch_preparation_file_sha256"
            ],
            "preparation_content_sha256": request["preparation_content_sha256"],
            "checkpoint_load_receipt": load_receipt,
            "next_update_receipt": update,
            "next_update_raw_reward_audit": audit,
            "state_sha256": fixed_launch._state_digest(cold.checkpoint_state()),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        fixed_launch._finite_tree(result, "a211_cold_child_result")
        fixed_launch._write_new_json(result_path.resolve(), result)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[A211-COLD-FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _execute(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not args.confirm_diagnostic_unauthorized:
        raise LaunchBlocked(
            "--execute-two-updates requires --confirm-diagnostic-unauthorized"
        )
    if args.num_envs > MAX_EXECUTE_ENVS:
        raise LaunchBlocked(
            f"execution is capped at {MAX_EXECUTE_ENVS} envs; 4096 is unmeasured"
        )
    output = args.output_dir.expanduser().resolve()
    canary_args = copy.copy(args)
    canary_args.reset_wait_min_steps = (
        shared_launch.FRESH_WAIT_BOOTSTRAP_CANARY_TICKS
    )
    canary_args.reset_wait_max_steps = (
        shared_launch.FRESH_WAIT_BOOTSTRAP_CANARY_TICKS
    )
    canary_env = _build_env(canary_args)
    canary_trainer = _build_trainer(canary_env, canary_args)
    bootstrap_canary = shared_launch._fresh_wait_bootstrap_canary(
        canary_env, canary_trainer, profile="A211"
    )
    del canary_trainer, canary_env
    gc.collect()
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    bootstrap_canary_path = output / "fresh_wait_bootstrap_canary.json"
    fixed_launch._write_new_json(bootstrap_canary_path, bootstrap_canary)
    source_env = _build_env(args)
    preparation = _launch_preparation_payload(source_env)
    preparation_path = output / "launch_preparation.json"
    fixed_launch._write_new_json(preparation_path, preparation)
    preparation_file_sha = shared_launch._sha256_file(preparation_path)
    source = _build_trainer(source_env, args)
    pre_updates = []
    pre_audits = []
    for _ in range(args.pre_checkpoint_updates):
        update, audit = _run_audited_update(source_env, source)
        if update.get("at_reset_boundary") is not True:
            raise LaunchBlocked("A211 update did not end at a reset boundary")
        pre_updates.append(update)
        pre_audits.append(audit)
    checkpoint_path = output / "reset_boundary.pt"
    save = checkpoint.ResetBoundaryCheckpoint().save(checkpoint_path, source)
    reference_update, reference_audit = _run_audited_update(source_env, source)
    reference_state = fixed_launch._state_digest(source.checkpoint_state())
    request = {
        "schema_version": 1,
        "kind": CHILD_REQUEST_KIND,
        "runtime_module_sha256s": _runtime_module_sha256s(),
        "parent_pid": os.getpid(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": shared_launch._sha256_file(checkpoint_path),
        "launch_preparation_path": str(preparation_path),
        "launch_preparation_file_sha256": preparation_file_sha,
        "preparation_content_sha256": preparation["preparation_content_sha256"],
        "args": _child_args(args),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    request_path = output / "cold_request.json"
    child_path = output / "cold_result.json"
    fixed_launch._write_new_json(request_path, request)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--profile",
            "A211",
            "--_cold-child",
            str(request_path),
            str(child_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise LaunchBlocked(
            "A211 cold child failed: " + (completed.stderr.strip() or "no stderr")
        )
    child = json.loads(child_path.read_text("utf-8"))
    if (
        child.get("kind") != CHILD_RESULT_KIND
        or child.get("parent_pid") != os.getpid()
        or child.get("pid") == os.getpid()
        or child.get("next_update_receipt") != reference_update
        or child.get("next_update_raw_reward_audit") != reference_audit
        or child.get("state_sha256") != reference_state
    ):
        raise LaunchBlocked("fresh-process A211 continuation differs")
    result = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "status": "A211_PARTIAL_ISAAC_REWARD_CHECKPOINT_DIAGNOSTIC_COMPLETE",
        "promotion_blocking_evidence": shared_launch._promotion_blocking_summary(
            profile="A211",
            canary=bootstrap_canary,
            update_receipts=[*pre_updates, reference_update],
            checkpoint_save_receipt=save,
        ),
        "plan": copy.deepcopy(dict(plan)),
        "pre_checkpoint_update_receipts": pre_updates,
        "pre_checkpoint_raw_reward_audits": pre_audits,
        "checkpoint_save_receipt": save,
        "checkpoint_sha256": request["checkpoint_sha256"],
        "launch_preparation_path": str(preparation_path),
        "launch_preparation_file_sha256": preparation_file_sha,
        "preparation_content_sha256": preparation["preparation_content_sha256"],
        "matched_next_update_receipt": reference_update,
        "matched_next_update_raw_reward_audit": reference_audit,
        "matched_state_sha256": reference_state,
        "cold_child_pid": child["pid"],
        "fresh_process_cold_load_exact": True,
        "fresh_process_update_2_exact": True,
        "fresh_wait_bootstrap_canary": bootstrap_canary,
        "fresh_wait_bootstrap_canary_path": str(bootstrap_canary_path),
        "fresh_wait_bootstrap_canary_file_sha256": shared_launch._sha256_file(
            bootstrap_canary_path
        ),
        "actor_width": abi.ACTOR_WIDTH,
        "critic_width": abi.CRITIC_WIDTH,
        "hidden_wait_ball_parked": True,
        "ball_only_atomic_sealed_launch_on_reveal": True,
        "robot_state_continuous_across_reveal": True,
        "reward_scope": action_ball_a211_env.A211_REWARD_SCOPE,
        "reward_parity_status": "partial_fail_closed",
        "a211_desired_contact_reward_available": True,
        "a211_achieved_outcome_reward_available": True,
        "complete_isaac_reward_parity_claimed": False,
        "matched_4096_runtime_measured": False,
        "formal_blockers": list(action_ball_a211_env.A211_FORMAL_BLOCKERS),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    fixed_launch._finite_tree(result, "a211_result")
    fixed_launch._write_new_json(output / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    # Keep one argument surface with the C/plan launcher so the same material
    # authorities and operation docs apply to both profiles.
    return shared_launch._parser()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._cold_child is not None:
        return _cold_child(Path(args._cold_child[0]), Path(args._cold_child[1]))
    try:
        if not args.execute_two_updates:
            task = shared_launch._task_authority(
                args.task_question_authority,
                args.expected_task_question_sha256,
            )
            result = shared_launch._static_plan(
                abi.A211_PROFILE,
                num_envs=args.num_envs,
                task_authority=task,
            )
        else:
            plan = _execution_plan(args)
            result = _execute(args, plan)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[MUJOCO-A211-BLOCKED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
