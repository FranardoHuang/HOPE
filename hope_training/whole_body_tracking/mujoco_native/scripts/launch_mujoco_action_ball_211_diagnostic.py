#!/usr/bin/env python3
"""Plan A211/C211 or execute their real-MjData partial-reward smokes.

Plan mode remains dependency-light and never manufactures missing columns.
C211 execution reopens every plant, physical-question, measured-motion and
immutable-task authority, constructs live 211-D actor / 319-D critic tensors,
runs one reset-boundary PPO update, saves, then proves update 2 exactly against
a cold load in a fresh Python process.  Each update reaches the immutable
nominal strike plus a conservative post-strike observation tail.  Its reward
is the explicitly partial Isaac-synonymous subset: always-on balance,
non-right-wrist measured full-body mimic and measured-paddle priors, plus the
C211 strike and achieved-flight task reward.  Foot/contact/torque terms,
cross-engine parity and 4096/GPU runtime remain explicit fail-closed blockers.
This runner never authorizes formal training, promotion, deployment, or
hardware use.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Keep the import order used by the already validated fixed-centre launcher.
try:  # pragma: no cover - host runtime guard
    import torch as _torch_import_order_guard  # noqa: F401
except ImportError:  # plan-only hosts need no torch
    _torch_import_order_guard = None


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hope_training.whole_body_tracking.mujoco_native import (  # noqa: E402
    action_ball_211_abi as abi,
)
from hope_training.whole_body_tracking.mujoco_native import (  # noqa: E402
    action_ball_c211_env,
)
from hope_training.whole_body_tracking.mujoco_native import checkpoint  # noqa: E402
from hope_training.whole_body_tracking.mujoco_native import trainer  # noqa: E402
from hope_training.whole_body_tracking.mujoco_native.scripts import (  # noqa: E402
    launch_mujoco_fixed_center_diagnostic as fixed_launch,
)


STATIC_PLAN_KIND = "action_ball_211_mujoco_diagnostic_plan_v2"
EXECUTION_PLAN_KIND = "action_ball_c211_mujoco_partial_isaac_reward_plan_v4"
RESULT_KIND = "action_ball_c211_mujoco_partial_isaac_reward_result_v4"
CHILD_REQUEST_KIND = "action_ball_c211_mujoco_cold_child_request_v4"
CHILD_RESULT_KIND = "action_ball_c211_mujoco_cold_child_result_v4"
MAX_EXECUTE_ENVS = 64
FRESH_WAIT_BOOTSTRAP_CANARY_KIND = (
    "action_ball_211_mujoco_fresh_wait_bootstrap_canary_v1"
)
FRESH_WAIT_BOOTSTRAP_CANARY_TICKS = 25
FRESH_WAIT_BOOTSTRAP_CANARY_SEED = 20260804
FRESH_WAIT_BOOTSTRAP_MAX_QDES_JUMP_RAD = 1.0e-6

_SOURCE_PATHS = {
    "A211": Path(
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/action_ball_a211_trainability.py"
    ),
    "C211": Path(
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/action_ball_c211_trainability.py"
    ),
}


class LaunchBlocked(RuntimeError):
    """The requested runtime operation lacks an exact authority or contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _source_lineage(profile: abi.ActionBall211Profile) -> dict[str, str]:
    path = (REPO_ROOT / _SOURCE_PATHS[profile.label]).resolve()
    actual = _sha256_file(path)
    if actual != profile.source_sha256:
        raise LaunchBlocked(
            f"{profile.label} mirrored ABI source SHA differs: "
            f"expected={profile.source_sha256} actual={actual}"
        )
    leaf_path = (
        REPO_ROOT
        / "hope_training/whole_body_tracking/cfg/task"
        / f"HOPEPingPongActionBall{profile.label}VendorV2N1Learnability.yaml"
    ).resolve()
    leaf_actual = _sha256_file(leaf_path)
    if leaf_actual != profile.task_leaf_sha256:
        raise LaunchBlocked(
            f"{profile.label} mirrored task leaf SHA differs: "
            f"expected={profile.task_leaf_sha256} actual={leaf_actual}"
        )
    # 人话:上一道门只说"源文件字节跟我钉的一样",不说"我这份手抄件抄对了"。
    # 源文件一动,把 SHA 重钉成新值是一行的事 —— 5ed998f1 就是这么让 table
    # 复刻停在原地两天的。这道门把手抄的身份串、逐行有序布局、两个宽度和
    # RESET_WAIT 掩码块跟活的叶子再对一遍,所以光重钉 SHA 不再放行。
    parity = abi.live_source_parity_blockers(profile, path)
    if parity:
        raise LaunchBlocked(
            f"{profile.label} mirrored ABI semantics differ from the live Isaac "
            "source (re-pinning the SHA does not port them): " + "; ".join(parity)
        )
    return {
        "repo_relative_path": str(_SOURCE_PATHS[profile.label]),
        "sha256": actual,
        "task_leaf_repo_relative_path": str(leaf_path.relative_to(REPO_ROOT)),
        "task_leaf_sha256": leaf_actual,
        "live_semantic_parity": "exact_identities_ordered_layouts_widths_wait_mask",
        "live_semantic_parity_symbols_compared": str(
            len(abi.MIRRORED_IDENTITY_SYMBOLS) + 6
        ),
    }


def _runtime_module_sha256s() -> dict[str, str]:
    """Seal every local module that can change parent/child update semantics."""

    return {
        "runner": _sha256_file(Path(__file__).resolve()),
        "abi": _sha256_file(Path(abi.__file__).resolve()),
        "c211_env": _sha256_file(Path(action_ball_c211_env.__file__).resolve()),
        "trainer": _sha256_file(Path(trainer.__file__).resolve()),
        "checkpoint": _sha256_file(Path(checkpoint.__file__).resolve()),
        "fixed_center_runner": _sha256_file(Path(fixed_launch.__file__).resolve()),
        "fixed_center_recipe": _sha256_file(
            Path(fixed_launch.fixed_center_recipe.__file__).resolve()
        ),
        "native_vec_env": _sha256_file(
            Path(fixed_launch.fixed_center_recipe.vec_env.__file__).resolve()
        ),
        "native_single_env": _sha256_file(
            Path(fixed_launch.fixed_center_recipe.single_env.__file__).resolve()
        ),
        "table_termination": _sha256_file(
            Path(
                fixed_launch.fixed_center_recipe.vec_env.table_termination.__file__
            ).resolve()
        ),
        "task_wait_schedule": _sha256_file(
            fixed_launch.fixed_center_recipe.TASK_WAIT_SOURCE
        ),
        "reward_event_kernel": _sha256_file(
            Path(action_ball_c211_env.n1_reward_event_kernel.__file__).resolve()
        ),
        "physical_ball_scene": _sha256_file(
            Path(action_ball_c211_env.physical_ball_scene.__file__).resolve()
        ),
        "virtual_ball": _sha256_file(action_ball_c211_env.VIRTUAL_BALL_PY),
    }


def _task_authority(path: Path | None, expected_sha256: str | None) -> dict | None:
    """Lightweight optional immutable-tape check retained for static plan mode."""

    if path is None and expected_sha256 is None:
        return None
    if path is None or expected_sha256 is None:
        raise LaunchBlocked(
            "task question authority path and expected SHA-256 must be supplied together"
        )
    source = path.expanduser().resolve()
    authority = fixed_launch._authority(
        source, expected_sha256, "task question authority"
    )
    try:
        payload = json.loads(source.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaunchBlocked("task question authority is not JSON") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("kind") != "action_ball_n1_immutable_single_question_tape"
        or payload.get("schema_version") != 1
        or payload.get("diagnostic_unauthorized") is not True
        or payload.get("row_count") != 1
    ):
        raise LaunchBlocked("task question authority schema differs")
    return {
        **authority,
        "kind": payload["kind"],
        "schema_version": payload["schema_version"],
        "canonical_sha256": payload.get("canonical_sha256"),
        "question_sha256": payload.get("question_sha256"),
    }


def _static_plan(
    profile: abi.ActionBall211Profile,
    *,
    num_envs: int,
    task_authority: dict | None,
) -> dict[str, Any]:
    construction = abi.construction_receipt(
        profile,
        num_envs=num_envs,
        task_question_sha256=(
            None if task_authority is None else task_authority["sha256"]
        ),
    )
    return {
        "schema_version": 2,
        "kind": STATIC_PLAN_KIND,
        "mode": "plan",
        "profile": profile.label,
        "source_lineage": _source_lineage(profile),
        "task_question_authority": task_authority,
        "construction": construction,
        "claims": {
            "ordered_actor_211_contract_constructed": True,
            "ordered_critic_319_contract_constructed": True,
            "c211_real_vecenv_adapter_implemented": True,
            "a211_real_vecenv_adapter_implemented": True,
            "runtime_tensor_materialized": False,
            "matched_4096_runtime_measured": False,
            "two_update_smoke_executed": False,
            "placeholder_or_zero_padded_columns_used": False,
            "safe_ready_formal_pass_claimed": False,
            "c211_achieved_outcome_reward_implemented": (
                profile.label == "C211"
            ),
            "c211_partial_isaac_synonymous_reward_implemented": (
                profile.label == "C211"
            ),
            "complete_isaac_reward_parity_claimed": False,
            "true_c211_achieved_outcome_reward_available": False,
            "true_c211_training_lane_ready": False,
        },
        "safe_ready_authority_status": (
            action_ball_c211_env.SAFE_READY_AUTHORITY_STATUS
        ),
        "runtime_blockers": construction["blockers"],
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
    }


def _execution_authorities(args: argparse.Namespace) -> dict[str, Any]:
    rows = fixed_launch._authorities(args)
    rows["immutable_task_tape"] = fixed_launch._authority(
        args.immutable_tape,
        args.expected_immutable_tape_sha256,
        "immutable C211 task tape",
    )
    rows["measured_motion"] = fixed_launch._authority(
        args.measured_motion,
        args.expected_measured_motion_sha256,
        "measured C211 motion",
    )
    return rows


def _execution_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile != "C211":
        raise LaunchBlocked(
            "execution is implemented only for the C211 observation provider; "
            "A211 remains plan-only"
        )
    if args.num_envs < 1 or args.required_active_steps < 1 or args.pre_checkpoint_updates != 1:
        raise LaunchBlocked(
            "C211 task-reward smoke requires positive env/active counts and "
            "exactly one pre-checkpoint update"
        )
    if (
        args.reset_wait_min_steps != 5
        or args.reset_wait_max_steps != 25
        or args.reset_wait_seed != 20260804
        or args.episode_horizon_steps != 500
        or args.required_active_steps != 200
    ):
        raise LaunchBlocked(
            "C211 smoke requires the frozen seeded WAIT schedule 20260804/5..25/500/200"
        )
    if args.output_dir is None:
        raise LaunchBlocked("execution output directory is required")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise LaunchBlocked("output directory already exists (no-clobber)")
    if not output.parent.is_dir():
        raise LaunchBlocked("output directory parent does not exist")
    task = action_ball_c211_env.C211TaskAuthority.load(
        args.immutable_tape,
        expected_file_sha256=args.expected_immutable_tape_sha256,
    )
    nominal_tick_float = (
        task.time_to_contact_s
        + args.reset_wait_max_steps * action_ball_c211_env.C211_POLICY_DT_S
    ) / action_ball_c211_env.C211_POLICY_DT_S
    nominal_tick = int(round(nominal_tick_float))
    if nominal_tick < 1 or abs(nominal_tick_float - nominal_tick) > 1.0e-12:
        raise LaunchBlocked("immutable C211 strike is not on one exact policy tick")
    rollout_steps = args.episode_horizon_steps
    maximum_rollout_steps = rollout_steps + args.episode_horizon_steps
    minimum_rollout_steps = nominal_tick + 2
    if rollout_steps < minimum_rollout_steps:
        raise LaunchBlocked(
            "C211 task-reward smoke must reach nominal strike tick "
            f"{nominal_tick} plus the post-strike tail; requires at least "
            f"{minimum_rollout_steps} rollout steps"
        )
    return {
        "schema_version": 1,
        "kind": EXECUTION_PLAN_KIND,
        "mode": "execute",
        "profile": "C211",
        "source_lineage": _source_lineage(abi.C211_PROFILE),
        "authorities": _execution_authorities(args),
        "runtime_module_sha256s": _runtime_module_sha256s(),
        "workload": {
            "num_envs": args.num_envs,
            "minimum_rollout_steps_per_update": rollout_steps,
            "maximum_rollout_steps_per_update": maximum_rollout_steps,
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
            "minimum_post_strike_tail_steps": 2,
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
            "immutable_c211_task_tuple": True,
            "same_motion_sha_fail_closed": True,
            "isaac_equivalent_fresh_actor_hold_bootstrap_required": True,
            "fresh_actor_max_wait_runtime_canary_required": True,
            "separate_actor_critic_normalizers": True,
            "fresh_process_update_2_exact": True,
            "placeholder_or_zero_padded_columns_used": False,
            "reward_scope": action_ball_c211_env.C211_REWARD_SCOPE,
            "reward_contract_identity": (
                action_ball_c211_env.C211_REWARD_CONTRACT_IDENTITY
            ),
            "reward_parity_status": "partial_fail_closed",
            "full_body_mimic_reward_consumed": True,
            "measured_paddle_prior_reward_consumed": True,
            "complete_isaac_reward_parity_claimed": False,
            "unavailable_isaac_reward_terms": [
                dict(row)
                for row in action_ball_c211_env.C211_UNAVAILABLE_ISAAC_REWARD_TERMS
            ],
            "cross_engine_reward_semantic_gaps": [
                dict(row)
                for row in action_ball_c211_env.C211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS
            ],
            "true_c211_achieved_outcome_reward_available": True,
            "true_c211_training_lane_ready": False,
            "safe_ready_formal_pass_claimed": False,
            "incoming_question_parity": False,
            "reset_boundary_resume_only": True,
            "mid_episode_resume": False,
            "formal_training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
        "safe_ready_authority_status": (
            action_ball_c211_env.SAFE_READY_AUTHORITY_STATUS
        ),
        "formal_blockers": list(action_ball_c211_env.FORMAL_BLOCKERS),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }


def _build_env(
    args: argparse.Namespace,
    *,
    expected_launch_preparation: Mapping[str, Any] | None = None,
) -> action_ball_c211_env.MujocoC211DiagnosticVecEnv:
    fixed = fixed_launch._build_env(args)
    task = action_ball_c211_env.C211TaskAuthority.load(
        args.immutable_tape,
        expected_file_sha256=args.expected_immutable_tape_sha256,
    )
    mimic = action_ball_c211_env.MeasuredC211MimicAuthority.load(
        args.measured_motion,
        expected_file_sha256=args.expected_measured_motion_sha256,
        task=task,
    )
    env = action_ball_c211_env.MujocoC211DiagnosticVecEnv(
        base_env=fixed,
        task_authority=task,
        mimic_authority=mimic,
    )
    if expected_launch_preparation is not None and (
        _launch_preparation_payload(env) != expected_launch_preparation
    ):
        raise LaunchBlocked("derived C211 launch differs from sealed preparation")
    return env


def _launch_preparation_payload(
    env: action_ball_c211_env.MujocoC211DiagnosticVecEnv,
) -> dict[str, Any]:
    fixed = fixed_launch._launch_preparation_payload(env.base)
    payload = {
        "schema_version": 1,
        "kind": "action_ball_c211_mujoco_launch_preparation_v1",
        "fixed_center_preparation": fixed,
        "task_authority": env.producer.task.receipt,
        "measured_mimic_authority": env.producer.mimic.receipt,
        "c211_reward_contract": copy.deepcopy(env._reward_contract),
        "observation_authorities_sha256": env.producer.authorities.content_sha256,
        "training_identity": env.diagnostic_training_identity(),
        "fresh_actor_bootstrap": env.fresh_actor_bootstrap_contract(),
        "canonical_reset_boundary_sha256": (env._canonical_boundary_sha256),
        "safe_ready_authority_status": (
            action_ball_c211_env.SAFE_READY_AUTHORITY_STATUS
        ),
        "safe_ready_formal_pass_claimed": False,
        "hidden_wait_ball_parked": True,
        "ball_only_atomic_sealed_launch_on_reveal": True,
        "robot_state_continuous_across_reveal": True,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    payload["preparation_content_sha256"] = _sha256_json(payload)
    fixed_launch._finite_tree(payload, "c211_launch_preparation")
    return payload


def _build_trainer(
    env: action_ball_c211_env.MujocoC211DiagnosticVecEnv,
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
        **abi.C211_PROFILE.trainer_config_kwargs(),
    )
    return trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=identity,
        config=config,
    )


def _fresh_wait_bootstrap_canary(
    env: action_ball_c211_env.MujocoC211DiagnosticVecEnv,
    diagnostic_trainer: trainer.MujocoDiagnosticPPOTrainer,
    *,
    profile: str,
) -> dict[str, Any]:
    """Exercise the actual fresh actor through a deterministic and noisy WAIT.

    The 4-sigma limit calculation is only a projection-risk forecast.  This is
    the launch gate: the installed float32 actor mean must exactly equal its
    sealed float32 hold bias, a 25-tick deterministic WAIT must remain legal,
    and a fresh ``std=0.02`` sample stream must expose projection/effort counts
    while producing no hard termination or non-finite transition.
    """

    torch = _torch_import_order_guard
    if torch is None:
        raise LaunchBlocked("fresh WAIT canary requires torch")
    schedule = getattr(env, "_wait_schedule", None)
    if (
        getattr(schedule, "min_wait_ticks", None)
        != FRESH_WAIT_BOOTSTRAP_CANARY_TICKS
        or getattr(schedule, "max_wait_ticks", None)
        != FRESH_WAIT_BOOTSTRAP_CANARY_TICKS
    ):
        raise LaunchBlocked("fresh WAIT canary requires a fixed 25-tick WAIT")
    bootstrap = env.fresh_actor_bootstrap_contract()
    expected_bias = torch.as_tensor(
        bootstrap["output_layer_bias"], dtype=torch.float32, device="cpu"
    )
    output = diagnostic_trainer.model.actor[-1]
    if (
        not isinstance(output, torch.nn.Linear)
        or output.bias is None
        or int(torch.count_nonzero(output.weight).item()) != 0
        or not torch.equal(output.bias.detach().cpu(), expected_bias)
    ):
        raise LaunchBlocked("fresh actor mean is not the sealed float32 hold")

    actor_obs, _reset_extras = env.reset(seed=FRESH_WAIT_BOOTSTRAP_CANARY_SEED)
    if (
        not isinstance(actor_obs, torch.Tensor)
        or tuple(actor_obs.shape) != (env.num_envs, env.num_observations)
        or not torch.isfinite(actor_obs).all()
    ):
        raise LaunchBlocked("fresh actor reset observation differs")
    # The complete final affine map has an exact-zero weight matrix, so its
    # output is algebraically the installed bias for every finite observation.
    # Avoiding a redundant BLAS call also keeps this safety preflight immune to
    # host-specific OpenMP initialization failures.
    installed_mean = output.bias.detach().cpu().expand(env.num_envs, -1)
    expected_mean = expected_bias.expand(env.num_envs, -1)
    if not torch.equal(installed_mean, expected_mean):
        raise LaunchBlocked("fresh actor output is not exactly its sealed hold bias")
    tape_bias = np.asarray(
        env.producer.robot_tape.history_fill_action, dtype=np.float64
    )
    actor_mean_np = installed_mean[0].detach().cpu().numpy().astype(np.float64)
    if not np.array_equal(
        actor_mean_np.astype(np.float32), tape_bias.astype(np.float32)
    ):
        raise LaunchBlocked("fresh actor hold differs at the execution dtype")
    installed_std = torch.exp(diagnostic_trainer.model.log_std.detach().cpu())
    expected_std = torch.full_like(installed_std, 0.02)
    if not torch.allclose(
        installed_std, expected_std, rtol=1.0e-6, atol=0.0
    ):
        raise LaunchBlocked("fresh actor installed std differs from 0.02")
    mean_projection_counts = []
    decoded_qdes_deltas = []
    for index, core in enumerate(env._native.cores):
        _raw, applied_mean, projection_count = core.binding.decode_action(actor_mean_np)
        _tape_raw, applied_tape, tape_projection_count = core.binding.decode_action(
            tape_bias
        )
        if projection_count != 0:
            raise LaunchBlocked(
                f"fresh actor mean requires qdes projection in env {index}"
            )
        if tape_projection_count != 0:
            raise LaunchBlocked(
                f"sealed tape hold requires qdes projection in env {index}"
            )
        qdes_delta = float(np.max(np.abs(applied_mean - applied_tape)))
        if qdes_delta > FRESH_WAIT_BOOTSTRAP_MAX_QDES_JUMP_RAD:
            raise LaunchBlocked(
                f"fresh actor/tape hold qdes jump exceeds tolerance in env {index}: "
                f"{qdes_delta}"
            )
        delay_state = np.asarray(core.plant.delay.state(), dtype=np.float64)
        if (
            delay_state.ndim != 2
            or delay_state.shape[1] != env.num_actions
            or not np.array_equal(
                delay_state,
                np.broadcast_to(tape_bias, delay_state.shape),
            )
        ):
            raise LaunchBlocked(
                f"fresh actor delay history is not the sealed hold in env {index}"
            )
        mean_projection_counts.append(projection_count)
        decoded_qdes_deltas.append(qdes_delta)

    def run_phase(*, stochastic: bool) -> dict[str, Any]:
        env.reset(seed=FRESH_WAIT_BOOTSTRAP_CANARY_SEED)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(FRESH_WAIT_BOOTSTRAP_CANARY_SEED)
        per_joint_projection = np.zeros(env.num_actions, dtype=np.int64)
        projected_env_transitions = 0
        hard_count = 0
        hard_reasons: dict[str, int] = {}
        promotion_blocked_samples: list[dict[str, Any]] = []
        final_effort_by_env = [0 for _ in range(env.num_envs)]
        completed = 0
        for tick in range(1, FRESH_WAIT_BOOTSTRAP_CANARY_TICKS + 1):
            actions = expected_mean.clone()
            if stochastic:
                actions = actions + torch.randn(
                    actions.shape,
                    dtype=actions.dtype,
                    device="cpu",
                    generator=generator,
                ) * installed_std
            try:
                observations, rewards, dones, extras = env.step(actions)
            except Exception as exc:  # noqa: BLE001 - evidence boundary
                raise LaunchBlocked(
                    f"fresh WAIT {'stochastic' if stochastic else 'mean'} "
                    f"transition failed at tick {tick}: {type(exc).__name__}: {exc}"
                ) from exc
            if (
                not torch.isfinite(observations).all()
                or not torch.isfinite(rewards).all()
            ):
                raise LaunchBlocked("fresh WAIT canary observed a non-finite tensor")
            masks = extras.get("diagnostic_qdes_projection_masks")
            if not isinstance(masks, tuple) or len(masks) != env.num_envs:
                raise LaunchBlocked("fresh WAIT canary lacks qdes projection masks")
            for row in masks:
                mask = np.asarray(row, dtype=np.bool_)
                if mask.shape != (env.num_actions,):
                    raise LaunchBlocked("fresh WAIT projection mask width differs")
                per_joint_projection += mask.astype(np.int64)
                projected_env_transitions += int(bool(np.any(mask)))
            expected_transition_valid = [False] * env.num_envs
            expected_next_valid = [
                tick == FRESH_WAIT_BOOTSTRAP_CANARY_TICKS
            ] * env.num_envs
            if (
                extras.get("task_valid_transition") != expected_transition_valid
                or extras.get("task_valid_next") != expected_next_valid
            ):
                raise LaunchBlocked(
                    f"fresh WAIT reveal timing differs at tick {tick}"
                )
            ledgers = extras.get("diagnostic_event_ledgers")
            if not isinstance(ledgers, tuple) or len(ledgers) != env.num_envs:
                raise LaunchBlocked("fresh WAIT canary lacks event ledgers")
            final_effort_by_env = []
            for ledger in ledgers:
                counters = ledger.get("plant_counters") if isinstance(ledger, Mapping) else None
                value = (
                    counters.get("effort_clip_joint_events")
                    if isinstance(counters, Mapping)
                    else None
                )
                if type(value) is not int or value < 0:
                    raise LaunchBlocked("fresh WAIT effort-clip counter differs")
                final_effort_by_env.append(value)
            # 人话:joint_actual_forbidden 改成"只记录不 reset"之后,它就不再进
            # hard_termination_count 了。这条 canary 的 passed 原本只看 hard,于是
            # "起手姿态贴着关节硬边"会静默通过。这里把 ledger 的结论位接进来补上后半句。
            promotion_blocked_samples.extend(
                trainer.promotion_blocking_samples_from_step(
                    extras=extras,
                    rollout_step_1based=tick,
                    num_envs=env.num_envs,
                )
            )
            hard = extras.get("diagnostic_exact_hard_terminations")
            reasons = extras.get("diagnostic_exact_hard_termination_reasons")
            if (
                not isinstance(hard, torch.Tensor)
                or tuple(hard.shape) != (env.num_envs,)
                or not isinstance(reasons, list)
                or len(reasons) != env.num_envs
            ):
                raise LaunchBlocked("fresh WAIT hard-termination evidence differs")
            for index, value in enumerate(hard.tolist()):
                if value:
                    hard_count += 1
                    reason = str(reasons[index])
                    hard_reasons[reason] = hard_reasons.get(reason, 0) + 1
            if bool(torch.any(dones).item()) and not bool(torch.any(hard).item()):
                raise LaunchBlocked("fresh WAIT ended before its 25-tick reveal")
            completed = tick
            if hard_count:
                break
        reset_observations, reset_extras = env.reset(
            seed=FRESH_WAIT_BOOTSTRAP_CANARY_SEED
        )
        reset_legal = bool(
            env.is_reset_boundary()
            and torch.isfinite(reset_observations).all()
            and reset_extras.get("task_valid") == [False] * env.num_envs
        )
        denominator = completed * env.num_envs
        promotion_evidence = trainer.promotion_blocking_evidence_receipt(
            promotion_blocked_samples,
            checked_sample_count=denominator,
        )
        return {
            "mode": "fresh_std_0p02" if stochastic else "sealed_mean",
            "requested_wait_ticks": FRESH_WAIT_BOOTSTRAP_CANARY_TICKS,
            "completed_wait_ticks": completed,
            "env_count": env.num_envs,
            "env_transition_denominator": denominator,
            "joint_projection_denominator": denominator * env.num_actions,
            "per_joint_projection_counts": per_joint_projection.tolist(),
            "per_joint_projection_rates": (
                (per_joint_projection / denominator).tolist()
                if denominator
                else [0.0] * env.num_actions
            ),
            "projected_env_transition_count": projected_env_transitions,
            "projected_env_transition_rate": (
                projected_env_transitions / denominator if denominator else 0.0
            ),
            "projected_joint_event_count": int(np.sum(per_joint_projection)),
            "projected_joint_event_rate": (
                float(np.sum(per_joint_projection))
                / float(denominator * env.num_actions)
                if denominator
                else 0.0
            ),
            "effort_clip_joint_events_by_env": final_effort_by_env,
            "effort_clip_joint_event_count": int(sum(final_effort_by_env)),
            "hard_termination_count": hard_count,
            "hard_termination_reasons": hard_reasons,
            "nonfinite_transition_count": 0,
            "reset_legal_after_phase": reset_legal,
            "promotion_blocking_evidence": promotion_evidence,
            "passed": bool(
                completed == FRESH_WAIT_BOOTSTRAP_CANARY_TICKS
                and hard_count == 0
                and reset_legal
                and promotion_evidence["promotion_blocked"] is False
            ),
        }

    deterministic = run_phase(stochastic=False)
    stochastic = run_phase(stochastic=True)
    receipt = {
        "schema_version": 1,
        "kind": FRESH_WAIT_BOOTSTRAP_CANARY_KIND,
        "profile": profile,
        "seed": FRESH_WAIT_BOOTSTRAP_CANARY_SEED,
        "actor_execution_dtype": "torch.float32",
        "actor_mean_exactly_sealed_at_execution_dtype": True,
        "first_action_command_discontinuity_claim": (
            "float32_canonical_action_exact_and_decoded_qdes_delta_within_"
            "preregistered_tolerance"
        ),
        "max_abs_float64_tape_to_float32_actor_bias_delta": float(
            np.max(np.abs(actor_mean_np - tape_bias))
        ),
        "max_abs_decoded_actor_mean_to_tape_qdes_delta_rad": float(
            max(decoded_qdes_deltas, default=0.0)
        ),
        "decoded_qdes_jump_tolerance_rad": (
            FRESH_WAIT_BOOTSTRAP_MAX_QDES_JUMP_RAD
        ),
        "installed_action_std": installed_std.tolist(),
        "mean_qdes_projection_counts": mean_projection_counts,
        "deterministic_max_wait": deterministic,
        "stochastic_fresh_wait": stochastic,
        "projection_is_reported_not_a_hidden_hard_gate": True,
        "hard_or_nonfinite_required_zero": True,
        "promotion_blocked": bool(
            deterministic["promotion_blocking_evidence"]["promotion_blocked"]
            or stochastic["promotion_blocking_evidence"]["promotion_blocked"]
        ),
        "passed": bool(deterministic["passed"] and stochastic["passed"]),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    receipt["content_sha256"] = _sha256_json(receipt)
    fixed_launch._finite_tree(receipt, "fresh_wait_bootstrap_canary")
    if receipt["passed"] is not True:
        raise LaunchBlocked(
            "fresh WAIT bootstrap canary failed: "
            + json.dumps(receipt, sort_keys=True, allow_nan=False)
        )
    return receipt


PROMOTION_SUMMARY_KIND = "a3_mujoco_launch_promotion_blocking_summary_v1"


def _promotion_blocking_summary(
    *,
    profile: str,
    canary: Mapping[str, Any],
    update_receipts: Sequence[Mapping[str, Any]],
    checkpoint_save_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Lift the promotion verdict to the top of the launch result, and WARN.

    人话:结论位埋在几千行收据里等于没有。这一格挂在 result 的最外层,并且只在
    "被卡住"时往 stderr 打一行带 WARN 的话——摘要抓异常不抓预期。

    Fail-closed: a source that cannot produce the verdict counts as blocked,
    because "we could not tell" and "not promotable" are the same answer here.
    """

    def _flag(payload: Any) -> bool:
        if not isinstance(payload, Mapping):
            return True
        value = payload.get("promotion_blocked")
        return True if type(value) is not bool else value

    sources: list[dict[str, Any]] = [
        {
            "name": "fresh_wait_bootstrap_canary",
            "promotion_blocked": _flag(canary),
        }
    ]
    for index, receipt in enumerate(update_receipts):
        sources.append(
            {
                "name": f"update_receipt[{index}]",
                "promotion_blocked": trainer.promotion_blocked_from_evidence(
                    receipt.get("promotion_blocking_evidence")
                    if isinstance(receipt, Mapping)
                    else None
                ),
            }
        )
    sources.append(
        {
            "name": "checkpoint_save_receipt",
            "promotion_blocked": _flag(checkpoint_save_receipt),
        }
    )
    blocked_sources = [row["name"] for row in sources if row["promotion_blocked"]]
    summary = {
        "schema_version": 1,
        "kind": PROMOTION_SUMMARY_KIND,
        "profile": profile,
        "promotion_blocked": bool(blocked_sources),
        "blocked_sources": blocked_sources,
        "sources": sources,
        "semantics": (
            "non-terminal hard-edge faults never shorten an episode; any blocked "
            "source bars checkpoint promotion, deployment and hardware use"
        ),
        "absent_verdict_counts_as_blocked": True,
    }
    if summary["promotion_blocked"]:
        print(
            f"[MUJOCO-{profile}] WARN promotion_blocked=True "
            f"blocked_sources={','.join(blocked_sources)} "
            "(non-terminal joint_actual_forbidden evidence; this run may not be "
            "promoted, deployed or run on hardware)",
            file=sys.stderr,
        )
    return summary


def _run_audited_update(
    env: action_ball_c211_env.MujocoC211DiagnosticVecEnv,
    diagnostic_trainer: trainer.MujocoDiagnosticPPOTrainer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one update and retain the raw/per-dt C211 term ledger it consumed."""

    env.reset_reward_audit()
    update_receipt = diagnostic_trainer.run_update()
    reward_audit = env.reward_audit_receipt()
    actual_rollout_steps = update_receipt.get("rollout_steps")
    if type(actual_rollout_steps) is not int or actual_rollout_steps < 1:
        raise LaunchBlocked("C211 update receipt omits its actual rollout length")
    expected_rows = actual_rollout_steps * env.num_envs
    if (
        reward_audit.get("transition_step_count")
        != actual_rollout_steps
        or reward_audit.get("row_count") != expected_rows
        or any(
            row.get("sample_count") != expected_rows
            for row in reward_audit.get("prior_terms", {}).values()
        )
    ):
        raise LaunchBlocked("C211 raw reward audit did not cover the complete update")
    return update_receipt, reward_audit


def _child_args(args: argparse.Namespace) -> dict[str, Any]:
    values = fixed_launch._child_args(args)
    values.update(
        {
            "profile": "C211",
            "immutable_tape": str(args.immutable_tape.expanduser().resolve()),
            "expected_immutable_tape_sha256": (args.expected_immutable_tape_sha256),
            "measured_motion": str(args.measured_motion.expanduser().resolve()),
            "expected_measured_motion_sha256": (args.expected_measured_motion_sha256),
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
        if _sha256_file(checkpoint_path) != request.get("checkpoint_sha256"):
            raise LaunchBlocked("cold-child checkpoint SHA differs")
        preparation_path = Path(request["launch_preparation_path"]).resolve()
        if _sha256_file(preparation_path) != request.get(
            "launch_preparation_file_sha256"
        ):
            raise LaunchBlocked("cold-child launch-preparation file SHA differs")
        sealed_preparation = json.loads(preparation_path.read_text("utf-8"))
        if sealed_preparation.get("preparation_content_sha256") != request.get(
            "preparation_content_sha256"
        ):
            raise LaunchBlocked("cold-child launch-preparation content differs")
        args = _args_from_child(request["args"])
        _execution_authorities(args)
        env = _build_env(args, expected_launch_preparation=sealed_preparation)
        cold = _build_trainer(env, args)
        load_receipt = checkpoint.ResetBoundaryCheckpoint().load(checkpoint_path, cold)
        update_receipt, reward_audit = _run_audited_update(env, cold)
        result = {
            "schema_version": 1,
            "kind": CHILD_RESULT_KIND,
            "pid": os.getpid(),
            "parent_pid": request["parent_pid"],
            "checkpoint_sha256": request["checkpoint_sha256"],
            "runtime_module_sha256s": copy.deepcopy(
                request["runtime_module_sha256s"]
            ),
            "launch_preparation_file_sha256": request["launch_preparation_file_sha256"],
            "preparation_content_sha256": request["preparation_content_sha256"],
            "checkpoint_load_receipt": load_receipt,
            "next_update_receipt": update_receipt,
            "next_update_raw_reward_audit": reward_audit,
            "state_sha256": fixed_launch._state_digest(cold.checkpoint_state()),
            "safe_ready_authority_status": (
                action_ball_c211_env.SAFE_READY_AUTHORITY_STATUS
            ),
            "safe_ready_formal_pass_claimed": False,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        fixed_launch._finite_tree(result, "cold_child_result")
        fixed_launch._write_new_json(result_path.resolve(), result)
        return 0
    except Exception as exc:  # noqa: BLE001 - fresh-process boundary
        print(f"[C211-COLD-FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
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
    canary_args.reset_wait_min_steps = FRESH_WAIT_BOOTSTRAP_CANARY_TICKS
    canary_args.reset_wait_max_steps = FRESH_WAIT_BOOTSTRAP_CANARY_TICKS
    canary_env = _build_env(canary_args)
    canary_trainer = _build_trainer(canary_env, canary_args)
    bootstrap_canary = _fresh_wait_bootstrap_canary(
        canary_env, canary_trainer, profile="C211"
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
    preparation_file_sha = _sha256_file(preparation_path)
    source = _build_trainer(source_env, args)
    pre_checkpoint = []
    pre_checkpoint_reward_audits = []
    for _ in range(args.pre_checkpoint_updates):
        receipt, reward_audit = _run_audited_update(source_env, source)
        if receipt.get("at_reset_boundary") is not True:
            raise LaunchBlocked("C211 update did not end at a reset boundary")
        pre_checkpoint.append(receipt)
        pre_checkpoint_reward_audits.append(reward_audit)
    checkpoint_path = output / "reset_boundary.pt"
    save_receipt = checkpoint.ResetBoundaryCheckpoint().save(checkpoint_path, source)
    reference_update, reference_reward_audit = _run_audited_update(source_env, source)
    reference_state_sha = fixed_launch._state_digest(source.checkpoint_state())
    request = {
        "schema_version": 1,
        "kind": CHILD_REQUEST_KIND,
        "runtime_module_sha256s": _runtime_module_sha256s(),
        "parent_pid": os.getpid(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
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
            "C211",
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
            "C211 cold child failed: " + (completed.stderr.strip() or "no stderr")
        )
    child = json.loads(child_path.read_text("utf-8"))
    if (
        child.get("kind") != CHILD_RESULT_KIND
        or child.get("parent_pid") != os.getpid()
        or child.get("pid") == os.getpid()
    ):
        raise LaunchBlocked("fresh-process C211 child identity differs")
    if child.get("next_update_receipt") != reference_update:
        raise LaunchBlocked("fresh-process C211 next-update receipt differs")
    if child.get("next_update_raw_reward_audit") != reference_reward_audit:
        raise LaunchBlocked("fresh-process C211 raw reward audit differs")
    if child.get("state_sha256") != reference_state_sha:
        raise LaunchBlocked("fresh-process C211 trainer state differs")
    result = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "status": "C211_PARTIAL_ISAAC_REWARD_CHECKPOINT_DIAGNOSTIC_COMPLETE",
        "promotion_blocking_evidence": _promotion_blocking_summary(
            profile="C211",
            canary=bootstrap_canary,
            update_receipts=[*pre_checkpoint, reference_update],
            checkpoint_save_receipt=save_receipt,
        ),
        "plan": copy.deepcopy(dict(plan)),
        "pre_checkpoint_update_receipts": pre_checkpoint,
        "pre_checkpoint_raw_reward_audits": pre_checkpoint_reward_audits,
        "checkpoint_save_receipt": save_receipt,
        "checkpoint_sha256": request["checkpoint_sha256"],
        "launch_preparation_path": str(preparation_path),
        "launch_preparation_file_sha256": preparation_file_sha,
        "preparation_content_sha256": preparation["preparation_content_sha256"],
        "matched_next_update_receipt": reference_update,
        "matched_next_update_raw_reward_audit": reference_reward_audit,
        "matched_state_sha256": reference_state_sha,
        "cold_child_pid": child["pid"],
        "fresh_process_cold_load_exact": True,
        "fresh_process_update_2_exact": True,
        "fresh_wait_bootstrap_canary": bootstrap_canary,
        "fresh_wait_bootstrap_canary_path": str(bootstrap_canary_path),
        "fresh_wait_bootstrap_canary_file_sha256": _sha256_file(
            bootstrap_canary_path
        ),
        "actor_width": abi.ACTOR_WIDTH,
        "critic_width": abi.CRITIC_WIDTH,
        "safe_ready_authority_status": (
            action_ball_c211_env.SAFE_READY_AUTHORITY_STATUS
        ),
        "safe_ready_formal_pass_claimed": False,
        "reward_scope": action_ball_c211_env.C211_REWARD_SCOPE,
        "reward_contract_identity": (
            action_ball_c211_env.C211_REWARD_CONTRACT_IDENTITY
        ),
        "reward_parity_status": "partial_fail_closed",
        "full_body_mimic_reward_consumed": True,
        "measured_paddle_prior_reward_consumed": True,
        "complete_isaac_reward_parity_claimed": False,
        "unavailable_isaac_reward_terms": [
            dict(row)
            for row in action_ball_c211_env.C211_UNAVAILABLE_ISAAC_REWARD_TERMS
        ],
        "cross_engine_reward_semantic_gaps": [
            dict(row)
            for row in action_ball_c211_env.C211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS
        ],
        "true_c211_achieved_outcome_reward_available": True,
        "true_c211_training_lane_ready": False,
        "formal_blockers": list(action_ball_c211_env.FORMAL_BLOCKERS),
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }
    fixed_launch._finite_tree(result, "c211_result")
    fixed_launch._write_new_json(output / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(abi.PROFILES), required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--task-question-authority", type=Path)
    parser.add_argument("--expected-task-question-sha256")
    for name in (
        "plant-contract",
        "robot-tape",
        "question",
        "selected-rubber-manifest",
        "mjcf",
        "immutable-tape",
        "measured-motion",
    ):
        parser.add_argument(f"--{name}", type=Path)
    for name in (
        "expected-plant-sha256",
        "expected-robot-tape-sha256",
        "expected-question-sha256",
        "expected-selected-rubber-manifest-sha256",
        "expected-mjcf-sha256",
        "expected-immutable-tape-sha256",
        "expected-measured-motion-sha256",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--phase-fidelity-reference-tape", type=Path)
    parser.add_argument("--expected-phase-fidelity-reference-tape-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reset-wait-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--active-steps", type=int, default=200, help=argparse.SUPPRESS)
    parser.add_argument("--reset-wait-min-steps", type=int, default=5)
    parser.add_argument("--reset-wait-max-steps", type=int, default=25)
    parser.add_argument("--reset-wait-seed", type=int, default=20260804)
    parser.add_argument("--episode-horizon-steps", type=int, default=500)
    parser.add_argument("--required-active-steps", type=int, default=200)
    parser.add_argument("--pre-checkpoint-updates", type=int, default=1)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(64, 64))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--initial-action-std", type=float, default=0.02)
    parser.add_argument("--execute-two-updates", action="store_true")
    parser.add_argument("--confirm-diagnostic-unauthorized", action="store_true")
    parser.add_argument("--_cold-child", nargs=2, metavar=("REQUEST", "RESULT"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.profile == "A211" and (
        args.execute_two_updates or args._cold_child is not None
    ):
        # Lazy import avoids a module cycle: the A runner deliberately reuses
        # this module's sealed source/authority helpers and static plan.
        from hope_training.whole_body_tracking.mujoco_native.scripts import (
            launch_mujoco_action_ball_a211_diagnostic as a211_launch,
        )

        return a211_launch.main(argv)
    if args._cold_child is not None:
        return _cold_child(Path(args._cold_child[0]), Path(args._cold_child[1]))
    try:
        profile = abi.PROFILES[args.profile]
        if args.execute_two_updates:
            plan = _execution_plan(args)
            result = _execute(args, plan)
        else:
            task = _task_authority(
                args.task_question_authority,
                args.expected_task_question_sha256,
            )
            result = _static_plan(
                profile,
                num_envs=args.num_envs,
                task_authority=task,
            )
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI failure boundary
        print(f"[MUJOCO-211-BLOCKED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
