"""Evaluator-owned schema-v3 BankExam companion leg for Isaac.

This entry point restores the trusted ``params/env.pkl``/``agent.pkl`` from one training run,
verifies the human-readable termination and schema-v3 checkpoint contracts, applies a versioned
nominal evaluation profile, and assigns exactly one immutable exam question to each environment.
The training command continues to load only its saved train split; the exam split is loaded and
installed through explicit runtime seams after the one nominal-stand reset.

Example (one diagnostic canary cell; historical checkpoints require the explicit inexact flag)::

    hope_isaac_py scripts/isaac_bank_exam.py task=HOPEPingPongVirtualBall headless=true \
      +run_dir=/workspace/.../run checkpoint=/workspace/.../run/model_16999.pt \
      +exam_bank=/workspace/.../s1_v4rg_v3_exam.npz +per_clip_quota=10 \
      +schedule_seed=0 +noise_scale=0.0 +allow_inexact_contract=true \
      +output_dir=/workspace/.../judge/schema3_isaac_canary
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaac_bank_exam_adapter import (  # noqa: E402
    IsaacBankExamError,
    apply_hold_aware_tracking_guard_profile,
    apply_nominal_eval_profile,
    canonical_sha256,
    configure_runtime_motion_loader,
    configure_runtime_train_bank_loader,
    finite_or_none,
    per_attempt_action_noise,
    policy_observation_tensor,
    ready_state_sha256,
    sha256_file,
    summarize_attempts,
    write_scorecard,
)


CLIP_NAMES = ("forehand", "backhand")
PHYSICAL_TERMS = ("base_fell_tilt", "base_too_low")
GUARD_TERMS = ("anchor_pos", "anchor_ori", "ee_body_pos")
TIMEOUT_TERM = "time_out"


def _cfg(cfg, key: str, default=None):
    value = cfg.get(key, default)
    return default if value is None else value


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _motion_files(motion_cmd) -> list[str]:
    raw = motion_cmd.cfg.motion_file
    return [str(raw)] if isinstance(raw, str) else [str(value) for value in raw]


def _strike_phases(racket_cmd, count: int) -> list[float]:
    raw = tuple(float(value) for value in racket_cmd.cfg.strike_phase_per_clip)
    if raw:
        if len(raw) != count:
            raise IsaacBankExamError(
                f"strike_phase_per_clip has {len(raw)} entries for {count} clips"
            )
        return list(raw)
    return [float(racket_cmd.cfg.strike_phase)] * count


def _artifact_paths(cfg, run_dir: Path) -> tuple[Path, Path, Path]:
    root = Path(str(_cfg(cfg, "output_dir", run_dir / "judge" / "isaac_bank_exam"))).expanduser()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = str(_cfg(cfg, "output_stem", "isaac_bank_exam"))
    return root / f"{stem}.json", root / f"{stem}.csv", root / f"{stem}.schedule.json"


def _load_saved_run(cfg, torch):
    run_dir = Path(str(_cfg(cfg, "run_dir", ""))).expanduser().resolve()
    if not run_dir.is_dir():
        raise IsaacBankExamError(f"run_dir is not a directory: {run_dir}")
    checkpoint = Path(str(_cfg(cfg, "checkpoint", ""))).expanduser().resolve()
    if not checkpoint.is_file():
        raise IsaacBankExamError(f"checkpoint does not exist: {checkpoint}")
    try:
        checkpoint.relative_to(run_dir)
    except ValueError as exc:
        raise IsaacBankExamError(
            f"checkpoint {checkpoint} is outside its trusted run_dir {run_dir}"
        ) from exc

    params = run_dir / "params"
    required = {
        "env_pickle": params / "env.pkl",
        "agent_pickle": params / "agent.pkl",
        "env_yaml": params / "env.yaml",
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise IsaacBankExamError("saved run is missing required artifacts: " + ", ".join(missing))
    with open(required["env_pickle"], "rb") as stream:
        env_cfg = pickle.load(stream)  # noqa: S301 -- user-authorized trusted training artifact
    with open(required["agent_pickle"], "rb") as stream:
        agent_cfg = pickle.load(stream)  # noqa: S301 -- same trusted run contract
    checkpoint_sha_before = sha256_file(checkpoint)
    checkpoint_obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_sha_after = sha256_file(checkpoint)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise IsaacBankExamError(
            "checkpoint changed while its contract payload was being loaded: "
            f"before={checkpoint_sha_before}, after={checkpoint_sha_after}"
        )
    return (
        run_dir,
        checkpoint,
        checkpoint_sha_after,
        required,
        env_cfg,
        agent_cfg,
        checkpoint_obj,
    )


def _contract_preflight(
    *, run_dir: Path, required: dict[str, Path], env_cfg: Any, checkpoint_obj: Any,
    allow_inexact: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, list[str]]:
    from termination_contract import (
        build_termination_contract,
        verify_runtime_env_cfg,
        verify_termination_contract,
    )

    termination = build_termination_contract(required["env_yaml"])
    verify_termination_contract(termination, env_yaml=required["env_yaml"])
    # Deliberately before gym.make: a stale env.pkl must fail before it can launch Isaac.
    verify_runtime_env_cfg(termination, env_cfg)

    contract_path = run_dir / "params" / "training_contract.json"
    contract = None
    contract_sha = None
    inexact_reasons: list[str] = []
    try:
        if not contract_path.is_file():
            raise ValueError(f"missing {contract_path}")
        contract_bytes = contract_path.read_bytes()
        contract = json.loads(contract_bytes.decode("utf-8"))
        from whole_body_tracking.utils.training_contract import (
            require_checkpoint_contract_binding,
            validate_schema3_contract,
        )

        validate_schema3_contract(contract)
        contract_sha = hashlib.sha256(contract_bytes).hexdigest()
        require_checkpoint_contract_binding(
            checkpoint_obj,
            schema=int(contract["schema_version"]),
            sha256=contract_sha,
            require_lineage_exact=True,
        )
    except (OSError, ValueError) as exc:
        if not allow_inexact:
            raise IsaacBankExamError(
                f"checkpoint is not exact schema-v3 lineage: {exc}; historical canaries require "
                "+allow_inexact_contract=true and remain non-bookable"
            ) from exc
        inexact_reasons.append(f"training/checkpoint contract: {exc}")
    return termination, contract, contract_sha, inexact_reasons


def _term_masks(termination_manager, names: tuple[str, ...], num_envs: int) -> dict[str, list[bool]]:
    masks = {}
    for name in names:
        value = termination_manager.get_term(name)
        if not hasattr(value, "numel") or int(value.numel()) != num_envs:
            raise IsaacBankExamError(
                f"termination term {name!r} has invalid shape {getattr(value, 'shape', None)}"
            )
        masks[name] = value.detach().to("cpu").bool().tolist()
    return masks


def _run(cfg, simulation_app):
    import torch
    import gymnasium as gym

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import whole_body_tracking.tasks  # noqa: F401 -- gym registrations
    from whole_body_tracking.tasks.tracking.actor_observation_contract import (
        infer_actor_observation_contract,
    )
    from whole_body_tracking.tasks.tracking.mdp.stage1_question_bank import (
        load_question_bank,
        validate_runtime_motion_contract,
    )
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner
    from whole_body_tracking.utils.training_contract import (
        RUNTIME_EXECUTION_KEYS,
        runtime_execution_facts,
    )

    allow_inexact = bool(_cfg(cfg, "allow_inexact_contract", False))
    (
        run_dir,
        checkpoint,
        checkpoint_sha,
        required,
        env_cfg,
        agent_cfg,
        checkpoint_obj,
    ) = _load_saved_run(cfg, torch)
    termination_contract, training_contract, training_contract_sha, inexact_reasons = _contract_preflight(
        run_dir=run_dir,
        required=required,
        env_cfg=env_cfg,
        checkpoint_obj=checkpoint_obj,
        allow_inexact=allow_inexact,
    )
    contract_path = run_dir / "params" / "training_contract.json"
    if training_contract_sha is not None and sha256_file(contract_path) != training_contract_sha:
        raise IsaacBankExamError("training contract changed immediately after preflight")

    exam_bank_path = Path(str(_cfg(cfg, "exam_bank", ""))).expanduser().resolve()
    if not exam_bank_path.is_file():
        raise IsaacBankExamError(f"schema-v3 exam bank does not exist: {exam_bank_path}")
    quota = int(_cfg(cfg, "per_clip_quota", 10))
    if quota <= 0:
        raise IsaacBankExamError("per_clip_quota must be positive")
    num_envs = len(CLIP_NAMES) * quota
    device = str(_cfg(cfg, "device", "cuda:0"))
    eval_seed = int(_cfg(cfg, "schedule_seed", 0))
    noise_scale = float(_cfg(cfg, "noise_scale", 0.0))
    if not np.isfinite(noise_scale) or noise_scale < 0.0:
        raise IsaacBankExamError("noise_scale must be finite and non-negative")

    profile = apply_nominal_eval_profile(env_cfg, num_envs=num_envs)
    motion_loader_profile = configure_runtime_motion_loader(
        env_cfg, allow_legacy_diagnostic=allow_inexact
    )
    profile["runtime_motion_loader"] = motion_loader_profile
    if motion_loader_profile["legacy_override"]:
        inexact_reasons.append(
            "saved runtime motions use untagged link-origin finite-difference velocities; "
            "diagnostic replay preserves their historical command semantics"
        )
    train_bank_loader_profile = configure_runtime_train_bank_loader(
        env_cfg, allow_legacy_diagnostic=allow_inexact
    )
    profile["runtime_train_bank_loader"] = train_bank_loader_profile
    if train_bank_loader_profile["legacy_override"]:
        inexact_reasons.append(
            "saved runtime train bank loader was explicitly opened for a legacy diagnostic bank"
        )
    from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
        bad_anchor_ori_hold_aware,
        bad_anchor_pos_z_only_hold_aware,
        bad_motion_body_pos_z_only_hold_aware,
    )

    guard_profile = apply_hold_aware_tracking_guard_profile(
        env_cfg,
        hold_aware_functions={
            "anchor_pos": bad_anchor_pos_z_only_hold_aware,
            "anchor_ori": bad_anchor_ori_hold_aware,
            "ee_body_pos": bad_motion_body_pos_z_only_hold_aware,
        },
        allow_legacy_override=allow_inexact,
    )
    profile.pop("sha256", None)
    profile["tracking_guard_profile"] = guard_profile
    profile["sha256"] = canonical_sha256(profile)
    if guard_profile["overridden_terms"]:
        inexact_reasons.append(
            "legacy tracking guards were evaluator-overridden to ignore nominal ready-stand holds: "
            + ",".join(guard_profile["overridden_terms"])
        )
    env_cfg.seed = eval_seed
    env_cfg.sim.device = device
    if hasattr(agent_cfg, "device"):
        agent_cfg.device = device
    if hasattr(agent_cfg, "seed"):
        agent_cfg.seed = eval_seed

    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)

    task_id = str(cfg.task.gym_task)  # registry entry only; no task YAML is reapplied
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    actor_contract = infer_actor_observation_contract(base)
    runtime_facts = runtime_execution_facts(base, actor_contract)
    history_lengths = tuple(int(value) for value in runtime_facts["observation_history_lengths"])
    if not history_lengths or any(value != 1 for value in history_lengths):
        raise IsaacBankExamError(
            "BankExam v1 observation refresh requires one current sample per actor term; "
            f"history lengths are {history_lengths}"
        )
    if training_contract is not None:
        mismatches = {
            key: (training_contract.get(key), runtime_facts.get(key))
            for key in RUNTIME_EXECUTION_KEYS
            if training_contract.get(key) != runtime_facts.get(key)
        }
        if mismatches:
            message = f"runtime execution facts differ from training contract: {mismatches}"
            if not allow_inexact:
                raise IsaacBankExamError(message)
            inexact_reasons.append(message)

    motion_cmd = base.command_manager.get_term("motion")
    racket_cmd = base.command_manager.get_term("racket_target")
    if not (
        bool(getattr(racket_cmd.cfg, "virtual_ball", False))
        or bool(getattr(racket_cmd.cfg, "vb_metrics_only", False))
    ):
        raise IsaacBankExamError(
            "Isaac BankExam requires virtual_ball or vb_metrics_only scoring in the saved task"
        )
    motion_files = _motion_files(motion_cmd)
    segment_lengths = [int(value) for value in motion_cmd.motion.seg_len.tolist()]
    strike_phases = _strike_phases(racket_cmd, len(motion_files))
    bank_sha_before = sha256_file(exam_bank_path)
    exam_bank = load_question_bank(
        str(exam_bank_path), device=device, clip_names=CLIP_NAMES,
        allow_legacy=False, expected_split="exam",
    )
    bank_sha_after = sha256_file(exam_bank_path)
    if bank_sha_before != bank_sha_after:
        raise IsaacBankExamError(
            "schema-v3 exam bank changed while its validated arrays were being loaded: "
            f"before={bank_sha_before}, after={bank_sha_after}"
        )
    validate_runtime_motion_contract(
        exam_bank.metadata, motion_files, segment_lengths, strike_phases
    )
    if len(exam_bank.counts) != len(CLIP_NAMES):
        raise IsaacBankExamError("BankExam v1 requires exactly forehand/backhand clips")

    family_sha = str(exam_bank.metadata.get("source_family_sha256", ""))
    trained_bank = (training_contract or {}).get("question_bank")
    runtime_train_bank = getattr(racket_cmd, "_question_bank", None)
    runtime_train_meta = getattr(runtime_train_bank, "metadata", None)
    if isinstance(runtime_train_meta, dict) and runtime_train_meta.get("schema_version") == 3:
        if (
            runtime_train_meta.get("split") != "train"
            or runtime_train_meta.get("source_family_sha256") != family_sha
        ):
            raise IsaacBankExamError(
                "runtime schema-v3 train bank is not the exam bank's source family"
            )
        runtime_train_path = Path(str(getattr(runtime_train_bank, "source_path", ""))).resolve()
        if not runtime_train_path.is_file():
            raise IsaacBankExamError(
                f"runtime schema-v3 train bank source is missing: {runtime_train_path}"
            )
        if isinstance(trained_bank, dict) and trained_bank.get("exact") is True:
            runtime_train_sha = sha256_file(runtime_train_path)
            if runtime_train_sha != trained_bank.get("sha256"):
                raise IsaacBankExamError(
                    "runtime schema-v3 train bank SHA differs from the checkpoint contract"
                )
    elif allow_inexact:
        inexact_reasons.append(
            "saved runtime train bank is legacy/unbound; exam family is constrained by the "
            "saved motion SHA/order/frame contract only"
        )
    else:
        raise IsaacBankExamError(
            "formal BankExam requires the saved runtime command to load a schema-v3 train bank"
        )
    if not (
        isinstance(trained_bank, dict)
        and trained_bank.get("exact") is True
        and trained_bank.get("split") == "train"
        and trained_bank.get("source_family_sha256") == family_sha
    ):
        message = "exam source family is not exact-bound to the checkpoint's train split"
        if not allow_inexact:
            raise IsaacBankExamError(message)
        inexact_reasons.append(message)

    from bank_exam_schedule import (  # standalone; venue_ball_sampler.py stays hash-stable
        HOLD_SEMANTICS,
        artifact_document,
        derive_question_bank_question_ids,
        load_schedule_artifact,
        materialize_balanced_bank_exam_schedule,
        write_schedule_artifact,
    )

    bank_sha = bank_sha_after
    question_ids = derive_question_bank_question_ids(exam_bank, bank_sha256=bank_sha)
    output_json, output_csv, schedule_path = _artifact_paths(cfg, run_dir)
    supplied_schedule = _cfg(cfg, "schedule_json", None)
    if supplied_schedule:
        schedule_artifact = load_schedule_artifact(
            supplied_schedule,
            expected_bank_sha256=bank_sha,
            expected_clip_names=CLIP_NAMES,
            expected_question_ids=question_ids,
        )
        if schedule_artifact.per_clip_quota != quota:
            raise IsaacBankExamError(
                f"schedule quota={schedule_artifact.per_clip_quota}, requested {quota}"
            )
    else:
        schedule_artifact = materialize_balanced_bank_exam_schedule(
            bank_sha256=bank_sha,
            clip_names=CLIP_NAMES,
            question_ids=question_ids,
            hold_range=tuple(int(value) for value in _cfg(cfg, "hold_steps_range", (0, 100))),
            schedule_seed=eval_seed,
            per_clip_quota=quota,
        )
        write_schedule_artifact(schedule_artifact, schedule_path)
    schedule = schedule_artifact.items

    env = RslRlVecEnvWrapper(env)
    runner = MotionOnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=device, registry_name=None
    )
    if sha256_file(checkpoint) != checkpoint_sha:
        raise IsaacBankExamError("checkpoint changed after contract preflight and before runner load")
    if inexact_reasons:
        from whole_body_tracking.utils.ckpt_compat import load_actor_tolerant

        load_actor_tolerant(runner, str(checkpoint))
    else:
        runner.load(str(checkpoint))
    if sha256_file(checkpoint) != checkpoint_sha:
        raise IsaacBankExamError("checkpoint changed while the policy runner was loading it")
    policy = runner.get_inference_policy(device=base.device)
    std_vec = None
    for name in ("std", "action_std"):
        value = getattr(runner.alg.policy, name, None)
        if value is not None:
            std_vec = value.detach().reshape(-1).to(device)
            break
    if noise_scale > 0.0 and std_vec is None:
        raise IsaacBankExamError("noise_scale>0 but the checkpoint exposes no learned action std")

    max_steps = max(
        int(item.hold_steps) + segment_lengths[int(item.clip)] + 3 for item in schedule
    )
    requested_cap = int(_cfg(cfg, "steps", 0))
    if requested_cap > 0:
        max_steps = requested_cap
    action_dim = int(base.action_manager.total_action_dim)
    if action_dim <= 0:
        raise IsaacBankExamError(f"invalid runtime action dimension {action_dim}")
    noise = per_attempt_action_noise(schedule, n_steps=max_steps, action_dim=action_dim)
    noise_t = torch.as_tensor(noise, dtype=torch.float32, device=device)

    with torch.inference_mode():
        env.reset()
    robot = base.scene["robot"]
    origins = base.scene.env_origins.detach().to("cpu").numpy()
    root = robot.data.root_state_w.detach().to("cpu").numpy().copy()
    root[:, :3] -= origins
    joints = robot.data.joint_pos.detach().to("cpu").numpy()
    joint_vel = robot.data.joint_vel.detach().to("cpu").numpy()
    ready_hashes = [
        ready_state_sha256(root[index], joints[index], joint_vel[index])
        for index in range(num_envs)
    ]
    if len(set(ready_hashes)) != 1:
        raise IsaacBankExamError(
            f"nominal stand reset produced {len(set(ready_hashes))} distinct ready-state hashes"
        )
    ready_sha = ready_hashes[0]

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    clip_ids = torch.as_tensor([int(item.clip) for item in schedule], device=device)
    bank_rows = torch.as_tensor([int(item.bank_row) for item in schedule], device=device)
    holds = torch.as_tensor([int(item.hold_steps) for item in schedule], device=device)
    motion_cmd.install_external_exam_timing(env_ids, clip_ids, holds)
    racket_cmd.install_external_exam_questions(env_ids, exam_bank, clip_ids, bank_rows)
    with torch.inference_mode():
        obs = policy_observation_tensor(env.get_observations(), device=device)

    records = []
    for env_id, item in enumerate(schedule):
        records.append({
            "schedule_index": int(item.schedule_index),
            "env_id": env_id,
            "clip": int(item.clip),
            "side": CLIP_NAMES[int(item.clip)],
            "bank_row": int(item.bank_row),
            "question_id": str(item.question_id),
            "repeat": int(item.repeat),
            "hold_steps": int(item.hold_steps),
            "attempt_seed": int(item.attempt_seed),
            "ready_state_sha256": ready_sha,
            "start_step": 0,
            "end_step": None,
            "finalize_reason": None,
            "finalized": False,
            "censored": False,
            "physical_fall": False,
            "guard_reset": False,
            "reached_exact": False,
            "hit": False,
            "returned": False,
            "pos_error_m": None,
            "vel_error_mps": None,
            "normal_error_deg": None,
            "landing_x": None,
            "landing_y": None,
            "net_clear": False,
        })
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    tm = base.termination_manager
    active_names = set(tm.active_terms)
    required_terms = {*PHYSICAL_TERMS, TIMEOUT_TERM}
    if not required_terms.issubset(active_names):
        raise IsaacBankExamError(
            f"formal evaluator requires {sorted(required_terms)}, active={sorted(active_names)}"
        )
    unexpected = active_names.difference({*PHYSICAL_TERMS, *GUARD_TERMS, TIMEOUT_TERM})
    if unexpected:
        raise IsaacBankExamError(
            f"formal evaluator cannot classify active termination terms {sorted(unexpected)}"
        )
    observed_terms = tuple(name for name in (*PHYSICAL_TERMS, *GUARD_TERMS, TIMEOUT_TERM)
                           if name in active_names)

    step = 0
    while simulation_app.is_running() and step < max_steps and bool(active.any()):
        with torch.inference_mode():
            mean = policy(obs)
            actions = mean
            if noise_scale > 0.0:
                actions = mean + noise_scale * std_vec * noise_t[:, step, :].to(mean.dtype)
            actions = torch.where(active.unsqueeze(-1), actions, torch.zeros_like(actions))
            held_before_step = motion_cmd.hold_counter > 0
            next_obs, _, dones, _ = env.step(actions.to(base.device))
            released = active & held_before_step & (motion_cmd.hold_counter == 0)
            if bool(released.any()):
                # Training keeps the completed hold snapshot for same-step reward/termination
                # accounting. The next shared-paper policy action must instead see raw frame 0:
                # exactly H stand actions, then frame 0, matching the MuJoCo companion leg.
                motion_cmd.metrics["in_hold"][released] = 0.0
                next_obs = env.get_observations()
            obs = policy_observation_tensor(next_obs, device=device)
        step += 1

        exact = racket_cmd.metrics["exact_strike_hit_rate"].detach() > 0.5
        hit = racket_cmd.vb_fired.detach().bool()
        returned = hit & racket_cmd.vb_net_clear.detach().bool() & racket_cmd.vb_on_opponent.detach().bool()
        for env_id in torch.where(active & exact)[0].detach().to("cpu").tolist():
            row = records[env_id]
            if row["reached_exact"]:
                raise IsaacBankExamError(f"attempt {env_id} produced more than one exact-strike frame")
            row["reached_exact"] = True
            row["hit"] = bool(hit[env_id])
            row["returned"] = bool(returned[env_id])
            row["pos_error_m"] = float(racket_cmd.metrics["racket_pos_error_exact_strike"][env_id])
            row["vel_error_mps"] = float(racket_cmd.metrics["racket_vel_error_exact_strike"][env_id])
            row["normal_error_deg"] = float(
                racket_cmd.metrics["racket_normal_error_deg_exact_strike"][env_id]
            )
            if bool(racket_cmd.vb_landing_valid[env_id]):
                row["landing_x"] = float(racket_cmd.vb_landing_xy[env_id, 0])
                row["landing_y"] = float(racket_cmd.vb_landing_xy[env_id, 1])
            row["net_clear"] = bool(racket_cmd.vb_net_clear[env_id])

        masks = _term_masks(tm, observed_terms, num_envs)
        done_list = dones.detach().to("cpu").bool().tolist()
        wrapped = motion_cmd.just_resampled.detach().to("cpu").bool().tolist()
        for env_id in torch.where(active)[0].detach().to("cpu").tolist():
            physical = any(masks[name][env_id] for name in PHYSICAL_TERMS)
            guards = any(masks[name][env_id] for name in GUARD_TERMS if name in masks)
            timeout = masks[TIMEOUT_TERM][env_id]
            reason = None
            if done_list[env_id]:
                if physical:
                    reason = "physical_fall"
                elif guards:
                    reason = "guard_reset"
                elif timeout:
                    reason = "episode_timeout"
                else:
                    raise IsaacBankExamError(
                        f"env {env_id} ended without a classified termination term"
                    )
            elif wrapped[env_id]:
                reason = "clip_complete"
            if reason is not None:
                row = records[env_id]
                row["end_step"] = step
                row["finalize_reason"] = reason
                row["finalized"] = True
                row["physical_fall"] = bool(physical)
                row["guard_reset"] = bool(guards)
                active[env_id] = False

    if bool(active.any()):
        for env_id in torch.where(active)[0].detach().to("cpu").tolist():
            records[env_id]["end_step"] = step
            records[env_id]["finalize_reason"] = "truncated_external_step_cap"
            records[env_id]["finalized"] = True
            records[env_id]["censored"] = True
        raise IsaacBankExamError(
            f"external step cap {max_steps} truncated schedule indices "
            f"{torch.where(active)[0].detach().to('cpu').tolist()}"
        )

    if sha256_file(checkpoint) != checkpoint_sha:
        raise IsaacBankExamError("checkpoint changed during evaluation; refusing mixed provenance")
    contract_path = run_dir / "params" / "training_contract.json"
    if training_contract_sha is not None and sha256_file(contract_path) != training_contract_sha:
        raise IsaacBankExamError(
            "training contract changed during evaluation; refusing mixed provenance"
        )

    repo = Path(__file__).resolve().parents[2]
    schedule_payload = artifact_document(schedule_artifact)
    metadata = {
        "evaluation_contract_exact": not inexact_reasons,
        "inexact_reasons": inexact_reasons,
        "simulator": "isaac",
        "protocol": "single",
        "noise_scale": noise_scale,
        "schedule": schedule_payload,
        "schedule_sha256": schedule_artifact.schedule_sha256,
        "hold_semantics": HOLD_SEMANTICS,
        "exam_bank": {
            "path": str(exam_bank_path),
            "sha256": bank_sha,
            "source_family_sha256": family_sha,
            "schema_version": 3,
            "split": "exam",
        },
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "training_contract_sha256": training_contract_sha,
        "termination_contract_id": termination_contract["contract_id"],
        "ready_state_sha256": ready_sha,
        "nominal_eval_profile": profile,
        "sources": {
            "git_head": _git_head(repo),
            "evaluator_sha256": sha256_file(__file__),
            "adapter_sha256": sha256_file(Path(__file__).with_name("isaac_bank_exam_adapter.py")),
            "schedule_module_sha256": sha256_file(Path(__file__).with_name("bank_exam_schedule.py")),
            "isaac_scorer_sha256": sha256_file(
                repo / "hope_training/whole_body_tracking/source/whole_body_tracking/"
                "whole_body_tracking/tasks/tracking/mdp/virtual_ball.py"
            ),
            "ball_physics_yaml_sha256": sha256_file(repo / "configs/ball_physics_venue.yaml"),
        },
    }
    document = write_scorecard(
        output_json=output_json,
        output_csv=output_csv,
        metadata=metadata,
        records=records,
        schedule=schedule,
        clip_names=CLIP_NAMES,
    )
    print(json.dumps(finite_or_none(document["summary"]), indent=2, sort_keys=True), flush=True)
    print(f"[isaac-bank-exam] JSON {output_json}", flush=True)
    print(f"[isaac-bank-exam] CSV  {output_csv}", flush=True)
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=False
    )
    simulation_app = launcher.app
    try:
        _run(cfg, simulation_app)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
