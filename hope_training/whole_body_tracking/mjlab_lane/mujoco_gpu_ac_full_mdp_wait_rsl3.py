"""Run upstream RSL-RL 3 on the real MuJoCo portable environment.

The default preserves the historical one-update WAIT ABI. ``--full-a`` exposes
the current R03/R06/R07/Reward20 engineering surface with fail-closed evidence;
it remains diagnostic rather than a promotion or physics authority.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import sys
import time


def _ppo_recipe_module():
    """Load the shared dependency-free recipe from this exact checkout."""

    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "action_ball_full_mdp_ppo_recipe.py"
    )
    name = "_hope_mujoco_action_ball_full_mdp_ppo_recipe"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(cached.__file__).resolve() != source:
            raise RuntimeError("cached FullMDP PPO recipe origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the FullMDP PPO recipe")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


FULL_MDP_PPO_RECIPE = (
    _ppo_recipe_module().ACTION_BALL_FULL_MDP_PPO_RECIPE
)
FULL_MDP_PPO_RECIPE_SHA256 = FULL_MDP_PPO_RECIPE.recipe_sha256()
RSL_RL_VERSION = "3.1.2"
COMPLETION_SCHEMA_VERSION = 3
NUM_STEPS_PER_ENV = FULL_MDP_PPO_RECIPE.num_steps_per_env
READY_POSE_SHA256 = "ab6b7e41ff129f91238835c533c8d589e68cc21f7e6184d639e95d8938d38069"
FULL_A_ACTION_UID = 6907688916670928
FULL_A_MOUNT_NORMAL_SIGN = 1
FULL_A_FAMILY = "forehand"
FULL_A_NUM_ENVS = 4096
FULL_A_NUM_UPDATES = FULL_MDP_PPO_RECIPE.max_iterations
FULL_A_SAVE_INTERVAL = FULL_MDP_PPO_RECIPE.save_interval
RSL_RL_SOURCE_SHA256 = {
    "rsl_rl/runners/on_policy_runner.py": "6ffaee7e154a49ae55eebf53a7b1549f0461a1742d92dd34af5bb4b785d19cf2",
    "rsl_rl/algorithms/ppo.py": "4373ac1b2f9fdf14d9da57516968fc95d8f605d2967fee01dc61bf0d09423478",
    "rsl_rl/modules/actor_critic.py": "614eb6e14d21c46504ce2046f04c9ab70a8c3cf679502ef2a220987e506a959a",
    "rsl_rl/modules/actor_critic_recurrent.py": "19fbe660f6a22a8df4e7d54dc13e6dc6d668f69c71f9303165553b9272aff5d0",
    "rsl_rl/storage/rollout_storage.py": "32d8b1b3cead87e0eeb96e2b334a5c75fd431309e43dae85a42a89b62c5dc5de",
    "rsl_rl/networks/mlp.py": "ead23a9b888bb70115c7ec17c085f21afa6903feeeb595d33aa9ce6c27534bfe",
}


def _run_identity(source_commit: str | None, run_namespace: str | None) -> dict:
    if (type(source_commit) is not str or re.fullmatch(
            r"[0-9a-f]{40}", source_commit) is None
            or type(run_namespace) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{15,159}", run_namespace) is None):
        raise ValueError("MuJoCo Full-A run identity differs")
    return {"source_commit": source_commit, "run_namespace": run_namespace}


def _stable_file_sha256(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before, digest = os.fstat(fd), hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or (before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns)
                or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)):
            raise RuntimeError("RSL-RL source file changed during verification")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _ready_pose_input() -> tuple[bytes, str]:
    raw = os.environ.get("ACTIONBALL_READY_POSE")
    if not raw:
        raise RuntimeError("MuJoCo WAIT ready-pose path is not bound")
    path = Path(raw)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeError("MuJoCo WAIT ready-pose path differs") from exc
    try:
        row = os.fstat(fd)
        resolved = path.resolve(strict=True)
        payload = b""
        while chunk := os.read(fd, 1024 * 1024):
            payload += chunk
        current = os.stat(path, follow_symlinks=False)
        if (
            not path.is_absolute()
            or not stat.S_ISREG(row.st_mode)
            or row.st_nlink != 1
            or current.st_nlink != 1
            or resolved != path
            or (row.st_dev, row.st_ino) != (current.st_dev, current.st_ino)
            or hashlib.sha256(payload).hexdigest() != READY_POSE_SHA256
        ):
            raise RuntimeError("MuJoCo WAIT ready-pose path differs")
        return payload, str(path)
    except OSError as exc:
        raise RuntimeError("MuJoCo WAIT ready-pose path differs") from exc
    finally:
        os.close(fd)


def build_train_cfg() -> dict:
    """Return the same RSL3 PPO surface used by the Isaac FullMDP run."""

    return FULL_MDP_PPO_RECIPE.mujoco_train_cfg()


def _snapshot_indices(num_updates: int, save_interval: int) -> tuple[int, ...]:
    """Return cadence checkpoints plus the final finite-run frontier."""

    indices = tuple(range(0, num_updates, save_interval))
    return indices if indices[-1] == num_updates - 1 else indices + (num_updates - 1,)


def _apply_full_a_policy_bootstrap(runner, torch_module) -> None:
    """Install the fresh Isaac A/C zero-mean policy prior exactly once."""

    policy = runner.alg.policy
    actor = getattr(policy, "actor", None)
    children = list(actor.children()) if isinstance(
        actor, torch_module.nn.Sequential
    ) else []
    output = children[-1] if children else None
    log_std = getattr(policy, "log_std", None)
    if (
        not isinstance(output, torch_module.nn.Linear)
        or output.out_features != 31
        or output.bias is None
        or not torch_module.is_tensor(log_std)
        or tuple(log_std.shape) != (31,)
        or getattr(policy, "noise_std_type", None) != "log"
    ):
        raise RuntimeError("MuJoCo Full-A policy bootstrap surface differs")
    with torch_module.no_grad():
        output.weight.zero_()
        output.bias.zero_()
    expected_std = torch_module.full_like(log_std, 0.02)
    if (
        int(torch_module.count_nonzero(output.weight).item()) != 0
        or int(torch_module.count_nonzero(output.bias).item()) != 0
        or not torch_module.allclose(
            torch_module.exp(log_std), expected_std, rtol=0.0, atol=1.0e-8
        )
    ):
        raise RuntimeError("MuJoCo Full-A zero-head/log-std bootstrap differs")


def _wait_module():
    module = importlib.import_module("mujoco_gpu_ac_full_mdp_initial_wait_env")
    expected = Path(__file__).with_name(
        "mujoco_gpu_ac_full_mdp_initial_wait_env.py"
    ).resolve()
    actual = Path(getattr(module, "__file__", "")).resolve()
    if actual != expected:
        raise RuntimeError("MuJoCo WAIT environment import origin differs")
    return module


def _update_ledger_module():
    module = importlib.import_module("mujoco_full_mdp_update_ledger")
    expected = Path(__file__).with_name("mujoco_full_mdp_update_ledger.py").resolve()
    if Path(getattr(module, "__file__", "")).resolve() != expected:
        raise RuntimeError("MuJoCo FullMDP update ledger import origin differs")
    return module


def _open_evidence_jsonl(raw: str | None) -> int | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("evidence JSONL path must be absolute")
    fd = os.open(
        path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        row, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        if (not stat.S_ISREG(row.st_mode) or row.st_nlink != 1
                or current.st_nlink != 1 or row.st_size != 0
                or (row.st_dev, row.st_ino) != (current.st_dev, current.st_ino)):
            raise ValueError("evidence JSONL must be a fresh regular file")
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        os.close(fd)
        raise
    return fd


def _snapshot_root(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    try:
        row = path.lstat()
    except OSError as exc:
        raise ValueError("snapshot directory differs") from exc
    if (not path.is_absolute() or not stat.S_ISDIR(row.st_mode)
            or path.resolve() != path or any(path.iterdir())):
        raise ValueError("snapshot directory differs")
    return path


def _save_snapshot(runner, root: Path, update_index: int, *, run_identity: dict,
                   prepared_update_sha256: str) -> dict:
    path = root / f"model_{update_index}.pt"
    dir_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = -1
    observed = runner.current_learning_iteration
    try:
        fd = os.open(path.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=dir_fd)
        runner.current_learning_iteration = update_index
        with os.fdopen(fd, "w+b", closefd=False) as stream:
            runner.save(stream, infos={"diagnostic_unauthorized": True,
                "checkpoint_authority": False, "resume_authority": False,
                "update_index": update_index, "completed_updates": update_index + 1,
                "run_identity": dict(run_identity),
                "action_ball_full_mdp_ppo_recipe_sha256": (
                    FULL_MDP_PPO_RECIPE_SHA256
                ),
                "prepared_update_sha256": prepared_update_sha256})
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        os.fsync(dir_fd)
        row = os.fstat(fd)
        current = os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)
        if (not stat.S_ISREG(row.st_mode) or row.st_nlink != 1 or row.st_size <= 0
                or (row.st_dev, row.st_ino) != (current.st_dev, current.st_ino)):
            raise RuntimeError("diagnostic snapshot differs")
    finally:
        runner.current_learning_iteration = observed
        if fd >= 0:
            os.close(fd)
        os.close(dir_fd)
    return {"name": path.name, "bytes": row.st_size,
            "sha256": digest.hexdigest()}


def _fd_inventory(fd: int, path: Path) -> dict:
    os.fsync(fd)
    row = os.fstat(fd)
    digest = hashlib.sha256()
    offset = 0
    while offset < row.st_size:
        chunk = os.pread(fd, min(1024 * 1024, row.st_size - offset), offset)
        if not chunk:
            raise RuntimeError("evidence JSONL short read")
        digest.update(chunk)
        offset += len(chunk)
    after, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
    if (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        after.st_ctime_ns,
    ) or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
        raise RuntimeError("evidence JSONL changed during finalization")
    return {"bytes": row.st_size, "sha256": digest.hexdigest()}


def _write_completion(raw: str, record: dict) -> None:
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("completion receipt path must be absolute")
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = -1
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent_fd)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            if stream.write(payload) != len(payload):
                raise OSError("completion receipt write made no progress")
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(parent_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)


def _optimizer_state_evidence(optimizer, torch_module) -> tuple[bool, bool]:
    try:
        parameters = tuple(
            parameter for group in optimizer.param_groups
            for parameter in group["params"]
        )
        required = {"step", "exp_avg", "exp_avg_sq"}
        present = bool(parameters) and set(optimizer.state) == set(parameters) and all(
            type(optimizer.state[parameter]) is dict
            and set(optimizer.state[parameter]) == required
            for parameter in parameters
        )
        if not present:
            return False, False
        for parameter in parameters:
            step, mean, square = (
                optimizer.state[parameter][name]
                for name in ("step", "exp_avg", "exp_avg_sq")
            )
            if (not all(isinstance(value, torch_module.Tensor)
                        for value in (step, mean, square))
                    or step.numel() != 1 or not bool(step.gt(0).all())
                    or tuple(mean.shape) != tuple(parameter.shape)
                    or tuple(square.shape) != tuple(parameter.shape)
                    or not all(bool(torch_module.isfinite(value).all())
                               for value in (step, mean, square))):
                return True, False
        return True, True
    except (KeyError, TypeError, AttributeError):
        return False, False


def _rsl3_runner():
    import torch

    distribution = importlib.metadata.distribution("rsl-rl-lib")
    if distribution.version != RSL_RL_VERSION:
        raise RuntimeError(
            f"MuJoCo WAIT requires RSL-RL {RSL_RL_VERSION}, got {distribution.version}"
        )
    module = importlib.import_module("rsl_rl.runners.on_policy_runner")
    expected = Path(
        distribution.locate_file("rsl_rl/runners/on_policy_runner.py")
    ).resolve()
    runner = getattr(module, "OnPolicyRunner", None)
    actual = Path(getattr(module, "__file__", "")).resolve()
    source = Path(inspect.getsourcefile(runner) or "").resolve()
    if actual != expected or source != expected:
        raise RuntimeError("MuJoCo WAIT RSL-RL import origin differs")
    ppo_module = importlib.import_module("rsl_rl.algorithms.ppo")
    actor_module = importlib.import_module("rsl_rl.modules.actor_critic")
    recurrent_module = importlib.import_module("rsl_rl.modules.actor_critic_recurrent")
    storage_module = importlib.import_module("rsl_rl.storage.rollout_storage")
    mlp_module = importlib.import_module("rsl_rl.networks.mlp")
    _require_rsl3_preconstruction(
        distribution, module, ppo_module, actor_module, recurrent_module,
        storage_module, mlp_module, torch,
    )
    return distribution.version, runner, distribution


def _require_rsl3_preconstruction(
    distribution,
    runner_module,
    ppo_module,
    actor_module,
    recurrent_module,
    storage_module,
    mlp_module,
    torch_module,
) -> None:
    expected_modules = (
        (runner_module, "rsl_rl/runners/on_policy_runner.py"),
        (ppo_module, "rsl_rl/algorithms/ppo.py"),
        (actor_module, "rsl_rl/modules/actor_critic.py"),
        (recurrent_module, "rsl_rl/modules/actor_critic_recurrent.py"),
        (storage_module, "rsl_rl/storage/rollout_storage.py"),
        (mlp_module, "rsl_rl/networks/mlp.py"),
    )
    if any(
        Path(getattr(module, "__file__", "")).resolve()
        != Path(distribution.locate_file(relative)).resolve()
        for module, relative in expected_modules
    ) or not (
        runner_module.PPO is ppo_module.PPO
        and runner_module.ActorCritic is actor_module.ActorCritic
        and runner_module.ActorCriticRecurrent is recurrent_module.ActorCriticRecurrent
        and ppo_module.RolloutStorage is storage_module.RolloutStorage
        and actor_module.MLP is mlp_module.MLP
        and ppo_module.optim.Adam is torch_module.optim.Adam
    ):
        raise RuntimeError("MuJoCo WAIT RSL-RL preconstruction origin differs")
    for relative, expected_sha256 in RSL_RL_SOURCE_SHA256.items():
        if _stable_file_sha256(
            Path(distribution.locate_file(relative)).resolve()
        ) != expected_sha256:
            raise RuntimeError("MuJoCo WAIT RSL-RL source bytes differ: " + relative)


def _require_rsl3_runtime(distribution, runner, torch_module) -> None:
    try:
        alg = runner.alg
        runtime = (
            (alg, "rsl_rl/algorithms/ppo.py"),
            (alg.policy, "rsl_rl/modules/actor_critic.py"),
            (alg.storage, "rsl_rl/storage/rollout_storage.py"),
        )
        optimizer = alg.optimizer
    except AttributeError as exc:
        raise RuntimeError("MuJoCo WAIT RSL-RL runtime origin differs") from exc
    if any(
        Path(inspect.getsourcefile(type(value)) or "").resolve()
        != Path(distribution.locate_file(relative)).resolve()
        for value, relative in runtime
    ) or type(optimizer) is not torch_module.optim.Adam or (
        getattr(alg.update, "__self__", None) is not alg
        or Path(inspect.getsourcefile(alg.update) or "").resolve()
        != Path(distribution.locate_file("rsl_rl/algorithms/ppo.py")).resolve()
    ):
        raise RuntimeError("MuJoCo WAIT RSL-RL runtime origin differs")


def main(
    *,
    num_envs: int = 2,
    num_updates: int = 1,
    full_a_mode: bool = False,
    evidence_jsonl: str | None = None,
    snapshot_dir: str | None = None,
    completion_json: str | None = None,
    source_commit: str | None = None,
    run_namespace: str | None = None,
    save_interval: int = FULL_A_SAVE_INTERVAL,
    _test_allow_small_full_a: bool = False,
) -> int:
    import torch

    num_steps_per_env = NUM_STEPS_PER_ENV
    if (
        type(num_envs) is not int
        or num_envs <= 0
        or type(num_updates) is not int
        or num_updates <= 0
        or type(full_a_mode) is not bool
        or type(_test_allow_small_full_a) is not bool
        or type(save_interval) is not int or save_interval != FULL_A_SAVE_INTERVAL
    ):
        raise ValueError("runner dimensions/mode differ")
    if full_a_mode and (
        evidence_jsonl is None or snapshot_dir is None or completion_json is None
    ):
        raise ValueError("MuJoCo Full-A requires evidence, snapshot and completion paths")
    if not full_a_mode and any(value is not None for value in (
        evidence_jsonl, snapshot_dir, completion_json, source_commit, run_namespace
    )):
        raise ValueError("MuJoCo Full-A artifact arguments require --full-a")
    if full_a_mode and not _test_allow_small_full_a and (
        num_envs != FULL_A_NUM_ENVS
        or num_updates != FULL_A_NUM_UPDATES
    ):
        raise ValueError(
            "production MuJoCo Full-A shape must be "
            f"{FULL_A_NUM_ENVS}x{NUM_STEPS_PER_ENV}x{FULL_A_NUM_UPDATES}"
        )
    identity = _run_identity(source_commit, run_namespace) if full_a_mode else None
    version, runner_type, distribution = _rsl3_runner()
    wait = _wait_module()
    ready_pose_payload, ready_pose_source = _ready_pose_input()
    torch.manual_seed(0)
    task = wait.TaskCfg(
        episode_length_s=30.0 if full_a_mode else 3.0,
        action_scale_mode="vendor",
        reset_joint_noise_rad=0.0,
        reset_joint_vel_noise=0.0,
        reset_root_xy_noise_m=0.0,
        reset_root_yaw_noise_rad=0.0,
    )
    env = wait.FullMdpInitialWaitVecEnv(
        wait.SimCfg(nworld=num_envs),
        task,
        device="cuda:0",
        seed=0,
        ready_pose_payload=ready_pose_payload,
        ready_pose_source=ready_pose_source,
        full_a_mode=full_a_mode,
    )
    action_contract = env.action_contract_identity if full_a_mode else None
    initial = env.get_observations()
    if (
        env.num_actions != 31
        or tuple(initial["policy"].shape) != (num_envs, 229)
        or tuple(initial["critic"].shape) != (num_envs, 399)
        or not bool(torch.isfinite(initial["policy"]).all())
        or not bool(torch.isfinite(initial["critic"]).all())
    ):
        raise RuntimeError("MuJoCo WAIT initial RSL3 surface differs")

    ledger = None
    if full_a_mode:
        ledger_module = _update_ledger_module()
        ledger = ledger_module.FullMdpUpdateLedger(
            torch_module=torch,
            num_envs=num_envs,
            num_steps_per_env=num_steps_per_env,
            device=env.device,
            termination_bits=wait.FULLMDP_TERMINATION_BITS,
            action_slot=0,
            action_uid=FULL_A_ACTION_UID,
            mount_normal_sign=FULL_A_MOUNT_NORMAL_SIGN,
            family=FULL_A_FAMILY,
            initial_reset_generation=env.reset_generation,
            source_commit=identity["source_commit"],
            run_namespace=identity["run_namespace"],
        )
        original_env_step = env.step

        def evidence_step(actions):
            result = original_env_step(actions)
            ledger.ingest(result)
            return result

        env.step = evidence_step

    runner = runner_type(
        env,
        build_train_cfg(),
        log_dir=None,
        device="cuda:0",
    )
    _require_rsl3_runtime(distribution, runner, torch)
    if full_a_mode:
        _apply_full_a_policy_bootstrap(runner, torch)
    runner.disable_logs = True
    # RSL-RL 3.1.2 initializes this field only when a logging writer exists,
    # but its stock save() reads it unconditionally before checking disable_logs.
    runner.logger_type = "tensorboard"
    updates = 0
    snapshots = _snapshot_root(snapshot_dir) if full_a_mode else None
    evidence_fd = _open_evidence_jsonl(evidence_jsonl) if full_a_mode else None
    snapshot_receipts = []
    snapshot_indices = (
        _snapshot_indices(num_updates, save_interval) if full_a_mode else ()
    )
    original_update = runner.alg.update
    run_started_at = time.perf_counter()
    iteration_started_at = run_started_at

    def counted_update():
        nonlocal iteration_started_at, updates
        collection_finished_at = time.perf_counter()
        if ledger is not None:
            storage = runner.alg.storage
            prepared = ledger.prepare(
                updates, environment_steps=env.common_step_counter,
                storage_step=storage.step,
                storage_tensors={"rewards": storage.rewards,
                    "returns": storage.returns, "advantages": storage.advantages},
                policy_std=runner.alg.policy.action_std,
            )
        else:
            prepared = None
        learning_started_at = time.perf_counter()
        result = original_update()
        learning_finished_at = time.perf_counter()
        updates += 1
        if ledger is not None:
            index = updates - 1
            receipt = None
            if index in snapshot_indices:
                prepared_sha256 = hashlib.sha256(prepared.payload).hexdigest()
                receipt = _save_snapshot(
                    runner, snapshots, index, run_identity=identity,
                    prepared_update_sha256=prepared_sha256,
                )
            ledger.ack(
                prepared, completed_updates=updates, evidence_fd=evidence_fd,
                optimizer_metrics=result, learning_rate=runner.alg.learning_rate,
                timings={
                    "collection_seconds": collection_finished_at - iteration_started_at,
                    "learning_seconds": learning_finished_at - learning_started_at,
                    "pre_ack_iteration_seconds": (
                        learning_finished_at - iteration_started_at
                    ),
                    "run_elapsed_pre_ack_seconds": (
                        learning_finished_at - run_started_at
                    ),
                },
                snapshot=receipt,
            )
            if receipt is not None:
                snapshot_receipts.append(receipt)
                print(
                    f"ACTION_BALL_MUJOCO_FULL_A_PROGRESS={index}:{receipt['name']}",
                    flush=True,
                )
        iteration_started_at = time.perf_counter()
        return result

    runner.alg.update = counted_update
    try:
        runner.learn(num_updates, init_at_random_ep_len=False)
        final = env.get_observations()
        storage = runner.alg.storage
        storage_finite = all(
            isinstance(value, torch.Tensor)
            and tuple(value.shape) == (num_steps_per_env, num_envs, 1)
            and bool(torch.isfinite(value).all())
            for value in (storage.rewards, storage.returns, storage.advantages)
        )
        optimizer_state_present, optimizer_state_finite = _optimizer_state_evidence(
            runner.alg.optimizer, torch
        )
        final_observation_finite = (
            bool(torch.isfinite(final["policy"]).all())
            and bool(torch.isfinite(final["critic"]).all())
        )
        if (
            updates != num_updates
            or env.common_step_counter != num_steps_per_env * num_updates
            or storage.step != 0
            or not storage_finite
            or not optimizer_state_present
            or not optimizer_state_finite
            or tuple(final["policy"].shape) != (num_envs, 229)
            or tuple(final["critic"].shape) != (num_envs, 399)
            or not final_observation_finite
        ):
            raise RuntimeError("MuJoCo WAIT RSL3 update evidence differs")
        if full_a_mode:
            if tuple(row["name"] for row in snapshot_receipts) != tuple(
                f"model_{index}.pt" for index in snapshot_indices
            ):
                raise RuntimeError("MuJoCo Full-A snapshot frontier differs")
            completion = {
                "schema_version": COMPLETION_SCHEMA_VERSION,
                "record_type": "mujoco_full_mdp_completion",
                "diagnostic_unauthorized": True,
                "checkpoint_authority": False, "resume_authority": False,
                "run_identity": dict(identity),
                "action_contract": action_contract,
                "action_ball_full_mdp_ppo_recipe_sha256": (
                    FULL_MDP_PPO_RECIPE_SHA256
                ),
                "num_envs": num_envs,
                "num_steps_per_env": num_steps_per_env,
                "completed_updates": updates,
                "environment_steps": env.common_step_counter,
                "transitions": env.common_step_counter * num_envs,
                "evidence_jsonl": _fd_inventory(evidence_fd, Path(evidence_jsonl)),
                "snapshot_receipts": snapshot_receipts,
                "final_observation_finite": final_observation_finite,
                "rollout_storage_finite": storage_finite,
                "optimizer_state_present": optimizer_state_present,
                "optimizer_state_finite": optimizer_state_finite,
            }
            _write_completion(completion_json, completion)
    finally:
        if evidence_fd is not None:
            os.close(evidence_fd)
    payload = {
        "diagnostic_unauthorized": True,
        "rsl_rl_version": version,
        "ppo_update_calls": updates,
        "environment_steps": env.common_step_counter,
        "transitions": env.common_step_counter * env.num_envs,
        "policy_width": final["policy"].shape[1],
        "critic_width": final["critic"].shape[1],
        "action_ball_full_mdp_ppo_recipe_sha256": (
            FULL_MDP_PPO_RECIPE_SHA256
        ),
        "task_lifecycle": "full_a_engineering_longrun_complete" if full_a_mode else "idle_wait_only",
    }
    if full_a_mode:
        payload.update({
            "engineering_run_complete": True, "full_a_update_ack_count": updates,
        })
    print(
        "ACTION_BALL_MUJOCO_WAIT_RSL3_JSON="
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-updates", type=int, default=1)
    parser.add_argument("--full-a", dest="full_a_mode", action="store_true")
    parser.add_argument("--evidence-jsonl")
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--completion-json")
    parser.add_argument("--source-commit")
    parser.add_argument("--run-namespace")
    parser.add_argument(
        "--save-interval", type=int, default=FULL_A_SAVE_INTERVAL
    )
    raise SystemExit(main(**vars(parser.parse_args())))
