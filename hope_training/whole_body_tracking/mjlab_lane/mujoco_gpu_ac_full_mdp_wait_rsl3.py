"""Run upstream RSL-RL 3 on the real MuJoCo portable environment.

The default preserves the historical one-update WAIT ABI. ``--full-a`` exposes
the current R03/R06/R07/ordered-reward-graph engineering surface with
fail-closed evidence;
``--diagnostic-rate-probe`` keeps that surface but measures a finite 61-update
profiler-off window without snapshots or completion authority.
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
import statistics
import subprocess
import sys
import time


def _best_effort_stdout_marker(marker: str) -> None:
    """Publish a human-readable marker without becoming training authority."""

    try:
        sys.stdout.write(marker + "\n")
        sys.stdout.flush()
    except (OSError, ValueError) as exc:
        # Optimizer state, evidence ACKs, snapshots and completion receipts are
        # already durable before these markers are emitted.  A closed pipe or
        # broken logging sink must not relabel that committed state as a failed
        # update.  stderr is also observational and may itself be unavailable.
        try:
            warning = {
                "event": "action_ball_stdout_marker_failed",
                "error_type": type(exc).__name__,
            }
            sys.stderr.write(
                json.dumps(warning, sort_keys=True, separators=(",", ":")) + "\n"
            )
            sys.stderr.flush()
        except (OSError, ValueError):
            pass


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


def _epa48_runtime_module():
    """Load the dependency-free Full-A runtime binder from this checkout."""

    source = Path(__file__).with_name("mujoco_full_mdp_epa48_runtime.py").resolve()
    name = "_hope_mujoco_full_mdp_epa48_runtime"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise RuntimeError("cached Full-A EPA48 runtime binder origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Full-A EPA48 runtime binder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _plant_contract_module():
    """Load the dependency-free plant wire contract from this checkout."""

    source = Path(__file__).with_name("mujoco_full_mdp_plant_contract.py").resolve()
    name = "_hope_mujoco_full_mdp_plant_contract"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise RuntimeError("cached MuJoCo plant contract origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load MuJoCo plant contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _canonical_mujoco_identity_module():
    """Load the complete source-closure/model verifier from this checkout."""

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "canonical_mujoco_identity.py"
    )
    name = "_hope_canonical_mujoco_identity"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise RuntimeError("cached canonical MuJoCo identity origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical MuJoCo identity verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _mujoco_module():
    return importlib.import_module("mujoco")


FULL_MDP_PPO_RECIPE = (
    _ppo_recipe_module().ACTION_BALL_FULL_MDP_PPO_RECIPE
)
FULL_MDP_PPO_RECIPE_SHA256 = FULL_MDP_PPO_RECIPE.recipe_sha256()
RSL_RL_VERSION = "3.1.2"
COMPLETION_SCHEMA_VERSION = 5
NUM_STEPS_PER_ENV = FULL_MDP_PPO_RECIPE.num_steps_per_env
READY_POSE_SHA256 = "b88d93c311b439bd61296b3b3a84198200d9c6938980471071992ec52d8df18f"
FULL_A_ACTION_UID = 2552478955674699
FULL_A_MOUNT_NORMAL_SIGN = 1
FULL_A_FAMILY = "backhand"
FULL_A_NUM_ENVS = FULL_MDP_PPO_RECIPE.num_envs
FULL_A_NUM_UPDATES = FULL_MDP_PPO_RECIPE.max_iterations
FULL_A_SAVE_INTERVAL = FULL_MDP_PPO_RECIPE.save_interval
DIAGNOSTIC_TEACHER_REPLAY_HANDOFF_SPLIT_READY_BRIDGE = (
    "split_ready_bridge"
)
DIAGNOSTIC_TEACHER_REPLAY_HANDOFF_DIRECT_FRAME0 = (
    "direct_frame0_playback"
)
DIAGNOSTIC_TEACHER_REPLAY_HANDOFF_MODES = (
    DIAGNOSTIC_TEACHER_REPLAY_HANDOFF_SPLIT_READY_BRIDGE,
    DIAGNOSTIC_TEACHER_REPLAY_HANDOFF_DIRECT_FRAME0,
)
RATE_PROBE_WARMUP_UPDATES = 10
RATE_PROBE_MEASURED_UPDATES = 50
RATE_PROBE_TAIL_UPDATES = 1
RATE_PROBE_NUM_UPDATES = (
    RATE_PROBE_WARMUP_UPDATES
    + RATE_PROBE_MEASURED_UPDATES
    + RATE_PROBE_TAIL_UPDATES
)
RATE_PROBE_PROFILE_ENVS = (
    "HOPE_ACTION_BALL_UPDATE_PROFILE",
    "HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES",
)
RSL_RL_SOURCE_SHA256 = {
    "rsl_rl/runners/on_policy_runner.py": "6ffaee7e154a49ae55eebf53a7b1549f0461a1742d92dd34af5bb4b785d19cf2",
    "rsl_rl/algorithms/ppo.py": "4373ac1b2f9fdf14d9da57516968fc95d8f605d2967fee01dc61bf0d09423478",
    "rsl_rl/modules/actor_critic.py": "614eb6e14d21c46504ce2046f04c9ab70a8c3cf679502ef2a220987e506a959a",
    "rsl_rl/modules/actor_critic_recurrent.py": "19fbe660f6a22a8df4e7d54dc13e6dc6d668f69c71f9303165553b9272aff5d0",
    "rsl_rl/storage/rollout_storage.py": "32d8b1b3cead87e0eeb96e2b334a5c75fd431309e43dae85a42a89b62c5dc5de",
    "rsl_rl/networks/mlp.py": "ead23a9b888bb70115c7ec17c085f21afa6903feeeb595d33aa9ce6c27534bfe",
}


def _canonical_payload_sha256(payload: dict[str, object]) -> str:
    body = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _rate_execution_recipe(
    *,
    num_envs: int = FULL_A_NUM_ENVS,
    num_steps_per_env: int = NUM_STEPS_PER_ENV,
    max_iterations: int = RATE_PROBE_NUM_UPDATES,
    save_interval: int = FULL_A_SAVE_INTERVAL,
) -> dict[str, object]:
    """Return the backend-neutral, actual finite rate execution identity."""

    effective_runner = {
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        "save_interval": save_interval,
    }
    candidate_runner = {
        "num_envs": FULL_A_NUM_ENVS,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "max_iterations": FULL_A_NUM_UPDATES,
        "save_interval": FULL_A_SAVE_INTERVAL,
    }
    return {
        "schema_version": 1,
        "kind": "action_ball_full_mdp_h48_rate_execution_v1",
        "candidate_production_execution_recipe_sha256": (
            FULL_MDP_PPO_RECIPE_SHA256
        ),
        "learning_recipe_sha256": (
            FULL_MDP_PPO_RECIPE.learning_recipe_sha256()
        ),
        "effective_runner": effective_runner,
        "runner_overrides": {
            name: {
                "candidate_production": candidate_runner[name],
                "rate_execution": effective_runner[name],
            }
            for name in candidate_runner
            if candidate_runner[name] != effective_runner[name]
        },
        "diagnostic_overrides": {
            "warmup_updates": RATE_PROBE_WARMUP_UPDATES,
            "measured_updates": RATE_PROBE_MEASURED_UPDATES,
            "tail_updates": RATE_PROBE_TAIL_UPDATES,
            "profiler_enabled": False,
            "diagnostic_unauthorized": True,
            "formal_evidence": False,
            "checkpoint_authority": False,
            "resume_authority": False,
        },
    }


def _require_run_identity_fields(
    source_commit: str | None, run_namespace: str | None
) -> None:
    if (type(source_commit) is not str or re.fullmatch(
            r"[0-9a-f]{40}", source_commit) is None
            or type(run_namespace) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{15,159}", run_namespace) is None):
        raise ValueError("MuJoCo Full-A run identity differs")


def _verified_source_checkout_commit(
    expected_commit: str, *, repo_root: Path | None = None,
) -> str:
    """Measure, rather than echo, the source checkout used by the runner."""

    if type(expected_commit) is not str or re.fullmatch(
        r"[0-9a-f]{40}", expected_commit
    ) is None:
        raise RuntimeError("MuJoCo Full-A source checkout differs")
    root = (
        Path(__file__).resolve().parents[3]
        if repo_root is None
        else Path(repo_root)
    )
    try:
        if not root.is_absolute() or root.resolve(strict=True) != root:
            raise RuntimeError("MuJoCo Full-A source checkout differs")
    except OSError as exc:
        raise RuntimeError("MuJoCo Full-A source checkout differs") from exc

    prefix = ["git", "--no-optional-locks", "-C", str(root)]

    def read(arguments: list[str]) -> str:
        try:
            result = subprocess.run(
                prefix + arguments,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except OSError as exc:
            raise RuntimeError("MuJoCo Full-A source checkout differs") from exc
        if result.returncode != 0:
            raise RuntimeError("MuJoCo Full-A source checkout differs")
        return result.stdout

    actual = read(["rev-parse", "HEAD"]).strip()
    status = read(["status", "--porcelain=v1", "--untracked-files=all"])
    if actual != expected_commit or status:
        raise RuntimeError("MuJoCo Full-A source checkout differs")
    return actual


def _run_identity(
    source_commit: str | None,
    run_namespace: str | None,
    runtime_stack: dict,
    plant_model: dict,
) -> dict:
    return {
        "source_commit": source_commit,
        "run_namespace": run_namespace,
        "runtime_stack": {
            "schema_version": runtime_stack["schema_version"],
            "mujoco_warp": dict(runtime_stack["mujoco_warp"]),
            "rsl_rl": dict(runtime_stack["rsl_rl"]),
            "mjlab": dict(runtime_stack["mjlab"]),
        },
        "plant_model": (
            _plant_contract_module().clone_plant_model_identity(plant_model)
        ),
    }


def _bind_full_a_runtime(raw: str | None, preimport_verification) -> dict:
    if type(raw) is not str or not raw:
        raise ValueError("MuJoCo Full-A runtime site is not bound")
    return _epa48_runtime_module().bind_fresh_epa48_runtime_site(
        Path(raw), preimport_verification=preimport_verification
    )


def _verify_full_a_runtime_postimport(runtime_stack: dict) -> None:
    """Close the MJLab pre/post import identity before evidence construction."""

    if (
        type(runtime_stack) is not dict
        or set(runtime_stack) != {"schema_version", "mujoco_warp", "rsl_rl", "mjlab"}
        or type(runtime_stack.get("schema_version")) is not int
        or runtime_stack["schema_version"] != 1
        or type(runtime_stack.get("mjlab")) is not dict
    ):
        raise RuntimeError("MuJoCo Full-A runtime stack identity differs")
    postimport = _epa48_runtime_module().verify_loaded_mjlab_runtime_modules()
    if postimport != runtime_stack["mjlab"]:
        raise RuntimeError("MuJoCo Full-A MJLab pre/post identity differs")


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


def _plant_xml_input() -> Path:
    raw = os.environ.get("A3_PINGPONG_XML")
    if not raw:
        raise RuntimeError("MuJoCo Full-A plant XML is not bound")
    path = Path(raw)
    expected = _plant_contract_module().expected_plant_model_identity()
    if (
        not path.is_absolute()
        or path.name != expected["source_plant"]["root_filename"]
    ):
        raise RuntimeError("MuJoCo Full-A plant XML path differs")
    return path


def _require_geometry_source_environment() -> None:
    """Reject the legacy diagnostic override before importing the court."""

    if "HOPE_GEOMETRY_PY" in os.environ:
        raise RuntimeError(
            "MuJoCo Full-A forbids the ambient HOPE_GEOMETRY_PY override"
        )


def _geometry_source_identity() -> str:
    """Bind the geometry module that actually constructed the live court."""

    contract = _plant_contract_module()
    court = importlib.import_module("a3_court_env")
    expected_court = Path(__file__).with_name("a3_court_env.py").resolve()
    if Path(getattr(court, "__file__", "")).resolve() != expected_court:
        raise RuntimeError("MuJoCo Full-A court module import origin differs")
    try:
        source = Path(court.geom.__source_path__)
        digest = contract.verify_geometry_source(source)
    except Exception as exc:
        raise RuntimeError("MuJoCo Full-A geometry source identity differs") from exc
    return digest


def _scan_plant_source(path: Path) -> dict:
    contract = _plant_contract_module()
    canonical = _canonical_mujoco_identity_module()
    try:
        closure = canonical.scan_mjcf_source_closure(path)
    except Exception as exc:
        raise RuntimeError("MuJoCo Full-A plant source-closure scan failed") from exc
    expected = contract.expected_plant_model_identity()["source_plant"]
    actual = {
        "root_path": str(closure.root_path),
        "root_filename": closure.root_filename,
        "root_mjcf_sha256": closure.root_mjcf_sha256,
        "source_closure_sha256": closure.closure_sha256,
        "source_member_count": closure.member_count,
        "source_total_bytes": closure.total_bytes,
    }
    if actual != {
        "root_path": str(path),
        **{key: expected[key] for key in (
            "root_filename", "root_mjcf_sha256", "source_closure_sha256",
            "source_member_count", "source_total_bytes",
        )},
    }:
        raise RuntimeError("MuJoCo Full-A plant source closure differs")
    return actual


def _plant_model_identity(
    env, path: Path, before, after, augmented_mjb: dict,
) -> dict:
    contract = _plant_contract_module()
    expected = contract.expected_plant_model_identity()
    source_expected = expected["source_plant"]
    runtime_expected = expected["runtime_attach"]
    try:
        env_path = Path(env.env.xml_path)
        resolved = path.resolve(strict=True)
        live_binding = dict(env._table_keepout.plant_identity_receipt)
    except (AttributeError, OSError, TypeError) as exc:
        raise RuntimeError("MuJoCo Full-A plant XML identity differs") from exc
    binding_keys = {
        "root_mjcf_sha256", "identity_manifest_sha256",
        "portable_identity_sha256", "verification_receipt_sha256",
        "owner_local_frame_sha256",
    }
    if (
        not path.is_absolute()
        or resolved != path
        or env_path != path
        or before != after
        or set(live_binding) != binding_keys
        or live_binding["root_mjcf_sha256"]
        != source_expected["root_mjcf_sha256"]
        or live_binding["identity_manifest_sha256"]
        != source_expected["manifest_sha256"]
        or live_binding["portable_identity_sha256"]
        != source_expected["portable_identity_sha256"]
    ):
        raise RuntimeError("MuJoCo Full-A plant XML identity differs")
    try:
        policy_clock = {
            "decimation": int(env.decimation),
            "step_dt": float(env.step_dt),
        }
        naconmax = int(env.naconmax_alloc)
        num_envs = int(env.num_envs)
        warp_capacity = {
            "njmax_per_world": int(env.njmax_alloc),
            "nconmax_per_world": naconmax // num_envs,
        }
        if num_envs <= 0 or naconmax != warp_capacity["nconmax_per_world"] * num_envs:
            raise RuntimeError("MuJoCo Full-A Warp capacity shape differs")
        geometry_source_sha256 = _geometry_source_identity()
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("MuJoCo Full-A runtime plant attachment differs") from exc
    runtime_actual = {
        "model_scope": runtime_expected["model_scope"],
        "contract_type": runtime_expected["contract_type"],
        "geometry_source_sha256": geometry_source_sha256,
        "policy_clock": policy_clock,
        "warp_capacity": warp_capacity,
        "final_augmented_mjb": augmented_mjb,
    }
    if not contract.runtime_attach_is_exact(runtime_actual):
        raise RuntimeError("MuJoCo Full-A runtime plant attachment differs")
    return contract.verified_plant_model_identity(
        verification_receipt_sha256=(
            live_binding["verification_receipt_sha256"]
        ),
        owner_local_frame_sha256=live_binding["owner_local_frame_sha256"],
        final_augmented_mjb=augmented_mjb,
    )


def build_train_cfg() -> dict:
    """Return the same RSL3 PPO surface used by the Isaac FullMDP run."""

    return FULL_MDP_PPO_RECIPE.mujoco_train_cfg()


def _snapshot_indices(num_updates: int, save_interval: int) -> tuple[int, ...]:
    """Return cadence checkpoints plus the final finite-run frontier."""

    indices = tuple(range(0, num_updates, save_interval))
    return indices if indices[-1] == num_updates - 1 else indices + (num_updates - 1,)


def _apply_full_a_policy_bootstrap(runner, torch_module, env) -> None:
    """Install the same-source Take061 dynamic-ready mean exactly once."""

    policy = runner.alg.policy
    actor = getattr(policy, "actor", None)
    children = list(actor.children()) if isinstance(
        actor, torch_module.nn.Sequential
    ) else []
    output = children[-1] if children else None
    log_std = getattr(policy, "log_std", None)
    bootstrap_action = getattr(env, "_full_a_policy_bootstrap_action", None)
    if (
        not isinstance(output, torch_module.nn.Linear)
        or output.out_features != 31
        or output.bias is None
        or not torch_module.is_tensor(log_std)
        or tuple(log_std.shape) != (31,)
        or getattr(policy, "noise_std_type", None)
        != FULL_MDP_PPO_RECIPE.noise_std_type
        or not torch_module.is_tensor(bootstrap_action)
        or tuple(bootstrap_action.shape) != (31,)
        or bootstrap_action.device != output.bias.device
        or bootstrap_action.dtype != output.bias.dtype
        or not bool(torch_module.isfinite(bootstrap_action).all())
    ):
        raise RuntimeError("MuJoCo Full-A policy bootstrap surface differs")
    with torch_module.no_grad():
        output.weight.zero_()
        output.bias.copy_(bootstrap_action)
    expected_std = torch_module.full_like(
        log_std, FULL_MDP_PPO_RECIPE.init_noise_std
    )
    if (
        int(torch_module.count_nonzero(output.weight).item()) != 0
        or not torch_module.equal(output.bias, bootstrap_action)
        or not torch_module.allclose(
            torch_module.exp(log_std), expected_std, rtol=0.0, atol=1.0e-8
        )
    ):
        raise RuntimeError("MuJoCo Full-A dynamic-ready-head/log-std bootstrap differs")


def _wait_module():
    module = importlib.import_module("mujoco_gpu_ac_full_mdp_initial_wait_env")
    expected = Path(__file__).with_name(
        "mujoco_gpu_ac_full_mdp_initial_wait_env.py"
    ).resolve()
    actual = Path(getattr(module, "__file__", "")).resolve()
    if actual != expected:
        raise RuntimeError("MuJoCo WAIT environment import origin differs")
    return module


def _teacher_replay_module():
    module = importlib.import_module("mujoco_full_mdp_teacher_replay")
    expected = Path(__file__).with_name(
        "mujoco_full_mdp_teacher_replay.py"
    ).resolve()
    if Path(getattr(module, "__file__", "")).resolve() != expected:
        raise RuntimeError("MuJoCo teacher replay helper origin differs")
    return module


def _update_ledger_module():
    module = importlib.import_module("mujoco_full_mdp_update_ledger")
    expected = Path(__file__).with_name("mujoco_full_mdp_update_ledger.py").resolve()
    if Path(getattr(module, "__file__", "")).resolve() != expected:
        raise RuntimeError("MuJoCo FullMDP update ledger import origin differs")
    return module


def _rollout_storage_views(storage) -> tuple[dict, object]:
    """Map the pinned RSL3 RolloutStorage object to the ledger wire names."""
    observations = getattr(storage, "observations", None)

    def observation(name):
        try:
            return observations[name]
        except (KeyError, TypeError):
            return None

    return {
        "observations_policy": observation("policy"),
        "observations_critic": observation("critic"),
        "actions": getattr(storage, "actions", None),
        "values": getattr(storage, "values", None),
        "actions_log_prob": getattr(storage, "actions_log_prob", None),
        "mu": getattr(storage, "mu", None),
        "sigma": getattr(storage, "sigma", None),
        "rewards": getattr(storage, "rewards", None),
        "returns": getattr(storage, "returns", None),
        "advantages": getattr(storage, "advantages", None),
    }, getattr(storage, "dones", None)


def _full_a_rollout_storage_finite(
    storage, *, ledger_module, torch_module, num_steps: int, num_envs: int,
    device,
) -> bool:
    """Recheck the complete Full-A storage ABI for the final completion seal."""
    tensors, dones = _rollout_storage_views(storage)
    if not ledger_module.storage_schema_is_exact(
        torch_module, num_steps=num_steps, num_envs=num_envs, device=device,
        storage_tensors=tensors, storage_dones=dones,
    ):
        return False
    health = torch_module.stack(tuple(
        torch_module.isfinite(tensors[name]).all()
        for name, _width in ledger_module.STORAGE_FLOAT_WIDTHS
    ) + ledger_module.storage_domain_validity(tensors, dones))
    return bool(health.all())


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


def _full_a_artifact_root(
    evidence_jsonl: str, snapshot_dir: str | None, completion_json: str | None,
) -> Path:
    """Resolve the one run root used by every relative artifact locator."""

    evidence = Path(evidence_jsonl)
    root = evidence.parent
    try:
        row = root.lstat()
        if (
            not evidence.is_absolute()
            or root.resolve(strict=True) != root
            or not stat.S_ISDIR(row.st_mode)
            or stat.S_ISLNK(row.st_mode)
        ):
            raise ValueError("MuJoCo Full-A artifact root differs")
    except OSError as exc:
        raise ValueError("MuJoCo Full-A artifact root differs") from exc
    if completion_json is not None:
        completion = Path(completion_json)
        if not completion.is_absolute() or completion.parent != root:
            raise ValueError("MuJoCo Full-A artifact root differs")
    if snapshot_dir is not None:
        snapshots = Path(snapshot_dir)
        if not snapshots.is_absolute() or snapshots.parent != root:
            raise ValueError("MuJoCo Full-A artifact root differs")
    return root


def _fresh_controller_trace_root(raw: str | None) -> Path:
    """Create one absent, canonical directory for a no-clobber diagnostic."""

    if type(raw) is not str or not raw:
        raise ValueError("controller trace output root is not bound")
    root = Path(raw)
    if not root.is_absolute() or root.name in ("", ".", "..") or os.path.lexists(root):
        raise ValueError("controller trace output root must be one absent absolute path")
    try:
        parent = root.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("controller trace output parent differs") from exc
    if parent != root.parent:
        raise ValueError("controller trace output parent differs")
    root.mkdir(mode=0o700)
    row = root.lstat()
    if (
        not stat.S_ISDIR(row.st_mode)
        or stat.S_ISLNK(row.st_mode)
        or root.resolve(strict=True) != root
    ):
        raise ValueError("controller trace output root differs")
    return root


def _write_fresh_bytes(path: Path, payload: bytes) -> None:
    """Durably write one regular no-clobber diagnostic artifact."""

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        row = os.fstat(fd)
        current = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(row.st_mode)
            or row.st_nlink != 1
            or row.st_size != len(payload)
            or (row.st_dev, row.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError("controller trace artifact differs")
    finally:
        os.close(fd)


def _run_controller_trace(
    *, runner, env, root: Path, checkpoint: str, steps: int,
    action_tape: str | None, termination_bit_contract: dict[str, int],
    torch_module,
) -> dict:
    """Trace the live MuJoCo PD owner under a frozen policy/action tape."""

    import io
    import numpy as np

    checkpoint_path = Path(checkpoint)
    if (
        not checkpoint_path.is_absolute()
        or checkpoint_path.resolve(strict=True) != checkpoint_path
        or not stat.S_ISREG(checkpoint_path.lstat().st_mode)
        or checkpoint_path.lstat().st_nlink != 1
    ):
        raise ValueError("controller trace checkpoint differs")
    checkpoint_sha256 = _stable_file_sha256(checkpoint_path)
    runner.load(str(checkpoint_path), load_optimizer=False, map_location=env.device)
    runner.eval_mode()

    selected_names = ("waist_pitch_joint", "waist_roll_joint", "left_ankle_roll_joint")
    names = tuple(env._action_joint_names)
    try:
        selected = tuple(names.index(name) for name in selected_names)
    except ValueError as exc:
        raise RuntimeError("controller trace selected joint contract differs") from exc

    replay_actions = None
    action_tape_sha256 = None
    if action_tape is not None:
        tape_path = Path(action_tape)
        if (
            not tape_path.is_absolute()
            or tape_path.resolve(strict=True) != tape_path
            or not stat.S_ISREG(tape_path.lstat().st_mode)
            or tape_path.lstat().st_nlink != 1
        ):
            raise ValueError("controller trace action tape differs")
        action_tape_sha256 = _stable_file_sha256(tape_path)
        with np.load(tape_path, allow_pickle=False) as archive:
            replay_actions = np.asarray(archive["actions"], dtype=np.float32)
        if replay_actions.shape != (steps, env.num_envs, env.num_actions):
            raise ValueError("controller trace replay action shape differs")
        if not np.isfinite(replay_actions).all():
            raise ValueError("controller trace replay actions are non-finite")

    env.enable_controller_trace()
    obs = env.get_observations()
    torch_module.manual_seed(20260829)
    rows = {name: [] for name in (
        "actions", "q_before", "dq_before", "pre_clamp_qdes",
        "executable_qdes", "q_min", "q_max", "q_after", "dq_after",
        "tau_raw_first", "tau_clamped_first", "tau_raw_last",
        "tau_clamped_last", "tau_raw_abs_max", "tau_clamped_abs_max",
        "tau_clamp_count", "hard_lower", "hard_upper",
    )}
    phases, dones, termination_bits = [], [], []
    selected_tensor = torch_module.tensor(selected, dtype=torch_module.long, device=env.device)
    with torch_module.inference_mode():
        for step in range(steps):
            if replay_actions is None:
                actions = runner.alg.policy.act(obs)
            else:
                actions = torch_module.from_numpy(replay_actions[step]).to(env.device)
            obs, _reward, done, extras = env.step(actions)
            trace = env.controller_trace()
            rows["actions"].append(actions.detach().cpu().numpy().copy())
            for name in rows:
                if name == "actions":
                    continue
                value = trace[name].index_select(1, selected_tensor)
                rows[name].append(value.detach().cpu().numpy().copy())
            phase = extras.get("full_a_phase_before_reset")
            bits = extras.get("termination_bits")
            if (
                not torch_module.is_tensor(phase)
                or not torch_module.is_tensor(bits)
                or tuple(bits.shape) != (env.num_envs,)
            ):
                raise RuntimeError("controller trace lifecycle surface differs")
            phases.append(phase.detach().cpu().numpy().copy())
            dones.append(done.detach().cpu().numpy().copy())
            termination_bits.append(bits.detach().cpu().numpy().copy())

    arrays = {name: np.stack(values) for name, values in rows.items()}
    arrays["phase"] = np.stack(phases)
    arrays["done"] = np.stack(dones)
    arrays["termination_bits"] = np.stack(termination_bits)
    arrays["selected_joint_indices"] = np.asarray(selected, dtype=np.int64)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    trace_path = root / "controller_trace.npz"
    _write_fresh_bytes(trace_path, buffer.getvalue())
    trace_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()

    hard = np.logical_or(arrays["hard_lower"], arrays["hard_upper"])
    clamp = arrays["tau_clamp_count"] > 0
    per_joint = []
    for ordinal, name in enumerate(selected_names):
        per_joint.append({
            "joint_name": name,
            "hard_rows": int(hard[:, :, ordinal].sum()),
            "lower_rows": int(arrays["hard_lower"][:, :, ordinal].sum()),
            "upper_rows": int(arrays["hard_upper"][:, :, ordinal].sum()),
            "torque_clamp_rows": int(clamp[:, :, ordinal].sum()),
            "maximum_abs_tau_raw": float(arrays["tau_raw_abs_max"][:, :, ordinal].max()),
            "maximum_abs_tau_clamped": float(
                arrays["tau_clamped_abs_max"][:, :, ordinal].max()
            ),
        })
    phase_counts = {
        str(int(code)): int((arrays["phase"] == code).sum())
        for code in np.unique(arrays["phase"])
    }
    if (
        type(termination_bit_contract) is not dict
        or not termination_bit_contract
        or any(
            type(name) is not str or not name or type(bit) is not int or bit <= 0
            for name, bit in termination_bit_contract.items()
        )
        or len(set(termination_bit_contract.values()))
        != len(termination_bit_contract)
    ):
        raise RuntimeError("controller trace termination-bit contract differs")
    termination_reason_rows = {
        name: int(np.count_nonzero(arrays["termination_bits"] & bit))
        for name, bit in termination_bit_contract.items()
    }
    known_termination_mask = sum(termination_bit_contract.values())
    termination_reason_rows["unknown_bits"] = int(np.count_nonzero(
        arrays["termination_bits"] & ~known_termination_mask
    ))
    termination_reason_rows["done_without_reason"] = int(np.count_nonzero(
        arrays["done"] & (arrays["termination_bits"] == 0)
    ))
    summary = {
        "schema_version": 1,
        "kind": "action_ball_mujoco_full_mdp_controller_trace_v1",
        "diagnostic_unauthorized": True,
        "training_authorized": False,
        "checkpoint_authority": False,
        "num_envs": int(env.num_envs),
        "policy_steps": int(steps),
        "joint_names": list(selected_names),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "action_source": "replay_npz" if replay_actions is not None else "seeded_checkpoint_policy",
        "input_action_tape_sha256": action_tape_sha256,
        "trace_npz": trace_path.name,
        "trace_npz_sha256": trace_sha256,
        "sample_rows": int(steps * env.num_envs),
        "done_rows": int(np.count_nonzero(arrays["done"])),
        "termination_reason_rows": termination_reason_rows,
        "phase_counts": phase_counts,
        "per_joint": per_joint,
    }
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    _write_fresh_bytes(root / "summary.json", payload + b"\n")
    return summary


def _direct_frame0_validation_report(arrays: dict) -> dict:
    """Evaluate every direct-frame0 post-run contract without hiding a field.

    The report is JSON-safe even when a measured float is non-finite.  Exact
    array predicates retain dtype/shape/content fingerprints and expose a
    finite max-absolute delta plus mismatch count instead of dumping a 31-DOF
    vector into an exception.
    """

    import numpy as np

    decoder_abi_atol_rad = float(
        _teacher_replay_module().TEACHER_REPLAY_DECODER_ABI_ATOL_RAD
    )
    required_bits = (
        "direct_frame0_ball_unchanged",
        "direct_frame0_task_unchanged",
        "direct_frame0_lifecycle_unchanged",
        "direct_frame0_frame0_pose_exact",
        "direct_frame0_robot_velocity_zero",
        "direct_frame0_robot_qacc_warmstart_zero",
        "direct_frame0_ctrl_zero",
        "direct_frame0_controller_history_exact",
        "direct_frame0_teacher_cache_refreshed",
        "direct_frame0_actuator_state_absent",
    )
    predicates: dict[str, dict[str, object]] = {}

    def add_exact(name, actual, expected) -> None:
        if isinstance(expected, bool):
            actual_value = bool(actual)
            expected_value = bool(expected)
            delta = int(actual_value) - int(expected_value)
        else:
            actual_value = int(actual)
            expected_value = int(expected)
            delta = actual_value - expected_value
        predicates[name] = {
            "evaluated": True,
            "actual": actual_value,
            "expected": expected_value,
            "comparison": "equal",
            "delta": delta,
            "passed": actual_value == expected_value,
        }

    def add_float(name, actual, expected=0.0) -> None:
        value = float(actual)
        finite = bool(np.isfinite(value))
        expected_value = float(expected)
        predicates[name] = {
            "evaluated": True,
            "actual": value if finite else None,
            "expected": expected_value,
            "comparison": "equal",
            "delta": value - expected_value if finite else None,
            "passed": finite and value == expected_value,
        }

    def add_float_at_most(name, actual, maximum) -> None:
        value = float(actual)
        finite = bool(np.isfinite(value))
        maximum_value = float(maximum)
        predicates[name] = {
            "evaluated": True,
            "actual": value if finite else None,
            "expected": maximum_value,
            "comparison": "less_than_or_equal",
            "delta": value - maximum_value if finite else None,
            "passed": finite and value <= maximum_value,
        }

    def fingerprint(value) -> dict[str, object]:
        row = np.ascontiguousarray(value)
        return {
            "dtype": str(row.dtype),
            "shape": [int(size) for size in row.shape],
            "sha256": hashlib.sha256(row.tobytes(order="C")).hexdigest(),
        }

    def add_array(name, actual, expected) -> None:
        left = np.asarray(actual)
        right = np.asarray(expected)
        same_shape = left.shape == right.shape
        mismatch_count = (
            int(np.count_nonzero(left != right)) if same_shape else None
        )
        finite = bool(
            same_shape
            and np.issubdtype(left.dtype, np.number)
            and np.issubdtype(right.dtype, np.number)
            and np.isfinite(left).all()
            and np.isfinite(right).all()
        )
        max_abs = (
            float(np.max(np.abs(left - right)))
            if finite and left.size > 0
            else (0.0 if finite else None)
        )
        predicates[name] = {
            "evaluated": True,
            "actual": fingerprint(left),
            "expected": fingerprint(right),
            "comparison": "array_equal",
            "delta": {
                "mismatch_count": mismatch_count,
                "max_abs": max_abs if max_abs is None or np.isfinite(max_abs)
                else None,
            },
            "passed": same_shape and bool(np.array_equal(left, right)),
        }

    def max_abs_delta(actual, expected) -> float:
        left = np.asarray(actual)
        right = np.asarray(expected)
        if (
            left.shape != right.shape
            or left.size == 0
            or not np.issubdtype(left.dtype, np.number)
            or not np.issubdtype(right.dtype, np.number)
        ):
            return float("nan")
        return float(np.max(np.abs(left - right)))

    def add_not_evaluated(name, expected, reason) -> None:
        predicates[name] = {
            "evaluated": False,
            "actual": None,
            "expected": expected,
            "comparison": "not_evaluated",
            "delta": None,
            "passed": None,
            "reason_not_evaluated": reason,
        }

    intervention_rows = np.flatnonzero(
        np.asarray(arrays["direct_frame0_intervention"])[:, 0]
    )
    add_exact(
        "intervention_count_exactly_one", int(intervention_rows.size), 1
    )
    index = int(intervention_rows[0]) if intervention_rows.size > 0 else None
    if index is None:
        for name in required_bits:
            add_not_evaluated(name, True, "no intervention row")
        for name, expected in (
            ("same_step_requested_q0_error_zero", 0.0),
            (
                "same_step_executable_guard_expected_exact",
                {
                    "relation": "array_equal",
                    "rhs": "guard_expected_executable_qdes[same_row]",
                },
            ),
            (
                "same_step_executable_teacher_error_within_decoder_abi",
                decoder_abi_atol_rad,
            ),
            ("same_step_qdes_guard_clear", False),
            ("same_step_done_clear", False),
            ("installed_frame0_table_keepout_clear", False),
            ("installed_frame0_resolved_table_contact_clear", False),
            ("joint_q0_error_after_zero", 0.0),
            ("root_position_q0_error_after_zero", 0.0),
            ("root_quaternion_q0_error_after_zero", 0.0),
            ("post_transition_teacher_frame_one", 1),
            ("next_trace_row_exists", True),
            ("next_pre_teacher_frame_one", 1),
            ("next_intervention_clear", False),
            (
                "next_requested_natural_teacher_exact",
                {
                    "relation": "array_equal",
                    "rhs": "pre_teacher_qdes[next_row]",
                },
            ),
            (
                "next_executable_guard_expected_exact",
                {
                    "relation": "array_equal",
                    "rhs": "guard_expected_executable_qdes[next_row]",
                },
            ),
            (
                "next_executable_teacher_error_within_decoder_abi",
                decoder_abi_atol_rad,
            ),
            ("next_qdes_guard_clear", False),
        ):
            add_not_evaluated(name, expected, "no intervention row")
        next_index = None
    else:
        for name in required_bits:
            add_exact(name, arrays[name][index, 0], True)
        add_float(
            "same_step_requested_q0_error_zero",
            arrays["direct_frame0_requested_q0_error_max_rad"][index, 0],
        )
        add_array(
            "same_step_executable_guard_expected_exact",
            arrays["executable_qdes"][index],
            arrays["guard_expected_executable_qdes"][index],
        )
        add_float_at_most(
            "same_step_executable_teacher_error_within_decoder_abi",
            arrays["direct_frame0_executable_q0_error_max_rad"][index, 0],
            decoder_abi_atol_rad,
        )
        add_exact(
            "same_step_qdes_guard_clear",
            arrays["qdes_guard_intervention"][index, 0],
            False,
        )
        add_exact("same_step_done_clear", arrays["done"][index, 0], False)
        add_exact(
            "installed_frame0_table_keepout_clear",
            arrays["direct_frame0_installed_table_keepout"][index, 0],
            False,
        )
        add_exact(
            "installed_frame0_resolved_table_contact_clear",
            arrays[
                "direct_frame0_installed_backend_resolved_table_contact"
            ][index, 0],
            False,
        )
        add_float(
            "joint_q0_error_after_zero",
            arrays["direct_frame0_joint_q0_error_max_after_rad"][index, 0],
        )
        add_float(
            "root_position_q0_error_after_zero",
            arrays["direct_frame0_root_position_q0_error_after_m"][index, 0],
        )
        add_float(
            "root_quaternion_q0_error_after_zero",
            arrays["direct_frame0_root_quaternion_q0_error_after"][index, 0],
        )
        add_exact(
            "post_transition_teacher_frame_one",
            arrays["post_teacher_frame"][index, 0],
            1,
        )
        next_index = index + 1
        next_exists = next_index < int(arrays["common_step"].shape[0])
        add_exact("next_trace_row_exists", next_exists, True)
        if next_exists:
            add_exact(
                "next_pre_teacher_frame_one",
                arrays["pre_teacher_frame"][next_index, 0],
                1,
            )
            add_exact(
                "next_intervention_clear",
                arrays["direct_frame0_intervention"][next_index, 0],
                False,
            )
            add_array(
                "next_requested_natural_teacher_exact",
                arrays["requested_qdes"][next_index],
                arrays["pre_teacher_qdes"][next_index],
            )
            add_array(
                "next_executable_guard_expected_exact",
                arrays["executable_qdes"][next_index],
                arrays["guard_expected_executable_qdes"][next_index],
            )
            add_float_at_most(
                "next_executable_teacher_error_within_decoder_abi",
                max_abs_delta(
                    arrays["executable_qdes"][next_index],
                    arrays["pre_teacher_qdes"][next_index],
                ),
                decoder_abi_atol_rad,
            )
            add_exact(
                "next_qdes_guard_clear",
                arrays["qdes_guard_intervention"][next_index, 0],
                False,
            )
        else:
            for name, expected in (
                ("next_pre_teacher_frame_one", 1),
                ("next_intervention_clear", False),
                (
                    "next_requested_natural_teacher_exact",
                    {
                        "relation": "array_equal",
                        "rhs": "pre_teacher_qdes[next_row]",
                    },
                ),
                (
                    "next_executable_guard_expected_exact",
                    {
                        "relation": "array_equal",
                        "rhs": "guard_expected_executable_qdes[next_row]",
                    },
                ),
                (
                    "next_executable_teacher_error_within_decoder_abi",
                    decoder_abi_atol_rad,
                ),
                ("next_qdes_guard_clear", False),
            ):
                add_not_evaluated(name, expected, "next trace row absent")

    failed = [
        name for name, row in predicates.items() if row["passed"] is False
    ]
    not_evaluated = [
        name for name, row in predicates.items() if not row["evaluated"]
    ]
    return {
        "schema_version": 2,
        "kind": "action_ball_mujoco_direct_frame0_validation_v2",
        "diagnostic_unauthorized": True,
        "training_authorized": False,
        "ppo_update_calls": 0,
        "decoder_abi_atol_rad": decoder_abi_atol_rad,
        "trace_rows": int(arrays["common_step"].shape[0]),
        "intervention_rows": [int(value) for value in intervention_rows],
        "trace_row": index,
        "next_trace_row": next_index,
        "predicates": predicates,
        "failed_predicates": failed,
        "not_evaluated_predicates": not_evaluated,
        "all_passed": not failed and not not_evaluated,
    }


def _write_direct_frame0_validation_failure(
    root: Path, report: dict, *, identity_context: dict
) -> tuple[Path, str]:
    """Persist one independently identifiable failure before raising."""

    if report.get("all_passed") is not False:
        raise ValueError("direct frame-zero validation failure report differs")
    identity_keys = {
        "run", "source", "runtime", "motion", "ready", "task", "plant",
    }
    if (
        type(identity_context) is not dict
        or set(identity_context) != identity_keys
        or any(type(identity_context[name]) is not dict for name in identity_keys)
    ):
        raise ValueError("direct frame-zero failure identity differs")
    document = {
        "schema_version": 1,
        "kind": "action_ball_mujoco_direct_frame0_validation_failure_v1",
        "diagnostic_unauthorized": True,
        "training_authorized": False,
        "ppo_update_calls": 0,
        "payload_sha256_scope": (
            "canonical_json_of_this_document_without_payload_sha256"
        ),
        "identity": identity_context,
        "validation": report,
    }
    unsigned_payload = json.dumps(
        document,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload_sha256 = hashlib.sha256(unsigned_payload).hexdigest()
    document["payload_sha256"] = payload_sha256
    payload = json.dumps(
        document,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path = root / "direct_frame0_validation_failure.json"
    _write_fresh_bytes(path, payload)
    directory_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path, payload_sha256


def _validate_or_raise_direct_frame0(
    *, arrays: dict, root: Path, identity_context: dict
) -> dict:
    """Run the production validator and durably identify every failure."""

    validation = _direct_frame0_validation_report(arrays)
    if validation["all_passed"]:
        return validation
    failure_path, failure_sha256 = _write_direct_frame0_validation_failure(
        root, validation, identity_context=identity_context
    )
    failed = ",".join(validation["failed_predicates"]) or "none"
    not_evaluated = ",".join(
        validation["not_evaluated_predicates"]
    ) or "none"
    marker = {"artifact": str(failure_path), "payload_sha256": failure_sha256}
    _best_effort_stdout_marker(
        "ACTION_BALL_MUJOCO_DIRECT_FRAME0_FAILURE_JSON="
        + json.dumps(marker, sort_keys=True, separators=(",", ":"))
    )
    raise RuntimeError(
        "direct frame-zero validation differs: "
        f"artifact={failure_path} payload_sha256={failure_sha256} "
        f"failed={failed} not_evaluated={not_evaluated}"
    )


def _run_teacher_replay(
    *, env, root: Path, identity: dict, ready_pose_payload: bytes,
    steps: int, handoff_mode: str, torch_module, physical_root_xy_m=None,
) -> dict:
    """Drive the live FullMDP owner with its own frozen measured teacher."""

    import io
    import numpy as np

    helper = _teacher_replay_module()
    if (
        env.num_envs != helper.TEACHER_REPLAY_NUM_ENVS
        or steps <= 0
        or helper.TEACHER_REPLAY_ACTION_DELAY_STEPS != 0
        or handoff_mode not in helper.TEACHER_REPLAY_HANDOFF_MODES
    ):
        raise ValueError("teacher replay dimensions/delay differ")
    env.enable_controller_trace()
    env.enable_diagnostic_first_generic_contact_patch()
    try:
        runtime_mjb_sha256 = identity["plant_model"]["runtime_attach"][
            "final_augmented_mjb"
        ]["sha256"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "teacher replay runtime MJB identity differs"
        ) from exc
    env.enable_diagnostic_exact_table_narrowphase(
        runtime_mjb_sha256=runtime_mjb_sha256
    )
    direct_frame0 = (
        handoff_mode == helper.TEACHER_REPLAY_HANDOFF_DIRECT_FRAME0
    )
    if direct_frame0:
        env.enable_diagnostic_direct_frame0_playback(
            physical_root_xy_m=physical_root_xy_m
        )
    elif physical_root_xy_m is not None:
        raise ValueError(
            "teacher replay root XY requires direct frame-zero playback"
        )
    previous_qdes = env._full_a_policy_bootstrap_qdes.unsqueeze(0).clone()
    hold_qdes = previous_qdes.clone()
    rows = {name: [] for name in (
        "common_step", "phase", "pre_task_valid", "pre_teacher_frame",
        "pre_motion_phase", "pre_teacher_qdes", "pre_reveal_tick",
        "pre_swing_wait_s", "post_task_valid", "post_teacher_frame",
        "post_motion_phase", "frozen_steps", "requested_qdes", "raw_action",
        "executable_qdes", "guard_expected_executable_qdes",
        "actual_joint_pos", "actual_joint_vel",
        "ball_center_w", "ball_lin_vel_w", "racket_site_w",
        "racket_velocity_w", "racket_normal_w", "scheduled_due", "reveal",
        "launch", "r03", "generic_contact", "selected_contact",
        "opposite_contact", "edge_contact", "between_contact",
        "invalid_contact", "crossing", "legal_landing", "done",
        "termination_bits", "qdes_guard_intervention", "keepout_source",
        "backend_resolved_table_contact",
        "resolved_any_substep",
    )}
    if direct_frame0:
        rows.update({name: [] for name in (
            "direct_frame0_intervention",
            "direct_frame0_joint_q0_error_max_before_rad",
            "direct_frame0_joint_q0_error_max_after_rad",
            "direct_frame0_root_position_q0_error_before_m",
            "direct_frame0_root_position_q0_error_after_m",
            "direct_frame0_root_quaternion_q0_error_before",
            "direct_frame0_root_quaternion_q0_error_after",
            "direct_frame0_ball_unchanged",
            "direct_frame0_task_unchanged",
            "direct_frame0_lifecycle_unchanged",
            "direct_frame0_frame0_pose_exact",
            "direct_frame0_robot_velocity_zero",
            "direct_frame0_robot_qacc_warmstart_zero",
            "direct_frame0_ctrl_zero",
            "direct_frame0_controller_history_exact",
            "direct_frame0_teacher_cache_refreshed",
            "direct_frame0_actuator_state_absent",
            "direct_frame0_installed_table_keepout",
            "direct_frame0_installed_backend_resolved_table_contact",
            "direct_frame0_requested_q0_error_max_rad",
            "direct_frame0_executable_q0_error_max_rad",
        )})
    direct_frame0_applied = None
    if direct_frame0:
        direct_frame0_applied = torch_module.zeros(
            env.num_envs, dtype=torch_module.bool, device=env.device
        )
    installed_physical_root_xy_m = None
    question = launch = None
    question_reset_generation = None
    shot_boundary_reason = None
    pre_due_done_count = 0
    with torch_module.inference_mode():
        for _ordinal in range(steps):
            pre = helper.capture_teacher_replay_pre_step(env)
            frozen = helper.remaining_teacher_frozen_steps(
                torch=torch_module,
                common_step=int(env.common_step_counter),
                reveal_tick=pre.reveal_tick,
                pre_swing_wait_s=pre.pre_swing_wait_s,
                step_dt=float(env.step_dt),
            )
            direct_receipt = None
            if direct_frame0:
                handoff_due = helper.direct_frame0_handoff_due(
                    torch=torch_module,
                    task_valid=pre.task_valid,
                    teacher_frame=pre.teacher_frame,
                    frozen_steps=frozen,
                    already_applied=direct_frame0_applied,
                )
                handoff_ids = handoff_due.nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if handoff_ids.numel() > 0:
                    direct_receipt = (
                        env.install_diagnostic_direct_frame0_playback(
                            handoff_ids
                        )
                    )
                    installed_physical_root_xy_m = [
                        float(value) for value in direct_receipt[
                            "physical_root_xy_m"
                        ][0].detach().cpu().tolist()
                    ]
                    direct_frame0_applied |= handoff_due
                requested = helper.direct_frame0_teacher_qdes(
                    torch=torch_module,
                    task_valid=pre.task_valid,
                    hold_qdes=hold_qdes,
                    teacher_qdes=pre.teacher_qdes,
                    frozen_steps=frozen,
                )
            else:
                requested = helper.frozen_teacher_qdes(
                    torch=torch_module,
                    task_valid=pre.task_valid,
                    hold_qdes=hold_qdes,
                    previous_qdes=previous_qdes,
                    teacher_qdes=pre.teacher_qdes,
                    frozen_steps=frozen,
                    bridge=(
                        _wait_module().portable_question
                        .step_diagnostic_split_ready_qdes_bridge
                    ),
                )
            action = helper.decode_teacher_qdes_to_action(
                torch=torch_module,
                qdes=requested,
                action_offset=env.action_offset,
                action_scale=env.act_scale,
            )
            _obs, _reward, done, extras = env.step(action)
            direct_same_step_invalid = direct_receipt is not None and (
                bool(done[0])
                or not bool(env._full_a_teacher_frame[0].eq(1))
            )
            table_attribution = env.diagnostic_table_attribution_tick(
                extras["termination_bits"]
            )
            if table_attribution["backend_resolved_table_contact"] != bool(
                extras["backend_resolved_table_contact"][0]
            ):
                raise RuntimeError("teacher replay resolved-table source differs")
            trace = env.controller_trace()
            racket = env._full_a_racket_kinematics()
            if bool(extras["full_a_reveal_event"][0]) and question is None:
                question = env._epoch_task_f32[0].detach().clone()
                launch = env._full_a_launch_state_f32[0].detach().clone()
                question_reset_generation = int(
                    extras["reset_generation"][0].detach().cpu().item()
                )

            def host(value):
                return value.detach().cpu().numpy().copy()

            rows["common_step"].append(int(env.common_step_counter))
            rows["phase"].append(host(extras["full_a_phase_before_reset"]))
            rows["pre_task_valid"].append(host(pre.task_valid))
            rows["pre_teacher_frame"].append(host(pre.teacher_frame))
            rows["pre_motion_phase"].append(host(pre.motion_phase))
            rows["pre_teacher_qdes"].append(host(pre.teacher_qdes))
            rows["pre_reveal_tick"].append(host(pre.reveal_tick))
            rows["pre_swing_wait_s"].append(host(pre.pre_swing_wait_s))
            rows["post_task_valid"].append(host(env._epoch_task_valid))
            rows["post_teacher_frame"].append(host(env._full_a_teacher_frame))
            rows["post_motion_phase"].append(host(env._full_a_motion_phase_code))
            rows["frozen_steps"].append(host(frozen))
            rows["requested_qdes"].append(host(requested))
            rows["raw_action"].append(host(action))
            rows["executable_qdes"].append(host(trace["executable_qdes"]))
            rows["guard_expected_executable_qdes"].append(host(
                trace["nominal_projected_qdes"]
            ))
            rows["actual_joint_pos"].append(host(env._qpos_act()))
            rows["actual_joint_vel"].append(host(env._qvel_act()))
            rows["ball_center_w"].append(host(env.sim.data.qpos[:, env.b_q:env.b_q + 3]))
            rows["ball_lin_vel_w"].append(host(env.sim.data.qvel[:, env.b_v:env.b_v + 3]))
            rows["racket_site_w"].append(host(racket[0] + env.env.scene.env_origins))
            rows["racket_velocity_w"].append(host(racket[1]))
            rows["racket_normal_w"].append(host(racket[2]))
            event_names = {
                "scheduled_due": "full_a_scheduled_due_event",
                "reveal": "full_a_reveal_event", "launch": "full_a_launch_event",
                "r03": "full_a_r03_present_event",
                "generic_contact": "full_a_racket_contact_event",
                "selected_contact": "full_a_selected_contact_event",
                "opposite_contact": "full_a_opposite_contact_event",
                "edge_contact": "full_a_edge_contact_event",
                "between_contact": "full_a_between_contact_event",
                "invalid_contact": "full_a_invalid_contact_event",
                "crossing": "full_a_landing_crossing_event",
            }
            for name, key in event_names.items():
                rows[name].append(host(extras[key]))
            rows["legal_landing"].append(host(
                extras["full_a_landing_on_opponent"]
                & extras["full_a_landing_opponent_bound"]
            ))
            rows["done"].append(host(done))
            rows["termination_bits"].append(host(extras["termination_bits"]))
            rows["qdes_guard_intervention"].append(host(
                extras["full_a_qdes_guard_intervention_event"]
            ))
            rows["keepout_source"].append([
                table_attribution["keepout_source"]
            ])
            rows["backend_resolved_table_contact"].append([
                table_attribution["backend_resolved_table_contact"]
            ])
            rows["resolved_any_substep"].append([
                table_attribution["resolved_any_substep"]
            ])
            if direct_frame0:
                direct_bool = torch_module.zeros(
                    env.num_envs,
                    dtype=torch_module.bool,
                    device=env.device,
                )
                direct_float = torch_module.zeros(
                    env.num_envs,
                    dtype=requested.dtype,
                    device=env.device,
                )
                direct_values = {
                    "direct_frame0_intervention": direct_bool,
                    "direct_frame0_joint_q0_error_max_before_rad": direct_float,
                    "direct_frame0_joint_q0_error_max_after_rad": direct_float,
                    "direct_frame0_root_position_q0_error_before_m": direct_float,
                    "direct_frame0_root_position_q0_error_after_m": direct_float,
                    "direct_frame0_root_quaternion_q0_error_before": direct_float,
                    "direct_frame0_root_quaternion_q0_error_after": direct_float,
                    "direct_frame0_ball_unchanged": direct_bool,
                    "direct_frame0_task_unchanged": direct_bool,
                    "direct_frame0_lifecycle_unchanged": direct_bool,
                    "direct_frame0_frame0_pose_exact": direct_bool,
                    "direct_frame0_robot_velocity_zero": direct_bool,
                    "direct_frame0_robot_qacc_warmstart_zero": direct_bool,
                    "direct_frame0_ctrl_zero": direct_bool,
                    "direct_frame0_controller_history_exact": direct_bool,
                    "direct_frame0_teacher_cache_refreshed": direct_bool,
                    "direct_frame0_actuator_state_absent": direct_bool,
                    "direct_frame0_installed_table_keepout": direct_bool,
                    "direct_frame0_installed_backend_resolved_table_contact": (
                        direct_bool
                    ),
                    "direct_frame0_requested_q0_error_max_rad": direct_float,
                    "direct_frame0_executable_q0_error_max_rad": direct_float,
                }
                if direct_receipt is not None:
                    direct_values.update({
                        "direct_frame0_intervention": direct_receipt["applied"],
                        "direct_frame0_joint_q0_error_max_before_rad": (
                            direct_receipt[
                                "joint_q0_error_max_before_rad"
                            ]
                        ),
                        "direct_frame0_joint_q0_error_max_after_rad": (
                            direct_receipt[
                                "joint_q0_error_max_after_rad"
                            ]
                        ),
                        "direct_frame0_root_position_q0_error_before_m": (
                            direct_receipt[
                                "root_position_q0_error_before_m"
                            ]
                        ),
                        "direct_frame0_root_position_q0_error_after_m": (
                            direct_receipt[
                                "root_position_q0_error_after_m"
                            ]
                        ),
                        "direct_frame0_root_quaternion_q0_error_before": (
                            direct_receipt[
                                "root_quaternion_q0_error_before"
                            ]
                        ),
                        "direct_frame0_root_quaternion_q0_error_after": (
                            direct_receipt[
                                "root_quaternion_q0_error_after"
                            ]
                        ),
                        "direct_frame0_ball_unchanged": direct_receipt[
                            "ball_unchanged"
                        ],
                        "direct_frame0_task_unchanged": direct_receipt[
                            "task_unchanged"
                        ],
                        "direct_frame0_lifecycle_unchanged": direct_receipt[
                            "lifecycle_unchanged"
                        ],
                        "direct_frame0_frame0_pose_exact": direct_receipt[
                            "frame0_pose_exact"
                        ],
                        "direct_frame0_robot_velocity_zero": direct_receipt[
                            "robot_velocity_zero"
                        ],
                        "direct_frame0_robot_qacc_warmstart_zero": direct_receipt[
                            "robot_qacc_warmstart_zero"
                        ],
                        "direct_frame0_ctrl_zero": direct_receipt[
                            "ctrl_zero"
                        ],
                        "direct_frame0_controller_history_exact": direct_receipt[
                            "controller_history_exact"
                        ],
                        "direct_frame0_teacher_cache_refreshed": direct_receipt[
                            "teacher_cache_refreshed"
                        ],
                        "direct_frame0_actuator_state_absent": direct_receipt[
                            "actuator_state_absent"
                        ],
                        "direct_frame0_installed_table_keepout": (
                            direct_receipt["installed_frame0_table_keepout"]
                        ),
                        (
                            "direct_frame0_installed_backend_resolved_"
                            "table_contact"
                        ): direct_receipt[
                            "installed_frame0_backend_resolved_table_contact"
                        ],
                        "direct_frame0_requested_q0_error_max_rad": (
                            torch_module.max(
                                torch_module.abs(
                                    requested
                                    - direct_receipt["frame0_joint_qdes"]
                                ),
                                dim=1,
                            ).values
                        ),
                        "direct_frame0_executable_q0_error_max_rad": (
                            torch_module.max(
                                torch_module.abs(
                                    trace["executable_qdes"]
                                    - direct_receipt["frame0_joint_qdes"]
                                ),
                                dim=1,
                            ).values
                        ),
                    })
                for name, value in direct_values.items():
                    rows[name].append(host(value))
            previous_qdes = requested
            if direct_same_step_invalid:
                break
            if bool(done[0]):
                if question is None:
                    pre_due_done_count += 1
                else:
                    shot_boundary_reason = "first_done"
                    break
            if bool(extras["full_a_completed_action_epoch_event"][0]):
                shot_boundary_reason = "first_completed_action_epoch"
                break
    contact_patch = env.diagnostic_first_generic_contact_patch()
    no_question = question is None or launch is None
    if no_question:
        if contact_patch.get("present", False):
            raise RuntimeError("contact patch exists without an accepted question")
        shot_boundary_reason = "step_budget_exhausted_no_question"
    else:
        helper.validate_contact_patch_shot(
            contact_patch=contact_patch,
            question_reset_generation=question_reset_generation,
            question_f32_sha256=helper.tensor_f32_sha256(question),
        )
    arrays = {
        name: np.asarray(values) if name == "common_step" else np.stack(values)
        for name, values in rows.items()
    }
    direct_frame0_receipt = None
    if direct_frame0:
        catalog = env._full_a_catalog
        plant = identity["plant_model"]
        failure_identity = {
                "run": {
                    "run_namespace": identity["run_namespace"],
                    "artifact_root": str(root),
                    "teacher_handoff_mode": handoff_mode,
                },
                "source": {"commit": identity["source_commit"]},
                "runtime": {
                    "identity": identity["runtime_stack"],
                    "canonical_sha256": _canonical_payload_sha256(
                        identity["runtime_stack"]
                    ),
                },
                "motion": {
                    "action_id": catalog.fresh_action.action_id,
                    "action_uid": int(catalog.fresh_action.action_uid),
                    "motion_sha256": catalog.fresh_action.motion_sha256,
                    "manifest_file_sha256": catalog.manifest_file_sha256,
                    "manifest_canonical_sha256": (
                        catalog.manifest_canonical_sha256
                    ),
                },
                "ready": {
                    "payload_sha256": hashlib.sha256(
                        ready_pose_payload
                    ).hexdigest(),
                },
                "task": {
                    "accepted_question_present": not no_question,
                    "question_f32_sha256": (
                        None if no_question
                        else helper.tensor_f32_sha256(question)
                    ),
                    "launch_f32_sha256": (
                        None if no_question
                        else helper.tensor_f32_sha256(launch)
                    ),
                    "question_reset_generation": question_reset_generation,
                },
                "plant": {
                    "identity": plant,
                    "canonical_sha256": _canonical_payload_sha256(plant),
                },
        }
        validation = _validate_or_raise_direct_frame0(
            arrays=arrays, root=root, identity_context=failure_identity
        )
        index = int(validation["trace_row"])
        next_index = int(validation["next_trace_row"])
        direct_frame0_receipt = {
            "schema_version": 1,
            "kind": "action_ball_mujoco_direct_frame0_intervention_v1",
            "diagnostic_unauthorized": True,
            "training_authorized": False,
            "physical_root_xy_m": [
                float(value) for value in installed_physical_root_xy_m
            ],
            "ppo_update_calls": 0,
            "applied": True,
            "validation": validation,
            "trace_row": index,
            "common_step_after_transition": int(arrays["common_step"][index]),
            "requested_frame0_same_step": bool(
                arrays[
                    "direct_frame0_requested_q0_error_max_rad"
                ][index, 0]
                == 0.0
            ),
            "executable_frame0_matches_guard_expected": bool(
                validation["predicates"][
                    "same_step_executable_guard_expected_exact"
                ]["passed"]
            ),
            "executable_frame0_teacher_error_max_rad": float(
                validation["predicates"][
                    "same_step_executable_teacher_error_within_decoder_abi"
                ]["actual"]
            ),
            "decoder_abi_atol_rad": float(
                validation["decoder_abi_atol_rad"]
            ),
            "post_transition_teacher_frame": int(
                arrays["post_teacher_frame"][index, 0]
            ),
            "next_action_trace_row": next_index,
            "next_action_teacher_frame": int(
                arrays["pre_teacher_frame"][next_index, 0]
            ),
            "next_action_uses_natural_teacher_qdes": True,
            "next_action_executable_matches_guard_expected": True,
            "next_action_executable_teacher_error_max_rad": float(
                validation["predicates"][
                    "next_executable_teacher_error_within_decoder_abi"
                ]["actual"]
            ),
            "next_action_executable_within_decoder_abi": True,
            "next_action_qdes_guard_intervention": False,
            "joint_q0_error_max_before_rad": float(
                    arrays[
                        "direct_frame0_joint_q0_error_max_before_rad"
                    ][index, 0]
            ),
            "joint_q0_error_max_after_rad": float(
                    arrays[
                        "direct_frame0_joint_q0_error_max_after_rad"
                    ][index, 0]
            ),
            "root_position_q0_error_before_m": float(
                    arrays[
                        "direct_frame0_root_position_q0_error_before_m"
                    ][index, 0]
            ),
            "root_position_q0_error_after_m": float(
                    arrays[
                        "direct_frame0_root_position_q0_error_after_m"
                    ][index, 0]
            ),
            "root_quaternion_q0_error_before": float(
                    arrays[
                        "direct_frame0_root_quaternion_q0_error_before"
                    ][index, 0]
            ),
            "root_quaternion_q0_error_after": float(
                    arrays[
                        "direct_frame0_root_quaternion_q0_error_after"
                    ][index, 0]
            ),
            "ball_unchanged": bool(
                arrays["direct_frame0_ball_unchanged"][index, 0]
            ),
            "task_unchanged": bool(
                arrays["direct_frame0_task_unchanged"][index, 0]
            ),
            "lifecycle_unchanged": bool(
                    arrays[
                        "direct_frame0_lifecycle_unchanged"
                    ][index, 0]
            ),
            "frame0_pose_exact": bool(
                    arrays[
                        "direct_frame0_frame0_pose_exact"
                    ][index, 0]
            ),
            "robot_velocity_zero": bool(
                    arrays[
                        "direct_frame0_robot_velocity_zero"
                    ][index, 0]
            ),
            "robot_qacc_warmstart_zero": bool(
                    arrays[
                        "direct_frame0_robot_qacc_warmstart_zero"
                    ][index, 0]
            ),
            "ctrl_zero": bool(
                    arrays[
                        "direct_frame0_ctrl_zero"
                    ][index, 0]
            ),
            "controller_history_exact": bool(
                    arrays[
                        "direct_frame0_controller_history_exact"
                    ][index, 0]
            ),
            "teacher_cache_refreshed": bool(
                    arrays[
                        "direct_frame0_teacher_cache_refreshed"
                    ][index, 0]
            ),
            "actuator_state_absent": bool(
                    arrays[
                        "direct_frame0_actuator_state_absent"
                    ][index, 0]
            ),
            "installed_frame0_table_attribution_boundary": (
                "post_forward_pre_transition"
            ),
            "installed_frame0_table_keepout": bool(
                arrays[
                    "direct_frame0_installed_table_keepout"
                ][index, 0]
            ),
            "installed_frame0_backend_resolved_table_contact": bool(
                arrays[
                    "direct_frame0_installed_backend_resolved_table_contact"
                ][index, 0]
            ),
        }
    arrays["question_f32"] = (
        np.empty((0,), dtype=np.float32) if no_question
        else question.detach().cpu().numpy().astype(np.float32)
    )
    arrays["launch_f32"] = (
        np.empty((0,), dtype=np.float32) if no_question
        else launch.detach().cpu().numpy().astype(np.float32)
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    trace_bytes = buffer.getvalue()
    _write_fresh_bytes(root / "teacher_replay.npz", trace_bytes)
    catalog = env._full_a_catalog
    plant = identity["plant_model"]
    summary = {
        "schema_version": helper.TEACHER_REPLAY_SCHEMA_VERSION,
        "kind": (
            "action_ball_mujoco_full_mdp_direct_frame0_teacher_replay_v1"
            if direct_frame0
            else "action_ball_mujoco_full_mdp_frozen_teacher_replay_v2"
        ),
        "diagnostic_unauthorized": True,
        "training_authorized": False,
        "checkpoint_authority": False,
        "ppo_update_calls": 0,
        "teacher_handoff_mode": handoff_mode,
        "num_envs": int(env.num_envs),
        "policy_steps": int(steps),
        "executed_policy_steps": int(len(rows["common_step"])),
        "shot_boundary_reason": shot_boundary_reason or "step_budget_exhausted",
        "pre_due_done_count": int(pre_due_done_count),
        "accepted_question_present": not no_question,
        "question_reset_generation": question_reset_generation,
        "run_identity": identity,
        "manifest_file_sha256": catalog.manifest_file_sha256,
        "manifest_canonical_sha256": catalog.manifest_canonical_sha256,
        "motion_sha256": catalog.fresh_action.motion_sha256,
        "ready_sha256": hashlib.sha256(ready_pose_payload).hexdigest(),
        "plant_mjb_sha256": plant["source_plant"]["compiled_mjb_sha256"],
        "runtime_mjb_sha256": plant["runtime_attach"]["final_augmented_mjb"]["sha256"],
        "runtime_stack_sha256": _canonical_payload_sha256(identity["runtime_stack"]),
        "question_f32_sha256": (
            None if no_question else helper.tensor_f32_sha256(question)
        ),
        "launch_f32_sha256": (
            None if no_question else helper.tensor_f32_sha256(launch)
        ),
        "action_delay": {"steps": 0, "source": "live MuJoCo action ABI"},
        "trace_npz": "teacher_replay.npz",
        "trace_npz_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "generic_contact_patch": contact_patch,
        "first_table_terminal_source": (
            table_attribution["first_table_terminal_source"]
            if rows["common_step"] else None
        ),
        "first_resolved_substep": (
            table_attribution["first_resolved_substep"]
            if rows["common_step"] else None
        ),
        "exact_table_narrowphase": (
            env.diagnostic_exact_table_narrowphase_receipt()
        ),
        "event_counts": {
            name: int(np.count_nonzero(arrays[name]))
            for name in (
                "scheduled_due", "reveal", "launch", "r03",
                "generic_contact", "selected_contact", "opposite_contact",
                "edge_contact", "between_contact", "invalid_contact",
                "crossing", "legal_landing", "done",
                "qdes_guard_intervention",
            )
        },
    }
    if direct_frame0:
        summary.update({
            "direct_frame0_intervention_count": int(
                np.count_nonzero(arrays["direct_frame0_intervention"])
            ),
            "direct_frame0_intervention": direct_frame0_receipt,
        })
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    _write_fresh_bytes(root / "summary.json", payload + b"\n")
    return summary


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
    mujoco_warp_runtime_site: str | None = None,
    save_interval: int = FULL_A_SAVE_INTERVAL,
    diagnostic_rate_probe: bool = False,
    diagnostic_controller_trace_checkpoint: str | None = None,
    diagnostic_controller_trace_output: str | None = None,
    diagnostic_controller_trace_action_tape: str | None = None,
    diagnostic_controller_trace_steps: int = 240,
    diagnostic_teacher_replay: bool = False,
    diagnostic_teacher_replay_output: str | None = None,
    diagnostic_teacher_replay_steps: int = 180,
    diagnostic_teacher_replay_handoff_mode: str = "split_ready_bridge",
    diagnostic_teacher_replay_physical_root_xy_m=None,
    _test_allow_small_full_a: bool = False,
) -> int:
    num_steps_per_env = NUM_STEPS_PER_ENV
    controller_trace = any(value is not None for value in (
        diagnostic_controller_trace_checkpoint,
        diagnostic_controller_trace_output,
        diagnostic_controller_trace_action_tape,
    ))
    teacher_replay = diagnostic_teacher_replay or (
        diagnostic_teacher_replay_output is not None
    )
    if (
        type(num_envs) is not int
        or num_envs <= 0
        or type(num_updates) is not int
        or num_updates <= 0
        or type(full_a_mode) is not bool
        or type(diagnostic_rate_probe) is not bool
        or type(diagnostic_controller_trace_steps) is not int
        or type(diagnostic_teacher_replay) is not bool
        or type(diagnostic_teacher_replay_steps) is not int
        or type(diagnostic_teacher_replay_handoff_mode) is not str
        or diagnostic_teacher_replay_handoff_mode
        not in DIAGNOSTIC_TEACHER_REPLAY_HANDOFF_MODES
        or type(_test_allow_small_full_a) is not bool
        or type(save_interval) is not int or save_interval != FULL_A_SAVE_INTERVAL
    ):
        raise ValueError("runner dimensions/mode differ")
    if (
        not teacher_replay
        and diagnostic_teacher_replay_handoff_mode
        != DIAGNOSTIC_TEACHER_REPLAY_HANDOFF_SPLIT_READY_BRIDGE
    ):
        raise ValueError(
            "direct frame-zero handoff requires diagnostic teacher replay"
        )
    if teacher_replay and (
        not diagnostic_teacher_replay
        or not full_a_mode
        or controller_trace
        or diagnostic_rate_probe
        or diagnostic_teacher_replay_output is None
        or diagnostic_teacher_replay_steps <= 0
        or num_envs != 1
        or num_updates != 1
    ):
        raise ValueError(
            "teacher replay requires --full-a, N=1, one unused update, one output, "
            "and no other diagnostic mode"
        )
    if diagnostic_rate_probe and not full_a_mode:
        raise ValueError("diagnostic rate probe requires --full-a")
    if controller_trace and (
        not full_a_mode
        or diagnostic_controller_trace_checkpoint is None
        or diagnostic_controller_trace_output is None
        or diagnostic_controller_trace_steps != 240
        or diagnostic_rate_probe
    ):
        raise ValueError(
            "controller trace requires --full-a, checkpoint/output, exactly 240 steps, "
            "and no rate probe"
        )
    production_full_a = (
        full_a_mode and not diagnostic_rate_probe and not controller_trace
        and not teacher_replay
    )
    if production_full_a and (
        evidence_jsonl is None or snapshot_dir is None or completion_json is None
    ):
        raise ValueError("MuJoCo Full-A requires evidence, snapshot and completion paths")
    if diagnostic_rate_probe and (
        evidence_jsonl is None or snapshot_dir is not None or completion_json is not None
    ):
        raise ValueError(
            "diagnostic rate probe requires evidence and forbids snapshot/completion paths"
        )
    if controller_trace and any(value is not None for value in (
        evidence_jsonl, snapshot_dir, completion_json,
    )):
        raise ValueError("controller trace forbids training evidence/snapshot/completion paths")
    if teacher_replay and any(value is not None for value in (
        evidence_jsonl, snapshot_dir, completion_json,
    )):
        raise ValueError("teacher replay forbids training evidence/snapshot/completion paths")
    if not full_a_mode and any(value is not None for value in (
        evidence_jsonl, snapshot_dir, completion_json, source_commit, run_namespace,
        mujoco_warp_runtime_site,
    )):
        raise ValueError("MuJoCo Full-A artifact arguments require --full-a")
    expected_updates = (
        RATE_PROBE_NUM_UPDATES if diagnostic_rate_probe else FULL_A_NUM_UPDATES
    )
    if full_a_mode and not _test_allow_small_full_a and not teacher_replay and (
        num_envs != FULL_A_NUM_ENVS
        or (not controller_trace and not teacher_replay and num_updates != expected_updates)
    ):
        label = (
            "teacher replay" if teacher_replay else
            "controller trace" if controller_trace else
            "diagnostic rate probe" if diagnostic_rate_probe else
            "production MuJoCo Full-A"
        )
        raise ValueError(
            f"{label} shape must be "
            f"{FULL_A_NUM_ENVS}x{NUM_STEPS_PER_ENV}x{expected_updates}"
        )
    if diagnostic_rate_probe and any(
        os.environ.get(name) not in (None, "0")
        for name in RATE_PROBE_PROFILE_ENVS
    ):
        raise ValueError("diagnostic rate probe requires profiler environment off")
    if full_a_mode and (
        type(mujoco_warp_runtime_site) is not str
        or not mujoco_warp_runtime_site
    ):
        raise ValueError("MuJoCo Full-A runtime site is not bound")
    if full_a_mode:
        _require_run_identity_fields(source_commit, run_namespace)
        _require_geometry_source_environment()
        if not _test_allow_small_full_a:
            source_commit = _verified_source_checkout_commit(source_commit)
    runtime_preimport = (
        _epa48_runtime_module().verify_runtime_stack_preimport()
        if full_a_mode else None
    )
    runtime_identity = (
        _bind_full_a_runtime(mujoco_warp_runtime_site, runtime_preimport)
        if full_a_mode else None
    )
    artifact_root = None
    if controller_trace:
        artifact_root = _fresh_controller_trace_root(
            diagnostic_controller_trace_output
        )
    elif teacher_replay:
        artifact_root = _fresh_controller_trace_root(
            diagnostic_teacher_replay_output
        )
    elif full_a_mode:
        artifact_root = _full_a_artifact_root(
            evidence_jsonl, snapshot_dir, completion_json
        )

    # Full-A must bind the exact EPA48/RSL3 site before importing torch or any
    # environment path that can import MJLab/MuJoCo-Warp.
    import torch

    version, runner_type, distribution = _rsl3_runner()
    wait = _wait_module()
    ready_pose_payload, ready_pose_source = _ready_pose_input()
    plant_path = _plant_xml_input() if full_a_mode else None
    plant_before = _scan_plant_source(plant_path) if full_a_mode else None
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
        xml_path=plant_path,
        seed=0,
        ready_pose_payload=ready_pose_payload,
        ready_pose_source=ready_pose_source,
        full_a_mode=full_a_mode,
    )
    if full_a_mode:
        _verify_full_a_runtime_postimport(runtime_identity)
    plant_after = _scan_plant_source(plant_path) if full_a_mode else None
    if full_a_mode and not _test_allow_small_full_a:
        source_commit = _verified_source_checkout_commit(source_commit)
    augmented_mjb = (
        _plant_contract_module().persist_augmented_runtime_mjb(
            _mujoco_module(), env.mj_model, artifact_root,
        )
        if full_a_mode else None
    )
    identity = (
        _run_identity(
            source_commit,
            run_namespace,
            runtime_identity,
            _plant_model_identity(
                env, plant_path, plant_before, plant_after, augmented_mjb,
            ),
        )
        if full_a_mode else None
    )
    action_contract = env.action_contract_identity if full_a_mode else None
    observation = wait.observation_contract
    policy_width = (
        observation.ACTOR_WIDTH_V3 if full_a_mode else observation.ACTOR_WIDTH_V1
    )
    critic_width = (
        observation.CRITIC_WIDTH_V3 if full_a_mode else observation.CRITIC_WIDTH_V1
    )
    initial = env.get_observations()
    if (
        env.num_actions != 31
        or tuple(initial["policy"].shape) != (num_envs, policy_width)
        or tuple(initial["critic"].shape) != (num_envs, critic_width)
        or not bool(torch.isfinite(initial["policy"]).all())
        or not bool(torch.isfinite(initial["critic"]).all())
    ):
        raise RuntimeError("MuJoCo WAIT initial RSL3 surface differs")

    if teacher_replay:
        summary = _run_teacher_replay(
            env=env,
            root=artifact_root,
            identity=identity,
            ready_pose_payload=ready_pose_payload,
            steps=diagnostic_teacher_replay_steps,
            handoff_mode=diagnostic_teacher_replay_handoff_mode,
            physical_root_xy_m=diagnostic_teacher_replay_physical_root_xy_m,
            torch_module=torch,
        )
        _best_effort_stdout_marker(
            "ACTION_BALL_MUJOCO_TEACHER_REPLAY_JSON="
            + json.dumps(summary, sort_keys=True)
        )
        return 0

    ledger = None
    if full_a_mode and not controller_trace:
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
            run_identity=identity,
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
        _apply_full_a_policy_bootstrap(runner, torch, env)
    if controller_trace:
        summary = _run_controller_trace(
            runner=runner,
            env=env,
            root=artifact_root,
            checkpoint=diagnostic_controller_trace_checkpoint,
            steps=diagnostic_controller_trace_steps,
            action_tape=diagnostic_controller_trace_action_tape,
            termination_bit_contract=wait.FULLMDP_TERMINATION_BITS,
            torch_module=torch,
        )
        _best_effort_stdout_marker(
            "ACTION_BALL_MUJOCO_CONTROLLER_TRACE_JSON="
            + json.dumps(summary, sort_keys=True)
        )
        return 0
    runner.disable_logs = True
    # RSL-RL 3.1.2 initializes this field only when a logging writer exists,
    # but its stock save() reads it unconditionally before checking disable_logs.
    runner.logger_type = "tensorboard"
    updates = 0
    snapshots = _snapshot_root(snapshot_dir) if production_full_a else None
    evidence_fd = _open_evidence_jsonl(evidence_jsonl) if full_a_mode else None
    snapshot_receipts = []
    snapshot_indices = (
        _snapshot_indices(num_updates, save_interval) if production_full_a else ()
    )
    original_update = runner.alg.update
    run_started_at = time.perf_counter()
    iteration_started_at = run_started_at
    rate_update_seconds = []

    def counted_update():
        nonlocal iteration_started_at, updates
        collection_finished_at = time.perf_counter()
        if ledger is not None:
            storage = runner.alg.storage
            storage_tensors, storage_dones = _rollout_storage_views(storage)
            prepared = ledger.prepare(
                updates, environment_steps=env.common_step_counter,
                storage_step=storage.step,
                storage_tensors=storage_tensors,
                storage_dones=storage_dones,
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
                _best_effort_stdout_marker(
                    f"ACTION_BALL_MUJOCO_FULL_A_PROGRESS={index}:{receipt['name']}"
                )
        update_finished_at = time.perf_counter()
        if diagnostic_rate_probe:
            rate_update_seconds.append(update_finished_at - iteration_started_at)
        iteration_started_at = update_finished_at
        return result

    runner.alg.update = counted_update
    try:
        runner.learn(num_updates, init_at_random_ep_len=False)
        final = env.get_observations()
        storage = runner.alg.storage
        storage_finite = (
            _full_a_rollout_storage_finite(
                storage, ledger_module=ledger_module, torch_module=torch,
                num_steps=num_steps_per_env, num_envs=num_envs,
                device=env.device,
            )
            if full_a_mode
            else all(
                isinstance(value, torch.Tensor)
                and tuple(value.shape) == (num_steps_per_env, num_envs, 1)
                and bool(torch.isfinite(value).all())
                for value in (
                    storage.rewards, storage.returns, storage.advantages
                )
            )
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
            or tuple(final["policy"].shape) != (num_envs, policy_width)
            or tuple(final["critic"].shape) != (num_envs, critic_width)
            or not final_observation_finite
        ):
            raise RuntimeError("MuJoCo WAIT RSL3 update evidence differs")
        if production_full_a:
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
        "task_lifecycle": (
            "full_a_diagnostic_rate_probe"
            if diagnostic_rate_probe
            else "full_a_engineering_longrun_complete"
            if full_a_mode
            else "idle_wait_only"
        ),
    }
    if diagnostic_rate_probe:
        rate_execution_recipe = _rate_execution_recipe(
            num_envs=num_envs,
            num_steps_per_env=num_steps_per_env,
            max_iterations=num_updates,
            save_interval=save_interval,
        )
        payload.update({
            "kind": "action_ball_mujoco_full_mdp_h48_rate_probe_v1",
            "schema_version": 1,
            "formal_evidence": False,
            "safety_gate": False,
            "source_commit": source_commit,
            "namespace": run_namespace,
            "learning_recipe_sha256": (
                FULL_MDP_PPO_RECIPE.learning_recipe_sha256()
            ),
            "candidate_production_execution_recipe": (
                FULL_MDP_PPO_RECIPE.execution_recipe()
            ),
            "candidate_production_execution_recipe_sha256": (
                FULL_MDP_PPO_RECIPE_SHA256
            ),
            "rate_execution_recipe": rate_execution_recipe,
            "rate_execution_recipe_sha256": (
                _canonical_payload_sha256(rate_execution_recipe)
            ),
        })
        measured = rate_update_seconds[
            RATE_PROBE_WARMUP_UPDATES:
            RATE_PROBE_WARMUP_UPDATES + RATE_PROBE_MEASURED_UPDATES
        ]
        if len(rate_update_seconds) != RATE_PROBE_NUM_UPDATES or len(measured) != (
            RATE_PROBE_MEASURED_UPDATES
        ):
            raise RuntimeError("diagnostic rate probe timing window differs")
        measured_wall = sum(measured)
        measured_transitions = (
            RATE_PROBE_MEASURED_UPDATES * num_steps_per_env * num_envs
        )
        rate = {
            "warmup_updates": RATE_PROBE_WARMUP_UPDATES,
            "measured_updates": RATE_PROBE_MEASURED_UPDATES,
            "tail_updates": RATE_PROBE_TAIL_UPDATES,
            "total_wall_seconds": time.perf_counter() - run_started_at,
            "measured_wall_seconds": measured_wall,
            "measured_update_seconds": measured,
            "measured_transitions": measured_transitions,
            "measured_transitions_per_second": measured_transitions / measured_wall,
            "update_seconds_p50": statistics.median(measured),
            "update_seconds_p90": statistics.quantiles(
                measured, n=10, method="inclusive"
            )[8],
        }
        if getattr(env.device, "type", None) == "cuda":
            rate["torch_cuda_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(env.device)
            )
        payload["rate_probe"] = rate
    elif full_a_mode:
        payload["action_ball_full_mdp_ppo_recipe_sha256"] = (
            FULL_MDP_PPO_RECIPE_SHA256
        )
        payload.update({
            "engineering_run_complete": True, "full_a_update_ack_count": updates,
        })
    else:
        payload["action_ball_full_mdp_ppo_recipe_sha256"] = (
            FULL_MDP_PPO_RECIPE_SHA256
        )
    _best_effort_stdout_marker(
        "ACTION_BALL_MUJOCO_WAIT_RSL3_JSON="
        + json.dumps(payload, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-updates", type=int, default=1)
    parser.add_argument("--full-a", dest="full_a_mode", action="store_true")
    parser.add_argument("--diagnostic-rate-probe", action="store_true")
    parser.add_argument("--diagnostic-controller-trace-checkpoint")
    parser.add_argument("--diagnostic-controller-trace-output")
    parser.add_argument("--diagnostic-controller-trace-action-tape")
    parser.add_argument(
        "--diagnostic-controller-trace-steps", type=int, default=240
    )
    parser.add_argument("--diagnostic-teacher-replay", action="store_true")
    parser.add_argument("--diagnostic-teacher-replay-output")
    parser.add_argument(
        "--diagnostic-teacher-replay-steps", type=int, default=180
    )
    parser.add_argument(
        "--diagnostic-teacher-replay-handoff-mode",
        choices=DIAGNOSTIC_TEACHER_REPLAY_HANDOFF_MODES,
        default=DIAGNOSTIC_TEACHER_REPLAY_HANDOFF_SPLIT_READY_BRIDGE,
    )
    parser.add_argument(
        "--diagnostic-teacher-replay-physical-root-xy-m",
        nargs=2,
        type=float,
        metavar=("X_M", "Y_M"),
    )
    parser.add_argument("--evidence-jsonl")
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--completion-json")
    parser.add_argument("--source-commit")
    parser.add_argument("--run-namespace")
    parser.add_argument("--mujoco-warp-runtime-site")
    parser.add_argument(
        "--save-interval", type=int, default=FULL_A_SAVE_INTERVAL
    )
    raise SystemExit(main(**vars(parser.parse_args())))
