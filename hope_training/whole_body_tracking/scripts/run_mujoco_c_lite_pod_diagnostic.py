#!/usr/bin/env python3
"""Run the real native MuJoCo C-lite PPO checkpoint plumbing diagnostic.

The default mode is a read-only plan.  ``--execute`` still refuses to create
anything unless ``--confirm-diagnostic-unauthorized`` is also present.  An
execution constructs fresh physical MuJoCo cores from exact authority files,
runs finite two-step diagnostic PPO episodes, saves only at a full-reset
boundary, cold-loads into another fresh trainer/core batch, and requires the
next update plus trainer state to match exactly.  The cold trainer is launched
in a fresh Python process, not reconstructed in the source process.

This runner is deliberately not a formal trainer, promotion gate, exporter,
deployment tool, or hardware tool.  Its C-lite reward has no motion/balance
term and its checkpoint does not contain mid-episode environment state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
if str(WBT_ROOT) not in sys.path:
    sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import checkpoint as diagnostic_checkpoint  # noqa: E402
from mujoco_native import single_env  # noqa: E402
from mujoco_native import trainer as diagnostic_trainer  # noqa: E402
from mujoco_native import vec_env  # noqa: E402


RESULT_KIND = "a3_mujoco_c_lite_pod_diagnostic_result_v1"
PLAN_KIND = "a3_mujoco_c_lite_pod_diagnostic_plan_v1"
FAILURE_KIND = "a3_mujoco_c_lite_pod_diagnostic_failure_v1"
COLD_CHILD_REQUEST_KIND = "a3_mujoco_c_lite_cold_child_request_v1"
COLD_CHILD_RESULT_KIND = "a3_mujoco_c_lite_cold_child_result_v1"
EPISODE_STEPS = 2


class RunnerError(RuntimeError):
    """The controlled diagnostic runner failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _plain_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunnerError(f"{name} must be a lowercase SHA-256")
    return value


def _authority_row(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    expected = _plain_sha256(expected_sha256, f"expected {name} SHA")
    if not source.is_file():
        raise RunnerError(f"{name} is not a file: {source}")
    actual = _sha256_file(source)
    if actual != expected:
        raise RunnerError(f"{name} file SHA differs from explicit authority")
    return {"path": str(source), "sha256": actual, "bytes": source.stat().st_size}


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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(str(path), flags, 0o644)
    except FileExistsError as exc:
        raise RunnerError(f"refusing to overwrite existing output: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _finite_tree(value: Any, path: str = "receipt") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunnerError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
        return
    raise RunnerError(
        f"{path} contains unsupported receipt type {type(value).__name__}"
    )


def _assert_exact_state(left: Any, right: Any, path: str) -> None:
    """Compare nested trainer state without tolerances or serialization guesses."""

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - real execution already needs NumPy
        np = None
    try:
        import torch
    except ImportError:  # pragma: no cover - real execution already needs Torch
        torch = None
    if torch is not None and isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise RunnerError(f"cold-load parity mismatch at {path}")
        return
    if np is not None and isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise RunnerError(f"cold-load parity mismatch at {path}")
        return
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or set(left) != set(right):
            raise RunnerError(f"cold-load parity mapping mismatch at {path}")
        for key in sorted(left, key=str):
            _assert_exact_state(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RunnerError(f"cold-load parity sequence mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_exact_state(left_item, right_item, f"{path}[{index}]")
        return
    if type(left) is not type(right) or left != right:
        raise RunnerError(f"cold-load parity mismatch at {path}")


def _state_digest(value: Any) -> str:
    """Hash nested model/optimizer/normalizer/RNG state across processes."""

    digest = hashlib.sha256()

    def add(item: Any) -> None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - execution already needs NumPy
            np = None
        try:
            import torch
        except ImportError:  # pragma: no cover - execution already needs Torch
            torch = None
        if torch is not None and isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"torch\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
            return
        if np is not None and isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
            return
        if isinstance(item, tuple):
            digest.update(b"tuple\0")
            for child in item:
                add(child)
            return
        if isinstance(item, list):
            digest.update(b"list\0")
            for child in item:
                add(child)
            return
        if item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(
                json.dumps(item, sort_keys=True, allow_nan=False).encode("utf-8")
            )
            return
        raise RunnerError(f"unsupported trainer state type {type(item).__name__}")

    add(value)
    return digest.hexdigest()


def _tensor_descriptor(value: Any, name: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - execution already needs Torch
        raise RunnerError("capturing a parity transcript requires torch") from exc
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise RunnerError(f"{name} must be a finite tensor")
    array = value.detach().cpu().contiguous().numpy()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


class _ParityCaptureEnv:
    """Delegate VecEnv operations while retaining matched reason/safety evidence."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.parity_transcript: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, **kwargs: Any) -> Any:
        return self._env.reset(**kwargs)

    def step(self, actions: Any) -> Any:
        result = self._env.step(actions)
        observations, rewards, dones, extras = result
        if not isinstance(extras, Mapping):
            raise RunnerError("VecEnv parity transcript extras are not a mapping")
        critic = extras.get("observations")
        critic = critic.get("critic") if isinstance(critic, Mapping) else None
        row = {
            "actions": _tensor_descriptor(actions, "step actions"),
            "observations": _tensor_descriptor(observations, "step observations"),
            "critic_observations": _tensor_descriptor(
                critic, "step critic observations"
            ),
            "rewards": _tensor_descriptor(rewards, "step rewards"),
            "dones": _tensor_descriptor(dones, "step dones"),
            "time_outs": _tensor_descriptor(extras.get("time_outs"), "time_outs"),
            "terminal_observations": _tensor_descriptor(
                extras.get("terminal_observations"), "terminal observations"
            ),
            "terminal_observation_mask": _tensor_descriptor(
                extras.get("terminal_observation_mask"),
                "terminal observation mask",
            ),
            "episode_done_reasons": extras.get("episode_done_reasons"),
            "reward_terms": extras.get("reward_terms"),
            "diagnostic_unauthorized": extras.get("diagnostic_unauthorized"),
            "formal_authorized": extras.get("formal_authorized"),
        }
        _finite_tree(row, "parity_transcript")
        self.parity_transcript.append(row)
        return result

    def clear_parity_transcript(self) -> None:
        self.parity_transcript.clear()


def _validate_selected_rubber_authority(
    env: Any, manifest_path: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    manifest = manifest_path.expanduser().resolve()
    expected_sha = _plain_sha256(
        expected_manifest_sha256, "expected selected-rubber manifest SHA"
    )
    lineages = []
    for index, question in enumerate(env.questions):
        lineage = getattr(question, "selected_rubber_action_lineage", None)
        if not isinstance(lineage, Mapping):
            raise RunnerError(
                f"question {index} lacks a selected-rubber action-lineage authority"
            )
        if lineage.get("action_manifest_sha256") != expected_sha:
            raise RunnerError(
                f"question {index} selected-rubber manifest SHA differs from CLI authority"
            )
        logical = lineage.get("action_manifest_repo_relative_path")
        if not isinstance(logical, str):
            raise RunnerError(
                f"question {index} selected-rubber manifest path is absent"
            )
        rebound = (REPO_ROOT / logical).resolve()
        if rebound != manifest:
            raise RunnerError(
                f"question {index} selected-rubber manifest path differs from CLI authority"
            )
        lineages.append(dict(lineage))
    first = lineages[0]
    if any(row != first for row in lineages[1:]):
        raise RunnerError("vector rows do not share one selected-rubber authority")
    return first


def _resolved_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_envs < 1:
        raise RunnerError("num_envs must be positive")
    if args.pre_checkpoint_updates < 1:
        raise RunnerError("pre_checkpoint_updates must be positive")
    if args.torch_device != "cpu":
        raise RunnerError(
            "current MujocoDiagnosticPPOTrainer is CPU-only; torch_device must be cpu"
        )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RunnerError("output directory already exists (no-clobber)")
    if not output_dir.parent.is_dir():
        raise RunnerError("output directory parent does not exist")
    authorities = _validated_authorities(args)
    return {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "mode": "execute" if args.execute else "plan",
        "authorities": authorities,
        "workload": {
            "num_envs": args.num_envs,
            "episode_steps": EPISODE_STEPS,
            "rollout_steps_per_update": EPISODE_STEPS,
            "pre_checkpoint_updates": args.pre_checkpoint_updates,
            "matched_updates_after_checkpoint": 1,
            "source_updates_total": args.pre_checkpoint_updates + 1,
            "cold_loaded_updates": 1,
            "torch_device": "cpu",
            "mujoco_physics_device": "cpu",
            "seed": args.seed,
            "hidden_dims": list(args.hidden_dims),
            "learning_rate": args.learning_rate,
            "initial_action_std": args.initial_action_std,
        },
        "outputs": {
            "directory": str(output_dir),
            "checkpoint": str(output_dir / "reset_boundary.pt"),
            "result": str(output_dir / "result.json"),
        },
        "claims": {
            "scope": "C-lite plumbing_and_learnability_smoke_only",
            "reset_boundary_resume_only": True,
            "mid_episode_resume": False,
            "formal_training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }


def _validated_authorities(args: argparse.Namespace) -> dict[str, Any]:
    authorities = {
        "plant_contract": _authority_row(
            args.plant_contract, args.expected_plant_sha256, "plant contract"
        ),
        "robot_tape": _authority_row(
            args.robot_tape, args.expected_robot_tape_sha256, "robot tape"
        ),
        "question": _authority_row(
            args.question, args.expected_question_sha256, "question"
        ),
        "selected_rubber_manifest": _authority_row(
            args.selected_rubber_manifest,
            args.expected_selected_rubber_manifest_sha256,
            "selected-rubber manifest",
        ),
        "mjcf": _authority_row(args.mjcf, args.expected_mjcf_sha256, "MJCF"),
    }
    phase_values = (
        args.phase_fidelity_reference_tape,
        args.expected_phase_fidelity_reference_tape_sha256,
    )
    if (phase_values[0] is None) != (phase_values[1] is None):
        raise RunnerError(
            "phase reference tape and expected SHA must be supplied together"
        )
    if phase_values[0] is not None:
        authorities["phase_fidelity_reference_tape"] = _authority_row(
            phase_values[0], phase_values[1], "phase-fidelity reference tape"
        )
    return authorities


def _build_env(args: argparse.Namespace) -> Any:
    return vec_env.MujocoN1DiagnosticVecEnv.from_authorities(
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
        diagnostic_episode_length=EPISODE_STEPS,
    )


def _build_trainer(env: Any, args: argparse.Namespace) -> Any:
    identity = diagnostic_trainer.TrainerIdentity(**env.diagnostic_training_identity())
    config = diagnostic_trainer.DiagnosticPPOConfig(
        observation_dim=env.num_observations,
        action_dim=env.num_actions,
        rollout_steps=EPISODE_STEPS,
        hidden_dims=tuple(args.hidden_dims),
        seed=args.seed,
        learning_rate=args.learning_rate,
        initial_action_std=args.initial_action_std,
    )
    return diagnostic_trainer.MujocoDiagnosticPPOTrainer(
        env=env, identity=identity, config=config
    )


def _child_request(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    path_fields = (
        "plant_contract",
        "robot_tape",
        "question",
        "selected_rubber_manifest",
        "mjcf",
        "phase_fidelity_reference_tape",
    )
    value_fields = (
        "expected_plant_sha256",
        "expected_robot_tape_sha256",
        "expected_question_sha256",
        "expected_selected_rubber_manifest_sha256",
        "expected_mjcf_sha256",
        "expected_phase_fidelity_reference_tape_sha256",
        "num_envs",
        "seed",
        "learning_rate",
        "initial_action_std",
        "torch_device",
    )
    child_args = {
        field: (
            None
            if getattr(args, field) is None
            else str(getattr(args, field).expanduser().resolve())
        )
        for field in path_fields
    }
    child_args.update({field: getattr(args, field) for field in value_fields})
    child_args["hidden_dims"] = list(args.hidden_dims)
    return {
        "schema_version": 1,
        "kind": COLD_CHILD_REQUEST_KIND,
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "parent_pid": os.getpid(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "args": child_args,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }


def _namespace_from_child_request(request: Mapping[str, Any]) -> argparse.Namespace:
    expected = {
        "schema_version",
        "kind",
        "runner_source_sha256",
        "parent_pid",
        "checkpoint_path",
        "checkpoint_sha256",
        "args",
        "diagnostic_unauthorized",
        "formal_authorized",
    }
    if set(request) != expected or request.get("schema_version") != 1:
        raise RunnerError("cold-child request schema differs")
    if request.get("kind") != COLD_CHILD_REQUEST_KIND:
        raise RunnerError("cold-child request kind differs")
    if request.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve()):
        raise RunnerError("cold child runner source differs from parent request")
    if request.get("parent_pid") == os.getpid():
        raise RunnerError("cold-load must run in a fresh process")
    if (
        request.get("diagnostic_unauthorized") is not True
        or request.get("formal_authorized") is not False
    ):
        raise RunnerError("cold-child authorization boundary differs")
    values = request.get("args")
    if not isinstance(values, Mapping):
        raise RunnerError("cold-child args are absent")
    values = dict(values)
    for field in (
        "plant_contract",
        "robot_tape",
        "question",
        "selected_rubber_manifest",
        "mjcf",
        "phase_fidelity_reference_tape",
    ):
        values[field] = None if values.get(field) is None else Path(values[field])
    return argparse.Namespace(**values)


def _cold_child(request_path: Path, result_path: Path) -> int:
    try:
        request = json.loads(request_path.expanduser().resolve().read_text("utf-8"))
        if not isinstance(request, Mapping):
            raise RunnerError("cold-child request root must be a mapping")
        args = _namespace_from_child_request(request)
        _validated_authorities(args)
        checkpoint_path = Path(request["checkpoint_path"]).expanduser().resolve()
        if _sha256_file(checkpoint_path) != request["checkpoint_sha256"]:
            raise RunnerError("cold-child checkpoint SHA differs from parent request")
        cold_env = _build_env(args)
        _validate_selected_rubber_authority(
            cold_env,
            args.selected_rubber_manifest,
            args.expected_selected_rubber_manifest_sha256,
        )
        cold_capture = _ParityCaptureEnv(cold_env)
        cold = _build_trainer(cold_capture, args)
        load_receipt = diagnostic_checkpoint.ResetBoundaryCheckpoint().load(
            checkpoint_path, cold
        )
        cold_capture.clear_parity_transcript()
        update_receipt = cold.run_update()
        _finite_tree(update_receipt)
        state = cold.checkpoint_state()
        result = {
            "schema_version": 1,
            "kind": COLD_CHILD_RESULT_KIND,
            "status": "COLD_CHILD_COMPLETE",
            "pid": os.getpid(),
            "parent_pid": request["parent_pid"],
            "runner_source_sha256": request["runner_source_sha256"],
            "checkpoint_sha256": request["checkpoint_sha256"],
            "identity": cold.identity.as_dict(),
            "checkpoint_load_receipt": load_receipt,
            "next_update_receipt": update_receipt,
            "transition_transcript": cold_capture.parity_transcript,
            "state_sha256": _state_digest(state),
            "model_state_sha256": _state_digest(cold.model.state_dict()),
            "optimizer_state_sha256": _state_digest(cold.optimizer.state_dict()),
            "normalizer_state_sha256": _state_digest(cold.normalizer.state_dict()),
            "update_counter": cold.update_counter,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
            "mid_episode_resume": False,
        }
        _finite_tree(result)
        _write_new_json(result_path.expanduser().resolve(), result)
        return 0
    except Exception as exc:  # noqa: BLE001 - isolated child failure boundary
        print(f"[COLD-CHILD-FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _execute(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not args.confirm_diagnostic_unauthorized:
        raise RunnerError(
            "--execute requires explicit --confirm-diagnostic-unauthorized"
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(mode=0o755, parents=False, exist_ok=False)
    start = time.monotonic()

    source_env = _build_env(args)
    selected_lineage = _validate_selected_rubber_authority(
        source_env,
        args.selected_rubber_manifest,
        args.expected_selected_rubber_manifest_sha256,
    )
    readiness = source_env.diagnostic_training_receipt()
    source_capture = _ParityCaptureEnv(source_env)
    source = _build_trainer(source_capture, args)
    pre_checkpoint_receipts = []
    for _index in range(args.pre_checkpoint_updates):
        receipt = source.run_update()
        _finite_tree(receipt)
        if receipt.get("at_reset_boundary") is not True:
            raise RunnerError("pre-checkpoint update did not end at a reset boundary")
        pre_checkpoint_receipts.append(receipt)

    checkpoint_path = output_dir / "reset_boundary.pt"
    checkpoint_api = diagnostic_checkpoint.ResetBoundaryCheckpoint()
    save_receipt = checkpoint_api.save(checkpoint_path, source)
    source_capture.clear_parity_transcript()
    reference_next = source.run_update()
    _finite_tree(reference_next)
    reference_transcript = list(source_capture.parity_transcript)
    source_state = source.checkpoint_state()
    source_state_digests = {
        "state_sha256": _state_digest(source_state),
        "model_state_sha256": _state_digest(source.model.state_dict()),
        "optimizer_state_sha256": _state_digest(source.optimizer.state_dict()),
        "normalizer_state_sha256": _state_digest(source.normalizer.state_dict()),
    }

    child_request_path = output_dir / "cold_child_request.json"
    child_result_path = output_dir / "cold_child_result.json"
    request = _child_request(args, checkpoint_path)
    _write_new_json(child_request_path, request)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_cold-child",
            "--request",
            str(child_request_path),
            "--result",
            str(child_result_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RunnerError(
            "fresh cold child failed: " + completed.stderr.strip()[-2000:]
        )
    try:
        cold_result = json.loads(child_result_path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError("fresh cold child result is unreadable") from exc
    if cold_result.get("pid") == os.getpid():
        raise RunnerError("cold-load did not execute in a fresh process")
    if cold_result.get("identity") != source.identity.as_dict():
        raise RunnerError("fresh cold trainer identity differs from source trainer")
    cold_next = cold_result.get("next_update_receipt")
    if cold_next != reference_next:
        raise RunnerError("cold-load next-update receipt differs from matched source")
    if cold_result.get("transition_transcript") != reference_transcript:
        raise RunnerError(
            "cold-load reason/safety transition transcript differs from matched source"
        )
    for name, expected in source_state_digests.items():
        if cold_result.get(name) != expected:
            raise RunnerError(f"cold-load {name} differs from matched source")
    if source.update_counter != cold_result.get("update_counter"):
        raise RunnerError("cold-load update counter differs from matched source")

    result = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "status": "PASS_EXACT_RESET_BOUNDARY_COLD_LOAD_NEXT_UPDATE_PARITY",
        "plan": dict(plan),
        "readiness_receipt": readiness,
        "selected_rubber_action_lineage": selected_lineage,
        "pre_checkpoint_update_receipts": pre_checkpoint_receipts,
        "checkpoint_save_receipt": save_receipt,
        "checkpoint_load_receipt": cold_result["checkpoint_load_receipt"],
        "fresh_cold_process": {
            "parent_pid": os.getpid(),
            "child_pid": cold_result["pid"],
            "runner_source_sha256": request["runner_source_sha256"],
            "request_path": str(child_request_path),
            "request_sha256": _sha256_file(child_request_path),
            "result_path": str(child_result_path),
            "result_sha256": _sha256_file(child_result_path),
            "natural_exit_code": completed.returncode,
        },
        "matched_source_next_update_receipt": reference_next,
        "cold_loaded_next_update_receipt": cold_next,
        "matched_transition_transcript": reference_transcript,
        "parity": {
            "finite_receipts": True,
            "next_update_receipt_exact": True,
            "reason_safety_transition_transcript_exact": True,
            "model_state_exact": True,
            "optimizer_state_exact": True,
            "normalizer_state_exact": True,
            "trainer_rng_and_full_boundary_state_exact": True,
            "update_counter_exact": True,
        },
        "elapsed_wall_s": time.monotonic() - start,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
        "mid_episode_resume": False,
        "authorization": {
            "formal_training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
    }
    _finite_tree(result)
    _write_new_json(output_dir / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant-contract", required=True, type=Path)
    parser.add_argument("--expected-plant-sha256", required=True)
    parser.add_argument("--robot-tape", required=True, type=Path)
    parser.add_argument("--expected-robot-tape-sha256", required=True)
    parser.add_argument("--question", required=True, type=Path)
    parser.add_argument("--expected-question-sha256", required=True)
    parser.add_argument("--selected-rubber-manifest", required=True, type=Path)
    parser.add_argument("--expected-selected-rubber-manifest-sha256", required=True)
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--expected-mjcf-sha256", required=True)
    parser.add_argument("--phase-fidelity-reference-tape", type=Path)
    parser.add_argument("--expected-phase-fidelity-reference-tape-sha256")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--pre-checkpoint-updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(8,))
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--initial-action-std", type=float, default=0.02)
    parser.add_argument(
        "--torch-device",
        default="cpu",
        help="Current controlled trainer supports only cpu; non-cpu fails closed.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create a new output directory and run; omitted means read-only plan.",
    )
    parser.add_argument(
        "--confirm-diagnostic-unauthorized",
        action="store_true",
        help="Required with --execute; acknowledges no formal/deploy authorization.",
    )
    return parser


def _cold_child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "_cold-child":
        child_args = _cold_child_parser().parse_args(raw_argv[1:])
        return _cold_child(child_args.request, child_args.result)
    args = _parser().parse_args(raw_argv)
    output_dir: Path | None = None
    try:
        plan = _resolved_plan(args)
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
            return 0
        output_dir = args.output_dir.expanduser().resolve()
        result = _execute(args, plan)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI failure boundary is fail-closed
        failure = {
            "schema_version": 1,
            "kind": FAILURE_KIND,
            "status": "FAIL_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }
        if output_dir is not None and output_dir.is_dir():
            try:
                _write_new_json(output_dir / "failure.json", failure)
            except Exception:
                pass
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
