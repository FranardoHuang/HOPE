"""Reset-boundary-only checkpoint for the controlled MuJoCo PPO shell.

The v3 checkpoint additionally saves an optional exact reset-boundary
environment continuation state.  C211 uses it for the counter-based WAIT
highwater/current assignment; this is still not mid-episode resume.
"""

from __future__ import annotations

import copy
import hashlib
import io
import os
from pathlib import Path
from typing import Any, Mapping

from .trainer import (
    DiagnosticPPOContractError,
    DiagnosticPPOError,
    MujocoDiagnosticPPOTrainer,
    ResetBoundaryRequired,
    _require_torch,
    promotion_blocked_from_evidence,
)


CHECKPOINT_KIND = "a3_mujoco_controlled_diagnostic_reset_boundary_checkpoint_v3"
CHECKPOINT_SCHEMA_VERSION = 3


class CheckpointRefused(DiagnosticPPOError):
    """A checkpoint could not be safely saved or loaded."""


def _checkpoint_promotion_blocked(trainer: MujocoDiagnosticPPOTrainer) -> bool:
    """Carry the update's promotion verdict onto the artifact it produced.

    人话:被晋级的是 checkpoint,不是 update。所以结论位必须跟着 checkpoint 走,
    否则读 checkpoint 的人得先回去翻 update 收据才知道这份权重能不能上机。
    没有 update 收据 / 收据里没有结论位,一律按"卡住"记——缺字段与 True 同义。
    """

    receipt = getattr(trainer, "_last_update_receipt", None)
    if not isinstance(receipt, Mapping):
        return True
    return promotion_blocked_from_evidence(
        receipt.get("promotion_blocking_evidence")
    )


def _torch_load_cpu(torch: Any, source: Any) -> Any:
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source, map_location="cpu")


def _all_finite_tensors(value: Any) -> bool:
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_all_finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite_tensors(item) for item in value)
    return True


def _validate_payload(
    payload: Any, trainer: MujocoDiagnosticPPOTrainer
) -> Mapping[str, Any]:
    torch = _require_torch()
    if not isinstance(payload, Mapping):
        raise CheckpointRefused("checkpoint root must be a mapping")
    required = {
        "schema_version",
        "kind",
        "identity",
        "config_sha256",
        "normalizer_identities",
        "model_state_dict",
        "optimizer_state_dict",
        "actor_normalizer_state_dict",
        "critic_normalizer_state_dict",
        "rng_state",
        "update_counter",
        "last_update_receipt",
        "environment_state",
        "boundary",
    }
    if set(payload) != required:
        raise CheckpointRefused("checkpoint field set differs from schema")
    if (
        payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or payload["kind"] != CHECKPOINT_KIND
    ):
        raise CheckpointRefused("checkpoint kind/schema differs")
    if payload["identity"] != trainer.identity.as_dict():
        raise CheckpointRefused(
            "checkpoint contract/observation/action/reward SHA differs"
        )
    if payload["config_sha256"] != trainer.config.content_sha256:
        raise CheckpointRefused("checkpoint PPO config SHA differs")
    if payload["normalizer_identities"] != {
        "actor": trainer.config.actor_normalizer_identity,
        "critic": trainer.config.critic_normalizer_identity,
    }:
        raise CheckpointRefused("checkpoint actor/critic normalizer identity differs")
    if payload["boundary"] != {
        "kind": "explicit_full_reset_boundary",
        "mid_episode_resume": False,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }:
        raise CheckpointRefused("checkpoint boundary declaration differs")
    if type(payload["update_counter"]) is not int or payload["update_counter"] < 0:
        raise CheckpointRefused("checkpoint update counter is invalid")
    model_state = payload["model_state_dict"]
    if not isinstance(model_state, Mapping):
        raise CheckpointRefused("checkpoint model state is absent")
    current_model = trainer.model.state_dict()
    if set(model_state) != set(current_model):
        raise CheckpointRefused("checkpoint model state keys differ")
    for key, current in current_model.items():
        value = model_state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(
            current.shape
        ):
            raise CheckpointRefused(f"checkpoint model tensor {key!r} shape differs")
    if not _all_finite_tensors(model_state) or not _all_finite_tensors(
        payload["optimizer_state_dict"]
    ):
        raise CheckpointRefused("checkpoint model/optimizer state is non-finite")
    try:
        trainer.actor_normalizer.validate_state_dict(
            payload["actor_normalizer_state_dict"]
        )
        trainer.critic_normalizer.validate_state_dict(
            payload["critic_normalizer_state_dict"]
        )
    except DiagnosticPPOContractError as exc:
        raise CheckpointRefused(str(exc)) from exc
    rng = payload["rng_state"]
    if not isinstance(rng, Mapping) or set(rng) != {
        "python",
        "numpy",
        "torch_cpu",
    }:
        raise CheckpointRefused("checkpoint RNG state schema differs")
    if (
        not isinstance(rng["torch_cpu"], torch.Tensor)
        or rng["torch_cpu"].dtype != torch.uint8
    ):
        raise CheckpointRefused("checkpoint torch CPU RNG state differs")
    live_env_checkpoint = callable(getattr(trainer.env, "checkpoint_state", None))
    if live_env_checkpoint != (payload["environment_state"] is not None):
        raise CheckpointRefused(
            "checkpoint environment continuation availability differs"
        )
    if payload["environment_state"] is not None and not isinstance(
        payload["environment_state"], Mapping
    ):
        raise CheckpointRefused("checkpoint environment state is malformed")
    return payload


class ResetBoundaryCheckpoint:
    """Save/load a diagnostic trainer only between complete episodes."""

    def save(
        self, path: Path | str, trainer: MujocoDiagnosticPPOTrainer
    ) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        if target.exists():
            raise CheckpointRefused("checkpoint target already exists (no-clobber)")
        if not target.parent.is_dir():
            raise CheckpointRefused("checkpoint parent directory does not exist")
        # Every authorization/identity/boundary check runs before serialization
        # and before opening the destination.
        state = trainer.checkpoint_state()
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": CHECKPOINT_KIND,
            "identity": trainer.identity.as_dict(),
            "config_sha256": trainer.config.content_sha256,
            "normalizer_identities": {
                "actor": trainer.config.actor_normalizer_identity,
                "critic": trainer.config.critic_normalizer_identity,
            },
            **state,
            "boundary": {
                "kind": "explicit_full_reset_boundary",
                "mid_episode_resume": False,
                "diagnostic_unauthorized": True,
                "formal_authorized": False,
            },
        }
        _validate_payload(payload, trainer)
        torch = _require_torch()
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        encoded = buffer.getvalue()
        try:
            with target.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:  # noqa: BLE001 - filesystem boundary
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass
            raise CheckpointRefused("checkpoint file write failed") from exc
        return {
            "schema_version": 3,
            "kind": "a3_mujoco_controlled_diagnostic_checkpoint_save_receipt_v3",
            "path": str(target),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "update_counter": trainer.update_counter,
            **trainer.identity.as_dict(),
            "promotion_blocked": _checkpoint_promotion_blocked(trainer),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
            "mid_episode_resume": False,
        }

    def load(
        self, path: Path | str, trainer: MujocoDiagnosticPPOTrainer
    ) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        # Validate the live receipt and boundary before reading or mutating any
        # trainer state.  In particular, no VecEnv reset/step occurs here.
        trainer._validated_readiness()
        trainer.assert_reset_boundary()
        try:
            encoded = source.read_bytes()
        except OSError as exc:
            raise CheckpointRefused("checkpoint file could not be read") from exc
        torch = _require_torch()
        try:
            payload = _torch_load_cpu(torch, io.BytesIO(encoded))
        except Exception as exc:  # noqa: BLE001 - untrusted checkpoint boundary
            raise CheckpointRefused("checkpoint payload could not be decoded") from exc
        payload = _validate_payload(payload, trainer)

        old_model = copy.deepcopy(trainer.model.state_dict())
        old_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
        old_actor_normalizer = trainer.actor_normalizer.state_dict()
        old_critic_normalizer = trainer.critic_normalizer.state_dict()
        old_counter = trainer.update_counter
        old_receipt = copy.deepcopy(trainer._last_update_receipt)
        old_python_rng = __import__("random").getstate()
        import numpy as np

        old_numpy_rng = np.random.get_state()
        old_torch_rng = torch.get_rng_state().clone()
        old_environment_state = None
        environment_checkpoint = getattr(trainer.env, "checkpoint_state", None)
        environment_loader = getattr(trainer.env, "load_checkpoint_state", None)
        if payload["environment_state"] is not None:
            if not callable(environment_checkpoint) or not callable(environment_loader):
                raise CheckpointRefused(
                    "live environment cannot restore checkpoint continuation state"
                )
            old_environment_state = copy.deepcopy(environment_checkpoint())
        try:
            trainer.model.load_state_dict(payload["model_state_dict"], strict=True)
            trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
            trainer.actor_normalizer.load_state_dict(
                payload["actor_normalizer_state_dict"]
            )
            trainer.critic_normalizer.load_state_dict(
                payload["critic_normalizer_state_dict"]
            )
            trainer.update_counter = payload["update_counter"]
            trainer._last_update_receipt = copy.deepcopy(payload["last_update_receipt"])
            __import__("random").setstate(payload["rng_state"]["python"])
            np.random.set_state(payload["rng_state"]["numpy"])
            torch.set_rng_state(payload["rng_state"]["torch_cpu"])
            if payload["environment_state"] is not None:
                environment_loader(copy.deepcopy(payload["environment_state"]))
            trainer._actor_observations = None
            trainer._critic_observations = None
        except Exception as exc:  # noqa: BLE001 - rollback keeps load transactional
            trainer.model.load_state_dict(old_model, strict=True)
            trainer.optimizer.load_state_dict(old_optimizer)
            trainer.actor_normalizer.load_state_dict(old_actor_normalizer)
            trainer.critic_normalizer.load_state_dict(old_critic_normalizer)
            trainer.update_counter = old_counter
            trainer._last_update_receipt = old_receipt
            __import__("random").setstate(old_python_rng)
            np.random.set_state(old_numpy_rng)
            torch.set_rng_state(old_torch_rng)
            if old_environment_state is not None:
                environment_loader(old_environment_state)
            raise CheckpointRefused("checkpoint state could not be installed") from exc
        return {
            "schema_version": 3,
            "kind": "a3_mujoco_controlled_diagnostic_checkpoint_load_receipt_v3",
            "path": str(source),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "update_counter": trainer.update_counter,
            **trainer.identity.as_dict(),
            "promotion_blocked": _checkpoint_promotion_blocked(trainer),
            "at_reset_boundary": True,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
            "mid_episode_resume": False,
        }


__all__ = [
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointRefused",
    "ResetBoundaryCheckpoint",
    "ResetBoundaryRequired",
]
