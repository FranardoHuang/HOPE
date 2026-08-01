#!/usr/bin/env python3
"""Materialize the static ActionBall reward/PPO economy receipt.

The receipt is intentionally narrower than a training launch claim.  It binds
the two registry-pinned r6 vendor runtime contracts, their common effective
reward receipt, the vendor task leaf, the repository PPO defaults, and the
exact rsl_rl 2.3.1 PPO/storage/actor-critic source files.  It then records the
adopted analytical contract used by the three fresh N1 lanes:

* no hidden reward rescale (``reward_global_scalar == 1.0``);
* reward-manager step integration at ``dt == 0.02``;
* whole-rollout, not per-minibatch, advantage normalization;
* a required final policy ABI of ``noise_std_type=log`` with realized initial
  sigma ``0.02``.

Both runtime contracts must already expose ``noise_std_type=log`` and realized
initial sigma 0.02.  The producer therefore remains intentionally unusable
while the r6 registry pins are ``None``; it cannot let the earlier r5 scalar
materialization carry the new policy ABI.

This tool never imports Isaac Lab or Torch and never starts a simulator.  The
only operator-selected semantic input is the installed rsl_rl package root;
all repository inputs are code-owned paths and SHA-256 pins.  Materialization
writes only the registry-planned fixed path and is no-clobber.  ``--verify``
rebuilds the receipt and requires byte-equivalent semantics, so a
package/source/config drift fails loudly.

Example on the exact Pod interpreter::

    python hope_training/whole_body_tracking/scripts/\
      materialize_action_ball_reward_ppo_economy_receipt.py \
      --rsl-rl-root /workspace/hope_isaac_venv/lib/python3.10/site-packages/rsl_rl \
      --materialize

The output is diagnostic/prelaunch evidence only.  It does not authorize
training, resume, promotion, export, judging, deployment, or hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

import yaml


SCHEMA_VERSION = 1
KIND = "action_ball_reward_ppo_economy_receipt_v1"
EXPECTED_RSL_RL_DISTRIBUTION = "rsl-rl-lib"
EXPECTED_RSL_RL_VERSION = "2.3.1"
REWARD_GLOBAL_SCALAR = 1.0
POLICY_STEP_DT_S = 0.02
EXPECTED_EFFECTIVE_REWARD_SHA256 = (
    "845d75b4f409725e9dfc7070b1070a6dd6385486c79a6c5a1aec60c41c42ff02"
)
RUNTIME_GATE_NUM_ENVS = 4096
RUNTIME_GATE_UPDATES = 5
REQUIRED_FINAL_NOISE_STD_TYPE = "log"
REQUIRED_INITIAL_REALIZED_SIGMA = 0.02

ACTION_IDS = ("bh_loop_c", "bh_block")
PRODUCER_SOURCE_PATH = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_action_ball_reward_ppo_economy_receipt.py"
)
ACTION_REGISTRY_PATH = (
    "hope_training/whole_body_tracking/scripts/a3_vendor_action_registry.py"
)
TASK_PROFILE_PATH = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV1.yaml"
)
PPO_CONFIG_PATH = "hope_training/whole_body_tracking/cfg/algo/ppo.yaml"
REPOSITORY_RUNTIME_SOURCE_PATHS = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py",
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_rewards.py",
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/rewards.py",
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/effective_reward_recipe.py",
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/my_on_policy_runner.py",
)
RSL_RUNTIME_SOURCE_PINS = (
    (
        "ppo",
        "algorithms/ppo.py",
        "ec0286fdbdd30360b21f2bea314f676058793f9241649d30545d656a961a22a2",
    ),
    (
        "rollout_storage",
        "storage/rollout_storage.py",
        "456450bbba03f9f5818f9ec43d2f3bd5d14d2d27f4bf2308838dd580f32ffcf0",
    ),
    (
        "actor_critic",
        "modules/actor_critic.py",
        "51ce8c6504a7434f86e5188ccb11e9fc981e46f764c00e32be3e8febe7112ae2",
    ),
)

EXPECTED_ALGORITHM = {
    "class_name": "PPO",
    "clip_param": 0.2,
    "desired_kl": 0.01,
    "entropy_coef": 0.01,
    "gamma": 0.99,
    "lam": 0.95,
    "learning_rate": 0.001,
    "max_grad_norm": 1.0,
    "normalize_advantage_per_mini_batch": False,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "rnd_cfg": None,
    "schedule": "adaptive",
    "symmetry_cfg": None,
    "use_clipped_value_loss": True,
    "value_loss_coef": 1.0,
}
EXPECTED_RUNNER = {
    "empirical_normalization": True,
    "init_at_random_ep_len": False,
    "num_steps_per_env": 24,
}
EXPECTED_EFFECTIVE_POLICY_EXCEPT_STD_TYPE = {
    "activation": "elu",
    "actor_hidden_dims": [512, 256, 128],
    "class_name": "ActorCritic",
    "critic_hidden_dims": [512, 256, 128],
    "init_noise_std": REQUIRED_INITIAL_REALIZED_SIGMA,
}
EXPECTED_EFFECTIVE_NOISE_STD_TYPE = REQUIRED_FINAL_NOISE_STD_TYPE


class ReceiptRefused(ValueError):
    """One code-owned source or runtime invariant differs from the receipt."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ReceiptRefused("receipt contains non-canonical JSON data") from exc


def _strict_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReceiptRefused(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptRefused(f"{name} is not strict UTF-8 JSON") from exc
    if type(document) is not dict:
        raise ReceiptRefused(f"{name} must be a JSON object")
    return document


def _read_regular_bytes(path: Path, *, name: str) -> bytes:
    if not path.is_absolute():
        raise ReceiptRefused(f"{name} path must be absolute")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ReceiptRefused(f"cannot stat {name}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ReceiptRefused(f"{name} must be a regular non-symlink file")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ReceiptRefused(f"cannot read {name}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ReceiptRefused(f"{name} changed while being read")
    return payload


def _read_pinned(
    repo_root: Path, relative_path: str, expected_sha256: str, *, name: str
) -> tuple[bytes, dict[str, str]]:
    payload = _read_regular_bytes(repo_root / relative_path, name=name)
    actual = _sha256(payload)
    if actual != expected_sha256:
        raise ReceiptRefused(
            f"{name} SHA-256 drift: expected {expected_sha256}, got {actual}"
        )
    return payload, {"path": relative_path, "sha256": actual}


def _read_source(
    repo_root: Path, relative_path: str, *, name: str
) -> tuple[bytes, dict[str, str]]:
    payload = _read_regular_bytes(repo_root / relative_path, name=name)
    return payload, {"path": relative_path, "sha256": _sha256(payload)}


def _git(
    repo_root: Path, args: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ReceiptRefused("cannot invoke git for source cleanliness") from exc


def _tracked_clean(repo_root: Path, relative_paths: Sequence[str]) -> None:
    paths = tuple(dict.fromkeys(relative_paths))
    for relative_path in paths:
        result = _git(
            repo_root,
            ("ls-files", "--error-unmatch", "--", relative_path),
        )
        if result.returncode != 0 or result.stdout.strip() != relative_path:
            raise ReceiptRefused(
                f"reward/PPO source is not tracked: {relative_path}"
            )
    for args, label in (
        (("diff", "--quiet", "--", *paths), "working tree"),
        (("diff", "--cached", "--quiet", "--", *paths), "index"),
    ):
        if _git(repo_root, args).returncode != 0:
            raise ReceiptRefused(
                f"reward/PPO sources differ from HEAD in {label}"
            )


def _load_registry(repo_root: Path):
    path = repo_root / ACTION_REGISTRY_PATH
    _read_regular_bytes(path, name="A3 vendor action registry")
    spec = importlib.util.spec_from_file_location(
        "_hope_reward_economy_action_registry", path
    )
    if spec is None or spec.loader is None:
        raise ReceiptRefused("cannot load A3 vendor action registry")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ReceiptRefused(f"cannot import A3 vendor action registry: {exc}") from exc
    return module


def _registry_pins(
    repo_root: Path,
) -> tuple[
    list[tuple[str, str, str]],
    dict[str, Optional[str]],
    list[dict[str, Any]],
]:
    registry = _load_registry(repo_root)
    action_configs = getattr(registry, "ACTION_CONFIGS", None)
    if not isinstance(action_configs, Mapping) or set(action_configs) != set(ACTION_IDS):
        raise ReceiptRefused("A3 vendor action registry action set drifted")
    contract_pins: list[tuple[str, str, str]] = []
    output_paths: set[str] = set()
    output_hashes: set[Optional[str]] = set()
    source_identities: list[dict[str, Any]] = []
    for action_id in ACTION_IDS:
        config = action_configs[action_id]
        runtime_pin = getattr(config, "runtime_contract", None)
        reward_pin = getattr(config, "reward_economy_receipt", None)
        try:
            runtime_path = runtime_pin.path
            runtime_sha256 = runtime_pin.sha256
            output_path = reward_pin.path
            output_sha256 = reward_pin.sha256
        except AttributeError as exc:
            raise ReceiptRefused(
                f"registry action {action_id!r} lacks r6 runtime/economy pins"
            ) from exc
        if type(runtime_path) is not str or not runtime_path:
            raise ReceiptRefused(f"registry action {action_id!r} runtime path is invalid")
        if (
            type(runtime_sha256) is not str
            or len(runtime_sha256) != 64
            or any(char not in "0123456789abcdef" for char in runtime_sha256)
        ):
            raise ReceiptRefused(
                f"registry action {action_id!r} r6 runtime contract is not materialized"
            )
        if type(output_path) is not str or not output_path:
            raise ReceiptRefused(f"registry action {action_id!r} economy path is invalid")
        if output_sha256 is not None and (
            type(output_sha256) is not str
            or len(output_sha256) != 64
            or any(char not in "0123456789abcdef" for char in output_sha256)
        ):
            raise ReceiptRefused(
                f"registry action {action_id!r} economy SHA is invalid"
            )
        contract_pins.append((action_id, runtime_path, runtime_sha256))
        try:
            identity = registry.action_source_identity(config)
            identity_sha256 = registry.action_source_identity_sha256(config)
        except Exception as exc:
            raise ReceiptRefused(
                f"registry action {action_id!r} source identity refused: {exc}"
            ) from exc
        try:
            identity_payload = json.loads(_canonical_bytes(identity).decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptRefused(
                f"registry action {action_id!r} source identity is not canonical"
            ) from exc
        if _sha256(_canonical_bytes(identity_payload)) != identity_sha256:
            raise ReceiptRefused(
                f"registry action {action_id!r} source identity SHA drifted"
            )
        source_identities.append(
            {
                "action_id": action_id,
                "identity": identity_payload,
                "sha256": identity_sha256,
            }
        )
        output_paths.add(output_path)
        output_hashes.add(output_sha256)
    if output_paths != {
        "configs/n1_reward_economy_20260802_r9/reward_economy.v1.json"
    }:
        raise ReceiptRefused("reward economy output path drifted")
    if len(output_hashes) != 1:
        raise ReceiptRefused("loop/block reward economy output pins disagree")
    return (
        contract_pins,
        {
            "path": next(iter(output_paths)),
            "sha256": next(iter(output_hashes)),
        },
        source_identities,
    )


def _parse_yaml(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        document = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReceiptRefused(f"{name} is not valid UTF-8 YAML") from exc
    if type(document) is not dict:
        raise ReceiptRefused(f"{name} must be a YAML mapping")
    return document


def _reward_receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if set(receipt) != {"schema_version", "terms", "sha256"}:
        raise ReceiptRefused("effective reward receipt keys drifted")
    if receipt.get("schema_version") != 1 or type(receipt.get("terms")) is not list:
        raise ReceiptRefused("effective reward receipt schema drifted")
    normalized = {
        "schema_version": receipt["schema_version"],
        "terms": receipt["terms"],
    }
    digest = _sha256(
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    if receipt.get("sha256") != digest:
        raise ReceiptRefused("effective reward receipt self-hash drifted")
    if digest != EXPECTED_EFFECTIVE_REWARD_SHA256:
        raise ReceiptRefused(
            "effective reward recipe differs from the adopted common coarse+fine recipe"
        )
    names: list[str] = []
    for index, term in enumerate(receipt["terms"]):
        if type(term) is not dict or set(term) != {
            "callable",
            "name",
            "params",
            "weight",
        }:
            raise ReceiptRefused(f"reward term {index} schema drifted")
        name = term["name"]
        weight = term["weight"]
        if type(name) is not str or not name:
            raise ReceiptRefused(f"reward term {index} name is invalid")
        if type(weight) not in (int, float) or not math.isfinite(float(weight)):
            raise ReceiptRefused(f"reward term {name!r} weight is invalid")
        if float(weight) == 0.0:
            raise ReceiptRefused(f"reward term {name!r} is zero but listed as active")
        if type(term["callable"]) is not str or type(term["params"]) is not dict:
            raise ReceiptRefused(f"reward term {name!r} callable/params are invalid")
        names.append(name)
    if names != sorted(names) or len(set(names)) != len(names):
        raise ReceiptRefused("effective reward terms are not ordered and unique")
    return normalized


def _term_by_name(reward_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {term["name"]: term for term in reward_payload["terms"]}


def _require_term(
    terms: Mapping[str, Mapping[str, Any]],
    name: str,
    *,
    weight: float,
    params_subset: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    term = terms.get(name)
    if term is None or float(term.get("weight", float("nan"))) != weight:
        raise ReceiptRefused(f"reward term {name!r} weight drifted")
    params = term.get("params")
    if type(params) is not dict:
        raise ReceiptRefused(f"reward term {name!r} params drifted")
    for key, expected in (params_subset or {}).items():
        if params.get(key) != expected:
            raise ReceiptRefused(f"reward term {name!r} param {key!r} drifted")
    return term


def _bound_row(
    *,
    name: str,
    members: Sequence[str],
    raw_min: float,
    raw_max: float,
    weighted_dt_min: float,
    weighted_dt_max: float,
    semantics: str,
    source_loci: Sequence[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "members": list(members),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "weighted_dt_min": weighted_dt_min,
        "weighted_dt_max": weighted_dt_max,
        "semantics": semantics,
        "evidence_class": "reviewed_analytic_assumption",
        "source_loci": list(source_loci),
    }


def _theoretical_weighted_dt_bounds(
    reward_payload: Mapping[str, Any], *, step_dt_s: float
) -> list[dict[str, Any]]:
    if step_dt_s != POLICY_STEP_DT_S:
        raise ReceiptRefused("reward step dt must remain exactly 0.02 s")
    terms = _term_by_name(reward_payload)
    _require_term(
        terms,
        "virtual_landing",
        weight=500.0,
        params_subset={"mode": "legal_base", "base_frac": 0.6},
    )
    _require_term(terms, "death_penalty", weight=-300.0)
    _require_term(
        terms,
        "qdes_limit_barrier",
        weight=-5.0,
        params_subset={"margin_frac": 0.08, "penalty_floor": 0.25},
    )
    _require_term(
        terms,
        "joint_limit",
        weight=-5.0,
        params_subset={
            "margin_frac": 0.08,
            "penalty_floor": 0.25,
            "expected_joint_count": 31,
        },
    )
    _require_term(
        terms,
        "action_rate_clamped",
        weight=-0.2,
        params_subset={"value_clamp": 9.0},
    )
    _require_term(
        terms,
        "racket_position",
        weight=4.0,
        params_subset={"std": 0.075},
    )
    _require_term(
        terms,
        "racket_position_coarse",
        weight=1.0,
        params_subset={"std": 0.3},
    )
    _require_term(
        terms,
        "racket_velocity",
        weight=0.5,
        params_subset={"std": 0.5},
    )
    _require_term(
        terms,
        "racket_normal",
        weight=0.5,
        params_subset={"std": 0.262},
    )
    _require_term(terms, "racket_progress", weight=10.0)
    imitation_members = (
        "motion_body_ang_vel",
        "motion_body_lin_vel",
        "motion_body_ori",
        "motion_body_pos",
    )
    for member in imitation_members:
        _require_term(terms, member, weight=1.0)

    # Keep the printed decimal contract human-auditable while checking that it
    # is still the actual weight*dt arithmetic.
    arithmetic = {
        "landing": 500.0 * step_dt_s,
        "death": -300.0 * step_dt_s,
        "qdes_limit": -5.0 * 31.0 * step_dt_s,
        "actual_limit": -5.0 * 31.0 * step_dt_s,
        "action_rate": -0.2 * 9.0 * step_dt_s,
        "fine": 4.0 * step_dt_s,
        "coarse": 1.0 * step_dt_s,
        "velocity": 0.5 * step_dt_s,
        "normal": 0.5 * step_dt_s,
        "progress": 10.0 * 0.15 * step_dt_s,
        "imitation": len(imitation_members) * 1.0 * step_dt_s,
    }
    expected = {
        "landing": 10.0,
        "death": -6.0,
        "qdes_limit": -3.1,
        "actual_limit": -3.1,
        "action_rate": -0.036,
        "fine": 0.08,
        "coarse": 0.02,
        "velocity": 0.01,
        "normal": 0.01,
        "progress": 0.03,
        "imitation": 0.08,
    }
    for key, value in arithmetic.items():
        if not math.isclose(value, expected[key], rel_tol=0.0, abs_tol=1.0e-12):
            raise ReceiptRefused(f"theoretical reward bound {key!r} drifted")

    return [
        _bound_row(
            name="virtual_landing_one_shot",
            members=("virtual_landing",),
            raw_min=0.0,
            raw_max=1.0,
            weighted_dt_min=0.0,
            weighted_dt_max=10.0,
            semantics="eligible exact-strike legal-base kernel",
            source_loci=("mdp/hope_rewards.py:virtual_landing",),
        ),
        _bound_row(
            name="hard_death_one_shot",
            members=("death_penalty",),
            raw_min=0.0,
            raw_max=1.0,
            weighted_dt_min=-6.0,
            weighted_dt_max=0.0,
            semantics="hard-safety termination indicator",
            source_loci=("mdp/hope_rewards.py:action_ball_safety_terminated",),
        ),
        _bound_row(
            name="qdes_limit_barrier_per_step",
            members=("qdes_limit_barrier",),
            raw_min=0.0,
            raw_max=31.0,
            weighted_dt_min=-3.1,
            weighted_dt_max=0.0,
            semantics="sum of bounded processed-qdes per-joint soft-band intrusions",
            source_loci=("mdp/hope_rewards.py:qdes_limit_barrier_v2",),
        ),
        _bound_row(
            name="actual_joint_limit_barrier_per_step",
            members=("joint_limit",),
            raw_min=0.0,
            raw_max=31.0,
            weighted_dt_min=-3.1,
            weighted_dt_max=0.0,
            semantics="sum of bounded actual-q per-joint soft-band intrusions",
            source_loci=("mdp/hope_rewards.py:actual_joint_limit_barrier_v2",),
        ),
        _bound_row(
            name="action_rate_clamped_per_step",
            members=("action_rate_clamped",),
            raw_min=0.0,
            raw_max=9.0,
            weighted_dt_min=-0.036,
            weighted_dt_max=0.0,
            semantics="clamped squared first action difference",
            source_loci=("mdp/hope_rewards.py:action_rate_l2_clamped",),
        ),
        _bound_row(
            name="racket_position_fine_per_step",
            members=("racket_position",),
            raw_min=0.0,
            raw_max=1.0,
            weighted_dt_min=0.0,
            weighted_dt_max=0.08,
            semantics="strike-window exponential kernel",
            source_loci=("mdp/hope_rewards.py:racket_position_tracking_exp",),
        ),
        _bound_row(
            name="racket_position_coarse_per_step",
            members=("racket_position_coarse",),
            raw_min=0.0,
            raw_max=1.0,
            weighted_dt_min=0.0,
            weighted_dt_max=0.02,
            semantics="strike-window exponential companion kernel",
            source_loci=(
                "mdp/hope_rewards.py:racket_position_coarse_tracking_exp",
            ),
        ),
        _bound_row(
            name="racket_velocity_per_step",
            members=("racket_velocity",),
            raw_min=0.0,
            raw_max=1.0,
            weighted_dt_min=0.0,
            weighted_dt_max=0.01,
            semantics="wide-window exponential kernel",
            source_loci=("mdp/hope_rewards.py:racket_velocity_tracking_exp",),
        ),
        _bound_row(
            name="racket_normal_per_step",
            members=("racket_normal",),
            raw_min=0.0,
            raw_max=1.0,
            weighted_dt_min=0.0,
            weighted_dt_max=0.01,
            semantics="wide-window exponential face kernel",
            source_loci=("mdp/hope_rewards.py:racket_normal_tracking_exp",),
        ),
        _bound_row(
            name="racket_progress_per_step",
            members=("racket_progress",),
            raw_min=-0.15,
            raw_max=0.15,
            weighted_dt_min=-0.03,
            weighted_dt_max=0.03,
            semantics="pre-strike previous-minus-current distance clamp",
            source_loci=(
                "mdp/hope_commands.py:RacketTargetCommand._update_footwork_signals",
                "mdp/hope_rewards.py:racket_progress",
            ),
        ),
        _bound_row(
            name="four_body_imitation_sum_per_step",
            members=imitation_members,
            raw_min=0.0,
            raw_max=4.0,
            weighted_dt_min=0.0,
            weighted_dt_max=0.08,
            semantics="sum of four unit-weight [0,1] body imitation kernels",
            source_loci=(
                "mdp/rewards.py:motion_relative_body_position_error_exp",
                "mdp/rewards.py:motion_relative_body_orientation_error_exp",
                "mdp/rewards.py:motion_global_body_linear_velocity_error_exp",
                "mdp/rewards.py:motion_global_body_angular_velocity_error_exp",
                "mdp/hope_rewards.py:motion_body_pos_swing_only",
                "mdp/hope_rewards.py:motion_body_ori_swing_only",
            ),
        ),
    ]


def _validate_task_profile(task: Mapping[str, Any]) -> None:
    if task.get("name") != "HOPEPingPongActionBallA3VendorV1":
        raise ReceiptRefused("vendor task profile name drifted")
    rewards = task.get("rewards")
    if type(rewards) is not dict:
        raise ReceiptRefused("vendor task rewards leaf is missing")
    expected = {
        "action_acc_weight": 0.0,
        "racket_position_coarse_weight": 1.0,
        "racket_position_coarse_std": 0.30,
    }
    for key, value in expected.items():
        if rewards.get(key) != value:
            raise ReceiptRefused(f"vendor task rewards.{key} drifted")
    forbidden_scale_keys = {
        "reward_global_scalar",
        "global_reward_scalar",
        "global_scalar",
    }
    if forbidden_scale_keys.intersection(rewards):
        raise ReceiptRefused("vendor task added a hidden global reward scalar")


def _validate_ppo_yaml(
    ppo: Mapping[str, Any],
    *,
    algorithm: Mapping[str, Any],
    runner: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    if set(ppo) != {"name", "runner", "policy", "algorithm"} or ppo["name"] != "ppo":
        raise ReceiptRefused("cfg/algo/ppo.yaml top-level schema drifted")
    base_runner = ppo["runner"]
    base_policy = ppo["policy"]
    base_algorithm = ppo["algorithm"]
    if not all(type(row) is dict for row in (base_runner, base_policy, base_algorithm)):
        raise ReceiptRefused("cfg/algo/ppo.yaml sections must be mappings")

    for key in (
        "num_steps_per_env",
        "empirical_normalization",
    ):
        if base_runner.get(key) != runner.get(key):
            raise ReceiptRefused(f"PPO runner field {key!r} differs from effective contract")
    for key in (
        "activation",
        "actor_hidden_dims",
        "critic_hidden_dims",
    ):
        if base_policy.get(key) != policy.get(key):
            raise ReceiptRefused(f"PPO policy field {key!r} differs from effective contract")
    if base_policy.get("init_noise_std") != 1.0:
        raise ReceiptRefused("base PPO init_noise_std is no longer the pinned 1.0 default")
    if base_policy.get("noise_std_type") != "scalar":
        raise ReceiptRefused(
            "base PPO noise_std_type must remain the explicit legacy scalar default"
        )
    for key in (
        "clip_param",
        "desired_kl",
        "entropy_coef",
        "gamma",
        "lam",
        "learning_rate",
        "max_grad_norm",
        "num_learning_epochs",
        "num_mini_batches",
        "schedule",
        "use_clipped_value_loss",
        "value_loss_coef",
    ):
        if base_algorithm.get(key) != algorithm.get(key):
            raise ReceiptRefused(f"PPO algorithm field {key!r} differs from effective contract")
    return {
        "base_policy_init_noise_std": 1.0,
        "base_policy_noise_std_type": "scalar",
        "n1_runtime_override_init_noise_std": policy["init_noise_std"],
        "n1_runtime_materialized_noise_std_type": policy["noise_std_type"],
    }


def _validate_contracts(
    repo_root: Path,
    contract_pins: Sequence[tuple[str, str, str]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    float,
]:
    source_rows: list[dict[str, Any]] = []
    common_reward: Optional[dict[str, Any]] = None
    common_algorithm: Optional[dict[str, Any]] = None
    common_runner: Optional[dict[str, Any]] = None
    common_policy: Optional[dict[str, Any]] = None
    step_dt: Optional[float] = None
    if tuple(action_id for action_id, _, _ in contract_pins) != ACTION_IDS:
        raise ReceiptRefused("r6 runtime contract pin order drifted")
    for action_id, path, digest in contract_pins:
        payload, pin = _read_pinned(
            repo_root, path, digest, name=f"r6 {action_id} training contract"
        )
        contract = _strict_json_bytes(payload, name=f"r6 {action_id} training contract")
        if contract.get("schema_version") != 3:
            raise ReceiptRefused(f"r6 {action_id} training contract schema drifted")
        if contract.get("policy_step_dt_s") != POLICY_STEP_DT_S:
            raise ReceiptRefused(f"r6 {action_id} policy_step_dt_s drifted")
        recipe_wrapper = contract.get("action_ball_ppo_runner_recipe")
        if type(recipe_wrapper) is not dict or set(recipe_wrapper) != {
            "schema_version",
            "sha256",
            "recipe",
        }:
            raise ReceiptRefused(f"r6 {action_id} PPO recipe wrapper drifted")
        recipe = recipe_wrapper.get("recipe")
        if type(recipe) is not dict:
            raise ReceiptRefused(f"r6 {action_id} PPO recipe is missing")
        if recipe_wrapper.get("sha256") != _sha256(_canonical_bytes(recipe)):
            raise ReceiptRefused(f"r6 {action_id} PPO recipe self-hash drifted")
        algorithm = recipe.get("algorithm")
        runner = recipe.get("runner")
        policy = recipe.get("policy")
        policy_initialization = recipe.get("policy_initialization")
        action_training = contract.get("action_ball_training")
        bootstrap = (
            action_training.get("policy_bootstrap")
            if type(action_training) is dict
            else None
        )
        bootstrap_initialization = (
            bootstrap.get("initialization")
            if type(bootstrap) is dict
            else None
        )
        if not all(
            type(row) is dict
            for row in (
                algorithm,
                runner,
                policy,
                policy_initialization,
                action_training,
                bootstrap,
                bootstrap_initialization,
            )
        ):
            raise ReceiptRefused(f"r6 {action_id} PPO recipe sections drifted")
        if algorithm != EXPECTED_ALGORITHM or runner != EXPECTED_RUNNER:
            raise ReceiptRefused(f"r6 {action_id} effective PPO algorithm/runner drifted")
        policy_without_std_type = dict(policy)
        noise_std_type = policy_without_std_type.pop("noise_std_type", None)
        if (
            policy_without_std_type != EXPECTED_EFFECTIVE_POLICY_EXCEPT_STD_TYPE
            or noise_std_type != EXPECTED_EFFECTIVE_NOISE_STD_TYPE
            or bootstrap.get("schema_version") != 1
            or bootstrap_initialization.get("noise_std_type") != "log"
            or bootstrap_initialization.get("init_noise_std") != 0.02
            or bootstrap_initialization.get("required_realized_init_noise_std")
            != 0.02
            or policy_initialization != bootstrap
        ):
            raise ReceiptRefused(
                f"r6 {action_id} runtime policy is not exact log/.02"
            )
        reward_receipt = contract.get("effective_reward_recipe")
        if type(reward_receipt) is not dict:
            raise ReceiptRefused(f"r6 {action_id} effective reward receipt is missing")
        reward_payload = _reward_receipt_payload(reward_receipt)
        if (
            action_training.get("effective_reward_recipe_sha256")
            != EXPECTED_EFFECTIVE_REWARD_SHA256
        ):
            raise ReceiptRefused(f"r6 {action_id} training/reward binding drifted")
        source_rows.append({"action_id": action_id, **pin})
        if common_reward is None:
            common_reward = dict(reward_payload)
            common_algorithm = dict(algorithm)
            common_runner = dict(runner)
            common_policy = dict(policy)
            step_dt = float(contract["policy_step_dt_s"])
        elif (
            reward_payload != common_reward
            or algorithm != common_algorithm
            or runner != common_runner
            or policy != common_policy
            or float(contract["policy_step_dt_s"]) != step_dt
        ):
            raise ReceiptRefused("loop/block reward/PPO economies are not identical")
    assert common_reward is not None
    assert common_algorithm is not None
    assert common_runner is not None
    assert common_policy is not None
    assert step_dt is not None
    return (
        source_rows,
        common_reward,
        common_algorithm,
        common_runner,
        common_policy,
        step_dt,
    )


def _runtime_distribution_version() -> str:
    try:
        return importlib_metadata.version(EXPECTED_RSL_RL_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as exc:
        raise ReceiptRefused(
            f"{EXPECTED_RSL_RL_DISTRIBUTION} distribution metadata is unavailable"
        ) from exc


_RSL_SOURCE_MARKERS = {
    "ppo": (
        b"normalize_advantage=not self.normalize_advantage_per_mini_batch",
        b"self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()",
        b"            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)\n",
    ),
    "rollout_storage": (
        b"self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)",
    ),
    "actor_critic": (
        b'elif self.noise_std_type == "log":',
        b"self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))",
        b"std = torch.exp(self.log_std).expand_as(mean)",
        b"return self.distribution.entropy().sum(dim=-1)",
    ),
}


def _validate_rsl_runtime_sources(
    rsl_rl_root: Path, *, distribution_version: str
) -> list[dict[str, str]]:
    if distribution_version != EXPECTED_RSL_RL_VERSION:
        raise ReceiptRefused(
            f"rsl_rl version drift: expected {EXPECTED_RSL_RL_VERSION}, "
            f"got {distribution_version!r}"
        )
    if not rsl_rl_root.is_absolute():
        raise ReceiptRefused("--rsl-rl-root must be absolute")
    try:
        root_stat = rsl_rl_root.lstat()
    except OSError as exc:
        raise ReceiptRefused(f"cannot stat rsl_rl root: {exc}") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or rsl_rl_root.is_symlink():
        raise ReceiptRefused("rsl_rl root must be a real non-symlink directory")
    root = rsl_rl_root.resolve(strict=True)
    rows: list[dict[str, str]] = []
    for role, relative_path, expected_sha256 in RSL_RUNTIME_SOURCE_PINS:
        path = root / relative_path
        payload = _read_regular_bytes(path, name=f"rsl_rl {role} source")
        digest = _sha256(payload)
        if digest != expected_sha256:
            raise ReceiptRefused(
                f"rsl_rl {role} source drift: expected {expected_sha256}, got {digest}"
            )
        for marker in _RSL_SOURCE_MARKERS[role]:
            if marker not in payload:
                raise ReceiptRefused(f"rsl_rl {role} semantic marker is missing")
        rows.append(
            {
                "role": role,
                "module_path": f"rsl_rl/{relative_path}",
                "sha256": digest,
            }
        )
    return rows


def _repository_runtime_sources(repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative_path in REPOSITORY_RUNTIME_SOURCE_PATHS:
        payload = _read_regular_bytes(
            repo_root / relative_path, name=f"repository runtime source {relative_path}"
        )
        rows.append({"path": relative_path, "sha256": _sha256(payload)})
    return rows


def _runtime_telemetry_consumer() -> dict[str, Any]:
    return {
        "status": "wired_probe_gate_runtime_evidence_required",
        "gate": {
            "num_envs": RUNTIME_GATE_NUM_ENVS,
            "ppo_updates": RUNTIME_GATE_UPDATES,
            "steps_per_env_per_update": EXPECTED_RUNNER["num_steps_per_env"],
            "rollout_samples_per_update": (
                RUNTIME_GATE_NUM_ENVS * EXPECTED_RUNNER["num_steps_per_env"]
            ),
        },
        "required_fields": {
            "reward": [
                "pre_advantage_reward_min_mean_p50_p95_p99_max",
                "return_min_mean_p50_p95_p99_max",
                "return_std",
                "explained_variance",
                "value_prediction_min_mean_p50_p95_p99_max",
                "value_residual_min_mean_p50_p95_p99_max",
                "per_term_raw_sum",
                "per_term_weighted_dt_sum",
                "per_term_eligible_denominator",
                "per_term_denominator_semantics",
                "reward_manager_total_sum",
                "per_term_closure_error",
                "reward_manager_closure_max_abs_error",
                "recipe_sha256",
                "pre_advantage_reward_semantics",
            ],
            "advantage": [
                "pre_normalization_mean_std_min_max",
                "post_normalization_mean_std_min_max",
                "post_normalization_finite",
            ],
            "ppo": [
                "surrogate_loss",
                "value_loss",
                "entropy_mean",
                "approx_kl",
                "learning_rate",
                "clip_fraction",
            ],
            "gradient": [
                "pre_clip_actor_mean_parameter_grad_norm",
                "pre_clip_critic_parameter_grad_norm",
                "pre_clip_std_parameter_grad_norm",
                "pre_clip_total_grad_norm",
                "post_clip_total_grad_norm",
                "max_grad_norm",
                "pre_clip_actor_mean_parameter_grad_norm_distribution",
                "pre_clip_critic_parameter_grad_norm_distribution",
                "pre_clip_std_parameter_grad_norm_distribution",
                "pre_clip_total_grad_norm_distribution",
                "post_clip_total_grad_norm_distribution",
                "clip_factor_distribution",
                "optimizer_minibatch_count",
            ],
            "policy": [
                "noise_std_type",
                "policy_std_min",
                "policy_std_mean",
                "policy_std_max",
            ],
        },
        "reject_if": [
            "any_required_field_missing",
            "any_required_value_nonfinite",
            "reward_sum_closure_nonzero_beyond_dtype_tolerance",
            "post_advantage_not_zero_mean_unit_std_beyond_dtype_tolerance",
            "noise_std_type_not_log",
            "policy_std_min_not_strictly_positive",
            "all_five_updates_learning_rate_at_1e-5_floor",
            "any_optimizer_minibatch_postclip_grad_norm_above_1_plus_tolerance",
            "policy_std_or_learning_rate_cross_source_marker_mismatch",
        ],
    }


def _build_payload(
    *,
    repo_root: Path,
    rsl_rl_root: Path,
    distribution_version: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    contract_pins, output_pin, registry_identities = _registry_pins(repo_root)
    tracked_inputs = (
        PRODUCER_SOURCE_PATH,
        ACTION_REGISTRY_PATH,
        TASK_PROFILE_PATH,
        PPO_CONFIG_PATH,
        *REPOSITORY_RUNTIME_SOURCE_PATHS,
        *(path for _, path, _ in contract_pins),
    )
    _tracked_clean(repo_root, tracked_inputs)
    (
        contract_sources,
        reward_payload,
        algorithm,
        runner,
        policy,
        step_dt,
    ) = _validate_contracts(repo_root, contract_pins)
    task_raw, task_pin = _read_source(
        repo_root, TASK_PROFILE_PATH, name="vendor ActionBall task profile"
    )
    ppo_raw, ppo_pin = _read_source(
        repo_root, PPO_CONFIG_PATH, name="repository PPO config"
    )
    _, producer_pin = _read_source(
        repo_root, PRODUCER_SOURCE_PATH, name="reward/PPO receipt producer"
    )
    task_profile = _parse_yaml(task_raw, name="vendor ActionBall task profile")
    ppo_config = _parse_yaml(ppo_raw, name="repository PPO config")
    _validate_task_profile(task_profile)
    std_provenance = _validate_ppo_yaml(
        ppo_config,
        algorithm=algorithm,
        runner=runner,
        policy=policy,
    )
    rsl_sources = _validate_rsl_runtime_sources(
        rsl_rl_root, distribution_version=distribution_version
    )
    bounds = _theoretical_weighted_dt_bounds(reward_payload, step_dt_s=step_dt)
    repository_sources = _repository_runtime_sources(repo_root)
    _tracked_clean(repo_root, tracked_inputs)

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "authorization": {
            "diagnostic_unauthorized": True,
            "training": False,
            "resume": False,
            "promotion": False,
            "export": False,
            "judge": False,
            "deployment": False,
            "hardware": False,
        },
        "sources": {
            "producer": producer_pin,
            "registry_action_source_identities": registry_identities,
            "runtime_training_contracts": contract_sources,
            "registry_output": {"path": output_pin["path"]},
            "task_profile": task_pin,
            "ppo_config": ppo_pin,
            "repository_runtime_sources": repository_sources,
            "rsl_rl": {
                "distribution": EXPECTED_RSL_RL_DISTRIBUTION,
                "version": distribution_version,
                "source_files": rsl_sources,
            },
        },
        "reward_economy": {
            "reward_global_scalar": REWARD_GLOBAL_SCALAR,
            "reward_global_scalar_source": (
                "identity_multiplier_no_runtime_global_override"
            ),
            "policy_step_dt_s": step_dt,
            "effective_reward_recipe_sha256": EXPECTED_EFFECTIVE_REWARD_SHA256,
            "ordered_nonzero_terms": reward_payload["terms"],
            "theoretical_bound_evidence_class": (
                "reviewed_analytic_assumption_bound_to_exact_sources"
            ),
            "theoretical_weighted_dt_bounds": bounds,
        },
        "ppo_economy": {
            "algorithm": dict(algorithm),
            "runner": dict(runner),
            "advantage_normalization": {
                "mode": "whole_rollout",
                "normalize_advantage_per_mini_batch": False,
                "storage_compute_returns_normalize_advantage": True,
                "rollout_samples_at_4096_env": (
                    RUNTIME_GATE_NUM_ENVS * runner["num_steps_per_env"]
                ),
                "source_role": "rollout_storage",
            },
            "loss_and_optimizer": {
                "entropy_coef": algorithm["entropy_coef"],
                "value_loss_coef": algorithm["value_loss_coef"],
                "use_clipped_value_loss": algorithm["use_clipped_value_loss"],
                "policy_and_value_clip_param": algorithm["clip_param"],
                "schedule": algorithm["schedule"],
                "desired_kl": algorithm["desired_kl"],
                "initial_learning_rate": algorithm["learning_rate"],
                "gamma": algorithm["gamma"],
                "lam": algorithm["lam"],
                "max_grad_norm": algorithm["max_grad_norm"],
            },
            "std_parameterization_provenance": std_provenance,
            "required_final_policy": {
                "fresh_only": True,
                "resume_from_scalar_checkpoint_prohibited": True,
                "noise_std_type": REQUIRED_FINAL_NOISE_STD_TYPE,
                "parameter_name": "log_std",
                "init_config_sigma": REQUIRED_INITIAL_REALIZED_SIGMA,
                "required_realized_init_noise_std": REQUIRED_INITIAL_REALIZED_SIGMA,
                "strictly_positive_by_construction": True,
            },
        },
        "runtime_4096x5_telemetry_consumer": _runtime_telemetry_consumer(),
    }


def build_receipt(
    *,
    repo_root: Path,
    rsl_rl_root: Path,
    distribution_version: Optional[str] = None,
) -> dict[str, Any]:
    payload = _build_payload(
        repo_root=repo_root,
        rsl_rl_root=rsl_rl_root,
        distribution_version=(
            _runtime_distribution_version()
            if distribution_version is None
            else distribution_version
        ),
    )
    return {**payload, "content_sha256": _sha256(_canonical_bytes(payload))}


def validate_receipt_document(document: Mapping[str, Any]) -> dict[str, Any]:
    if type(document) is not dict:
        raise ReceiptRefused("reward/PPO economy receipt must be an object")
    expected_keys = {
        "schema_version",
        "kind",
        "authorization",
        "sources",
        "reward_economy",
        "ppo_economy",
        "runtime_4096x5_telemetry_consumer",
        "content_sha256",
    }
    if set(document) != expected_keys:
        raise ReceiptRefused("reward/PPO economy receipt keys drifted")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("kind") != KIND:
        raise ReceiptRefused("reward/PPO economy receipt schema/kind drifted")
    digest = document.get("content_sha256")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ReceiptRefused("reward/PPO economy receipt SHA is invalid")
    payload = {
        key: value for key, value in document.items() if key != "content_sha256"
    }
    if _sha256(_canonical_bytes(payload)) != digest:
        raise ReceiptRefused("reward/PPO economy receipt self-hash drifted")
    return dict(document)


def _absolute_path(raw: str, *, name: str, must_exist: bool) -> Path:
    if type(raw) is not str or not raw:
        raise ReceiptRefused(f"{name} must be a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        raise ReceiptRefused(f"{name} must be absolute")
    if must_exist and not path.exists():
        raise ReceiptRefused(f"{name} does not exist: {path}")
    return path


def _write_no_clobber(path: Path, document: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise ReceiptRefused(f"output path is already spent: {path}")
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except OSError as exc:
        raise ReceiptRefused(f"cannot stat output parent: {exc}") from exc
    if not stat.S_ISDIR(parent_stat.st_mode) or parent.is_symlink():
        raise ReceiptRefused("output parent must be a real non-symlink directory")
    payload = _canonical_bytes(document) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ReceiptRefused(f"cannot write receipt without clobber: {exc}") from exc


def _registry_output_path(repo_root: Path, output_pin: Mapping[str, Any]) -> Path:
    expected_relative = (
        "configs/n1_reward_economy_20260802_r9/reward_economy.v1.json"
    )
    if output_pin.get("path") != expected_relative:
        raise ReceiptRefused("registry reward economy output path drifted")
    output = repo_root / expected_relative
    expected_parent = repo_root / "configs/n1_reward_economy_20260802_r9"
    if output.parent != expected_parent:
        raise ReceiptRefused("registry reward economy output escaped its fixed directory")
    return output


def _ensure_registry_output_parent(repo_root: Path, output: Path) -> None:
    parent = output.parent
    if not os.path.lexists(parent):
        configs = repo_root / "configs"
        try:
            configs_stat = configs.lstat()
        except OSError as exc:
            raise ReceiptRefused(f"cannot stat configs directory: {exc}") from exc
        if not stat.S_ISDIR(configs_stat.st_mode) or configs.is_symlink():
            raise ReceiptRefused("configs must be a real non-symlink directory")
        try:
            parent.mkdir()
        except OSError as exc:
            raise ReceiptRefused(f"cannot create fixed receipt directory: {exc}") from exc
    try:
        parent_stat = parent.lstat()
    except OSError as exc:
        raise ReceiptRefused(f"cannot stat fixed receipt directory: {exc}") from exc
    if not stat.S_ISDIR(parent_stat.st_mode) or parent.is_symlink():
        raise ReceiptRefused("fixed receipt directory must be real and non-symlink")


def _verify_file(
    path: Path,
    *,
    repo_root: Path,
    rsl_rl_root: Path,
    distribution_version: str,
    expected_file_sha256: str,
) -> dict[str, Any]:
    raw = _read_regular_bytes(path, name="reward/PPO economy receipt")
    if _sha256(raw) != expected_file_sha256:
        raise ReceiptRefused("reward/PPO economy receipt differs from registry file SHA")
    document = validate_receipt_document(
        _strict_json_bytes(raw, name="reward/PPO economy receipt")
    )
    if raw != _canonical_bytes(document) + b"\n":
        raise ReceiptRefused("reward/PPO economy receipt is not canonical JSON plus newline")
    actual = build_receipt(
        repo_root=repo_root,
        rsl_rl_root=rsl_rl_root,
        distribution_version=distribution_version,
    )
    if document != actual:
        raise ReceiptRefused("reward/PPO economy receipt differs from live pinned inputs")
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rsl-rl-root",
        required=True,
        help="absolute installed rsl_rl package directory for the exact Pod interpreter",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--materialize",
        action="store_true",
        help="write the registry-planned no-clobber receipt (registry SHA must be None)",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="revalidate the registry-pinned receipt (registry SHA must be materialized)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rsl_root = _absolute_path(
            args.rsl_rl_root, name="--rsl-rl-root", must_exist=True
        )
        repo_root = _repo_root()
        version = _runtime_distribution_version()
        _, output_pin, _ = _registry_pins(repo_root)
        output = _registry_output_path(repo_root, output_pin)
        if args.materialize:
            if output_pin["sha256"] is not None:
                raise ReceiptRefused(
                    "registry reward economy receipt is already materialized; use --verify"
                )
            _ensure_registry_output_parent(repo_root, output)
            receipt = build_receipt(
                repo_root=repo_root,
                rsl_rl_root=rsl_root,
                distribution_version=version,
            )
            _write_no_clobber(output, receipt)
            print(
                "ACTION_BALL_REWARD_PPO_ECONOMY_RECEIPT="
                + json.dumps(
                    {
                        "path": str(output),
                        "content_sha256": receipt["content_sha256"],
                        "file_sha256": _sha256(_canonical_bytes(receipt) + b"\n"),
                        "effective_reward_recipe_sha256": (
                            receipt["reward_economy"][
                                "effective_reward_recipe_sha256"
                            ]
                        ),
                        "reward_global_scalar": REWARD_GLOBAL_SCALAR,
                        "required_noise_std_type": REQUIRED_FINAL_NOISE_STD_TYPE,
                    },
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            expected_file_sha = output_pin["sha256"]
            if type(expected_file_sha) is not str:
                raise ReceiptRefused(
                    "registry reward economy receipt SHA is not materialized"
                )
            receipt = _verify_file(
                output,
                repo_root=repo_root,
                rsl_rl_root=rsl_root,
                distribution_version=version,
                expected_file_sha256=expected_file_sha,
            )
            print(
                "ACTION_BALL_REWARD_PPO_ECONOMY_VERIFIED="
                + json.dumps(
                    {
                        "path": str(output),
                        "content_sha256": receipt["content_sha256"],
                        "file_sha256": expected_file_sha,
                    },
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0
    except ReceiptRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
