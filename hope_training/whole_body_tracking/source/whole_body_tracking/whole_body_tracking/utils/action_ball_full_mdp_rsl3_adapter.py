"""One RSL-RL 3 optimizer edge for diagnostic FullMDP.

The environment owns every semantic fact.  This adapter owns only the order
between the zero-argument PPO update and the two durable WAL records.  It is
not a checkpoint, a readiness receipt, or a second telemetry authority.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Callable

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


RUN_MODE = "single_action_lean"
TELEMETRY_SCHEMA_VERSION = 13
TELEMETRY_KIND = "action_ball_epoch_optimizer_update_ack_telemetry_v13"
TRAINING_CONTRACT_SCHEMA_VERSION = 3
ACTOR_OBSERVATION_CONTRACT = "action_ball_full_mdp_semantic_actor_v3"
ACTOR_OBSERVATION_WIDTH = 215
CRITIC_OBSERVATION_CONTRACT = "action_ball_full_mdp_semantic_critic_v3"
CRITIC_OBSERVATION_WIDTH = 231
OBSERVATION_KIND = "action_ball_full_mdp_semantic_observation_v3"
SNAPSHOT_RECEIPT_SCHEMA_VERSION = 2
SNAPSHOT_RECEIPT_KIND = "action_ball_full_mdp_diagnostic_snapshot_receipt_v2"
SNAPSHOT_PAYLOAD_KIND = "policy_optimizer_diagnostic_nonresumable_v2"
_STDOUT_WARNING_PREFIX = "HOPE_NONAUTHORITATIVE_STDOUT_WARNING_JSON="
_SHOT_SAMPLE_LIMIT = 4
_RESET_SAMPLE_LIMIT = 8

_SNAPSHOT_IDENTITY_KEYS = {
    "action_ball_full_mdp_snapshot_kind",
    "fresh_full_mdp_observation_kind",
    "actor_obs_contract",
    "actor_obs_total_dim",
    "critic_obs_contract",
    "critic_obs_total_dim",
    "training_contract_schema_version",
    "training_contract_sha256",
    "diagnostic_unauthorized",
    "checkpoint_authority",
    "resume_authority",
}


def _emit_non_authoritative_stdout_marker(
    marker: str,
    *,
    marker_name: str,
    durable_wal_authoritative: bool,
) -> None:
    """Best-effort stdout delivery with an explicit durability scope."""

    if type(durable_wal_authoritative) is not bool:
        raise TypeError("stdout marker durability scope must be an exact bool")

    reported_characters = None
    failure = None
    try:
        reported_characters = sys.stdout.write(marker)
        if (
            type(reported_characters) is not int
            or reported_characters != len(marker)
        ):
            failure = "short_write"
        else:
            sys.stdout.flush()
            return
    except Exception as exc:
        # Stdout is explicitly outside the training transaction.
        failure = type(exc).__module__ + "." + type(exc).__qualname__

    warning = {
        "event": "hope_nonauthoritative_stdout_delivery_warning",
        "schema_version": 1,
        "marker": marker_name,
        "failure": failure,
        "expected_character_count": len(marker),
        "reported_character_count": (
            reported_characters if type(reported_characters) is int else None
        ),
        "stdout_authoritative": False,
        "durable_wal_authoritative": durable_wal_authoritative,
        "durable_scope": (
            "action_epoch" if durable_wal_authoritative else "none"
        ),
        "training_transaction": (
            "epoch_ack_committed"
            if durable_wal_authoritative
            else "completed_in_process_not_durable"
        ),
    }
    try:
        sys.stderr.write(
            _STDOUT_WARNING_PREFIX
            + json.dumps(warning, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        sys.stderr.flush()
    except Exception:
        # A second non-authoritative stream failure must not undo or poison a
        # completed in-process update either.
        pass


def _validate_training_contract_identity(
    schema_version: object, sha256: object
) -> tuple[int, str]:
    if (
        type(schema_version) is not int
        or schema_version != TRAINING_CONTRACT_SCHEMA_VERSION
    ):
        raise RuntimeError("FullMDP RSL3 training-contract schema differs")
    if (
        type(sha256) is not str
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise RuntimeError("FullMDP RSL3 training-contract SHA differs")
    return schema_version, sha256


def _validate_observation_identity(env: object) -> dict[str, object]:
    observations = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_observation_cfg"
    )
    resolver = getattr(observations, "installed_observation_facts", None)
    if not callable(resolver):
        raise RuntimeError("FullMDP RSL3 observation identity resolver differs")
    facts = resolver(env)
    expected = {
        "actor_obs_contract": ACTOR_OBSERVATION_CONTRACT,
        "actor_obs_mode": "action_ball_full_mdp",
        "actor_obs_total_dim": ACTOR_OBSERVATION_WIDTH,
        "actor_obs_term_names": ["action_epoch"],
        "actor_obs_term_dims": [ACTOR_OBSERVATION_WIDTH],
        "critic_obs_contract": CRITIC_OBSERVATION_CONTRACT,
        "critic_obs_total_dim": CRITIC_OBSERVATION_WIDTH,
        "critic_obs_term_names": ["action_epoch"],
        "critic_obs_term_dims": [CRITIC_OBSERVATION_WIDTH],
        "fresh_full_mdp_observation_kind": OBSERVATION_KIND,
        "fresh_full_mdp_diagnostic_unauthorized": True,
        "fresh_full_mdp_launch_authorized": False,
        "fresh_full_mdp_no_capacity_receipt_or_sha_authority": True,
    }
    if type(facts) is not dict or facts != expected:
        raise RuntimeError("FullMDP RSL3 semantic observation identity differs")
    return {
        "fresh_full_mdp_observation_kind": facts[
            "fresh_full_mdp_observation_kind"
        ],
        "actor_obs_contract": facts["actor_obs_contract"],
        "actor_obs_total_dim": facts["actor_obs_total_dim"],
        "critic_obs_contract": facts["critic_obs_contract"],
        "critic_obs_total_dim": facts["critic_obs_total_dim"],
    }


def _validate_snapshot_identity_infos(
    required_infos: object,
) -> dict[str, object]:
    if (
        type(required_infos) is not dict
        or set(required_infos) != _SNAPSHOT_IDENTITY_KEYS
    ):
        raise RuntimeError("single_action_lean diagnostic snapshot identity differs")
    schema_version, sha256 = _validate_training_contract_identity(
        required_infos["training_contract_schema_version"],
        required_infos["training_contract_sha256"],
    )
    expected = {
        "action_ball_full_mdp_snapshot_kind": SNAPSHOT_PAYLOAD_KIND,
        "fresh_full_mdp_observation_kind": OBSERVATION_KIND,
        "actor_obs_contract": ACTOR_OBSERVATION_CONTRACT,
        "actor_obs_total_dim": ACTOR_OBSERVATION_WIDTH,
        "critic_obs_contract": CRITIC_OBSERVATION_CONTRACT,
        "critic_obs_total_dim": CRITIC_OBSERVATION_WIDTH,
        "training_contract_schema_version": schema_version,
        "training_contract_sha256": sha256,
        "diagnostic_unauthorized": True,
        "checkpoint_authority": False,
        "resume_authority": False,
    }
    if required_infos != expected:
        raise RuntimeError("single_action_lean diagnostic snapshot identity differs")
    return expected


def _write_snapshot_receipt(
    snapshot: Path, *, learning_iteration: int, required_infos: dict[str, object]
) -> Path:
    """Bind one completed upstream save to a durable, non-authoritative receipt."""

    snapshot_identity = _validate_snapshot_identity_infos(required_infos)
    torch = importlib.import_module("torch")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(snapshot, flags)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
        ):
            raise RuntimeError(
                "single_action_lean diagnostic snapshot is not a non-empty "
                "one-link regular file"
            )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(
                    "single_action_lean diagnostic snapshot shortened during read"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as payload_stream:
            payload = torch.load(
                payload_stream, map_location="cpu", weights_only=True
            )
        if type(payload) is not dict or set(payload) != {
            "model_state_dict",
            "optimizer_state_dict",
            "iter",
            "infos",
        }:
            raise RuntimeError(
                "single_action_lean diagnostic snapshot payload shape differs"
            )
        if type(payload["iter"]) is not int or payload["iter"] != learning_iteration:
            raise RuntimeError(
                "single_action_lean diagnostic snapshot iteration differs"
            )
        infos = payload["infos"]
        if type(infos) is not dict or any(
            infos.get(key) != value for key, value in snapshot_identity.items()
        ):
            raise RuntimeError(
                "single_action_lean diagnostic snapshot metadata differs"
            )

        def count_finite_tensors(value: object) -> int:
            if isinstance(value, torch.Tensor):
                if not bool(torch.isfinite(value).all().item()):
                    raise RuntimeError(
                        "single_action_lean diagnostic snapshot tensor is non-finite"
                    )
                return 1
            if isinstance(value, dict):
                if any(type(key) not in (str, int) for key in value):
                    raise RuntimeError(
                        "single_action_lean diagnostic snapshot mapping key differs"
                    )
                return sum(count_finite_tensors(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(count_finite_tensors(item) for item in value)
            if type(value) is float and not math.isfinite(value):
                raise RuntimeError(
                    "single_action_lean diagnostic snapshot scalar is non-finite"
                )
            if value is not None and type(value) not in (str, int, float, bool):
                raise RuntimeError(
                    "single_action_lean diagnostic snapshot value type differs"
                )
            return 0

        model_state = payload["model_state_dict"]
        optimizer_state = payload["optimizer_state_dict"]
        if not isinstance(model_state, dict) or not model_state:
            raise RuntimeError(
                "single_action_lean diagnostic snapshot model state differs"
            )
        if not isinstance(optimizer_state, dict) or not optimizer_state:
            raise RuntimeError(
                "single_action_lean diagnostic snapshot optimizer state differs"
            )
        model_tensor_count = count_finite_tensors(model_state)
        optimizer_tensor_count = count_finite_tensors(optimizer_state)
        if model_tensor_count <= 0 or optimizer_tensor_count <= 0:
            raise RuntimeError(
                "single_action_lean diagnostic snapshot tensor inventory differs"
            )
        os.lseek(fd, 0, os.SEEK_SET)
        verification_digest = hashlib.sha256()
        verification_remaining = before.st_size
        while verification_remaining:
            chunk = os.read(fd, min(1024 * 1024, verification_remaining))
            if not chunk:
                raise RuntimeError(
                    "single_action_lean diagnostic snapshot shortened after decode"
                )
            verification_digest.update(chunk)
            verification_remaining -= len(chunk)
        if verification_digest.digest() != digest.digest():
            raise RuntimeError(
                "single_action_lean diagnostic snapshot bytes changed during decode"
            )
        after = os.fstat(fd)
        current = os.stat(snapshot, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size)
        if identity != (after.st_dev, after.st_ino, after.st_size) or identity[:2] != (
            current.st_dev,
            current.st_ino,
        ):
            raise RuntimeError(
                "single_action_lean diagnostic snapshot identity changed"
            )
    finally:
        os.close(fd)

    receipt = {
        "schema_version": SNAPSHOT_RECEIPT_SCHEMA_VERSION,
        "kind": SNAPSHOT_RECEIPT_KIND,
        "snapshot_name": snapshot.name,
        "learning_iteration": learning_iteration,
        "snapshot_size_bytes": before.st_size,
        "snapshot_sha256": digest.hexdigest(),
        "model_tensor_count": model_tensor_count,
        "optimizer_tensor_count": optimizer_tensor_count,
        "all_tensors_finite": True,
        **snapshot_identity,
    }
    encoded = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    receipt_path = snapshot.with_name(snapshot.name + ".receipt.json")
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    write_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    receipt_fd = os.open(receipt_path, write_flags, 0o600)
    try:
        written = 0
        while written < len(encoded):
            count = os.write(receipt_fd, encoded[written:])
            if count <= 0:
                raise OSError("single_action_lean snapshot receipt write was short")
            written += count
        os.fsync(receipt_fd)
        info = os.fstat(receipt_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(
                "single_action_lean snapshot receipt is not a unique regular file"
            )
    finally:
        os.close(receipt_fd)
    directory_fd = os.open(
        receipt_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return receipt_path


def _bound(owner: object, name: str) -> Callable:
    method = getattr(owner, name, None)
    function = vars(type(owner)).get(name)
    if (
        not callable(method)
        or not callable(function)
        or getattr(method, "__self__", None) is not owner
        or getattr(method, "__func__", None) is not function
    ):
        raise RuntimeError(f"FullMDP owner lacks exact bound method {name}")
    return method


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"FullMDP compact joint-safety {name} differs")
    return value


def _tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    kind: str,
) -> object:
    if tuple(getattr(value, "shape", ())) != shape:
        raise RuntimeError(f"FullMDP compact joint-safety {name} shape differs")
    dtype = str(getattr(value, "dtype", "")).rsplit(".", 1)[-1]
    allowed = {
        "integer": {"int64", "long"},
        "boolean": {"bool"},
        "floating": {"float32"},
    }[kind]
    if dtype not in allowed:
        raise RuntimeError(f"FullMDP compact joint-safety {name} dtype differs")
    for method_name in ("eq", "ge", "gt", "le", "all", "sum", "item"):
        if not callable(getattr(value, method_name, None)):
            raise RuntimeError(
                f"FullMDP compact joint-safety {name} tensor API differs"
            )
    if kind == "floating" and not callable(getattr(value, "isfinite", None)):
        raise RuntimeError(
            f"FullMDP compact joint-safety {name} finite API differs"
        )
    return value


def _scalar_bool(value: object, *, name: str) -> bool:
    item = getattr(value, "item", None)
    result = item() if callable(item) else None
    if type(result) is not bool:
        raise RuntimeError(f"FullMDP compact joint-safety {name} scalar differs")
    return result


def _scalar_number(value: object, *, name: str) -> int | float:
    item = getattr(value, "item", None)
    result = item() if callable(item) else None
    if isinstance(result, bool) or type(result) not in (int, float):
        raise RuntimeError(f"FullMDP compact joint-safety {name} scalar differs")
    return result


def _validate_compact_joint_safety(
    snapshot: object,
    *,
    expected_num_envs: int,
    expected_policy_steps: int,
    expected_apply_calls: int,
    previous_consume_sequence: int | None,
    previous_policy_step_sequence: int | None,
) -> dict[str, object]:
    """Validate only the typed compact producer; never infer legacy attribution."""

    top_keys = {
        "schema_version",
        "enabled",
        "diagnostic_compact_evidence",
        "diagnostic_first_policy_step_sequence",
        "diagnostic_last_policy_step_sequence",
        "since_last_consume",
        "terminal_archives",
        "identity_bound_policy_steps",
        "policy_step_summary_capacity",
        "policy_step_summary_used",
        "policy_step_summary_payload_bytes",
        "policy_step_summary_overflow_latch",
        "policy_step_summary_overflow_count",
        "terminal_archive_capacity",
        "terminal_archive_used",
        "terminal_archive_payload_bytes",
        "terminal_archive_overflow_latch",
        "terminal_archive_overflow_count",
    }
    if type(snapshot) is not dict or set(snapshot) != top_keys:
        raise RuntimeError("FullMDP compact joint-safety snapshot ABI differs")
    if (
        snapshot["schema_version"] != 1
        or snapshot["enabled"] is not True
        or snapshot["diagnostic_compact_evidence"] is not True
        or snapshot["terminal_archives"] != ()
        or snapshot["identity_bound_policy_steps"] != ()
    ):
        raise RuntimeError("FullMDP compact joint-safety producer mode differs")
    for prefix in ("policy_step_summary", "terminal_archive"):
        _exact_int(snapshot[f"{prefix}_capacity"], name=f"{prefix}_capacity", minimum=1)
        if (
            snapshot[f"{prefix}_used"] != 0
            or snapshot[f"{prefix}_payload_bytes"] != 0
            or snapshot[f"{prefix}_overflow_latch"] is not False
            or snapshot[f"{prefix}_overflow_count"] != 0
        ):
            raise RuntimeError(
                f"FullMDP compact joint-safety {prefix} is not empty"
            )

    since_keys = {
        "consume_sequence",
        "has_data",
        "identity_bound_policy_step_count",
        "policy_step_count",
        "complete_policy_step_count",
        "incomplete_policy_step_count",
        "apply_readback_count",
        "post_readback_count",
        "timestamp_invariant_pass_count",
        "hard_crossing_latch",
        "actual_hard_edge_latch",
        "qdes_joint_count",
        "policy_crossing_joint_count",
        "substep_hard_crossing_joint_count",
        "actual_hard_edge_joint_count",
        "minimum_hard_lower_gap",
        "minimum_hard_upper_gap",
    }
    since = snapshot["since_last_consume"]
    if type(since) is not dict or set(since) != since_keys:
        raise RuntimeError("FullMDP compact joint-safety aggregate ABI differs")
    if since["has_data"] is not True or since["identity_bound_policy_step_count"] != 0:
        raise RuntimeError("FullMDP compact joint-safety aggregate is absent or dense")
    consume_sequence = _exact_int(
        since["consume_sequence"], name="consume_sequence"
    )
    expected_consume_sequence = (
        0 if previous_consume_sequence is None else previous_consume_sequence + 1
    )
    if consume_sequence != expected_consume_sequence:
        raise RuntimeError("FullMDP compact joint-safety consume sequence differs")

    env_shape = (expected_num_envs,)
    policy_steps = _tensor(
        since["policy_step_count"],
        name="policy_step_count",
        shape=env_shape,
        kind="integer",
    )
    complete_steps = _tensor(
        since["complete_policy_step_count"],
        name="complete_policy_step_count",
        shape=env_shape,
        kind="integer",
    )
    incomplete_steps = _tensor(
        since["incomplete_policy_step_count"],
        name="incomplete_policy_step_count",
        shape=env_shape,
        kind="integer",
    )
    apply_readbacks = _tensor(
        since["apply_readback_count"],
        name="apply_readback_count",
        shape=env_shape,
        kind="integer",
    )
    post_readbacks = _tensor(
        since["post_readback_count"],
        name="post_readback_count",
        shape=env_shape,
        kind="integer",
    )
    timestamp_passes = _tensor(
        since["timestamp_invariant_pass_count"],
        name="timestamp_invariant_pass_count",
        shape=env_shape,
        kind="integer",
    )
    if not all(
        _scalar_bool(check, name=name)
        for name, check in (
            ("policy_step_count", policy_steps.eq(expected_policy_steps).all()),
            ("complete_policy_step_count", complete_steps.eq(expected_policy_steps).all()),
            ("incomplete_policy_step_count", incomplete_steps.eq(0).all()),
            (
                "apply_readback_count",
                apply_readbacks.eq(
                    expected_policy_steps * expected_apply_calls
                ).all(),
            ),
            ("post_readback_count", post_readbacks.eq(expected_policy_steps).all()),
            ("timestamp_invariant_pass_count", timestamp_passes.eq(expected_policy_steps).all()),
        )
    ):
        raise RuntimeError("FullMDP compact joint-safety rollout is incomplete")

    qdes = since["qdes_joint_count"]
    joint_shape = tuple(getattr(qdes, "shape", ()))
    if (
        len(joint_shape) != 2
        or joint_shape[0] != expected_num_envs
        or type(joint_shape[1]) is not int
        or joint_shape[1] <= 0
    ):
        raise RuntimeError("FullMDP compact joint-safety joint shape differs")
    integer_joint_names = (
        "qdes_joint_count",
        "policy_crossing_joint_count",
        "substep_hard_crossing_joint_count",
        "actual_hard_edge_joint_count",
    )
    joint_counts = {
        name: _tensor(since[name], name=name, shape=joint_shape, kind="integer")
        for name in integer_joint_names
    }
    if any(
        not _scalar_bool(value.ge(0).all(), name=f"{name}_nonnegative")
        for name, value in joint_counts.items()
    ):
        raise RuntimeError("FullMDP compact joint-safety counter is negative")
    hard_latch = _tensor(
        since["hard_crossing_latch"],
        name="hard_crossing_latch",
        shape=env_shape,
        kind="boolean",
    )
    actual_latch = _tensor(
        since["actual_hard_edge_latch"],
        name="actual_hard_edge_latch",
        shape=env_shape,
        kind="boolean",
    )
    minimum_lower = _tensor(
        since["minimum_hard_lower_gap"],
        name="minimum_hard_lower_gap",
        shape=joint_shape,
        kind="floating",
    )
    minimum_upper = _tensor(
        since["minimum_hard_upper_gap"],
        name="minimum_hard_upper_gap",
        shape=joint_shape,
        kind="floating",
    )
    if not (
        _scalar_bool(minimum_lower.isfinite().all(), name="minimum_lower_finite")
        and _scalar_bool(minimum_upper.isfinite().all(), name="minimum_upper_finite")
    ):
        raise RuntimeError("FullMDP compact joint-safety gap is non-finite")
    substep_rows = joint_counts["substep_hard_crossing_joint_count"].gt(0).any(dim=1)
    actual_rows = joint_counts["actual_hard_edge_joint_count"].gt(0).any(dim=1)
    actual_from_gap = minimum_lower.le(0) | minimum_upper.le(0)
    if not (
        _scalar_bool(hard_latch.eq(substep_rows).all(), name="hard_latch")
        and _scalar_bool(actual_latch.eq(actual_rows).all(), name="actual_latch")
        and _scalar_bool(
            joint_counts["actual_hard_edge_joint_count"]
            .gt(0)
            .eq(actual_from_gap)
            .all(),
            name="actual_edge_gap_equivalence",
        )
    ):
        raise RuntimeError("FullMDP compact joint-safety edge evidence differs")

    first_sequence = _exact_int(
        snapshot["diagnostic_first_policy_step_sequence"],
        name="first_policy_step_sequence",
    )
    last_sequence = _exact_int(
        snapshot["diagnostic_last_policy_step_sequence"],
        name="last_policy_step_sequence",
    )
    if (
        last_sequence - first_sequence + 1 != expected_policy_steps
        or (
            previous_policy_step_sequence is None
            and first_sequence != 0
        )
        or (
            previous_policy_step_sequence is not None
            and first_sequence != previous_policy_step_sequence + 1
        )
    ):
        raise RuntimeError("FullMDP compact joint-safety policy-step span differs")

    totals = {
        name: int(_scalar_number(value.sum(), name=f"{name}_total"))
        for name, value in joint_counts.items()
    }
    minimum_gap = min(
        float(_scalar_number(minimum_lower.amin(), name="minimum_lower")),
        float(_scalar_number(minimum_upper.amin(), name="minimum_upper")),
    )
    return {
        "consume_sequence": consume_sequence,
        "first_policy_step_sequence": first_sequence,
        "last_policy_step_sequence": last_sequence,
        "counter_totals": totals,
        "minimum_hard_gap_rad": minimum_gap,
    }


def _shot_row(shot: object, *, lifecycle_flags: tuple[str, ...]) -> dict:
    row = asdict(shot)
    evidence = row["evidence"]
    bits = evidence["lifecycle_bits"]
    lifecycle = {
        name: bool(bits & (1 << ordinal))
        for ordinal, name in enumerate(lifecycle_flags)
    }
    evidence["lifecycle"] = lifecycle
    evidence["contact_face"] = {"availability": "not_produced"}
    evidence["recovery_horizon"] = {"availability": "not_produced"}
    if not lifecycle["physical_launched"]:
        row["target_x_m"] = None
        row["target_y_m"] = None
    return row


def _action_identity(row: object) -> tuple[int, int, str, bool]:
    """Read the one owner-produced action/side identity used for strata."""

    values = (
        getattr(row, "action_uid", None),
        getattr(row, "action_slot", None),
        getattr(row, "stroke_family", None),
        getattr(
            row,
            "attribution_valid",
            getattr(row, "action_attribution_valid", None),
        ),
    )
    if (
        type(values[0]) is not int
        or type(values[1]) is not int
        or type(values[2]) is not str
        or not values[2]
        or type(values[3]) is not bool
    ):
        raise RuntimeError("FullMDP telemetry action stratum identity differs")
    return values


def _identity_payload(identity: tuple[int, int, str, bool]) -> dict:
    return {
        "action_uid": identity[0],
        "action_slot": identity[1],
        "stroke_family": identity[2],
        "action_attribution_valid": identity[3],
    }


def _opportunity_strata(rows: tuple[object, ...]) -> list[dict]:
    """Replace unbounded D05 rows with exact per-action/per-side counters."""

    grouped: dict[tuple[int, int, str, bool], dict[str, int]] = {}
    flags = ("selected", "accepted", "censored", "rejected", "deferred")
    for row in rows:
        identity = _action_identity(row)
        counts = grouped.setdefault(
            identity,
            {"opportunity_rows": 0, **{name + "_rows": 0 for name in flags}},
        )
        counts["opportunity_rows"] += 1
        for name in flags:
            value = getattr(row, name, None)
            if type(value) is not bool:
                raise RuntimeError("FullMDP opportunity flag differs")
            counts[name + "_rows"] += int(value)
    return [
        {**_identity_payload(identity), **grouped[identity]}
        for identity in sorted(grouped)
    ]


def _increment_histogram(histogram: dict[str, int], value: int) -> None:
    key = str(value)
    histogram[key] = histogram.get(key, 0) + 1


def _shot_strata(
    rows: tuple[object, ...], *, lifecycle_flags: tuple[str, ...]
) -> list[dict]:
    """Keep exact lifecycle/bit denominators without serializing every shot."""

    grouped: dict[tuple[int, int, str, bool], dict] = {}
    for row in rows:
        identity = _action_identity(row)
        counts = grouped.setdefault(
            identity,
            {
                "shot_rows": 0,
                "lifecycle_flag_counts": {name: 0 for name in lifecycle_flags},
                "r03_valid_bits_counts": {},
                "physical_valid_bits_counts": {},
                "r06_valid_bits_counts": {},
                "r06_outcome_code_counts": {},
                "r06_predicate_bits_counts": {},
                "motion_close_reason_counts": {},
            },
        )
        evidence = getattr(row, "evidence", None)
        bits = getattr(evidence, "lifecycle_bits", None)
        if type(bits) is not int:
            raise RuntimeError("FullMDP shot lifecycle evidence differs")
        counts["shot_rows"] += 1
        for ordinal, name in enumerate(lifecycle_flags):
            counts["lifecycle_flag_counts"][name] += int(
                bool(bits & (1 << ordinal))
            )
        for field in (
            "r03_valid_bits",
            "physical_valid_bits",
            "r06_valid_bits",
            "r06_outcome_code",
            "r06_predicate_bits",
        ):
            value = getattr(evidence, field, None)
            if type(value) is not int:
                raise RuntimeError("FullMDP shot evidence bit field differs")
            _increment_histogram(counts[field + "_counts"], value)
        close_reason = getattr(row, "motion_close_reason", None)
        if type(close_reason) is not int:
            raise RuntimeError("FullMDP shot close reason differs")
        _increment_histogram(counts["motion_close_reason_counts"], close_reason)
    return [
        {**_identity_payload(identity), **grouped[identity]}
        for identity in sorted(grouped)
    ]


def _bounded_rows(
    rows: tuple[object, ...], *, limit: int, projector: Callable[[object], dict]
) -> dict:
    sample = [projector(row) for row in rows[:limit]]
    return {
        "row_count": len(rows),
        "sample_limit": limit,
        "sample_rows": sample,
        "dropped_row_count": len(rows) - len(sample),
    }


def _telemetry(summary: object, runtime_module: object) -> dict:
    """Project one exact owner-produced summary without re-owning its facts."""

    summary_type = runtime_module.ActionEpochPpoBoundarySummary
    frontier_type = runtime_module.EpochDrainFrontier
    if type(summary) is not summary_type or type(summary.frontier) is not frontier_type:
        raise RuntimeError("FullMDP owner returned a foreign summary")
    frontier = summary.frontier
    drain = runtime_module.drain_v2
    lifecycle_flags = drain.SHOT_LIFECYCLE_FLAGS
    if type(lifecycle_flags) is not tuple:
        raise RuntimeError("FullMDP lifecycle ABI differs")
    rewards = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_rewards"
    )
    reset_rows = tuple(summary.terminal_resets)
    reset_counts = {
        "terminal_reset_reason_time_out_count": 0,
        "terminal_reset_reason_base_fell_tilt_count": 0,
        "terminal_reset_reason_base_too_low_count": 0,
        "terminal_reset_reason_joint_qdes_forbidden_count": 0,
        "terminal_reset_reason_robot_hit_table_count": 0,
    }
    for reset in reset_rows:
        row = asdict(reset)
        for name, bit in zip(reset_counts, (1, 2, 4, 8, 16)):
            reset_counts[name] += int(bool(row["reason_bits"] & bit))
    settlement = summary.settlement
    commits = summary.reveal_commit
    lifecycle = summary.lifecycle
    faults = summary.owner_faults
    continuation = summary.continuation
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "kind": TELEMETRY_KIND,
        "diagnostic_unauthorized": True,
        "ppo_update": frontier.update_index,
        "completed_environment_steps": frontier.completed_environment_steps,
        "epoch_operation_sequence": frontier.operation_sequence,
        "epoch_drain_sequence": frontier.drain_sequence,
        "epoch_commit_start": frontier.start_commit,
        "epoch_commit_end": frontier.end_commit,
        "shot_slot_capacity": frontier.shot_slot_capacity,
        "d05_transactions": settlement.transactions,
        "d05_scheduled_due_rows": (
            settlement.due_rows + frontier.due_terminal_overlap_rows
        ),
        "d05_due_rows": settlement.due_rows,
        "d05_due_terminal_overlap_rows": (
            frontier.due_terminal_overlap_rows
        ),
        "d05_selected_rows": settlement.selected_rows,
        "d05_accepted_rows": settlement.accepted,
        "d05_censored_rows": settlement.censored,
        "d05_rejected_rows": settlement.rejected,
        "d05_deferred_rows": settlement.deferred,
        "d05_not_ready_rows": settlement.not_ready,
        "motion_committed_rows": commits.motion_committed_rows,
        "racket_committed_rows": commits.racket_committed_rows,
        "r05_committed_rows": commits.r05_committed_rows,
        "playback_started_rows": lifecycle.playback_started_rows,
        "closed_unplayed_rows": lifecycle.closed_unplayed_rows,
        "physical_launch_rows": lifecycle.physical_launch_rows,
        "outcome_settled_rows": lifecycle.outcome_settled_rows,
        "payment_recorded_rows": lifecycle.payment_recorded_rows,
        "retired_rows": lifecycle.retired_rows,
        "terminal_shot_rows": lifecycle.terminal_shot_rows,
        "attributed_fault_rows": faults.attributed_fault_rows,
        "active_before": continuation.active_before,
        "active_after": continuation.active_after,
        "awaiting_playback_after": continuation.awaiting_playback_after,
        "awaiting_outcome_after": continuation.awaiting_outcome_after,
        "awaiting_payment_after": continuation.awaiting_payment_after,
        "action_opportunity_strata": _opportunity_strata(
            summary.action_opportunities
        ),
        "completed_shot_strata": _shot_strata(
            summary.completed_shots, lifecycle_flags=lifecycle_flags
        ),
        "terminal_shot_strata": _shot_strata(
            summary.terminal_shots, lifecycle_flags=lifecycle_flags
        ),
        "completed_shot_diagnostic_sample": _bounded_rows(
            summary.completed_shots,
            limit=_SHOT_SAMPLE_LIMIT,
            projector=lambda row: _shot_row(
                row, lifecycle_flags=lifecycle_flags
            ),
        ),
        "terminal_shot_diagnostic_sample": _bounded_rows(
            summary.terminal_shots,
            limit=_SHOT_SAMPLE_LIMIT,
            projector=lambda row: _shot_row(
                row, lifecycle_flags=lifecycle_flags
            ),
        ),
        "terminal_reset_diagnostic_sample": _bounded_rows(
            reset_rows,
            limit=_RESET_SAMPLE_LIMIT,
            projector=asdict,
        ),
        "terminal_reset_rows": len(reset_rows),
        "milestone": summary.milestone.as_json(tuple(rewards.MANAGER_NAMES)),
        **reset_counts,
    }


class ActionBallFullMdpRsl3Adapter:
    """Bind one exact lean owner to one RSL-RL 3 optimizer call."""

    def __init__(self, *, env: object, owner: object, log_dir: str) -> None:
        runtime = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_runtime"
        )
        if type(owner) is not runtime.ActionBallFullMdpLeanRuntimeOwner:
            raise RuntimeError("single_action_lean requires the exact runtime owner")
        runtime_env = getattr(env, "unwrapped", env)
        lease = getattr(runtime_env, "action_ball_full_mdp_runtime_lease", None)
        getter = getattr(runtime_env, "action_ball_full_mdp_ppo_drain_owner", None)
        if (
            lease is None
            or owner.full_mdp_runtime_env is not runtime_env
            or owner.full_mdp_runtime_lease is not lease
            or not callable(getter)
            or getattr(getter, "__self__", None) is not runtime_env
            or getter(lease) is not owner
            or owner.epoch_owner.num_envs != runtime_env.num_envs
            or owner.epoch_owner.shot_slot_capacity != 1
        ):
            raise RuntimeError("single_action_lean runtime graph identity differs")
        self._runtime = runtime
        self._owner = owner
        self._require_healthy = _bound(owner, "require_healthy")
        self._require_owner_idle = _bound(
            owner, "require_optimizer_boundary_idle"
        )
        self._prepare = _bound(owner, "prepare_pre_optimizer_ppo_boundary")
        self._mark_returned = _bound(owner, "mark_optimizer_returned")
        self._prepare_summary = _bound(owner, "prepare_post_update_summary")
        self._ack = _bound(owner, "acknowledge_post_update")
        self._latch = _bound(owner, "_record_durable_epoch_ack_span")
        self._poison = _bound(owner, "poison_optimizer_boundary")
        command_manager = getattr(runtime_env, "command_manager", None)
        get_command_term = getattr(command_manager, "get_term", None)
        command_term = (
            get_command_term("racket_target")
            if callable(get_command_term)
            else None
        )
        command_module = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.hope_commands"
        )
        command_type = vars(command_module).get("RacketTargetCommand")
        command_mismatches = []
        if not isinstance(command_type, type):
            command_mismatches.append("export_type")
        elif (
            command_type.__name__ != "RacketTargetCommand"
            or command_type.__module__ != command_module.__name__
        ):
            command_mismatches.append("export_identity")
        if command_manager is None:
            command_mismatches.append("command_manager")
        elif not callable(get_command_term):
            command_mismatches.append("get_term")
        if command_term is None:
            command_mismatches.append("racket_target")
        elif isinstance(command_type, type) and type(command_term) is not command_type:
            command_mismatches.append(
                "racket_target_type="
                + type(command_term).__module__
                + "."
                + type(command_term).__qualname__
            )
        if command_term is not getattr(owner, "_racket", None):
            command_mismatches.append("runtime_owner_racket_identity")
        deferred_predicate = None
        if command_term is not None:
            try:
                deferred_predicate = _bound(
                    command_term,
                    "_action_ball_full_mdp_deferred_exact_metrics_enabled",
                )
            except RuntimeError:
                command_mismatches.append("deferred_exact_metrics")
        if deferred_predicate is not None and deferred_predicate() is not True:
            disabled = {
                "full_mdp": getattr(command_term, "_action_ball_full_mdp_enabled", None),
                "target_mode": getattr(getattr(command_term, "cfg", None), "target_mode", None),
                "action_ball": getattr(command_term, "_action_ball_enabled", None),
                "task_first": getattr(command_term, "_task_first_enabled", None),
                "diagnostic": getattr(command_term, "_action_ball_diagnostic_unauthorized", None),
                "d05_owner": getattr(command_term, "_action_ball_full_mdp_device_r05_owner", None) is not None,
                "epoch_owner": getattr(command_term, "_action_ball_full_mdp_racket_epoch_owner", None) is not None,
                "adaptive_sigma": getattr(getattr(command_term, "cfg", None), "adaptive_sigma", None),
                "adaptive_sigma_monotonic": getattr(getattr(command_term, "cfg", None), "adaptive_sigma_monotonic", None),
                "adaptive_sigma_normal": getattr(getattr(command_term, "cfg", None), "adaptive_sigma_normal", None),
                "virtual_ball": getattr(getattr(command_term, "cfg", None), "virtual_ball", None),
                "vb_metrics_only": getattr(getattr(command_term, "cfg", None), "vb_metrics_only", None),
                "shadow_ball": getattr(getattr(command_term, "cfg", None), "shadow_ball", None),
                "physical_ball": getattr(getattr(command_term, "cfg", None), "physical_ball", None),
                "shadow_runtime": getattr(command_term, "_shadow", None) is not None,
                "physical_runtime": getattr(command_term, "_physical", None) is not None,
            }
            command_mismatches.append(
                "deferred_exact_metrics_disabled=" + repr(disabled)
            )
        if command_mismatches:
            raise RuntimeError(
                "single_action_lean requires the exact deferred-metric command "
                "producer: " + ",".join(command_mismatches)
            )
        self._command_term = command_term
        self._command_materialize = _bound(
            command_term,
            "materialize_action_ball_diagnostic_metrics_for_report",
        )
        self._command_assert_materialized = _bound(
            command_term,
            "assert_action_ball_diagnostic_metrics_materialized_for_report",
        )
        action_manager = getattr(runtime_env, "action_manager", None)
        get_action_term = getattr(action_manager, "get_term", None)
        action_term = get_action_term("joint_pos") if callable(get_action_term) else None
        action_module = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.hope_actions"
        )
        action_type = vars(action_module).get("ClampedJointPositionAction")
        producer_mismatches = []
        if not isinstance(action_type, type):
            producer_mismatches.append("export_type")
        elif (
            action_type.__name__ != "ClampedJointPositionAction"
            or action_type.__module__ != action_module.__name__
        ):
            producer_mismatches.append("export_identity")
        if action_manager is None:
            producer_mismatches.append("action_manager")
        elif not callable(get_action_term):
            producer_mismatches.append("get_term")
        if action_term is None:
            producer_mismatches.append("joint_pos")
        elif isinstance(action_type, type) and type(action_term) is not action_type:
            producer_mismatches.append(
                "joint_pos_type="
                + type(action_term).__module__
                + "."
                + type(action_term).__qualname__
            )
        if (
            action_term is not None
            and getattr(
                action_term, "_joint_safety_diagnostic_compact_evidence", None
            )
            is not True
        ):
            producer_mismatches.append("compact_evidence")
        if producer_mismatches:
            raise RuntimeError(
                "single_action_lean requires the compact joint-safety action "
                "producer: " + ",".join(producer_mismatches)
            )
        self._safety_term = action_term
        self._safety_expected_apply_calls = _exact_int(
            getattr(action_term, "_pre_apply_guard_decimation", None),
            name="bound guard decimation",
            minimum=1,
        )
        self._safety_prepare = _bound(
            action_term, "prepare_joint_safety_ledger_consume"
        )
        self._safety_assert_idle = _bound(
            action_term, "assert_joint_safety_ledger_consume_idle"
        )
        self._safety_ack = _bound(action_term, "acknowledge_joint_safety_ledger")
        self._wal = importlib.import_module(
            "whole_body_tracking.utils.action_ball_full_mdp_durable_wal"
        )
        self._rank = 0
        self._num_envs = int(owner.epoch_owner.num_envs)
        self._path, self._identity, self._segment = self._create_wal(log_dir)
        self._size = 0
        self._last_update = -1
        self._update_in_progress = False
        self._failure_reason = None
        self._safety_pending = None
        self._safety_last_completed_environment_steps = 0
        self._safety_last_consume_sequence = None
        self._safety_last_policy_step_sequence = None
        self._require_healthy()

    @staticmethod
    def _create_wal(log_dir: str) -> tuple[Path, tuple[int, int], str]:
        if type(log_dir) is not str or not log_dir:
            raise RuntimeError("single_action_lean requires an exact log directory")
        root = Path(log_dir)
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError("single_action_lean log directory differs")
        directory = root / "action_ball_epoch_durable_wal"
        directory.mkdir(mode=0o700)
        path = directory / "rank_0000.jsonl"
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError("single_action_lean WAL is not a unique regular file")
            os.fsync(fd)
        finally:
            os.close(fd)
        dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        # The child fsync persists the new file within ``directory``.  The
        # parent fsync separately persists the directory entry itself; without
        # it, a power loss can erase the entire otherwise-durable WAL tree.
        root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        root_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        root_fd = os.open(root, root_flags)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        identity = (info.st_dev, info.st_ino)
        return path, identity, f"{info.st_dev:x}:{info.st_ino:x}"

    def _append(self, line: bytes) -> tuple[int, int]:
        if type(line) is not bytes or line.count(b"\n") != 1 or not line.endswith(b"\n"):
            raise RuntimeError("single_action_lean WAL row is not one JSONL line")
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self._path, flags)
        start = self._size
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or (info.st_dev, info.st_ino) != self._identity
                or info.st_nlink != 1
                or info.st_size != start
            ):
                raise RuntimeError("single_action_lean WAL frontier changed")
            written = os.write(fd, line)
            if written != len(line):
                raise OSError("single_action_lean WAL append was short")
            os.fsync(fd)
            end = start + len(line)
            if os.fstat(fd).st_size != end:
                raise RuntimeError("single_action_lean WAL append frontier differs")
        except BaseException:
            try:
                if os.fstat(fd).st_size > start:
                    os.ftruncate(fd, start)
                    os.fsync(fd)
            except BaseException:
                pass
            raise
        finally:
            os.close(fd)
        self._size = end
        return start, end

    def update(
        self,
        algorithm_update: Callable[[], object],
        *,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        if (
            not callable(algorithm_update)
            or type(update_index) is not int
            or update_index != self._last_update + 1
            or type(completed_environment_steps) is not int
            or completed_environment_steps <= 0
        ):
            raise RuntimeError("single_action_lean optimizer chronology differs")
        if self._update_in_progress:
            raise RuntimeError(
                "single_action_lean optimizer boundary is already active"
            )
        if self._failure_reason is not None:
            raise RuntimeError(self._failure_reason)
        self._update_in_progress = True
        boundary = None
        try:
            self._require_healthy()
            delta_steps = (
                completed_environment_steps
                - self._safety_last_completed_environment_steps
            )
            if delta_steps <= 0 or delta_steps % self._num_envs != 0:
                raise RuntimeError(
                    "single_action_lean joint-safety rollout span differs"
                )
            rollout_steps = delta_steps // self._num_envs
            expected_metric_rows = (
                (rollout_steps + 1,)
                if self._last_update == -1
                else (rollout_steps,)
            )
            boundary = self._prepare(
                update_index=update_index,
                completed_environment_steps=completed_environment_steps,
            )
            # Only the exact FullMDP metric rows are deferred from H/H+1
            # command computes.  Drain that chronological device tape once,
            # after the owner has opened its valid optimizer transaction and
            # before any optimizer or destructive ledger operation.  A
            # materialize/assert failure therefore poisons this exact active
            # boundary and can never be retried against partially replayed
            # Python EMA state.
            self._command_materialize(
                expected_full_mdp_exact_row_counts=expected_metric_rows
            )
            self._command_assert_materialized()
            if self._safety_pending is not None:
                raise RuntimeError(
                    "single_action_lean joint-safety evidence is already frozen"
                )
            safety_token, safety_snapshot = self._safety_prepare()
            if type(safety_token) is not tuple:
                raise RuntimeError(
                    "single_action_lean joint-safety prepare token differs"
                )
            safety_pending = {
                "term": self._safety_term,
                "token": safety_token,
                "validated": None,
            }
            self._safety_pending = safety_pending
            safety_pending["validated"] = _validate_compact_joint_safety(
                safety_snapshot,
                expected_num_envs=self._num_envs,
                expected_policy_steps=delta_steps // self._num_envs,
                expected_apply_calls=self._safety_expected_apply_calls,
                previous_consume_sequence=self._safety_last_consume_sequence,
                previous_policy_step_sequence=(
                    self._safety_last_policy_step_sequence
                ),
            )
            result = algorithm_update()
            self._mark_returned(boundary, update_index=update_index)
            summary = self._prepare_summary(boundary, update_index=update_index)
            record = _telemetry(summary, self._runtime)
            validated = safety_pending["validated"]
            record["joint_safety"] = {
                "event": "hope_joint_safety_diagnostic_compact_update",
                "schema_version": 1,
                "status": "diagnostic_compact_prepared_before_optimizer",
                "ppo_update": update_index,
                "consume_sequence": validated["consume_sequence"],
                "num_envs": self._num_envs,
                "policy_step_count": delta_steps // self._num_envs,
                "first_policy_step_sequence": validated[
                    "first_policy_step_sequence"
                ],
                "last_policy_step_sequence": validated[
                    "last_policy_step_sequence"
                ],
                "counter_totals": validated["counter_totals"],
                "minimum_hard_gap_rad": validated["minimum_hard_gap_rad"],
                "terminal_archive_count": 0,
                "identity_bound_policy_step_count": 0,
                "formal_authority": False,
            }
            canonical, pending_line = self._wal.encode_pending(
                segment_id=self._segment, rank=self._rank, telemetry=record
            )
            safety_receipt = {
                **record["joint_safety"],
                "status": (
                    "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
                ),
                "completed_environment_steps": record[
                    "completed_environment_steps"
                ],
                "epoch_operation_sequence": record[
                    "epoch_operation_sequence"
                ],
                "epoch_drain_sequence": record["epoch_drain_sequence"],
                "epoch_commit_start": record["epoch_commit_start"],
                "epoch_commit_end": record["epoch_commit_end"],
            }
            safety_marker = (
                "HOPE_JOINT_SAFETY_UPDATE_JSON="
                + json.dumps(
                    safety_receipt, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
            marker = (
                "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON="
                + canonical.decode("utf-8")
                + "\n"
            )
            pending_start, pending_end = self._append(pending_line)
            self._ack(boundary, summary, update_index=update_index)
            if (
                safety_pending is not self._safety_pending
                or safety_pending["term"] is not self._safety_term
            ):
                raise RuntimeError(
                    "single_action_lean joint-safety prepared owner changed"
                )
            ack_line = self._wal.encode_epoch_ack(
                pending_line=pending_line,
                pending_byte_start=pending_start,
                pending_byte_end=pending_end,
            )
            ack_start, ack_end = self._append(ack_line)
            self._latch(
                summary,
                update_index=update_index,
                segment_id=self._segment,
                rank=self._rank,
                pending_byte_start=pending_start,
                pending_byte_end=pending_end,
                ack_byte_start=ack_start,
                ack_byte_end=ack_end,
            )
            # The action term is the existing typed compact-evidence owner.  Its
            # destructive clear stays last in the durable owner transaction, so
            # optimizer, PENDING, owner-ACK, EPOCH_ACK or latch failure leaves
            # the exact generation frozen and retry-forbidden.
            self._safety_ack(safety_pending["token"])
            self._safety_last_completed_environment_steps = (
                completed_environment_steps
            )
            self._safety_last_consume_sequence = validated["consume_sequence"]
            self._safety_last_policy_step_sequence = validated[
                "last_policy_step_sequence"
            ]
            self._safety_pending = None
            self._last_update = update_index
        except BaseException as exc:
            reason = (
                "single_action_lean optimizer boundary failed; retry forbidden: "
                f"{type(exc).__module__}.{type(exc).__qualname__}"
            )
            self._failure_reason = reason
            try:
                self._poison(boundary, update_index=update_index, reason=reason)
            except BaseException:
                pass
            self._update_in_progress = False
            raise RuntimeError(reason) from exc
        self._update_in_progress = False
        _emit_non_authoritative_stdout_marker(
            safety_marker,
            marker_name="HOPE_JOINT_SAFETY_UPDATE_JSON",
            durable_wal_authoritative=False,
        )
        _emit_non_authoritative_stdout_marker(
            marker,
            marker_name="HOPE_ACTION_EPOCH_UPDATE_ACK_JSON",
            durable_wal_authoritative=True,
        )
        return result

    def assert_snapshot_boundary_clean(self) -> None:
        """Reject a snapshot with active/poisoned or pending update state."""

        if self._update_in_progress:
            raise RuntimeError(
                "single_action_lean snapshot crossed an active optimizer boundary"
            )
        if self._failure_reason is not None:
            raise RuntimeError(self._failure_reason)
        self._require_owner_idle()
        if self._safety_pending is not None:
            raise RuntimeError(
                "single_action_lean snapshot crossed an active optimizer boundary"
            )
        self._safety_assert_idle()
        # This assertion covers the exact FullMDP metric-row tape only.  It
        # does not claim that every command-side D2H in the process is batched.
        self._command_assert_materialized()


class ActionBallFullMdpRsl3Runner(OnPolicyRunner):
    """Unmodified RSL-RL 3 loop with one exact optimizer-boundary adapter."""

    def __init__(
        self,
        env: object,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: object = None,
        *,
        training_contract_schema_version: int | None = None,
        training_contract_sha256: str | None = None,
        training_contract_lineage_exact: bool = False,
        training_launch_claim_sha256: str | None = None,
        require_exact_resume_state: bool = False,
        action_ball_r10_checkpoint_adapter: object = None,
        action_ball_r10_cold_restore_capsule: object = None,
        action_ball_full_mdp_runtime_owner: object = None,
        action_ball_full_mdp_run_mode: object = None,
    ) -> None:
        if (
            action_ball_full_mdp_run_mode != RUN_MODE
            or action_ball_full_mdp_runtime_owner is None
            or action_ball_r10_checkpoint_adapter is not None
            or action_ball_r10_cold_restore_capsule is not None
            or training_contract_lineage_exact is not False
            or training_launch_claim_sha256 is not None
            or require_exact_resume_state is not False
        ):
            raise RuntimeError("FullMDP RSL3 runner accepts only fresh single_action_lean")
        contract_schema, contract_sha256 = _validate_training_contract_identity(
            training_contract_schema_version,
            training_contract_sha256,
        )
        runtime_env = getattr(env, "unwrapped", env)
        observation_identity = _validate_observation_identity(runtime_env)
        self.training_contract_schema_version = contract_schema
        self.training_contract_sha256 = contract_sha256
        self._full_mdp_observation_identity = observation_identity
        super().__init__(env, train_cfg, log_dir, device)
        if self.is_distributed:
            raise RuntimeError("single_action_lean RSL3 runner is single-process only")
        self.registry_name = registry_name
        self.empirical_normalization = bool(train_cfg.get("empirical_normalization"))
        if log_dir is None:
            raise RuntimeError("single_action_lean RSL3 runner requires log_dir")
        self._full_mdp_adapter = ActionBallFullMdpRsl3Adapter(
            env=env, owner=action_ball_full_mdp_runtime_owner, log_dir=log_dir
        )
        original_update = self.alg.update
        if not callable(original_update) or getattr(original_update, "__self__", None) is not self.alg:
            raise RuntimeError("RSL3 PPO update is not the exact bound algorithm method")

        def update_with_full_mdp_boundary():
            update_index = self._full_mdp_adapter._last_update + 1
            return self._full_mdp_adapter.update(
                original_update,
                update_index=update_index,
                completed_environment_steps=(
                    (update_index + 1) * int(self.env.num_envs) * int(self.num_steps_per_env)
                ),
            )

        self.alg.update = update_with_full_mdp_boundary
        self._full_mdp_update_profiler = None
        profile_raw = os.environ.get(
            "HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES"
        )
        if profile_raw not in (None, "", "0"):
            profiler_module = importlib.import_module(
                "whole_body_tracking.utils."
                "action_ball_full_mdp_update_profiler"
            )
            requested_updates = (
                profiler_module.parse_full_mdp_profile_updates(os.environ)
            )
            self._full_mdp_update_profiler = (
                profiler_module.install_full_mdp_update_profiler(
                    self.env,
                    requested_updates=requested_updates,
                    emit_line=lambda line: print(line, flush=True),
                )
            )

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Emit bounded real-FullMDP attribution after upstream logging."""

        super().log(locs, width=width, pad=pad)
        profiler = self._full_mdp_update_profiler
        if profiler is None:
            return
        profiler.emit_update(
            update=int(locs["it"]),
            collection_time_s=locs["collection_time"],
            learning_time_s=locs["learn_time"],
            expected_env_step_calls=int(self.num_steps_per_env),
        )
        if profiler.closed:
            self._full_mdp_update_profiler = None

    @staticmethod
    def _normalizer_aliases(role: str) -> tuple[str, ...]:
        if role == "actor":
            return ("actor_obs_normalizer",)
        if role == "critic":
            return ("critic_obs_normalizer",)
        raise ValueError("normalizer role must be actor or critic")

    def _resolve_runtime_normalizer(self, role: str):
        aliases = self._normalizer_aliases(role)
        policy = self.alg.policy
        name = aliases[0]
        value = getattr(policy, name, None)
        return name, value, aliases

    def save(self, path: str, infos: dict | None = None) -> None:
        """Persist policy/optimizer bytes without claiming environment restore."""

        if infos is not None:
            raise RuntimeError(
                "single_action_lean diagnostic snapshot forbids caller infos"
            )
        self._full_mdp_adapter.assert_snapshot_boundary_clean()
        requested = Path(path)
        requested_stem = requested.stem
        if (
            not requested.is_absolute()
            or requested.suffix != ".pt"
            or not requested_stem.startswith("model_")
            or not requested_stem[6:].isdigit()
        ):
            raise RuntimeError("single_action_lean diagnostic snapshot path differs")
        learning_iteration = int(requested_stem[6:])
        snapshot = requested.with_name(
            requested_stem + ".diagnostic_nonresumable.pt"
        )
        contract_schema, contract_sha256 = _validate_training_contract_identity(
            getattr(self, "training_contract_schema_version", None),
            getattr(self, "training_contract_sha256", None),
        )
        observation_identity = getattr(
            self, "_full_mdp_observation_identity", None
        )
        if type(observation_identity) is not dict:
            raise RuntimeError(
                "single_action_lean diagnostic snapshot observation identity differs"
            )
        snapshot_identity = _validate_snapshot_identity_infos(
            {
                "action_ball_full_mdp_snapshot_kind": SNAPSHOT_PAYLOAD_KIND,
                **observation_identity,
                "training_contract_schema_version": contract_schema,
                "training_contract_sha256": contract_sha256,
                "diagnostic_unauthorized": True,
                "checkpoint_authority": False,
                "resume_authority": False,
            }
        )
        super().save(str(snapshot), infos=snapshot_identity)
        receipt = _write_snapshot_receipt(
            snapshot,
            learning_iteration=learning_iteration,
            required_infos=snapshot_identity,
        )
        print(
            "[ActionBallFullMdpRsl3Runner] wrote non-resumable diagnostic snapshot: "
            f"{snapshot} receipt={receipt}",
            flush=True,
        )

    def load(self, path: str, load_optimizer: bool = True, map_location=None):
        raise RuntimeError("single_action_lean forbids checkpoint load/resume")
