#!/usr/bin/env python3
"""Produce signed, source-derived ActionBall stage evidence.

This program has three deliberately narrow entry points:

``mint-authority``
    Derive an Ed25519 public key from an externally provisioned PKCS8 private
    key and publish the exact authority document consumed by
    ``launch_action_ball_curriculum.py``.  The private key must be a stable,
    owner-only, non-symlink regular file.  The authority binds a clean exact
    Git commit and the committed launcher bytes.

``attest-prelaunch``
    Re-open the formal MuJoCo fitted-ball and Isaac table-smoke receipts and
    derive every per-action gate from their detailed evidence.  There is no
    command-line field for a PASS boolean, count, time, speed, or aggregate.

``attest-stage``
    Re-open the launch claim, supervisor terminal receipt, trainer artifacts,
    checkpoint, Reward activation evidence, and every V4 request/evidence/ACK
    transcript.  Aggregate metrics are recomputed from raw attempt rows.  The
    signed receipt has the exact schema consumed by the launcher.

The producer is intentionally fail-closed.  In particular, a canary/long
stage cannot be signed without one accepted formal 320-canary + 960-heldout
pair per action, zero infrastructure/unsafe outcomes, a PASS Reward audit,
and a finite structurally exact-resumable checkpoint.  A caller cannot turn a
missing upstream artifact into a boolean command-line assertion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as _datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_action_ball_curriculum.py"
)
INBOX_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_evaluation_inbox.py"
)
AUDIT_REWARD_SOURCE = (
    "hope_training/whole_body_tracking/scripts/audit_reward_run.py"
)
TABLE_SMOKE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py"
)
RUNTIME_BOOTSTRAP_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_runtime_bootstrap.py"
)
EXACT_RESUME_VERIFIER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_exact_resume_verifier.py"
)
EXACT_RESUME_FACTORY_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_frozen_eval_sidecar.py"
)
STAGE_EVIDENCE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/action_ball_stage_evidence.py"
)
ACTION_SET_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_action_set_contract.py"
)
ACTION_ORDER = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
ARM_KEYS = (
    "time_to_contact_lower",
    "time_to_contact_upper",
    "contact_x_lower",
    "contact_x_upper",
    "contact_y_lower",
    "contact_y_upper",
    "contact_z_lower",
    "contact_z_upper",
    "incoming_speed_lower",
    "incoming_speed_upper",
    "spin_magnitude_lower",
    "spin_magnitude_upper",
    "base_spawn_x_lower",
    "base_spawn_x_upper",
    "base_spawn_y_lower",
    "base_spawn_y_upper",
    "base_travel_x_lower",
    "base_travel_x_upper",
    "base_travel_y_lower",
    "base_travel_y_upper",
    "landing_aim_x_lower",
    "landing_aim_x_upper",
    "landing_aim_y_lower",
    "landing_aim_y_upper",
    "incoming_direction_u_neg",
    "incoming_direction_u_pos",
    "incoming_direction_v_neg",
    "incoming_direction_v_pos",
    "spin_direction_u_neg",
    "spin_direction_u_pos",
    "spin_direction_v_neg",
    "spin_direction_v_pos",
)
NO_MOVE_ARMS = tuple(
    arm for arm in ARM_KEYS if not arm.startswith("base_travel_")
)
ACTION_BALL_LEDGER_NAMES = (
    "P",
    "A",
    "I",
    "S",
    "C",
    "L",
    "F",
    "U_table",
    "U_fall",
    "U_collision",
    "U_joint_qdes",
    "U_joint_actual",
    "X",
)
CURRICULUM_TOP_KEYS = (
    "schema_version",
    "contract_sha256",
    "profile_order",
    "arm_catalog",
    "arm_catalog_sha256",
    "scheduler_config",
    "scheduler_contract_sha256",
    "sampler_sha256",
    "solver_sha256",
    "policy_contract_sha256",
    "config",
    "evaluator_authority_contract_sha256",
    "evaluator_authority_state_owner_sha256",
    "evaluator_authority_state",
    "evaluator_authority_state_sha256",
    "drain_reset_authority_contract_sha256",
    "drain_reset_launch_receipt_sha256",
    "drain_reset_authority_state_owner_sha256",
    "drain_reset_authority_state",
    "drain_reset_authority_state_sha256",
    "next_barrier_serial",
    "issued_global_pre_reset_barriers",
    "progress",
    "state_sha256",
)
CURRICULUM_PROGRESS_KEYS = (
    "key",
    "phase",
    "arm_frontier_indices",
    "arm_status",
    "arm_probe_indices",
    "arm_epochs",
    "selected_arm_key",
    "selection_round",
    "last_selected_round",
    "center_epoch",
    "joint_epoch",
    "joint_probe_index",
    "joint_rho_index",
    "center_failures",
    "domain_release_epoch",
    "pending_canary_window_sha256",
    "pending_release",
    "release_receipts",
    "formal_receipts",
    "scheduler_receipts",
    "event_hash_chain_sha256",
    "last_certified",
)
CURRICULUM_EVIDENCE_KEYS = (
    "schema_version",
    "key",
    "arm_catalog_sha256",
    "scheduler_contract_sha256",
    "sampler_sha256",
    "solver_sha256",
    "policy_contract_sha256",
    "policy_checkpoint_sha256",
    "policy_generation",
    "evidence_role",
    "domain_epoch",
    "stratum",
    "selected_arm_key",
    "selection_round",
    "arm_levels",
    "rho",
    "seed_block_start",
    "seed_block_end_exclusive",
    "sample_id_start",
    "sample_id_end_exclusive",
    "sample_receipt_root_sha256",
    "unique_birth_count",
    "birth_receipt_root_sha256",
    "seq",
    "window_id",
    "ledger",
)
CURRICULUM_LEDGER_KEYS = (
    "P",
    "A",
    "I",
    "S",
    "C",
    "L",
    "F",
    "U_table",
    "U_fall",
    "U_collision",
    "U_joint_qdes",
    "U_joint_actual",
    "X",
    "NB",
    "NB_F",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CHECKPOINT_RE = re.compile(r"^model_([0-9]+)\.pt$")
RSL_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
MAX_JSON_BYTES = 512 * 1024 * 1024


class EvidenceError(RuntimeError):
    """The evidence producer cannot prove a requested PASS."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EvidenceError("value is not finite canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("duplicate JSON key is forbidden: {!r}".format(key))
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise EvidenceError("non-finite JSON constant is forbidden: {!r}".format(token))


def _strict_json_bytes(raw: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except EvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("{} is not strict UTF-8 JSON".format(label)) from exc
    if type(value) is not dict:
        raise EvidenceError("{} must contain one JSON object".format(label))
    _canonical_bytes(value)
    return value


def _count_nonfinite_tree(value: Any) -> int:
    if type(value) is float:
        return 0 if math.isfinite(value) else 1
    if type(value) is dict:
        return sum(
            _count_nonfinite_tree(key) + _count_nonfinite_tree(item)
            for key, item in value.items()
        )
    if type(value) in (list, tuple):
        return sum(_count_nonfinite_tree(item) for item in value)
    return 0


def _exact_dict(value: Any, keys: Iterable[str], label: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise EvidenceError("{} must be a plain JSON object".format(label))
    wanted = set(keys)
    actual = set(value)
    if actual != wanted:
        raise EvidenceError(
            "{} has invalid keys (missing={}, unknown={})".format(
                label, sorted(wanted - actual), sorted(actual - wanted)
            )
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise EvidenceError("{} must be one lowercase SHA-256".format(label))
    return value


def _commit(value: Any, label: str) -> str:
    if type(value) is not str or COMMIT_RE.fullmatch(value) is None:
        raise EvidenceError("{} must be one full lowercase Git commit".format(label))
    return value


def _plain_int(
    value: Any, label: str, minimum: int = 0, maximum: Optional[int] = None
) -> int:
    if type(value) is not int or value < minimum:
        raise EvidenceError("{} must be a plain integer >= {}".format(label, minimum))
    if maximum is not None and value > maximum:
        raise EvidenceError("{} must be <= {}".format(label, maximum))
    return value


def _validate_numpy_rng_state(value: Any) -> None:
    row = _exact_dict(
        value,
        (
            "schema_version",
            "bit_generator",
            "state_uint32",
            "position",
            "has_gauss",
            "cached_gaussian",
        ),
        "checkpoint NumPy RNG state",
    )
    values = row["state_uint32"]
    if (
        row["schema_version"] != 1
        or row["bit_generator"] != "MT19937"
        or type(values) is not list
        or len(values) != 624
        or any(
            type(item) is not int or item < 0 or item > 0xFFFFFFFF
            for item in values
        )
        or type(row["position"]) is not int
        or not 0 <= row["position"] <= 624
        or type(row["has_gauss"]) is not int
        or row["has_gauss"] not in (0, 1)
        or type(row["cached_gaussian"]) not in (int, float)
        or not math.isfinite(float(row["cached_gaussian"]))
    ):
        raise EvidenceError(
            "checkpoint NumPy RNG state is not safe schema 1"
        )


def _finite(
    value: Any,
    label: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if type(value) not in (int, float):
        raise EvidenceError("{} must be a plain finite number".format(label))
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EvidenceError("{} must be finite".format(label))
    if minimum is not None and numeric < minimum:
        raise EvidenceError("{} must be >= {}".format(label, minimum))
    if maximum is not None and numeric > maximum:
        raise EvidenceError("{} must be <= {}".format(label, maximum))
    return numeric


def _normalized_absolute(value: Any, label: str, must_exist: bool = True) -> Path:
    if type(value) is not str or not value:
        raise EvidenceError("{} must be a non-empty absolute path".format(label))
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise EvidenceError("{} must be an absolute normalized path".format(label))
    if must_exist and not os.path.lexists(path):
        raise EvidenceError("{} does not exist: {}".format(label, path))
    return path


def _assert_real_directory_chain(path: Path, label: str) -> None:
    """Reject a directory whose existing path contains a symlink component."""

    candidate = path
    chain: List[Path] = []
    while candidate != candidate.parent:
        chain.append(candidate)
        candidate = candidate.parent
    chain.append(candidate)
    for component in reversed(chain):
        try:
            info = component.lstat()
        except OSError as exc:
            raise EvidenceError(
                "{} directory component is missing: {}".format(label, component)
            ) from exc
        if not stat.S_ISDIR(info.st_mode):
            raise EvidenceError(
                "{} directory chain contains a symlink/non-directory: {}".format(
                    label, component
                )
            )


def _snapshot_file(
    path_value: Any,
    label: str,
    max_bytes: int = MAX_JSON_BYTES,
) -> Dict[str, Any]:
    path = (
        path_value
        if isinstance(path_value, Path)
        else _normalized_absolute(path_value, label)
    )
    try:
        before = path.lstat()
    except OSError as exc:
        raise EvidenceError("{} cannot be inspected: {}".format(label, path)) from exc
    if not stat.S_ISREG(before.st_mode):
        raise EvidenceError("{} must be a regular non-symlink file".format(label))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise EvidenceError("{} cannot be opened safely".format(label)) from exc
    try:
        opened = os.fstat(fd)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
        )
        if opened_identity != identity:
            raise EvidenceError("{} changed before safe open".format(label))
        if opened.st_size > max_bytes:
            raise EvidenceError("{} exceeds byte budget".format(label))
        chunks: List[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise EvidenceError("{} was truncated while reading".format(label))
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise EvidenceError("{} grew while reading".format(label))
        after = os.fstat(fd)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        if after_identity != identity:
            raise EvidenceError("{} changed while reading".format(label))
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    return {
        "path": path,
        "raw": raw,
        "sha256": _sha256_bytes(raw),
        "stat": before,
    }


def _snapshot_json(path_value: Any, label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    snapshot = _snapshot_file(path_value, label)
    return _strict_json_bytes(snapshot["raw"], label), snapshot


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_exclusive_json(path_value: Any, value: Mapping[str, Any]) -> str:
    path = _normalized_absolute(path_value, "output path", must_exist=False)
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except OSError as exc:
        raise EvidenceError("output parent does not exist: {}".format(parent)) from exc
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise EvidenceError("output parent must be a real directory")
    _assert_real_directory_chain(parent, "output parent")
    if os.path.lexists(path):
        raise EvidenceError("refusing to clobber output: {}".format(path))
    payload = _canonical_bytes(dict(value)) + b"\n"
    temporary = parent / ".{}.{}.tmp".format(path.name, secrets.token_hex(12))
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(str(temporary), flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    try:
        os.link(str(temporary), str(path), follow_symlinks=False)
        _fsync_directory(parent)
    except FileExistsError as exc:
        raise EvidenceError("refusing to clobber output: {}".format(path)) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(parent)
    reopened = _snapshot_file(path, "published output")
    if reopened["raw"] != payload:
        raise EvidenceError("published output readback differs from signed bytes")
    return reopened["sha256"]


def _git(checkout: Path, *args: str, binary: bool = False) -> Any:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        stderr = result.stderr if not binary else result.stderr.decode(
            "utf-8", errors="replace"
        )
        raise EvidenceError(
            "git {} failed: {}".format(" ".join(args), stderr.strip()[-4000:])
        )
    return result.stdout if binary else result.stdout.strip()


def _verify_clean_checkout(checkout_value: Any, commit_value: Any) -> Tuple[Path, str]:
    checkout = _normalized_absolute(checkout_value, "checkout")
    try:
        resolved = checkout.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError("checkout cannot be resolved") from exc
    if resolved != checkout or not checkout.is_dir():
        raise EvidenceError("checkout must be one real absolute directory")
    top = Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve()
    if top != checkout:
        raise EvidenceError("checkout must be the exact Git worktree root")
    commit = _commit(commit_value, "source commit")
    if _git(checkout, "rev-parse", "--verify", "HEAD") != commit:
        raise EvidenceError("checkout HEAD differs from source commit")
    dirty = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise EvidenceError(
            "checkout is not exact clean; first entry={!r}".format(
                dirty.splitlines()[0]
            )
        )
    return checkout, commit


def _committed_file(
    checkout: Path, commit: str, relative_value: Any, label: str
) -> Dict[str, Any]:
    if type(relative_value) is not str or not relative_value:
        raise EvidenceError("{} path must be repo-relative text".format(label))
    pure = PurePosixPath(relative_value)
    if (
        pure.is_absolute()
        or "\\" in relative_value
        or ":" in relative_value
        or pure.as_posix() != relative_value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise EvidenceError("{} path must be normalized POSIX relative".format(label))
    path = checkout.joinpath(*pure.parts)
    snapshot = _snapshot_file(path, label)
    committed = _git(
        checkout, "cat-file", "blob", "{}:{}".format(commit, relative_value), binary=True
    )
    if snapshot["raw"] != committed:
        raise EvidenceError("{} worktree bytes differ from commit".format(label))
    snapshot["relative"] = relative_value
    return snapshot


def _committed_absolute_file(
    checkout: Path,
    source_commit: str,
    path_value: Any,
    label: str,
) -> Dict[str, Any]:
    path = (
        path_value
        if isinstance(path_value, Path)
        else _normalized_absolute(path_value, label)
    )
    try:
        relative = path.relative_to(checkout).as_posix()
    except ValueError as exc:
        raise EvidenceError("{} must be inside the exact checkout".format(label)) from exc
    return _committed_file(checkout, source_commit, relative, label)


def _load_action_set_contract(
    checkout: Path, source_commit: str, profile_id: str
) -> Dict[str, Any]:
    snapshot = _committed_file(
        checkout,
        source_commit,
        ACTION_SET_CONTRACT_SOURCE,
        "action-set contract source",
    )
    module_name = "_action_ball_stage_action_set_contract_{}".format(
        secrets.token_hex(6)
    )
    spec = importlib.util.spec_from_file_location(
        module_name, str(snapshot["path"])
    )
    if spec is None or spec.loader is None:
        raise EvidenceError("cannot load action-set contract source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.load_contract_from_source(
            snapshot["raw"], profile_id
        )
    except Exception as exc:
        raise EvidenceError(
            "code-owned action-set contract is invalid: {}".format(exc)
        ) from exc


def _load_private_key(path_value: Any) -> Tuple[Any, Dict[str, Any]]:
    snapshot = _snapshot_file(path_value, "Ed25519 PKCS8 private key", 64 * 1024)
    mode = stat.S_IMODE(snapshot["stat"].st_mode)
    if mode != 0o600:
        raise EvidenceError("Ed25519 private key mode must be exactly 0600")
    if snapshot["stat"].st_uid != os.geteuid():
        raise EvidenceError("Ed25519 private key must be owned by the current uid")
    if snapshot["stat"].st_nlink != 1:
        raise EvidenceError("Ed25519 private key must have exactly one hard link")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:
        raise EvidenceError("cryptography Ed25519 support is unavailable") from exc
    raw = snapshot["raw"]
    loaders = (
        serialization.load_pem_private_key,
        serialization.load_der_private_key,
    )
    key = None
    for loader in loaders:
        try:
            key = loader(raw, password=None)
            break
        except (TypeError, ValueError):
            continue
    if type(key) is not Ed25519PrivateKey:
        raise EvidenceError("private key must be an unencrypted Ed25519 PKCS8 key")
    return key, snapshot


def _public_key_bytes(private_key: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def mint_authority(
    *,
    checkout: str,
    source_commit: str,
    private_key_path: str,
    evaluator_id: str,
    output_path: str,
) -> Dict[str, Any]:
    root, commit = _verify_clean_checkout(checkout, source_commit)
    if (
        type(evaluator_id) is not str
        or not evaluator_id
        or "\n" in evaluator_id
        or "\x00" in evaluator_id
    ):
        raise EvidenceError("evaluator_id must be non-empty single-line text")
    launcher = _committed_file(root, commit, LAUNCHER_SOURCE, "launcher source")
    private_key, _key_snapshot = _load_private_key(private_key_path)
    unsigned = {
        "schema_version": 1,
        "kind": "action_ball_frozen_stage_evaluator_authority",
        "evaluator_id": evaluator_id,
        "public_key_ed25519_hex": _public_key_bytes(private_key).hex(),
        "evaluator_source_path": LAUNCHER_SOURCE,
        "evaluator_source_sha256": launcher["sha256"],
    }
    document = {**unsigned, "canonical_sha256": canonical_sha256(unsigned)}
    _publish_exclusive_json(output_path, document)
    return document


def _validate_authority(
    authority_path: Any,
    *,
    checkout: Path,
    source_commit: str,
    private_key: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    document, snapshot = _snapshot_json(authority_path, "stage evaluator authority")
    committed = _committed_absolute_file(
        checkout,
        source_commit,
        snapshot["path"],
        "stage evaluator authority",
    )
    if committed["sha256"] != snapshot["sha256"]:
        raise EvidenceError(
            "stage evaluator authority bytes differ from the exact commit"
        )
    row = _exact_dict(
        document,
        (
            "schema_version",
            "kind",
            "evaluator_id",
            "public_key_ed25519_hex",
            "evaluator_source_path",
            "evaluator_source_sha256",
            "canonical_sha256",
        ),
        "stage evaluator authority",
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != "action_ball_frozen_stage_evaluator_authority"
        or type(row["evaluator_id"]) is not str
        or not row["evaluator_id"]
        or row["evaluator_source_path"] != LAUNCHER_SOURCE
    ):
        raise EvidenceError("stage evaluator authority identity is invalid")
    public_hex = _sha256(row["public_key_ed25519_hex"], "authority public key")
    source_sha = _sha256(row["evaluator_source_sha256"], "authority source SHA")
    launcher = _committed_file(
        checkout, source_commit, LAUNCHER_SOURCE, "authority launcher source"
    )
    if launcher["sha256"] != source_sha:
        raise EvidenceError("authority launcher bytes differ from exact commit")
    unsigned = dict(row)
    del unsigned["canonical_sha256"]
    if (
        _sha256(row["canonical_sha256"], "authority canonical SHA")
        != canonical_sha256(unsigned)
    ):
        raise EvidenceError("authority canonical SHA is false")
    public = bytes.fromhex(public_hex)
    if private_key is not None and _public_key_bytes(private_key) != public:
        raise EvidenceError("private key does not match stage evaluator authority")
    return row, snapshot, public


def _width_pair(
    initial: Any,
    maximum: Any,
    *,
    label: str,
    unit: str,
) -> Dict[str, Any]:
    lo = _finite(initial, label + " initial", minimum=0.0)
    hi = _finite(maximum, label + " max", minimum=0.0)
    if hi < lo:
        raise EvidenceError("{} max width is below its initial width".format(label))
    return {"initial": lo, "maximum": hi, "unit": unit}


def _vector_component(value: Any, index: int, length: int, label: str) -> float:
    if type(value) is not list or len(value) != length:
        raise EvidenceError("{} must be a {}-vector".format(label, length))
    return _finite(value[index], "{}[{}]".format(label, index), minimum=0.0)


def _manifest_arm_width_spec(
    *,
    manifest: Mapping[str, Any],
    action: Mapping[str, Any],
    action_id: str,
) -> Dict[str, Dict[str, Any]]:
    profile = action.get("ball_profile")
    landing = manifest.get("landing_aim")
    if type(profile) is not dict or type(landing) is not dict:
        raise EvidenceError(
            "manifest action {} lacks ball_profile/landing_aim".format(action_id)
        )
    result: Dict[str, Dict[str, Any]] = {}

    def scalar(
        arm: str,
        initial_key: str,
        maximum_key: str,
        unit: str,
    ) -> None:
        result[arm] = _width_pair(
            profile.get(initial_key),
            profile.get(maximum_key),
            label="{}.{}".format(action_id, arm),
            unit=unit,
        )

    scalar(
        "time_to_contact_lower",
        "time_to_contact_std_lower_initial_s",
        "time_to_contact_std_lower_max_s",
        "s",
    )
    scalar(
        "time_to_contact_upper",
        "time_to_contact_std_upper_initial_s",
        "time_to_contact_std_upper_max_s",
        "s",
    )
    scalar(
        "incoming_speed_lower",
        "incoming_speed_std_lower_initial_mps",
        "incoming_speed_std_lower_max_mps",
        "m/s",
    )
    scalar(
        "incoming_speed_upper",
        "incoming_speed_std_upper_initial_mps",
        "incoming_speed_std_upper_max_mps",
        "m/s",
    )
    scalar(
        "spin_magnitude_lower",
        "spin_magnitude_std_lower_initial_radps",
        "spin_magnitude_std_lower_max_radps",
        "rad/s",
    )
    scalar(
        "spin_magnitude_upper",
        "spin_magnitude_std_upper_initial_radps",
        "spin_magnitude_std_upper_max_radps",
        "rad/s",
    )

    vector_families = (
        (
            "contact",
            3,
            "contact_offset_std_{}_initial_m",
            "contact_offset_std_{}_max_m",
            ("x", "y", "z"),
            "m",
        ),
        (
            "base_spawn",
            2,
            "base_spawn_std_{}_initial_m",
            "base_spawn_std_{}_max_m",
            ("x", "y"),
            "m",
        ),
        (
            "base_travel",
            2,
            "base_travel_std_{}_initial_m",
            "base_travel_std_{}_max_m",
            ("x", "y"),
            "m",
        ),
    )
    for prefix, length, initial_template, maximum_template, axes, unit in (
        vector_families
    ):
        for side in ("lower", "upper"):
            initial = profile.get(initial_template.format(side))
            maximum = profile.get(maximum_template.format(side))
            for index, axis in enumerate(axes):
                arm = "{}_{}_{}".format(prefix, axis, side)
                result[arm] = _width_pair(
                    _vector_component(
                        initial,
                        index,
                        length,
                        "{}.{}".format(action_id, initial_template.format(side)),
                    ),
                    _vector_component(
                        maximum,
                        index,
                        length,
                        "{}.{}".format(action_id, maximum_template.format(side)),
                    ),
                    label="{}.{}".format(action_id, arm),
                    unit=unit,
                )

    for side in ("lower", "upper"):
        initial = landing.get("std_{}_initial_m".format(side))
        maximum = landing.get("std_{}_max_m".format(side))
        for index, axis in enumerate(("x", "y")):
            arm = "landing_aim_{}_{}".format(axis, side)
            result[arm] = _width_pair(
                _vector_component(
                    initial,
                    index,
                    2,
                    "landing_aim.std_{}_initial_m".format(side),
                ),
                _vector_component(
                    maximum,
                    index,
                    2,
                    "landing_aim.std_{}_max_m".format(side),
                ),
                label="{}.{}".format(action_id, arm),
                unit="m",
            )

    for prefix in ("incoming_direction", "spin_direction"):
        for tangent in ("u", "v"):
            for side in ("neg", "pos"):
                arm = "{}_{}_{}".format(prefix, tangent, side)
                result[arm] = _width_pair(
                    profile.get(
                        "{}_tangent_{}_{}_initial_deg".format(
                            prefix, tangent, side
                        )
                    ),
                    profile.get(
                        "{}_tangent_{}_{}_max_deg".format(
                            prefix, tangent, side
                        )
                    ),
                    label="{}.{}".format(action_id, arm),
                    unit="deg",
                )
    if set(result) != set(ARM_KEYS):
        raise EvidenceError(
            "manifest arm-width mapping is incomplete for {}".format(action_id)
        )
    return {arm: result[arm] for arm in ARM_KEYS}


def _manifest_bindings(
    *,
    manifest_path: Any,
    checkout: Path,
    source_commit: str,
    action_set_contract: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    manifest_absolute = (
        manifest_path
        if isinstance(manifest_path, Path)
        else _normalized_absolute(manifest_path, "ActionBall manifest")
    )
    try:
        manifest_relative = manifest_absolute.relative_to(checkout).as_posix()
    except ValueError as exc:
        raise EvidenceError("ActionBall manifest must be inside the checkout") from exc
    snapshot = _committed_file(
        checkout,
        source_commit,
        manifest_relative,
        "ActionBall manifest",
    )
    document = _strict_json_bytes(snapshot["raw"], "ActionBall manifest")
    order = action_set_contract["ordered_action_ids"]
    uids = action_set_contract["ordered_action_uids"]
    if (
        snapshot["relative"] != action_set_contract["manifest_path"]
        or snapshot["sha256"] != action_set_contract["manifest_sha256"]
        or document.get("schema_version") != 3
        or document.get("mobility_mode")
        != action_set_contract["mobility_mode"]
        or document.get("action_order") != order
        or type(document.get("prototype")) is not dict
        or document["prototype"].get("scope")
        != action_set_contract["scope"]
    ):
        raise EvidenceError(
            "manifest differs from the code-owned action-set contract"
        )
    actions = document.get("actions")
    if type(actions) is not list or len(actions) != len(order):
        raise EvidenceError("manifest must contain exact contracted N rows")
    result: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for index, (raw, action_id, action_uid) in enumerate(
        zip(actions, order, uids)
    ):
        if type(raw) is not dict or raw.get("action_id") != action_id:
            raise EvidenceError("manifest action order is invalid at {}".format(index))
        uid = _plain_int(
            raw.get("action_uid"),
            "manifest.actions[{}].action_uid".format(index),
            minimum=1,
            maximum=(1 << 53) - 1,
        )
        if uid != action_uid:
            raise EvidenceError(
                "manifest action UID differs from contract at {}".format(index)
            )
        if uid in seen:
            raise EvidenceError("manifest action_uid values are not unique")
        seen.add(uid)
        motion_sha = _sha256(
            raw.get("motion_sha256"),
            "manifest.actions[{}].motion_sha256".format(index),
        )
        motion = _committed_file(
            checkout,
            source_commit,
            raw.get("motion_path"),
            "manifest motion {}".format(action_id),
        )
        if motion["sha256"] != motion_sha:
            raise EvidenceError("manifest motion SHA is false for {}".format(action_id))
        result.append(
            {
                "motion_id": action_id,
                "action_uid": uid,
                "motion_sha256": motion_sha,
                "scope": action_set_contract["scope"],
                "mobility_mode": action_set_contract["mobility_mode"],
                "arm_width_spec": _manifest_arm_width_spec(
                    manifest=document,
                    action=raw,
                    action_id=action_id,
                ),
            }
        )
    return result, document, snapshot


_FITTED_TOP_KEYS = {
    "schema_version",
    "gate",
    "contact_authority",
    "native_ball_contact_enabled",
    "selector_executed",
    "ball_to_task_solver_executed",
    "ball_to_task_solver_executed_by_gate",
    "pre_registered_ball_to_task_solver_receipt_consumed",
    "solver_execution_receipt_authority",
    "analytic_return_scorer_executed",
    "expected_actions",
    "expected_action_order",
    "preflight",
    "authorization",
    "runtime_code_identity",
    "formal_gate_executed",
    "runtime_environment",
    "runtime_input_snapshot",
    "runtime_code_identity_post_runtime",
    "runtime_code_identity_final",
    "status",
    "verdict",
    "manifest_id",
    "action_order",
    "base_mujoco_portable_identity_sha256",
    "base_mujoco_verification_receipt_sha256",
    "compiler_mesh_assets",
    "scene_contracts",
    "venue",
    "contact_model",
    "actions",
    "receipt_payload_sha256",
}
_FITTED_ACTION_KEYS = {
    "action_id",
    "action_uid",
    "motion_path",
    "motion_sha256",
    "launch",
    "face_geometry",
    "t_hit_s",
    "t_cycle_s",
    "reference_racket_site_speed_mps",
    "dt_results",
    "convergence",
    "physical_task_binding",
    "shared_ready_joint_linf_rad",
    "recovery_joint_linf_rad",
    "video",
    "verdict",
    "failure_reasons",
}
_TABLE_TOP_KEYS = {
    "schema_version",
    "receipt_class",
    "verdict",
    "task_id",
    "with_table",
    "scope",
    "mobility_mode",
    "action_set_contract",
    "manifest",
    "profile_contract",
    "ordered_action_ids",
    "motion_sha256",
    "runtime_contract",
    "actions",
    "authorization",
    "non_claims",
    "receipt_payload_sha256",
}
_TABLE_ACTION_KEYS = {
    "motion_id",
    "action_uid",
    "scope",
    "robot_body_contract_count",
    "motion_sha256",
    "complete_cycle",
    "isaac_filtered_contact_pass",
    "table_contact_count",
    "fall_count",
    "hard_limit_count",
    "unsafe_count",
    "verdict",
}
_ACTION_BALL_SOLVER_SOURCE_NAMES = {
    "continuous_questions.py",
    "hope_commands.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "virtual_ball.py",
}
_PHYSICAL_TASK_CASE_ROLES = (
    "center_positive_seed_0",
    "center_positive_seed_1",
    "support_positive",
    "negative_t_hit_offset",
    "negative_face_sign",
    "negative_ball_state_mismatch",
)
_PHYSICAL_TASK_NEGATIVE_REASON = {
    "negative_t_hit_offset": "teacher_task_contact_time_mismatch",
    "negative_face_sign": "teacher_task_face_sign_mismatch",
    "negative_ball_state_mismatch": "teacher_task_ball_state_mismatch",
}
_PHYSICAL_TASK_BINDING_KEYS = {
    "ball_profile_sha256",
    "solver_profile_sha256",
    "physics_profile_sha256",
    "solver_source_sha256",
    "solver_execution_receipt",
    "cases_sha256",
    "case_order",
    "cases",
}
_PHYSICAL_TASK_SUMMARY_CASE_KEYS = {
    "case_id",
    "case_role",
    "sample_seed",
    "expected_physical_verdict",
    "expected_failure_reason",
    "ball_proposal_sha256",
    "task_payload_sha256",
    "solved_task_geometry_sha256",
    "case_binding_sha256",
    "solver_execution_identity",
    "task_timing",
    "task_geometry",
    "dt_results",
    "convergence",
    "control",
    "observed_physical_verdict",
    "control_verdict",
    "failure_reasons",
}
_RACKET_GEOMETRY_CONTRACT_KEYS = {
    "schema_version",
    "semantics",
    "ball_target_point",
    "site_target_mapping",
    "face_velocity_mapping",
    "source_path",
    "source_sha256",
    "geometry_source_sha256",
}
_TABLE_RUNTIME_KEYS = {
    "source_commit_sha",
    "isaac_version",
    "python_executable",
    "runtime_source",
    "gpu_identity",
    "physics_steps",
    "real_physx_contacts",
    "full_action_ball_assembly",
    "all_five_table_sources_with_explicit_robot_body_filters",
    "action_robot_body_contract_rows",
    "all_five_obstacles",
    "all_four_substeps",
    "positive_control_pass",
    "negative_control_pass",
    "zero_reset_leakage",
}


def _validate_payload_seal(document: Dict[str, Any], label: str) -> None:
    declared = _sha256(document.get("receipt_payload_sha256"), label + " payload SHA")
    unsigned = dict(document)
    del unsigned["receipt_payload_sha256"]
    if canonical_sha256(unsigned) != declared:
        raise EvidenceError("{} payload seal is false".format(label))


def _validate_false_authorization(value: Any, label: str) -> None:
    expected = {
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }
    if value != expected:
        raise EvidenceError("{} self-authorization is forbidden".format(label))


def _validate_fitted_physical_task_binding(
    value: Any,
    *,
    manifest_action: Mapping[str, Any],
    action: Mapping[str, Any],
    action_index: int,
    manifest: Mapping[str, Any],
    profile_pins: Mapping[str, Any],
    checkout: Path,
    source_commit: str,
) -> Dict[str, Any]:
    """Close the preregistered ball->task solver and six physical controls."""

    label = "fitted-ball actions[{}].physical_task_binding".format(
        action_index
    )
    summary = _exact_dict(value, _PHYSICAL_TASK_BINDING_KEYS, label)
    raw = _exact_dict(
        manifest_action.get("physical_task_binding"),
        (
            "schema_version",
            "authority",
            "action_id",
            "action_uid",
            "motion_sha256",
            "ball_profile_sha256",
            "solver_profile_sha256",
            "physics_profile_sha256",
            "solver_implementation_source_sha256",
            "solver_execution_receipt_path",
            "solver_execution_receipt_sha256",
            "solver_execution_identity",
            "solver_execution_identity_sha256",
            "selector_executed",
            "action_identity_frozen",
            "cases",
            "cases_sha256",
        ),
        label + " manifest binding",
    )
    ball_profile_sha = canonical_sha256(manifest_action["ball_profile"])
    solver_sha = _sha256(
        manifest.get("solver_profile_sha256"),
        label + " manifest solver profile SHA",
    )
    physics_sha = _sha256(
        manifest.get("physics_profile_sha256"),
        label + " manifest physics profile SHA",
    )
    source_map = profile_pins.get(
        "solver_implementation_source_sha256"
    )
    if (
        type(source_map) is not dict
        or set(source_map) != _ACTION_BALL_SOLVER_SOURCE_NAMES
    ):
        raise EvidenceError(
            "{} profile pins lack exact five solver source identities".format(
                label
            )
        )
    normalized_sources = {
        name: _sha256(digest, "{} source {}".format(label, name))
        for name, digest in source_map.items()
    }
    source_base = (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp"
    )
    for name in sorted(_ACTION_BALL_SOLVER_SOURCE_NAMES):
        committed = _committed_file(
            checkout,
            source_commit,
            "{}/{}".format(source_base, name),
            "{} solver source {}".format(label, name),
        )
        if committed["sha256"] != normalized_sources[name]:
            raise EvidenceError(
                "{} solver source {} differs from exact commit".format(
                    label, name
                )
            )
    if (
        raw["schema_version"] != 1
        or raw["authority"]
        != "pre_registered_frozen_action_ball_solver_receipt_v1"
        or raw["action_id"] != action["action_id"]
        or raw["action_uid"] != action["action_uid"]
        or raw["motion_sha256"] != action["motion_sha256"]
        or raw["ball_profile_sha256"] != ball_profile_sha
        or raw["solver_profile_sha256"] != solver_sha
        or raw["physics_profile_sha256"] != physics_sha
        or raw["solver_implementation_source_sha256"]
        != normalized_sources
        or raw["selector_executed"] is not False
        or raw["action_identity_frozen"] is not True
        or summary["ball_profile_sha256"] != ball_profile_sha
        or summary["solver_profile_sha256"] != solver_sha
        or summary["physics_profile_sha256"] != physics_sha
        or summary["solver_source_sha256"] != normalized_sources
    ):
        raise EvidenceError(
            "{} action/profile/frozen-solver identity does not close".format(
                label
            )
        )

    execution_identity = _exact_dict(
        raw["solver_execution_identity"],
        (
            "artifact_type",
            "execution_id",
            "executed_before_gate",
            "solver_replayed_exact",
            "selector_executed",
            "action_identity_frozen",
            "action_switching_allowed",
            "hardware_authorized",
        ),
        label + " solver execution identity",
    )
    execution_identity_sha = _sha256(
        raw["solver_execution_identity_sha256"],
        label + " solver execution identity SHA",
    )
    if (
        execution_identity["artifact_type"]
        != "frozen_ball_to_task_solver_execution_v1"
        or type(execution_identity["execution_id"]) is not str
        or not execution_identity["execution_id"]
        or execution_identity["executed_before_gate"] is not True
        or execution_identity["solver_replayed_exact"] is not True
        or execution_identity["selector_executed"] is not False
        or execution_identity["action_identity_frozen"] is not True
        or execution_identity["action_switching_allowed"] is not False
        or execution_identity["hardware_authorized"] is not False
        or canonical_sha256(execution_identity) != execution_identity_sha
    ):
        raise EvidenceError(
            "{} solver execution identity is not frozen simulation-only".format(
                label
            )
        )

    raw_cases = raw["cases"]
    summary_cases = summary["cases"]
    if (
        type(raw_cases) is not list
        or type(summary_cases) is not list
        or len(raw_cases) != len(_PHYSICAL_TASK_CASE_ROLES)
        or len(summary_cases) != len(raw_cases)
        or [
            item.get("case_role") if type(item) is dict else None
            for item in raw_cases
        ]
        != list(_PHYSICAL_TASK_CASE_ROLES)
        or summary["case_order"] != list(_PHYSICAL_TASK_CASE_ROLES)
        or canonical_sha256(raw_cases) != raw["cases_sha256"]
        or raw["cases_sha256"] != summary["cases_sha256"]
    ):
        raise EvidenceError(
            "{} does not contain the exact sealed 3-positive/3-negative "
            "case order".format(label)
        )

    receipt_binding = _exact_dict(
        summary["solver_execution_receipt"],
        ("path", "sha256", "receipt_payload_sha256"),
        label + " external solver receipt binding",
    )
    if (
        receipt_binding["path"]
        != raw["solver_execution_receipt_path"]
        or receipt_binding["sha256"]
        != raw["solver_execution_receipt_sha256"]
    ):
        raise EvidenceError(
            "{} external solver receipt differs from manifest".format(label)
        )
    solver_receipt_file = _committed_file(
        checkout,
        source_commit,
        receipt_binding["path"],
        label + " external solver execution receipt",
    )
    if solver_receipt_file["sha256"] != _sha256(
        receipt_binding["sha256"], label + " external receipt SHA"
    ):
        raise EvidenceError(
            "{} external solver receipt bytes differ from exact commit".format(
                label
            )
        )
    solver_receipt = _strict_json_bytes(
        solver_receipt_file["raw"], label + " external solver receipt"
    )
    solver_receipt = _exact_dict(
        solver_receipt,
        (
            "schema_version",
            "artifact_type",
            "producer",
            "action_identity",
            "profile_identity",
            "solver_execution_identity",
            "cases",
            "receipt_payload_sha256",
        ),
        label + " external solver receipt",
    )
    receipt_unsigned = dict(solver_receipt)
    external_payload_sha = _sha256(
        receipt_unsigned.pop("receipt_payload_sha256"),
        label + " external receipt payload SHA",
    )
    producer = _exact_dict(
        solver_receipt["producer"],
        (
            "source_path",
            "source_sha256",
            "runtime_receipt_type",
            "exact_solver_replay_required",
            "selector_executed",
            "hardware_authorized",
        ),
        label + " external receipt producer",
    )
    if (
        solver_receipt["schema_version"] != 1
        or solver_receipt["artifact_type"]
        != "frozen_action_ball_solver_execution_receipt_v1"
        or canonical_sha256(receipt_unsigned) != external_payload_sha
        or receipt_binding["receipt_payload_sha256"]
        != external_payload_sha
        or producer["source_path"]
        != "{}/hope_commands.py".format(source_base)
        or producer["source_sha256"]
        != normalized_sources["hope_commands.py"]
        or producer["runtime_receipt_type"] != "ActionBallTaskReceipt"
        or producer["exact_solver_replay_required"] is not True
        or producer["selector_executed"] is not False
        or producer["hardware_authorized"] is not False
        or solver_receipt["action_identity"]
        != {
            "action_id": action["action_id"],
            "action_uid": action["action_uid"],
            "motion_sha256": action["motion_sha256"],
        }
        or solver_receipt["solver_execution_identity"]
        != execution_identity
        or solver_receipt["cases"] != raw_cases
    ):
        raise EvidenceError(
            "{} external solver receipt provenance/cases do not close".format(
                label
            )
        )
    profile_identity = _exact_dict(
        solver_receipt["profile_identity"],
        (
            "ball_profile_sha256",
            "solver_profile_sha256",
            "physics_profile_sha256",
            "solver_implementation_source_sha256",
            "geometry_source_sha256",
        ),
        label + " external receipt profile",
    )
    if (
        profile_identity["ball_profile_sha256"] != ball_profile_sha
        or profile_identity["solver_profile_sha256"] != solver_sha
        or profile_identity["physics_profile_sha256"] != physics_sha
        or profile_identity["solver_implementation_source_sha256"]
        != normalized_sources
    ):
        raise EvidenceError(
            "{} external receipt profile identity drifted".format(label)
        )

    positive_count = 0
    negative_count = 0
    for case_index, (role, raw_case_value, replay_value) in enumerate(
        zip(_PHYSICAL_TASK_CASE_ROLES, raw_cases, summary_cases)
    ):
        case_label = "{}.cases[{}]".format(label, case_index)
        raw_case = _exact_dict(
            raw_case_value,
            (
                "case_id",
                "case_role",
                "sample_seed",
                "expected_physical_verdict",
                "expected_failure_reason",
                "ball_proposal",
                "ball_proposal_sha256",
                "task_payload",
                "task_payload_sha256",
                "fault_injection",
                "case_binding_sha256",
            ),
            case_label + " manifest case",
        )
        replay = _exact_dict(
            replay_value, _PHYSICAL_TASK_SUMMARY_CASE_KEYS, case_label
        )
        positive = role in _PHYSICAL_TASK_CASE_ROLES[:3]
        expected_verdict = "PASS" if positive else "FAIL"
        expected_reason = (
            None if positive else _PHYSICAL_TASK_NEGATIVE_REASON[role]
        )
        sample_seed = _plain_int(
            raw_case["sample_seed"], case_label + " sample_seed"
        )
        proposal_sha = _sha256(
            raw_case["ball_proposal_sha256"],
            case_label + " ball proposal SHA",
        )
        task_sha = _sha256(
            raw_case["task_payload_sha256"],
            case_label + " task payload SHA",
        )
        proposal = raw_case["ball_proposal"]
        task = raw_case["task_payload"]
        if (
            type(raw_case["case_id"]) is not str
            or not raw_case["case_id"]
            or raw_case["case_role"] != role
            or raw_case["expected_physical_verdict"] != expected_verdict
            or raw_case["expected_failure_reason"] != expected_reason
            or type(proposal) is not dict
            or canonical_sha256(proposal) != proposal_sha
            or type(task) is not dict
            or canonical_sha256(task) != task_sha
            or proposal.get("action_id") != action["action_id"]
            or proposal.get("action_uid") != action["action_uid"]
            or proposal.get("motion_sha256") != action["motion_sha256"]
            or proposal.get("sample_seed") != sample_seed
            or task.get("action_id") != action["action_id"]
            or task.get("action_uid") != action["action_uid"]
            or task.get("motion_sha256") != action["motion_sha256"]
            or task.get("ball_proposal_sha256") != proposal_sha
            or task.get("solver_profile_sha256") != solver_sha
            or task.get("physics_profile_sha256") != physics_sha
            or replay["case_id"] != raw_case["case_id"]
            or replay["case_role"] != role
            or replay["sample_seed"] != sample_seed
            or replay["expected_physical_verdict"] != expected_verdict
            or replay["expected_failure_reason"] != expected_reason
            or replay["ball_proposal_sha256"] != proposal_sha
            or replay["task_payload_sha256"] != task_sha
            or replay["case_binding_sha256"]
            != raw_case["case_binding_sha256"]
            or replay["solver_execution_identity"]
            != execution_identity
            or replay["observed_physical_verdict"] != expected_verdict
            or replay["control_verdict"] != "PASS"
            or replay["failure_reasons"] != []
        ):
            raise EvidenceError(
                "{} solver/task/case/control identity is false".format(
                    case_label
                )
            )
        control = _exact_dict(
            replay["control"],
            (
                "expected_physical_verdict",
                "expected_failure_reason",
                "observed_physical_verdict",
                "observed_failure_reason",
                "observed_dt_verdicts",
                "fault_application",
                "convergence_required",
                "convergence_pass",
                "control_verdict",
                "failure_reasons",
            ),
            case_label + " control",
        )
        if (
            control["expected_physical_verdict"] != expected_verdict
            or control["expected_failure_reason"] != expected_reason
            or control["observed_physical_verdict"] != expected_verdict
            or control["observed_failure_reason"] != expected_reason
            or control["observed_dt_verdicts"]
            != {"0.0010": expected_verdict, "0.0005": expected_verdict}
            or control["convergence_required"] is not positive
            or (
                positive
                and control["convergence_pass"] is not True
            )
            or (
                not positive
                and control["convergence_pass"] is not None
            )
            or control["control_verdict"] != "PASS"
            or control["failure_reasons"] != []
        ):
            raise EvidenceError(
                "{} positive/negative control ledger is false".format(
                    case_label
                )
            )
        dt_results = replay["dt_results"]
        if (
            type(dt_results) is not dict
            or set(dt_results) != {"0.0010", "0.0005"}
        ):
            raise EvidenceError(
                "{} lacks both physical replay timesteps".format(case_label)
            )
        for dt_name, dt_result in dt_results.items():
            if (
                type(dt_result) is not dict
                or dt_result.get("verdict") != expected_verdict
                or (
                    positive
                    and dt_result.get("failure_reasons") != []
                )
                or (
                    not positive
                    and (
                        type(dt_result.get("failure_reasons")) is not list
                        or not dt_result["failure_reasons"]
                    )
                )
                or dt_result.get("robot_obstacle_contacts") != []
                or dt_result.get("self_contacts") != []
                or dt_result.get(
                    "shadow_robot_obstacle_near_contacts"
                )
                != []
                or dt_result.get("shadow_self_near_contacts") != []
                or dt_result.get("joint_limit_violation") is not None
                or dt_result.get("fall") is not None
                or dt_result.get("event_order_violations") != []
                or dt_result.get("ball_forbidden_contacts") != []
                or dt_result.get("native_ball_contact_count") != 0
            ):
                raise EvidenceError(
                    "{} {} physical replay is unsafe/wrong control "
                    "verdict".format(case_label, dt_name)
                )
        if positive:
            positive_count += 1
        else:
            negative_count += 1
    if positive_count != 3 or negative_count != 3:
        raise EvidenceError(
            "{} does not prove exactly 3 positive and 3 negative cases".format(
                label
            )
        )
    if action["dt_results"] != summary_cases[0]["dt_results"]:
        raise EvidenceError(
            "{} top-level center replay is not the physical solver case".format(
                label
            )
        )
    return {
        "ball_profile_sha256": ball_profile_sha,
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "solver_execution_receipt_sha256": solver_receipt_file[
            "sha256"
        ],
        "cases_sha256": raw["cases_sha256"],
        "positive_case_count": positive_count,
        "negative_case_count": negative_count,
    }


def _derive_fitted_rows(
    document: Dict[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    profile_pins: Mapping[str, Any],
    checkout: Path,
    source_commit: str,
    action_set_contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    _exact_dict(document, _FITTED_TOP_KEYS, "fitted-ball receipt")
    _validate_payload_seal(document, "fitted-ball receipt")
    preflight = document["preflight"]
    if (
        document["schema_version"] != 1
        or document["gate"] != "mujoco_teacher_motion_fitted_ball_gate"
        or document["contact_authority"] != "venue_fitted_swept_selected_face_v2"
        or document["native_ball_contact_enabled"] is not False
        or document["selector_executed"] is not False
        or document["ball_to_task_solver_executed"] is not False
        or document["ball_to_task_solver_executed_by_gate"] is not False
        or document[
            "pre_registered_ball_to_task_solver_receipt_consumed"
        ]
        is not True
        or document["solver_execution_receipt_authority"]
        != "pre_registered_frozen_action_ball_solver_receipt_v1"
        or document["analytic_return_scorer_executed"] is not False
        or document["expected_actions"] != len(bindings)
        or document["expected_action_order"]
        != action_set_contract["ordered_action_ids"]
        or type(preflight) is not dict
        or preflight.get("status") != "PASS"
        or preflight.get("blockers") != []
        or document["formal_gate_executed"] is not True
        or document["status"] != "PASS"
        or document["verdict"] != "PASS"
        or document["action_order"]
        != action_set_contract["ordered_action_ids"]
    ):
        raise EvidenceError(
            "fitted-ball receipt is not an exact contracted formal PASS"
        )
    _validate_false_authorization(document["authorization"], "fitted-ball receipt")
    actions = document["actions"]
    manifest_actions = manifest.get("actions")
    if (
        type(actions) is not list
        or len(actions) != len(bindings)
        or type(manifest_actions) is not list
        or len(manifest_actions) != len(bindings)
        or canonical_sha256(profile_pins.get("solver_payload"))
        != manifest.get("solver_profile_sha256")
        or canonical_sha256(profile_pins.get("physics_payload"))
        != manifest.get("physics_profile_sha256")
        or profile_pins.get("solver_profile_sha256")
        != manifest.get("solver_profile_sha256")
        or profile_pins.get("physics_profile_sha256")
        != manifest.get("physics_profile_sha256")
    ):
        raise EvidenceError(
            "fitted-ball receipt lacks exact contracted action rows"
        )
    derived: List[Dict[str, Any]] = []
    for index, (raw, binding, manifest_action) in enumerate(
        zip(actions, bindings, manifest_actions)
    ):
        action = _exact_dict(
            raw, _FITTED_ACTION_KEYS, "fitted-ball actions[{}]".format(index)
        )
        if (
            action["action_id"] != binding["motion_id"]
            or action["action_uid"] != binding["action_uid"]
            or action["motion_sha256"] != binding["motion_sha256"]
            or action["verdict"] != "PASS"
            or action["failure_reasons"] != []
        ):
            raise EvidenceError("fitted-ball action identity/verdict drifted")
        physical_task = _validate_fitted_physical_task_binding(
            action["physical_task_binding"],
            manifest_action=manifest_action,
            action=action,
            action_index=index,
            manifest=manifest,
            profile_pins=profile_pins,
            checkout=checkout,
            source_commit=source_commit,
        )
        t_hit = _finite(action["t_hit_s"], "t_hit_s", minimum=0.0)
        t_cycle = _finite(action["t_cycle_s"], "t_cycle_s", minimum=0.0)
        speed = _finite(
            action["reference_racket_site_speed_mps"],
            "reference racket speed",
            minimum=0.0,
        )
        ready = _finite(
            action["shared_ready_joint_linf_rad"],
            "shared-ready error",
            minimum=0.0,
        )
        recovery = _finite(
            action["recovery_joint_linf_rad"],
            "recovery error",
            minimum=0.0,
        )
        if not (t_hit < t_cycle and speed > 0.0 and ready <= 1.0e-6 and recovery <= 1.0e-6):
            raise EvidenceError("fitted-ball timing/speed/recovery gate failed")
        dt_results = action["dt_results"]
        if type(dt_results) is not dict or set(dt_results) != {"0.0010", "0.0005"}:
            raise EvidenceError("fitted-ball action lacks both dt replays")
        total_steps = 0
        for dt_name, result in sorted(dt_results.items()):
            if type(result) is not dict:
                raise EvidenceError("fitted-ball dt result must be an object")
            window = result.get("simulation_window")
            mandatory = result.get("mandatory_gates")
            contacts = (
                result.get("robot_obstacle_contacts"),
                result.get("self_contacts"),
                result.get("shadow_robot_obstacle_near_contacts"),
                result.get("shadow_self_near_contacts"),
            )
            if (
                result.get("verdict") != "PASS"
                or result.get("failure_reasons") != []
                or result.get("joint_limit_violation") is not None
                or result.get("fall") is not None
                or any(value != [] for value in contacts)
                or type(window) is not dict
                or type(mandatory) is not dict
            ):
                raise EvidenceError("fitted-ball dt replay contains unsafe raw evidence")
            steps = _plain_int(
                window.get("physics_steps"),
                "fitted-ball {} physics_steps".format(dt_name),
                minimum=1,
            )
            executed = _finite(
                window.get("executed_end_time_s"),
                "fitted-ball executed end time",
                minimum=0.0,
            )
            required = _finite(
                window.get("required_ready_to_recovery_end_time_s"),
                "fitted-ball required recovery end time",
                minimum=0.0,
            )
            if (
                executed < required
                or mandatory.get(
                    "physical_ball_selected_face_return_and_first_landing"
                )
                is not True
                or mandatory.get(
                    "teacher_robot_and_racket_table_net_post_clearance"
                )
                is not True
            ):
                raise EvidenceError("fitted-ball physical return/recovery gate failed")
            total_steps += steps
        derived.append(
            {
                "action_id": binding["motion_id"],
                "action_uid": binding["action_uid"],
                "motion_sha256": binding["motion_sha256"],
                "t_hit_s": t_hit,
                "t_cycle_s": t_cycle,
                "physical_racket_site_speed_mps": speed,
                "fitted_physics_steps": total_steps,
                "physical_task_binding": physical_task,
            }
        )
    return derived


def _validate_table_rows(
    document: Dict[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    *,
    manifest_relative: str,
    manifest_document: Mapping[str, Any],
    profile_relative: str,
    profile_sha256: str,
    profile_document: Mapping[str, Any],
    checkout: Path,
    source_commit: str,
    action_set_contract: Mapping[str, Any],
) -> None:
    _exact_dict(document, _TABLE_TOP_KEYS, "Isaac table-smoke receipt")
    _validate_payload_seal(document, "Isaac table-smoke receipt")
    motions = [row["motion_sha256"] for row in bindings]
    action_ids = [row["motion_id"] for row in bindings]
    action_uids = [row["action_uid"] for row in bindings]
    receipt_action_set_contract = _exact_dict(
        document["action_set_contract"],
        action_set_contract.keys(),
        "Isaac table-smoke action-set contract",
    )
    if (
        document["schema_version"] != 3
        or document["receipt_class"]
        != "isaac_action_ball_table_filtered_smoke_v3"
        or document["verdict"] != "PASS"
        or document["task_id"] != "HOPE-PingPong-ActionBall-AgibotA3-v0"
        or document["with_table"] is not True
        or document["scope"] != action_set_contract["scope"]
        or document["mobility_mode"] != action_set_contract["mobility_mode"]
        or receipt_action_set_contract != dict(action_set_contract)
        or document["ordered_action_ids"]
        != action_set_contract["ordered_action_ids"]
        or action_set_contract.get("expected_n") != len(bindings)
        or action_set_contract.get("ordered_action_ids") != action_ids
        or action_set_contract.get("ordered_action_uids") != action_uids
        or document["motion_sha256"] != motions
    ):
        raise EvidenceError("Isaac table-smoke receipt identity is not exact")
    if document["non_claims"] != [
        "training_authorization",
        "deployment_authorization",
        "hardware_authorization",
    ]:
        raise EvidenceError("Isaac table-smoke non-claims are not exact")
    manifest = _exact_dict(
        document["manifest"], ("path", "sha256"), "table manifest binding"
    )
    if manifest != {
        "path": manifest_relative,
        "sha256": manifest_sha256,
    }:
        raise EvidenceError("Isaac table-smoke manifest binding drifted")
    profile_contract = _exact_dict(
        document["profile_contract"],
        (
            "profile_pins",
            "solver_profile_sha256",
            "physics_profile_sha256",
            "solver_implementation_sources",
            "racket_geometry_contract",
        ),
        "Isaac table-smoke profile contract",
    )
    profile_binding = _exact_dict(
        profile_contract["profile_pins"],
        ("path", "sha256"),
        "Isaac table-smoke profile pins",
    )
    solver_payload = profile_document.get("solver_payload")
    physics_payload = profile_document.get("physics_payload")
    source_map = profile_document.get(
        "solver_implementation_source_sha256"
    )
    solver_sha = canonical_sha256(solver_payload)
    physics_sha = canonical_sha256(physics_payload)
    if (
        profile_binding
        != {"path": profile_relative, "sha256": profile_sha256}
        or profile_contract["solver_profile_sha256"] != solver_sha
        or profile_contract["physics_profile_sha256"] != physics_sha
        or profile_document.get("solver_profile_sha256") != solver_sha
        or profile_document.get("physics_profile_sha256") != physics_sha
        or manifest_document.get("solver_profile_sha256") != solver_sha
        or manifest_document.get("physics_profile_sha256") != physics_sha
        or type(source_map) is not dict
        or set(source_map) != _ACTION_BALL_SOLVER_SOURCE_NAMES
        or (
            type(solver_payload) is not dict
            or solver_payload.get("implementation_source_sha256")
            != source_map
        )
    ):
        raise EvidenceError(
            "Isaac table-smoke solver/physics profile identity does not close"
        )
    source_rows = profile_contract["solver_implementation_sources"]
    expected_source_names = sorted(_ACTION_BALL_SOLVER_SOURCE_NAMES)
    if (
        type(source_rows) is not list
        or len(source_rows) != len(expected_source_names)
    ):
        raise EvidenceError(
            "Isaac table-smoke solver source closure is incomplete"
        )
    source_by_name: Dict[str, Dict[str, Any]] = {}
    source_base = (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp"
    )
    for index, (raw_source, expected_name) in enumerate(
        zip(source_rows, expected_source_names)
    ):
        source = _exact_dict(
            raw_source,
            ("name", "path", "sha256"),
            "Isaac table-smoke solver source[{}]".format(index),
        )
        expected_path = "{}/{}".format(source_base, expected_name)
        committed = _committed_file(
            checkout,
            source_commit,
            expected_path,
            "Isaac table-smoke solver source {}".format(expected_name),
        )
        if (
            source["name"] != expected_name
            or source["path"] != expected_path
            or source["sha256"] != source_map[expected_name]
            or source["sha256"] != committed["sha256"]
        ):
            raise EvidenceError(
                "Isaac table-smoke solver source order/hash changed"
            )
        source_by_name[expected_name] = dict(source)
    geometry = _exact_dict(
        profile_contract["racket_geometry_contract"],
        _RACKET_GEOMETRY_CONTRACT_KEYS,
        "Isaac table-smoke racket geometry contract",
    )
    manifest_geometry = manifest_document.get(
        "racket_geometry_contract"
    )
    geometry_source = source_by_name["racket_contact_geometry.py"]
    if (
        type(manifest_geometry) is not dict
        or geometry != manifest_geometry
        or geometry["source_path"] != geometry_source["path"]
        or geometry["source_sha256"] != geometry_source["sha256"]
        or geometry["geometry_source_sha256"]
        == geometry["source_sha256"]
    ):
        raise EvidenceError(
            "Isaac table-smoke physical racket geometry does not close"
        )
    _validate_false_authorization(document["authorization"], "Isaac table-smoke receipt")
    runtime = _exact_dict(
        document["runtime_contract"], _TABLE_RUNTIME_KEYS, "table runtime contract"
    )
    if _commit(runtime["source_commit_sha"], "table runtime source commit") != source_commit:
        raise EvidenceError("table runtime source commit differs from launch commit")
    for field in ("isaac_version", "python_executable"):
        if type(runtime[field]) is not str or not runtime[field]:
            raise EvidenceError("table runtime {} is missing".format(field))
    runtime_source = _exact_dict(
        runtime["runtime_source"],
        ("path", "sha256"),
        "table runtime source binding",
    )
    if runtime_source["path"] != TABLE_SMOKE_SOURCE:
        raise EvidenceError("table runtime source path is not the formal producer")
    committed_source = _committed_file(
        checkout,
        source_commit,
        TABLE_SMOKE_SOURCE,
        "table runtime source",
    )
    if (
        _sha256(runtime_source["sha256"], "table runtime source SHA")
        != committed_source["sha256"]
    ):
        raise EvidenceError("table runtime source bytes differ from exact commit")
    gpu = _exact_dict(
        runtime["gpu_identity"],
        (
            "physical_index",
            "logical_index",
            "cuda_visible_devices",
            "gpu_uuid",
            "gpu_name",
            "driver_version",
            "nvml_verified",
        ),
        "table runtime GPU identity",
    )
    physical_index = _plain_int(
        gpu["physical_index"], "table runtime physical GPU index"
    )
    if (
        gpu["logical_index"] != 0
        or gpu["cuda_visible_devices"] != str(physical_index)
        or type(gpu["gpu_uuid"]) is not str
        or not gpu["gpu_uuid"].startswith("GPU-")
        or type(gpu["gpu_name"]) is not str
        or not gpu["gpu_name"]
        or type(gpu["driver_version"]) is not str
        or not gpu["driver_version"]
        or gpu["nvml_verified"] is not True
    ):
        raise EvidenceError("table runtime GPU identity is not an NVML-bound slot")
    _plain_int(runtime["physics_steps"], "table runtime physics_steps", minimum=1)
    if (
        _plain_int(
            runtime["action_robot_body_contract_rows"],
            "table runtime action_robot_body_contract_rows",
            minimum=1,
        )
        != 32 * len(bindings)
    ):
        raise EvidenceError(
            "table runtime does not bind the exact 32 x N A3 Robot-body "
            "contract rows"
        )
    for field in (
        "real_physx_contacts",
        "full_action_ball_assembly",
        "all_five_table_sources_with_explicit_robot_body_filters",
        "all_five_obstacles",
        "all_four_substeps",
        "positive_control_pass",
        "negative_control_pass",
        "zero_reset_leakage",
    ):
        if runtime[field] is not True:
            raise EvidenceError("table runtime control {} did not pass".format(field))
    actions = document["actions"]
    if type(actions) is not list or len(actions) != len(bindings):
        raise EvidenceError(
            "table-smoke receipt lacks exact contracted action rows"
        )
    for index, (raw, binding) in enumerate(zip(actions, bindings)):
        action = _exact_dict(
            raw, _TABLE_ACTION_KEYS, "table-smoke actions[{}]".format(index)
        )
        expected = {
            "motion_id": binding["motion_id"],
            "action_uid": binding["action_uid"],
            "scope": action_set_contract["scope"],
            "robot_body_contract_count": 32,
            "motion_sha256": binding["motion_sha256"],
            "complete_cycle": True,
            "isaac_filtered_contact_pass": True,
            "table_contact_count": 0,
            "fall_count": 0,
            "hard_limit_count": 0,
            "unsafe_count": 0,
            "verdict": "PASS",
        }
        if action != expected:
            raise EvidenceError(
                "table-smoke action is partial/unsafe: {}".format(binding["motion_id"])
            )


def derive_prelaunch_payload(
    *,
    checkout: Path,
    source_commit: str,
    launch_profile: str,
    manifest_path: Any,
    profile_pins_path: Any,
    launch_trust_spec_path: Any,
    launch_trust_root_path: Any,
    fitted_gate_path: Any,
    table_smoke_path: Any,
    authority_path: Any,
    private_key: Any,
) -> Dict[str, Any]:
    action_set_contract = _load_action_set_contract(
        checkout, source_commit, launch_profile
    )
    bindings, manifest_document, manifest_snapshot = _manifest_bindings(
        manifest_path=manifest_path,
        checkout=checkout,
        source_commit=source_commit,
        action_set_contract=action_set_contract,
    )
    authority, authority_snapshot, _public = _validate_authority(
        authority_path,
        checkout=checkout,
        source_commit=source_commit,
        private_key=private_key,
    )
    del authority
    profile = _committed_absolute_file(
        checkout,
        source_commit,
        profile_pins_path,
        "fitted-ball profile pins",
    )
    profile_document = _strict_json_bytes(
        profile["raw"], "fitted-ball profile pins"
    )
    trust_spec = _committed_absolute_file(
        checkout,
        source_commit,
        launch_trust_spec_path,
        "launch trust spec",
    )
    trust_root = _committed_absolute_file(
        checkout,
        source_commit,
        launch_trust_root_path,
        "launch trust root",
    )
    fitted_document, fitted_snapshot = _snapshot_json(
        fitted_gate_path, "formal fitted-ball receipt"
    )
    table_document, table_snapshot = _snapshot_json(
        table_smoke_path, "formal Isaac table-smoke receipt"
    )
    fitted_rows = _derive_fitted_rows(
        fitted_document,
        bindings,
        manifest=manifest_document,
        profile_pins=profile_document,
        checkout=checkout,
        source_commit=source_commit,
        action_set_contract=action_set_contract,
    )
    _validate_table_rows(
        table_document,
        bindings,
        manifest_snapshot["sha256"],
        manifest_relative=manifest_snapshot["relative"],
        manifest_document=manifest_document,
        profile_relative=profile["relative"],
        profile_sha256=profile["sha256"],
        profile_document=profile_document,
        checkout=checkout,
        source_commit=source_commit,
        action_set_contract=action_set_contract,
    )
    per_action = []
    for row in fitted_rows:
        per_action.append(
            {
                "action_id": row["action_id"],
                "action_uid": row["action_uid"],
                "motion_sha256": row["motion_sha256"],
                "t_hit_s": row["t_hit_s"],
                "t_cycle_s": row["t_cycle_s"],
                "physical_racket_site_speed_mps": row[
                    "physical_racket_site_speed_mps"
                ],
                "all_body_table_pair_count": 32 * 5,
                "table_contact_count": 0,
                "fall_count": 0,
                "hard_limit_count": 0,
                "unsafe_count": 0,
                "t_hit_pass": True,
                "t_cycle_pass": True,
                "physical_racket_site_speed_pass": True,
                "shared_ready_recovery_pass": True,
                "recorded_incoming_ball_returned_to_table": True,
                "no_table_contact": True,
                "grounded_safety_pass": True,
                "hard_limit_pass": True,
                "isaac_filtered_contact_pass": True,
            }
        )
    return {
        "schema_version": 1,
        "kind": "action_ball_prelaunch_safety_attestation",
        "status": "passed",
        "source_commit_sha": source_commit,
        "launch_profile": launch_profile,
        "action_set_contract_sha256": action_set_contract[
            "contract_sha256"
        ],
        "ordered_action_ids": list(
            action_set_contract["ordered_action_ids"]
        ),
        "manifest_sha256": manifest_snapshot["sha256"],
        "profile_pins_sha256": profile["sha256"],
        "fitted_ball_launch_trust_spec_sha256": trust_spec["sha256"],
        "fitted_ball_launch_trust_root_sha256": trust_root["sha256"],
        "fitted_ball_gate_receipt_sha256": fitted_snapshot["sha256"],
        "isaac_table_smoke_receipt_sha256": table_snapshot["sha256"],
        "stage_evaluator_authority_sha256": authority_snapshot["sha256"],
        "per_action": per_action,
    }


def _signed_envelope(kind: str, payload: Dict[str, Any], private_key: Any) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "payload": payload,
        "signature_ed25519_hex": private_key.sign(_canonical_bytes(payload)).hex(),
    }


def attest_prelaunch(
    *,
    checkout: str,
    source_commit: str,
    launch_profile: str,
    manifest_path: str,
    profile_pins_path: str,
    launch_trust_spec_path: str,
    launch_trust_root_path: str,
    fitted_gate_path: str,
    table_smoke_path: str,
    authority_path: str,
    private_key_path: str,
    output_path: str,
) -> Dict[str, Any]:
    root, commit = _verify_clean_checkout(checkout, source_commit)
    private_key, _snapshot = _load_private_key(private_key_path)
    payload = derive_prelaunch_payload(
        checkout=root,
        source_commit=commit,
        launch_profile=launch_profile,
        manifest_path=manifest_path,
        profile_pins_path=profile_pins_path,
        launch_trust_spec_path=launch_trust_spec_path,
        launch_trust_root_path=launch_trust_root_path,
        fitted_gate_path=fitted_gate_path,
        table_smoke_path=table_smoke_path,
        authority_path=authority_path,
        private_key=private_key,
    )
    envelope = _signed_envelope(
        "action_ball_signed_prelaunch_safety_attestation",
        payload,
        private_key,
    )
    _publish_exclusive_json(output_path, envelope)
    return envelope


def _parse_arm(arm: Any) -> Tuple[str, str]:
    if type(arm) is not str or arm not in ARM_KEYS:
        raise EvidenceError("formal request selected_arm_key is missing or unknown")
    for suffix, side in (
        ("_lower", "lower"),
        ("_upper", "upper"),
        ("_neg", "negative"),
        ("_pos", "positive"),
    ):
        if arm.endswith(suffix):
            return arm[: -len(suffix)], side
    raise EvidenceError("formal selected arm has no explicit side")


def _parse_domain_axis_side(arm: Any, stratum: Any) -> Tuple[str, str]:
    if arm:
        axis, side = _parse_arm(arm)
        if stratum != "marginal:{}".format(arm):
            raise EvidenceError(
                "formal marginal stratum does not bind selected_arm_key"
            )
        return axis, side
    if stratum not in ("center", "joint", "steady"):
        raise EvidenceError(
            "formal non-marginal request has an unknown stratum"
        )
    return str(stratum), "not_applicable"


def _physical_width_rows(
    binding: Mapping[str, Any], levels: Sequence[float]
) -> Dict[str, Dict[str, Any]]:
    specs = binding.get("arm_width_spec")
    if specs is None:
        # Dependency-light unit callers may exercise the statistical reducer
        # without re-constructing a full manifest.  The formal attestor always
        # receives specs from _manifest_bindings().
        return {}
    if type(specs) is not dict or set(specs) != set(ARM_KEYS):
        raise EvidenceError("formal action binding lacks the exact arm width spec")
    result: Dict[str, Dict[str, Any]] = {}
    for index, arm in enumerate(ARM_KEYS):
        spec = _exact_dict(
            specs[arm],
            ("initial", "maximum", "unit"),
            "arm width spec {}".format(arm),
        )
        initial = _finite(spec["initial"], arm + " initial", minimum=0.0)
        maximum = _finite(spec["maximum"], arm + " maximum", minimum=initial)
        unit = spec["unit"]
        if type(unit) is not str or not unit:
            raise EvidenceError("arm width unit is missing")
        level = levels[index]
        result[arm] = {
            "level": level,
            "width": initial + level * (maximum - initial),
            "initial": initial,
            "maximum": maximum,
            "unit": unit,
        }
    return result


def _percentile(values: Sequence[int], probability: float) -> float:
    if not values:
        raise EvidenceError("starvation percentile has zero samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _wilson_lower(successes: int, attempts: int, z: float = 1.96) -> float:
    if attempts <= 0:
        raise EvidenceError("Wilson LCB requires non-zero safe-closed attempts")
    proportion = float(successes) / attempts
    z2 = z * z
    denominator = 1.0 + z2 / attempts
    center = proportion + z2 / (2.0 * attempts)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / attempts
        + z2 / (4.0 * attempts * attempts)
    )
    return max(0.0, (center - radius) / denominator)


def derive_stage_metrics(
    *,
    stage: str,
    max_iterations: int,
    interval_updates: int,
    action_bindings: Sequence[Mapping[str, Any]],
    formal_records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if stage not in ("smoke", "canary", "long"):
        raise EvidenceError("completed stage is invalid")
    iterations = _plain_int(max_iterations, "max_iterations", minimum=1)
    interval = _plain_int(interval_updates, "interval_updates", minimum=1)
    if interval > iterations:
        raise EvidenceError("frozen evaluation interval exceeds stage budget")
    expected_uids = [row["action_uid"] for row in action_bindings]
    if (
        not expected_uids
        or len(expected_uids) != len(set(expected_uids))
        or any(type(uid) is not int or uid < 1 for uid in expected_uids)
    ):
        raise EvidenceError("action bindings must contain unique positive UIDs")
    uid_to_id = {
        row["action_uid"]: row["motion_id"] for row in action_bindings
    }
    uid_to_binding = {row["action_uid"]: row for row in action_bindings}
    uid_to_mobility = {
        row["action_uid"]: row.get("mobility_mode")
        for row in action_bindings
    }
    if any(
        mobility not in ("no_move", "move")
        for mobility in uid_to_mobility.values()
    ):
        raise EvidenceError("action binding mobility_mode is invalid")
    if not formal_records:
        raise EvidenceError("formal evaluation has zero samples")
    if stage != "smoke":
        if iterations // interval < len(expected_uids):
            raise EvidenceError(
                "stage budget cannot cover one formal evaluation per action"
            )

    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    seen_actions: set[int] = set()
    latest_selected_round: Dict[Tuple[int, str], int] = {}
    last_selection_round = 0
    aggregate = {
        "proposed": 0,
        "physics_invalid": 0,
        "solver_rejected": 0,
        "solver_admitted": 0,
        "installed": 0,
        "started": 0,
        "closed": 0,
        "legal_return": 0,
        "safe_nonreturn": 0,
        "table_hit": 0,
        "fall": 0,
        "collision": 0,
        "joint_qdes_limit": 0,
        "joint_actual_limit": 0,
        "infrastructure_invalid": 0,
        "nan_count": 0,
        "counter_violation_count": 0,
        "accepted_ack_count": 0,
        "physics_invalid_reasons": {},
        "solver_reject_reasons": {},
    }
    for record_index, record in enumerate(formal_records):
        if type(record) is not dict:
            raise EvidenceError("formal record must be a plain object")
        uid = _plain_int(
            record.get("action_uid"),
            "formal record action_uid",
            minimum=1,
        )
        if uid not in uid_to_id:
            raise EvidenceError("formal record targets an unknown action")
        arm = record.get("selected_arm_key")
        stratum = record.get("stratum")
        axis, side = _parse_domain_axis_side(arm, stratum)
        expected_mobility = uid_to_mobility[uid]
        active_arms = NO_MOVE_ARMS if expected_mobility == "no_move" else ARM_KEYS
        if arm and arm not in active_arms:
            raise EvidenceError(
                "formal record selected an arm forbidden by mobility"
            )
        profile_sha = _sha256(
            record.get("profile_sha256"),
            "formal record profile_sha256",
        )
        if record.get("mobility_mode") != expected_mobility:
            raise EvidenceError(
                "formal record mobility differs from action binding"
            )
        domain_epoch = _plain_int(
            record.get("domain_epoch"),
            "formal record domain_epoch",
            minimum=0,
        )
        selection_round = _plain_int(
            record.get("selection_round"),
            "formal record selection_round",
            minimum=0,
        )
        last_selection_round = max(last_selection_round, selection_round)
        if arm:
            arm_round_key = (uid, arm)
            latest_selected_round[arm_round_key] = max(
                selection_round,
                latest_selected_round.get(arm_round_key, -1),
            )
        levels_raw = record.get("arm_levels")
        if type(levels_raw) is not list or len(levels_raw) != len(ARM_KEYS):
            raise EvidenceError(
                "formal record arm_levels must contain all 32 asymmetric arms"
            )
        levels = [
            _finite(
                value,
                "formal arm_levels[{}]".format(index),
                minimum=0.0,
                maximum=1.0,
            )
            for index, value in enumerate(levels_raw)
        ]
        if any(
            levels[ARM_KEYS.index(base_arm)] != 0.0
            for base_arm in ARM_KEYS
            if base_arm.startswith("base_travel_")
        ):
            raise EvidenceError("no_move formal domain enabled a base-travel level")
        rho = _finite(
            record.get("rho"),
            "formal record rho",
            minimum=0.0,
            maximum=1.0,
        )
        if stratum == "center" and (any(levels) or rho != 0.0):
            raise EvidenceError("center formal domain is not level-zero")
        if arm:
            selected_index = ARM_KEYS.index(arm)
            if levels[selected_index] <= 0.0 or any(
                value != 0.0
                for index, value in enumerate(levels)
                if index != selected_index
            ):
                raise EvidenceError(
                    "marginal formal domain must activate only its selected arm"
                )
        ack = _exact_dict(
            record.get("accepted_ack"),
            (
                "decision",
                "consumer_code_sha256",
                "consumer_state_sha256",
                "consumer_checkpoint",
            ),
            "formal accepted ACK",
        )
        checkpoint_receipt = _exact_dict(
            ack["consumer_checkpoint"],
            ("path", "sha256", "size_bytes"),
            "formal ACK consumer checkpoint",
        )
        if (
            ack["decision"] != "accepted"
            or type(checkpoint_receipt["path"]) is not str
            or not checkpoint_receipt["path"]
        ):
            raise EvidenceError("formal evidence lacks an accepted consumer ACK")
        _sha256(ack["consumer_code_sha256"], "formal ACK consumer code SHA")
        _sha256(ack["consumer_state_sha256"], "formal ACK consumer state SHA")
        _sha256(checkpoint_receipt["sha256"], "formal ACK checkpoint SHA")
        _plain_int(
            checkpoint_receipt["size_bytes"],
            "formal ACK checkpoint size",
            minimum=1,
        )
        # The V4 inbox permits an ACK only after the curriculum consumer has
        # accepted this exact request/evidence and persisted its state
        # checkpoint. Missing/stale evidence cannot manufacture this row.
        aggregate["accepted_ack_count"] += 1
        seen_actions.add(uid)
        windows = record.get("windows")
        if type(windows) is not list or [
            row.get("role") if type(row) is dict else None for row in windows
        ] != ["frozen_canary", "frozen_heldout"]:
            raise EvidenceError("formal record must contain canary then heldout")
        widths = _physical_width_rows(uid_to_binding[uid], levels)
        key = (
            uid,
            profile_sha,
            domain_epoch,
            stratum,
            arm,
            selection_round,
            tuple(levels),
            rho,
        )
        grouped = groups.setdefault(
            key,
            {
                "action_id": uid_to_id[uid],
                "action_uid": uid,
                "profile_sha256": profile_sha,
                "mobility_mode": expected_mobility,
                "domain_epoch": domain_epoch,
                "stratum": stratum,
                "axis": axis,
                "side": side,
                "selected_arm_key": arm,
                "selection_round": selection_round,
                "arm_levels": {
                    name: levels[index] for index, name in enumerate(ARM_KEYS)
                },
                "arm_physical_widths": widths,
                "selected_arm_physical_width": (
                    None if not arm or not widths else widths[arm]
                ),
                "rho": rho,
                "request_count": 0,
                "windows": {
                    "frozen_canary": {},
                    "frozen_heldout": {},
                },
            },
        )
        grouped["request_count"] += 1
        for window in windows:
            role = window["role"]
            ledger = window.get("ledger")
            if type(ledger) is not dict:
                raise EvidenceError("formal evidence ledger is missing")
            expected_count = 320 if role == "frozen_canary" else 960
            if ledger.get("proposed") != expected_count:
                raise EvidenceError(
                    "{} ledger must contain exactly {} proposals".format(
                        role, expected_count
                    )
                )
            raw_count = _plain_int(
                window.get("raw_attempt_count"),
                "{} raw_attempt_count".format(role),
                minimum=1,
            )
            raw_sha = _sha256(
                window.get("raw_attempts_sha256"),
                "{} raw attempts SHA".format(role),
            )
            raw_root = _sha256(
                window.get("attempt_receipt_root_sha256"),
                "{} attempt receipt root".format(role),
            )
            del raw_sha, raw_root
            raw_nonfinite = _plain_int(
                window.get("raw_nonfinite_count"),
                "{} raw_nonfinite_count".format(role),
                minimum=0,
            )
            aggregate["nan_count"] += raw_nonfinite
            if raw_count != expected_count or raw_count != ledger["proposed"]:
                aggregate["counter_violation_count"] += 1
                raise EvidenceError(
                    "{} raw attempt count differs from its fixed proposal window".format(
                        role
                    )
                )
            if raw_nonfinite != 0:
                raise EvidenceError(
                    "{} raw attempts contain non-finite values".format(role)
                )
            for name in (
                "proposed",
                "physics_invalid",
                "solver_rejected",
                "solver_admitted",
                "installed",
                "started",
                "closed",
                "legal_return",
                "safe_nonreturn",
                "table_hit",
                "fall",
                "collision",
                "joint_qdes_limit",
                "joint_actual_limit",
                "infrastructure_invalid",
            ):
                count = _plain_int(
                    ledger.get(name),
                    "formal ledger {}".format(name),
                    minimum=0,
                )
                aggregate[name] += count
            if (
                ledger["proposed"]
                != ledger["physics_invalid"]
                + ledger["solver_rejected"]
                + ledger["solver_admitted"]
            ):
                aggregate["counter_violation_count"] += 1
                raise EvidenceError("formal proposal ledger does not conserve")
            if (
                ledger["closed"]
                != ledger["legal_return"]
                + ledger["safe_nonreturn"]
                + ledger["table_hit"]
                + ledger["fall"]
                + ledger["collision"]
                + ledger["joint_qdes_limit"]
                + ledger["joint_actual_limit"]
            ):
                aggregate["counter_violation_count"] += 1
                raise EvidenceError("formal closed/outcome ledger does not conserve")
            if not (
                ledger["closed"] <= ledger["started"] <= ledger["installed"]
                <= ledger["solver_admitted"]
            ):
                aggregate["counter_violation_count"] += 1
                raise EvidenceError("formal attempt lifecycle is not monotonic")
            if (
                ledger["closed"] + ledger["infrastructure_invalid"]
                != ledger["solver_admitted"]
            ):
                aggregate["counter_violation_count"] += 1
                raise EvidenceError("formal admitted lifecycle does not conserve")
            for reason_field in (
                "physics_invalid_reasons",
                "solver_reject_reasons",
            ):
                reasons = ledger.get(reason_field)
                if type(reasons) is not dict or any(
                    type(reason) is not str
                    or not reason
                    or type(count) is not int
                    or count < 0
                    for reason, count in reasons.items()
                ):
                    raise EvidenceError("formal rejection reasons are invalid")
                expected_total = (
                    ledger["physics_invalid"]
                    if reason_field == "physics_invalid_reasons"
                    else ledger["solver_rejected"]
                )
                if sum(reasons.values()) != expected_total:
                    aggregate["counter_violation_count"] += 1
                    raise EvidenceError("formal rejection reason denominator is false")
                target = aggregate[reason_field]
                for reason, count in reasons.items():
                    target[reason] = target.get(reason, 0) + count
            role_row = grouped["windows"][role]
            for name in (
                "proposed",
                "physics_invalid",
                "solver_rejected",
                "solver_admitted",
                "installed",
                "started",
                "closed",
                "legal_return",
                "safe_nonreturn",
                "table_hit",
                "fall",
                "collision",
                "joint_qdes_limit",
                "joint_actual_limit",
                "infrastructure_invalid",
            ):
                role_row[name] = role_row.get(name, 0) + ledger[name]
            for reason_field in (
                "physics_invalid_reasons",
                "solver_reject_reasons",
            ):
                target = role_row.setdefault(reason_field, {})
                for reason, count in ledger[reason_field].items():
                    target[reason] = target.get(reason, 0) + count

    if stage != "smoke":
        missing = [
            uid_to_id[uid] for uid in expected_uids if uid not in seen_actions
        ]
        if missing:
            raise EvidenceError(
                "formal evaluation omitted actions: {}".format(missing)
            )
    if any(
        aggregate[name] != 0
        for name in (
            "table_hit",
            "fall",
            "collision",
            "joint_qdes_limit",
            "joint_actual_limit",
            "infrastructure_invalid",
        )
    ):
        raise EvidenceError("formal evaluation contains unsafe/infrastructure outcomes")
    attempts = aggregate["legal_return"] + aggregate["safe_nonreturn"]
    if attempts <= 0:
        raise EvidenceError("formal evaluation has zero safe-closed samples")

    reason_counts: Dict[str, int] = {}
    for prefix, field in (
        ("physics_invalid", "physics_invalid_reasons"),
        ("solver_rejected", "solver_reject_reasons"),
    ):
        for reason, count in sorted(aggregate[field].items()):
            reason_counts["{}/{}".format(prefix, reason)] = count
    rejected = aggregate["physics_invalid"] + aggregate["solver_rejected"]
    ack_stale_count = abs(
        aggregate["accepted_ack_count"] - len(formal_records)
    )
    aggregate["domain_epoch_stale_count"] = ack_stale_count
    metrics = {
        "proposed_count": aggregate["proposed"],
        "solver_admitted_count": aggregate["solver_admitted"],
        "solver_rejected_count": rejected,
        "solver_rejection_reason_counts": reason_counts,
        "attempt_count": attempts,
        "return_success_count": aggregate["legal_return"],
        "policy_return_failure_count": aggregate["safe_nonreturn"],
        "return_success_lcb": (
            _wilson_lower(aggregate["legal_return"], attempts)
            if attempts
            else 0.0
        ),
        "policy_return_failure_rate": (
            float(aggregate["safe_nonreturn"]) / attempts if attempts else 0.0
        ),
        "unsafe_count": aggregate["collision"],
        "table_hit_count": aggregate["table_hit"],
        "fall_count": aggregate["fall"],
        "hard_limit_count": (
            aggregate["joint_qdes_limit"] + aggregate["joint_actual_limit"]
        ),
        "nan_count": aggregate["nan_count"],
        "counter_violation_count": aggregate["counter_violation_count"],
        "domain_epoch_stale_count": aggregate["domain_epoch_stale_count"],
        "curriculum_counter_invariants_passed": (
            aggregate["counter_violation_count"] == 0
            and aggregate["accepted_ack_count"] == len(formal_records)
        ),
    }
    detailed = {
        "formal_request_count": len(formal_records),
        "formal_window_count": 2 * len(formal_records),
        "accepted_ack_implies_domain_epoch_stale_count": ack_stale_count,
        "per_action_axis_side": [
            groups[key] for key in sorted(groups)
        ],
        "aggregate_raw": aggregate,
    }
    starvation_rows = []
    ages = []
    for uid in expected_uids:
        for arm in NO_MOVE_ARMS:
            selected = latest_selected_round.get((uid, arm))
            age = (
                last_selection_round + 1
                if selected is None
                else last_selection_round - selected
            )
            ages.append(age)
            starvation_rows.append(
                {
                    "action_id": uid_to_id[uid],
                    "action_uid": uid,
                    "arm": arm,
                    "last_selected_round": selected,
                    "age": age,
                }
            )
    starvation = {
        "definition": (
            "formal-request coverage age in selection_round units for every "
            "action x active no_move arm; null arms are age current_round+1"
        ),
        "terminal_selection_round": last_selection_round,
        "sample_count": len(ages),
        "p5": _percentile(ages, 0.05),
        "p50": _percentile(ages, 0.50),
        "p95": _percentile(ages, 0.95),
        "rows": starvation_rows,
    }
    return metrics, detailed, starvation


def _tensor_audit(value: Any, torch_module: Any) -> Dict[str, int]:
    counts = {
        "tensor_count": 0,
        "floating_tensor_count": 0,
        "floating_element_count": 0,
        "nonfinite_floating_elements": 0,
    }
    seen: set[int] = set()

    def walk(item: Any) -> None:
        identity = id(item)
        if identity in seen:
            return
        if isinstance(item, (dict, list, tuple, set)):
            seen.add(identity)
        if torch_module.is_tensor(item):
            counts["tensor_count"] += 1
            if bool(item.is_floating_point()) or bool(item.is_complex()):
                counts["floating_tensor_count"] += 1
                counts["floating_element_count"] += int(item.numel())
                finite = torch_module.isfinite(item)
                counts["nonfinite_floating_elements"] += int(
                    item.numel() - int(finite.sum().item())
                )
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                walk(key)
                walk(nested)
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                walk(nested)

    walk(value)
    return counts


def audit_checkpoint_object(
    *,
    checkpoint: Any,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    training_contract_sha256: str,
    launch_claim_sha256: str,
    torch_module: Any,
) -> Dict[str, Any]:
    if type(checkpoint) is not dict:
        raise EvidenceError("checkpoint must load as one plain mapping")
    match = CHECKPOINT_RE.fullmatch(checkpoint_path.name)
    if match is None:
        raise EvidenceError("checkpoint filename must be model_<N>.pt")
    filename_iteration = int(match.group(1))
    embedded = _plain_int(checkpoint.get("iter"), "checkpoint iter", minimum=0)
    if embedded != filename_iteration:
        raise EvidenceError("checkpoint filename and embedded iteration differ")
    infos = checkpoint.get("infos")
    if type(infos) is not dict:
        raise EvidenceError("checkpoint infos are missing")
    if (
        infos.get("training_contract_schema_version") != 3
        or infos.get("training_contract_sha256") != training_contract_sha256
        or infos.get("training_contract_lineage_exact") != 1
        or infos.get("training_launch_claim_sha256") != launch_claim_sha256
    ):
        raise EvidenceError("checkpoint training/launch lineage is not exact")
    state = infos.get("hope_exact_resume_state")
    if state is None:
        state = checkpoint.get("hope_exact_resume_state")
    required = {
        "schema_version",
        "next_learning_iteration",
        "tot_timesteps",
        "tot_time",
        "algorithm_learning_rate",
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
        "torch_cuda_random_states",
        "torch_cuda_device_count",
        "environment_resume_state",
    }
    if (
        type(state) is not dict
        or state.get("schema_version") != 3
        or not required.issubset(state)
        or state.get("next_learning_iteration") != embedded + 1
    ):
        raise EvidenceError("checkpoint exact-resume schema/iteration is incomplete")
    _plain_int(state["tot_timesteps"], "exact-resume tot_timesteps", minimum=0)
    _finite(state["tot_time"], "exact-resume tot_time", minimum=0.0)
    _finite(
        state["algorithm_learning_rate"],
        "exact-resume learning rate",
        minimum=0.0,
    )
    if float(state["algorithm_learning_rate"]) <= 0.0:
        raise EvidenceError("exact-resume learning rate must be positive")
    _validate_numpy_rng_state(state["numpy_random_state"])
    environment = state["environment_resume_state"]
    environment = _exact_dict(
        environment,
        (
            "schema_version",
            "common_step_counter",
            "active_term_names",
            "command_terms",
        ),
        "checkpoint environment exact-resume state",
    )
    if environment["schema_version"] != 3:
        raise EvidenceError("checkpoint environment exact-resume schema is not 3")
    _plain_int(
        environment["common_step_counter"],
        "environment common_step_counter",
        minimum=0,
    )
    active = environment["active_term_names"]
    terms = environment["command_terms"]
    if (
        type(active) is not list
        or any(type(name) is not str or not name for name in active)
        or len(active) != len(set(active))
        or type(terms) is not dict
        or list(terms) != active
        or not {"racket_target", "motion"}.issubset(active)
    ):
        raise EvidenceError("checkpoint command-term exact-resume identity is incomplete")
    for term_name in active:
        term = terms[term_name]
        if (
            type(term) is not dict
            or term.get("capture_mode") != "explicit"
            or type(term.get("term_type")) is not str
            or not term["term_type"]
            or type(term.get("exact_state")) is not dict
        ):
            raise EvidenceError(
                "checkpoint command term {!r} is not explicit exact state".format(
                    term_name
                )
            )
    if type(checkpoint.get("optimizer_state_dict")) is not dict or not checkpoint[
        "optimizer_state_dict"
    ]:
        raise EvidenceError("checkpoint optimizer state is missing")
    tensor_counts = _tensor_audit(checkpoint, torch_module)
    if (
        tensor_counts["floating_tensor_count"] <= 0
        or tensor_counts["nonfinite_floating_elements"] != 0
    ):
        raise EvidenceError("checkpoint floating tensors are empty or non-finite")
    return {
        "path": checkpoint_path.name,
        "sha256": _sha256(checkpoint_sha256, "checkpoint SHA"),
        "filename_iteration": filename_iteration,
        "embedded_iteration": embedded,
        **tensor_counts,
        "exact_resume_schema_version": 3,
        "environment_resume_schema_version": 3,
        "explicit_command_terms": list(active),
        "finite": True,
        # This is deliberately only a structural result.  Formal
        # exact_resume_passed is granted later only after the independent
        # Isaac restore -> no-step save -> reload roundtrip receipt.
        "exact_resume_structure_passed": True,
    }


def _load_checkpoint(
    path: Path,
    *,
    training_contract_sha256: str,
    launch_claim_sha256: str,
    torch_module: Optional[Any] = None,
    checkpoint_loader: Optional[Callable[[bytes], Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    snapshot_before = _snapshot_file(path, "stage checkpoint", max_bytes=32 << 30)
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise EvidenceError(
                "torch is required to inspect the actual checkpoint"
            ) from exc
    if checkpoint_loader is None:
        try:
            checkpoint = torch_module.load(
                io.BytesIO(snapshot_before["raw"]),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise EvidenceError(
                "safe weights-only checkpoint load failed; unsigned pickle "
                "execution is forbidden"
            ) from exc
    else:
        checkpoint = checkpoint_loader(snapshot_before["raw"])
    snapshot_after = _snapshot_file(path, "stage checkpoint", max_bytes=32 << 30)
    if (
        snapshot_after["sha256"] != snapshot_before["sha256"]
        or snapshot_after["stat"].st_dev != snapshot_before["stat"].st_dev
        or snapshot_after["stat"].st_ino != snapshot_before["stat"].st_ino
    ):
        raise EvidenceError("checkpoint changed while being audited")
    audit = audit_checkpoint_object(
        checkpoint=checkpoint,
        checkpoint_path=path,
        checkpoint_sha256=snapshot_before["sha256"],
        training_contract_sha256=training_contract_sha256,
        launch_claim_sha256=launch_claim_sha256,
        torch_module=torch_module,
    )
    return audit, checkpoint


def _action_ball_racket_state(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    infos = checkpoint.get("infos")
    if type(infos) is not dict:
        raise EvidenceError("checkpoint infos are missing")
    resume = infos.get("hope_exact_resume_state")
    if type(resume) is not dict:
        raise EvidenceError("checkpoint exact-resume state is missing")
    environment = resume.get("environment_resume_state")
    if type(environment) is not dict:
        raise EvidenceError("checkpoint environment exact-resume state is missing")
    terms = environment.get("command_terms")
    if type(terms) is not dict:
        raise EvidenceError("checkpoint command terms are missing")
    racket = terms.get("racket_target")
    if type(racket) is not dict or racket.get("capture_mode") != "explicit":
        raise EvidenceError("checkpoint racket_target is not explicit exact state")
    state = racket.get("exact_state")
    if type(state) is not dict:
        raise EvidenceError("checkpoint racket_target exact state is missing")
    expected = {
        "schema_version",
        "kind",
        "manifest_sha256",
        "hard_contract",
        "action_order",
        "action_uids",
        "num_envs",
        "solver",
        "physics",
        "curriculum",
        "frozen_evaluation",
        "mutable_state",
        "broker",
        "pool",
        "ledger",
        "env_state",
        "last_rollout_step",
        "integrity_sha256",
    }
    if set(state) != expected:
        raise EvidenceError(
            "action-ball racket exact state has invalid keys "
            "(missing={}, unknown={})".format(
                sorted(expected - set(state)), sorted(set(state) - expected)
            )
        )
    unsigned = {key: state[key] for key in expected if key != "integrity_sha256"}
    digest = _sha256(
        state["integrity_sha256"], "action-ball exact-state integrity SHA"
    )
    if canonical_sha256(unsigned) != digest:
        raise EvidenceError("action-ball exact-state integrity SHA is false")
    if (
        state["schema_version"] != 5
        or state["kind"] != "whole_body_tracking.RacketTargetCommand.action_ball"
    ):
        raise EvidenceError("action-ball exact-state schema/kind is stale")
    _plain_int(state["num_envs"], "action-ball exact-state num_envs", minimum=1)
    return state


def _validate_artifact_binding(
    value: Any,
    *,
    label: str,
    expected_path: Optional[Path] = None,
    max_bytes: int = MAX_JSON_BYTES,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    row = _exact_dict(value, ("path", "sha256", "size_bytes"), label)
    path = _normalized_absolute(row["path"], label + " path")
    if expected_path is not None and path != expected_path:
        raise EvidenceError("{} path differs from the exact runtime path".format(label))
    digest = _sha256(row["sha256"], label + " SHA")
    size = _plain_int(row["size_bytes"], label + " size", minimum=1)
    snapshot = _snapshot_file(path, label, max_bytes=max_bytes)
    if (
        snapshot["sha256"] != digest
        or snapshot["stat"].st_size != size
        or snapshot["stat"].st_nlink != 1
    ):
        raise EvidenceError("{} bytes/size/link count drifted".format(label))
    return row, snapshot


def _checkpoint_runtime_bootstrap_evidence(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    checkout: Path,
    source_commit: str,
    claim_path: Path,
    claim_sha256: str,
    training_contract_path: Path,
    training_contract_sha256: str,
    claim_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    infos = checkpoint.get("infos")
    resume = infos.get("hope_exact_resume_state") if type(infos) is dict else None
    keys = (
        "runtime_bootstrap_receipt_sha256",
        "runtime_bootstrap_lineage_payload_sha256",
        "runtime_bootstrap_receipt",
    )
    if type(infos) is not dict or type(resume) is not dict:
        raise EvidenceError("checkpoint bootstrap infos/exact state are missing")
    try:
        infos_binding = {key: infos[key] for key in keys}
        resume_binding = {key: resume[key] for key in keys}
    except KeyError as exc:
        raise EvidenceError(
            "checkpoint lacks runtime-bootstrap exact-resume binding"
        ) from exc
    if infos_binding != resume_binding:
        raise EvidenceError(
            "checkpoint bootstrap infos and exact-resume bindings differ"
        )
    content_sha = _sha256(
        infos_binding["runtime_bootstrap_receipt_sha256"],
        "checkpoint runtime bootstrap content SHA",
    )
    lineage_sha = _sha256(
        infos_binding["runtime_bootstrap_lineage_payload_sha256"],
        "checkpoint runtime bootstrap lineage SHA",
    )
    expected_receipt_path = (
        checkpoint_path.parent
        / "params/action_ball_runtime_bootstrap_receipt.json"
    )
    artifact, artifact_snapshot = _validate_artifact_binding(
        infos_binding["runtime_bootstrap_receipt"],
        label="checkpoint runtime bootstrap receipt",
        expected_path=expected_receipt_path,
    )
    document = _strict_json_bytes(
        artifact_snapshot["raw"], "checkpoint runtime bootstrap receipt"
    )
    envelope = _exact_dict(
        document,
        ("schema_version", "kind", "content", "content_sha256"),
        "checkpoint runtime bootstrap receipt",
    )
    content_keys = (
        "runtime_bootstrap_contract_sha256",
        "task_id",
        "training_launch_claim_sha256",
        "launch_claim",
        "training_contract",
        "environment_config_pickle",
        "agent_config_pickle",
        "runtime_identity",
        "runtime_inventory",
        "runtime_identity_content_sha256",
        "runtime_identity_contract_sha256",
        "source",
        "lineage_payload",
        "lineage_payload_sha256",
    )
    content = _exact_dict(
        envelope["content"], content_keys, "runtime bootstrap content"
    )
    if (
        envelope["schema_version"] != 1
        or envelope["kind"] != "action_ball_runtime_bootstrap_receipt_v1"
        or envelope["content_sha256"] != content_sha
        or canonical_sha256(content) != content_sha
        or content["training_launch_claim_sha256"] != claim_sha256
        or content["lineage_payload_sha256"] != lineage_sha
        or canonical_sha256(content["lineage_payload"]) != lineage_sha
        or content["task_id"] != "HOPE-PingPong-ActionBall-AgibotA3-v0"
    ):
        raise EvidenceError("runtime bootstrap envelope/claim/lineage is invalid")
    for name in (
        "runtime_bootstrap_contract_sha256",
        "runtime_identity_content_sha256",
        "runtime_identity_contract_sha256",
    ):
        _sha256(content[name], "runtime bootstrap {}".format(name))
    isaac_runtime = claim_payload.get("isaac_python_runtime")
    inventory_identity = (
        isaac_runtime.get("runtime_inventory")
        if type(isaac_runtime) is dict
        else None
    )
    if type(inventory_identity) is not dict:
        raise EvidenceError("claim lacks runtime inventory identity")
    inventory_identity = _exact_dict(
        inventory_identity,
        ("path", "file_sha256", "content_sha256", "kind"),
        "claim runtime inventory identity",
    )
    _sha256(
        inventory_identity["file_sha256"],
        "claim runtime inventory file SHA",
    )
    _sha256(
        inventory_identity["content_sha256"],
        "claim runtime inventory content SHA",
    )
    if inventory_identity["kind"] != "action_ball_runtime_inventory_v1":
        raise EvidenceError("claim runtime inventory kind is invalid")
    inventory_path = _normalized_absolute(
        inventory_identity.get("path"), "claim runtime inventory path"
    )
    expected_paths = {
        "launch_claim": claim_path,
        "training_contract": training_contract_path,
        "environment_config_pickle": (
            checkpoint_path.parent / "params/env.pkl"
        ),
        "agent_config_pickle": (
            checkpoint_path.parent / "params/agent.pkl"
        ),
        "runtime_identity": (
            checkpoint_path.parent
            / "params/action_ball_frozen_eval_runtime.json"
        ),
        "runtime_inventory": inventory_path,
    }
    artifacts: Dict[str, Any] = {}
    for name, expected_path in expected_paths.items():
        row, snapshot = _validate_artifact_binding(
            content[name],
            label="runtime bootstrap {}".format(name),
            expected_path=expected_path,
            max_bytes=(32 << 30 if name in ("environment_config_pickle", "agent_config_pickle") else MAX_JSON_BYTES),
        )
        artifacts[name] = {
            "path": row["path"],
            "sha256": snapshot["sha256"],
            "size_bytes": snapshot["stat"].st_size,
        }
    if (
        artifacts["launch_claim"]["sha256"]
        != _snapshot_file(claim_path, "runtime bootstrap claim cross-check")[
            "sha256"
        ]
        or artifacts["training_contract"]["sha256"]
        != training_contract_sha256
        or artifacts["runtime_inventory"]["sha256"]
        != inventory_identity.get("file_sha256")
    ):
        raise EvidenceError(
            "runtime bootstrap claim/contract/inventory bytes differ"
        )
    source = _exact_dict(
        content["source"],
        ("repo_root", "object_format", "head_commit_oid", "detached", "clean"),
        "runtime bootstrap source",
    )
    if (
        source["repo_root"] != str(checkout)
        or source["head_commit_oid"] != source_commit
        or source["detached"] is not True
        or source["clean"] is not True
        or source["object_format"] not in ("sha1", "sha256")
    ):
        raise EvidenceError("runtime bootstrap source is not exact clean detached commit")
    runtime_shas = claim_payload.get("runtime_code_sha256")
    if type(runtime_shas) is not dict:
        raise EvidenceError("claim lacks runtime code pins for bootstrap verifier")
    bootstrap_source = _committed_file(
        checkout,
        source_commit,
        RUNTIME_BOOTSTRAP_SOURCE,
        "runtime bootstrap source",
    )
    if runtime_shas.get(RUNTIME_BOOTSTRAP_SOURCE) != bootstrap_source["sha256"]:
        raise EvidenceError("claim runtime-bootstrap source pin differs from commit")
    return {
        "content_sha256": content_sha,
        "lineage_payload_sha256": lineage_sha,
        "receipt_artifact": dict(artifact),
        "runtime_bootstrap_source_sha256": bootstrap_source["sha256"],
        "artifacts": artifacts,
        "source": dict(source),
    }


def _exact_resume_verification_evidence(
    *,
    namespace: Path,
    checkpoint_path: Path,
    checkpoint_audit: Mapping[str, Any],
    runtime_bootstrap: Mapping[str, Any],
    checkout: Path,
    source_commit: str,
    claim_sha256: str,
    stage: str,
    claim_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Re-open the independent real restore/no-step-save/reload receipt.

    Structural checkpoint inspection is intentionally insufficient here.  The
    only source of ``exact_resume_passed`` is this exact, committed-verifier
    receipt, whose source and roundtrip checkpoint bytes are both reopened.
    """

    receipt_path = namespace / "exact_resume_verification.json"
    document, receipt_snapshot = _snapshot_json(
        receipt_path, "exact-resume verification receipt"
    )
    row = _exact_dict(
        document,
        (
            "schema_version",
            "kind",
            "status",
            "source_commit_sha",
            "launch_claim_sha256",
            "stage",
            "namespace",
            "verifier",
            "source_checkpoint",
            "roundtrip_checkpoint",
            "runtime_bootstrap",
            "restore",
            "state",
            "natural_exit",
            "receipt_payload_sha256",
        ),
        "exact-resume verification receipt",
    )
    payload_sha = _sha256(
        row["receipt_payload_sha256"],
        "exact-resume receipt payload SHA",
    )
    unsigned = {
        key: value
        for key, value in row.items()
        if key != "receipt_payload_sha256"
    }
    if (
        row["schema_version"] != 1
        or row["kind"]
        != "action_ball_exact_resume_verification_v1"
        or row["status"] != "passed"
        or row["source_commit_sha"] != source_commit
        or row["launch_claim_sha256"] != claim_sha256
        or row["stage"] != stage
        or row["namespace"] != str(namespace)
        or row["natural_exit"] is not True
        or canonical_sha256(unsigned) != payload_sha
    ):
        raise EvidenceError(
            "exact-resume verification identity/payload is invalid"
        )

    verifier = _exact_dict(
        row["verifier"],
        (
            "source_path",
            "source_sha256",
            "runtime_factory_source_path",
            "runtime_factory_source_sha256",
        ),
        "exact-resume verifier identity",
    )
    verifier_source = _committed_file(
        checkout,
        source_commit,
        EXACT_RESUME_VERIFIER_SOURCE,
        "exact-resume verifier source",
    )
    factory_source = _committed_file(
        checkout,
        source_commit,
        EXACT_RESUME_FACTORY_SOURCE,
        "exact-resume runtime factory source",
    )
    runtime_shas = claim_payload.get("runtime_code_sha256")
    if (
        type(runtime_shas) is not dict
        or verifier["source_path"] != EXACT_RESUME_VERIFIER_SOURCE
        or verifier["source_sha256"] != verifier_source["sha256"]
        or runtime_shas.get(EXACT_RESUME_VERIFIER_SOURCE)
        != verifier_source["sha256"]
        or verifier["runtime_factory_source_path"]
        != EXACT_RESUME_FACTORY_SOURCE
        or verifier["runtime_factory_source_sha256"]
        != factory_source["sha256"]
        or runtime_shas.get(EXACT_RESUME_FACTORY_SOURCE)
        != factory_source["sha256"]
    ):
        raise EvidenceError(
            "exact-resume verifier/factory source pins differ from commit"
        )

    source = _exact_dict(
        row["source_checkpoint"],
        ("path", "sha256", "size_bytes", "embedded_iteration"),
        "exact-resume source checkpoint",
    )
    source_path = _normalized_absolute(
        source["path"], "exact-resume source checkpoint path"
    )
    source_snapshot = _snapshot_file(
        source_path, "exact-resume source checkpoint", max_bytes=32 << 30
    )
    source_iteration = _plain_int(
        source["embedded_iteration"],
        "exact-resume source embedded iteration",
        minimum=0,
    )
    if (
        source_path != checkpoint_path
        or source["sha256"] != checkpoint_audit["sha256"]
        or source_snapshot["sha256"] != checkpoint_audit["sha256"]
        or source["size_bytes"] != source_snapshot["stat"].st_size
        or source_iteration != checkpoint_audit["embedded_iteration"]
    ):
        raise EvidenceError(
            "exact-resume source checkpoint differs from selected final bytes"
        )

    roundtrip = _exact_dict(
        row["roundtrip_checkpoint"],
        ("path", "sha256", "size_bytes", "embedded_iteration"),
        "exact-resume roundtrip checkpoint",
    )
    expected_roundtrip_path = (
        checkpoint_path.parent
        / (
            "exact_resume_roundtrip_"
            + claim_sha256[:16]
        )
        / checkpoint_path.name
    )
    roundtrip_path = _normalized_absolute(
        roundtrip["path"], "exact-resume roundtrip checkpoint path"
    )
    roundtrip_snapshot = _snapshot_file(
        roundtrip_path,
        "exact-resume roundtrip checkpoint",
        max_bytes=32 << 30,
    )
    if (
        roundtrip_path != expected_roundtrip_path
        or roundtrip["sha256"] != roundtrip_snapshot["sha256"]
        or roundtrip["size_bytes"] != roundtrip_snapshot["stat"].st_size
        or roundtrip["embedded_iteration"] != source_iteration
    ):
        raise EvidenceError(
            "exact-resume roundtrip checkpoint artifact/iteration drifted"
        )

    bootstrap = _exact_dict(
        row["runtime_bootstrap"],
        ("content_sha256", "lineage_payload_sha256"),
        "exact-resume runtime bootstrap",
    )
    if (
        _sha256(
            bootstrap["content_sha256"],
            "exact-resume bootstrap content SHA",
        )
        != runtime_bootstrap["content_sha256"]
        or _sha256(
            bootstrap["lineage_payload_sha256"],
            "exact-resume bootstrap lineage SHA",
        )
        != runtime_bootstrap["lineage_payload_sha256"]
    ):
        raise EvidenceError(
            "exact-resume runtime bootstrap differs from final checkpoint"
        )

    restore = _exact_dict(
        row["restore"],
        (
            "factory_call_count",
            "closed_runtime_count",
            "load_optimizer",
            "fresh_strict_load_token_consumed",
            "roundtrip_save_api",
            "roundtrip_save_receipt_sha256",
            "source_construction_receipt_sha256",
            "roundtrip_construction_receipt_sha256",
            "runtime_inventory_live_verification_sha256",
            "source_live_state_receipt_sha256",
            "roundtrip_live_state_receipt_sha256",
            "live_core_sha256",
            "common_step_counter",
            "common_step_counter_delta",
        ),
        "exact-resume restore proof",
    )
    if (
        restore["factory_call_count"] != 2
        or restore["closed_runtime_count"] != 2
        or restore["load_optimizer"] is not True
        or restore["fresh_strict_load_token_consumed"] is not True
        or restore["roundtrip_save_api"]
        != "save_exact_resume_roundtrip"
        or restore["source_live_state_receipt_sha256"]
        != restore["roundtrip_live_state_receipt_sha256"]
        or restore["common_step_counter_delta"] != 0
    ):
        raise EvidenceError(
            "exact-resume receipt does not prove two closed zero-step restores"
        )
    for key in (
        "roundtrip_save_receipt_sha256",
        "source_construction_receipt_sha256",
        "roundtrip_construction_receipt_sha256",
        "runtime_inventory_live_verification_sha256",
        "source_live_state_receipt_sha256",
        "roundtrip_live_state_receipt_sha256",
        "live_core_sha256",
    ):
        _sha256(restore[key], "exact-resume restore {}".format(key))
    _plain_int(
        restore["common_step_counter"],
        "exact-resume restore common step counter",
        minimum=0,
    )

    state = _exact_dict(
        row["state"],
        (
            "source_core_sha256",
            "roundtrip_core_sha256",
            "source_exact_resume_sha256",
            "roundtrip_exact_resume_sha256",
            "model_state_sha256",
            "optimizer_state_sha256",
            "normalizer_state_sha256",
        ),
        "exact-resume state proof",
    )
    for key, value in state.items():
        _sha256(value, "exact-resume state {}".format(key))
    if (
        state["source_core_sha256"]
        != state["roundtrip_core_sha256"]
        or state["source_exact_resume_sha256"]
        != state["roundtrip_exact_resume_sha256"]
    ):
        raise EvidenceError(
            "exact-resume checkpoint core/exact environment state drifted"
        )

    return {
        "exact_resume_passed": True,
        "receipt_artifact": {
            "path": receipt_path.name,
            "sha256": receipt_snapshot["sha256"],
            "size_bytes": receipt_snapshot["stat"].st_size,
        },
        "receipt_payload_sha256": payload_sha,
        "verifier": dict(verifier),
        "source_checkpoint": dict(source),
        "roundtrip_checkpoint": dict(roundtrip),
        "runtime_bootstrap": dict(bootstrap),
        "restore": dict(restore),
        "state": dict(state),
        "natural_exit": True,
    }


def _validate_short_ledger(
    value: Any, *, label: str
) -> Dict[str, int]:
    row = _exact_dict(value, CURRICULUM_LEDGER_KEYS, label)
    result = {
        name: _plain_int(row[name], "{}.{}".format(label, name), minimum=0)
        for name in CURRICULUM_LEDGER_KEYS
    }
    if not (
        result["P"]
        >= result["A"]
        >= result["I"]
        >= result["S"]
        >= result["C"]
    ):
        raise EvidenceError("{} lifecycle is not monotonic".format(label))
    unsafe_closures = result["C"] - result["L"] - result["F"]
    raw_unsafe = [
        result[name]
        for name in (
            "U_table",
            "U_fall",
            "U_collision",
            "U_joint_qdes",
            "U_joint_actual",
        )
    ]
    # Raw sticky safety channels may overlap on one closure (for example a
    # joint-limit and table hit on the same sample).  Conservation therefore
    # uses the unique unsafe closure count C-L-F; requiring a sum equality
    # would erase exactly the overlaps schema 10 was introduced to retain.
    if (
        unsafe_closures < 0
        or (
            unsafe_closures > 0
            and (
                max(raw_unsafe) > unsafe_closures
                or sum(raw_unsafe) < unsafe_closures
            )
        )
        or (unsafe_closures == 0 and any(raw_unsafe))
    ):
        raise EvidenceError(
            "{} raw sticky safety signals do not cover unique unsafe "
            "closures".format(label)
        )
    if (
        result["X"] > result["P"]
        or result["NB"] > result["L"] + result["F"]
        or result["NB_F"] > result["NB"]
        or result["NB_F"] > result["F"]
    ):
        raise EvidenceError("{} new-band/infrastructure counts are invalid".format(label))
    return result


def _scheduler_attempt_ledger(
    attempts: Any, *, label: str
) -> Dict[str, int]:
    if type(attempts) is not list:
        raise EvidenceError("{} must be a list".format(label))
    terminal_order = (
        "joint_actual_limit",
        "joint_qdes_limit",
        "fall",
        "table_hit",
        "collision",
        "legal_return",
    )
    signal_keys = (
        "infrastructure_invalid",
        *terminal_order,
    )
    result = {
        name: 0 for name in CURRICULUM_LEDGER_KEYS
    }
    result["P"] = len(attempts)
    for index, raw in enumerate(attempts):
        row_label = "{}[{}]".format(label, index)
        row = _exact_dict(
            raw,
            (
                "sample_receipt_sha256",
                "birth_receipt_sha256",
                "solver_admitted",
                "installed",
                "started",
                "closed",
                "terminal_outcome",
                "infrastructure_invalid",
                "in_new_band",
                "terminal_signals",
            ),
            row_label,
        )
        _sha256(
            row["sample_receipt_sha256"],
            row_label + " sample receipt SHA",
        )
        _sha256(
            row["birth_receipt_sha256"],
            row_label + " birth receipt SHA",
        )
        for key in (
            "solver_admitted",
            "installed",
            "started",
            "closed",
            "infrastructure_invalid",
            "in_new_band",
        ):
            if type(row[key]) is not bool:
                raise EvidenceError(
                    "{}.{} must be bool".format(row_label, key)
                )
        if not (
            row["solver_admitted"]
            >= row["installed"]
            >= row["started"]
            >= row["closed"]
        ):
            raise EvidenceError(
                "{} lifecycle flags are not contained".format(row_label)
            )
        signals_raw = row["terminal_signals"]
        if signals_raw is None:
            signals = None
        else:
            signals = _exact_dict(
                signals_raw, signal_keys, row_label + " terminal signals"
            )
            if any(type(signals[key]) is not bool for key in signal_keys):
                raise EvidenceError(
                    "{} terminal signals must be bool".format(row_label)
                )
            if (
                signals["infrastructure_invalid"]
                != row["infrastructure_invalid"]
            ):
                raise EvidenceError(
                    "{} infrastructure signal differs".format(row_label)
                )
        terminal = row["terminal_outcome"]
        if row["closed"]:
            if signals is None or signals["infrastructure_invalid"]:
                raise EvidenceError(
                    "{} closed row lacks physical terminal signals".format(
                        row_label
                    )
                )
            primary = "safe_nonreturn"
            for candidate in terminal_order:
                if signals[candidate]:
                    primary = candidate
                    break
            if terminal != primary:
                raise EvidenceError(
                    "{} primary terminal differs from raw sticky signals".format(
                        row_label
                    )
                )
        elif terminal is not None:
            raise EvidenceError(
                "{} nonclosed row has terminal outcome".format(row_label)
            )
        elif signals is not None:
            if row["infrastructure_invalid"]:
                if any(signals[key] for key in terminal_order):
                    raise EvidenceError(
                        "{} infrastructure burn has physical signals".format(
                            row_label
                        )
                    )
            elif any(signals.values()):
                raise EvidenceError(
                    "{} unsettled row has terminal signals".format(row_label)
                )

        result["A"] += int(row["solver_admitted"])
        result["I"] += int(row["installed"])
        result["S"] += int(row["started"])
        result["C"] += int(row["closed"])
        result["X"] += int(row["infrastructure_invalid"])
        terminal_to_ledger = {
            "legal_return": "L",
            "safe_nonreturn": "F",
        }
        if terminal in terminal_to_ledger:
            result[terminal_to_ledger[terminal]] += 1
        if signals is not None:
            for signal, ledger_key in (
                ("table_hit", "U_table"),
                ("fall", "U_fall"),
                ("collision", "U_collision"),
                ("joint_qdes_limit", "U_joint_qdes"),
                ("joint_actual_limit", "U_joint_actual"),
            ):
                result[ledger_key] += int(signals[signal])
        if (
            row["in_new_band"]
            and row["closed"]
            and terminal in ("legal_return", "safe_nonreturn")
        ):
            result["NB"] += 1
            result["NB_F"] += int(terminal == "safe_nonreturn")
    return _validate_short_ledger(result, label=label + " raw ledger")


def _formal_short_ledger(
    ledger: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    *,
    selected_arm_key: str,
) -> Dict[str, int]:
    new_band = 0
    new_band_failures = 0
    for index, attempt in enumerate(attempts):
        if type(attempt) is not dict:
            raise EvidenceError(
                "formal raw attempt {} is not a plain object".format(index)
            )
        terminal = attempt.get("terminal_signals")
        if type(terminal) is not dict:
            raise EvidenceError(
                "formal raw attempt {} lacks terminal signals".format(index)
            )
        eligible = (
            bool(selected_arm_key)
            and attempt.get("sampling_stratum") == "frontier"
            and attempt.get("frontier_arm") == selected_arm_key
            and attempt.get("closed") is True
            and terminal.get("infrastructure_invalid") is False
            and not any(
                terminal.get(name) is True
                for name in (
                    "joint_actual_limit",
                    "joint_qdes_limit",
                    "fall",
                    "table_hit",
                    "collision",
                )
            )
        )
        if eligible:
            new_band += 1
            if terminal.get("legal_return") is False:
                new_band_failures += 1
    return _validate_short_ledger(
        {
            "P": ledger["proposed"],
            "A": ledger["solver_admitted"],
            "I": ledger["installed"],
            "S": ledger["started"],
            "C": ledger["closed"],
            "L": ledger["legal_return"],
            "F": ledger["safe_nonreturn"],
            "U_table": ledger["table_hit"],
            "U_fall": ledger["fall"],
            "U_collision": ledger["collision"],
            "U_joint_qdes": ledger["joint_qdes_limit"],
            "U_joint_actual": ledger["joint_actual_limit"],
            "X": ledger["infrastructure_invalid"],
            "NB": new_band,
            "NB_F": new_band_failures,
        },
        label="formal reconstructed curriculum ledger",
    )


def _ordered_receipt_root(
    receipts: Sequence[Any], *, kind: str, field: str
) -> str:
    normalized = [
        _sha256(value, "{}[{}]".format(field, index))
        for index, value in enumerate(receipts)
    ]
    if not normalized:
        raise EvidenceError("{} must not be empty".format(field))
    if len(normalized) != len(set(normalized)):
        raise EvidenceError("{} contains duplicate receipts".format(field))
    return canonical_sha256(
        {
            "schema_version": 3,
            "kind": kind,
            "count": len(normalized),
            field: normalized,
        }
    )


def _formal_window_match_document(
    *,
    record: Mapping[str, Any],
    window: Mapping[str, Any],
    arm_catalog_sha256: str,
    scheduler_contract_sha256: str,
) -> Dict[str, Any]:
    allocation = window.get("allocation")
    if type(allocation) is not dict:
        raise EvidenceError("formal raw window allocation is missing")
    attempts = window.get("raw_attempts")
    if type(attempts) is not list or not attempts:
        raise EvidenceError("formal raw window attempts are missing")
    samples = window.get("ordered_sample_receipt_sha256")
    births = window.get("ordered_birth_receipt_sha256")
    if type(samples) is not list or type(births) is not list:
        raise EvidenceError("formal raw receipt lists are missing")
    sample_root = _ordered_receipt_root(
        samples,
        kind="action_ball_ordered_sample_receipts",
        field="ordered_sample_receipt_sha256",
    )
    birth_root = _ordered_receipt_root(
        births,
        kind="action_ball_ordered_birth_receipts",
        field="ordered_birth_receipt_sha256",
    )
    ledger = _formal_short_ledger(
        window["ledger"],
        attempts,
        selected_arm_key=str(record["selected_arm_key"]),
    )
    role = window["role"]
    return {
        "key": {
            "action_uid": record["action_uid"],
            "profile_sha256": record["profile_sha256"],
            "mobility": record["mobility_mode"],
        },
        "arm_catalog_sha256": arm_catalog_sha256,
        "scheduler_contract_sha256": scheduler_contract_sha256,
        "sampler_sha256": record["sampler_sha256"],
        "solver_sha256": record["solver_sha256"],
        "policy_contract_sha256": record["policy_contract_sha256"],
        "policy_checkpoint_sha256": record["policy_checkpoint_sha256"],
        "policy_generation": record["policy_generation"],
        "evidence_role": role,
        "domain_epoch": record["domain_epoch"],
        "stratum": record["stratum"],
        "selected_arm_key": record["selected_arm_key"],
        "selection_round": record["selection_round"],
        "arm_levels": list(record["arm_levels"]),
        "rho": record["rho"],
        "seed_block_start": allocation["seed_start"],
        "seed_block_end_exclusive": allocation["seed_end_exclusive"],
        "sample_id_start": allocation["sample_start"],
        "sample_id_end_exclusive": allocation["sample_end_exclusive"],
        "sample_receipt_root_sha256": sample_root,
        "unique_birth_count": len(set(births)),
        "birth_receipt_root_sha256": birth_root,
        "ledger": ledger,
    }


def _curriculum_receipt_match_document(
    receipt: Any,
    *,
    label: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    row = _exact_dict(
        receipt,
        ("evidence", "window_sha256", "certified"),
        label,
    )
    if type(row["certified"]) is not bool:
        raise EvidenceError("{}.certified must be an exact boolean".format(label))
    evidence = _exact_dict(row["evidence"], CURRICULUM_EVIDENCE_KEYS, label + ".evidence")
    window_sha = _sha256(row["window_sha256"], label + ".window_sha256")
    if canonical_sha256(evidence) != window_sha:
        raise EvidenceError("{} window SHA is false".format(label))
    if evidence["schema_version"] != 4:
        raise EvidenceError("{} evidence schema is stale".format(label))
    key = _exact_dict(
        evidence["key"],
        ("action_uid", "profile_sha256", "mobility"),
        label + ".evidence.key",
    )
    _plain_int(key["action_uid"], label + " action_uid", minimum=1)
    _sha256(key["profile_sha256"], label + " profile_sha256")
    if key["mobility"] not in ("no_move", "move"):
        raise EvidenceError("{} mobility is invalid".format(label))
    levels = evidence["arm_levels"]
    if type(levels) is not list or len(levels) != len(ARM_KEYS):
        raise EvidenceError("{} arm_levels shape is invalid".format(label))
    for index, value in enumerate(levels):
        _finite(
            value,
            "{} arm_levels[{}]".format(label, index),
            minimum=0.0,
            maximum=1.0,
        )
    _validate_short_ledger(evidence["ledger"], label=label + ".ledger")
    for name in (
        "arm_catalog_sha256",
        "scheduler_contract_sha256",
        "sampler_sha256",
        "solver_sha256",
        "policy_contract_sha256",
        "policy_checkpoint_sha256",
        "sample_receipt_root_sha256",
        "birth_receipt_root_sha256",
        "window_id",
    ):
        _sha256(evidence[name], "{}.{}".format(label, name))
    for name in (
        "policy_generation",
        "domain_epoch",
        "selection_round",
        "seed_block_start",
        "seed_block_end_exclusive",
        "sample_id_start",
        "sample_id_end_exclusive",
        "unique_birth_count",
        "seq",
    ):
        _plain_int(evidence[name], "{}.{}".format(label, name), minimum=0)
    if evidence["unique_birth_count"] <= 0:
        raise EvidenceError("{} unique birth count is zero".format(label))
    match = {
        name: evidence[name]
        for name in (
            "key",
            "arm_catalog_sha256",
            "scheduler_contract_sha256",
            "sampler_sha256",
            "solver_sha256",
            "policy_contract_sha256",
            "policy_checkpoint_sha256",
            "policy_generation",
            "evidence_role",
            "domain_epoch",
            "stratum",
            "selected_arm_key",
            "selection_round",
            "arm_levels",
            "rho",
            "seed_block_start",
            "seed_block_end_exclusive",
            "sample_id_start",
            "sample_id_end_exclusive",
            "sample_receipt_root_sha256",
            "unique_birth_count",
            "birth_receipt_root_sha256",
            "ledger",
        )
    }
    return evidence, match


def _derive_checkpoint_curriculum_evidence(
    *,
    checkpoint: Mapping[str, Any],
    action_bindings: Sequence[Mapping[str, Any]],
    formal_records: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
) -> Dict[str, Any]:
    state = _action_ball_racket_state(checkpoint)
    expected_ids = [row["motion_id"] for row in action_bindings]
    expected_uids = [row["action_uid"] for row in action_bindings]
    expected_mobility = [
        row.get("mobility_mode") for row in action_bindings
    ]
    if any(item not in ("no_move", "move") for item in expected_mobility):
        raise EvidenceError("checkpoint action binding mobility is invalid")
    if (
        state["manifest_sha256"] != manifest_sha256
        or state["action_order"] != expected_ids
        or state["action_uids"] != expected_uids
    ):
        raise EvidenceError("checkpoint action/manifest identity differs from claim")
    curriculum = _exact_dict(
        state["curriculum"], CURRICULUM_TOP_KEYS, "checkpoint curriculum state"
    )
    curriculum_sha = _sha256(
        curriculum["state_sha256"], "checkpoint curriculum state SHA"
    )
    unsigned_curriculum = dict(curriculum)
    del unsigned_curriculum["state_sha256"]
    if canonical_sha256(unsigned_curriculum) != curriculum_sha:
        raise EvidenceError("checkpoint curriculum state SHA is false")
    if curriculum["schema_version"] != 10:
        raise EvidenceError("checkpoint curriculum schema is stale")
    catalog = {
        "schema_version": 3,
        "arm_keys": list(ARM_KEYS),
    }
    arm_catalog_sha = _sha256(
        curriculum["arm_catalog_sha256"], "checkpoint arm catalog SHA"
    )
    if curriculum["arm_catalog"] != catalog or canonical_sha256(catalog) != arm_catalog_sha:
        raise EvidenceError("checkpoint curriculum arm catalog is not exact")
    scheduler_sha = _sha256(
        curriculum["scheduler_contract_sha256"],
        "checkpoint scheduler contract SHA",
    )
    for name in (
        "contract_sha256",
        "sampler_sha256",
        "solver_sha256",
        "policy_contract_sha256",
    ):
        _sha256(curriculum[name], "checkpoint curriculum {}".format(name))
    profiles = curriculum["profile_order"]
    progress_rows = curriculum["progress"]
    if (
        type(profiles) is not list
        or type(progress_rows) is not list
        or len(profiles) != len(expected_uids)
        or len(progress_rows) != len(expected_uids)
    ):
        raise EvidenceError(
            "checkpoint curriculum is not exact contracted order"
        )

    observed_formal_matches: List[Dict[str, Any]] = []
    starvation_rows: List[Dict[str, Any]] = []
    starvation_ages: List[int] = []
    coverage_count = 0
    profile_rows: List[Dict[str, Any]] = []
    for slot, (
        uid,
        action_id,
        mobility_mode,
        profile_raw,
        progress_raw,
    ) in enumerate(
        zip(
            expected_uids,
            expected_ids,
            expected_mobility,
            profiles,
            progress_rows,
        )
    ):
        profile = _exact_dict(
            profile_raw,
            ("action_uid", "profile_sha256", "mobility"),
            "checkpoint profile_order[{}]".format(slot),
        )
        profile_sha = _sha256(
            profile["profile_sha256"],
            "checkpoint profile_order[{}].profile_sha256".format(slot),
        )
        if (
            profile["action_uid"] != uid
            or profile["mobility"] != mobility_mode
        ):
            raise EvidenceError(
                "checkpoint profile order/mobility differs from contract"
            )
        progress = _exact_dict(
            progress_raw,
            CURRICULUM_PROGRESS_KEYS,
            "checkpoint progress[{}]".format(slot),
        )
        if progress["key"] != profile:
            raise EvidenceError("checkpoint progress key differs from profile order")
        phase = progress["phase"]
        if phase not in ("center", "marginal", "joint", "steady", "stalled"):
            raise EvidenceError("checkpoint curriculum phase is invalid")
        vector_names = (
            "arm_frontier_indices",
            "arm_status",
            "arm_probe_indices",
            "arm_epochs",
            "last_selected_round",
        )
        if any(
            type(progress[name]) is not list
            or len(progress[name]) != len(ARM_KEYS)
            for name in vector_names
        ):
            raise EvidenceError("checkpoint curriculum arm vectors are not length 32")
        for name in (
            "arm_frontier_indices",
            "arm_probe_indices",
        ):
            for index, value in enumerate(progress[name]):
                _plain_int(
                    value,
                    "checkpoint progress {}[{}]".format(name, index),
                    minimum=0,
                    maximum=4,
                )
        for name in ("arm_epochs", "last_selected_round"):
            for index, value in enumerate(progress[name]):
                _plain_int(
                    value,
                    "checkpoint progress {}[{}]".format(name, index),
                    minimum=0,
                )
        if any(
            status not in ("pending", "probing", "decided", "disabled")
            for status in progress["arm_status"]
        ):
            raise EvidenceError("checkpoint curriculum arm status is invalid")
        selection_round = _plain_int(
            progress["selection_round"],
            "checkpoint progress selection_round",
            minimum=0,
        )
        selected_arm = progress["selected_arm_key"]
        if type(selected_arm) is not str or (
            selected_arm and selected_arm not in NO_MOVE_ARMS
        ):
            raise EvidenceError("checkpoint selected arm is not active no_move arm")
        for arm_index, arm in enumerate(ARM_KEYS):
            last_selected = progress["last_selected_round"][arm_index]
            if last_selected > selection_round:
                raise EvidenceError("checkpoint last-selected round is in the future")
            if arm not in NO_MOVE_ARMS:
                if (
                    progress["arm_status"][arm_index] != "disabled"
                    or progress["arm_frontier_indices"][arm_index] != 0
                    or progress["arm_probe_indices"][arm_index] != 0
                    or progress["arm_epochs"][arm_index] != 0
                    or last_selected != 0
                ):
                    raise EvidenceError(
                        "checkpoint no_move curriculum enabled base-travel state"
                    )
                continue
            covered = last_selected > 0
            age = (
                selection_round - last_selected
                if covered
                else selection_round + 1
            )
            coverage_count += int(covered)
            starvation_ages.append(age)
            starvation_rows.append(
                {
                    "action_id": action_id,
                    "action_uid": uid,
                    "profile_sha256": profile_sha,
                    "arm": arm,
                    "covered": covered,
                    "last_selected_round": (
                        last_selected if covered else None
                    ),
                    "selection_round": selection_round,
                    "age": age,
                }
            )
        if max(progress["last_selected_round"]) != selection_round:
            if not (selection_round == 0 and max(progress["last_selected_round"]) == 0):
                raise EvidenceError(
                    "checkpoint selection round lacks a matching selected-arm round"
                )
        if selected_arm:
            selected_index = ARM_KEYS.index(selected_arm)
            if progress["last_selected_round"][selected_index] != selection_round:
                raise EvidenceError("checkpoint current arm/selection round diverged")

        receipts = progress["formal_receipts"]
        if type(receipts) is not list:
            raise EvidenceError("checkpoint formal receipts are not a list")
        receipt_rounds: Dict[str, int] = {}
        for receipt_index, receipt in enumerate(receipts):
            evidence, match = _curriculum_receipt_match_document(
                receipt,
                label="checkpoint progress[{}].formal_receipts[{}]".format(
                    slot, receipt_index
                ),
            )
            if (
                evidence["key"] != profile
                or evidence["arm_catalog_sha256"] != arm_catalog_sha
                or evidence["scheduler_contract_sha256"] != scheduler_sha
                or evidence["sampler_sha256"] != curriculum["sampler_sha256"]
                or evidence["solver_sha256"] != curriculum["solver_sha256"]
                or evidence["policy_contract_sha256"]
                != curriculum["policy_contract_sha256"]
                or evidence["evidence_role"]
                not in ("frozen_canary", "frozen_heldout")
            ):
                raise EvidenceError("checkpoint formal receipt identity drifted")
            arm = evidence["selected_arm_key"]
            if arm:
                receipt_rounds[arm] = max(
                    receipt_rounds.get(arm, 0),
                    evidence["selection_round"],
                )
            observed_formal_matches.append(match)
        scheduler_receipts = progress["scheduler_receipts"]
        if type(scheduler_receipts) is not list:
            raise EvidenceError("checkpoint scheduler receipts are not a list")
        for receipt_index, receipt in enumerate(scheduler_receipts):
            scheduler = _exact_dict(
                receipt,
                ("evidence", "window_sha256", "attempts"),
                "checkpoint progress[{}].scheduler_receipts[{}]".format(
                    slot, receipt_index
                ),
            )
            synthetic = {
                "evidence": scheduler["evidence"],
                "window_sha256": scheduler["window_sha256"],
                "certified": False,
            }
            evidence, _match = _curriculum_receipt_match_document(
                synthetic,
                label="checkpoint progress[{}].scheduler_receipts[{}]".format(
                    slot, receipt_index
                ),
            )
            raw_scheduler_ledger = _scheduler_attempt_ledger(
                scheduler["attempts"],
                label=(
                    "checkpoint progress[{}].scheduler_receipts[{}]"
                    ".attempts"
                ).format(slot, receipt_index),
            )
            if (
                evidence["key"] != profile
                or evidence["evidence_role"] != "scheduler"
                or len(scheduler["attempts"]) != evidence["ledger"]["P"]
                or raw_scheduler_ledger != evidence["ledger"]
            ):
                raise EvidenceError("checkpoint scheduler receipt identity drifted")
            arm = evidence["selected_arm_key"]
            if arm:
                receipt_rounds[arm] = max(
                    receipt_rounds.get(arm, 0),
                    evidence["selection_round"],
                )
        if selected_arm:
            receipt_rounds[selected_arm] = max(
                receipt_rounds.get(selected_arm, 0), selection_round
            )
        active_arms = (
            NO_MOVE_ARMS if mobility_mode == "no_move" else ARM_KEYS
        )
        for arm in active_arms:
            observed_round = progress["last_selected_round"][ARM_KEYS.index(arm)]
            if receipt_rounds.get(arm, 0) != observed_round:
                raise EvidenceError(
                    "checkpoint last-selected round is not backed by "
                    "formal/scheduler/current state"
                )
        profile_rows.append(
            {
                "action_id": action_id,
                "action_uid": uid,
                "profile_sha256": profile_sha,
                "mobility_mode": mobility_mode,
                "phase": phase,
                "selection_round": selection_round,
                "selected_arm_key": selected_arm,
                "domain_epoch": _plain_int(
                    progress["domain_release_epoch"],
                    "checkpoint domain_release_epoch",
                    minimum=0,
                ),
                "formal_receipt_count": len(receipts),
                "scheduler_receipt_count": len(scheduler_receipts),
                "frontier_levels": {
                    arm: (0.0, 0.25, 0.5, 0.75, 1.0)[
                        progress["arm_frontier_indices"][arm_index]
                    ]
                    for arm_index, arm in enumerate(ARM_KEYS)
                },
                "candidate_levels": {
                    arm: (0.0, 0.25, 0.5, 0.75, 1.0)[
                        progress["arm_probe_indices"][arm_index]
                    ]
                    for arm_index, arm in enumerate(ARM_KEYS)
                },
                "frontier_physical_widths": _physical_width_rows(
                    action_bindings[slot],
                    [
                        (0.0, 0.25, 0.5, 0.75, 1.0)[index]
                        for index in progress["arm_frontier_indices"]
                    ],
                ),
                "candidate_physical_widths": _physical_width_rows(
                    action_bindings[slot],
                    [
                        (0.0, 0.25, 0.5, 0.75, 1.0)[index]
                        for index in progress["arm_probe_indices"]
                    ],
                ),
                "arm_status": {
                    arm: progress["arm_status"][arm_index]
                    for arm_index, arm in enumerate(ARM_KEYS)
                },
            }
        )

    expected_formal_matches: List[Dict[str, Any]] = []
    for record in formal_records:
        if type(record) is not dict:
            raise EvidenceError("formal record is not a plain object")
        for window in record["windows"]:
            expected_formal_matches.append(
                _formal_window_match_document(
                    record=record,
                    window=window,
                    arm_catalog_sha256=arm_catalog_sha,
                    scheduler_contract_sha256=scheduler_sha,
                )
            )
    expected_digests = sorted(
        canonical_sha256(row) for row in expected_formal_matches
    )
    observed_digests = sorted(
        canonical_sha256(row) for row in observed_formal_matches
    )
    expected_counts = Counter(expected_digests)
    observed_counts = Counter(observed_digests)
    stale_count = sum(
        abs(expected_counts[digest] - observed_counts[digest])
        for digest in set(expected_counts).union(observed_counts)
    )
    if stale_count:
        raise EvidenceError(
            "checkpoint curriculum formal receipts differ from accepted raw inbox"
        )
    starvation = {
        "definition": (
            "checkpoint curriculum selection_round - last_selected_round for "
            "every active contracted action x mobility-enabled arm; "
            "never-selected arms are "
            "explicit uncovered age selection_round+1"
        ),
        "source": "frozen checkpoint curriculum state",
        "sample_count": len(starvation_ages),
        "coverage_count": coverage_count,
        "uncovered_count": len(starvation_ages) - coverage_count,
        "max_age": max(starvation_ages),
        "p5": _percentile(starvation_ages, 0.05),
        "p50": _percentile(starvation_ages, 0.50),
        "p95": _percentile(starvation_ages, 0.95),
        "rows": starvation_rows,
    }
    return {
        "racket_exact_state_sha256": state["integrity_sha256"],
        "curriculum_state_sha256": curriculum_sha,
        "arm_catalog_sha256": arm_catalog_sha,
        "scheduler_contract_sha256": scheduler_sha,
        "profile_rows": profile_rows,
        "formal_receipt_count": len(observed_formal_matches),
        "accepted_inbox_window_count": len(expected_formal_matches),
        "domain_epoch_stale_count": stale_count,
        "counter_invariants_passed": (
            len(starvation_rows) == len(expected_uids) * len(NO_MOVE_ARMS)
            and stale_count == 0
        ),
        "starvation": starvation,
    }


def _trainer_ledger_evidence(
    *,
    log_path: Path,
    checkpoint: Mapping[str, Any],
    action_bindings: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
) -> Dict[str, Any]:
    snapshot = _snapshot_file(log_path, "trainer ledger log")
    try:
        text = snapshot["raw"].decode("utf-8")
    except UnicodeError as exc:
        raise EvidenceError("trainer ledger log is not UTF-8") from exc
    events: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            document = _strict_json_bytes(
                line.encode("utf-8"),
                "trainer log line {}".format(line_number),
            )
        except EvidenceError:
            if "action_ball_training_ledger" in line:
                raise EvidenceError(
                    "trainer ledger event line {} is malformed/non-finite".format(
                        line_number
                    )
                )
            continue
        if document.get("event") == "action_ball_training_ledger":
            events.append(document)
    if not events:
        raise EvidenceError("trainer log contains no action_ball_training_ledger")
    action_ids = [row["motion_id"] for row in action_bindings]
    action_uids = [row["action_uid"] for row in action_bindings]
    previous_step = -1
    normalized_terminal: Optional[Dict[str, Dict[str, int]]] = None
    for event_index, event in enumerate(events):
        step = _plain_int(
            event.get("step"),
            "trainer ledger event step",
            minimum=0,
        )
        if step <= previous_step:
            raise EvidenceError("trainer ledger steps are not strictly increasing")
        previous_step = step
        if (
            event.get("schema_version") != 1
            or event.get("manifest_sha256") != manifest_sha256
            or event.get("action_order") != action_ids
            or event.get("status")
            != "report_only_requires_frozen_checkpoint_evidence"
            or event.get("diagnostic_unauthorized") is not None
        ):
            raise EvidenceError("trainer ledger event identity is invalid")
        raw_ledger = event.get("ledger")
        raw_rejections = event.get("solver_rejections")
        raw_pool = event.get("pool")
        if (
            type(raw_ledger) is not dict
            or list(raw_ledger) != action_ids
            or type(raw_rejections) is not dict
            or set(raw_rejections) != {str(uid) for uid in action_uids}
            or type(raw_pool) is not dict
            or set(raw_pool) != {str(uid) for uid in action_uids}
        ):
            raise EvidenceError(
                "trainer ledger event contracted action shape is invalid"
            )
        normalized: Dict[str, Dict[str, int]] = {}
        for slot, (action_id, uid) in enumerate(zip(action_ids, action_uids)):
            row = _exact_dict(
                raw_ledger[action_id],
                ACTION_BALL_LEDGER_NAMES,
                "trainer ledger {} action {}".format(event_index, action_id),
            )
            values = {
                name: _plain_int(
                    row[name],
                    "trainer ledger {}.{}".format(action_id, name),
                    minimum=0,
                )
                for name in ACTION_BALL_LEDGER_NAMES
            }
            _validate_short_ledger(
                {**values, "NB": 0, "NB_F": 0},
                label="trainer ledger {}".format(action_id),
            )
            reasons = raw_rejections[str(uid)]
            if type(reasons) is not dict or any(
                type(reason) is not str
                or not reason
                or type(count) is not int
                or count < 0
                for reason, count in reasons.items()
            ):
                raise EvidenceError("trainer solver rejection reasons are invalid")
            if sum(reasons.values()) != values["P"] - values["A"]:
                raise EvidenceError(
                    "trainer solver rejection denominator differs from P-A"
                )
            pool = _exact_dict(
                raw_pool[str(uid)],
                (
                    "requests",
                    "refill_calls",
                    "proposed",
                    "admitted",
                    "issued",
                    "discarded",
                    "pending",
                ),
                "trainer pool {}".format(uid),
            )
            pool_values = {
                name: _plain_int(
                    pool[name],
                    "trainer pool {}.{}".format(uid, name),
                    minimum=0,
                )
                for name in pool
            }
            if (
                pool_values["proposed"] != values["P"]
                or pool_values["admitted"] != values["A"]
                or pool_values["issued"] != values["I"]
                or pool_values["issued"] != values["S"]
                or pool_values["requests"] != pool_values["issued"]
                or pool_values["proposed"] < pool_values["admitted"]
                or pool_values["admitted"]
                < pool_values["issued"] + pool_values["discarded"]
            ):
                raise EvidenceError("trainer pool/attempt ledger differs")
            del slot
            normalized[action_id] = values
        normalized_terminal = normalized
    assert normalized_terminal is not None
    state = _action_ball_racket_state(checkpoint)
    checkpoint_ledger = _exact_dict(
        state["ledger"],
        ACTION_BALL_LEDGER_NAMES,
        "checkpoint action-ball ledger",
    )
    for slot, action_id in enumerate(action_ids):
        for name in ACTION_BALL_LEDGER_NAMES:
            vector = checkpoint_ledger[name]
            if (
                type(vector) is not list
                or len(vector) != len(action_ids)
                or any(type(value) is not int or value < 0 for value in vector)
                or vector[slot] != normalized_terminal[action_id][name]
            ):
                raise EvidenceError(
                    "terminal trainer ledger differs from checkpoint exact state"
                )
    last_rollout_step = state["last_rollout_step"]
    if last_rollout_step != previous_step:
        raise EvidenceError(
            "terminal trainer ledger step differs from checkpoint exact state"
        )
    nonfinite_count = _count_nonfinite_tree(events)
    if nonfinite_count:
        raise EvidenceError("trainer ledger contains non-finite values")
    counter_violation_count = sum(
        int(
            not (
                row["P"] >= row["A"] >= row["I"] >= row["S"] >= row["C"]
                and row["C"]
                == sum(
                    row[name]
                    for name in (
                        "L",
                        "F",
                        "U_table",
                        "U_fall",
                        "U_collision",
                        "U_joint_qdes",
                        "U_joint_actual",
                    )
                )
            )
        )
        for row in normalized_terminal.values()
    )
    return {
        "log_sha256": snapshot["sha256"],
        "event_count": len(events),
        "terminal_step": previous_step,
        "terminal_event_sha256": canonical_sha256(events[-1]),
        "terminal_ledger": normalized_terminal,
        "nonfinite_count": nonfinite_count,
        "counter_violation_count": counter_violation_count,
        "counter_invariants_passed": counter_violation_count == 0,
    }


def _load_module(path: Path, expected_sha: str, name: str) -> Any:
    snapshot = _snapshot_file(path, name)
    if snapshot["sha256"] != expected_sha:
        raise EvidenceError("{} bytes differ from pinned SHA".format(name))
    module_name = "_action_ball_evidence_{}_{}".format(
        name.replace(" ", "_"), secrets.token_hex(6)
    )
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise EvidenceError("cannot construct module loader for {}".format(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_claim_document(
    claim_path: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    claim, snapshot = _snapshot_json(claim_path, "launch claim")
    row = _exact_dict(
        claim,
        (
            "schema_version",
            "kind",
            "launch_claim_sha256",
            "canonical_payload",
            "argv",
            "confirmation_claim_sha256",
        ),
        "launch claim",
    )
    if row["schema_version"] != 3 or row["kind"] != "action_ball_no_clobber_launch_claim_v3":
        raise EvidenceError("launch claim schema/kind is invalid")
    claim_sha = _sha256(row["launch_claim_sha256"], "launch claim SHA")
    if (
        canonical_sha256(row["canonical_payload"]) != claim_sha
        or row["confirmation_claim_sha256"] != claim_sha
        or type(row["argv"]) is not list
        or not row["argv"]
        or row["argv"][-1]
        != "++training_launch_claim_sha256={}".format(claim_sha)
    ):
        raise EvidenceError("launch claim canonical/self-confirmation binding is false")
    payload = row["canonical_payload"]
    if type(payload) is not dict:
        raise EvidenceError("launch claim canonical payload must be an object")
    namespace = payload.get("namespace")
    expected_path_override = (
        "++training_launch_claim_path={}/launch_claim.json".format(namespace)
    )
    argv_without_claim = payload.get("argv_without_launch_claim")
    if (
        type(namespace) is not str
        or not namespace
        or type(argv_without_claim) is not list
        or row["argv"]
        != [
            *argv_without_claim,
            "++training_launch_claim_sha256={}".format(claim_sha),
        ]
        or argv_without_claim.count(expected_path_override) != 1
        or sum(
            type(item) is str
            and item.startswith("++training_launch_claim_path=")
            for item in argv_without_claim
        )
        != 1
        or any(
            type(item) is str
            and item.startswith("++training_launch_claim_sha256=")
            for item in argv_without_claim
        )
    ):
        raise EvidenceError("launch claim argv does not bind its exact claim path")
    return row, payload, snapshot


def _validate_supervisor_terminal(
    document: Dict[str, Any],
    *,
    claim_sha256: str,
    source_commit: str,
    stage: str,
    namespace: Path,
) -> None:
    row = _exact_dict(
        document,
        (
            "schema_version",
            "kind",
            "terminal_utc",
            "status",
            "claim_sha256",
            "source_commit_sha",
            "stage",
            "namespace",
            "trainer_returncode",
            "evaluator_returncode",
            "cleanup",
            "processes",
        ),
        "supervisor terminal",
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != "action_ball_stage_supervisor_terminal"
        or row["status"] != "completed"
        or row["claim_sha256"] != claim_sha256
        or row["source_commit_sha"] != source_commit
        or row["stage"] != stage
        or row["namespace"] != str(namespace)
        or row["trainer_returncode"] != 0
        or row["evaluator_returncode"] not in (0, -15)
        or type(row["cleanup"]) is not list
        or not row["cleanup"]
        or any(
            type(item) is not dict or item.get("forced_kill") is not False
            for item in row["cleanup"]
        )
    ):
        raise EvidenceError("supervisor terminal does not prove clean stage completion")


def _trainer_output_from_log(
    *,
    checkout: Path,
    namespace: Path,
    experiment_name: str,
) -> Tuple[Dict[str, Any], Path]:
    log_path = namespace / "train.log"
    log_snapshot = _snapshot_file(log_path, "trainer log")
    try:
        text = log_snapshot["raw"].decode("utf-8")
    except UnicodeError as exc:
        raise EvidenceError("trainer log is not UTF-8") from exc
    prefix = (
        "[INFO] Task: HOPE-PingPong-ActionBall-AgibotA3-v0 | "
        "experiment: {} | log: ".format(experiment_name)
    )
    outputs = [
        line[len(prefix) :]
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(outputs) != 1:
        raise EvidenceError("trainer log must name exactly one RSL output")
    rsl_log_dir = _normalized_absolute(outputs[0], "trainer RSL output")
    expected_parent = (
        checkout
        / "hope_training/whole_body_tracking/logs/rsl_rl"
        / experiment_name
    )
    if rsl_log_dir.parent != expected_parent:
        raise EvidenceError("trainer output escaped dedicated experiment root")
    suffix = "_" + namespace.name
    if not rsl_log_dir.name.endswith(suffix):
        raise EvidenceError("trainer output basename does not bind namespace")
    timestamp = rsl_log_dir.name[: -len(suffix)]
    if RSL_TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise EvidenceError("trainer output timestamp is not canonical")
    try:
        parsed = _datetime.datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
    except ValueError as exc:
        raise EvidenceError("trainer output timestamp is not a real date") from exc
    if parsed.strftime("%Y-%m-%d_%H-%M-%S") != timestamp:
        raise EvidenceError("trainer output timestamp round-trip failed")
    contract_path = rsl_log_dir / "params/training_contract.json"
    reward_path = rsl_log_dir / "params/effective_reward_recipe.json"
    contract = _snapshot_file(contract_path, "training contract")
    reward = _snapshot_file(reward_path, "effective Reward recipe")
    row = {
        "rsl_log_dir": str(rsl_log_dir),
        "timestamp_prefix": timestamp,
        "run_name": namespace.name,
        "launcher_log_sha256": log_snapshot["sha256"],
        "training_contract": {
            "path": "params/training_contract.json",
            "sha256": contract["sha256"],
        },
        "effective_reward_recipe": {
            "path": "params/effective_reward_recipe.json",
            "sha256": reward["sha256"],
        },
    }
    return row, rsl_log_dir


def _select_final_checkpoint(rsl_log_dir: Path, max_iterations: int) -> Path:
    rows: List[Tuple[int, Path]] = []
    for path in rsl_log_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is not None:
            rows.append((int(match.group(1)), path))
    if not rows:
        raise EvidenceError("trainer output contains no model_<N>.pt checkpoint")
    iteration, path = max(rows)
    if iteration < max_iterations - 1 or iteration > max_iterations:
        raise EvidenceError(
            "final checkpoint iteration is outside the completed stage budget"
        )
    return path


def _reward_audit(
    *,
    checkout: Path,
    source_commit: str,
    claim_payload: Mapping[str, Any],
    rsl_log_dir: Path,
    namespace: Path,
) -> Tuple[Dict[str, Any], str]:
    source = _committed_file(
        checkout, source_commit, AUDIT_REWARD_SOURCE, "Reward audit source"
    )
    module = _load_module(
        source["path"], source["sha256"], "Reward audit source"
    )
    report = module.audit_reward_run(
        recipe_path=rsl_log_dir / "params/effective_reward_recipe.json",
        event_paths=[namespace / "train.log"],
        manifest_path=checkout / claim_payload["manifest"]["path"],
        run_dir=rsl_log_dir,
    )
    if type(report) is not dict or report.get("status") != "PASS":
        failures = report.get("failures") if type(report) is dict else None
        raise EvidenceError(
            "Reward activation/negative-semantics audit failed closed: {!r}".format(
                failures
            )
        )
    declared = _sha256(report.get("report_sha256"), "Reward audit report SHA")
    unsigned = dict(report)
    del unsigned["report_sha256"]
    if canonical_sha256(unsigned) != declared:
        raise EvidenceError("Reward audit report SHA is false")
    return report, source["sha256"]


def _formal_evaluation_records(
    *,
    checkout: Path,
    source_commit: str,
    claim_sha256: str,
    claim_payload: Mapping[str, Any],
    action_bindings: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    runtime_shas = claim_payload.get("runtime_code_sha256")
    if type(runtime_shas) is not dict:
        raise EvidenceError("claim lacks runtime_code_sha256")
    inbox_sha = _sha256(runtime_shas.get(INBOX_SOURCE), "inbox runtime SHA")
    inbox_source = _committed_file(
        checkout, source_commit, INBOX_SOURCE, "evaluation inbox source"
    )
    if inbox_source["sha256"] != inbox_sha:
        raise EvidenceError("claim inbox runtime SHA differs from exact commit")
    inbox_module = _load_module(
        inbox_source["path"], inbox_sha, "evaluation inbox source"
    )
    evaluation = claim_payload.get("frozen_evaluation_runtime")
    if type(evaluation) is not dict:
        raise EvidenceError("claim lacks frozen_evaluation_runtime")
    root = _normalized_absolute(evaluation.get("inbox_root"), "evaluation inbox root")
    owner = evaluation.get("owner_id")
    run_id = evaluation.get("run_id")
    queue = inbox_module.EvaluationInbox(root)
    history = queue._validated_history(owner, run_id)
    manifest_uids = [row["action_uid"] for row in action_bindings]
    manifest_motions = [row["motion_sha256"] for row in action_bindings]
    formal: List[Dict[str, Any]] = []
    artifact_rows: List[Dict[str, Any]] = []
    for sequence, request_document in enumerate(history):
        request = inbox_module.validate_request_document(
            request_document,
            expected_owner_id=owner,
            expected_run_id=run_id,
            expected_request_seq=sequence,
        )
        bindings = request["bindings"]
        if (
            bindings["training_launch_claim_sha256"] != claim_sha256
            or bindings["manifest_sha256"] != claim_payload["manifest"]["sha256"]
            or bindings["policy_contract_sha256"]
            != claim_payload["policy_contract_sha256"]
            or bindings["action_order"] != manifest_uids
            or [
                row["motion"]["sha256"] for row in bindings["actions"]
            ]
            != manifest_motions
        ):
            raise EvidenceError("formal request bindings differ from launch claim")
        evidence_document = queue.load_evidence(owner, run_id, sequence)
        ack_document = queue.load_ack(owner, run_id, sequence)
        evidence = inbox_module.validate_evidence_document(
            evidence_document,
            request_document=request_document,
        )
        ack = inbox_module.validate_ack_document(
            ack_document,
            request_document=request_document,
            evidence_document=evidence_document,
        )
        request_path = queue.request_path(owner, run_id, sequence)
        evidence_path = queue.evidence_path(owner, run_id, sequence)
        ack_path = queue.ack_path(owner, run_id, sequence)
        artifact_rows.append(
            {
                "request_seq": sequence,
                "request_sha256": _snapshot_file(
                    request_path, "formal request"
                )["sha256"],
                "evidence_sha256": _snapshot_file(
                    evidence_path, "formal evidence"
                )["sha256"],
                "ack_sha256": _snapshot_file(ack_path, "formal ACK")["sha256"],
            }
        )
        roles = [window["role"] for window in request["windows"]]
        if roles == ["frozen_canary", "frozen_heldout"]:
            target = request["target"]
            bindings = request["bindings"]
            formal.append(
                {
                    "request_seq": sequence,
                    "action_uid": target["action_uid"],
                    "profile_sha256": target["profile_sha256"],
                    "mobility_mode": target["mobility_mode"],
                    "domain_epoch": target["domain_epoch"],
                    "stratum": target["stratum"],
                    "selected_arm_key": target["selected_arm_key"],
                    "selection_round": target["selection_round"],
                    "arm_levels": list(target["arm_levels"]),
                    "rho": target["rho"],
                    "policy_checkpoint_sha256": bindings["checkpoint"][
                        "sha256"
                    ],
                    "policy_generation": bindings["policy_generation"],
                    "sampler_sha256": bindings["sampler_sha256"],
                    "solver_sha256": bindings["solver_sha256"],
                    "policy_contract_sha256": bindings[
                        "policy_contract_sha256"
                    ],
                    "accepted_ack": {
                        "decision": ack["decision"],
                        "consumer_code_sha256": ack[
                            "consumer_code_sha256"
                        ],
                        "consumer_state_sha256": ack[
                            "consumer_state_sha256"
                        ],
                        "consumer_checkpoint": dict(
                            ack["consumer_checkpoint"]
                        ),
                    },
                    "windows": [
                        {
                            "role": window["allocation"]["role"],
                            "allocation": dict(window["allocation"]),
                            "ledger": inbox_module._derive_ledger(
                                window["attempts"]
                            ),
                            "raw_attempt_count": len(window["attempts"]),
                            "raw_attempts_sha256": canonical_sha256(
                                window["attempts"]
                            ),
                            "raw_nonfinite_count": _count_nonfinite_tree(
                                window["attempts"]
                            ),
                            "attempt_receipt_root_sha256": window[
                                "attempt_receipt_root_sha256"
                            ],
                            "ordered_sample_receipt_sha256": [
                                row["sample_receipt_sha256"]
                                for row in window["attempts"]
                            ],
                            "ordered_birth_receipt_sha256": [
                                row["birth_receipt_sha256"]
                                for row in window["attempts"]
                            ],
                            "raw_attempts": [
                                dict(row) for row in window["attempts"]
                            ],
                        }
                        for window in evidence["windows"]
                    ],
                }
            )
    return formal, artifact_rows, inbox_sha


def attest_stage(
    *,
    claim_path: str,
    authority_path: str,
    private_key_path: str,
    output_path: str,
    torch_module: Optional[Any] = None,
    checkpoint_loader: Optional[Callable[[bytes], Any]] = None,
) -> Dict[str, Any]:
    claim, payload, _claim_snapshot = _validate_claim_document(claim_path)
    checkout, source_commit = _verify_clean_checkout(
        payload.get("source_checkout"), payload.get("source_commit_sha")
    )
    action_set_contract = _load_action_set_contract(
        checkout, source_commit, payload.get("launch_profile")
    )
    if payload.get("action_set_contract") != action_set_contract:
        raise EvidenceError(
            "claim action-set contract differs from committed registry"
        )
    private_key, _private_snapshot = _load_private_key(private_key_path)
    _authority, authority_snapshot, _public = _validate_authority(
        authority_path,
        checkout=checkout,
        source_commit=source_commit,
        private_key=private_key,
    )
    namespace = _normalized_absolute(payload.get("namespace"), "stage namespace")
    if Path(claim_path) != namespace / "launch_claim.json":
        raise EvidenceError("claim path is not the stage namespace launch_claim.json")
    stage = payload.get("stage")
    if stage not in ("smoke", "canary", "long"):
        raise EvidenceError("claim stage is invalid")
    terminal, terminal_snapshot = _snapshot_json(
        namespace / "supervisor_terminal.json", "supervisor terminal"
    )
    if os.path.lexists(namespace / "supervisor_failed.json"):
        raise EvidenceError("supervisor failure receipt exists")
    _validate_supervisor_terminal(
        terminal,
        claim_sha256=claim["launch_claim_sha256"],
        source_commit=source_commit,
        stage=stage,
        namespace=namespace,
    )
    launcher_source = _committed_file(
        checkout, source_commit, LAUNCHER_SOURCE, "launcher source"
    )
    launcher = _load_module(
        launcher_source["path"], launcher_source["sha256"], "launcher source"
    )
    trainer_output, rsl_log_dir = _trainer_output_from_log(
        checkout=checkout,
        namespace=namespace,
        experiment_name=action_set_contract["experiment_name"],
    )
    recipe = payload["training_recipe"]
    launcher._validate_trainer_output(
        trainer_output,
        checkout=checkout,
        completed_namespace=namespace,
        expected_run_name=namespace.name,
        expected_experiment_name=action_set_contract["experiment_name"],
        expected_claim_sha=claim["launch_claim_sha256"],
        expected_ground_plant_sha=recipe["ground_plant_contract_sha256"],
        expected_reward_sha=recipe["effective_reward_recipe_sha256"],
        expected_ppo_sha=recipe["ppo_recipe_sha256"],
    )
    reward_report, reward_audit_source_sha = _reward_audit(
        checkout=checkout,
        source_commit=source_commit,
        claim_payload=payload,
        rsl_log_dir=rsl_log_dir,
        namespace=namespace,
    )
    contract_snapshot = _snapshot_file(
        rsl_log_dir / "params/training_contract.json", "training contract"
    )
    checkpoint_path = _select_final_checkpoint(
        rsl_log_dir, payload["stage_budget"]["max_iterations"]
    )
    checkpoint_audit, checkpoint_object = _load_checkpoint(
        checkpoint_path,
        training_contract_sha256=contract_snapshot["sha256"],
        launch_claim_sha256=claim["launch_claim_sha256"],
        torch_module=torch_module,
        checkpoint_loader=checkpoint_loader,
    )
    runtime_bootstrap = _checkpoint_runtime_bootstrap_evidence(
        checkpoint=checkpoint_object,
        checkpoint_path=checkpoint_path,
        checkout=checkout,
        source_commit=source_commit,
        claim_path=Path(claim_path),
        claim_sha256=claim["launch_claim_sha256"],
        training_contract_path=(
            rsl_log_dir / "params/training_contract.json"
        ),
        training_contract_sha256=contract_snapshot["sha256"],
        claim_payload=payload,
    )
    exact_resume = _exact_resume_verification_evidence(
        namespace=namespace,
        checkpoint_path=checkpoint_path,
        checkpoint_audit=checkpoint_audit,
        runtime_bootstrap=runtime_bootstrap,
        checkout=checkout,
        source_commit=source_commit,
        claim_sha256=claim["launch_claim_sha256"],
        stage=stage,
        claim_payload=payload,
    )
    bindings, _manifest, _manifest_snapshot = _manifest_bindings(
        manifest_path=checkout / payload["manifest"]["path"],
        checkout=checkout,
        source_commit=source_commit,
        action_set_contract=action_set_contract,
    )
    formal, eval_artifacts, inbox_source_sha = _formal_evaluation_records(
        checkout=checkout,
        source_commit=source_commit,
        claim_sha256=claim["launch_claim_sha256"],
        claim_payload=payload,
        action_bindings=bindings,
    )
    evaluation = payload["frozen_evaluation_runtime"]
    metrics, detailed, starvation = derive_stage_metrics(
        stage=stage,
        max_iterations=payload["stage_budget"]["max_iterations"],
        interval_updates=evaluation["interval_updates"],
        action_bindings=bindings,
        formal_records=formal,
    )
    curriculum_evidence = _derive_checkpoint_curriculum_evidence(
        checkpoint=checkpoint_object,
        action_bindings=bindings,
        formal_records=formal,
        manifest_sha256=payload["manifest"]["sha256"],
    )
    trainer_ledger = _trainer_ledger_evidence(
        log_path=namespace / "train.log",
        checkpoint=checkpoint_object,
        action_bindings=bindings,
        manifest_sha256=payload["manifest"]["sha256"],
    )
    request_schedule_diagnostic = starvation
    starvation = curriculum_evidence["starvation"]
    metrics["nan_count"] += (
        checkpoint_audit["nonfinite_floating_elements"]
        + trainer_ledger["nonfinite_count"]
    )
    metrics["counter_violation_count"] += trainer_ledger[
        "counter_violation_count"
    ]
    metrics["domain_epoch_stale_count"] = curriculum_evidence[
        "domain_epoch_stale_count"
    ]
    metrics["curriculum_counter_invariants_passed"] = bool(
        metrics["curriculum_counter_invariants_passed"]
        and curriculum_evidence["counter_invariants_passed"]
        and trainer_ledger["counter_invariants_passed"]
        and metrics["nan_count"] == 0
        and metrics["counter_violation_count"] == 0
        and metrics["domain_epoch_stale_count"] == 0
    )
    detailed["accepted_ack_implies_domain_epoch_stale_count"] = (
        curriculum_evidence["domain_epoch_stale_count"]
    )
    producer_source = _committed_file(
        checkout, source_commit, STAGE_EVIDENCE_SOURCE, "stage evidence source"
    )
    evidence_document = {
        "schema_version": 1,
        "kind": "action_ball_stage_evidence_v1",
        "status": "passed",
        "source_commit_sha": source_commit,
        "stage": stage,
        "namespace": str(namespace),
        "launch_claim_sha256": claim["launch_claim_sha256"],
        "source_code": {
            "producer_sha256": producer_source["sha256"],
            "launcher_sha256": launcher_source["sha256"],
            "evaluation_inbox_sha256": inbox_source_sha,
            "reward_audit_sha256": reward_audit_source_sha,
        },
        "supervisor_terminal": {
            "path": "supervisor_terminal.json",
            "sha256": terminal_snapshot["sha256"],
        },
        "evaluation_artifacts": eval_artifacts,
        "evaluation": detailed,
        "request_schedule_diagnostic": request_schedule_diagnostic,
        "starvation": starvation,
        "checkpoint_curriculum": curriculum_evidence,
        "trainer_ledger": trainer_ledger,
        "reward_activation_audit": reward_report,
        "checkpoint_audit": checkpoint_audit,
        "runtime_bootstrap": runtime_bootstrap,
        "exact_resume_verification": exact_resume,
        "aggregate_metrics": metrics,
    }
    evidence_document["content_sha256"] = canonical_sha256(evidence_document)
    evidence_path = namespace / "stage_evidence.json"
    _publish_exclusive_json(str(evidence_path), evidence_document)
    evidence_snapshot = _snapshot_file(evidence_path, "published stage evidence")

    row = {
        "schema_version": 1,
        "kind": "action_ball_stage_result",
        "status": "passed",
        "completed_stage": stage,
        "launch_profile": payload["launch_profile"],
        "action_set_contract_sha256": action_set_contract[
            "contract_sha256"
        ],
        "source_commit_sha": source_commit,
        "ordered_action_ids": payload["ordered_action_ids"],
        "manifest_sha256": payload["manifest"]["sha256"],
        "prototype_sha256": payload["prototype"]["sha256"],
        "motion_admission_receipt_sha256": payload[
            "motion_admission_receipt"
        ]["file_sha256"],
        "evaluator_launch_receipt_sha256": payload[
            "evaluator_launch_receipt"
        ]["file_sha256"],
        "sidecar_launch_receipt_sha256": payload[
            "sidecar_launch_receipt"
        ]["file_sha256"],
        "drain_reset_launch_receipt_sha256": payload[
            "drain_reset_launch_receipt"
        ]["file_sha256"],
        "policy_contract_sha256": payload["policy_contract_sha256"],
        "fitted_ball_profile_pins_sha256": payload[
            "fitted_ball_profile_pins_sha256"
        ],
        "fitted_ball_launch_trust_spec_sha256": payload[
            "fitted_ball_launch_trust_spec_sha256"
        ],
        "fitted_ball_launch_trust_root_sha256": payload[
            "fitted_ball_launch_trust_root_sha256"
        ],
        "fitted_ball_gate_receipt_sha256": payload[
            "fitted_ball_gate_receipt_sha256"
        ],
        "isaac_table_smoke_receipt_sha256": payload[
            "isaac_table_smoke_receipt_sha256"
        ],
        "prelaunch_safety_attestation_sha256": payload[
            "prelaunch_safety_attestation_sha256"
        ],
        "stage_evaluator_authority_sha256": authority_snapshot["sha256"],
        "namespace": str(namespace),
        "launch_claim_sha256": claim["launch_claim_sha256"],
        "stage_budget": payload["stage_budget"],
        "training_recipe_sha256": canonical_sha256(payload["training_recipe"]),
        "isaac_python_runtime_sha256": canonical_sha256(
            payload["isaac_python_runtime"]
        ),
        "frozen_evaluation_runtime_sha256": canonical_sha256(
            payload["frozen_evaluation_runtime"]
        ),
        "gpu_roles_sha256": canonical_sha256(payload["gpus"]),
        "isolated_training_entrypoint_sha256": canonical_sha256(
            payload["isolated_training_entrypoint"]
        ),
        "trainer_output": trainer_output,
        "metrics_evidence": {
            "path": evidence_path.name,
            "sha256": evidence_snapshot["sha256"],
        },
        "checkpoint": {
            "path": checkpoint_audit["path"],
            "sha256": checkpoint_audit["sha256"],
            "finite": checkpoint_audit["finite"],
            "exact_resume_passed": exact_resume["exact_resume_passed"],
        },
        "metrics": metrics,
    }
    envelope = _signed_envelope("action_ball_signed_stage_result", row, private_key)
    _publish_exclusive_json(output_path, envelope)
    return envelope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mint = sub.add_parser("mint-authority")
    mint.add_argument("--checkout", required=True)
    mint.add_argument("--source-commit", required=True)
    mint.add_argument("--private-key", required=True)
    mint.add_argument("--evaluator-id", required=True)
    mint.add_argument("--out", required=True)

    prelaunch = sub.add_parser("attest-prelaunch")
    prelaunch.add_argument("--checkout", required=True)
    prelaunch.add_argument("--source-commit", required=True)
    prelaunch.add_argument("--launch-profile", required=True)
    prelaunch.add_argument("--manifest", required=True)
    prelaunch.add_argument("--profile-pins", required=True)
    prelaunch.add_argument("--launch-trust-spec", required=True)
    prelaunch.add_argument("--launch-trust-root", required=True)
    prelaunch.add_argument("--fitted-gate", required=True)
    prelaunch.add_argument("--isaac-table-smoke", required=True)
    prelaunch.add_argument("--authority", required=True)
    prelaunch.add_argument("--private-key", required=True)
    prelaunch.add_argument("--out", required=True)

    stage = sub.add_parser("attest-stage")
    stage.add_argument("--claim", required=True)
    stage.add_argument("--authority", required=True)
    stage.add_argument("--private-key", required=True)
    stage.add_argument("--out", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "mint-authority":
            mint_authority(
                checkout=args.checkout,
                source_commit=args.source_commit,
                private_key_path=args.private_key,
                evaluator_id=args.evaluator_id,
                output_path=args.out,
            )
        elif args.command == "attest-prelaunch":
            attest_prelaunch(
                checkout=args.checkout,
                source_commit=args.source_commit,
                launch_profile=args.launch_profile,
                manifest_path=args.manifest,
                profile_pins_path=args.profile_pins,
                launch_trust_spec_path=args.launch_trust_spec,
                launch_trust_root_path=args.launch_trust_root,
                fitted_gate_path=args.fitted_gate,
                table_smoke_path=args.isaac_table_smoke,
                authority_path=args.authority,
                private_key_path=args.private_key,
                output_path=args.out,
            )
        else:
            attest_stage(
                claim_path=args.claim,
                authority_path=args.authority,
                private_key_path=args.private_key,
                output_path=args.out,
            )
    except Exception as exc:
        print(
            "ACTION_BALL_STAGE_EVIDENCE_REFUSED: {}: {}".format(
                type(exc).__name__, exc
            ),
            file=sys.stderr,
        )
        return 2
    print("ACTION_BALL_STAGE_EVIDENCE_PASS command={}".format(args.command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
