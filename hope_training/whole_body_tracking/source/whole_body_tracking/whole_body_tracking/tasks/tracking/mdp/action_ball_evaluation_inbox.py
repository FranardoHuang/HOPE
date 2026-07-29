"""Append-only transport for action-ball frozen evaluation.

The trainer and the frozen evaluator deliberately do not share mutable Python
objects.  They communicate through three immutable JSON records:

``request -> evidence -> acknowledgement``.

Every record is content addressed, written through a durable temporary file,
and installed without replacing an existing name.  The protocol is strict on
purpose: duplicate JSON keys, non-finite numbers, partial writes, sequence
gaps, replayed identities, overlapping evaluator allocations, and unpinned
sidecar code all fail closed.

This module is dependency-light so the inbox can be audited and tested on a
CPU-only host.  It does not evaluate a policy and it cannot authorize a
curriculum release by itself.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
SCHEDULER_PROPOSALS = 100
CANARY_PROPOSALS = 320
HELDOUT_PROPOSALS = 960
CANARY_SAFE_CLOSED_MIN = 256
HELDOUT_SAFE_CLOSED_MIN = 768
MAX_JSON_BYTES = 128 * 1024 * 1024
INT64_MAX = (1 << 63) - 1

REQUEST_KIND = "action_ball_frozen_eval_request"
EVIDENCE_KIND = "action_ball_frozen_eval_evidence"
ACK_KIND = "action_ball_frozen_eval_ack"
LAUNCH_KIND = "action_ball_frozen_eval_sidecar_launch"

TERMINAL_OUTCOMES = (
    "legal_return",
    "safe_nonreturn",
    "table_hit",
    "fall",
    "collision",
    "joint_qdes_limit",
    "joint_actual_limit",
)

POLICY_EVALUATION_CONTRACT = {
    "actor_mode": "eval",
    "critic_mode": "eval",
    "action_selection": "deterministic_mean",
    "torch_inference_mode": True,
    "stochastic_action_sampling": False,
    "action_noise": False,
    "recurrent_state": "reset_at_each_birth",
}

RESOLVED_EVALUATION_RECIPE_CONTRACT = {
    "schema_version": 1,
    "kind": "action_ball_frozen_eval_resolved_recipe",
    "inputs": (
        "exact training_contract.json, env.pkl, agent.pkl, checkpoint, "
        "ordered motion bytes, and trainer runtime-identity receipts"
    ),
    "construction": (
        "load the saved env/agent config without Hydra defaults; preserve "
        "the checkpoint environment count required by exact resume, map each "
        "fixed window through deterministic evaluator-hook batches, and only "
        "replace the sim/runner device with the evaluator-owned device"
    ),
    "batching": (
        "preserve proposal order, denominators, and receipts while mapping "
        "each fixed window through batches no larger than the checkpoint "
        "environment count; unused environments cannot emit evidence"
    ),
    "policy": POLICY_EVALUATION_CONTRACT,
    "action_identity": (
        "freeze target action UID before true reset; the solver may not "
        "change the action"
    ),
    "domain": (
        "install the request target domain and exact 20/60/20 "
        "center/interior/selected-frontier allocation"
    ),
    "outcomes": (
        "raw code-owned legal-return/table/fall/collision/joint-hard/"
        "infrastructure signals; rollout reward is not evidence"
    ),
}

RUNTIME_IDENTITY_CONTRACT = {
    "schema_version": 1,
    "kind": "action_ball_frozen_eval_runtime_identity",
    "interpreter": (
        "absolute executable path, executable bytes, implementation, and "
        "full version"
    ),
    "packages": (
        "torch/CUDA build, Isaac Lab, Isaac Lab RL, RSL-RL, Gymnasium, "
        "NumPy, and Python package versions"
    ),
    "source": (
        "absolute repository root, explicit Git object format, exact HEAD "
        "object ID, detached, and clean"
    ),
    "launch": "exact training launch-claim SHA-256",
    "device_ordinal": (
        "excluded from equality so trainer physical GPU0 and evaluator "
        "physical GPU1 may each be exposed locally as cuda:0"
    ),
}

FORMAL_ISAAC_BACKEND_CONTRACT = {
    "schema_version": 1,
    "kind": "action_ball_frozen_eval_formal_isaac_backend",
    "task_id": "HOPE-PingPong-ActionBall-AgibotA3-v0",
    "runtime": (
        "one independent headless Isaac process on the evaluator-owned CUDA "
        "device; one process may serve consecutive arbitrary action UID/"
        "profile requests without restarting Kit"
    ),
    "reconstruction": (
        "strictly verify the request-bound runtime-bootstrap receipt and its "
        "location-free lineage, then load its env.pkl, agent.pkl, runtime "
        "identity, training contract, checkpoint, and ordered motion bytes"
    ),
    "policy": POLICY_EVALUATION_CONTRACT,
    "proposal_hook": (
        "RacketTargetCommand.action_ball_frozen_evaluator_execute_v1 owns "
        "exact one-proposal sampling, solver disposition, task install, raw "
        "sensor closure, and proposal-denominator preservation"
    ),
    "batching": (
        "request target is arbitrary action_uid/profile; implementation may "
        "map one window through multiple environment batches but may not "
        "redraw or hide proposals"
    ),
    "evidence": (
        "only the runtime hook's raw proposal/solver/install/start/terminal "
        "transcript may become formal evidence"
    ),
}

SIDECAR_HEARTBEAT_CONTRACT = {
    "schema_version": 1,
    "heartbeat_interval_seconds": 5.0,
    "heartbeat_stale_after_seconds": 120.0,
    "request_deadline_seconds": 7200.0,
}

WINDOW_CONTRACT = {
    "optional_stopping": False,
    "scheduler_proposals": SCHEDULER_PROPOSALS,
    "canary_proposals": CANARY_PROPOSALS,
    "canary_safe_closed_min": CANARY_SAFE_CLOSED_MIN,
    "heldout_proposals": HELDOUT_PROPOSALS,
    "heldout_safe_closed_min": HELDOUT_SAFE_CLOSED_MIN,
    "allocation": (
        "authority supplied disjoint contiguous seed/sample/birth ranges"
    ),
    "sampling_mixture": {
        "center": 0.20,
        "interior": 0.60,
        "frontier": 0.20,
    },
}

_PROTOCOL_DOCUMENT = {
    "schema_version": SCHEMA_VERSION,
    "kind": "action_ball_frozen_eval_inbox_contract",
    "transport": "append-only request/evidence/ack",
    "identity": "owner_id/run_id/request_seq",
    "publication": "temp-write/fsync/atomic-no-replace/fsync-directory",
    "ack_barrier": (
        "accepted evidence ACK binds the exact consumer resume-state SHA "
        "and the already-persisted no-clobber checkpoint bytes"
    ),
    "request_binding": (
        "checkpoint/training-contract/env-pickle/agent-pickle/runtime-identity/"
        "runtime-bootstrap-receipt/runtime-bootstrap-lineage/"
        "launch-claim/policy-generation/policy-state/actor-normalizer/"
        "critic-normalizer/PPO/policy-eval/"
        "ordered-action-UID-motion-bytes/manifest/sampler/solver/physics/"
        "reward/curriculum"
    ),
    "windows": WINDOW_CONTRACT,
    "accounting": (
        "physics-invalid and solver-rejected are disjoint proposal outcomes; "
        "neither is a policy return failure"
    ),
    "trust": "exact sidecar code SHA and launch SHA are code-pinned",
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvaluationInboxError(RuntimeError):
    """The frozen-evaluation transport failed closed."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationInboxError(
            "value is not finite canonical JSON data"
        ) from exc
    return (encoded + "\n").encode("ascii")


def canonical_sha256(value: object) -> str:
    """Return the SHA256 of canonical JSON without its transport newline."""

    return hashlib.sha256(_canonical_json_bytes(value)[:-1]).hexdigest()


EVALUATION_INBOX_CONTRACT_SHA256 = canonical_sha256(_PROTOCOL_DOCUMENT)
POLICY_EVALUATION_CONTRACT_SHA256 = canonical_sha256(
    POLICY_EVALUATION_CONTRACT
)
RESOLVED_EVALUATION_RECIPE_CONTRACT_SHA256 = canonical_sha256(
    RESOLVED_EVALUATION_RECIPE_CONTRACT
)
RUNTIME_IDENTITY_CONTRACT_SHA256 = canonical_sha256(
    RUNTIME_IDENTITY_CONTRACT
)
FORMAL_ISAAC_BACKEND_CONTRACT_SHA256 = canonical_sha256(
    FORMAL_ISAAC_BACKEND_CONTRACT
)

# Reviewed production pins are deliberately absent on this branch.  Tests may
# replace these module globals in-process; the CLI offers no flag that can add
# a caller-provided trust entry.
TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_CODE_SHA256 = frozenset()
TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_LAUNCH_SHA256 = frozenset()


def _exact_dict(
    value: object,
    keys: Sequence[str],
    *,
    label: str,
) -> Dict[str, object]:
    if type(value) is not dict:
        raise EvaluationInboxError("{} must be a JSON object".format(label))
    wanted = set(keys)
    actual = set(value)
    if actual != wanted:
        raise EvaluationInboxError(
            "{} has invalid keys (missing={}, unknown={})".format(
                label,
                sorted(wanted - actual),
                sorted(actual - wanted),
            )
        )
    return value


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int = INT64_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise EvaluationInboxError(
            "{} must be a plain integer in [{}, {}]".format(
                label, minimum, maximum
            )
        )
    return value


def _finite_number(
    value: object,
    *,
    label: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if type(value) not in (int, float):
        raise EvaluationInboxError("{} must be a finite number".format(label))
    converted = float(value)
    if not math.isfinite(converted):
        raise EvaluationInboxError("{} must be finite".format(label))
    if minimum is not None and converted < minimum:
        raise EvaluationInboxError(
            "{} must be >= {}".format(label, minimum)
        )
    if maximum is not None and converted > maximum:
        raise EvaluationInboxError(
            "{} must be <= {}".format(label, maximum)
        )
    return converted


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise EvaluationInboxError(
            "{} must be 64 lowercase hexadecimal characters".format(label)
        )
    return value


def _identifier(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or _IDENTIFIER_RE.fullmatch(value) is None
        or value in (".", "..")
    ):
        raise EvaluationInboxError(
            "{} must be a safe 1-128 character identifier".format(label)
        )
    return value


def _text(value: object, *, label: str, allow_empty: bool = False) -> str:
    if (
        type(value) is not str
        or (not allow_empty and not value)
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise EvaluationInboxError(
            "{} must be {}single-line text".format(
                label, "" if allow_empty else "non-empty "
            )
        )
    return value


def _reject_duplicate_pairs(
    pairs: Sequence[Tuple[str, object]],
) -> Dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationInboxError(
                "duplicate JSON key is forbidden: {!r}".format(key)
            )
        result[key] = value
    return result


def _reject_json_constant(token: str) -> object:
    raise EvaluationInboxError(
        "non-finite JSON constant is forbidden: {}".format(token)
    )


def _assert_finite_tree(value: object, *, label: str = "JSON") -> None:
    if type(value) is float and not math.isfinite(value):
        raise EvaluationInboxError("{} contains a non-finite number".format(label))
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_finite_tree(
                item, label="{}[{}]".format(label, index)
            )
    elif type(value) is dict:
        for key, item in value.items():
            _assert_finite_tree(
                item, label="{}.{}".format(label, key)
            )


def strict_json_loads(
    raw: object,
    *,
    label: str = "JSON document",
    require_canonical: bool = True,
) -> Dict[str, object]:
    """Decode UTF-8 JSON while rejecting duplicates and all non-finite forms."""

    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise EvaluationInboxError("{} must be raw bytes".format(label))
    payload = bytes(raw)
    if not payload or len(payload) > MAX_JSON_BYTES:
        raise EvaluationInboxError(
            "{} byte length is outside [1, {}]".format(
                label, MAX_JSON_BYTES
            )
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationInboxError("{} is not UTF-8".format(label)) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except EvaluationInboxError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise EvaluationInboxError(
            "{} is incomplete or invalid JSON".format(label)
        ) from exc
    _assert_finite_tree(value, label=label)
    if type(value) is not dict:
        raise EvaluationInboxError(
            "{} must contain one JSON object".format(label)
        )
    if require_canonical and payload != _canonical_json_bytes(value):
        raise EvaluationInboxError(
            "{} is not the canonical newline-terminated encoding".format(
                label
            )
        )
    return value


def _stat_signature(info: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
    )


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationInboxError(
            "{} is missing: {}".format(label, path)
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise EvaluationInboxError(
            "{} must be a regular non-symlink file: {}".format(label, path)
        )
    if before.st_nlink != 1:
        raise EvaluationInboxError(
            "{} must have exactly one filesystem link".format(label)
        )
    if before.st_size > MAX_JSON_BYTES:
        raise EvaluationInboxError(
            "{} exceeds the JSON size limit".format(label)
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise EvaluationInboxError(
            "cannot open {} without following links: {}".format(label, path)
        ) from exc
    chunks = []
    try:
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != _stat_signature(before):
            raise EvaluationInboxError(
                "{} changed while opening".format(label)
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationInboxError(
            "{} vanished while reading".format(label)
        ) from exc
    if (
        _stat_signature(before) != _stat_signature(after_descriptor)
        or _stat_signature(before) != _stat_signature(after_path)
    ):
        raise EvaluationInboxError(
            "{} changed while reading".format(label)
        )
    return b"".join(chunks)


def strict_read_json(path: object, *, label: str) -> Dict[str, object]:
    candidate = Path(os.fspath(path))
    return strict_json_loads(
        _read_regular_bytes(candidate, label=label),
        label=label,
        require_canonical=True,
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _native_rename_noreplace(source: Path, destination: Path) -> bool:
    """Use the platform's atomic no-replace rename when it is available."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(str(source))
    destination_bytes = os.fsencode(str(destination))
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        renameat2 = library.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,
        )
        if result == 0:
            return True
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                "refusing to replace existing artifact: {}".format(
                    destination
                )
            )
        if error not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise OSError(error, os.strerror(error), str(destination))
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        renamex_np = library.renamex_np
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
        if result == 0:
            return True
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                "refusing to replace existing artifact: {}".format(
                    destination
                )
            )
        if error not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise OSError(error, os.strerror(error), str(destination))
    return False


def _rename_noreplace(source: Path, destination: Path) -> None:
    if _native_rename_noreplace(source, destination):
        return
    # Hard-link installation has the same atomic no-clobber property.  It is
    # only a fallback for filesystems/platforms without renameat2/renamex_np.
    try:
        os.link(str(source), str(destination), follow_symlinks=False)
    except FileExistsError as exc:
        raise FileExistsError(
            "refusing to replace existing artifact: {}".format(destination)
        ) from exc
    os.unlink(str(source))


def _atomic_publish_json(path: Path, value: Mapping[str, object]) -> Path:
    payload = _canonical_json_bytes(value)
    parent = path.parent
    try:
        parent_info = parent.lstat()
    except FileNotFoundError as exc:
        raise EvaluationInboxError(
            "artifact parent is missing: {}".format(parent)
        ) from exc
    if not stat.S_ISDIR(parent_info.st_mode):
        raise EvaluationInboxError(
            "artifact parent must be a real directory: {}".format(parent)
        )
    temporary = parent / ".{}.tmp.{}.{}".format(
        path.name, os.getpid(), secrets.token_hex(12)
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(str(temporary), flags, 0o600)
    installed = False
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise EvaluationInboxError(
                    "temporary artifact write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _rename_noreplace(temporary, path)
        installed = True
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not installed:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    reopened = strict_read_json(path, label="published inbox artifact")
    if reopened != value:
        raise EvaluationInboxError(
            "published inbox artifact differs from the in-memory document"
        )
    return path


def _normalized_absolute_path(value: object, *, label: str) -> Path:
    text = _text(value, label=label)
    candidate = Path(text)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise EvaluationInboxError(
            "{} must be an absolute normalized path".format(label)
        )
    normalized = os.path.normpath(text)
    if normalized != text.rstrip(os.sep):
        raise EvaluationInboxError(
            "{} must already be normalized".format(label)
        )
    return Path(normalized)


def _hash_regular_file(path: Path, *, label: str) -> Tuple[str, int]:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationInboxError(
            "{} is missing: {}".format(label, path)
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise EvaluationInboxError(
            "{} must be a regular non-symlink file".format(label)
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(str(path), flags)
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != _stat_signature(before):
            raise EvaluationInboxError(
                "{} changed while opening".format(label)
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    if (
        _stat_signature(before) != _stat_signature(after_descriptor)
        or _stat_signature(before) != _stat_signature(after_path)
        or total != before.st_size
    ):
        raise EvaluationInboxError(
            "{} changed while hashing".format(label)
        )
    return digest.hexdigest(), total


def artifact_receipt(path: object) -> Dict[str, object]:
    """Hash one stable regular file into a request artifact binding."""

    candidate = Path(os.path.abspath(os.fspath(path)))
    digest, size = _hash_regular_file(candidate, label="bound artifact")
    return {
        "path": str(candidate),
        "sha256": digest,
        "size_bytes": size,
    }


def _validate_artifact_receipt(
    value: object, *, label: str
) -> Dict[str, object]:
    row = _exact_dict(
        value,
        ("path", "sha256", "size_bytes"),
        label=label,
    )
    _normalized_absolute_path(row["path"], label="{}.path".format(label))
    _sha256(row["sha256"], label="{}.sha256".format(label))
    _plain_int(row["size_bytes"], label="{}.size_bytes".format(label))
    return row


def verify_artifact_receipt(value: object, *, label: str) -> None:
    row = _validate_artifact_receipt(value, label=label)
    path = _normalized_absolute_path(
        row["path"], label="{}.path".format(label)
    )
    digest, size = _hash_regular_file(path, label=label)
    if digest != row["sha256"] or size != row["size_bytes"]:
        raise EvaluationInboxError(
            "{} bytes differ from the request receipt".format(label)
        )


def read_artifact_receipt_bytes(
    value: object, *, label: str
) -> bytes:
    """Read exact bound bytes without a path-hash/read TOCTOU window.

    Frozen evaluator checkpoints may be larger than the JSON transport limit,
    so this deliberately does not reuse ``_read_regular_bytes``.  The open
    descriptor, path entry, size, link count, and digest must remain identical
    for the whole read.
    """

    row = _validate_artifact_receipt(value, label=label)
    path = _normalized_absolute_path(
        row["path"], label="{}.path".format(label)
    )
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationInboxError(
            "{} is missing: {}".format(label, path)
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != row["size_bytes"]
    ):
        raise EvaluationInboxError(
            "{} is not the exact bound regular file".format(label)
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise EvaluationInboxError(
            "cannot open {} without following links: {}".format(label, path)
        ) from exc
    chunks = []
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != _stat_signature(before):
            raise EvaluationInboxError(
                "{} changed while opening".format(label)
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationInboxError(
            "{} vanished while reading".format(label)
        ) from exc
    if (
        _stat_signature(before) != _stat_signature(after_descriptor)
        or _stat_signature(before) != _stat_signature(after_path)
        or size != row["size_bytes"]
        or digest.hexdigest() != row["sha256"]
    ):
        raise EvaluationInboxError(
            "{} bytes changed or differ from the receipt".format(label)
        )
    return b"".join(chunks)


def state_binding(*, sha256: str, size_bytes: int) -> Dict[str, object]:
    """Bind normalized state extracted from a checkpoint."""

    result = {"sha256": sha256, "size_bytes": size_bytes}
    _validate_state_binding(result, label="state binding")
    return result


def _validate_state_binding(
    value: object, *, label: str
) -> Dict[str, object]:
    row = _exact_dict(value, ("sha256", "size_bytes"), label=label)
    _sha256(row["sha256"], label="{}.sha256".format(label))
    _plain_int(row["size_bytes"], label="{}.size_bytes".format(label))
    return row


def build_sidecar_launch_document(
    *,
    sidecar_code_sha256: str,
    backend_contract_sha256: str,
) -> Dict[str, object]:
    """Build, but do not authorize, one exact sidecar launch receipt."""

    content = {
        "protocol_contract_sha256": EVALUATION_INBOX_CONTRACT_SHA256,
        "sidecar_code_sha256": _sha256(
            sidecar_code_sha256, label="sidecar_code_sha256"
        ),
        "backend_contract_sha256": _sha256(
            backend_contract_sha256, label="backend_contract_sha256"
        ),
        "policy_evaluation_contract_sha256": (
            POLICY_EVALUATION_CONTRACT_SHA256
        ),
        "resolved_recipe_contract_sha256": (
            RESOLVED_EVALUATION_RECIPE_CONTRACT_SHA256
        ),
        "runtime_identity_contract_sha256": (
            RUNTIME_IDENTITY_CONTRACT_SHA256
        ),
        "heartbeat_contract": dict(SIDECAR_HEARTBEAT_CONTRACT),
        "window_contract": dict(WINDOW_CONTRACT),
    }
    return _envelope(LAUNCH_KIND, content)


def validate_sidecar_launch_document(
    document: object,
    *,
    actual_sidecar_code_sha256: str,
    backend_contract_sha256: str,
    require_trust: bool = True,
) -> Dict[str, object]:
    content = _validate_envelope(document, LAUNCH_KIND)
    row = _exact_dict(
        content,
        (
            "protocol_contract_sha256",
            "sidecar_code_sha256",
            "backend_contract_sha256",
            "policy_evaluation_contract_sha256",
            "resolved_recipe_contract_sha256",
            "runtime_identity_contract_sha256",
            "heartbeat_contract",
            "window_contract",
        ),
        label="sidecar launch content",
    )
    actual_code = _sha256(
        actual_sidecar_code_sha256,
        label="actual_sidecar_code_sha256",
    )
    backend_contract = _sha256(
        backend_contract_sha256,
        label="backend_contract_sha256",
    )
    heartbeat = _exact_dict(
        row["heartbeat_contract"],
        (
            "schema_version",
            "heartbeat_interval_seconds",
            "heartbeat_stale_after_seconds",
            "request_deadline_seconds",
        ),
        label="sidecar heartbeat contract",
    )
    if (
        type(heartbeat["schema_version"]) is not int
        or heartbeat["schema_version"] != 1
        or any(
            type(heartbeat[name]) is not float
            or heartbeat[name] != SIDECAR_HEARTBEAT_CONTRACT[name]
            for name in (
                "heartbeat_interval_seconds",
                "heartbeat_stale_after_seconds",
                "request_deadline_seconds",
            )
        )
    ):
        raise EvaluationInboxError(
            "sidecar heartbeat contract is not exact"
        )
    if (
        row["protocol_contract_sha256"]
        != EVALUATION_INBOX_CONTRACT_SHA256
        or row["policy_evaluation_contract_sha256"]
        != POLICY_EVALUATION_CONTRACT_SHA256
        or row["resolved_recipe_contract_sha256"]
        != RESOLVED_EVALUATION_RECIPE_CONTRACT_SHA256
        or row["runtime_identity_contract_sha256"]
        != RUNTIME_IDENTITY_CONTRACT_SHA256
        or heartbeat != SIDECAR_HEARTBEAT_CONTRACT
        or row["window_contract"] != WINDOW_CONTRACT
        or row["sidecar_code_sha256"] != actual_code
        or row["backend_contract_sha256"] != backend_contract
    ):
        raise EvaluationInboxError(
            "sidecar launch contract or code binding mismatch"
        )
    if require_trust:
        if (
            actual_code
            not in TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_CODE_SHA256
        ):
            raise EvaluationInboxError(
                "sidecar code SHA is not code-pinned"
            )
        launch_sha256 = document["content_sha256"]
        if (
            launch_sha256
            not in TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_LAUNCH_SHA256
        ):
            raise EvaluationInboxError(
                "sidecar launch SHA is not code-pinned"
            )
    return row


def _envelope(kind: str, content: Mapping[str, object]) -> Dict[str, object]:
    copied = json.loads(_canonical_json_bytes(content).decode("ascii"))
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "content": copied,
        "content_sha256": canonical_sha256(copied),
    }
    return result


def _validate_envelope(
    document: object, expected_kind: str
) -> Dict[str, object]:
    row = _exact_dict(
        document,
        ("schema_version", "kind", "content", "content_sha256"),
        label="{} envelope".format(expected_kind),
    )
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != SCHEMA_VERSION
    ):
        raise EvaluationInboxError(
            "{} schema_version mismatch".format(expected_kind)
        )
    if row["kind"] != expected_kind:
        raise EvaluationInboxError(
            "expected {}, observed {!r}".format(
                expected_kind, row["kind"]
            )
        )
    content = row["content"]
    if type(content) is not dict:
        raise EvaluationInboxError(
            "{} content must be a JSON object".format(expected_kind)
        )
    declared = _sha256(
        row["content_sha256"],
        label="{}.content_sha256".format(expected_kind),
    )
    if canonical_sha256(content) != declared:
        raise EvaluationInboxError(
            "{} content digest mismatch".format(expected_kind)
        )
    return content


def make_window_allocations(
    *,
    seed_start: int,
    sample_start: int,
    birth_start: int,
) -> List[Dict[str, object]]:
    seed = _plain_int(seed_start, label="seed_start")
    sample = _plain_int(sample_start, label="sample_start")
    birth = _plain_int(birth_start, label="birth_start")
    canary = {
        "role": "frozen_canary",
        "proposal_count": CANARY_PROPOSALS,
        "seed_start": seed,
        "seed_end_exclusive": seed + CANARY_PROPOSALS,
        "sample_start": sample,
        "sample_end_exclusive": sample + CANARY_PROPOSALS,
        "birth_start": birth,
        "birth_end_exclusive": birth + CANARY_PROPOSALS,
    }
    heldout = {
        "role": "frozen_heldout",
        "proposal_count": HELDOUT_PROPOSALS,
        "seed_start": canary["seed_end_exclusive"],
        "seed_end_exclusive": (
            canary["seed_end_exclusive"] + HELDOUT_PROPOSALS
        ),
        "sample_start": canary["sample_end_exclusive"],
        "sample_end_exclusive": (
            canary["sample_end_exclusive"] + HELDOUT_PROPOSALS
        ),
        "birth_start": canary["birth_end_exclusive"],
        "birth_end_exclusive": (
            canary["birth_end_exclusive"] + HELDOUT_PROPOSALS
        ),
    }
    for key in (
        "seed_end_exclusive",
        "sample_end_exclusive",
        "birth_end_exclusive",
    ):
        if heldout[key] > INT64_MAX:
            raise EvaluationInboxError(
                "window allocation exceeds the int64 range"
            )
    return [canary, heldout]


def make_scheduler_allocation(
    *,
    seed_start: int,
    sample_start: int,
    birth_start: int,
) -> List[Dict[str, object]]:
    """Allocate the fixed rolling-100 scheduler evidence window."""

    starts = {
        "seed": _plain_int(seed_start, label="seed_start"),
        "sample": _plain_int(sample_start, label="sample_start"),
        "birth": _plain_int(birth_start, label="birth_start"),
    }
    row = {
        "role": "scheduler",
        "proposal_count": SCHEDULER_PROPOSALS,
    }
    for axis, start in starts.items():
        end = start + SCHEDULER_PROPOSALS
        if end > INT64_MAX:
            raise EvaluationInboxError(
                "scheduler allocation exceeds the int64 range"
            )
        row[f"{axis}_start"] = start
        row[f"{axis}_end_exclusive"] = end
    return [row]


_WINDOW_KEYS = (
    "role",
    "proposal_count",
    "seed_start",
    "seed_end_exclusive",
    "sample_start",
    "sample_end_exclusive",
    "birth_start",
    "birth_end_exclusive",
)


def _validate_window_allocations(
    value: object,
) -> List[Dict[str, object]]:
    if type(value) is not list:
        raise EvaluationInboxError(
            "request windows must be a plain list"
        )
    observed_roles = tuple(
        raw.get("role") if type(raw) is dict else None
        for raw in value
    )
    if observed_roles == ("scheduler",):
        expected_roles = (("scheduler", SCHEDULER_PROPOSALS),)
    elif observed_roles == ("frozen_canary", "frozen_heldout"):
        expected_roles = (
            ("frozen_canary", CANARY_PROPOSALS),
            ("frozen_heldout", HELDOUT_PROPOSALS),
        )
    else:
        raise EvaluationInboxError(
            "request windows must contain exactly scheduler or "
            "canary then heldout"
        )
    result = []
    for index, (raw, expected) in enumerate(zip(value, expected_roles)):
        row = _exact_dict(
            raw, _WINDOW_KEYS, label="windows[{}]".format(index)
        )
        role, count = expected
        if row["role"] != role:
            raise EvaluationInboxError(
                "windows[{}].role must be {}".format(index, role)
            )
        proposal_count = _plain_int(
            row["proposal_count"],
            label="windows[{}].proposal_count".format(index),
            minimum=count,
            maximum=count,
        )
        for axis in ("seed", "sample", "birth"):
            start = _plain_int(
                row["{}_start".format(axis)],
                label="windows[{}].{}_start".format(index, axis),
            )
            end = _plain_int(
                row["{}_end_exclusive".format(axis)],
                label="windows[{}].{}_end_exclusive".format(index, axis),
            )
            if end - start != proposal_count:
                raise EvaluationInboxError(
                    "windows[{}] {} range must contain exactly {} values".format(
                        index, axis, proposal_count
                    )
                )
        result.append(row)
    if len(result) == 2:
        for axis in ("seed", "sample", "birth"):
            if (
                result[1]["{}_start".format(axis)]
                != result[0]["{}_end_exclusive".format(axis)]
            ):
                raise EvaluationInboxError(
                    "heldout {} range must start after the canary "
                    "range".format(axis)
                )
    return result


def _validate_bindings(value: object) -> Dict[str, object]:
    row = _exact_dict(
        value,
        (
            "checkpoint",
            "training_contract",
            "environment_config_pickle",
            "agent_config_pickle",
            "runtime_identity",
            "runtime_bootstrap_receipt_sha256",
            "runtime_bootstrap_lineage_payload_sha256",
            "runtime_bootstrap_receipt",
            "training_launch_claim_sha256",
            "policy_generation",
            "policy_state",
            "actor_obs_normalizer",
            "critic_obs_normalizer",
            "ppo_recipe_sha256",
            "policy_contract_sha256",
            "action_order",
            "actions",
            "manifest_sha256",
            "sampler_sha256",
            "proposal_sampler_contract_sha256",
            "solver_sha256",
            "physics_sha256",
            "reward_sha256",
            "curriculum_sha256",
        ),
        label="request bindings",
    )
    _validate_artifact_receipt(row["checkpoint"], label="checkpoint")
    for name in (
        "training_contract",
        "environment_config_pickle",
        "agent_config_pickle",
        "runtime_identity",
        "runtime_bootstrap_receipt",
    ):
        _validate_artifact_receipt(
            row[name], label="bindings.{}".format(name)
        )
    for name in (
        "runtime_bootstrap_receipt_sha256",
        "runtime_bootstrap_lineage_payload_sha256",
    ):
        _sha256(row[name], label="bindings.{}".format(name))
    _sha256(
        row["training_launch_claim_sha256"],
        label="bindings.training_launch_claim_sha256",
    )
    _plain_int(
        row["policy_generation"],
        label="bindings.policy_generation",
    )
    _validate_state_binding(
        row["policy_state"],
        label="policy_state",
    )
    _validate_state_binding(
        row["actor_obs_normalizer"],
        label="actor_obs_normalizer",
    )
    _validate_state_binding(
        row["critic_obs_normalizer"],
        label="critic_obs_normalizer",
    )
    for name in (
        "ppo_recipe_sha256",
        "policy_contract_sha256",
        "manifest_sha256",
        "sampler_sha256",
        "proposal_sampler_contract_sha256",
        "solver_sha256",
        "physics_sha256",
        "reward_sha256",
        "curriculum_sha256",
    ):
        _sha256(row[name], label="bindings.{}".format(name))
    order = row["action_order"]
    if (
        type(order) is not list
        or not order
        or any(type(uid) is not int or uid < 1 for uid in order)
        or len(order) != len(set(order))
    ):
        raise EvaluationInboxError(
            "action_order must be a non-empty unique plain-integer list"
        )
    actions = row["actions"]
    if type(actions) is not list or len(actions) != len(order):
        raise EvaluationInboxError(
            "actions must contain one ordered motion receipt per action UID"
        )
    observed_order = []
    for index, raw_action in enumerate(actions):
        action = _exact_dict(
            raw_action,
            ("action_uid", "motion"),
            label="actions[{}]".format(index),
        )
        uid = _plain_int(
            action["action_uid"],
            label="actions[{}].action_uid".format(index),
            minimum=1,
        )
        _validate_artifact_receipt(
            action["motion"],
            label="actions[{}].motion".format(index),
        )
        observed_order.append(uid)
    if observed_order != order:
        raise EvaluationInboxError(
            "actions order does not equal action_order"
        )
    return row


def _static_run_contract(
    bindings: Mapping[str, object],
    sidecar_launch_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "sidecar_launch_sha256": sidecar_launch_sha256,
            "training_contract": bindings["training_contract"],
            "environment_config_pickle": bindings[
                "environment_config_pickle"
            ],
            "agent_config_pickle": bindings["agent_config_pickle"],
            "runtime_identity": bindings["runtime_identity"],
            "runtime_bootstrap_receipt_sha256": bindings[
                "runtime_bootstrap_receipt_sha256"
            ],
            "runtime_bootstrap_lineage_payload_sha256": bindings[
                "runtime_bootstrap_lineage_payload_sha256"
            ],
            "runtime_bootstrap_receipt": bindings[
                "runtime_bootstrap_receipt"
            ],
            "training_launch_claim_sha256": bindings[
                "training_launch_claim_sha256"
            ],
            "ppo_recipe_sha256": bindings["ppo_recipe_sha256"],
            "policy_contract_sha256": bindings["policy_contract_sha256"],
            "action_order": bindings["action_order"],
            "actions": bindings["actions"],
            "manifest_sha256": bindings["manifest_sha256"],
            "sampler_sha256": bindings["sampler_sha256"],
            "proposal_sampler_contract_sha256": bindings[
                "proposal_sampler_contract_sha256"
            ],
            "solver_sha256": bindings["solver_sha256"],
            "physics_sha256": bindings["physics_sha256"],
            "reward_sha256": bindings["reward_sha256"],
            "curriculum_sha256": bindings["curriculum_sha256"],
            "policy_evaluation_contract_sha256": (
                POLICY_EVALUATION_CONTRACT_SHA256
            ),
        }
    )


def _validate_target(
    value: object, bindings: Mapping[str, object]
) -> Dict[str, object]:
    row = _exact_dict(
        value,
        (
            "action_uid",
            "profile_sha256",
            "mobility_mode",
            "domain_epoch",
            "stratum",
            "selected_arm_key",
            "selection_round",
            "arm_levels",
            "rho",
        ),
        label="request target",
    )
    action_uid = _plain_int(
        row["action_uid"], label="target.action_uid", minimum=1
    )
    if action_uid not in bindings["action_order"]:
        raise EvaluationInboxError(
            "target.action_uid is outside the frozen action order"
        )
    _sha256(row["profile_sha256"], label="target.profile_sha256")
    if row["mobility_mode"] not in ("no_move", "move"):
        raise EvaluationInboxError(
            "target.mobility_mode must be no_move or move"
        )
    _plain_int(row["domain_epoch"], label="target.domain_epoch")
    _text(row["stratum"], label="target.stratum")
    _text(
        row["selected_arm_key"],
        label="target.selected_arm_key",
        allow_empty=True,
    )
    _plain_int(
        row["selection_round"], label="target.selection_round"
    )
    levels = row["arm_levels"]
    if type(levels) is not list or not levels:
        raise EvaluationInboxError(
            "target.arm_levels must be a non-empty list"
        )
    for index, level in enumerate(levels):
        _finite_number(
            level,
            label="target.arm_levels[{}]".format(index),
            minimum=0.0,
            maximum=1.0,
        )
    _finite_number(
        row["rho"], label="target.rho", minimum=0.0, maximum=1.0
    )
    return row


def build_request_document(
    *,
    owner_id: str,
    run_id: str,
    request_seq: int,
    sidecar_launch_sha256: str,
    bindings: Mapping[str, object],
    target: Mapping[str, object],
    seed_start: int,
    sample_start: int,
    birth_start: int,
    request_kind: str = "formal",
) -> Dict[str, object]:
    """Build one fixed scheduler or canary+heldout request."""

    normalized_bindings = json.loads(
        _canonical_json_bytes(bindings).decode("ascii")
    )
    normalized_target = json.loads(
        _canonical_json_bytes(target).decode("ascii")
    )
    _validate_bindings(normalized_bindings)
    _validate_target(normalized_target, normalized_bindings)
    launch_sha256 = _sha256(
        sidecar_launch_sha256, label="sidecar_launch_sha256"
    )
    if request_kind == "formal":
        windows = make_window_allocations(
            seed_start=seed_start,
            sample_start=sample_start,
            birth_start=birth_start,
        )
    elif request_kind == "scheduler":
        windows = make_scheduler_allocation(
            seed_start=seed_start,
            sample_start=sample_start,
            birth_start=birth_start,
        )
    else:
        raise EvaluationInboxError(
            "request_kind must be formal or scheduler"
        )
    content = {
        "owner_id": _identifier(owner_id, label="owner_id"),
        "run_id": _identifier(run_id, label="run_id"),
        "request_seq": _plain_int(request_seq, label="request_seq"),
        "sidecar_launch_sha256": launch_sha256,
        "run_contract_sha256": _static_run_contract(
            normalized_bindings, launch_sha256
        ),
        "bindings": normalized_bindings,
        "target": normalized_target,
        "policy_evaluation": dict(POLICY_EVALUATION_CONTRACT),
        "windows": windows,
    }
    document = _envelope(REQUEST_KIND, content)
    validate_request_document(document)
    return document


def validate_request_document(
    document: object,
    *,
    expected_owner_id: Optional[str] = None,
    expected_run_id: Optional[str] = None,
    expected_request_seq: Optional[int] = None,
    expected_sidecar_launch_sha256: Optional[str] = None,
) -> Dict[str, object]:
    content = _validate_envelope(document, REQUEST_KIND)
    row = _exact_dict(
        content,
        (
            "owner_id",
            "run_id",
            "request_seq",
            "sidecar_launch_sha256",
            "run_contract_sha256",
            "bindings",
            "target",
            "policy_evaluation",
            "windows",
        ),
        label="request content",
    )
    owner = _identifier(row["owner_id"], label="owner_id")
    run = _identifier(row["run_id"], label="run_id")
    seq = _plain_int(row["request_seq"], label="request_seq")
    launch_sha = _sha256(
        row["sidecar_launch_sha256"],
        label="sidecar_launch_sha256",
    )
    bindings = _validate_bindings(row["bindings"])
    _validate_target(row["target"], bindings)
    if row["policy_evaluation"] != POLICY_EVALUATION_CONTRACT:
        raise EvaluationInboxError(
            "request policy-evaluation determinism contract drifted"
        )
    _validate_window_allocations(row["windows"])
    run_contract = _sha256(
        row["run_contract_sha256"],
        label="run_contract_sha256",
    )
    if run_contract != _static_run_contract(bindings, launch_sha):
        raise EvaluationInboxError(
            "request run contract digest mismatch"
        )
    if expected_owner_id is not None and owner != _identifier(
        expected_owner_id, label="expected_owner_id"
    ):
        raise EvaluationInboxError("request belongs to another owner")
    if expected_run_id is not None and run != _identifier(
        expected_run_id, label="expected_run_id"
    ):
        raise EvaluationInboxError("request belongs to another run")
    if (
        expected_request_seq is not None
        and seq
        != _plain_int(
            expected_request_seq, label="expected_request_seq"
        )
    ):
        raise EvaluationInboxError(
            "request sequence does not match its inbox identity"
        )
    if (
        expected_sidecar_launch_sha256 is not None
        and launch_sha
        != _sha256(
            expected_sidecar_launch_sha256,
            label="expected_sidecar_launch_sha256",
        )
    ):
        raise EvaluationInboxError(
            "request targets another sidecar launch"
        )
    return row


def verify_request_artifacts(request_document: object) -> None:
    """Re-hash checkpoint and every ordered motion before evaluation."""

    content = validate_request_document(request_document)
    bindings = content["bindings"]
    verify_artifact_receipt(
        bindings["checkpoint"], label="checkpoint"
    )
    for name in (
        "training_contract",
        "environment_config_pickle",
        "agent_config_pickle",
        "runtime_identity",
        "runtime_bootstrap_receipt",
    ):
        verify_artifact_receipt(
            bindings[name], label="bindings.{}".format(name)
        )
    receipt_document = strict_read_json(
        bindings["runtime_bootstrap_receipt"]["path"],
        label="bindings.runtime_bootstrap_receipt",
    )
    if (
        receipt_document.get("content_sha256")
        != bindings["runtime_bootstrap_receipt_sha256"]
        or not isinstance(receipt_document.get("content"), dict)
        or receipt_document["content"].get("lineage_payload_sha256")
        != bindings["runtime_bootstrap_lineage_payload_sha256"]
    ):
        raise EvaluationInboxError(
            "runtime bootstrap receipt content/lineage differs from request"
        )
    for index, action in enumerate(bindings["actions"]):
        verify_artifact_receipt(
            action["motion"],
            label="actions[{}].motion".format(index),
        )


_ATTEMPT_KEYS = (
    "proposal_offset",
    "seed",
    "sample_id",
    "birth_id",
    "proposal_sampler_contract_sha256",
    "proposal_receipt_sha256",
    "sample_receipt_sha256",
    "birth_receipt_sha256",
    "sampling_stratum",
    "frontier_arm",
    "solver_disposition",
    "reject_reason",
    "task_receipt_sha256",
    "installed",
    "started",
    "closed",
    "terminal_signals",
)

_TERMINAL_SIGNAL_KEYS = (
    "infrastructure_invalid",
    "joint_actual_limit",
    "joint_qdes_limit",
    "fall",
    "table_hit",
    "collision",
    "legal_return",
)


def _validate_terminal_signals(
    value: object, *, label: str
) -> Dict[str, bool]:
    row = _exact_dict(value, _TERMINAL_SIGNAL_KEYS, label=label)
    for name in _TERMINAL_SIGNAL_KEYS:
        if type(row[name]) is not bool:
            raise EvaluationInboxError(
                "{}.{} must be a raw boolean".format(label, name)
            )
    return row


def classify_terminal_signals(signals: object) -> Optional[str]:
    """Apply the runtime's single hard-outcome precedence to raw booleans."""

    row = _validate_terminal_signals(
        signals, label="terminal_signals"
    )
    if row["infrastructure_invalid"]:
        return None
    if row["joint_actual_limit"]:
        return "joint_actual_limit"
    if row["joint_qdes_limit"]:
        return "joint_qdes_limit"
    if row["fall"]:
        return "fall"
    if row["table_hit"]:
        return "table_hit"
    if row["collision"]:
        return "collision"
    if row["legal_return"]:
        return "legal_return"
    return "safe_nonreturn"


def _validate_attempt(
    value: object,
    *,
    allocation: Mapping[str, object],
    offset: int,
    proposal_sampler_contract_sha256: str,
) -> Dict[str, object]:
    row = _exact_dict(
        value, _ATTEMPT_KEYS, label="attempt[{}]".format(offset)
    )
    expected_numbers = {
        "proposal_offset": offset,
        "seed": allocation["seed_start"] + offset,
        "sample_id": allocation["sample_start"] + offset,
        "birth_id": allocation["birth_start"] + offset,
    }
    for field, expected in expected_numbers.items():
        actual = _plain_int(
            row[field], label="attempt[{}].{}".format(offset, field)
        )
        if actual != expected:
            raise EvaluationInboxError(
                "attempt[{}].{} does not match its allocation".format(
                    offset, field
                )
            )
    proposal_contract = _sha256(
        row["proposal_sampler_contract_sha256"],
        label=(
            "attempt[{}].proposal_sampler_contract_sha256".format(
                offset
            )
        ),
    )
    if proposal_contract != _sha256(
        proposal_sampler_contract_sha256,
        label="request proposal_sampler_contract_sha256",
    ):
        raise EvaluationInboxError(
            "attempt proposal sampler contract differs from request"
        )
    _sha256(
        row["proposal_receipt_sha256"],
        label="attempt[{}].proposal_receipt_sha256".format(offset),
    )
    _sha256(
        row["sample_receipt_sha256"],
        label="attempt[{}].sample_receipt_sha256".format(offset),
    )
    _sha256(
        row["birth_receipt_sha256"],
        label="attempt[{}].birth_receipt_sha256".format(offset),
    )
    stratum = row["sampling_stratum"]
    if stratum not in ("center", "interior", "frontier"):
        raise EvaluationInboxError(
            "attempt sampling_stratum is invalid"
        )
    frontier_arm = _text(
        row["frontier_arm"],
        label="attempt[{}].frontier_arm".format(offset),
        allow_empty=True,
    )
    if (stratum == "frontier") != bool(frontier_arm):
        raise EvaluationInboxError(
            "attempt frontier arm/stratum accounting is inconsistent"
        )
    disposition = row["solver_disposition"]
    if disposition not in ("physics_invalid", "rejected", "admitted"):
        raise EvaluationInboxError(
            "attempt solver_disposition is invalid"
        )
    reject_reason = _text(
        row["reject_reason"],
        label="attempt[{}].reject_reason".format(offset),
        allow_empty=True,
    )
    task_receipt = row["task_receipt_sha256"]
    if type(task_receipt) is not str:
        raise EvaluationInboxError(
            "attempt task_receipt_sha256 must be text"
        )
    booleans = {}
    for field in ("installed", "started", "closed"):
        if type(row[field]) is not bool:
            raise EvaluationInboxError(
                "attempt[{}].{} must be boolean".format(offset, field)
            )
        booleans[field] = row[field]
    signals = _validate_terminal_signals(
        row["terminal_signals"],
        label="attempt[{}].terminal_signals".format(offset),
    )
    outcome = classify_terminal_signals(signals)
    if disposition in ("physics_invalid", "rejected"):
        if (
            not reject_reason
            or task_receipt
            or any(booleans.values())
            or any(signals.values())
        ):
            raise EvaluationInboxError(
                "{} attempt has invalid task/lifecycle accounting".format(
                    disposition
                )
            )
    else:
        _sha256(
            task_receipt,
            label="attempt[{}].task_receipt_sha256".format(offset),
        )
        if reject_reason:
            raise EvaluationInboxError(
                "admitted attempt must not carry a reject reason"
            )
        if booleans["started"] and not booleans["installed"]:
            raise EvaluationInboxError(
                "started attempt was not installed"
            )
        if booleans["closed"] and not booleans["started"]:
            raise EvaluationInboxError(
                "closed attempt was not started"
            )
        if booleans["closed"]:
            if (
                signals["infrastructure_invalid"]
                or outcome not in TERMINAL_OUTCOMES
            ):
                raise EvaluationInboxError(
                    "closed attempt needs one non-infrastructure terminal"
                )
        else:
            if (
                not signals["infrastructure_invalid"]
                or outcome is not None
                or any(
                    signals[name]
                    for name in _TERMINAL_SIGNAL_KEYS
                    if name != "infrastructure_invalid"
                )
            ):
                raise EvaluationInboxError(
                    "unfinished admitted attempt must burn as pure "
                    "infrastructure invalid"
                )
    return row


def _reason_counts(
    rows: Sequence[Mapping[str, object]], disposition: str
) -> Dict[str, int]:
    result = {}
    for row in rows:
        if row["solver_disposition"] == disposition:
            reason = row["reject_reason"]
            result[reason] = result.get(reason, 0) + 1
    return dict(sorted(result.items()))


def _validate_window_sampling_contract(
    rows: Sequence[Mapping[str, object]],
    *,
    count: int,
    selected_arm_key: str,
) -> None:
    expected_mixture = {
        "center": count // 5,
        "interior": 3 * count // 5,
        "frontier": count // 5,
    }
    observed_mixture = {
        stratum: sum(
            row["sampling_stratum"] == stratum for row in rows
        )
        for stratum in expected_mixture
    }
    if observed_mixture != expected_mixture:
        raise EvaluationInboxError(
            "window sampling mixture must be exact 20/60/20: "
            f"expected={expected_mixture}, observed={observed_mixture}"
        )
    if selected_arm_key and any(
        row["sampling_stratum"] == "frontier"
        and row["frontier_arm"] != selected_arm_key
        for row in rows
    ):
        raise EvaluationInboxError(
            "frontier samples must exercise the selected action-axis-side arm"
        )


def _derive_ledger(
    rows: Sequence[Mapping[str, object]]
) -> Dict[str, object]:
    outcomes = [
        classify_terminal_signals(row["terminal_signals"])
        if row["closed"]
        else None
        for row in rows
    ]
    terminal = {
        name: sum(outcome == name for outcome in outcomes)
        for name in TERMINAL_OUTCOMES
    }
    return {
        "proposed": len(rows),
        "physics_invalid": sum(
            row["solver_disposition"] == "physics_invalid"
            for row in rows
        ),
        "solver_rejected": sum(
            row["solver_disposition"] == "rejected" for row in rows
        ),
        "solver_admitted": sum(
            row["solver_disposition"] == "admitted" for row in rows
        ),
        "installed": sum(bool(row["installed"]) for row in rows),
        "started": sum(bool(row["started"]) for row in rows),
        "closed": sum(bool(row["closed"]) for row in rows),
        "legal_return": terminal["legal_return"],
        "safe_nonreturn": terminal["safe_nonreturn"],
        # Safety channels are raw sticky sensor dimensions, not the
        # mutually-exclusive terminal-precedence label.  One closure may
        # therefore increment (for example) both joint_actual_limit and
        # table_hit while still contributing exactly once to ``closed``.
        "table_hit": sum(
            bool(row["terminal_signals"]["table_hit"]) for row in rows
        ),
        "fall": sum(
            bool(row["terminal_signals"]["fall"]) for row in rows
        ),
        "collision": sum(
            bool(row["terminal_signals"]["collision"]) for row in rows
        ),
        "joint_qdes_limit": sum(
            bool(row["terminal_signals"]["joint_qdes_limit"])
            for row in rows
        ),
        "joint_actual_limit": sum(
            bool(row["terminal_signals"]["joint_actual_limit"])
            for row in rows
        ),
        "infrastructure_invalid": sum(
            bool(row["terminal_signals"]["infrastructure_invalid"])
            for row in rows
        ),
        "physics_invalid_reasons": _reason_counts(
            rows, "physics_invalid"
        ),
        "solver_reject_reasons": _reason_counts(rows, "rejected"),
    }


_LEDGER_KEYS = (
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
    "physics_invalid_reasons",
    "solver_reject_reasons",
)


def _validate_ledger(
    value: object,
    *,
    rows: Sequence[Mapping[str, object]],
    role: str,
) -> Dict[str, object]:
    row = _exact_dict(value, _LEDGER_KEYS, label="window ledger")
    expected = _derive_ledger(rows)
    if row != expected:
        raise EvaluationInboxError(
            "window ledger does not equal the exact attempt transcript"
        )
    if (
        row["proposed"]
        != row["physics_invalid"]
        + row["solver_rejected"]
        + row["solver_admitted"]
    ):
        raise EvaluationInboxError(
            "proposal disposition ledger does not conserve P"
        )
    if not (
        row["closed"] <= row["started"] <= row["installed"]
        <= row["solver_admitted"]
    ):
        raise EvaluationInboxError(
            "attempt lifecycle counts are not monotonic"
        )
    if (
        row["closed"] + row["infrastructure_invalid"]
        != row["solver_admitted"]
    ):
        raise EvaluationInboxError(
            "every admitted proposal must close or burn as infrastructure "
            "invalid"
        )
    safe_closed = row["legal_return"] + row["safe_nonreturn"]
    floor = {
        "scheduler": 0,
        "frozen_canary": CANARY_SAFE_CLOSED_MIN,
        "frozen_heldout": HELDOUT_SAFE_CLOSED_MIN,
    }.get(role)
    if floor is None:
        raise EvaluationInboxError("window ledger role is invalid")
    if safe_closed < floor:
        raise EvaluationInboxError(
            "{} safe-closed floor is {}, observed {}".format(
                role, floor, safe_closed
            )
        )
    return row


def _attempt_receipt_root(
    rows: Sequence[Mapping[str, object]]
) -> str:
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "ordered_action_ball_frozen_eval_attempt_receipts",
            "count": len(rows),
            "ordered_attempt_sha256": [
                canonical_sha256(row) for row in rows
            ],
        }
    )


def _build_evidence_window(
    allocation: Mapping[str, object],
    raw_attempts: object,
    *,
    selected_arm_key: str,
    proposal_sampler_contract_sha256: str,
) -> Dict[str, object]:
    if type(raw_attempts) is not list:
        raise EvaluationInboxError(
            "sidecar attempts must be a plain list"
        )
    count = allocation["proposal_count"]
    if len(raw_attempts) != count:
        raise EvaluationInboxError(
            "{} requires exactly {} attempts".format(
                allocation["role"], count
            )
        )
    rows = [
        _validate_attempt(
            raw,
            allocation=allocation,
            offset=index,
            proposal_sampler_contract_sha256=(
                proposal_sampler_contract_sha256
            ),
        )
        for index, raw in enumerate(raw_attempts)
    ]
    _validate_window_sampling_contract(
        rows,
        count=count,
        selected_arm_key=selected_arm_key,
    )
    sample_receipts = [row["sample_receipt_sha256"] for row in rows]
    birth_receipts = [row["birth_receipt_sha256"] for row in rows]
    proposal_receipts = [
        row["proposal_receipt_sha256"] for row in rows
    ]
    admitted_tasks = [
        row["task_receipt_sha256"]
        for row in rows
        if row["solver_disposition"] == "admitted"
    ]
    if len(proposal_receipts) != len(set(proposal_receipts)):
        raise EvaluationInboxError(
            "evaluation window reused a proposal receipt"
        )
    if len(sample_receipts) != len(set(sample_receipts)):
        raise EvaluationInboxError(
            "evaluation window reused a sample receipt"
        )
    if len(birth_receipts) != len(set(birth_receipts)):
        raise EvaluationInboxError(
            "evaluation window reused a birth receipt"
        )
    if len(admitted_tasks) != len(set(admitted_tasks)):
        raise EvaluationInboxError(
            "evaluation window reused an admitted task receipt"
        )
    return {
        "allocation": dict(allocation),
        "attempt_receipt_root_sha256": _attempt_receipt_root(rows),
        "ledger": _derive_ledger(rows),
        "attempts": rows,
    }


def build_evidence_document(
    request_document: object,
    *,
    sidecar_launch_sha256: str,
    attempts_by_role: Mapping[str, object],
) -> Dict[str, object]:
    """Build exact evidence from full fixed-window attempt transcripts."""

    request = validate_request_document(
        request_document,
        expected_sidecar_launch_sha256=sidecar_launch_sha256,
    )
    expected_roles = {
        allocation["role"] for allocation in request["windows"]
    }
    if (
        type(attempts_by_role) is not dict
        or set(attempts_by_role) != expected_roles
    ):
        raise EvaluationInboxError(
            "attempts_by_role must exactly match the requested windows"
        )
    windows = []
    for allocation in request["windows"]:
        windows.append(
            _build_evidence_window(
                allocation,
                attempts_by_role[allocation["role"]],
                selected_arm_key=request["target"][
                    "selected_arm_key"
                ],
                proposal_sampler_contract_sha256=request["bindings"][
                    "proposal_sampler_contract_sha256"
                ],
            )
        )
    all_proposals = [
        attempt["proposal_receipt_sha256"]
        for window in windows
        for attempt in window["attempts"]
    ]
    all_samples = [
        attempt["sample_receipt_sha256"]
        for window in windows
        for attempt in window["attempts"]
    ]
    all_births = [
        attempt["birth_receipt_sha256"]
        for window in windows
        for attempt in window["attempts"]
    ]
    if len(all_proposals) != len(set(all_proposals)):
        raise EvaluationInboxError(
            "canary and heldout reused a proposal receipt"
        )
    if len(all_samples) != len(set(all_samples)):
        raise EvaluationInboxError(
            "canary and heldout reused a sample receipt"
        )
    if len(all_births) != len(set(all_births)):
        raise EvaluationInboxError(
            "canary and heldout reused a birth receipt"
        )
    content = {
        "owner_id": request["owner_id"],
        "run_id": request["run_id"],
        "request_seq": request["request_seq"],
        "request_sha256": request_document["content_sha256"],
        "sidecar_launch_sha256": request["sidecar_launch_sha256"],
        "bindings_sha256": canonical_sha256(request["bindings"]),
        "target_sha256": canonical_sha256(request["target"]),
        "policy_evaluation_sha256": (
            POLICY_EVALUATION_CONTRACT_SHA256
        ),
        "windows": windows,
    }
    document = _envelope(EVIDENCE_KIND, content)
    validate_evidence_document(
        document,
        request_document=request_document,
        expected_sidecar_launch_sha256=sidecar_launch_sha256,
    )
    return document


_EVIDENCE_WINDOW_KEYS = (
    "allocation",
    "attempt_receipt_root_sha256",
    "ledger",
    "attempts",
)


def validate_evidence_document(
    document: object,
    *,
    request_document: object,
    expected_sidecar_launch_sha256: Optional[str] = None,
) -> Dict[str, object]:
    evidence = _validate_envelope(document, EVIDENCE_KIND)
    row = _exact_dict(
        evidence,
        (
            "owner_id",
            "run_id",
            "request_seq",
            "request_sha256",
            "sidecar_launch_sha256",
            "bindings_sha256",
            "target_sha256",
            "policy_evaluation_sha256",
            "windows",
        ),
        label="evidence content",
    )
    request = validate_request_document(
        request_document,
        expected_sidecar_launch_sha256=expected_sidecar_launch_sha256,
    )
    expected_identity = {
        "owner_id": request["owner_id"],
        "run_id": request["run_id"],
        "request_seq": request["request_seq"],
        "request_sha256": request_document["content_sha256"],
        "sidecar_launch_sha256": request["sidecar_launch_sha256"],
        "bindings_sha256": canonical_sha256(request["bindings"]),
        "target_sha256": canonical_sha256(request["target"]),
        "policy_evaluation_sha256": (
            POLICY_EVALUATION_CONTRACT_SHA256
        ),
    }
    for field, expected in expected_identity.items():
        if row[field] != expected:
            raise EvaluationInboxError(
                "evidence {} does not bind the exact request".format(field)
            )
    windows = row["windows"]
    if (
        type(windows) is not list
        or len(windows) != len(request["windows"])
    ):
        raise EvaluationInboxError(
            "evidence must contain every exact requested window"
        )
    all_samples = []
    all_births = []
    all_proposals = []
    for index, (raw, allocation) in enumerate(
        zip(windows, request["windows"])
    ):
        window = _exact_dict(
            raw,
            _EVIDENCE_WINDOW_KEYS,
            label="evidence.windows[{}]".format(index),
        )
        if window["allocation"] != allocation:
            raise EvaluationInboxError(
                "evidence window allocation differs from request"
            )
        attempts = window["attempts"]
        if type(attempts) is not list or len(attempts) != allocation[
            "proposal_count"
        ]:
            raise EvaluationInboxError(
                "evidence attempt count differs from fixed allocation"
            )
        rows = [
            _validate_attempt(
                attempt,
                allocation=allocation,
                offset=offset,
                proposal_sampler_contract_sha256=request["bindings"][
                    "proposal_sampler_contract_sha256"
                ],
            )
            for offset, attempt in enumerate(attempts)
        ]
        _validate_window_sampling_contract(
            rows,
            count=allocation["proposal_count"],
            selected_arm_key=request["target"][
                "selected_arm_key"
            ],
        )
        root = _sha256(
            window["attempt_receipt_root_sha256"],
            label="attempt_receipt_root_sha256",
        )
        if root != _attempt_receipt_root(rows):
            raise EvaluationInboxError(
                "evidence attempt receipt root mismatch"
            )
        _validate_ledger(
            window["ledger"], rows=rows, role=allocation["role"]
        )
        proposals = [
            item["proposal_receipt_sha256"] for item in rows
        ]
        samples = [item["sample_receipt_sha256"] for item in rows]
        births = [item["birth_receipt_sha256"] for item in rows]
        tasks = [
            item["task_receipt_sha256"]
            for item in rows
            if item["solver_disposition"] == "admitted"
        ]
        if (
            len(proposals) != len(set(proposals))
            or len(samples) != len(set(samples))
            or len(births) != len(set(births))
            or len(tasks) != len(set(tasks))
        ):
            raise EvaluationInboxError(
                "evidence reused a proposal, sample, birth, or task receipt"
            )
        all_proposals.extend(proposals)
        all_samples.extend(samples)
        all_births.extend(births)
    if (
        len(all_proposals) != len(set(all_proposals))
        or len(all_samples) != len(set(all_samples))
        or len(all_births) != len(set(all_births))
    ):
        raise EvaluationInboxError(
            "formal windows reused proposal, sample, or birth receipts"
        )
    return row


def build_ack_document(
    request_document: object,
    evidence_document: object,
    *,
    consumer_code_sha256: str,
    consumer_state_sha256: str,
    consumer_checkpoint: Mapping[str, object],
) -> Dict[str, object]:
    request = validate_request_document(request_document)
    validate_evidence_document(
        evidence_document, request_document=request_document
    )
    content = {
        "owner_id": request["owner_id"],
        "run_id": request["run_id"],
        "request_seq": request["request_seq"],
        "request_sha256": request_document["content_sha256"],
        "evidence_sha256": evidence_document["content_sha256"],
        "consumer_code_sha256": _sha256(
            consumer_code_sha256, label="consumer_code_sha256"
        ),
        "consumer_state_sha256": _sha256(
            consumer_state_sha256,
            label="consumer_state_sha256",
        ),
        "consumer_checkpoint": json.loads(
            _canonical_json_bytes(consumer_checkpoint).decode("ascii")
        ),
        "decision": "accepted",
    }
    verify_artifact_receipt(
        content["consumer_checkpoint"],
        label="ack.consumer_checkpoint",
    )
    document = _envelope(ACK_KIND, content)
    validate_ack_document(
        document,
        request_document=request_document,
        evidence_document=evidence_document,
    )
    return document


def validate_ack_document(
    document: object,
    *,
    request_document: object,
    evidence_document: object,
) -> Dict[str, object]:
    content = _validate_envelope(document, ACK_KIND)
    row = _exact_dict(
        content,
        (
            "owner_id",
            "run_id",
            "request_seq",
            "request_sha256",
            "evidence_sha256",
            "consumer_code_sha256",
            "consumer_state_sha256",
            "consumer_checkpoint",
            "decision",
        ),
        label="ack content",
    )
    request = validate_request_document(request_document)
    validate_evidence_document(
        evidence_document, request_document=request_document
    )
    expected = {
        "owner_id": request["owner_id"],
        "run_id": request["run_id"],
        "request_seq": request["request_seq"],
        "request_sha256": request_document["content_sha256"],
        "evidence_sha256": evidence_document["content_sha256"],
        "decision": "accepted",
    }
    for field, wanted in expected.items():
        if row[field] != wanted:
            raise EvaluationInboxError(
                "ack {} does not bind accepted evidence".format(field)
            )
    _sha256(
        row["consumer_code_sha256"],
        label="ack.consumer_code_sha256",
    )
    _sha256(
        row["consumer_state_sha256"],
        label="ack.consumer_state_sha256",
    )
    verify_artifact_receipt(
        row["consumer_checkpoint"],
        label="ack.consumer_checkpoint",
    )
    return row


def _intervals_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return left_start < right_end and right_start < left_end


class EvaluationInbox:
    """One append-only owner/run namespace."""

    def __init__(self, root: object) -> None:
        raw = Path(os.path.abspath(os.fspath(root)))
        self.root = raw

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("requests", "evidence", "acks"):
            (self.root / name).mkdir(exist_ok=True)
        for path in (
            self.root,
            self.root / "requests",
            self.root / "evidence",
            self.root / "acks",
        ):
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise EvaluationInboxError(
                    "inbox path must be a real directory: {}".format(path)
                )

    def _namespace(
        self, category: str, owner_id: str, run_id: str
    ) -> Path:
        if category not in ("requests", "evidence", "acks"):
            raise EvaluationInboxError("invalid inbox category")
        owner = _identifier(owner_id, label="owner_id")
        run = _identifier(run_id, label="run_id")
        return self.root / category / owner / run

    def _ensure_namespace(
        self, category: str, owner_id: str, run_id: str
    ) -> Path:
        self.initialize()
        namespace = self._namespace(category, owner_id, run_id)
        namespace.mkdir(parents=True, exist_ok=True)
        current = namespace
        while current != self.root:
            info = current.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise EvaluationInboxError(
                    "inbox namespace contains a symlink or non-directory"
                )
            current = current.parent
        return namespace

    def _path(
        self,
        category: str,
        owner_id: str,
        run_id: str,
        request_seq: int,
    ) -> Path:
        namespace = self._namespace(category, owner_id, run_id)
        seq = _plain_int(request_seq, label="request_seq")
        return namespace / "{:020d}.json".format(seq)

    def request_path(
        self, owner_id: str, run_id: str, request_seq: int
    ) -> Path:
        return self._path("requests", owner_id, run_id, request_seq)

    def evidence_path(
        self, owner_id: str, run_id: str, request_seq: int
    ) -> Path:
        return self._path("evidence", owner_id, run_id, request_seq)

    def ack_path(
        self, owner_id: str, run_id: str, request_seq: int
    ) -> Path:
        return self._path("acks", owner_id, run_id, request_seq)

    def load_request(
        self, owner_id: str, run_id: str, request_seq: int
    ) -> Dict[str, object]:
        document = strict_read_json(
            self.request_path(owner_id, run_id, request_seq),
            label="frozen-eval request",
        )
        validate_request_document(
            document,
            expected_owner_id=owner_id,
            expected_run_id=run_id,
            expected_request_seq=request_seq,
        )
        return document

    def load_evidence(
        self, owner_id: str, run_id: str, request_seq: int
    ) -> Dict[str, object]:
        request = self.load_request(owner_id, run_id, request_seq)
        document = strict_read_json(
            self.evidence_path(owner_id, run_id, request_seq),
            label="frozen-eval evidence",
        )
        validate_evidence_document(
            document, request_document=request
        )
        return document

    def load_ack(
        self, owner_id: str, run_id: str, request_seq: int
    ) -> Dict[str, object]:
        request = self.load_request(owner_id, run_id, request_seq)
        evidence = self.load_evidence(owner_id, run_id, request_seq)
        document = strict_read_json(
            self.ack_path(owner_id, run_id, request_seq),
            label="frozen-eval ack",
        )
        validate_ack_document(
            document,
            request_document=request,
            evidence_document=evidence,
        )
        return document

    def _request_files(
        self, owner_id: str, run_id: str
    ) -> List[Path]:
        namespace = self._namespace("requests", owner_id, run_id)
        if not namespace.exists():
            return []
        info = namespace.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise EvaluationInboxError(
                "request namespace is not a real directory"
            )
        result = []
        for path in namespace.iterdir():
            if not path.is_file() or path.is_symlink():
                raise EvaluationInboxError(
                    "request namespace contains a non-regular entry"
                )
            if re.fullmatch(r"[0-9]{20}\.json", path.name) is None:
                raise EvaluationInboxError(
                    "request namespace contains an unknown file"
                )
            result.append(path)
        return sorted(result)

    def _validated_history(
        self, owner_id: str, run_id: str
    ) -> List[Dict[str, object]]:
        files = self._request_files(owner_id, run_id)
        requests = []
        ranges = {"seed": [], "sample": [], "birth": []}
        run_contract = None
        for expected_seq, path in enumerate(files):
            if path.name != "{:020d}.json".format(expected_seq):
                raise EvaluationInboxError(
                    "request sequence has a gap or replayed filename"
                )
            document = self.load_request(
                owner_id, run_id, expected_seq
            )
            content = document["content"]
            if run_contract is None:
                run_contract = content["run_contract_sha256"]
            elif content["run_contract_sha256"] != run_contract:
                raise EvaluationInboxError(
                    "run contract changed inside one owner/run namespace"
                )
            for window in content["windows"]:
                for axis in ("seed", "sample", "birth"):
                    start = window["{}_start".format(axis)]
                    end = window["{}_end_exclusive".format(axis)]
                    if any(
                        _intervals_overlap(start, end, old_start, old_end)
                        for old_start, old_end in ranges[axis]
                    ):
                        raise EvaluationInboxError(
                            "{} ranges overlap across requests".format(axis)
                        )
                    ranges[axis].append((start, end))
            if expected_seq < len(files) - 1:
                self.load_ack(owner_id, run_id, expected_seq)
            requests.append(document)
        return requests

    def publish_request(
        self, document: Mapping[str, object]
    ) -> Path:
        content = validate_request_document(document)
        owner = content["owner_id"]
        run = content["run_id"]
        seq = content["request_seq"]
        self._ensure_namespace("requests", owner, run)
        self._ensure_namespace("evidence", owner, run)
        self._ensure_namespace("acks", owner, run)
        history = self._validated_history(owner, run)
        if seq != len(history):
            raise EvaluationInboxError(
                "request_seq must append exactly after current history"
            )
        if history:
            previous_seq = seq - 1
            self.load_ack(owner, run, previous_seq)
            if (
                content["run_contract_sha256"]
                != history[0]["content"]["run_contract_sha256"]
            ):
                raise EvaluationInboxError(
                    "run contract changed inside one owner/run namespace"
                )
            prior_ranges = {"seed": [], "sample": [], "birth": []}
            for request in history:
                for window in request["content"]["windows"]:
                    for axis in prior_ranges:
                        prior_ranges[axis].append(
                            (
                                window["{}_start".format(axis)],
                                window[
                                    "{}_end_exclusive".format(axis)
                                ],
                            )
                        )
            for window in content["windows"]:
                for axis in prior_ranges:
                    start = window["{}_start".format(axis)]
                    end = window["{}_end_exclusive".format(axis)]
                    if any(
                        _intervals_overlap(start, end, old_start, old_end)
                        for old_start, old_end in prior_ranges[axis]
                    ):
                        raise EvaluationInboxError(
                            "{} ranges overlap across requests".format(axis)
                        )
        return _atomic_publish_json(
            self.request_path(owner, run, seq), document
        )

    def publish_evidence(
        self, document: Mapping[str, object]
    ) -> Path:
        content = _validate_envelope(document, EVIDENCE_KIND)
        owner = _identifier(content["owner_id"], label="owner_id")
        run = _identifier(content["run_id"], label="run_id")
        seq = _plain_int(content["request_seq"], label="request_seq")
        self._validated_history(owner, run)
        request = self.load_request(owner, run, seq)
        validate_evidence_document(
            document, request_document=request
        )
        if self.ack_path(owner, run, seq).exists():
            raise EvaluationInboxError(
                "cannot append evidence after an acknowledgement"
            )
        self._ensure_namespace("evidence", owner, run)
        return _atomic_publish_json(
            self.evidence_path(owner, run, seq), document
        )

    def publish_ack(
        self, document: Mapping[str, object]
    ) -> Path:
        content = _validate_envelope(document, ACK_KIND)
        owner = _identifier(content["owner_id"], label="owner_id")
        run = _identifier(content["run_id"], label="run_id")
        seq = _plain_int(content["request_seq"], label="request_seq")
        self._validated_history(owner, run)
        request = self.load_request(owner, run, seq)
        evidence = self.load_evidence(owner, run, seq)
        validate_ack_document(
            document,
            request_document=request,
            evidence_document=evidence,
        )
        self._ensure_namespace("acks", owner, run)
        return _atomic_publish_json(
            self.ack_path(owner, run, seq), document
        )

    def next_pending_request(
        self, owner_id: str, run_id: str
    ) -> Optional[Dict[str, object]]:
        owner = _identifier(owner_id, label="owner_id")
        run = _identifier(run_id, label="run_id")
        history = self._validated_history(owner, run)
        for seq, request in enumerate(history):
            evidence_path = self.evidence_path(owner, run, seq)
            ack_path = self.ack_path(owner, run, seq)
            if ack_path.exists():
                self.load_ack(owner, run, seq)
                continue
            if evidence_path.exists():
                self.load_evidence(owner, run, seq)
                return None
            return request
        return None


_COORDINATOR_DOCUMENT = {
    "schema_version": SCHEMA_VERSION,
    "kind": "action_ball_frozen_eval_inbox_coordinator",
    "authority": (
        "V4 privately allocates every seed/sample/birth range; coordinator "
        "only serializes the authority-derived request plan"
    ),
    "modes": (
        "one fixed rolling-100 scheduler window or one fixed "
        "canary320+heldout960 formal pair"
    ),
    "commit": (
        "curriculum must consume the opaque capability/release before ACK; "
        "ACK is separately prepared, then published only after an exact "
        "resume checkpoint exists"
    ),
    "resume": (
        "records bind request/evidence/allocation/result/ACK identities; "
        "V4 and source independently replay and validate their event tapes"
    ),
}
FROZEN_EVALUATION_INBOX_COORDINATOR_CONTRACT_SHA256 = (
    canonical_sha256(_COORDINATOR_DOCUMENT)
)


class FrozenEvaluationInboxCoordinator:
    """Two-phase trainer adapter for periodic sidecar evaluation."""

    _STAGE_REVISION = {
        "published": 1,
        "result_ready": 2,
        "curriculum_consumed": 3,
        "ack_prepared": 4,
        "acked": 5,
    }

    def __init__(
        self,
        *,
        inbox: EvaluationInbox,
        owner_id: str,
        run_id: str,
        sidecar_launch_sha256: str,
        consumer_code_sha256: str,
        evaluator_authority: object,
    ) -> None:
        if not isinstance(inbox, EvaluationInbox):
            raise EvaluationInboxError(
                "coordinator requires an EvaluationInbox"
            )
        for method in (
            "freeze_checkpoint",
            "open_window",
            "sidecar_request_plan",
            "complete_sidecar_window",
            "issue_or_resume_sidecar_release",
            "pending_capability",
            "pending_release",
            "assert_sidecar_result_consumed",
            "assert_sidecar_request_consumed",
            "state_dict",
            "load_state_dict",
        ):
            if not callable(getattr(evaluator_authority, method, None)):
                raise EvaluationInboxError(
                    f"evaluator authority lacks {method}()"
                )
        authority_owner = _sha256(
            getattr(evaluator_authority, "state_owner_sha256", None),
            label="evaluator_authority.state_owner_sha256",
        )
        self._inbox = inbox
        self._inbox.initialize()
        self._owner_id = _identifier(owner_id, label="owner_id")
        self._run_id = _identifier(run_id, label="run_id")
        self._sidecar_launch_sha256 = _sha256(
            sidecar_launch_sha256,
            label="sidecar_launch_sha256",
        )
        self._consumer_code_sha256 = _sha256(
            consumer_code_sha256,
            label="consumer_code_sha256",
        )
        self._authority = evaluator_authority
        self._authority_owner_sha256 = authority_owner
        self._records: Dict[int, Dict[str, object]] = {}
        self._revision = 0
        self.state_owner_sha256 = canonical_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "coordinator_contract_sha256": (
                    FROZEN_EVALUATION_INBOX_COORDINATOR_CONTRACT_SHA256
                ),
                "inbox_root": str(self._inbox.root),
                "owner_id": self._owner_id,
                "run_id": self._run_id,
                "sidecar_launch_sha256": self._sidecar_launch_sha256,
                "consumer_code_sha256": self._consumer_code_sha256,
                "evaluator_state_owner_sha256": authority_owner,
            }
        )

    @staticmethod
    def _empty_record(
        *,
        seq: int,
        request_kind: str,
        request_sha256: str,
        allocations: Sequence[str],
    ) -> Dict[str, object]:
        return {
            "request_seq": seq,
            "request_kind": request_kind,
            "request_sha256": request_sha256,
            "allocation_sha256": list(allocations),
            "evidence_sha256": "",
            "result_id": "",
            "stage": "published",
            "consumer_state_sha256": "",
            "consumer_checkpoint": None,
            "ack_sha256": "",
        }

    def state_fingerprint(self) -> int:
        return self._revision

    def state_dict(self) -> Dict[str, object]:
        document = {
            "schema_version": SCHEMA_VERSION,
            "state_owner_sha256": self.state_owner_sha256,
            "records": [
                dict(self._records[seq]) for seq in sorted(self._records)
            ],
            "revision": self._revision,
        }
        document["state_sha256"] = canonical_sha256(document)
        return document

    def _validate_record(self, raw: object, *, expected_seq: int) -> Dict[str, object]:
        record = _exact_dict(
            raw,
            (
                "request_seq",
                "request_kind",
                "request_sha256",
                "allocation_sha256",
                "evidence_sha256",
                "result_id",
                "stage",
                "consumer_state_sha256",
                "consumer_checkpoint",
                "ack_sha256",
            ),
            label=f"coordinator.records[{expected_seq}]",
        )
        seq = _plain_int(
            record["request_seq"],
            label="coordinator request_seq",
        )
        if seq != expected_seq:
            raise EvaluationInboxError(
                "coordinator request sequence is not contiguous"
            )
        kind = record["request_kind"]
        allocations = record["allocation_sha256"]
        expected_allocations = 1 if kind == "scheduler" else 2
        if (
            kind not in ("scheduler", "formal")
            or type(allocations) is not list
            or len(allocations) != expected_allocations
        ):
            raise EvaluationInboxError(
                "coordinator request kind/allocation shape is invalid"
            )
        for digest in allocations:
            _sha256(digest, label="coordinator allocation_sha256")
        request_sha = _sha256(
            record["request_sha256"],
            label="coordinator request_sha256",
        )
        request = self._inbox.load_request(
            self._owner_id,
            self._run_id,
            seq,
        )
        if request["content_sha256"] != request_sha:
            raise EvaluationInboxError(
                "coordinator request bytes drifted"
            )
        roles = tuple(
            window["role"] for window in request["content"]["windows"]
        )
        if (
            (kind == "scheduler" and roles != ("scheduler",))
            or (
                kind == "formal"
                and roles != ("frozen_canary", "frozen_heldout")
            )
        ):
            raise EvaluationInboxError(
                "coordinator request role differs from state"
            )
        stage = record["stage"]
        if stage not in self._STAGE_REVISION:
            raise EvaluationInboxError(
                "coordinator record stage is invalid"
            )
        if stage != "published":
            evidence = self._inbox.load_evidence(
                self._owner_id,
                self._run_id,
                seq,
            )
            evidence_sha = _sha256(
                record["evidence_sha256"],
                label="coordinator evidence_sha256",
            )
            result_id = _sha256(
                record["result_id"],
                label="coordinator result_id",
            )
            if evidence["content_sha256"] != evidence_sha:
                raise EvaluationInboxError(
                    "coordinator evidence bytes drifted"
                )
            consumed_evidence = (
                self._authority.assert_sidecar_request_consumed(seq)
            )
            if consumed_evidence != evidence_sha:
                raise EvaluationInboxError(
                    "coordinator/source evidence identity drifted"
                )
            if stage == "result_ready":
                if kind == "scheduler":
                    pending = self._authority.pending_capability(result_id)
                    pending_id = pending.capability_id
                else:
                    pending = self._authority.pending_release(result_id)
                    pending_id = pending.release_id
                if pending_id != result_id:
                    raise EvaluationInboxError(
                        "coordinator pending result identity drifted"
                    )
            elif stage in (
                "curriculum_consumed",
                "ack_prepared",
                "acked",
            ):
                self._authority.assert_sidecar_result_consumed(
                    request_kind=kind,
                    result_id=result_id,
                )
        else:
            if any(
                record[name]
                for name in (
                    "evidence_sha256",
                    "result_id",
                    "consumer_state_sha256",
                    "ack_sha256",
                )
            ) or record["consumer_checkpoint"] is not None:
                raise EvaluationInboxError(
                    "published coordinator record carries future state"
                )
        if stage == "ack_prepared":
            if any(
                record[name]
                for name in (
                    "consumer_state_sha256",
                    "ack_sha256",
                )
            ) or record["consumer_checkpoint"] is not None:
                raise EvaluationInboxError(
                    "prepared ACK record carries an uncommitted checkpoint"
                )
        elif stage == "acked":
            state_sha = _sha256(
                record["consumer_state_sha256"],
                label="coordinator consumer_state_sha256",
            )
            checkpoint = record["consumer_checkpoint"]
            verify_artifact_receipt(
                checkpoint,
                label="coordinator consumer_checkpoint",
            )
            ack = self._inbox.load_ack(
                self._owner_id,
                self._run_id,
                seq,
            )
            ack_sha = _sha256(
                record["ack_sha256"],
                label="coordinator ack_sha256",
            )
            if (
                ack["content_sha256"] != ack_sha
                or ack["content"]["consumer_code_sha256"]
                != self._consumer_code_sha256
                or ack["content"]["consumer_state_sha256"] != state_sha
                or ack["content"]["consumer_checkpoint"] != checkpoint
            ):
                raise EvaluationInboxError(
                    "coordinator ACK barrier identity drifted"
                )
        elif stage not in ("ack_prepared",):
            if any(
                record[name]
                for name in (
                    "consumer_state_sha256",
                    "ack_sha256",
                )
            ) or record["consumer_checkpoint"] is not None:
                raise EvaluationInboxError(
                    "coordinator record carries an early ACK barrier"
                )
        return dict(record)

    def load_state_dict(self, value: object) -> None:
        row = _exact_dict(
            value,
            (
                "schema_version",
                "state_owner_sha256",
                "records",
                "revision",
                "state_sha256",
            ),
            label="evaluation inbox coordinator state",
        )
        unsigned = dict(row)
        digest = _sha256(
            unsigned.pop("state_sha256"),
            label="coordinator state_sha256",
        )
        if canonical_sha256(unsigned) != digest:
            raise EvaluationInboxError(
                "coordinator state digest mismatch"
            )
        if (
            row["schema_version"] != SCHEMA_VERSION
            or row["state_owner_sha256"] != self.state_owner_sha256
            or type(row["records"]) is not list
        ):
            raise EvaluationInboxError(
                "coordinator state binding/schema mismatch"
            )
        parsed = {
            seq: self._validate_record(raw, expected_seq=seq)
            for seq, raw in enumerate(row["records"])
        }
        expected_revision = sum(
            self._STAGE_REVISION[record["stage"]]
            for record in parsed.values()
        )
        revision = _plain_int(
            row["revision"],
            label="coordinator revision",
        )
        if revision != expected_revision:
            raise EvaluationInboxError(
                "coordinator revision differs from its exact tape"
            )
        self._records = parsed
        self._revision = revision

    def publish_sessions(
        self,
        *,
        sessions: Sequence[object],
        bindings: Mapping[str, object],
    ) -> int:
        if self._records and self._records[max(self._records)][
            "stage"
        ] != "acked":
            raise EvaluationInboxError(
                "previous evaluation request is not checkpointed and ACKed"
            )
        plan = self._authority.sidecar_request_plan(sessions)
        normalized_bindings = json.loads(
            _canonical_json_bytes(bindings).decode("ascii")
        )
        _validate_bindings(normalized_bindings)
        if (
            normalized_bindings["checkpoint"]["sha256"]
            != plan["checkpoint_sha256"]
        ):
            raise EvaluationInboxError(
                "sidecar request checkpoint differs from V4 snapshot bytes"
            )
        if (
            normalized_bindings["policy_generation"]
            != plan["policy_generation"]
        ):
            raise EvaluationInboxError(
                "sidecar request policy generation differs from V4 "
                "snapshot generation"
            )
        seq = len(self._records)
        request = build_request_document(
            owner_id=self._owner_id,
            run_id=self._run_id,
            request_seq=seq,
            sidecar_launch_sha256=self._sidecar_launch_sha256,
            bindings=normalized_bindings,
            target=plan["target"],
            seed_start=plan["seed_start"],
            sample_start=plan["sample_start"],
            birth_start=plan["birth_start"],
            request_kind=plan["request_kind"],
        )
        self._inbox.publish_request(request)
        self._records[seq] = self._empty_record(
            seq=seq,
            request_kind=plan["request_kind"],
            request_sha256=request["content_sha256"],
            allocations=plan["allocation_sha256"],
        )
        self._revision += 1
        return seq

    def reconcile_published_request(self) -> Optional[int]:
        """Recover the one crash window after request fsync but before save.

        The caller resumes from the immediately preceding exact checkpoint.
        Therefore the authority tape has not yet frozen this policy or opened
        its windows, while the append-only request may already exist.  Rebuild
        those opaque allocations from the request's exact checkpoint bytes and
        target, then require the complete regenerated request to be byte-equal.
        No evidence is consumed here.
        """

        if self._records and self._records[max(self._records)][
            "stage"
        ] != "acked":
            return None
        seq = len(self._records)
        path = self._inbox.request_path(
            self._owner_id, self._run_id, seq
        )
        if not path.exists():
            return None
        if self._inbox.request_path(
            self._owner_id, self._run_id, seq + 1
        ).exists():
            raise EvaluationInboxError(
                "cannot reconcile a request sequence gap"
            )
        request = self._inbox.load_request(
            self._owner_id,
            self._run_id,
            seq,
        )
        content = validate_request_document(
            request,
            expected_owner_id=self._owner_id,
            expected_run_id=self._run_id,
            expected_request_seq=seq,
            expected_sidecar_launch_sha256=(
                self._sidecar_launch_sha256
            ),
        )
        bindings = content["bindings"]
        checkpoint_bytes = read_artifact_receipt_bytes(
            bindings["checkpoint"],
            label="reconciled frozen checkpoint",
        )
        target = content["target"]
        windows = content["windows"]
        if tuple(window["role"] for window in windows) == (
            "scheduler",
        ):
            request_kind = "scheduler"
        else:
            request_kind = "formal"

        from whole_body_tracking.tasks.tracking.mdp.action_ball_curriculum import (
            ActionProfileKey,
        )

        key = ActionProfileKey(
            action_uid=target["action_uid"],
            profile_sha256=target["profile_sha256"],
            mobility=target["mobility_mode"],
        )
        before = self._authority.state_dict()
        try:
            snapshot = self._authority.freeze_checkpoint(
                checkpoint_bytes,
                policy_generation=bindings["policy_generation"],
            )
            sessions = tuple(
                self._authority.open_window(
                    snapshot=snapshot,
                    key=key,
                    evidence_role=window["role"],
                    domain_epoch=target["domain_epoch"],
                    stratum=target["stratum"],
                    selected_arm_key=target["selected_arm_key"],
                    selection_round=target["selection_round"],
                    arm_levels=tuple(target["arm_levels"]),
                    rho=target["rho"],
                )
                for window in windows
            )
            plan = self._authority.sidecar_request_plan(sessions)
            rebuilt = build_request_document(
                owner_id=self._owner_id,
                run_id=self._run_id,
                request_seq=seq,
                sidecar_launch_sha256=self._sidecar_launch_sha256,
                bindings=bindings,
                target=plan["target"],
                seed_start=plan["seed_start"],
                sample_start=plan["sample_start"],
                birth_start=plan["birth_start"],
                request_kind=plan["request_kind"],
            )
            if (
                plan["request_kind"] != request_kind
                or rebuilt != request
            ):
                raise EvaluationInboxError(
                    "published request cannot be regenerated from the "
                    "preceding exact authority state"
                )
            self._records[seq] = self._empty_record(
                seq=seq,
                request_kind=plan["request_kind"],
                request_sha256=request["content_sha256"],
                allocations=plan["allocation_sha256"],
            )
            self._revision += 1
            return seq
        except Exception:
            self._authority.load_state_dict(before)
            raise

    def consume_evidence(self, request_seq: int) -> object:
        seq = _plain_int(request_seq, label="request_seq")
        try:
            record = self._records[seq]
        except KeyError as exc:
            raise EvaluationInboxError(
                "coordinator request is unknown"
            ) from exc
        if record["stage"] != "published":
            raise EvaluationInboxError(
                "coordinator evidence was already consumed"
            )
        evidence = self._inbox.load_evidence(
            self._owner_id,
            self._run_id,
            seq,
        )
        capabilities = [
            self._authority.complete_sidecar_window(allocation)
            for allocation in record["allocation_sha256"]
        ]
        if record["request_kind"] == "scheduler":
            result = capabilities[0]
            result_id = result.capability_id
        else:
            result = self._authority.issue_or_resume_sidecar_release(
                canary=capabilities[0],
                heldout=capabilities[1],
            )
            result_id = result.release_id
        consumed_evidence = (
            self._authority.assert_sidecar_request_consumed(seq)
        )
        if consumed_evidence != evidence["content_sha256"]:
            raise EvaluationInboxError(
                "V4/source consumed another evidence document"
            )
        record["evidence_sha256"] = evidence["content_sha256"]
        record["result_id"] = result_id
        record["stage"] = "result_ready"
        self._revision += 1
        return result

    def pending_result(self, request_seq: int) -> object:
        seq = _plain_int(request_seq, label="request_seq")
        record = self._records.get(seq)
        if record is None or record["stage"] != "result_ready":
            raise EvaluationInboxError(
                "coordinator has no pending opaque result"
            )
        if record["request_kind"] == "scheduler":
            return self._authority.pending_capability(
                record["result_id"]
            )
        return self._authority.pending_release(record["result_id"])

    def mark_curriculum_consumed(self, request_seq: int) -> None:
        seq = _plain_int(request_seq, label="request_seq")
        record = self._records.get(seq)
        if record is None or record["stage"] != "result_ready":
            raise EvaluationInboxError(
                "coordinator result is not ready for curriculum commit"
            )
        self._authority.assert_sidecar_result_consumed(
            request_kind=record["request_kind"],
            result_id=record["result_id"],
        )
        record["stage"] = "curriculum_consumed"
        self._revision += 1

    def prepare_ack(self, request_seq: int) -> None:
        seq = _plain_int(request_seq, label="request_seq")
        record = self._records.get(seq)
        if record is None or record["stage"] != "curriculum_consumed":
            raise EvaluationInboxError(
                "ACK preparation requires committed curriculum ingestion"
            )
        evidence_sha = self._authority.assert_sidecar_request_consumed(
            seq
        )
        if evidence_sha != record["evidence_sha256"]:
            raise EvaluationInboxError(
                "ACK preparation evidence identity drifted"
            )
        record["stage"] = "ack_prepared"
        self._revision += 1

    def publish_ack(
        self,
        request_seq: int,
        *,
        consumer_state_sha256: str,
        consumer_checkpoint: Mapping[str, object],
    ) -> Path:
        seq = _plain_int(request_seq, label="request_seq")
        record = self._records.get(seq)
        if record is None or record["stage"] != "ack_prepared":
            raise EvaluationInboxError(
                "ACK publish requires a prepared persistence barrier"
            )
        request = self._inbox.load_request(
            self._owner_id,
            self._run_id,
            seq,
        )
        evidence = self._inbox.load_evidence(
            self._owner_id,
            self._run_id,
            seq,
        )
        checkpoint = json.loads(
            _canonical_json_bytes(consumer_checkpoint).decode("ascii")
        )
        verify_artifact_receipt(
            checkpoint,
            label="coordinator consumer_checkpoint",
        )
        state_sha = _sha256(
            consumer_state_sha256,
            label="consumer_state_sha256",
        )
        ack = build_ack_document(
            request,
            evidence,
            consumer_code_sha256=self._consumer_code_sha256,
            consumer_state_sha256=state_sha,
            consumer_checkpoint=checkpoint,
        )
        path = self._inbox.publish_ack(ack)
        record["consumer_state_sha256"] = state_sha
        record["consumer_checkpoint"] = checkpoint
        record["ack_sha256"] = ack["content_sha256"]
        record["stage"] = "acked"
        self._revision += 1
        return path

    def reconcile_ack(self, request_seq: int) -> None:
        """Recover the post-checkpoint ACK transition after a crash."""

        seq = _plain_int(request_seq, label="request_seq")
        record = self._records.get(seq)
        if record is None or record["stage"] != "ack_prepared":
            raise EvaluationInboxError(
                "ACK reconciliation requires prepared state"
            )
        ack = self._inbox.load_ack(
            self._owner_id,
            self._run_id,
            seq,
        )
        if (
            ack["content"]["consumer_code_sha256"]
            != self._consumer_code_sha256
        ):
            raise EvaluationInboxError(
                "ACK was published by another consumer code root"
            )
        record["consumer_state_sha256"] = ack["content"][
            "consumer_state_sha256"
        ]
        record["consumer_checkpoint"] = ack["content"][
            "consumer_checkpoint"
        ]
        record["ack_sha256"] = ack["content_sha256"]
        record["stage"] = "acked"
        self._revision += 1


_ATTEMPT_SOURCE_DOCUMENT = {
    "schema_version": SCHEMA_VERSION,
    "kind": "action_ball_frozen_eval_inbox_attempt_source",
    "protocol_contract_sha256": EVALUATION_INBOX_CONTRACT_SHA256,
    "authority_boundary": (
        "wire JSON is raw evidence only; runtime Frozen* events are rebuilt "
        "inside the V4 evaluator process"
    ),
    "terminal": (
        "worker supplies raw booleans; runtime classifier owns the outcome"
    ),
    "state": (
        "stable owner/run namespace identity independent of append history; "
        "exact request/evidence/row/reservation replay tape plus lifecycle "
        "stage; optional fixed request-seq compatibility filter"
    ),
    "matching": (
        "dynamic multi-request lookup by checkpoint/role/ranges/target/"
        "domain-levels with one immutable evidence row per reservation"
    ),
    "lifecycle_replay": (
        "source-owned next-event stage drives solver/install/start/terminal; "
        "infrastructure-invalid may close before install or start without "
        "becoming a policy failure"
    ),
}
FROZEN_EVALUATION_INBOX_ATTEMPT_SOURCE_CONTRACT_SHA256 = (
    canonical_sha256(_ATTEMPT_SOURCE_DOCUMENT)
)
FROZEN_EVALUATION_INBOX_ATTEMPT_SOURCE_PATH = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/"
    "action_ball_evaluation_inbox.py"
)


def _runtime_module(explicit: object = None) -> object:
    if explicit is not None:
        return explicit
    try:
        from . import action_ball_runtime as runtime  # type: ignore

        return runtime
    except (ImportError, ValueError):
        runtime = sys.modules.get("action_ball_runtime")
        if runtime is None:
            raise EvaluationInboxError(
                "FrozenSidecarInboxAttemptSource requires "
                "action_ball_runtime"
            )
        return runtime


def _source_code_sha256() -> str:
    digest, _size = _hash_regular_file(
        Path(__file__).resolve(),
        label="evaluation inbox attempt-source code",
    )
    return digest


class FrozenSidecarInboxAttemptSource:
    """Replay accepted raw sidecar evidence through the V4 source API.

    The class never deserializes a capability.  It reconstructs each
    ``FrozenIssuedProposal``/solver/lifecycle/terminal event by calling the
    runtime dataclass ``create`` methods, so the evaluator remains the sole
    minter of opaque handles and release receipts.
    """

    source_contract_sha256 = (
        FROZEN_EVALUATION_INBOX_ATTEMPT_SOURCE_CONTRACT_SHA256
    )
    source_path = FROZEN_EVALUATION_INBOX_ATTEMPT_SOURCE_PATH

    def __init__(
        self,
        *,
        inbox: EvaluationInbox,
        owner_id: str,
        run_id: str,
        request_seq: Optional[int] = None,
        runtime_module: object = None,
    ) -> None:
        if not isinstance(inbox, EvaluationInbox):
            raise EvaluationInboxError(
                "attempt source requires an EvaluationInbox"
            )
        self._runtime = _runtime_module(runtime_module)
        self.source_code_sha256 = _source_code_sha256()
        self._inbox = inbox
        self._inbox.initialize()
        self._owner_id = _identifier(owner_id, label="owner_id")
        self._run_id = _identifier(run_id, label="run_id")
        self._fixed_request_seq = (
            None
            if request_seq is None
            else _plain_int(request_seq, label="request_seq")
        )
        self._records = {}
        self._revision = 0
        self._history_cache_signature = None
        self._history_cache = ()
        self._document_cache = {}
        self.state_owner_sha256 = canonical_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "source_contract_sha256": self.source_contract_sha256,
                "source_code_sha256": self.source_code_sha256,
                "inbox_root": str(self._inbox.root),
                "owner_id": self._owner_id,
                "run_id": self._run_id,
                "fixed_request_seq": self._fixed_request_seq,
            }
        )

    def state_fingerprint(self) -> int:
        return self._revision

    def state_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "state_owner_sha256": self.state_owner_sha256,
            "fixed_request_seq": self._fixed_request_seq,
            "records": {
                reservation: dict(record)
                for reservation, record in sorted(
                    self._records.items()
                )
            },
            "revision": self._revision,
        }

    def load_state_dict(self, value: object) -> None:
        row = _exact_dict(
            value,
            (
                "schema_version",
                "state_owner_sha256",
                "fixed_request_seq",
                "records",
                "revision",
            ),
            label="inbox attempt-source state",
        )
        expected = {
            "schema_version": SCHEMA_VERSION,
            "state_owner_sha256": self.state_owner_sha256,
            "fixed_request_seq": self._fixed_request_seq,
        }
        if (
            type(row["schema_version"]) is not int
            or row["schema_version"] != SCHEMA_VERSION
        ):
            raise EvaluationInboxError(
                "attempt-source state schema_version mismatch"
            )
        for name, wanted in expected.items():
            if row[name] != wanted:
                raise EvaluationInboxError(
                    "attempt-source state {} mismatch".format(name)
                )
        records = row["records"]
        if type(records) is not dict:
            raise EvaluationInboxError(
                "attempt-source records must be a JSON object"
            )
        parsed = {}
        allowed_stages = {
            "issued",
            "solver_rejected",
            "solver_admitted",
            "installed",
            "started",
            "terminal",
        }
        stage_revisions = {
            "issued": 1,
            "solver_rejected": 2,
            "solver_admitted": 2,
            "installed": 3,
            "started": 4,
        }
        used_keys = set()
        expected_revision = 0
        for reservation, raw in records.items():
            _sha256(reservation, label="reservation_sha256")
            record = _exact_dict(
                raw,
                (
                    "request_seq",
                    "request_sha256",
                    "evidence_sha256",
                    "attempt_sha256",
                    "role",
                    "offset",
                    "stage",
                ),
                label="attempt-source record",
            )
            request_seq = _plain_int(
                record["request_seq"],
                label="attempt-source request_seq",
            )
            if (
                self._fixed_request_seq is not None
                and request_seq != self._fixed_request_seq
            ):
                raise EvaluationInboxError(
                    "attempt-source state escapes fixed request_seq"
                )
            _sha256(
                record["request_sha256"],
                label="attempt-source request_sha256",
            )
            _sha256(
                record["evidence_sha256"],
                label="attempt-source evidence_sha256",
            )
            _sha256(
                record["attempt_sha256"],
                label="attempt-source attempt_sha256",
            )
            role = record["role"]
            if role not in (
                "scheduler",
                "frozen_canary",
                "frozen_heldout",
            ):
                raise EvaluationInboxError(
                    "attempt-source record role is invalid"
                )
            offset = _plain_int(
                record["offset"], label="attempt-source offset"
            )
            if record["stage"] not in allowed_stages:
                raise EvaluationInboxError(
                    "attempt-source record stage is invalid"
                )
            row_key = (request_seq, role, offset)
            if row_key in used_keys:
                raise EvaluationInboxError(
                    "attempt-source state reuses one evidence row"
                )
            used_keys.add(row_key)
            evidence_row = self._load_record_row(record)
            disposition = evidence_row["solver_disposition"]
            if disposition in ("physics_invalid", "rejected"):
                allowed_for_row = ("issued", "solver_rejected")
            else:
                allowed_for_row = (
                    "issued",
                    "solver_admitted",
                    "installed",
                    "started",
                    "terminal",
                )
            if record["stage"] not in allowed_for_row:
                raise EvaluationInboxError(
                    "attempt-source stage disagrees with evidence disposition"
                )
            if (
                record["stage"] in ("installed", "started")
                and not evidence_row["installed"]
            ):
                raise EvaluationInboxError(
                    "attempt-source state invents an install event"
                )
            if (
                record["stage"] == "started"
                and not evidence_row["started"]
            ):
                raise EvaluationInboxError(
                    "attempt-source state invents a start event"
                )
            if record["stage"] == "terminal":
                expected_revision += (
                    3
                    + int(bool(evidence_row["installed"]))
                    + int(bool(evidence_row["started"]))
                )
            else:
                expected_revision += stage_revisions[record["stage"]]
            parsed[reservation] = {
                "request_seq": request_seq,
                "request_sha256": record["request_sha256"],
                "evidence_sha256": record["evidence_sha256"],
                "attempt_sha256": record["attempt_sha256"],
                "role": role,
                "offset": offset,
                "stage": record["stage"],
            }
        revision = _plain_int(
            row["revision"], label="attempt-source revision"
        )
        if revision != expected_revision:
            raise EvaluationInboxError(
                "attempt-source revision does not equal its exact event tape"
            )
        self._records = parsed
        self._revision = revision

    def _assert_runtime_request(self, request: object) -> None:
        runtime = self._runtime
        request_type = getattr(
            runtime, "FrozenEvaluationProposalRequest", None
        )
        if request_type is None or not isinstance(request, request_type):
            raise EvaluationInboxError(
                "attempt source requires FrozenEvaluationProposalRequest"
            )
    @staticmethod
    def _allocation_for_role(
        request_content: Mapping[str, object], role: str
    ) -> Mapping[str, object]:
        matches = [
            window
            for window in request_content["windows"]
            if window["role"] == role
        ]
        if len(matches) != 1:
            raise EvaluationInboxError(
                "request has no exact allocation for {}".format(role)
            )
        return matches[0]

    def _request_matches(
        self,
        runtime_request: object,
        request_content: Mapping[str, object],
        allocation: Mapping[str, object],
        evidence_row: Mapping[str, object],
    ) -> bool:
        runtime = self._runtime
        if runtime_request.evidence_role != allocation["role"]:
            return False
        offset = runtime_request.proposal_offset
        if not 0 <= offset < allocation["proposal_count"]:
            return False
        target = request_content["target"]
        checkpoint_sha256 = request_content["bindings"]["checkpoint"][
            "sha256"
        ]
        expected = (
            checkpoint_sha256,
            request_content["bindings"]["policy_generation"],
            allocation["seed_start"] + offset,
            allocation["sample_start"] + offset,
            allocation["birth_start"] + offset,
            target["action_uid"],
            target["profile_sha256"],
            target["mobility_mode"],
            target["domain_epoch"],
            target["selected_arm_key"],
        )
        actual = (
            runtime_request.policy_checkpoint_sha256,
            runtime_request.policy_generation,
            runtime_request.seed,
            runtime_request.sample_index,
            runtime_request.birth_index,
            runtime_request.action_uid,
            runtime_request.profile_sha256,
            runtime_request.mobility_mode,
            runtime_request.domain_epoch,
            runtime_request.selected_arm_key,
        )
        if actual != expected:
            return False
        arm_keys = tuple(getattr(runtime, "ARM_KEYS", ()))
        if not arm_keys:
            raise EvaluationInboxError(
                "runtime ARM_KEYS contract is unavailable"
            )
        levels = tuple(
            float(getattr(runtime_request.domain_levels, name))
            for name in arm_keys
        )
        if levels != tuple(
            float(value) for value in target["arm_levels"]
        ):
            return False
        if (
            evidence_row["sampling_stratum"] == "frontier"
            and evidence_row["frontier_arm"] not in arm_keys
        ):
            raise EvaluationInboxError(
                "sidecar frontier arm is outside runtime ARM_KEYS"
            )
        return True

    def _history_with_evidence(
        self,
    ) -> Iterable[
        Tuple[
            int,
            Dict[str, object],
            Dict[str, object],
            Dict[str, object],
            Dict[str, object],
        ]
    ]:
        signature_before = self._namespace_signature()
        if signature_before == self._history_cache_signature:
            return iter(self._history_cache)
        requests = self._inbox._validated_history(
            self._owner_id, self._run_id
        )
        loaded = []
        for request_document in requests:
            request_content = validate_request_document(
                request_document,
                expected_owner_id=self._owner_id,
                expected_run_id=self._run_id,
            )
            request_seq = request_content["request_seq"]
            if (
                self._fixed_request_seq is not None
                and request_seq != self._fixed_request_seq
            ):
                continue
            evidence_path = self._inbox.evidence_path(
                self._owner_id, self._run_id, request_seq
            )
            if not evidence_path.exists():
                continue
            evidence_document = self._inbox.load_evidence(
                self._owner_id, self._run_id, request_seq
            )
            evidence_content = validate_evidence_document(
                evidence_document,
                request_document=request_document,
            )
            self._document_cache[request_seq] = (
                self._document_signature(
                    self._inbox.request_path(
                        self._owner_id,
                        self._run_id,
                        request_seq,
                    )
                ),
                self._document_signature(evidence_path),
                request_document,
                evidence_document,
                request_content,
                evidence_content,
            )
            loaded.append(
                (
                    request_seq,
                    request_document,
                    request_content,
                    evidence_document,
                    evidence_content,
                )
            )
        signature_after = self._namespace_signature()
        if signature_after != signature_before:
            raise EvaluationInboxError(
                "inbox history changed while refreshing attempt-source tape"
            )
        self._history_cache = tuple(loaded)
        self._history_cache_signature = signature_after
        return iter(self._history_cache)

    def _namespace_signature(self) -> Tuple[object, ...]:
        result = []
        for category in ("requests", "evidence", "acks"):
            namespace = self._inbox._namespace(
                category, self._owner_id, self._run_id
            )
            if not namespace.exists():
                result.append((category, None))
                continue
            info = namespace.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise EvaluationInboxError(
                    "attempt-source namespace is not a real directory"
                )
            entries = []
            for path in sorted(namespace.iterdir()):
                item = path.lstat()
                entries.append(
                    (
                        path.name,
                        item.st_dev,
                        item.st_ino,
                        item.st_mode,
                        item.st_size,
                        item.st_mtime_ns,
                        item.st_nlink,
                    )
                )
            result.append((category, tuple(entries)))
        return tuple(result)

    @staticmethod
    def _document_signature(path: Path) -> Tuple[int, ...]:
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise EvaluationInboxError(
                "attempt-source document is missing: {}".format(path)
            ) from exc
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise EvaluationInboxError(
                "attempt-source document is not immutable regular data"
            )
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            getattr(info, "st_ctime_ns", int(info.st_ctime * 1e9)),
            info.st_nlink,
        )

    def _load_request_evidence_documents(
        self, request_seq: int
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        request_path = self._inbox.request_path(
            self._owner_id, self._run_id, request_seq
        )
        evidence_path = self._inbox.evidence_path(
            self._owner_id, self._run_id, request_seq
        )
        request_signature = self._document_signature(request_path)
        evidence_signature = self._document_signature(evidence_path)
        cached = self._document_cache.get(request_seq)
        if (
            cached is not None
            and cached[0] == request_signature
            and cached[1] == evidence_signature
        ):
            return cached[2], cached[3]
        request_document = self._inbox.load_request(
            self._owner_id, self._run_id, request_seq
        )
        evidence_document = self._inbox.load_evidence(
            self._owner_id, self._run_id, request_seq
        )
        request_content = validate_request_document(
            request_document,
            expected_owner_id=self._owner_id,
            expected_run_id=self._run_id,
            expected_request_seq=request_seq,
        )
        evidence_content = validate_evidence_document(
            evidence_document,
            request_document=request_document,
        )
        if (
            self._document_signature(request_path) != request_signature
            or self._document_signature(evidence_path)
            != evidence_signature
        ):
            raise EvaluationInboxError(
                "attempt-source document changed while loading"
            )
        self._document_cache[request_seq] = (
            request_signature,
            evidence_signature,
            request_document,
            evidence_document,
            request_content,
            evidence_content,
        )
        return request_document, evidence_document

    def _load_request_evidence_contents(
        self, request_seq: int
    ) -> Tuple[
        Dict[str, object],
        Dict[str, object],
        Dict[str, object],
        Dict[str, object],
    ]:
        request_document, evidence_document = (
            self._load_request_evidence_documents(request_seq)
        )
        cached = self._document_cache.get(request_seq)
        if (
            cached is None
            or len(cached) != 6
            or cached[2] is not request_document
            or cached[3] is not evidence_document
        ):
            raise EvaluationInboxError(
                "attempt-source validated document cache drifted"
            )
        return (
            request_document,
            evidence_document,
            cached[4],
            cached[5],
        )

    def _find_row_for_request(
        self, request: object
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        self._assert_runtime_request(request)
        matches = []
        for (
            request_seq,
            request_document,
            request_content,
            evidence_document,
            evidence_content,
        ) in self._history_with_evidence():
            allocation = self._allocation_for_role(
                request_content, request.evidence_role
            )
            window_matches = [
                window
                for window in evidence_content["windows"]
                if window["allocation"]["role"]
                == request.evidence_role
            ]
            if len(window_matches) != 1:
                raise EvaluationInboxError(
                    "evidence has no exact role window"
                )
            row = window_matches[0]["attempts"][
                request.proposal_offset
            ] if request.proposal_offset < len(
                window_matches[0]["attempts"]
            ) else None
            if row is None or not self._request_matches(
                request, request_content, allocation, row
            ):
                continue
            matches.append(
                (
                    row,
                    {
                        "request_seq": request_seq,
                        "request_sha256": request_document[
                            "content_sha256"
                        ],
                        "evidence_sha256": evidence_document[
                            "content_sha256"
                        ],
                        "attempt_sha256": canonical_sha256(row),
                        "role": request.evidence_role,
                        "offset": request.proposal_offset,
                    },
                )
            )
        if not matches:
            raise EvaluationInboxError(
                "no accepted sidecar evidence row matches the V4 proposal"
            )
        if len(matches) != 1:
            raise EvaluationInboxError(
                "multiple sidecar evidence rows match one V4 proposal"
            )
        return matches[0]

    def _load_record_row(
        self, record: Mapping[str, object]
    ) -> Dict[str, object]:
        request_seq = record["request_seq"]
        (
            request_document,
            evidence_document,
            _request,
            evidence,
        ) = self._load_request_evidence_contents(request_seq)
        if (
            request_document["content_sha256"]
            != record["request_sha256"]
        ):
            raise EvaluationInboxError(
                "consumed request bytes changed after reservation"
            )
        if (
            evidence_document["content_sha256"]
            != record["evidence_sha256"]
        ):
            raise EvaluationInboxError(
                "consumed evidence bytes changed after reservation"
            )
        windows = [
            window
            for window in evidence["windows"]
            if window["allocation"]["role"] == record["role"]
        ]
        if len(windows) != 1:
            raise EvaluationInboxError(
                "consumed evidence role is no longer exact"
            )
        offset = record["offset"]
        attempts = windows[0]["attempts"]
        if offset >= len(attempts):
            raise EvaluationInboxError(
                "consumed evidence offset is outside its window"
            )
        row = attempts[offset]
        if canonical_sha256(row) != record["attempt_sha256"]:
            raise EvaluationInboxError(
                "consumed attempt row changed after reservation"
            )
        return row

    def _record(
        self, request: object
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        self._assert_runtime_request(request)
        try:
            record = self._records[request.reservation_sha256]
        except KeyError as exc:
            raise EvaluationInboxError(
                "proposal reservation has not been issued"
            ) from exc
        if (
            record["role"] != request.evidence_role
            or record["offset"] != request.proposal_offset
        ):
            raise EvaluationInboxError(
                "proposal reservation was replayed against another row"
            )
        row = self._load_record_row(record)
        (
            request_document,
            _evidence_document,
            request_content,
            _evidence_content,
        ) = self._load_request_evidence_contents(
            record["request_seq"]
        )
        allocation = self._allocation_for_role(
            request_content, record["role"]
        )
        if not self._request_matches(
            request, request_content, allocation, row
        ):
            raise EvaluationInboxError(
                "proposal reservation no longer matches its immutable row"
            )
        return row, record

    def _proposal_object(
        self, request: object, row: Mapping[str, object]
    ) -> object:
        return self._runtime.FrozenIssuedProposal.create(
            reservation_sha256=request.reservation_sha256,
            source_contract_sha256=self.source_contract_sha256,
            sample_receipt_sha256=row["sample_receipt_sha256"],
            birth_receipt_sha256=row["birth_receipt_sha256"],
            action_uid=request.action_uid,
            profile_sha256=request.profile_sha256,
            mobility_mode=request.mobility_mode,
            domain_epoch=request.domain_epoch,
            levels_sha256=request.domain_levels.canonical_sha256,
            sample_index=request.sample_index,
            birth_index=request.birth_index,
            sampling_stratum=row["sampling_stratum"],
            frontier_arm=row["frontier_arm"],
        )

    def issue_proposal(self, request: object) -> object:
        row, reference = self._find_row_for_request(request)
        reservation = request.reservation_sha256
        if reservation in self._records or any(
            record["request_seq"] == reference["request_seq"]
            and record["role"] == reference["role"]
            and record["offset"] == reference["offset"]
            for record in self._records.values()
        ):
            raise EvaluationInboxError(
                "sidecar evidence proposal was replayed"
            )
        proposal = self._proposal_object(request, row)
        self._records[reservation] = {
            **reference,
            "stage": "issued",
        }
        self._revision += 1
        return proposal

    def assert_exact_proposal(
        self, request: object, proposal: object
    ) -> None:
        row, record = self._record(request)
        if record["stage"] not in (
            "issued",
            "solver_rejected",
            "solver_admitted",
            "installed",
            "started",
            "terminal",
        ):
            raise EvaluationInboxError(
                "proposal source stage is invalid"
            )
        expected = self._proposal_object(request, row)
        if proposal != expected:
            raise EvaluationInboxError(
                "proposal differs from accepted sidecar evidence"
            )

    def _solver_object(
        self,
        proposal: object,
        row: Mapping[str, object],
    ) -> object:
        disposition = row["solver_disposition"]
        if disposition == "admitted":
            solver_disposition = "admitted"
            reason = ""
            task_receipt = row["task_receipt_sha256"]
        else:
            solver_disposition = "rejected"
            prefix = (
                "physics_invalid/"
                if disposition == "physics_invalid"
                else ""
            )
            reason = prefix + row["reject_reason"]
            task_receipt = ""
        return self._runtime.FrozenSolverEvent.create(
            proposal_receipt_sha256=proposal.source_receipt_sha256,
            source_contract_sha256=self.source_contract_sha256,
            disposition=solver_disposition,
            reject_reason=reason,
            task_receipt_sha256=task_receipt,
        )

    def solver_event(
        self, request: object, proposal: object
    ) -> object:
        row, record = self._record(request)
        if record["stage"] != "issued":
            raise EvaluationInboxError(
                "solver event is out of order or replayed"
            )
        self.assert_exact_proposal(request, proposal)
        event = self._solver_object(proposal, row)
        record["stage"] = (
            "solver_admitted"
            if event.disposition == "admitted"
            else "solver_rejected"
        )
        self._revision += 1
        return event

    def assert_solver_event(
        self,
        request: object,
        proposal: object,
        event: object,
    ) -> None:
        row, record = self._record(request)
        expected = self._solver_object(proposal, row)
        if event != expected or record["stage"] not in (
            "solver_rejected",
            "solver_admitted",
            "installed",
            "started",
            "terminal",
        ):
            raise EvaluationInboxError(
                "solver event differs from accepted sidecar evidence"
            )

    def _lifecycle_object(
        self, proposal: object, solver: object, stage: str
    ) -> object:
        return self._runtime.FrozenLifecycleEvent.create(
            proposal_receipt_sha256=proposal.source_receipt_sha256,
            task_receipt_sha256=solver.task_receipt_sha256,
            source_contract_sha256=self.source_contract_sha256,
            stage=stage,
        )

    def lifecycle_event(
        self,
        request: object,
        proposal: object,
        solver: object,
        stage: str,
    ) -> object:
        row, record = self._record(request)
        self.assert_exact_proposal(request, proposal)
        self.assert_solver_event(request, proposal, solver)
        if stage == "installed":
            if record["stage"] != "solver_admitted" or not row["installed"]:
                raise EvaluationInboxError(
                    "sidecar evidence has no install event"
                )
        elif stage == "started":
            if record["stage"] != "installed" or not row["started"]:
                raise EvaluationInboxError(
                    "sidecar evidence has no start event"
                )
        else:
            raise EvaluationInboxError(
                "lifecycle stage must be installed or started"
            )
        event = self._lifecycle_object(proposal, solver, stage)
        record["stage"] = stage
        self._revision += 1
        return event

    def assert_lifecycle_event(
        self,
        request: object,
        proposal: object,
        solver: object,
        event: object,
    ) -> None:
        _row, record = self._record(request)
        expected = self._lifecycle_object(
            proposal, solver, event.stage
        )
        allowed = (
            ("installed", "started", "terminal")
            if event.stage == "installed"
            else ("started", "terminal")
        )
        if event != expected or record["stage"] not in allowed:
            raise EvaluationInboxError(
                "lifecycle event differs from accepted sidecar evidence"
            )

    def _terminal_object(
        self,
        proposal: object,
        solver: object,
        row: Mapping[str, object],
    ) -> object:
        signals = self._runtime.FrozenTerminalSignals(
            **dict(row["terminal_signals"])
        )
        return self._runtime.FrozenTerminalEvent.create(
            proposal_receipt_sha256=proposal.source_receipt_sha256,
            task_receipt_sha256=solver.task_receipt_sha256,
            source_contract_sha256=self.source_contract_sha256,
            signals=signals,
        )

    def next_event_stage(self, request: object) -> str:
        """Return the next source-owned lifecycle step without mutation."""

        row, record = self._record(request)
        stage = record["stage"]
        if stage == "issued":
            return "solver"
        if stage in ("solver_rejected", "terminal"):
            return "settled"
        if stage == "solver_admitted":
            if row["installed"]:
                return "installed"
            if (
                not row["closed"]
                and row["terminal_signals"]["infrastructure_invalid"]
            ):
                return "terminal"
        elif stage == "installed":
            if row["started"]:
                return "started"
            if (
                not row["closed"]
                and row["terminal_signals"]["infrastructure_invalid"]
            ):
                return "terminal"
        elif stage == "started":
            return "terminal"
        raise EvaluationInboxError(
            "accepted sidecar lifecycle transcript cannot advance exactly"
        )

    def terminal_event(
        self,
        request: object,
        proposal: object,
        solver: object,
    ) -> object:
        row, record = self._record(request)
        self.assert_exact_proposal(request, proposal)
        self.assert_solver_event(request, proposal, solver)
        if record["stage"] == "started":
            pass
        elif record["stage"] in ("solver_admitted", "installed"):
            if (
                row["closed"]
                or not row["terminal_signals"][
                    "infrastructure_invalid"
                ]
                or (
                    record["stage"] == "solver_admitted"
                    and row["installed"]
                )
                or (
                    record["stage"] == "installed"
                    and (
                        not row["installed"] or row["started"]
                    )
                )
            ):
                raise EvaluationInboxError(
                    "pre-start terminal event is not an exact "
                    "infrastructure burn"
                )
        else:
            raise EvaluationInboxError("terminal event is out of order")
        event = self._terminal_object(proposal, solver, row)
        if (
            bool(event.signals.infrastructure_invalid)
            != (not row["closed"])
        ):
            raise EvaluationInboxError(
                "terminal infrastructure/closed accounting drifted"
            )
        record["stage"] = "terminal"
        self._revision += 1
        return event

    def assert_terminal_event(
        self,
        request: object,
        proposal: object,
        solver: object,
        event: object,
    ) -> None:
        row, record = self._record(request)
        expected = self._terminal_object(proposal, solver, row)
        if event != expected or record["stage"] != "terminal":
            raise EvaluationInboxError(
                "terminal event differs from accepted raw sidecar signals"
            )

    def assert_request_consumed(self, request_seq: int) -> str:
        """Prove every row in one request reached its exact final stage."""

        seq = _plain_int(request_seq, label="request_seq")
        (
            request_document,
            evidence_document,
            request,
            evidence,
        ) = (
            self._load_request_evidence_contents(seq)
        )
        records = [
            record
            for record in self._records.values()
            if record["request_seq"] == seq
        ]
        expected_count = sum(
            allocation["proposal_count"]
            for allocation in request["windows"]
        )
        if len(records) != expected_count:
            raise EvaluationInboxError(
                "request is not fully consumed by the V4 evaluator"
            )
        observed = {
            (record["role"], record["offset"]) for record in records
        }
        expected = {
            (allocation["role"], offset)
            for allocation in request["windows"]
            for offset in range(allocation["proposal_count"])
        }
        if observed != expected:
            raise EvaluationInboxError(
                "request consumption tape has a missing or duplicate row"
            )
        rows_by_key = {
            (
                window["allocation"]["role"],
                row["proposal_offset"],
            ): row
            for window in evidence["windows"]
            for row in window["attempts"]
        }
        for record in records:
            row = rows_by_key[(record["role"], record["offset"])]
            expected_stage = (
                "solver_rejected"
                if row["solver_disposition"]
                in ("physics_invalid", "rejected")
                else "terminal"
            )
            if (
                record["stage"] != expected_stage
                or record["request_sha256"]
                != request_document["content_sha256"]
                or record["evidence_sha256"]
                != evidence_document["content_sha256"]
                or record["attempt_sha256"] != canonical_sha256(row)
            ):
                raise EvaluationInboxError(
                    "request consumption tape is incomplete or drifted"
                )
        return evidence_document["content_sha256"]


__all__ = [
    "ACK_KIND",
    "CANARY_PROPOSALS",
    "CANARY_SAFE_CLOSED_MIN",
    "EVALUATION_INBOX_CONTRACT_SHA256",
    "EVIDENCE_KIND",
    "EvaluationInbox",
    "EvaluationInboxError",
    "FROZEN_EVALUATION_INBOX_COORDINATOR_CONTRACT_SHA256",
    "FROZEN_EVALUATION_INBOX_ATTEMPT_SOURCE_CONTRACT_SHA256",
    "FROZEN_EVALUATION_INBOX_ATTEMPT_SOURCE_PATH",
    "FORMAL_ISAAC_BACKEND_CONTRACT",
    "FORMAL_ISAAC_BACKEND_CONTRACT_SHA256",
    "FrozenEvaluationInboxCoordinator",
    "FrozenSidecarInboxAttemptSource",
    "HELDOUT_PROPOSALS",
    "HELDOUT_SAFE_CLOSED_MIN",
    "LAUNCH_KIND",
    "POLICY_EVALUATION_CONTRACT",
    "POLICY_EVALUATION_CONTRACT_SHA256",
    "RESOLVED_EVALUATION_RECIPE_CONTRACT",
    "RESOLVED_EVALUATION_RECIPE_CONTRACT_SHA256",
    "REQUEST_KIND",
    "RUNTIME_IDENTITY_CONTRACT",
    "RUNTIME_IDENTITY_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "SCHEDULER_PROPOSALS",
    "SIDECAR_HEARTBEAT_CONTRACT",
    "TERMINAL_OUTCOMES",
    "TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_CODE_SHA256",
    "TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_LAUNCH_SHA256",
    "WINDOW_CONTRACT",
    "artifact_receipt",
    "build_ack_document",
    "build_evidence_document",
    "build_request_document",
    "build_sidecar_launch_document",
    "canonical_sha256",
    "classify_terminal_signals",
    "make_scheduler_allocation",
    "make_window_allocations",
    "state_binding",
    "strict_json_loads",
    "strict_read_json",
    "validate_ack_document",
    "validate_evidence_document",
    "validate_request_document",
    "validate_sidecar_launch_document",
    "verify_artifact_receipt",
    "verify_request_artifacts",
]
