#!/usr/bin/env python3
"""Verify one formal ActionBall checkpoint by a real no-step Isaac roundtrip.

The verifier is deliberately not a checkpoint-structure linter.  It launches
the same claim-bound Isaac runtime constructor as the frozen evaluator,
strictly restores policy, optimizer, normalizers, RNG and the complete
ActionBall environment state, performs no reset/step/update, saves through the
runner's production no-step roundtrip API, then constructs a second runtime and
strictly restores the roundtrip bytes.  Only an exact core-state match can
produce the no-clobber receipt consumed by the signed stage attestor.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import types
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
RECEIPT_KIND = "action_ball_exact_resume_verification_v1"
TASK_ID = "HOPE-PingPong-ActionBall-AgibotA3-v0"
EXPERIMENT_NAME = "agibot_a3_hope_action_ball_fresh_n5"
VERIFIER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_exact_resume_verifier.py"
)
LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_action_ball_curriculum.py"
)
SIDECAR_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_frozen_eval_sidecar.py"
)
RUNTIME_INVENTORY_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_runtime_inventory.py"
)
NOSITE_BOOTSTRAP_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_python_nosite_bootstrap.py"
)
ACTION_SET_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_action_set_contract.py"
)
EVALUATION_INBOX_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_evaluation_inbox.py"
)
CHECKPOINT_RE = re.compile(r"^model_([0-9]+)\.pt$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_JSON_BYTES = 512 * 1024 * 1024
_SANITIZED_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "TZ",
        "USER",
    }
)
_INVENTORY_STDIN_WRAPPER = (
    "import sys\n"
    "_raw=sys.stdin.buffer.read()\n"
    "_path=sys.argv.pop(1)\n"
    "_globals={'__name__':'__main__','__file__':_path,'__package__':None}\n"
    "exec(compile(_raw,_path,'exec',dont_inherit=True,optimize=0),_globals)\n"
)


class ExactResumeVerificationError(RuntimeError):
    """The final checkpoint cannot prove a true no-step exact roundtrip."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ExactResumeVerificationError(
            "value is not finite canonical JSON"
        ) from exc


def _canonical_utf8_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ExactResumeVerificationError(
            "value is not finite canonical UTF-8 JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExactResumeVerificationError(
                "duplicate JSON key is forbidden: {!r}".format(key)
            )
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ExactResumeVerificationError(
        "non-finite JSON constant is forbidden: {!r}".format(token)
    )


def _strict_json_bytes(raw: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except ExactResumeVerificationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExactResumeVerificationError(
            "{} is not strict UTF-8 JSON".format(label)
        ) from exc
    if type(value) is not dict:
        raise ExactResumeVerificationError(
            "{} must contain one JSON object".format(label)
        )
    _canonical_bytes(value)
    return value


def _exact_dict(value: Any, keys: Iterable[str], label: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise ExactResumeVerificationError(
            "{} must be a plain object".format(label)
        )
    expected = set(keys)
    if set(value) != expected:
        raise ExactResumeVerificationError(
            "{} has invalid keys (missing={}, unknown={})".format(
                label,
                sorted(expected - set(value)),
                sorted(set(value) - expected),
            )
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ExactResumeVerificationError(
            "{} must be one lowercase SHA-256".format(label)
        )
    return value


def _plain_int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ExactResumeVerificationError(
            "{} must be a plain integer >= {}".format(label, minimum)
        )
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
        raise ExactResumeVerificationError(
            "checkpoint NumPy RNG state is not safe schema 1"
        )


def _absolute_path(value: Any, label: str, must_exist: bool = True) -> Path:
    if type(value) is not str or not value:
        raise ExactResumeVerificationError(
            "{} must be a non-empty absolute path".format(label)
        )
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or os.path.normpath(value) != value
    ):
        raise ExactResumeVerificationError(
            "{} must be an absolute normalized path".format(label)
        )
    if must_exist and not os.path.lexists(path):
        raise ExactResumeVerificationError(
            "{} does not exist: {}".format(label, path)
        )
    return path


def _real_directory_chain(path: Path, label: str) -> None:
    chain: List[Path] = []
    current = path
    while current != current.parent:
        chain.append(current)
        current = current.parent
    chain.append(current)
    for component in reversed(chain):
        try:
            info = component.lstat()
        except OSError as exc:
            raise ExactResumeVerificationError(
                "{} directory component is missing: {}".format(
                    label, component
                )
            ) from exc
        if not stat.S_ISDIR(info.st_mode):
            raise ExactResumeVerificationError(
                "{} contains a symlink/non-directory: {}".format(
                    label, component
                )
            )


def _snapshot_file(
    path_value: Any,
    label: str,
    *,
    max_bytes: int = 32 << 30,
) -> Dict[str, Any]:
    path = (
        path_value
        if isinstance(path_value, Path)
        else _absolute_path(path_value, label)
    )
    _real_directory_chain(path.parent, label + " parent")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ExactResumeVerificationError(
            "{} cannot be inspected".format(label)
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > max_bytes
    ):
        raise ExactResumeVerificationError(
            "{} must be a nonempty single-link regular file".format(label)
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(str(path), flags)
    try:
        opened = os.fstat(descriptor)
        signature = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
        )
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
        ) != signature:
            raise ExactResumeVerificationError(
                "{} changed before safe open".format(label)
            )
        digest = hashlib.sha256()
        chunks: List[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise ExactResumeVerificationError(
                    "{} was truncated while reading".format(label)
                )
            digest.update(chunk)
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ExactResumeVerificationError(
                "{} grew while reading".format(label)
            )
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
        ) != signature:
            raise ExactResumeVerificationError(
                "{} changed while reading".format(label)
            )
    finally:
        os.close(descriptor)
    return {
        "path": path,
        "raw": b"".join(chunks),
        "sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
        "stat": before,
    }


def _snapshot_json(path: Path, label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    snapshot = _snapshot_file(path, label, max_bytes=MAX_JSON_BYTES)
    return _strict_json_bytes(snapshot["raw"], label), snapshot


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        str(path),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive_json(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute():
        raise ExactResumeVerificationError(
            "receipt output path must be absolute"
        )
    _real_directory_chain(path.parent, "receipt output parent")
    if os.path.lexists(path):
        raise ExactResumeVerificationError(
            "receipt namespace is already spent: {}".format(path)
        )
    payload = _canonical_bytes(dict(value)) + b"\n"
    temporary = path.parent / ".{}.{}.tmp".format(
        path.name, secrets.token_hex(12)
    )
    descriptor = os.open(
        str(temporary),
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ExactResumeVerificationError(
                    "receipt write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        actual = b""
        while len(actual) < len(payload):
            chunk = os.read(descriptor, len(payload) - len(actual))
            if not chunk:
                break
            actual += chunk
        if actual != payload or os.read(descriptor, 1):
            raise ExactResumeVerificationError(
                "receipt descriptor readback differs"
            )
    finally:
        os.close(descriptor)
    installed = False
    try:
        os.link(str(temporary), str(path), follow_symlinks=False)
        installed = True
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise ExactResumeVerificationError(
            "receipt namespace is already spent: {}".format(path)
        ) from exc
    finally:
        if installed:
            temporary.unlink()
            _fsync_directory(path.parent)
    reopened = _snapshot_file(path, "published exact-resume receipt")
    if reopened["raw"] != payload:
        raise ExactResumeVerificationError(
            "published exact-resume receipt bytes drifted"
        )
    return reopened["sha256"]


def _git(checkout: Path, *args: str, binary: bool = False) -> Any:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=not binary,
        env={
            "PATH": os.defpath,
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    if result.returncode != 0:
        stderr = (
            result.stderr
            if not binary
            else result.stderr.decode("utf-8", errors="replace")
        )
        raise ExactResumeVerificationError(
            "git {} failed: {}".format(
                " ".join(args), stderr.strip()[-4000:]
            )
        )
    return result.stdout if binary else result.stdout.strip()


def _committed_source(
    checkout: Path, commit: str, relative: str, label: str
) -> Dict[str, Any]:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ExactResumeVerificationError(
            "{} path is not repository-relative".format(label)
        )
    raw = _git(checkout, "cat-file", "-p", "{}:{}".format(commit, relative), binary=True)
    mode_row = _git(
        checkout, "ls-tree", commit, "--", relative
    ).split()
    if len(mode_row) < 3 or mode_row[1] != "blob":
        raise ExactResumeVerificationError(
            "{} is not one committed regular blob".format(label)
        )
    live = _snapshot_file(checkout / relative, label)
    digest = hashlib.sha256(raw).hexdigest()
    if live["sha256"] != digest:
        raise ExactResumeVerificationError(
            "{} live bytes differ from exact commit".format(label)
        )
    return {
        "path": live["path"],
        "sha256": digest,
        "size_bytes": len(raw),
        "raw": raw,
    }


def _load_action_set_contract_module(path: Path) -> Any:
    name = "_action_ball_exact_resume_action_set_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExactResumeVerificationError(
            "cannot load exact action-set contract module"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_nosite_bootstrap_module(path: Path) -> Any:
    name = "_action_ball_exact_resume_nosite_bootstrap"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExactResumeVerificationError(
            "cannot load exact no-site bootstrap module"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ExactResumeVerificationError(
            "cannot execute exact no-site bootstrap module"
        ) from exc
    return module


def _decode_nosite_argv_contract(
    argv: Any, label: str
) -> Tuple[Dict[str, Any], str]:
    """Decode an untrusted no-site argv before its committed validator loads."""

    if (
        type(argv) is not list
        or len(argv) != 10
        or any(type(item) is not str for item in argv)
        or argv[1:5] != ["-I", "-B", "-S", "-c"]
    ):
        raise ExactResumeVerificationError(
            "{} is not one exact 10-token no-site argv".format(label)
        )
    contract_sha = _sha256(argv[8], "{} contract SHA".format(label))
    try:
        raw = base64.b64decode(argv[9].encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ExactResumeVerificationError(
            "{} contract is not canonical base64".format(label)
        ) from exc
    if (
        base64.b64encode(raw).decode("ascii") != argv[9]
        or hashlib.sha256(raw).hexdigest() != contract_sha
    ):
        raise ExactResumeVerificationError(
            "{} contract base64/SHA differs".format(label)
        )
    contract = _exact_dict(
        _strict_json_bytes(raw, "{} contract".format(label)),
        (
            "schema_version",
            "kind",
            "bootstrap",
            "entrypoint",
            "import_roots",
            "entrypoint_argv",
        ),
        "{} contract".format(label),
    )
    if (
        contract["schema_version"] != 1
        or contract["kind"] != "action_ball_python_nosite_argv_contract_v1"
        or _canonical_utf8_bytes(contract) != raw
        or type(contract["entrypoint_argv"]) is not list
        or any(
            type(item) is not str or not item
            for item in contract["entrypoint_argv"]
        )
    ):
        raise ExactResumeVerificationError(
            "{} contract schema/arguments are invalid".format(label)
        )
    return contract, contract_sha


def _hydra_overrides(
    entrypoint_argv: Sequence[str],
) -> Dict[str, str]:
    if "--" not in entrypoint_argv:
        raise ExactResumeVerificationError(
            "launch claim embedded argv lacks one Hydra boundary"
        )
    boundary = entrypoint_argv.index("--")
    if "--" in entrypoint_argv[boundary + 1 :]:
        raise ExactResumeVerificationError(
            "launch claim embedded argv has multiple Hydra boundaries"
        )
    pattern = re.compile(
        r"^(?:\+\+|\+)?([A-Za-z_][A-Za-z0-9_.-]*)=(.*)$"
    )
    overrides: Dict[str, str] = {}
    for token in entrypoint_argv[boundary + 1 :]:
        if type(token) is not str or token.startswith("~"):
            raise ExactResumeVerificationError(
                "Hydra deletion/non-string override is forbidden"
            )
        match = pattern.fullmatch(token)
        if match is None:
            raise ExactResumeVerificationError(
                "malformed Hydra override: {!r}".format(token)
            )
        key, value = match.group(1), match.group(2)
        if key in overrides:
            raise ExactResumeVerificationError(
                "duplicate Hydra override: {}".format(key)
            )
        overrides[key] = value
    return overrides


def _validate_claim_argv(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    claim_sha256: str,
    claim_path: Path,
) -> None:
    argv_without_claim = payload.get("argv_without_launch_claim")
    expected_path = "++training_launch_claim_path={}".format(claim_path)
    base_contract, _base_sha = _decode_nosite_argv_contract(
        argv_without_claim, "base trainer argv"
    )
    final_contract, _final_sha = _decode_nosite_argv_contract(
        row.get("argv"), "claim-bound trainer argv"
    )
    base_args = base_contract["entrypoint_argv"]
    expected_final_arg = "++training_launch_claim_sha256={}".format(
        claim_sha256
    )
    final_args = final_contract["entrypoint_argv"]
    base_without_args = dict(base_contract)
    final_without_args = dict(final_contract)
    del base_without_args["entrypoint_argv"]
    del final_without_args["entrypoint_argv"]
    if (
        final_without_args != base_without_args
        or final_args != [*base_args, expected_final_arg]
        or base_args.count(expected_path) != 1
        or sum(
            type(item) is str
            and item.startswith("++training_launch_claim_path=")
            for item in base_args
        )
        != 1
        or any(
            type(item) is str
            and item.startswith("++training_launch_claim_sha256=")
            for item in base_args
        )
        or sum(
            type(item) is str
            and item.startswith("++training_launch_claim_sha256=")
            for item in final_args
        )
        != 1
    ):
        raise ExactResumeVerificationError(
            "launch claim argv does not have one exact path and final SHA"
        )
    overrides = _hydra_overrides(final_args)
    contract = payload.get("action_set_contract")
    manifest = payload.get("manifest")
    if type(contract) is not dict or type(manifest) is not dict:
        raise ExactResumeVerificationError(
            "claim lacks action-set/manifest identity"
        )
    expected = {
        "task.experiment_name": contract.get("experiment_name"),
        "task.actor_obs_contract": contract.get("actor_obs_contract"),
        "task.racket.action_ball_manifest_path": contract.get(
            "manifest_path"
        ),
        "task.racket.action_ball_manifest_sha256": contract.get(
            "manifest_sha256"
        ),
        "training_launch_claim_path": str(claim_path),
    }
    for key, value in expected.items():
        if type(value) is not str or overrides.get(key) != value:
            raise ExactResumeVerificationError(
                "claim Hydra override {} differs from action-set identity".format(
                    key
                )
            )
    try:
        clip_names = json.loads(overrides["task.racket.clip_names"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExactResumeVerificationError(
            "claim clip_names is not strict JSON"
        ) from exc
    if (
        clip_names != contract.get("ordered_action_ids")
        or payload.get("ordered_action_ids") != clip_names
        or manifest.get("path") != contract.get("manifest_path")
        or manifest.get("sha256") != contract.get("manifest_sha256")
    ):
        raise ExactResumeVerificationError(
            "claim order/manifest differs from action-set contract"
        )


def _validate_claim(
    claim_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Path, str]:
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
    claim_sha = _sha256(row["launch_claim_sha256"], "launch claim SHA")
    payload = row["canonical_payload"]
    if (
        row["schema_version"] != 3
        or row["kind"] != "action_ball_no_clobber_launch_claim_v3"
        or type(payload) is not dict
        or canonical_sha256(payload) != claim_sha
        or row["confirmation_claim_sha256"] != claim_sha
        or type(row["argv"]) is not list
        or not row["argv"]
    ):
        raise ExactResumeVerificationError(
            "launch claim canonical binding is invalid"
        )
    namespace = _absolute_path(payload.get("namespace"), "stage namespace")
    if claim_path != namespace / "launch_claim.json":
        raise ExactResumeVerificationError(
            "claim is not its namespace launch_claim.json"
        )
    _validate_claim_argv(
        row,
        payload,
        claim_sha256=claim_sha,
        claim_path=claim_path,
    )
    checkout = _absolute_path(payload.get("source_checkout"), "source checkout")
    if checkout.resolve(strict=True) != checkout:
        raise ExactResumeVerificationError(
            "source checkout is not one real absolute path"
        )
    commit = payload.get("source_commit_sha")
    if type(commit) is not str or COMMIT_RE.fullmatch(commit) is None:
        raise ExactResumeVerificationError(
            "source commit is not one full SHA-1 object id"
        )
    if (
        _git(checkout, "rev-parse", "--verify", "HEAD") != commit
        or _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise ExactResumeVerificationError(
            "source checkout is not exact clean claim commit"
        )
    runtime_shas = payload.get("runtime_code_sha256")
    if type(runtime_shas) is not dict:
        raise ExactResumeVerificationError(
            "claim lacks runtime code pins"
        )
    verifier_source = _committed_source(
        checkout, commit, VERIFIER_SOURCE, "exact-resume verifier source"
    )
    launcher_source = _committed_source(
        checkout, commit, LAUNCHER_SOURCE, "training launcher source"
    )
    sidecar_source = _committed_source(
        checkout, commit, SIDECAR_SOURCE, "exact-resume runtime factory source"
    )
    inventory_source = _committed_source(
        checkout,
        commit,
        RUNTIME_INVENTORY_SOURCE,
        "runtime inventory verifier source",
    )
    action_set_contract_source = _committed_source(
        checkout,
        commit,
        ACTION_SET_CONTRACT_SOURCE,
        "action-set contract source",
    )
    nosite_bootstrap_source = _committed_source(
        checkout,
        commit,
        NOSITE_BOOTSTRAP_SOURCE,
        "no-site bootstrap source",
    )
    inbox_source = _committed_source(
        checkout,
        commit,
        EVALUATION_INBOX_SOURCE,
        "evaluation inbox protocol source",
    )
    if (
        runtime_shas.get(VERIFIER_SOURCE) != verifier_source["sha256"]
        or runtime_shas.get(LAUNCHER_SOURCE) != launcher_source["sha256"]
        or runtime_shas.get(SIDECAR_SOURCE) != sidecar_source["sha256"]
        or runtime_shas.get(RUNTIME_INVENTORY_SOURCE)
        != inventory_source["sha256"]
        or runtime_shas.get(ACTION_SET_CONTRACT_SOURCE)
        != action_set_contract_source["sha256"]
        or runtime_shas.get(NOSITE_BOOTSTRAP_SOURCE)
        != nosite_bootstrap_source["sha256"]
        or runtime_shas.get(EVALUATION_INBOX_SOURCE)
        != inbox_source["sha256"]
    ):
        raise ExactResumeVerificationError(
            "claim runtime source pins differ from exact commit"
        )
    runtime = payload.get("isaac_python_runtime")
    inventory_identity = (
        runtime.get("runtime_inventory") if type(runtime) is dict else None
    )
    isolated_identity = payload.get("isolated_training_entrypoint")
    sidecar_nosite_identity = payload.get("sidecar_nosite_execution")
    sidecar_argv = payload.get("sidecar_argv")
    if (
        type(runtime) is not dict
        or type(runtime.get("path")) is not str
        or type(inventory_identity) is not dict
        or type(inventory_identity.get("import_roots")) is not list
        or type(isolated_identity) is not dict
        or type(sidecar_nosite_identity) is not dict
        or type(sidecar_argv) is not list
    ):
        raise ExactResumeVerificationError(
            "claim lacks exact no-site runtime identities"
        )
    nosite = _load_nosite_bootstrap_module(
        nosite_bootstrap_source["path"]
    )
    bootstrap_binding = {
        "path": str(nosite_bootstrap_source["path"]),
        "byte_count": nosite_bootstrap_source["size_bytes"],
        "sha256": nosite_bootstrap_source["sha256"],
    }
    launcher_binding = {
        "path": str(launcher_source["path"]),
        "byte_count": launcher_source["size_bytes"],
        "sha256": launcher_source["sha256"],
    }
    sidecar_binding = {
        "path": str(sidecar_source["path"]),
        "byte_count": sidecar_source["size_bytes"],
        "sha256": sidecar_source["sha256"],
    }
    try:
        base_command = nosite.validate_exact_nosite_argv(
            payload["argv_without_launch_claim"],
            expected_python=Path(runtime["path"]),
            expected_bootstrap=bootstrap_binding,
            expected_entrypoint=launcher_binding,
            expected_import_roots=inventory_identity["import_roots"],
            verify_live=True,
        )
        final_command = nosite.validate_exact_nosite_argv(
            row["argv"],
            expected_python=Path(runtime["path"]),
            expected_bootstrap=bootstrap_binding,
            expected_entrypoint=launcher_binding,
            expected_import_roots=inventory_identity["import_roots"],
            expected_entrypoint_argv=[
                *base_command.contract["entrypoint_argv"],
                "++training_launch_claim_sha256={}".format(claim_sha),
            ],
            verify_live=True,
        )
        sidecar_command = nosite.validate_exact_nosite_argv(
            sidecar_argv,
            expected_python=Path(runtime["path"]),
            expected_bootstrap=bootstrap_binding,
            expected_entrypoint=sidecar_binding,
            expected_import_roots=inventory_identity["import_roots"],
            verify_live=True,
        )
    except Exception as exc:
        raise ExactResumeVerificationError(
            "claim no-site argv validation failed: {}".format(exc)
        ) from exc
    if (
        isolated_identity.get("nosite_argv_contract")
        != base_command.contract
        or isolated_identity.get("nosite_argv_contract_sha256")
        != base_command.contract_sha256
        or sidecar_nosite_identity
        != {
            "nosite_argv_contract_sha256": (
                sidecar_command.contract_sha256
            ),
            "nosite_argv_contract": dict(sidecar_command.contract),
        }
        or final_command.contract["entrypoint_argv"][-1]
        != "++training_launch_claim_sha256={}".format(claim_sha)
    ):
        raise ExactResumeVerificationError(
            "claim no-site identities differ from exact argv contracts"
        )
    contract_module = _load_action_set_contract_module(
        action_set_contract_source["path"]
    )
    try:
        exact_contract = contract_module.load_contract_from_source(
            action_set_contract_source["raw"],
            payload.get("launch_profile"),
        )
    except Exception as exc:
        raise ExactResumeVerificationError(
            "committed action-set contract is invalid: {}".format(exc)
        ) from exc
    if payload.get("action_set_contract") != exact_contract:
        raise ExactResumeVerificationError(
            "claim action-set contract differs from committed registry"
        )
    return row, payload, snapshot, checkout, commit


def _preimport_runtime_inventory_verification(
    *,
    payload: Mapping[str, Any],
    checkout: Path,
    inventory_source: Mapping[str, Any],
    nosite_bootstrap_source: Mapping[str, Any],
) -> Dict[str, Any]:
    """Establish runtime provenance before importing torch, Isaac, or project code."""

    forbidden = (
        "torch",
        "isaaclab",
        "carb",
        "omni",
        "action_ball_evaluation_inbox",
    )
    already_imported = [
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in forbidden)
    ]
    if already_imported:
        raise ExactResumeVerificationError(
            "runtime inventory must run before third-party/project imports: "
            + ", ".join(sorted(already_imported)[:8])
        )
    runtime_code = payload.get("runtime_code_sha256")
    runtime = payload.get("isaac_python_runtime")
    inventory_identity = (
        runtime.get("runtime_inventory") if type(runtime) is dict else None
    )
    if (
        type(runtime_code) is not dict
        or type(runtime) is not dict
        or type(inventory_identity) is not dict
        or set(inventory_identity)
        != {
            "path",
            "file_sha256",
            "content_sha256",
            "kind",
            "import_roots",
            "nosite_verification_contract_sha256",
        }
    ):
        raise ExactResumeVerificationError(
            "claim lacks exact pre-import runtime inventory identity"
        )
    source_path = checkout / RUNTIME_INVENTORY_SOURCE
    source_raw = inventory_source.get("raw")
    source_sha = inventory_source.get("sha256")
    source_size = inventory_source.get("size_bytes")
    if (
        inventory_source.get("path") != source_path
        or type(source_raw) is not bytes
        or type(source_size) is not int
        or source_size <= 0
        or len(source_raw) != source_size
        or hashlib.sha256(source_raw).hexdigest() != source_sha
        or source_sha != runtime_code.get(RUNTIME_INVENTORY_SOURCE)
    ):
        raise ExactResumeVerificationError(
            "runtime inventory verifier is not the claim-pinned commit bytes"
        )
    bootstrap_path = checkout / NOSITE_BOOTSTRAP_SOURCE
    bootstrap_raw = nosite_bootstrap_source.get("raw")
    bootstrap_sha = nosite_bootstrap_source.get("sha256")
    bootstrap_size = nosite_bootstrap_source.get("size_bytes")
    if (
        nosite_bootstrap_source.get("path") != bootstrap_path
        or type(bootstrap_raw) is not bytes
        or type(bootstrap_size) is not int
        or bootstrap_size <= 0
        or len(bootstrap_raw) != bootstrap_size
        or hashlib.sha256(bootstrap_raw).hexdigest() != bootstrap_sha
        or bootstrap_sha != runtime_code.get(NOSITE_BOOTSTRAP_SOURCE)
    ):
        raise ExactResumeVerificationError(
            "no-site bootstrap is not the claim-pinned commit bytes"
        )
    requested_interpreter = runtime.get("path")
    current_interpreter = os.path.normpath(os.path.abspath(sys.executable))
    if (
        type(requested_interpreter) is not str
        or requested_interpreter != current_interpreter
    ):
        raise ExactResumeVerificationError(
            "exact-resume verifier interpreter differs from claim runtime"
        )
    inventory_path = _absolute_path(
        inventory_identity.get("path"), "runtime inventory receipt"
    )
    inventory_document, inventory_snapshot = _snapshot_json(
        inventory_path, "runtime inventory receipt"
    )
    inventory_row = _exact_dict(
        inventory_document,
        ("schema_version", "kind", "content", "content_sha256"),
        "runtime inventory receipt",
    )
    inventory_content = inventory_row["content"]
    python_identity = (
        inventory_content.get("python")
        if type(inventory_content) is dict
        else None
    )
    if (
        inventory_snapshot["raw"]
        != _canonical_utf8_bytes(inventory_document) + b"\n"
        or inventory_snapshot["sha256"]
        != inventory_identity.get("file_sha256")
        or inventory_row["schema_version"] != 2
        or inventory_row["kind"] != "action_ball_runtime_inventory_v2"
        or inventory_row["kind"] != inventory_identity.get("kind")
        or type(inventory_content) is not dict
        or inventory_row["content_sha256"]
        != hashlib.sha256(
            _canonical_utf8_bytes(inventory_content)
        ).hexdigest()
        or inventory_row["content_sha256"]
        != inventory_identity.get("content_sha256")
        or type(python_identity) is not dict
        or python_identity.get("requested_path") != requested_interpreter
    ):
        raise ExactResumeVerificationError(
            "runtime inventory artifact differs from the claim"
        )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in _SANITIZED_ENV_ALLOWLIST
    }
    environment["PATH"] = os.defpath
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        outer_execution = python_identity["probe"]["no_site_execution"][
            "outer"
        ]
        import_roots = outer_execution["import_roots"]
        if import_roots != inventory_identity["import_roots"]:
            raise ExactResumeVerificationError(
                "runtime inventory import roots differ from claim identity"
            )
        nosite = _load_nosite_bootstrap_module(bootstrap_path)
        command = nosite.build_exact_nosite_argv(
            python=Path(current_interpreter),
            bootstrap=bootstrap_path,
            bootstrap_sha256=bootstrap_sha,
            entrypoint=source_path,
            entrypoint_sha256=source_sha,
            import_roots=import_roots,
            entrypoint_argv=[
                "verify",
                "--receipt",
                str(inventory_path),
            ],
        )
        nosite.validate_exact_nosite_argv(
            command.argv,
            expected_python=Path(current_interpreter),
            expected_bootstrap=command.contract["bootstrap"],
            expected_entrypoint=command.contract["entrypoint"],
            expected_import_roots=import_roots,
            expected_entrypoint_argv=[
                "verify",
                "--receipt",
                str(inventory_path),
            ],
            expected_contract_sha256=(
                inventory_identity[
                    "nosite_verification_contract_sha256"
                ]
            ),
            verify_live=True,
        )
        if (
            command.contract_sha256
            != inventory_identity[
                "nosite_verification_contract_sha256"
            ]
        ):
            raise ExactResumeVerificationError(
                "runtime inventory no-site contract differs from claim"
            )
        completed = subprocess.run(
            list(command.argv),
            cwd=os.sep,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=600,
            check=False,
        )
    except ExactResumeVerificationError:
        raise
    except (KeyError, TypeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        raise ExactResumeVerificationError(
            "pre-import runtime inventory verifier could not execute"
        ) from exc
    if completed.returncode != 0:
        raise ExactResumeVerificationError(
            "pre-import runtime inventory verification failed: {}".format(
                completed.stderr.decode("utf-8", errors="replace")[-4000:]
            )
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise ExactResumeVerificationError(
            "pre-import runtime inventory emitted no unique result"
        )
    result = _strict_json_bytes(lines[0], "runtime inventory verification result")
    expected_result = {
        "ok": True,
        "kind": "action_ball_runtime_inventory_v2",
        "content_sha256": inventory_identity["content_sha256"],
        "receipt_path": str(inventory_path),
        "receipt_sha256": inventory_identity["file_sha256"],
    }
    if (
        result != expected_result
        or lines[0] != _canonical_utf8_bytes(expected_result)
    ):
        raise ExactResumeVerificationError(
            "pre-import runtime inventory result differs from its claim"
        )
    content = {
        "schema_version": 1,
        "kind": "action_ball_runtime_inventory_live_verification",
        "verifier_source": {
            "path": str(source_path),
            "sha256": source_sha,
            "size_bytes": source_size,
        },
        "inventory_artifact": {
            "path": str(inventory_path),
            "sha256": inventory_snapshot["sha256"],
            "size_bytes": inventory_snapshot["size_bytes"],
        },
        "inventory_content_sha256": inventory_identity["content_sha256"],
        "current_interpreter": current_interpreter,
        "verification_result": result,
    }
    return {
        "schema_version": 1,
        "kind": "action_ball_runtime_inventory_live_verification",
        "content": content,
        "content_sha256": canonical_sha256(content),
    }


def _rsl_log_dir(
    *,
    checkout: Path,
    namespace: Path,
    experiment_name: str,
) -> Path:
    log = _snapshot_file(namespace / "train.log", "trainer log")
    try:
        text = log["raw"].decode("utf-8")
    except UnicodeError as exc:
        raise ExactResumeVerificationError(
            "trainer log is not UTF-8"
        ) from exc
    prefix = (
        "[INFO] Task: {} | experiment: {} | log: ".format(
            TASK_ID, experiment_name
        )
    )
    rows = [
        line[len(prefix) :]
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(rows) != 1:
        raise ExactResumeVerificationError(
            "trainer log must name exactly one claim-bound RSL output"
        )
    output = _absolute_path(rows[0], "RSL output")
    expected_parent = (
        checkout
        / "hope_training/whole_body_tracking/logs/rsl_rl"
        / experiment_name
    )
    if output.parent != expected_parent:
        raise ExactResumeVerificationError(
            "RSL output escaped the dedicated experiment root"
        )
    suffix = "_" + namespace.name
    if not output.name.endswith(suffix):
        raise ExactResumeVerificationError(
            "RSL output basename does not bind namespace"
        )
    timestamp = output.name[: -len(suffix)]
    if TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise ExactResumeVerificationError(
            "RSL output timestamp is not canonical"
        )
    _real_directory_chain(output, "RSL output")
    return output


def _validate_final_checkpoint_path(
    *,
    checkpoint_path: Path,
    rsl_log_dir: Path,
    max_iterations: int,
) -> int:
    if checkpoint_path.parent != rsl_log_dir:
        raise ExactResumeVerificationError(
            "source checkpoint is not in the claim-bound RSL output"
        )
    match = CHECKPOINT_RE.fullmatch(checkpoint_path.name)
    if match is None:
        raise ExactResumeVerificationError(
            "source checkpoint is not model_<N>.pt"
        )
    iteration = int(match.group(1))
    candidates = []
    for path in rsl_log_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        candidate = CHECKPOINT_RE.fullmatch(path.name)
        if candidate is not None:
            candidates.append(int(candidate.group(1)))
    if not candidates or iteration != max(candidates):
        raise ExactResumeVerificationError(
            "source checkpoint is not the terminal model_<N>.pt"
        )
    if iteration < max_iterations - 1 or iteration > max_iterations:
        raise ExactResumeVerificationError(
            "terminal checkpoint iteration is outside stage budget"
        )
    return iteration


def _load_checkpoint(
    snapshot: Mapping[str, Any],
    *,
    checkpoint_loader: Optional[Callable[[bytes], Any]] = None,
) -> Dict[str, Any]:
    if checkpoint_loader is None:
        try:
            import torch

            checkpoint = torch.load(
                io.BytesIO(snapshot["raw"]),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ExactResumeVerificationError(
                "checkpoint cannot be decoded by the safe weights-only loader"
            ) from exc
    else:
        checkpoint = checkpoint_loader(snapshot["raw"])
    if type(checkpoint) is not dict:
        raise ExactResumeVerificationError(
            "checkpoint must decode to one plain mapping"
        )
    return checkpoint


def _tree_digest(value: Any, *, torch_module: Any = None) -> str:
    digest = hashlib.sha256()
    seen: set[int] = set()

    def emit(raw: bytes) -> None:
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)

    def walk(item: Any) -> None:
        if torch_module is not None and torch_module.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            emit(b"tensor")
            emit(str(tensor.dtype).encode("ascii"))
            emit(_canonical_bytes(list(tensor.shape)))
            try:
                raw = tensor.view(torch_module.uint8).numpy().tobytes()
            except Exception as exc:
                raise ExactResumeVerificationError(
                    "tensor cannot be hashed losslessly"
                ) from exc
            emit(raw)
            return
        identity = id(item)
        if isinstance(item, (dict, list, tuple, set)):
            if identity in seen:
                raise ExactResumeVerificationError(
                    "checkpoint core contains a cyclic container"
                )
            seen.add(identity)
        if item is None:
            emit(b"none")
        elif type(item) is bool:
            emit(b"bool1" if item else b"bool0")
        elif type(item) is int:
            emit(b"int")
            emit(str(item).encode("ascii"))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ExactResumeVerificationError(
                    "checkpoint core contains a non-finite float"
                )
            emit(b"float")
            emit(item.hex().encode("ascii"))
        elif type(item) is str:
            emit(b"str")
            emit(item.encode("utf-8"))
        elif type(item) is bytes:
            emit(b"bytes")
            emit(item)
        elif isinstance(item, Mapping):
            emit(b"mapping")
            keyed = [(_tree_digest(key, torch_module=torch_module), key) for key in item]
            for key_digest, key in sorted(keyed, key=lambda row: row[0]):
                emit(key_digest.encode("ascii"))
                walk(key)
                walk(item[key])
        elif isinstance(item, tuple):
            emit(b"tuple")
            for nested in item:
                walk(nested)
        elif isinstance(item, list):
            emit(b"list")
            for nested in item:
                walk(nested)
        else:
            try:
                import numpy as np
            except ImportError:
                np = None
            if np is not None and isinstance(item, np.ndarray):
                emit(b"ndarray")
                emit(str(item.dtype).encode("ascii"))
                emit(_canonical_bytes(list(item.shape)))
                emit(item.tobytes(order="C"))
            elif np is not None and isinstance(item, np.generic):
                walk(item.item())
            else:
                raise ExactResumeVerificationError(
                    "checkpoint core contains unsupported type {}.{}".format(
                        type(item).__module__, type(item).__qualname__
                    )
                )
        if isinstance(item, (dict, list, tuple, set)):
            seen.remove(identity)

    walk(value)
    return digest.hexdigest()


def _checkpoint_core(
    checkpoint: Mapping[str, Any],
    *,
    expected_iteration: int,
    claim_sha256: str,
    torch_module: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if checkpoint.get("iter") != expected_iteration:
        raise ExactResumeVerificationError(
            "checkpoint embedded iteration differs"
        )
    infos = checkpoint.get("infos")
    if type(infos) is not dict:
        raise ExactResumeVerificationError(
            "checkpoint infos are missing"
        )
    exact_state = infos.get("hope_exact_resume_state")
    if (
        type(exact_state) is not dict
        or exact_state.get("schema_version") != 3
        or exact_state.get("next_learning_iteration")
        != expected_iteration + 1
        or infos.get("training_launch_claim_sha256") != claim_sha256
        or infos.get("training_contract_schema_version") != 3
        or infos.get("training_contract_lineage_exact") != 1
    ):
        raise ExactResumeVerificationError(
            "checkpoint exact resume/training lineage is incomplete"
        )
    bootstrap_keys = (
        "runtime_bootstrap_receipt_sha256",
        "runtime_bootstrap_lineage_payload_sha256",
        "runtime_bootstrap_receipt",
    )
    try:
        info_bootstrap = {key: infos[key] for key in bootstrap_keys}
        state_bootstrap = {key: exact_state[key] for key in bootstrap_keys}
    except KeyError as exc:
        raise ExactResumeVerificationError(
            "checkpoint runtime bootstrap binding is absent"
        ) from exc
    if info_bootstrap != state_bootstrap:
        raise ExactResumeVerificationError(
            "checkpoint infos/exact runtime bootstrap bindings differ"
        )
    _sha256(
        info_bootstrap["runtime_bootstrap_receipt_sha256"],
        "runtime bootstrap content SHA",
    )
    _sha256(
        info_bootstrap["runtime_bootstrap_lineage_payload_sha256"],
        "runtime bootstrap lineage SHA",
    )
    if (
        type(checkpoint.get("model_state_dict")) is not dict
        or not checkpoint["model_state_dict"]
        or type(checkpoint.get("optimizer_state_dict")) is not dict
        or not checkpoint["optimizer_state_dict"]
    ):
        raise ExactResumeVerificationError(
            "checkpoint policy/optimizer state is empty"
        )
    _validate_numpy_rng_state(exact_state.get("numpy_random_state"))
    normalizer_keys = sorted(
        key
        for key in checkpoint
        if type(key) is str
        and "norm" in key.lower()
        and key.endswith("state_dict")
    )
    core = {
        "iter": checkpoint["iter"],
        "model_state_dict": checkpoint["model_state_dict"],
        "optimizer_state_dict": checkpoint["optimizer_state_dict"],
        "normalizers": {
            key: checkpoint[key] for key in normalizer_keys
        },
        "exact_resume_state": exact_state,
        "training_contract_sha256": infos.get(
            "training_contract_sha256"
        ),
        "training_launch_claim_sha256": infos.get(
            "training_launch_claim_sha256"
        ),
        "runtime_bootstrap": info_bootstrap,
    }
    return core, {
        "core_sha256": _tree_digest(core, torch_module=torch_module),
        "exact_resume_state_sha256": _tree_digest(
            exact_state, torch_module=torch_module
        ),
        "model_state_sha256": _tree_digest(
            checkpoint["model_state_dict"], torch_module=torch_module
        ),
        "optimizer_state_sha256": _tree_digest(
            checkpoint["optimizer_state_dict"], torch_module=torch_module
        ),
        "normalizer_state_sha256": _tree_digest(
            core["normalizers"], torch_module=torch_module
        ),
        "runtime_bootstrap_receipt_sha256": info_bootstrap[
            "runtime_bootstrap_receipt_sha256"
        ],
        "runtime_bootstrap_lineage_payload_sha256": info_bootstrap[
            "runtime_bootstrap_lineage_payload_sha256"
        ],
    }


def _validate_construction_receipt(
    value: Any,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_size_bytes: int,
    iteration: int,
    claim_sha256: str,
    bootstrap_content_sha256: str,
    bootstrap_lineage_sha256: str,
    bootstrap_artifact_sha256: str,
    bootstrap_artifact_size_bytes: int,
    checkout: Path,
    runtime_inventory_identity: Mapping[str, Any],
    runtime_inventory_source_sha256: str,
    expected_interpreter: str,
) -> Dict[str, Any]:
    envelope = _exact_dict(
        value,
        ("schema_version", "kind", "content", "content_sha256"),
        "runtime construction receipt",
    )
    content = _exact_dict(
        envelope["content"],
        (
            "schema_version",
            "kind",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
            "checkpoint_iteration",
            "load_optimizer",
            "bootstrap_content_sha256",
            "bootstrap_artifact_sha256",
            "bootstrap_artifact_size_bytes",
            "bootstrap_lineage_payload_sha256",
            "runtime_inventory_live_verification",
            "exact_resume_live_state",
            "training_contract_sha256",
            "training_launch_claim_sha256",
            "environment_count",
            "runner_current_learning_iteration",
        ),
        "runtime construction receipt content",
    )
    if (
        envelope["schema_version"] != 1
        or envelope["kind"]
        != "action_ball_exact_resume_runtime_construction"
        or envelope["content_sha256"] != canonical_sha256(content)
        or content["schema_version"] != 1
        or content["kind"]
        != "action_ball_exact_resume_runtime_construction"
        or content["checkpoint_path"] != str(checkpoint_path)
        or content["checkpoint_sha256"] != checkpoint_sha256
        or content["checkpoint_size_bytes"]
        != checkpoint_size_bytes
        or content["checkpoint_iteration"] != iteration
        or content["load_optimizer"] is not True
        or content["bootstrap_content_sha256"]
        != bootstrap_content_sha256
        or content["bootstrap_artifact_sha256"]
        != bootstrap_artifact_sha256
        or content["bootstrap_artifact_size_bytes"]
        != bootstrap_artifact_size_bytes
        or content["bootstrap_lineage_payload_sha256"]
        != bootstrap_lineage_sha256
        or content["training_launch_claim_sha256"] != claim_sha256
        or content["runner_current_learning_iteration"] != iteration + 1
    ):
        raise ExactResumeVerificationError(
            "runtime construction receipt does not prove strict no-step load"
        )
    _sha256(
        content["training_contract_sha256"],
        "construction training contract SHA",
    )
    _plain_int(
        content["environment_count"],
        "construction environment count",
        minimum=1,
    )
    live = _exact_dict(
        content["runtime_inventory_live_verification"],
        ("schema_version", "kind", "content", "content_sha256"),
        "runtime inventory live verification",
    )
    live_content = _exact_dict(
        live["content"],
        (
            "schema_version",
            "kind",
            "verifier_source",
            "inventory_artifact",
            "inventory_content_sha256",
            "current_interpreter",
            "verification_result",
        ),
        "runtime inventory live verification content",
    )
    verifier_artifact = _exact_dict(
        live_content["verifier_source"],
        ("path", "sha256", "size_bytes"),
        "runtime inventory live verifier artifact",
    )
    inventory_artifact = _exact_dict(
        live_content["inventory_artifact"],
        ("path", "sha256", "size_bytes"),
        "runtime inventory live artifact",
    )
    verification_result = _exact_dict(
        live_content["verification_result"],
        (
            "ok",
            "kind",
            "content_sha256",
            "receipt_path",
            "receipt_sha256",
        ),
        "runtime inventory live verification result",
    )
    inventory_path = runtime_inventory_identity.get("path")
    inventory_file_sha = runtime_inventory_identity.get("file_sha256")
    inventory_content_sha = runtime_inventory_identity.get(
        "content_sha256"
    )
    if (
        live["schema_version"] != 1
        or live["kind"]
        != "action_ball_runtime_inventory_live_verification"
        or live["content_sha256"] != canonical_sha256(live_content)
        or live_content["schema_version"] != 1
        or live_content["kind"]
        != "action_ball_runtime_inventory_live_verification"
        or verifier_artifact["path"]
        != str(checkout / RUNTIME_INVENTORY_SOURCE)
        or verifier_artifact["sha256"]
        != runtime_inventory_source_sha256
        or _plain_int(
            verifier_artifact["size_bytes"],
            "runtime inventory verifier size",
            minimum=1,
        )
        != verifier_artifact["size_bytes"]
        or inventory_artifact["path"] != inventory_path
        or inventory_artifact["sha256"] != inventory_file_sha
        or _plain_int(
            inventory_artifact["size_bytes"],
            "runtime inventory artifact size",
            minimum=1,
        )
        != inventory_artifact["size_bytes"]
        or live_content["inventory_content_sha256"]
        != inventory_content_sha
        or live_content["current_interpreter"] != expected_interpreter
        or verification_result
        != {
            "ok": True,
            "kind": "action_ball_runtime_inventory_v1",
            "content_sha256": inventory_content_sha,
            "receipt_path": inventory_path,
            "receipt_sha256": inventory_file_sha,
        }
    ):
        raise ExactResumeVerificationError(
            "runtime inventory live verification does not bind the claim"
        )
    live_state = _exact_dict(
        content["exact_resume_live_state"],
        ("schema_version", "kind", "content", "content_sha256"),
        "exact-resume live-state receipt",
    )
    live_state_content = _exact_dict(
        live_state["content"],
        (
            "schema_version",
            "kind",
            "source_embedded_iteration",
            "current_learning_iteration",
            "roundtrip_pending",
            "resume_reset_pending",
            "model_state_sha256",
            "optimizer_state_sha256",
            "actor_normalizer_state_sha256",
            "critic_normalizer_state_sha256",
            "exact_resume_state_sha256",
            "environment_resume_state_sha256",
            "rng_state_sha256",
            "runtime_bootstrap_binding_sha256",
            "common_step_counter",
            "common_step_counter_delta",
            "live_core_sha256",
        ),
        "exact-resume live-state content",
    )
    digest_fields = (
        "model_state_sha256",
        "optimizer_state_sha256",
        "actor_normalizer_state_sha256",
        "critic_normalizer_state_sha256",
        "exact_resume_state_sha256",
        "environment_resume_state_sha256",
        "rng_state_sha256",
        "runtime_bootstrap_binding_sha256",
        "live_core_sha256",
    )
    if (
        live_state["schema_version"] != 1
        or live_state["kind"] != "action_ball_exact_resume_live_state"
        or live_state["content_sha256"]
        != canonical_sha256(live_state_content)
        or live_state_content["schema_version"] != 1
        or live_state_content["kind"]
        != "action_ball_exact_resume_live_state"
        or live_state_content["source_embedded_iteration"] != iteration
        or live_state_content["current_learning_iteration"]
        != iteration + 1
        or live_state_content["roundtrip_pending"] is not True
        or live_state_content["resume_reset_pending"] is not True
        or live_state_content["common_step_counter_delta"] != 0
    ):
        raise ExactResumeVerificationError(
            "runtime live-state receipt does not prove an unused no-step load"
        )
    _plain_int(
        live_state_content["common_step_counter"],
        "runtime live-state common step counter",
        minimum=0,
    )
    for field in digest_fields:
        _sha256(
            live_state_content[field],
            "runtime live-state {}".format(field),
        )
    return content


def _load_committed_module(
    source: Mapping[str, Any],
    *,
    module_name: str,
) -> Any:
    path = source.get("path")
    raw = source.get("raw")
    digest = source.get("sha256")
    size_bytes = source.get("size_bytes")
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or type(raw) is not bytes
        or type(size_bytes) is not int
        or size_bytes <= 0
        or len(raw) != size_bytes
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        raise ExactResumeVerificationError(
            "committed module snapshot is not internally exact"
        )
    if module_name in sys.modules:
        raise ExactResumeVerificationError(
            "committed module name is already occupied: {}".format(module_name)
        )
    try:
        code = compile(
            raw,
            str(path),
            "exec",
            dont_inherit=True,
            optimize=0,
        )
    except (SyntaxError, ValueError) as exc:
        raise ExactResumeVerificationError(
            "cannot compile committed module bytes: {}".format(path)
        ) from exc
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_sidecar_factory(
    sidecar_source: Mapping[str, Any],
    inbox_source: Mapping[str, Any],
) -> Callable[..., Any]:
    inbox_module = _load_committed_module(
        inbox_source,
        module_name="action_ball_evaluation_inbox",
    )
    module_name = "_action_ball_exact_resume_sidecar_{}".format(
        secrets.token_hex(8)
    )
    try:
        module = _load_committed_module(
            sidecar_source,
            module_name=module_name,
        )
    except BaseException:
        sys.modules.pop("action_ball_evaluation_inbox", None)
        raise
    if getattr(module, "inbox_protocol", None) is not inbox_module:
        sys.modules.pop(module_name, None)
        sys.modules.pop("action_ball_evaluation_inbox", None)
        raise ExactResumeVerificationError(
            "sidecar did not bind the committed inbox protocol snapshot"
        )
    factory = getattr(
        module, "build_exact_resume_runtime_from_claim", None
    )
    if not callable(factory) or getattr(factory, "__module__", None) != module_name:
        sys.modules.pop(module_name, None)
        sys.modules.pop("action_ball_evaluation_inbox", None)
        raise ExactResumeVerificationError(
            "sidecar lacks shared exact-resume factory"
        )
    return factory


def _production_torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ExactResumeVerificationError(
            "torch is required to verify the actual checkpoint"
        ) from exc
    return torch


def _launch_isaac_app() -> Any:
    try:
        from isaaclab.app import AppLauncher

        return AppLauncher(
            headless=True,
            device="cuda:0",
            enable_cameras=False,
        ).app
    except Exception as exc:
        raise ExactResumeVerificationError(
            "cannot launch the claim-bound Isaac runtime"
        ) from exc


def _runtime_parts(runtime: Any) -> Tuple[Any, Any, Mapping[str, Any], Callable[[], None]]:
    wrapped = getattr(runtime, "wrapped_env", None)
    runner = getattr(runtime, "runner", None)
    receipt = getattr(runtime, "construction_receipt", None)
    close = getattr(runtime, "close", None)
    if (
        wrapped is None
        or runner is None
        or type(receipt) is not dict
        or not callable(close)
    ):
        raise ExactResumeVerificationError(
            "shared factory returned an invalid ExactResumeRuntime"
        )
    return wrapped, runner, receipt, close


def verify_exact_resume(
    *,
    claim_path: str,
    checkpoint_path: str,
    output_path: str,
) -> Dict[str, Any]:
    claim_file = _absolute_path(claim_path, "launch claim")
    output = _absolute_path(
        output_path, "exact-resume receipt output", must_exist=False
    )
    claim, payload, _claim_snapshot, checkout, source_commit = (
        _validate_claim(claim_file)
    )
    namespace = _absolute_path(payload["namespace"], "stage namespace")
    if output != namespace / "exact_resume_verification.json":
        raise ExactResumeVerificationError(
            "receipt output is not the canonical namespace path"
        )
    if os.path.lexists(output):
        raise ExactResumeVerificationError(
            "exact-resume receipt namespace is already spent"
        )
    stage = payload.get("stage")
    budget = payload.get("stage_budget")
    if (
        stage not in ("smoke", "canary", "long")
        or type(budget) is not dict
    ):
        raise ExactResumeVerificationError(
            "claim stage/budget is invalid"
        )
    max_iterations = _plain_int(
        budget.get("max_iterations"), "stage max_iterations", minimum=1
    )
    rsl_log_dir = _rsl_log_dir(
        checkout=checkout,
        namespace=namespace,
        experiment_name=payload["action_set_contract"]["experiment_name"],
    )
    source_checkpoint_path = _absolute_path(
        checkpoint_path, "source checkpoint"
    )
    iteration = _validate_final_checkpoint_path(
        checkpoint_path=source_checkpoint_path,
        rsl_log_dir=rsl_log_dir,
        max_iterations=max_iterations,
    )
    verifier_source = _committed_source(
        checkout,
        source_commit,
        VERIFIER_SOURCE,
        "exact-resume verifier source",
    )
    sidecar_source = _committed_source(
        checkout,
        source_commit,
        SIDECAR_SOURCE,
        "exact-resume runtime factory source",
    )
    inventory_source = _committed_source(
        checkout,
        source_commit,
        RUNTIME_INVENTORY_SOURCE,
        "runtime inventory verifier source",
    )
    inbox_source = _committed_source(
        checkout,
        source_commit,
        EVALUATION_INBOX_SOURCE,
        "evaluation inbox protocol source",
    )
    isaac_runtime = payload.get("isaac_python_runtime")
    runtime_code_sha256 = payload.get("runtime_code_sha256")
    runtime_inventory_identity = (
        isaac_runtime.get("runtime_inventory")
        if type(isaac_runtime) is dict
        else None
    )
    expected_interpreter = (
        isaac_runtime.get("path")
        if type(isaac_runtime) is dict
        else None
    )
    runtime_inventory_source_sha256 = (
        runtime_code_sha256.get(RUNTIME_INVENTORY_SOURCE)
        if type(runtime_code_sha256) is dict
        else None
    )
    if (
        type(runtime_inventory_identity) is not dict
        or set(runtime_inventory_identity)
        != {"path", "file_sha256", "content_sha256", "kind"}
        or type(expected_interpreter) is not str
        or not expected_interpreter
    ):
        raise ExactResumeVerificationError(
            "claim lacks exact runtime inventory/interpreter identity"
        )
    _sha256(
        runtime_inventory_source_sha256,
        "runtime inventory verifier source SHA",
    )
    preimport_live_inventory = _preimport_runtime_inventory_verification(
        payload=payload,
        checkout=checkout,
        inventory_source=inventory_source,
    )
    factory = _load_sidecar_factory(sidecar_source, inbox_source)
    source_snapshot = _snapshot_file(
        source_checkpoint_path, "source final checkpoint"
    )
    source_checkpoint = _load_checkpoint(source_snapshot)
    # The public production entry point has no injectable loader/factory.  Unit
    # tests replace these private module functions, but no injected callable
    # can be passed through the API that publishes a formal PASS receipt.
    torch_module = _production_torch_module()
    source_core, source_state = _checkpoint_core(
        source_checkpoint,
        expected_iteration=iteration,
        claim_sha256=claim["launch_claim_sha256"],
        torch_module=torch_module,
    )
    bootstrap_artifact = _exact_dict(
        source_checkpoint["infos"]["runtime_bootstrap_receipt"],
        ("path", "sha256", "size_bytes"),
        "source checkpoint runtime bootstrap artifact",
    )
    _sha256(
        bootstrap_artifact["sha256"],
        "source checkpoint runtime bootstrap artifact SHA",
    )
    _plain_int(
        bootstrap_artifact["size_bytes"],
        "source checkpoint runtime bootstrap artifact size",
        minimum=1,
    )
    source_bootstrap_receipt = source_checkpoint["infos"][
        "runtime_bootstrap_receipt"
    ]
    # Do not retain a second decoded/raw copy while constructing a full Isaac
    # scene.  Long stages can have multi-gigabyte checkpoints.
    del source_core, source_checkpoint
    source_snapshot.pop("raw", None)
    roundtrip_dir = rsl_log_dir / (
        "exact_resume_roundtrip_"
        + claim["launch_claim_sha256"][:16]
    )
    try:
        os.mkdir(roundtrip_dir, 0o700)
    except FileExistsError as exc:
        raise ExactResumeVerificationError(
            "roundtrip directory namespace is already spent: {}".format(
                roundtrip_dir
            )
        ) from exc
    _fsync_directory(roundtrip_dir.parent)
    roundtrip_path = roundtrip_dir / source_checkpoint_path.name

    runtime_one = None
    runtime_two = None
    simulation_app = None
    factory_call_count = 0
    closed_runtime_count = 0
    fresh_strict_load_token_consumed = False
    try:
        simulation_app = _launch_isaac_app()
        runtime_one = factory(
            claim_document=claim,
            final_checkpoint_path=str(source_checkpoint_path),
            device="cuda:0",
            _preimport_live_inventory_verification=(
                preimport_live_inventory
            ),
        )
        factory_call_count += 1
        _wrapped_one, runner_one, construction_one_raw, close_one = (
            _runtime_parts(runtime_one)
        )
        construction_one = _validate_construction_receipt(
            construction_one_raw,
            checkpoint_path=source_checkpoint_path,
            checkpoint_sha256=source_snapshot["sha256"],
            checkpoint_size_bytes=source_snapshot["size_bytes"],
            iteration=iteration,
            claim_sha256=claim["launch_claim_sha256"],
            bootstrap_content_sha256=source_state[
                "runtime_bootstrap_receipt_sha256"
            ],
            bootstrap_lineage_sha256=source_state[
                "runtime_bootstrap_lineage_payload_sha256"
            ],
            bootstrap_artifact_sha256=bootstrap_artifact["sha256"],
            bootstrap_artifact_size_bytes=bootstrap_artifact[
                "size_bytes"
            ],
            checkout=checkout,
            runtime_inventory_identity=runtime_inventory_identity,
            runtime_inventory_source_sha256=(
                runtime_inventory_source_sha256
            ),
            expected_interpreter=expected_interpreter,
        )
        if (
            construction_one["runtime_inventory_live_verification"]
            != preimport_live_inventory
        ):
            raise ExactResumeVerificationError(
                "source restore did not consume the pre-import inventory proof"
            )
        roundtrip_save = getattr(
            runner_one, "save_exact_resume_roundtrip", None
        )
        if not callable(roundtrip_save):
            raise ExactResumeVerificationError(
                "runner lacks production no-step roundtrip save"
            )
        save_receipt = roundtrip_save(str(roundtrip_path))
        close_one()
        closed_runtime_count += 1
        runtime_one = None
        if type(save_receipt) is not dict:
            raise ExactResumeVerificationError(
                "roundtrip save did not return a receipt"
            )
        expected_save_keys = {
            "checkpoint",
            "source_embedded_iteration",
            "before_current_learning_iteration",
            "after_current_learning_iteration",
            "output_embedded_iteration",
            "output_next_learning_iteration",
            "runtime_bootstrap_receipt_sha256",
            "runtime_bootstrap_lineage_payload_sha256",
            "runtime_bootstrap_receipt",
        }
        if (
            set(save_receipt) != expected_save_keys
            or save_receipt["source_embedded_iteration"] != iteration
            or save_receipt["before_current_learning_iteration"]
            != iteration + 1
            or save_receipt["after_current_learning_iteration"]
            != iteration + 1
            or save_receipt["output_embedded_iteration"] != iteration
            or save_receipt["output_next_learning_iteration"]
            != iteration + 1
            or save_receipt["runtime_bootstrap_receipt_sha256"]
            != source_state["runtime_bootstrap_receipt_sha256"]
            or save_receipt[
                "runtime_bootstrap_lineage_payload_sha256"
            ]
            != source_state[
                "runtime_bootstrap_lineage_payload_sha256"
            ]
            or save_receipt["runtime_bootstrap_receipt"]
            != source_bootstrap_receipt
        ):
            raise ExactResumeVerificationError(
                "roundtrip save receipt does not prove zero-step identity"
            )
        fresh_strict_load_token_consumed = (
            getattr(
                runner_one, "_exact_resume_roundtrip_pending", None
            )
            is False
        )
        if not fresh_strict_load_token_consumed:
            raise ExactResumeVerificationError(
                "roundtrip save did not consume the fresh strict-load token"
            )
        roundtrip_snapshot = _snapshot_file(
            roundtrip_path, "roundtrip checkpoint"
        )
        checkpoint_receipt = _exact_dict(
            save_receipt["checkpoint"],
            ("path", "sha256", "size_bytes"),
            "roundtrip checkpoint artifact receipt",
        )
        if (
            checkpoint_receipt["path"] != str(roundtrip_path)
            or checkpoint_receipt["sha256"]
            != roundtrip_snapshot["sha256"]
            or checkpoint_receipt["size_bytes"]
            != roundtrip_snapshot["size_bytes"]
        ):
            raise ExactResumeVerificationError(
                "roundtrip save artifact receipt differs from installed bytes"
            )
        roundtrip_checkpoint = _load_checkpoint(
            roundtrip_snapshot
        )
        roundtrip_core, roundtrip_state = _checkpoint_core(
            roundtrip_checkpoint,
            expected_iteration=iteration,
            claim_sha256=claim["launch_claim_sha256"],
            torch_module=torch_module,
        )
        del roundtrip_core, roundtrip_checkpoint
        roundtrip_snapshot.pop("raw", None)
        if source_state != roundtrip_state:
            raise ExactResumeVerificationError(
                "roundtrip policy/optimizer/normalizer/RNG/env core drifted"
            )
        del _wrapped_one, runner_one, close_one, roundtrip_save
        gc.collect()
        cuda_module = getattr(torch_module, "cuda", None)
        empty_cache = (
            getattr(cuda_module, "empty_cache", None)
            if cuda_module is not None
            else None
        )
        if callable(empty_cache):
            empty_cache()
        runtime_two = factory(
            claim_document=claim,
            final_checkpoint_path=str(roundtrip_path),
            device="cuda:0",
            _preimport_live_inventory_verification=(
                preimport_live_inventory
            ),
        )
        factory_call_count += 1
        _wrapped_two, _runner_two, construction_two_raw, close_two = (
            _runtime_parts(runtime_two)
        )
        construction_two = _validate_construction_receipt(
            construction_two_raw,
            checkpoint_path=roundtrip_path,
            checkpoint_sha256=roundtrip_snapshot["sha256"],
            checkpoint_size_bytes=roundtrip_snapshot["size_bytes"],
            iteration=iteration,
            claim_sha256=claim["launch_claim_sha256"],
            bootstrap_content_sha256=source_state[
                "runtime_bootstrap_receipt_sha256"
            ],
            bootstrap_lineage_sha256=source_state[
                "runtime_bootstrap_lineage_payload_sha256"
            ],
            bootstrap_artifact_sha256=bootstrap_artifact["sha256"],
            bootstrap_artifact_size_bytes=bootstrap_artifact[
                "size_bytes"
            ],
            checkout=checkout,
            runtime_inventory_identity=runtime_inventory_identity,
            runtime_inventory_source_sha256=(
                runtime_inventory_source_sha256
            ),
            expected_interpreter=expected_interpreter,
        )
        if (
            construction_two["runtime_inventory_live_verification"]
            != preimport_live_inventory
        ):
            raise ExactResumeVerificationError(
                "roundtrip restore did not consume the pre-import inventory proof"
            )
        if (
            construction_one["runtime_inventory_live_verification"]
            != construction_two["runtime_inventory_live_verification"]
        ):
            raise ExactResumeVerificationError(
                "two restores observed different live runtime inventories"
            )
        source_live_state = construction_one[
            "exact_resume_live_state"
        ]
        roundtrip_live_state = construction_two[
            "exact_resume_live_state"
        ]
        if source_live_state["content"] != roundtrip_live_state["content"]:
            raise ExactResumeVerificationError(
                "two strict restores produced different live continuation state"
            )
        close_two()
        closed_runtime_count += 1
        runtime_two = None
        del _wrapped_two, _runner_two, close_two
        gc.collect()
    finally:
        for runtime in (runtime_two, runtime_one):
            if runtime is not None:
                close = getattr(runtime, "close", None)
                if callable(close):
                    close()
        if simulation_app is not None:
            simulation_app.close()

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": "passed",
        "source_commit_sha": source_commit,
        "launch_claim_sha256": claim["launch_claim_sha256"],
        "stage": stage,
        "namespace": str(namespace),
        "verifier": {
            "source_path": VERIFIER_SOURCE,
            "source_sha256": verifier_source["sha256"],
            "runtime_factory_source_path": SIDECAR_SOURCE,
            "runtime_factory_source_sha256": sidecar_source["sha256"],
        },
        "source_checkpoint": {
            "path": str(source_checkpoint_path),
            "sha256": source_snapshot["sha256"],
            "size_bytes": source_snapshot["size_bytes"],
            "embedded_iteration": iteration,
        },
        "roundtrip_checkpoint": {
            "path": str(roundtrip_path),
            "sha256": roundtrip_snapshot["sha256"],
            "size_bytes": roundtrip_snapshot["size_bytes"],
            "embedded_iteration": iteration,
        },
        "runtime_bootstrap": {
            "content_sha256": source_state[
                "runtime_bootstrap_receipt_sha256"
            ],
            "lineage_payload_sha256": source_state[
                "runtime_bootstrap_lineage_payload_sha256"
            ],
        },
        "restore": {
            "factory_call_count": factory_call_count,
            "closed_runtime_count": closed_runtime_count,
            "load_optimizer": (
                construction_one["load_optimizer"]
                and construction_two["load_optimizer"]
            ),
            # The production runner invalidates this one-use token at the
            # entry to ``learn`` and consumes it after the no-step save.  A
            # successful save plus this observed postcondition is therefore
            # direct evidence, not a caller-supplied PASS switch.
            "fresh_strict_load_token_consumed": (
                fresh_strict_load_token_consumed
            ),
            "roundtrip_save_api": "save_exact_resume_roundtrip",
            "roundtrip_save_receipt_sha256": canonical_sha256(
                save_receipt
            ),
            "source_construction_receipt_sha256": canonical_sha256(
                construction_one_raw
            ),
            "roundtrip_construction_receipt_sha256": canonical_sha256(
                construction_two_raw
            ),
            "runtime_inventory_live_verification_sha256": (
                construction_one[
                    "runtime_inventory_live_verification"
                ]["content_sha256"]
            ),
            "source_live_state_receipt_sha256": source_live_state[
                "content_sha256"
            ],
            "roundtrip_live_state_receipt_sha256": (
                roundtrip_live_state["content_sha256"]
            ),
            "live_core_sha256": source_live_state["content"][
                "live_core_sha256"
            ],
            "common_step_counter": source_live_state["content"][
                "common_step_counter"
            ],
            "common_step_counter_delta": source_live_state["content"][
                "common_step_counter_delta"
            ],
        },
        "state": {
            "source_core_sha256": source_state["core_sha256"],
            "roundtrip_core_sha256": roundtrip_state["core_sha256"],
            "source_exact_resume_sha256": source_state[
                "exact_resume_state_sha256"
            ],
            "roundtrip_exact_resume_sha256": roundtrip_state[
                "exact_resume_state_sha256"
            ],
            "model_state_sha256": source_state["model_state_sha256"],
            "optimizer_state_sha256": source_state[
                "optimizer_state_sha256"
            ],
            "normalizer_state_sha256": source_state[
                "normalizer_state_sha256"
            ],
        },
        "natural_exit": (
            factory_call_count == 2 and closed_runtime_count == 2
        ),
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    _publish_exclusive_json(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = verify_exact_resume(
            claim_path=args.claim,
            checkpoint_path=args.checkpoint,
            output_path=args.out,
        )
    except ExactResumeVerificationError as exc:
        print(
            "ACTION_BALL_EXACT_RESUME_REFUSED: {}".format(exc),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
