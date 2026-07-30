"""Exact runtime identity shared by the ActionBall trainer and evaluator."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Dict

from . import action_ball_evaluation_inbox as inbox_protocol


TASK_ID = "HOPE-PingPong-ActionBall-AgibotA3-v0"
RUNTIME_IDENTITY_KIND = "action_ball_frozen_eval_runtime_identity"
_SHA256_CHARS = frozenset("0123456789abcdef")
_GIT_OBJECT_FORMAT_LENGTHS = {
    "sha1": 40,
    "sha256": 64,
}
_PACKAGE_NAMES = (
    "torch",
    "isaaclab",
    "isaaclab-rl",
    "rsl-rl-lib",
    "gymnasium",
    "numpy",
)


class FrozenEvaluationRuntimeIdentityError(RuntimeError):
    """The trainer/evaluator runtime or resolved recipe is not identical."""


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise FrozenEvaluationRuntimeIdentityError(
            "{} must be 64 lowercase hexadecimal characters".format(name)
        )
    return value


def _git_object_id(
    value: object,
    *,
    object_format: object,
    name: str,
) -> str:
    if (
        type(object_format) is not str
        or object_format not in _GIT_OBJECT_FORMAT_LENGTHS
    ):
        raise FrozenEvaluationRuntimeIdentityError(
            "{} uses unsupported Git object format {!r}".format(
                name, object_format
            )
        )
    expected_length = _GIT_OBJECT_FORMAT_LENGTHS[object_format]
    if (
        type(value) is not str
        or len(value) != expected_length
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise FrozenEvaluationRuntimeIdentityError(
            "{} must be a {}-character lowercase {} Git object ID".format(
                name,
                expected_length,
                object_format,
            )
        )
    return value


def _stable_file_receipt(path: object, *, name: str) -> Dict[str, object]:
    candidate = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = candidate.lstat()
        descriptor = os.open(str(candidate), flags)
    except OSError as exc:
        raise FrozenEvaluationRuntimeIdentityError(
            "cannot open {}: {}".format(name, candidate)
        ) from exc
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_size,
            )
            != (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
            )
        ):
            raise FrozenEvaluationRuntimeIdentityError(
                "{} changed while opening".format(name)
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = candidate.lstat()
    signature = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        signature(before) != signature(after_descriptor)
        or signature(before) != signature(after_path)
        or size != before.st_size
    ):
        raise FrozenEvaluationRuntimeIdentityError(
            "{} changed while hashing".format(name)
        )
    return {
        "path": str(candidate),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FrozenEvaluationRuntimeIdentityError(
            "cannot inspect source checkout with git {}".format(" ".join(args))
        ) from exc


def _package_versions() -> Dict[str, str]:
    versions = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = ""
    try:
        import torch

        versions["torch_runtime"] = str(torch.__version__)
        versions["torch_cuda_build"] = str(torch.version.cuda or "")
    except ImportError as exc:
        raise FrozenEvaluationRuntimeIdentityError(
            "frozen evaluation requires torch"
        ) from exc
    return versions


def build_runtime_identity_document(
    *,
    repo_root: object,
    task_id: str,
    training_launch_claim_sha256: str,
    training_contract_path: object,
    environment_config_pickle_path: object,
    agent_config_pickle_path: object,
) -> Dict[str, object]:
    """Observe one exact, GPU-ordinal-independent trainer/evaluator runtime."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    if task_id != TASK_ID:
        raise FrozenEvaluationRuntimeIdentityError(
            "formal frozen evaluation requires task_id {!r}".format(TASK_ID)
        )
    claim_sha = _sha256(
        training_launch_claim_sha256,
        name="training_launch_claim_sha256",
    )
    executable = Path(sys.executable).expanduser().resolve(strict=True)
    object_format = _git(root, "rev-parse", "--show-object-format")
    head = _git_object_id(
        _git(root, "rev-parse", "HEAD"),
        object_format=object_format,
        name="source HEAD",
    )
    symbolic = subprocess.run(
        ("git", "symbolic-ref", "-q", "HEAD"),
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if symbolic.returncode not in (0, 1):
        raise FrozenEvaluationRuntimeIdentityError(
            "cannot determine whether the source checkout is detached"
        )
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    content = {
        "runtime_identity_contract_sha256": (
            inbox_protocol.RUNTIME_IDENTITY_CONTRACT_SHA256
        ),
        "resolved_recipe_contract_sha256": (
            inbox_protocol.RESOLVED_EVALUATION_RECIPE_CONTRACT_SHA256
        ),
        "task_id": task_id,
        "training_launch_claim_sha256": claim_sha,
        "training_contract": _stable_file_receipt(
            training_contract_path,
            name="training_contract.json",
        ),
        "environment_config_pickle": _stable_file_receipt(
            environment_config_pickle_path,
            name="env.pkl",
        ),
        "agent_config_pickle": _stable_file_receipt(
            agent_config_pickle_path,
            name="agent.pkl",
        ),
        "interpreter": {
            **_stable_file_receipt(
                executable,
                name="Python executable",
            ),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": str(sys.implementation.cache_tag),
        },
        "packages": _package_versions(),
        "source": {
            "repo_root": str(root),
            "object_format": object_format,
            "head_commit_oid": head,
            "detached": symbolic.returncode == 1,
            "clean": not bool(dirty),
        },
    }
    document = {
        "schema_version": 1,
        "kind": RUNTIME_IDENTITY_KIND,
        "content": content,
        "content_sha256": inbox_protocol.canonical_sha256(content),
    }
    return document


def validate_runtime_identity_document(
    document: object,
    *,
    repo_root: object,
    task_id: str,
    training_launch_claim_sha256: str,
    training_contract_path: object,
    environment_config_pickle_path: object,
    agent_config_pickle_path: object,
) -> Dict[str, object]:
    """Require byte-for-byte equality with a freshly observed live identity."""

    expected = build_runtime_identity_document(
        repo_root=repo_root,
        task_id=task_id,
        training_launch_claim_sha256=training_launch_claim_sha256,
        training_contract_path=training_contract_path,
        environment_config_pickle_path=environment_config_pickle_path,
        agent_config_pickle_path=agent_config_pickle_path,
    )
    if document != expected:
        raise FrozenEvaluationRuntimeIdentityError(
            "live evaluator runtime/resolved recipe differs from the trainer"
        )
    return dict(expected["content"])


def canonical_document_bytes(document: object) -> bytes:
    """Serialize the exact identity with one transport newline."""

    try:
        return (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise FrozenEvaluationRuntimeIdentityError(
            "runtime identity is not canonical JSON data"
        ) from exc


__all__ = [
    "FrozenEvaluationRuntimeIdentityError",
    "RUNTIME_IDENTITY_KIND",
    "TASK_ID",
    "build_runtime_identity_document",
    "canonical_document_bytes",
    "validate_runtime_identity_document",
]
