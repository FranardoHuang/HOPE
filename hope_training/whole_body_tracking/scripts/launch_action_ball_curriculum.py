#!/usr/bin/env python3
"""Fail-closed launcher for a code-owned ActionBall action-set curriculum.

This module is deliberately dependency-light.  ``plan`` performs every static
check without touching a GPU, a lock, or a run namespace.  ``launch`` repeats
the checks while holding the shared per-GPU lifetime lock, verifies that the
physical GPU is empty, atomically claims a never-before-used namespace, writes
an exclusive launch claim, and delegates only the Kit boot window to
``launch_kit_training_locked.sh``.

The launcher owns all scientific identity overrides.  It has no arbitrary CLI
override seam and rejects bank/exam keys even when they are present in the
content-addressed spec.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Iterable, Optional


SCHEMA_VERSION = 3
SPEC_KIND = "action_ball_no_clobber_launch_spec_v3"
CLAIM_KIND = "action_ball_no_clobber_launch_claim_v3"
LAUNCH_PROFILE = "fresh_upper_nomove_n5_v3"
ACTION_ORDER = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
STAGE_ORDER = ("smoke", "canary", "long")
ACTION_BALL_EXPERIMENT_NAME = "agibot_a3_hope_action_ball_fresh_n5"
TASK_PROFILE_ID = "HOPEPingPongActionBallA3VendorV1"
TASK_PROFILE_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV1.yaml"
)
LONG_MIN_NUM_ENVS = 4096
LONG_MIN_ITERATIONS = 20001
LONG_MAX_SAVE_INTERVAL = 100
GROUND_PLANT_ABSENT_SHA256 = hashlib.sha256(
    b'{"ground_plant":"absent-historical-default"}'
).hexdigest()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ED25519_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
OVERRIDE_KEY_RE = re.compile(r"^[+~]?[A-Za-z_][A-Za-z0-9_.-]*$")
RUN_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RSL_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

PROMOTION_TRUST_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "canonical_motion_admission.py"
)
EVALUATOR_TRUST_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_evaluation.py"
)
EVALUATION_INBOX_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_evaluation_inbox.py"
)
CURRICULUM_TRUST_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_curriculum.py"
)
HOPE_COMMANDS_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
)
SIDECAR_CODE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_frozen_eval_sidecar.py"
)
STAGE_SUPERVISOR_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_stage_supervisor.py"
)
EXACT_RESUME_VERIFIER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_exact_resume_verifier.py"
)
RUNTIME_INVENTORY_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_runtime_inventory.py"
)
PROPOSAL_SAMPLER_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_sampling.py"
)
RUNTIME_BOOTSTRAP_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_runtime_bootstrap.py"
)
PPO_RUNNER_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/my_on_policy_runner.py"
)
FRESH_ORDER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate.py"
)
FRESH_ORDER_NAME = "FRESH_N5_ORDER"
ACTION_SET_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_action_set_contract.py"
)
NOSITE_BOOTSTRAP_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_python_nosite_bootstrap.py"
)
ACTION_SET_CONTRACTS_NAME = "ACTION_SET_CONTRACTS"
ACTION_SET_PROFILE_POLICIES_NAME = "ACTION_SET_PROFILE_POLICIES"
ACTION_SET_ACTOR_OBS_CONTRACT_NAME = "ACTOR_OBS_CONTRACT"
ACTION_SET_ACTOR_OBS_WIDTH_NAME = "ACTOR_OBS_WIDTH"
PROMOTION_TRUST_NAME = "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256"
EVALUATOR_TRUST_NAME = (
    "TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256"
)
SIDECAR_CODE_TRUST_NAME = (
    "TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_CODE_SHA256"
)
SIDECAR_LAUNCH_TRUST_NAME = (
    "TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_LAUNCH_SHA256"
)
DRAIN_RESET_TRUST_NAME = "TRUSTED_DRAIN_RESET_LAUNCH_RECEIPT_SHA256"
LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_action_ball_curriculum.py"
)
TRAIN_SOURCE = "hope_training/whole_body_tracking/scripts/train.py"
KIT_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_kit_training_locked.sh"
)
PROCESS_GROUP_SOURCE = (
    "hope_training/whole_body_tracking/scripts/exact_process_group.py"
)
SETUP_SOURCE = "hope_training/whole_body_tracking/setup_train_env.sh"
RUNTIME_CODE_SOURCES = (
    LAUNCHER_SOURCE,
    ACTION_SET_CONTRACT_SOURCE,
    NOSITE_BOOTSTRAP_SOURCE,
    TRAIN_SOURCE,
    KIT_LAUNCHER_SOURCE,
    PROCESS_GROUP_SOURCE,
    SETUP_SOURCE,
    STAGE_SUPERVISOR_SOURCE,
    EXACT_RESUME_VERIFIER_SOURCE,
    RUNTIME_INVENTORY_SOURCE,
    PROPOSAL_SAMPLER_SOURCE,
    RUNTIME_BOOTSTRAP_SOURCE,
    PPO_RUNNER_SOURCE,
)
V4_EVALUATOR_WINDOW_CONTRACT = {
    "optional_stopping": False,
    "scheduler_proposals": 100,
    "canary_proposals": 320,
    "canary_safe_closed_min": 256,
    "heldout_proposals": 960,
    "heldout_safe_closed_min": 768,
    "sampling_mixture": {
        "center": 0.20,
        "interior": 0.60,
        "frontier": 0.20,
    },
}
SIDECAR_WINDOW_CONTRACT = {
    **V4_EVALUATOR_WINDOW_CONTRACT,
    "allocation": (
        "authority supplied disjoint contiguous seed/sample/birth ranges"
    ),
}
SIDECAR_HEARTBEAT_CONTRACT = {
    "schema_version": 1,
    "heartbeat_interval_seconds": 5.0,
    "heartbeat_stale_after_seconds": 120.0,
    "request_deadline_seconds": 7200.0,
}
_SIDECAR_HEARTBEAT_CONTENT_KEYS = (
    "owner_id",
    "run_id",
    "pid",
    "sidecar_code_sha256",
    "launch_sha256",
    "backend_contract_sha256",
    "heartbeat_seq",
    "phase",
    "request_seq",
    "request_sha256",
    "attempts_completed",
    "attempts_total",
    "request_started_unix_ns",
    "request_started_monotonic_ns",
    "request_deadline_unix_ns",
    "request_deadline_monotonic_ns",
    "heartbeat_unix_ns",
    "heartbeat_monotonic_ns",
    "error_type",
)
_PYTHON_PROBE = (
    "import json,platform,sys;"
    "print(json.dumps({"
    "'version':platform.python_version(),"
    "'cache_tag':sys.implementation.cache_tag,"
    "'import_roots':sys.path,"
    "'isolated':bool(sys.flags.isolated),"
    # ``safe_path`` became a named flag in Python 3.11.  On older supported
    # Isaac/host Pythons, ``-I`` already omits the script/current directory
    # from sys.path (and implies -E/-s), but the attribute is absent.  Report
    # the effective isolation guarantee without making Python 3.8 crash.
    "'safe_path':bool(getattr(sys.flags,'safe_path',sys.flags.isolated))"
    "},sort_keys=True,separators=(',',':')))"
)
_SANITIZED_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "TZ",
        "USER",
    }
)

_TOP_KEYS = (
    "schema_version",
    "kind",
    "launch_profile",
    "source",
    "action_set",
    "inputs",
    "policy_contract_sha256",
    "train",
    "gpus",
    "stages",
)
_OWNED_OVERRIDE_KEYS = frozenset(
    {
        "task",
        "algo",
        "headless",
        "video",
        "device",
        "seed",
        "motion_file",
        "num_envs",
        "max_iterations",
        "run_name",
        "training_launch_claim_path",
        "training_launch_claim_sha256",
        "expected_effective_reward_recipe_sha256",
        "action_ball_shared_ready_bootstrap",
        "algo.policy.init_noise_std",
        "algo.runner.save_interval",
        "task.experiment_name",
        "task.actor_obs_contract",
        "task.rewards.full_body_mimic",
        "task.racket.clip_names",
        "task.racket.target_mode",
        "task.racket.question_bank",
        "task.racket.question_bank_allow_legacy",
        "task.racket.cq_anchor_bank",
        "task.racket.exam_bank",
        "task.racket.action_ball_manifest_path",
        "task.racket.action_ball_manifest_sha256",
        "task.racket.action_ball_policy_contract_sha256",
        "task.racket.action_ball_evaluator_launch_receipt_path",
        "task.racket.action_ball_evaluator_launch_receipt_file_sha256",
        "task.racket.action_ball_sidecar_launch_receipt_path",
        "task.racket.action_ball_sidecar_launch_receipt_file_sha256",
        "task.racket.action_ball_drain_reset_launch_receipt_path",
        "task.racket.action_ball_drain_reset_launch_receipt_file_sha256",
        "task.racket.action_ball_evaluation_inbox_root",
        "task.racket.action_ball_evaluation_owner_id",
        "task.racket.action_ball_evaluation_run_id",
        "task.racket.action_ball_frozen_eval_interval_updates",
        "task.racket.action_ball_diagnostic_unauthorized",
        "task.racket.reference_guard_mode",
        "task.racket.action_ball_seed",
        "task.motion.canonical_registry_path",
        "task.motion.canonical_registry_repo_root",
        "task.motion.canonical_registry_sha256",
        "task.motion.canonical_registry_alignment_sha256",
        "task.motion.canonical_ready_sha256",
        "task.motion.canonical_ready_fk_sha256",
        "task.motion.canonical_promotion_certificate_path",
    }
)
_OWNED_OVERRIDE_PREFIXES = (
    "task.motion.",
    "task.racket.action_ball_",
)
# The fresh formal launch owns every scientific and source-root setting.
# ``logger=tensorboard`` only selects the already-local metrics sink and is the
# sole caller-provided non-scientific override admitted by this profile.
_ALLOWED_EXTRA_OVERRIDES = frozenset(("logger=tensorboard",))


class LaunchRefused(RuntimeError):
    """A fail-closed refusal with a user-actionable reason."""


class LaunchClosureUnknown(LaunchRefused):
    """The run was not accepted, but exact child closure is not yet proven."""


def _utc_now() -> str:
    return _datetime.datetime.now(
        _datetime.timezone.utc
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_module_without_bytecode(
    path: Path,
    *,
    name: str,
    purpose: str,
    retain_in_sys_modules: bool = False,
) -> Any:
    """Execute source without creating ``__pycache__`` in a clean checkout."""

    module_spec = importlib.util.spec_from_file_location(name, str(path))
    if module_spec is None or module_spec.loader is None:
        raise LaunchRefused(f"cannot load {purpose}")
    module = importlib.util.module_from_spec(module_spec)
    previous = sys.modules.get(name)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.modules[name] = module
    sys.dont_write_bytecode = True
    loaded = False
    try:
        module_spec.loader.exec_module(module)
        loaded = True
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        if not retain_in_sys_modules or not loaded:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LaunchRefused(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise LaunchRefused(f"non-finite JSON number {token!r}")
    return value


def _reject_constant(token: str) -> None:
    raise LaunchRefused(f"non-finite JSON constant {token!r}")


def _load_strict_json_bytes(raw: bytes, *, name: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise LaunchRefused(f"{name} must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise LaunchRefused(f"{name} is not valid JSON: {exc}") from exc


def load_strict_json(path: Path, *, name: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be read: {path}: {exc}") from exc
    return _load_strict_json_bytes(raw, name=name)


def _exact_dict(
    value: Any, keys: Iterable[str], *, name: str
) -> dict[str, Any]:
    expected = frozenset(keys)
    if type(value) is not dict:
        raise LaunchRefused(f"{name} must be a plain JSON object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise LaunchRefused(
            f"{name} keys differ: missing={missing}, extra={extra}"
        )
    return value


def _plain_int(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise LaunchRefused(f"{name} must be a plain integer")
    if minimum is not None and value < minimum:
        raise LaunchRefused(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise LaunchRefused(f"{name} must be <= {maximum}")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise LaunchRefused(
            f"{name} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _commit(value: Any, *, name: str) -> str:
    if type(value) is not str or COMMIT_RE.fullmatch(value) is None:
        raise LaunchRefused(
            f"{name} must be a full 40-character lowercase Git commit"
        )
    return value


def _assert_no_symlink_components(
    path: Path, *, start: Path, name: str
) -> None:
    try:
        relative = path.relative_to(start)
    except ValueError as exc:
        raise LaunchRefused(f"{name} escapes its trusted root") from exc
    current = start
    for component in relative.parts:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise LaunchRefused(
                f"{name} path component cannot be inspected: {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise LaunchRefused(
                f"{name} must not contain a symlink: {current}"
            )


def _absolute_normalized_path(
    value: Any, *, name: str, must_exist: bool
) -> Path:
    if type(value) is not str or not value:
        raise LaunchRefused(f"{name} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise LaunchRefused(f"{name} must be an absolute normalized path")
    if must_exist:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise LaunchRefused(f"{name} does not exist: {path}") from exc
        if resolved != path:
            raise LaunchRefused(
                f"{name} must not resolve through symlinks: {path}"
            )
    return path


def _repo_file(
    checkout: Path, relative_value: Any, *, name: str
) -> tuple[Path, str]:
    if type(relative_value) is not str or not relative_value:
        raise LaunchRefused(f"{name} must be a non-empty repo-relative path")
    pure = Path(relative_value)
    if (
        pure.is_absolute()
        or "\\" in relative_value
        or ":" in relative_value
        or "\x00" in relative_value
        or relative_value != pure.as_posix()
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise LaunchRefused(
            f"{name} must be a normalized POSIX repo-relative path"
        )
    candidate = checkout / pure
    _assert_no_symlink_components(
        candidate, start=checkout, name=name
    )
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be inspected: {candidate}") from exc
    if not stat.S_ISREG(mode):
        raise LaunchRefused(f"{name} must resolve to a regular file")
    return candidate, relative_value


def _external_regular_file(value: Any, *, name: str) -> Path:
    path = _absolute_normalized_path(value, name=name, must_exist=True)
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be inspected: {path}") from exc
    if not stat.S_ISREG(mode):
        raise LaunchRefused(f"{name} must be a regular non-symlink file")
    return path


def _trusted_system_executable(name: str) -> dict[str, str]:
    """Resolve one host tool without consulting caller-controlled PATH."""

    located = shutil.which(name, path=os.defpath)
    if located is None:
        raise LaunchRefused(
            f"required system executable is unavailable on os.defpath: {name}"
        )
    path = Path(located)
    if not path.is_absolute():
        raise LaunchRefused(f"system executable path is not absolute: {located}")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise LaunchRefused(
            f"system executable cannot be inspected: {path}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or not os.access(resolved, os.X_OK)
    ):
        raise LaunchRefused(
            f"system executable must resolve to one executable regular file: {path}"
        )
    return {
        "name": name,
        "requested_path": str(path),
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def _git_environment() -> dict[str, str]:
    """Return a fixed Git environment with caller repository/config seams cut."""

    return {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _verify_repo_pin(
    checkout: Path,
    source_commit: str,
    value: Any,
    *,
    name: str,
) -> tuple[Path, str, str]:
    row = _exact_dict(value, ("path", "sha256"), name=name)
    expected = _sha256(row["sha256"], name=f"{name}.sha256")
    path, relative, actual, _mode = _verify_repo_blob(
        checkout,
        source_commit,
        row["path"],
        name=f"{name}.path",
    )
    if actual != expected:
        raise LaunchRefused(
            f"{name} byte SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return path, relative, actual


def _verify_external_pin(value: Any, *, name: str) -> tuple[Path, str]:
    row = _exact_dict(value, ("path", "sha256"), name=name)
    expected = _sha256(row["sha256"], name=f"{name}.sha256")
    path = _external_regular_file(row["path"], name=f"{name}.path")
    actual = sha256_file(path)
    if actual != expected:
        raise LaunchRefused(
            f"{name} byte SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return path, actual


def _git_output(checkout: Path, *args: str) -> str:
    git = _trusted_system_executable("git")
    try:
        result = subprocess.run(
            [git["path"], "-C", str(checkout), *args],
            check=False,
            capture_output=True,
            text=True,
            env=_git_environment(),
        )
    except OSError as exc:
        raise LaunchRefused(f"git cannot inspect {checkout}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise LaunchRefused(
            f"git {' '.join(args)} failed for {checkout}: {detail}"
        )
    return result.stdout.strip()


def _git_bytes(checkout: Path, *args: str) -> bytes:
    git = _trusted_system_executable("git")
    try:
        result = subprocess.run(
            [git["path"], "-C", str(checkout), *args],
            check=False,
            capture_output=True,
            env=_git_environment(),
        )
    except OSError as exc:
        raise LaunchRefused(f"git cannot inspect {checkout}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-4000:].decode(
            "utf-8", errors="replace"
        ).strip()
        raise LaunchRefused(
            f"git {' '.join(args)} failed for {checkout}: {detail}"
        )
    return result.stdout


def _verify_repo_blob(
    checkout: Path,
    source_commit: str,
    relative_value: Any,
    *,
    name: str,
    executable: bool | None = None,
) -> tuple[Path, str, str, str]:
    """Bind one worktree file to an ordinary blob in the exact commit tree.

    `git status` intentionally ignores ignored files.  It is therefore not
    evidence that a file is committed.  Every scientific input and every
    executable/source opened by this launcher goes through this function.
    """

    path, relative = _repo_file(checkout, relative_value, name=name)
    listing = _git_bytes(
        checkout, "ls-tree", "-z", source_commit, "--", relative
    )
    rows = [row for row in listing.split(b"\x00") if row]
    if len(rows) != 1:
        raise LaunchRefused(
            f"{name} must exist exactly once in commit {source_commit}"
        )
    try:
        metadata, listed_path = rows[0].split(b"\t", 1)
        mode_raw, object_type, _object_id = metadata.split(b" ", 2)
        mode = mode_raw.decode("ascii")
        listed = listed_path.decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise LaunchRefused(
            f"{name} has an unparseable Git tree entry"
        ) from exc
    if listed != relative or object_type != b"blob" or mode not in {
        "100644",
        "100755",
    }:
        raise LaunchRefused(
            f"{name} must be one ordinary committed file; "
            f"path={listed!r}, type={object_type!r}, mode={mode!r}"
        )
    if executable is True and mode != "100755":
        raise LaunchRefused(f"{name} must be executable in the commit tree")
    if executable is False and mode != "100644":
        raise LaunchRefused(f"{name} must be non-executable in the commit tree")
    committed = _git_bytes(
        checkout, "cat-file", "blob", f"{source_commit}:{relative}"
    )
    try:
        working = path.read_bytes()
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be read: {path}: {exc}") from exc
    if working != committed:
        raise LaunchRefused(
            f"{name} worktree bytes differ from exact commit-tree bytes"
        )
    return path, relative, hashlib.sha256(committed).hexdigest(), mode


def _verify_checkout(source: Any) -> tuple[Path, str]:
    row = _exact_dict(
        source, ("checkout", "commit_sha"), name="spec.source"
    )
    checkout = _absolute_normalized_path(
        row["checkout"], name="spec.source.checkout", must_exist=True
    )
    if not checkout.is_dir():
        raise LaunchRefused("spec.source.checkout must be a directory")
    top = Path(_git_output(checkout, "rev-parse", "--show-toplevel")).resolve()
    if top != checkout:
        raise LaunchRefused(
            "spec.source.checkout must be the exact Git worktree root"
        )
    expected = _commit(
        row["commit_sha"], name="spec.source.commit_sha"
    )
    actual = _git_output(checkout, "rev-parse", "--verify", "HEAD")
    if actual != expected:
        raise LaunchRefused(
            f"checkout HEAD mismatch: expected={expected}, actual={actual}"
        )
    status = _git_output(
        checkout, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        first = status.splitlines()[0]
        raise LaunchRefused(
            "checkout is dirty (staged, unstaged, and untracked files are "
            f"all forbidden); first entry: {first}"
        )
    return checkout, expected


def _parse_literal_trust_set(
    source: Path, variable: str
) -> frozenset[str]:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), str(source))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise LaunchRefused(
            f"cannot parse code-owned trust source {source}: {exc}"
        ) from exc
    values: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == variable
                for target in node.targets
            ):
                values.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == variable
            and node.value is not None
        ):
            values.append(node.value)
    if len(values) != 1:
        raise LaunchRefused(
            f"{variable} must have exactly one top-level assignment"
        )
    expression = values[0]
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "frozenset"
        and not expression.keywords
        and len(expression.args) in (0, 1)
    ):
        raise LaunchRefused(
            f"{variable} RHS must be frozenset(<literal strings>)"
        )
    try:
        raw = (
            ()
            if not expression.args
            else ast.literal_eval(expression.args[0])
        )
    except (ValueError, SyntaxError) as exc:
        raise LaunchRefused(
            f"{variable} must contain only literal strings"
        ) from exc
    if type(raw) not in (tuple, list, set, frozenset):
        raise LaunchRefused(
            f"{variable} literal must be a tuple/list/set of strings"
        )
    trusted = frozenset(raw)
    if len(trusted) != len(raw) or any(
        type(item) is not str or SHA256_RE.fullmatch(item) is None
        for item in raw
    ):
        raise LaunchRefused(
            f"{variable} must contain unique lowercase SHA-256 strings"
        )
    return trusted


def _require_fresh_order_sentinel(
    checkout: Path, source_commit: str
) -> str:
    """Reject a clean checkout whose independent fresh-N5 sentinel is stale."""

    source, _, source_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        FRESH_ORDER_SOURCE,
        name=f"{FRESH_ORDER_NAME} source",
    )
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), str(source))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise LaunchRefused(
            f"cannot parse fresh-N5 order source {source}: {exc}"
        ) from exc
    values: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == FRESH_ORDER_NAME
            for target in node.targets
        ):
            values.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == FRESH_ORDER_NAME
            and node.value is not None
        ):
            values.append(node.value)
    if len(values) != 1:
        raise LaunchRefused(
            f"{FRESH_ORDER_NAME} must have exactly one top-level assignment"
        )
    try:
        raw = ast.literal_eval(values[0])
    except (ValueError, SyntaxError) as exc:
        raise LaunchRefused(
            f"{FRESH_ORDER_NAME} must be a literal ordered sequence"
        ) from exc
    if type(raw) not in (tuple, list) or any(
        type(item) is not str for item in raw
    ):
        raise LaunchRefused(
            f"{FRESH_ORDER_NAME} must be a literal string tuple/list"
        )
    if tuple(raw) != ACTION_ORDER:
        raise LaunchRefused(
            f"stale {FRESH_ORDER_NAME}: expected={list(ACTION_ORDER)}, "
            f"actual={list(raw)}"
        )
    return source_sha


def _require_exact_trust(
    checkout: Path,
    *,
    source_commit: str,
    source_relative: str,
    variable: str,
    expected_digest: str,
) -> str:
    source, _, source_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        source_relative,
        name=f"{variable} source",
    )
    trusted = _parse_literal_trust_set(source, variable)
    if not trusted:
        raise LaunchRefused(
            f"code-owned trust set {variable} is empty; formal launch is forbidden"
        )
    expected = frozenset((expected_digest,))
    if trusted != expected:
        raise LaunchRefused(
            f"code-owned trust set {variable} must equal the one approved "
            f"digest; expected={sorted(expected)}, actual={sorted(trusted)}"
        )
    return source_sha


def _literal_assignment(source: Path, variable: str) -> Any:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), str(source))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise LaunchRefused(
            f"cannot parse action-set contract source {source}: {exc}"
        ) from exc
    values: list[ast.AST] = []
    for node in tree.body:
        targets: list[ast.AST] = []
        value: Optional[ast.AST] = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is not None and any(
            isinstance(target, ast.Name) and target.id == variable
            for target in targets
        ):
            values.append(value)
    if len(values) != 1:
        raise LaunchRefused(
            f"{variable} must have exactly one top-level assignment"
        )
    try:
        return ast.literal_eval(values[0])
    except (ValueError, SyntaxError) as exc:
        raise LaunchRefused(f"{variable} must be a Python literal") from exc


def _order_uid_digest(order: list[str], uids: list[int]) -> str:
    rows = [
        {"index": index, "action_id": action_id, "action_uid": action_uid}
        for index, (action_id, action_uid) in enumerate(zip(order, uids))
    ]
    return canonical_sha256({"schema_version": 1, "ordered_actions": rows})


def _load_action_set_contract(
    checkout: Path,
    source_commit: str,
    spec_value: Any,
) -> tuple[dict[str, Any], str]:
    spec_row = _exact_dict(
        spec_value, ("contract_profile",), name="spec.action_set"
    )
    profile_id = spec_row["contract_profile"]
    if (
        type(profile_id) is not str
        or RUN_COMPONENT_RE.fullmatch(profile_id) is None
    ):
        raise LaunchRefused(
            "spec.action_set.contract_profile must be a safe profile ID"
        )
    source, _, source_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        ACTION_SET_CONTRACT_SOURCE,
        name="code-owned action-set contract registry",
    )
    registry = _literal_assignment(source, ACTION_SET_CONTRACTS_NAME)
    policies = _literal_assignment(source, ACTION_SET_PROFILE_POLICIES_NAME)
    actor_obs_contract = _literal_assignment(
        source, ACTION_SET_ACTOR_OBS_CONTRACT_NAME
    )
    actor_obs_width = _literal_assignment(
        source, ACTION_SET_ACTOR_OBS_WIDTH_NAME
    )
    if type(registry) is not dict or type(policies) is not dict:
        raise LaunchRefused("action-set contract registry/policies must be dicts")
    if type(actor_obs_contract) is not str or not actor_obs_contract:
        raise LaunchRefused("action-set actor observation contract is invalid")
    actor_obs_width = _plain_int(
        actor_obs_width,
        name="action-set actor observation width",
        minimum=1,
    )
    if profile_id not in registry:
        raise LaunchRefused(
            f"unregistered action-set contract profile: {profile_id}"
        )
    row = _exact_dict(
        registry[profile_id],
        (
            "profile_id",
            "expected_n",
            "scope",
            "mobility_mode",
            "ordered_action_ids",
            "ordered_action_uids",
            "order_uid_digest_sha256",
            "manifest_path",
            "manifest_sha256",
            "experiment_name",
        ),
        name=f"action-set contract {profile_id}",
    )
    if row["profile_id"] != profile_id:
        raise LaunchRefused("action-set contract profile_id mismatch")
    expected_n = _plain_int(
        row["expected_n"], name="action-set contract expected_n", minimum=1
    )
    if actor_obs_contract.endswith("_v2") and expected_n != 1:
        raise LaunchRefused(
            "fixed-194 ActionBall v2 is N=1-only; formal multi-action "
            "launch remains blocked until the final fixed-width teacher-"
            "trajectory/ball/task ABI and N2/N3 shared-policy validation exist; "
            "no motion ID or synthetic intent code is required"
        )
    order = row["ordered_action_ids"]
    uids = row["ordered_action_uids"]
    if (
        type(order) is not list
        or len(order) != expected_n
        or any(type(item) is not str or not item for item in order)
        or len(order) != len(set(order))
    ):
        raise LaunchRefused(
            "action-set contract ordered_action_ids must be exact unique N"
        )
    if (
        type(uids) is not list
        or len(uids) != expected_n
        or any(
            type(item) is not int or isinstance(item, bool) or item < 0
            for item in uids
        )
        or len(uids) != len(set(uids))
    ):
        raise LaunchRefused(
            "action-set contract ordered_action_uids must be exact unique N"
        )
    if row["scope"] not in ("upper", "full"):
        raise LaunchRefused("action-set contract scope must be upper/full")
    if row["mobility_mode"] not in ("no_move", "move"):
        raise LaunchRefused(
            "action-set contract mobility_mode must be no_move/move"
        )
    digest = _sha256(
        row["order_uid_digest_sha256"],
        name="action-set contract order_uid_digest_sha256",
    )
    if digest != _order_uid_digest(order, uids):
        raise LaunchRefused("action-set contract order/UID digest mismatch")
    _manifest_contract_path, manifest_path = _repo_file(
        checkout,
        row["manifest_path"],
        name="action-set contract manifest_path",
    )
    manifest_sha = _sha256(
        row["manifest_sha256"],
        name="action-set contract manifest_sha256",
    )
    experiment_name = row["experiment_name"]
    if (
        type(experiment_name) is not str
        or RUN_COMPONENT_RE.fullmatch(experiment_name) is None
    ):
        raise LaunchRefused("action-set contract experiment_name is invalid")

    if profile_id in policies:
        policy = _exact_dict(
            policies[profile_id],
            (
                "expected_n",
                "scope",
                "mobility_mode",
                "required_action_ids",
                "retired_action_ids",
            ),
            name=f"action-set profile policy {profile_id}",
        )
        required = policy["required_action_ids"]
        retired = policy["retired_action_ids"]
        if (
            policy["expected_n"] != expected_n
            or policy["scope"] != row["scope"]
            or policy["mobility_mode"] != row["mobility_mode"]
            or type(required) is not list
            or any(type(item) is not str or not item for item in required)
            or len(required) != len(set(required))
            or (required and required != order)
            or type(retired) is not list
            or any(type(item) is not str or not item for item in retired)
            or len(retired) != len(set(retired))
        ):
            raise LaunchRefused(
                "action-set contract violates its code-owned profile policy"
            )
        stale = sorted(set(order).intersection(retired))
        if stale:
            raise LaunchRefused(
                f"retired actions are forbidden by profile policy: {stale}"
            )

    identity = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.action_set_contract",
        "profile_id": profile_id,
        "expected_n": expected_n,
        "scope": row["scope"],
        "mobility_mode": row["mobility_mode"],
        "ordered_action_ids": list(order),
        "ordered_action_uids": list(uids),
        "order_uid_digest_sha256": digest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "experiment_name": experiment_name,
        "actor_obs_contract": actor_obs_contract,
        "actor_obs_width": actor_obs_width,
        "namespace_identity": f"n{expected_n}-{digest[:12]}",
    }
    identity["contract_sha256"] = canonical_sha256(identity)
    return identity, source_sha


def _validate_manifest(
    document: Any,
    *,
    checkout: Path,
    source_commit: str,
    order: tuple[str, ...],
    ordered_action_uids: tuple[int, ...],
    scope: str,
    mobility_mode: str,
    prototype_relative: str,
    prototype_sha256: str,
) -> tuple[list[dict[str, Any]], str, str]:
    if type(document) is not dict:
        raise LaunchRefused("manifest must be a JSON object")
    top_level_keys = (
        "schema_version",
        "manifest_id",
        "mobility_mode",
        "action_order",
        "prototype",
        "solver_profile_sha256",
        "physics_profile_sha256",
        "landing_aim",
        "actions",
        "curriculum",
        "holdout",
        "notes",
    )
    has_counter_rally = "counter_rally_objective" in document
    _exact_dict(
        document,
        (
            *top_level_keys,
            *(("counter_rally_objective",) if has_counter_rally else ()),
        ),
        name="manifest",
    )
    if document.get("schema_version") != 3:
        raise LaunchRefused("manifest schema_version must be 3")
    if (
        type(document.get("manifest_id")) is not str
        or not document["manifest_id"]
        or document["manifest_id"].strip() != document["manifest_id"]
    ):
        raise LaunchRefused("manifest.manifest_id must be non-empty and trimmed")
    if type(document.get("notes")) is not str:
        raise LaunchRefused("manifest.notes must be a string")
    if type(document.get("landing_aim")) is not dict:
        raise LaunchRefused("manifest.landing_aim must be an object")
    if has_counter_rally and len(order) != 1:
        raise LaunchRefused(
            "manifest.counter_rally_objective is restricted to exact N=1"
        )
    if has_counter_rally and type(document["counter_rally_objective"]) is not dict:
        raise LaunchRefused(
            "manifest.counter_rally_objective must be an object"
        )
    if document.get("mobility_mode") != mobility_mode:
        raise LaunchRefused(
            "manifest mobility_mode differs from the action-set contract"
        )
    if document.get("action_order") != list(order):
        raise LaunchRefused(
            "manifest.action_order differs from the exact launch order"
        )
    prototype = _exact_dict(
        document.get("prototype"),
        ("path", "sha256", "scope"),
        name="manifest.prototype",
    )
    if (
        prototype["path"] != prototype_relative
        or prototype["sha256"] != prototype_sha256
        or prototype["scope"] != scope
    ):
        raise LaunchRefused(
            "manifest prototype path/SHA/scope differs from the pinned prototype"
        )
    actions = document.get("actions")
    if type(actions) is not list or len(actions) != len(order):
        raise LaunchRefused(
            "manifest.actions must contain exact contracted N ordered rows"
        )
    bindings: list[dict[str, Any]] = []
    seen_uids: set[int] = set()
    for index, (action, action_id) in enumerate(zip(actions, order)):
        if type(action) is not dict:
            raise LaunchRefused(f"manifest.actions[{index}] must be an object")
        if action.get("action_id") != action_id:
            raise LaunchRefused(
                f"manifest.actions[{index}].action_id is out of order"
            )
        uid = _plain_int(
            action.get("action_uid"),
            name=f"manifest.actions[{index}].action_uid",
            minimum=1,
            maximum=(1 << 53) - 1,
        )
        if uid != ordered_action_uids[index]:
            raise LaunchRefused(
                f"manifest.actions[{index}].action_uid differs from contract"
            )
        if uid in seen_uids:
            raise LaunchRefused("manifest action_uid values must be unique")
        seen_uids.add(uid)
        family = action.get("family")
        if family not in ("forehand", "backhand"):
            raise LaunchRefused(
                f"manifest.actions[{index}].family is invalid"
            )
        motion_sha = _sha256(
            action.get("motion_sha256"),
            name=f"manifest.actions[{index}].motion_sha256",
        )
        motion_path, motion_relative, actual_motion_sha, _ = _verify_repo_blob(
            checkout,
            source_commit,
            action.get("motion_path"),
            name=f"manifest.actions[{index}].motion_path",
        )
        if actual_motion_sha != motion_sha:
            raise LaunchRefused(
                f"manifest motion[{index}] SHA mismatch: "
                f"expected={motion_sha}, actual={actual_motion_sha}"
            )
        bindings.append(
            {
                "motion_id": action_id,
                "action_uid": uid,
                "family": family,
                "motion_path": motion_relative,
                "motion_sha256": motion_sha,
            }
        )
    solver = _sha256(
        document.get("solver_profile_sha256"),
        name="manifest.solver_profile_sha256",
    )
    physics = _sha256(
        document.get("physics_profile_sha256"),
        name="manifest.physics_profile_sha256",
    )
    curriculum = _exact_dict(
        document.get("curriculum"),
        (
            "min_proposals",
            "min_safe_closed",
            "target_failure_rate",
            "failure_band_half_width",
            "min_solver_admit_rate",
            "min_install_rate",
            "min_start_rate",
            "min_close_rate",
            "max_other_unsafe_rate",
            "confidence_z",
            "max_center_failures",
        ),
        name="manifest.curriculum",
    )
    min_proposals = _plain_int(
        curriculum["min_proposals"],
        name="manifest.curriculum.min_proposals",
        minimum=1,
    )
    min_safe_closed = _plain_int(
        curriculum["min_safe_closed"],
        name="manifest.curriculum.min_safe_closed",
        minimum=1,
    )
    holdout = _exact_dict(
        document.get("holdout"),
        ("seed", "samples_per_action", "split_id"),
        name="manifest.holdout",
    )
    _plain_int(
        holdout["seed"], name="manifest.holdout.seed", minimum=0
    )
    holdout_samples = _plain_int(
        holdout["samples_per_action"],
        name="manifest.holdout.samples_per_action",
        minimum=1,
    )
    if (
        type(holdout["split_id"]) is not str
        or not holdout["split_id"]
        or holdout["split_id"].strip() != holdout["split_id"]
    ):
        raise LaunchRefused(
            "manifest.holdout.split_id must be non-empty and trimmed"
        )
    required_holdout = max(768, min_proposals, min_safe_closed)
    if holdout_samples < required_holdout:
        raise LaunchRefused(
            "manifest holdout is not a formal per-action window: "
            f"need at least {required_holdout}, got {holdout_samples}"
        )
    return bindings, solver, physics


def _validate_prototype(
    document: Any,
    *,
    order: tuple[str, ...],
    bindings: list[dict[str, Any]],
    scope: str,
) -> None:
    if type(document) is not dict or document.get("schema_version") != 2:
        raise LaunchRefused(
            "prototype must use schema v2 selected-rubber face-centre semantics"
        )
    scopes = document.get("scopes")
    if type(scopes) is not dict:
        raise LaunchRefused("prototype.scopes must be an object")
    rows = scopes.get(scope)
    if type(rows) is not list or len(rows) != len(order):
        raise LaunchRefused(
            "prototype contracted scope must contain exact N rows"
        )
    for index, (row, action_id, binding) in enumerate(
        zip(rows, order, bindings)
    ):
        if type(row) is not dict:
            raise LaunchRefused(
                f"prototype {scope} row[{index}] must be an object"
            )
        if (
            row.get("motion_id") != action_id
            or row.get("scope") != scope
            or row.get("clip_index") != index
            or row.get("family") != binding["family"]
            or row.get("npz_sha256") != binding["motion_sha256"]
        ):
            raise LaunchRefused(
                f"prototype {scope} row[{index}] identity/order/motion SHA drifted"
            )
    declared = _sha256(
        document.get("derived_sha256"), name="prototype.derived_sha256"
    )
    actual = canonical_sha256(scopes)
    if declared != actual:
        raise LaunchRefused(
            "prototype.derived_sha256 does not match prototype.scopes"
        )


def _validate_registry(
    document: Any,
    *,
    order: tuple[str, ...],
    bindings: list[dict[str, Any]],
    scope: str,
    expected_ready_sha256: str,
    expected_ready_fk_sha256: str,
) -> None:
    if type(document) is not dict:
        raise LaunchRefused("canonical registry must be an object")
    if document.get("scope") != scope:
        raise LaunchRefused("canonical registry scope differs from contract")
    if document.get("canonical_ready_sha256") != expected_ready_sha256:
        raise LaunchRefused("canonical registry ready SHA pin drifted")
    if (
        document.get("canonical_ready_fk_sha256")
        != expected_ready_fk_sha256
    ):
        raise LaunchRefused("canonical registry ready-FK SHA pin drifted")
    entries = document.get("entries")
    if type(entries) is not list or len(entries) != len(order):
        raise LaunchRefused(
            "canonical registry must contain exact contracted N entries"
        )
    for index, (entry, action_id, binding) in enumerate(
        zip(entries, order, bindings)
    ):
        if type(entry) is not dict:
            raise LaunchRefused(
                f"canonical registry entry[{index}] must be an object"
            )
        motion_path = entry.get("motion_path", entry.get("npz_path"))
        motion_sha = entry.get("motion_sha256", entry.get("npz_sha256"))
        if (
            entry.get("motion_id") != action_id
            or entry.get("scope") != scope
            or motion_path != binding["motion_path"]
            or motion_sha != binding["motion_sha256"]
            or entry.get("training_authorized") is not True
        ):
            raise LaunchRefused(
                f"canonical registry entry[{index}] is not the exact "
                "training-authorized contracted action binding"
            )


def _validate_admission_receipt(
    document: Any,
    *,
    order: tuple[str, ...],
    bindings: list[dict[str, Any]],
    scope: str,
    mobility_mode: str,
    registry_sha256: str,
    promotion_sha256: str,
) -> str:
    row = _exact_dict(
        document,
        (
            "schema_version",
            "kind",
            "authorization_purpose",
            "scope",
            "mobility_mode",
            "ordered_action_ids",
            "registry_sha256",
            "promotion_certificate_sha256",
            "motion_rows",
            "canonical_sha256",
        ),
        name="motion admission launch receipt",
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != "action_ball_static_motion_admission_launch"
        or row["authorization_purpose"] != "training"
        or row["scope"] != scope
        or row["mobility_mode"] != mobility_mode
        or row["ordered_action_ids"] != list(order)
        or row["registry_sha256"] != registry_sha256
        or row["promotion_certificate_sha256"] != promotion_sha256
    ):
        raise LaunchRefused(
            "motion admission receipt identity/purpose/order pins drifted"
        )
    expected_rows = [
        {
            "motion_id": binding["motion_id"],
            "action_uid": binding["action_uid"],
            "motion_path": binding["motion_path"],
            "motion_sha256": binding["motion_sha256"],
        }
        for binding in bindings
    ]
    if row["motion_rows"] != expected_rows:
        raise LaunchRefused(
            "motion admission receipt rows differ from the manifest"
        )
    declared = _sha256(
        row["canonical_sha256"],
        name="motion admission receipt canonical_sha256",
    )
    unsigned = dict(row)
    del unsigned["canonical_sha256"]
    actual = canonical_sha256(unsigned)
    if actual != declared:
        raise LaunchRefused(
            "motion admission receipt canonical SHA-256 mismatch"
        )
    return declared


def _validate_profile_order(
    value: Any,
    *,
    bindings: list[dict[str, Any]],
    mobility_mode: str,
    name: str,
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(bindings):
        raise LaunchRefused(f"{name} must contain exact contracted N rows")
    result: list[dict[str, Any]] = []
    for index, (profile, binding) in enumerate(zip(value, bindings)):
        row = _exact_dict(
            profile,
            ("action_uid", "profile_sha256", "mobility"),
            name=f"{name}[{index}]",
        )
        _plain_int(
            row["action_uid"],
            name=f"{name}[{index}].action_uid",
            minimum=1,
        )
        _sha256(
            row["profile_sha256"],
            name=f"{name}[{index}].profile_sha256",
        )
        if (
            row["action_uid"] != binding["action_uid"]
            or row["mobility"] != mobility_mode
        ):
            raise LaunchRefused(f"{name}[{index}] identity drifted")
        result.append(dict(row))
    if len({row["action_uid"] for row in result}) != len(result):
        raise LaunchRefused(f"{name} action_uid values must be unique")
    return result


def _validate_evaluator_receipt(
    document: Any,
    *,
    checkout: Path,
    source_commit: str,
    bindings: list[dict[str, Any]],
    mobility_mode: str,
    solver_sha256: str,
    policy_sha256: str,
) -> tuple[str, dict[str, Any]]:
    keys = (
        "schema_version",
        "kind",
        "authority_contract_sha256",
        "curriculum_contract_sha256",
        "profile_order",
        "arm_catalog_sha256",
        "scheduler_contract_sha256",
        "sampler_sha256",
        "solver_sha256",
        "policy_contract_sha256",
        "attempt_source_contract_sha256",
        "attempt_source_path",
        "attempt_source_sha256",
        "window_contract",
    )
    row = _exact_dict(
        document, keys, name="frozen evaluator launch receipt"
    )
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != 4
        or row["kind"] != "action_ball_frozen_evaluator_v4_launch"
    ):
        raise LaunchRefused(
            "frozen evaluator receipt schema/kind is invalid"
        )
    for field in (
        "authority_contract_sha256",
        "curriculum_contract_sha256",
        "arm_catalog_sha256",
        "scheduler_contract_sha256",
        "sampler_sha256",
        "attempt_source_contract_sha256",
    ):
        _sha256(row[field], name=f"frozen evaluator receipt {field}")
    if (
        row["solver_sha256"] != solver_sha256
        or row["policy_contract_sha256"] != policy_sha256
    ):
        raise LaunchRefused(
            "frozen evaluator solver/policy pin differs from launch identity"
        )
    profiles = _validate_profile_order(
        row["profile_order"],
        bindings=bindings,
        mobility_mode=mobility_mode,
        name="frozen evaluator profile_order",
    )
    if row["window_contract"] != V4_EVALUATOR_WINDOW_CONTRACT:
        raise LaunchRefused(
            "frozen evaluator window_contract is not the exact V4 "
            "100/320/960 non-optional contract"
        )
    if row["attempt_source_path"] != EVALUATION_INBOX_SOURCE:
        raise LaunchRefused(
            "formal V4 evaluator attempt source must be the frozen evaluation "
            "inbox implementation"
        )
    expected_source_sha = _sha256(
        row["attempt_source_sha256"],
        name="frozen evaluator receipt attempt_source_sha256",
    )
    source, _, actual_source_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        row["attempt_source_path"],
        name="frozen evaluator receipt attempt_source_path",
    )
    if actual_source_sha != expected_source_sha:
        raise LaunchRefused(
            "frozen evaluator attempt-source bytes drifted"
        )
    identity = {
        "curriculum_contract_sha256": row["curriculum_contract_sha256"],
        "profile_order": profiles,
        "arm_catalog_sha256": row["arm_catalog_sha256"],
        "scheduler_contract_sha256": row["scheduler_contract_sha256"],
        "sampler_sha256": row["sampler_sha256"],
        "solver_sha256": row["solver_sha256"],
        "policy_contract_sha256": row["policy_contract_sha256"],
        "attempt_source_contract_sha256": row[
            "attempt_source_contract_sha256"
        ],
        "attempt_source_path": row["attempt_source_path"],
        "attempt_source_sha256": row["attempt_source_sha256"],
        "window_contract": row["window_contract"],
    }
    return canonical_sha256(row), identity


def _validate_sidecar_launch_receipt(
    document: Any,
    *,
    checkout: Path,
    source_commit: str,
) -> tuple[str, str, str, dict[str, Any]]:
    row = _exact_dict(
        document,
        ("schema_version", "kind", "content", "content_sha256"),
        name="frozen evaluation sidecar launch receipt",
    )
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != 1
        or row["kind"] != "action_ball_frozen_eval_sidecar_launch"
    ):
        raise LaunchRefused(
            "frozen evaluation sidecar launch schema/kind is invalid"
        )
    content = _exact_dict(
        row["content"],
        (
            "protocol_contract_sha256",
            "sidecar_code_sha256",
            "backend_contract_sha256",
            "policy_evaluation_contract_sha256",
            "resolved_recipe_contract_sha256",
            "runtime_identity_contract_sha256",
            "window_contract",
            "heartbeat_contract",
        ),
        name="frozen evaluation sidecar launch content",
    )
    for field in (
        "protocol_contract_sha256",
        "sidecar_code_sha256",
        "backend_contract_sha256",
        "policy_evaluation_contract_sha256",
        "resolved_recipe_contract_sha256",
        "runtime_identity_contract_sha256",
    ):
        _sha256(content[field], name=f"sidecar launch content {field}")
    launch_content_sha = _sha256(
        row["content_sha256"], name="sidecar launch content_sha256"
    )
    if canonical_sha256(content) != launch_content_sha:
        raise LaunchRefused("sidecar launch content SHA-256 mismatch")
    if content["window_contract"] != SIDECAR_WINDOW_CONTRACT:
        raise LaunchRefused(
            "sidecar window_contract is not the exact append-only V4 window "
            "allocation contract"
        )
    heartbeat_contract = content["heartbeat_contract"]
    if (
        type(heartbeat_contract) is not dict
        or set(heartbeat_contract) != set(SIDECAR_HEARTBEAT_CONTRACT)
        or type(heartbeat_contract["schema_version"]) is not int
        or any(
            type(heartbeat_contract[field]) is not float
            for field in (
                "heartbeat_interval_seconds",
                "heartbeat_stale_after_seconds",
                "request_deadline_seconds",
            )
        )
        or heartbeat_contract != SIDECAR_HEARTBEAT_CONTRACT
    ):
        raise LaunchRefused(
            "sidecar heartbeat_contract is not the exact formal "
            "5s/120s/7200s liveness contract"
        )
    _sidecar_path, _relative, actual_code_sha, _mode = _verify_repo_blob(
        checkout,
        source_commit,
        SIDECAR_CODE_SOURCE,
        name="frozen evaluation sidecar code",
    )
    if content["sidecar_code_sha256"] != actual_code_sha:
        raise LaunchRefused(
            "sidecar launch receipt does not bind the exact committed sidecar "
            "code bytes"
        )
    return (
        canonical_sha256(row),
        launch_content_sha,
        actual_code_sha,
        dict(content),
    )


def _validate_drain_reset_launch_receipt(
    document: Any,
    *,
    checkout: Path,
    source_commit: str,
    bindings: list[dict[str, Any]],
    mobility_mode: str,
    evaluator_identity: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    keys = (
        "schema_version",
        "kind",
        "authority_contract_sha256",
        "curriculum_contract_sha256",
        "profile_order",
        "arm_catalog_sha256",
        "scheduler_contract_sha256",
        "sampler_sha256",
        "solver_sha256",
        "policy_contract_sha256",
        "runtime_source_contract_sha256",
        "runtime_source_path",
        "runtime_source_sha256",
        "broker_contract_sha256",
        "attempt_pool_contract_sha256",
        "task_receipt_pool_contract_sha256",
        "env_reset_contract_sha256",
    )
    row = _exact_dict(
        document, keys, name="drain/reset launch receipt"
    )
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != 1
        or row["kind"] != "action_ball_drain_reset_launch"
    ):
        raise LaunchRefused("drain/reset launch receipt schema/kind is invalid")
    for field in (
        "authority_contract_sha256",
        "curriculum_contract_sha256",
        "arm_catalog_sha256",
        "scheduler_contract_sha256",
        "sampler_sha256",
        "solver_sha256",
        "policy_contract_sha256",
        "runtime_source_contract_sha256",
        "runtime_source_sha256",
        "broker_contract_sha256",
        "attempt_pool_contract_sha256",
        "task_receipt_pool_contract_sha256",
        "env_reset_contract_sha256",
    ):
        _sha256(row[field], name=f"drain/reset launch receipt {field}")
    profiles = _validate_profile_order(
        row["profile_order"],
        bindings=bindings,
        mobility_mode=mobility_mode,
        name="drain/reset profile_order",
    )
    shared_identity = {
        "curriculum_contract_sha256": row["curriculum_contract_sha256"],
        "profile_order": profiles,
        "arm_catalog_sha256": row["arm_catalog_sha256"],
        "scheduler_contract_sha256": row["scheduler_contract_sha256"],
        "sampler_sha256": row["sampler_sha256"],
        "solver_sha256": row["solver_sha256"],
        "policy_contract_sha256": row["policy_contract_sha256"],
    }
    expected_shared = {
        key: evaluator_identity[key] for key in shared_identity
    }
    if shared_identity != expected_shared:
        raise LaunchRefused(
            "drain/reset launch scientific identity differs from the exact V4 "
            "evaluator identity"
        )
    if row["runtime_source_path"] != HOPE_COMMANDS_SOURCE:
        raise LaunchRefused(
            "drain/reset runtime source must be the formal hope_commands "
            "coordinator"
        )
    _runtime_path, _relative, actual_runtime_sha, _mode = _verify_repo_blob(
        checkout,
        source_commit,
        row["runtime_source_path"],
        name="drain/reset runtime source",
    )
    if row["runtime_source_sha256"] != actual_runtime_sha:
        raise LaunchRefused(
            "drain/reset launch runtime source bytes drifted"
        )
    operational_identity = {
        "authority_contract_sha256": row["authority_contract_sha256"],
        "runtime_source_contract_sha256": row[
            "runtime_source_contract_sha256"
        ],
        "runtime_source_path": row["runtime_source_path"],
        "runtime_source_sha256": row["runtime_source_sha256"],
        "broker_contract_sha256": row["broker_contract_sha256"],
        "attempt_pool_contract_sha256": row[
            "attempt_pool_contract_sha256"
        ],
        "task_receipt_pool_contract_sha256": row[
            "task_receipt_pool_contract_sha256"
        ],
        "env_reset_contract_sha256": row["env_reset_contract_sha256"],
    }
    return canonical_sha256(row), operational_identity


def _validate_python_import_roots(value: Any, *, name: str) -> list[str]:
    if type(value) is not list or not value:
        raise LaunchRefused(f"{name} must be a non-empty ordered list")
    roots: list[str] = []
    for index, item in enumerate(value):
        if (
            type(item) is not str
            or not item
            or "\x00" in item
            or "\n" in item
            or not Path(item).is_absolute()
            or os.path.normpath(item) != item
        ):
            raise LaunchRefused(
                f"{name}[{index}] must be one normalized absolute path"
            )
        roots.append(item)
    if len(roots) != len(set(roots)):
        raise LaunchRefused(f"{name} must not contain duplicates")
    return roots


def _probe_python_runtime(path: Path) -> dict[str, Any]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _SANITIZED_ENV_ALLOWLIST
    }
    environment["PATH"] = os.defpath
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [str(path), "-I", "-B", "-S", "-c", _PYTHON_PROBE],
            check=False,
            capture_output=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchRefused(
            f"isolated Isaac Python identity probe failed: {exc}"
        ) from exc
    if result.returncode != 0:
        raise LaunchRefused(
            "isolated Isaac Python identity probe returned nonzero: "
            + result.stderr[-4000:].decode("utf-8", errors="replace")
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise LaunchRefused(
            "isolated Isaac Python identity probe must emit exactly one JSON line"
        )
    row = _exact_dict(
        _load_strict_json_bytes(lines[0], name="Isaac Python identity"),
        ("version", "cache_tag", "import_roots", "isolated", "safe_path"),
        name="Isaac Python identity",
    )
    if row["isolated"] is not True or row["safe_path"] is not True:
        raise LaunchRefused(
            "Isaac Python must support -I isolated/safe-path execution"
        )
    if (
        type(row["version"]) is not str
        or not row["version"]
        or type(row["cache_tag"]) is not str
        or not row["cache_tag"]
    ):
        raise LaunchRefused("Isaac Python version/cache tag is invalid")
    row["import_roots"] = _validate_python_import_roots(
        row["import_roots"], name="Isaac Python observed import_roots"
    )
    return row


def _lstat_identity(path: Path, *, name: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be lstat'd: {path}: {exc}") from exc
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def _ancestor_identities(path: Path, *, name: str) -> list[dict[str, Any]]:
    current = Path(path.anchor)
    root_identity = _lstat_identity(current, name=f"{name} ancestor")
    if not stat.S_ISDIR(root_identity["mode"]):
        raise LaunchRefused(f"{name} filesystem root is not a directory")
    ancestors: list[dict[str, Any]] = [root_identity]
    for component in path.parent.parts[1:]:
        current = current / component
        identity = _lstat_identity(current, name=f"{name} ancestor")
        if not stat.S_ISDIR(identity["mode"]):
            raise LaunchRefused(
                f"{name} ancestor must be a real non-symlink directory: "
                f"{current}"
            )
        ancestors.append(identity)
    return ancestors


def _resolve_python_executable(
    value: Any, *, name: str
) -> tuple[Path, Path, dict[str, Any]]:
    if type(value) is not str or not value:
        raise LaunchRefused(f"{name} must be a non-empty absolute path")
    lexical = Path(value)
    if not lexical.is_absolute() or os.path.normpath(value) != value:
        raise LaunchRefused(f"{name} must be an absolute normalized path")
    current = lexical
    seen: set[str] = set()
    chain: list[dict[str, Any]] = []
    for depth in range(17):
        current_text = str(current)
        if current_text in seen:
            raise LaunchRefused(f"{name} contains a symlink loop")
        seen.add(current_text)
        ancestors = _ancestor_identities(current, name=name)
        identity = _lstat_identity(current, name=name)
        mode = identity["mode"]
        if stat.S_ISLNK(mode):
            if depth == 16:
                raise LaunchRefused(f"{name} symlink chain exceeds 16 hops")
            try:
                link_text = os.readlink(current)
            except OSError as exc:
                raise LaunchRefused(
                    f"{name} symlink text cannot be read: {current}: {exc}"
                ) from exc
            target = Path(link_text)
            if (
                not link_text
                or "\x00" in link_text
                or any(part == ".." for part in target.parts)
                or os.path.normpath(link_text) != link_text
            ):
                raise LaunchRefused(
                    f"{name} symlink target must be normalized and contain no "
                    f"parent traversal: {current} -> {link_text!r}"
                )
            next_path = target if target.is_absolute() else current.parent / target
            next_path = Path(os.path.normpath(str(next_path)))
            if not next_path.is_absolute():
                raise LaunchRefused(f"{name} symlink target escaped absolute paths")
            chain.append(
                {
                    "kind": "symlink",
                    "lstat": identity,
                    "ancestors": ancestors,
                    "link_text": link_text,
                    "resolved_target": str(next_path),
                }
            )
            current = next_path
            continue
        if not stat.S_ISREG(mode):
            raise LaunchRefused(
                f"{name} final target must be one regular executable file"
            )
        if not os.access(current, os.X_OK):
            raise LaunchRefused(f"{name} final target must be executable")
        actual_sha = sha256_file(current)
        after = _lstat_identity(current, name=name)
        if after != identity:
            raise LaunchRefused(f"{name} final target changed while hashing")
        chain.append(
            {
                "kind": "regular",
                "lstat": identity,
                "ancestors": ancestors,
                "sha256": actual_sha,
            }
        )
        return lexical, current, {
            "requested_path": str(lexical),
            "resolved_path": str(current),
            "resolution_chain": chain,
            "final_sha256": actual_sha,
        }
    raise LaunchRefused(f"{name} symlink resolution failed")


def _validate_python_runtime(value: Any) -> tuple[Path, dict[str, Any]]:
    row = _exact_dict(
        value,
        ("path", "sha256", "version", "cache_tag", "import_roots"),
        name="spec.train.isaac_python",
    )
    path, resolved_path, resolution = _resolve_python_executable(
        row["path"], name="spec.train.isaac_python.path"
    )
    expected_sha = _sha256(
        row["sha256"], name="spec.train.isaac_python.sha256"
    )
    actual_sha = resolution["final_sha256"]
    if actual_sha != expected_sha:
        raise LaunchRefused(
            "spec.train.isaac_python byte SHA-256 mismatch: "
            f"expected={expected_sha}, actual={actual_sha}"
        )
    expected_roots = _validate_python_import_roots(
        row["import_roots"],
        name="spec.train.isaac_python.import_roots",
    )
    expected_identity = {
        "version": row["version"],
        "cache_tag": row["cache_tag"],
        "import_roots": expected_roots,
        "isolated": True,
        "safe_path": True,
    }
    if (
        type(row["version"]) is not str
        or not row["version"]
        or type(row["cache_tag"]) is not str
        or not row["cache_tag"]
    ):
        raise LaunchRefused(
            "Isaac Python version/cache identity is invalid"
        )
    observed_identity = _probe_python_runtime(resolved_path)
    if observed_identity != expected_identity:
        raise LaunchRefused(
            "live isolated Isaac Python identity differs from "
            "spec.train.isaac_python"
        )
    return path, {
        "path": str(path),
        "resolved_path": str(resolved_path),
        "resolution": resolution,
        "sha256": actual_sha,
        **expected_identity,
    }


def _validate_runtime_inventory_receipt(
    pin: Any,
    *,
    inventory_script: Path,
    isaac_python: Path,
    expected_version: str,
    expected_cache_tag: str,
    nosite_bootstrap_script: Path,
    nosite_bootstrap_sha256: str,
) -> dict[str, Any]:
    """Recompute the committed runtime inventory and bind its exact receipt.

    The inventory implementation is part of ``RUNTIME_CODE_SOURCES`` and was
    already proven to be the exact commit-tree blob before this function is
    called.  Running that blob with the requested Isaac Python makes both
    ``plan`` and the locked re-plan re-open every Python/IsaacLab identity in
    the receipt instead of trusting a caller-provided summary.
    """

    receipt_path, receipt_file_sha = _verify_external_pin(
        pin, name="spec.train.runtime_inventory"
    )
    receipt = _exact_dict(
        load_strict_json(receipt_path, name="runtime inventory receipt"),
        ("schema_version", "kind", "content", "content_sha256"),
        name="runtime inventory receipt",
    )
    if (
        receipt["schema_version"] != 2
        or receipt["kind"] != "action_ball_runtime_inventory_v2"
    ):
        raise LaunchRefused("runtime inventory receipt schema/kind is invalid")
    if type(receipt["content"]) is not dict:
        raise LaunchRefused("runtime inventory receipt content must be an object")
    content_sha = _sha256(
        receipt["content_sha256"],
        name="runtime inventory receipt content_sha256",
    )
    if canonical_sha256(receipt["content"]) != content_sha:
        raise LaunchRefused(
            "runtime inventory receipt content_sha256 does not match content"
        )
    python_content = receipt["content"].get("python")
    probe_content = (
        python_content.get("probe")
        if type(python_content) is dict
        else None
    )
    if (
        type(python_content) is not dict
        or python_content.get("requested_path") != str(isaac_python)
        or type(probe_content) is not dict
        or probe_content.get("version") != expected_version
        or probe_content.get("cache_tag") != expected_cache_tag
    ):
        raise LaunchRefused(
            "runtime inventory receipt does not bind the requested Isaac Python"
        )

    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _SANITIZED_ENV_ALLOWLIST
    }
    environment["PATH"] = os.defpath
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        module_name = "_action_ball_launcher_nosite_bootstrap"
        nosite = _load_source_module_without_bytecode(
            nosite_bootstrap_script,
            name=module_name,
            purpose="no-site bootstrap source",
        )
        outer = python_content["probe"]["no_site_execution"]["outer"]
        import_roots = outer["import_roots"]
        command_identity = nosite.build_exact_nosite_argv(
            python=isaac_python,
            bootstrap=nosite_bootstrap_script,
            bootstrap_sha256=nosite_bootstrap_sha256,
            entrypoint=inventory_script,
            entrypoint_sha256=sha256_file(inventory_script),
            import_roots=import_roots,
            entrypoint_argv=[
                "verify",
                "--receipt",
                str(receipt_path),
            ],
        )
        nosite.validate_exact_nosite_argv(
            command_identity.argv,
            expected_python=isaac_python,
            expected_bootstrap=command_identity.contract["bootstrap"],
            expected_entrypoint=command_identity.contract["entrypoint"],
            expected_import_roots=import_roots,
            expected_entrypoint_argv=[
                "verify",
                "--receipt",
                str(receipt_path),
            ],
            expected_contract_sha256=command_identity.contract_sha256,
        )
        command = list(command_identity.argv)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise LaunchRefused(
            "runtime inventory receipt lacks exact no-site import roots"
        ) from exc
    except Exception as exc:
        raise LaunchRefused(
            f"runtime inventory no-site command is invalid: {exc}"
        ) from exc
    try:
        completed = subprocess.run(
            command,
            cwd=inventory_script.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchRefused(
            "committed runtime inventory verifier could not complete"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise LaunchRefused(
            "live Python/IsaacLab runtime differs from the frozen inventory: "
            + detail
        )
    output_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if len(output_lines) != 1:
        raise LaunchRefused(
            "runtime inventory verifier did not emit exactly one result"
        )
    result = _exact_dict(
        _load_strict_json_bytes(
            output_lines[0], name="runtime inventory verifier result"
        ),
        (
            "ok",
            "kind",
            "content_sha256",
            "receipt_path",
            "receipt_sha256",
        ),
        name="runtime inventory verifier result",
    )
    expected_result = {
        "ok": True,
        "kind": "action_ball_runtime_inventory_v2",
        "content_sha256": content_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_file_sha,
    }
    if result != expected_result:
        raise LaunchRefused(
            "runtime inventory verifier result is not bound to the exact receipt"
        )
    return {
        "path": str(receipt_path),
        "file_sha256": receipt_file_sha,
        "content_sha256": content_sha,
        "kind": receipt["kind"],
        "import_roots": list(import_roots),
        "nosite_verification_contract_sha256": (
            command_identity.contract_sha256
        ),
    }


def _verify_ed25519_signature(
    *,
    public_key: bytes,
    payload: dict[str, Any],
    signature_hex: Any,
    name: str,
) -> None:
    if (
        type(signature_hex) is not str
        or ED25519_SIGNATURE_RE.fullmatch(signature_hex) is None
    ):
        raise LaunchRefused(
            f"{name}.signature_ed25519_hex must be 128 lowercase hex characters"
        )
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise LaunchRefused(
            "cryptography Ed25519 verifier is unavailable; signed authority "
            "cannot be authenticated"
        ) from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(signature_hex), _canonical_bytes(payload)
        )
    except (InvalidSignature, ValueError) as exc:
        raise LaunchRefused(f"{name} Ed25519 signature is invalid") from exc


def _validate_stage_evaluator_authority(
    document: Any,
    *,
    checkout: Path,
    source_commit: str,
) -> tuple[str, bytes]:
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
        name="frozen stage evaluator authority",
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != "action_ball_frozen_stage_evaluator_authority"
        or type(row["evaluator_id"]) is not str
        or not row["evaluator_id"]
    ):
        raise LaunchRefused("frozen stage evaluator authority identity is invalid")
    public_hex = row["public_key_ed25519_hex"]
    if type(public_hex) is not str or SHA256_RE.fullmatch(public_hex) is None:
        raise LaunchRefused(
            "frozen stage evaluator public key must be 32-byte lowercase hex"
        )
    source_sha = _sha256(
        row["evaluator_source_sha256"],
        name="frozen stage evaluator source SHA",
    )
    if row["evaluator_source_path"] != LAUNCHER_SOURCE:
        raise LaunchRefused(
            "frozen stage evaluator authority must pin the committed launcher "
            "that verifies trainer output and signed stage receipts"
        )
    _source, _relative, actual_source_sha, _mode = _verify_repo_blob(
        checkout,
        source_commit,
        row["evaluator_source_path"],
        name="frozen stage evaluator source",
    )
    if actual_source_sha != source_sha:
        raise LaunchRefused(
            "frozen stage evaluator source differs from authority pin"
        )
    declared = _sha256(
        row["canonical_sha256"],
        name="frozen stage evaluator authority canonical_sha256",
    )
    unsigned = dict(row)
    del unsigned["canonical_sha256"]
    actual = canonical_sha256(unsigned)
    if declared != actual:
        raise LaunchRefused(
            "frozen stage evaluator authority canonical SHA-256 mismatch"
        )
    return declared, bytes.fromhex(public_hex)


def _load_signed_payload(
    document: Any,
    *,
    expected_kind: str,
    public_key: bytes,
    name: str,
) -> dict[str, Any]:
    envelope = _exact_dict(
        document,
        ("schema_version", "kind", "payload", "signature_ed25519_hex"),
        name=name,
    )
    if envelope["schema_version"] != 1 or envelope["kind"] != expected_kind:
        raise LaunchRefused(f"{name} schema/kind is invalid")
    if type(envelope["payload"]) is not dict:
        raise LaunchRefused(f"{name}.payload must be a plain JSON object")
    _verify_ed25519_signature(
        public_key=public_key,
        payload=envelope["payload"],
        signature_hex=envelope["signature_ed25519_hex"],
        name=name,
    )
    return envelope["payload"]


def _finite_number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in (int, float):
        raise LaunchRefused(f"{name} must be a plain finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise LaunchRefused(f"{name} must be finite")
    if minimum is not None and numeric < minimum:
        raise LaunchRefused(f"{name} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise LaunchRefused(f"{name} must be <= {maximum}")
    return numeric


def _validate_prelaunch_safety_attestation(
    document: Any,
    *,
    public_key: bytes,
    authority_file_sha: str,
    source_commit: str,
    launch_profile: str,
    action_set_contract: dict[str, Any],
    order: tuple[str, ...],
    bindings: list[dict[str, Any]],
    manifest_sha: str,
    profile_pins_sha: str,
    launch_trust_spec_sha: str,
    launch_trust_root_sha: str,
    fitted_gate_sha: str,
    isaac_table_smoke_sha: str,
) -> str:
    payload = _load_signed_payload(
        document,
        expected_kind="action_ball_signed_prelaunch_safety_attestation",
        public_key=public_key,
        name="prelaunch safety attestation",
    )
    row = _exact_dict(
        payload,
        (
            "schema_version",
            "kind",
            "status",
            "source_commit_sha",
            "launch_profile",
            "action_set_contract_sha256",
            "ordered_action_ids",
            "manifest_sha256",
            "profile_pins_sha256",
            "fitted_ball_launch_trust_spec_sha256",
            "fitted_ball_launch_trust_root_sha256",
            "fitted_ball_gate_receipt_sha256",
            "isaac_table_smoke_receipt_sha256",
            "stage_evaluator_authority_sha256",
            "per_action",
        ),
        name="prelaunch safety attestation payload",
    )
    expected_identity = {
        "schema_version": 1,
        "kind": "action_ball_prelaunch_safety_attestation",
        "status": "passed",
        "source_commit_sha": source_commit,
        "launch_profile": launch_profile,
        "action_set_contract_sha256": action_set_contract[
            "contract_sha256"
        ],
        "ordered_action_ids": list(order),
        "manifest_sha256": manifest_sha,
        "profile_pins_sha256": profile_pins_sha,
        "fitted_ball_launch_trust_spec_sha256": launch_trust_spec_sha,
        "fitted_ball_launch_trust_root_sha256": launch_trust_root_sha,
        "fitted_ball_gate_receipt_sha256": fitted_gate_sha,
        "isaac_table_smoke_receipt_sha256": isaac_table_smoke_sha,
        "stage_evaluator_authority_sha256": authority_file_sha,
    }
    for key, expected in expected_identity.items():
        if row[key] != expected:
            raise LaunchRefused(
                f"prelaunch safety attestation {key} differs from launch identity"
            )
    actions = row["per_action"]
    if type(actions) is not list or len(actions) != len(bindings):
        raise LaunchRefused(
            "prelaunch safety attestation must contain exact contracted action rows"
        )
    boolean_gates = (
        "t_hit_pass",
        "t_cycle_pass",
        "physical_racket_site_speed_pass",
        "shared_ready_recovery_pass",
        "recorded_incoming_ball_returned_to_table",
        "no_table_contact",
        "grounded_safety_pass",
        "hard_limit_pass",
        "isaac_pose_obb_pass",
    )
    for index, (action, binding) in enumerate(zip(actions, bindings)):
        action = _exact_dict(
            action,
            (
                "action_id",
                "action_uid",
                "motion_sha256",
                "t_hit_s",
                "t_cycle_s",
                "physical_racket_site_speed_mps",
                "all_body_table_pair_count",
                "table_contact_count",
                "fall_count",
                "hard_limit_count",
                "unsafe_count",
                *boolean_gates,
            ),
            name=f"prelaunch safety per_action[{index}]",
        )
        if (
            action["action_id"] != binding["motion_id"]
            or action["action_uid"] != binding["action_uid"]
            or action["motion_sha256"] != binding["motion_sha256"]
        ):
            raise LaunchRefused(
                f"prelaunch safety per_action[{index}] identity drifted"
            )
        t_hit = _finite_number(
            action["t_hit_s"],
            name=f"prelaunch safety per_action[{index}].t_hit_s",
            minimum=0.0,
        )
        t_cycle = _finite_number(
            action["t_cycle_s"],
            name=f"prelaunch safety per_action[{index}].t_cycle_s",
            minimum=0.0,
        )
        _finite_number(
            action["physical_racket_site_speed_mps"],
            name=(
                f"prelaunch safety per_action[{index}]."
                "physical_racket_site_speed_mps"
            ),
            minimum=0.0,
        )
        if not t_hit < t_cycle:
            raise LaunchRefused(
                f"prelaunch safety per_action[{index}] requires t_hit < t_cycle"
            )
        _plain_int(
            action["all_body_table_pair_count"],
            name=(
                f"prelaunch safety per_action[{index}]."
                "all_body_table_pair_count"
            ),
            minimum=1,
        )
        for counter in (
            "table_contact_count",
            "fall_count",
            "hard_limit_count",
            "unsafe_count",
        ):
            if (
                _plain_int(
                    action[counter],
                    name=f"prelaunch safety per_action[{index}].{counter}",
                    minimum=0,
                )
                != 0
            ):
                raise LaunchRefused(
                    f"prelaunch safety per_action[{index}].{counter} must be zero"
                )
        failed = [gate for gate in boolean_gates if action[gate] is not True]
        if failed:
            raise LaunchRefused(
                f"prelaunch safety per_action[{index}] failed gates: {failed}"
            )
    return canonical_sha256(payload)


def _validate_override_list(value: Any) -> list[str]:
    if type(value) is not list:
        raise LaunchRefused("spec.train.extra_overrides must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if (
            type(item) is not str
            or not item
            or "\x00" in item
            or "\n" in item
            or "=" not in item
        ):
            raise LaunchRefused(
                f"spec.train.extra_overrides[{index}] must be one key=value string"
            )
        key, _ = item.split("=", 1)
        if OVERRIDE_KEY_RE.fullmatch(key) is None:
            raise LaunchRefused(
                f"invalid Hydra override key at extra_overrides[{index}]: {key!r}"
            )
        normalized = key.lstrip("+~")
        components = normalized.lower().split(".")
        if any("bank" in component or "exam" in component for component in components):
            raise LaunchRefused(
                f"generic bank/exam injection is forbidden: {key}"
            )
        if any(
            token in component
            for component in components
            for token in ("checkpoint", "resume")
        ) or normalized in {"load_run", "load_checkpoint"}:
            raise LaunchRefused(
                f"fresh profile forbids checkpoint/resume injection: {key}"
            )
        if normalized in _OWNED_OVERRIDE_KEYS or any(
            normalized.startswith(prefix)
            for prefix in _OWNED_OVERRIDE_PREFIXES
        ):
            raise LaunchRefused(
                f"launcher-owned override cannot be injected: {key}"
            )
        if normalized in seen:
            raise LaunchRefused(
                f"duplicate extra override key is forbidden: {key}"
            )
        if item not in _ALLOWED_EXTRA_OVERRIDES:
            raise LaunchRefused(
                "formal fresh launch permits only the exact non-scientific "
                f"extra override logger=tensorboard; rejected: {key}"
            )
        seen.add(normalized)
        result.append(item)
    return result


def _validate_stage_budgets(
    stages: Any, action_set_contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    stages = _exact_dict(stages, STAGE_ORDER, name="spec.stages")
    normalized: dict[str, dict[str, Any]] = {}
    namespaces: list[str] = []
    for stage in STAGE_ORDER:
        row = _exact_dict(
            stages[stage],
            (
                "namespace",
                "num_envs",
                "max_iterations",
                "save_interval",
                "evaluation_inbox_root",
                "evaluation_owner_id",
                "evaluation_run_id",
                "frozen_eval_interval_updates",
                "trainer_gpu_owner_receipt",
                "evaluator_gpu_owner_receipt",
                "predecessor_receipt",
            ),
            name=f"spec.stages.{stage}",
        )
        namespace = _absolute_normalized_path(
            row["namespace"],
            name=f"spec.stages.{stage}.namespace",
            must_exist=False,
        )
        if (
            len(namespace.name) > 128
            or RUN_COMPONENT_RE.fullmatch(namespace.name) is None
        ):
            raise LaunchRefused(
                f"spec.stages.{stage}.namespace basename is not a safe run name"
            )
        namespace_identity = action_set_contract["namespace_identity"]
        if namespace_identity not in namespace.name:
            raise LaunchRefused(
                f"spec.stages.{stage}.namespace basename must contain exact "
                f"action-set identity {namespace_identity}"
            )
        inbox_root = _absolute_normalized_path(
            row["evaluation_inbox_root"],
            name=f"spec.stages.{stage}.evaluation_inbox_root",
            must_exist=False,
        )
        expected_inbox_root = namespace / "frozen_eval_inbox"
        if inbox_root != expected_inbox_root:
            raise LaunchRefused(
                f"spec.stages.{stage}.evaluation_inbox_root must be the "
                f"stage-owned no-clobber path {expected_inbox_root}"
            )
        evaluation_owner_id = row["evaluation_owner_id"]
        if (
            type(evaluation_owner_id) is not str
            or len(evaluation_owner_id) > 128
            or RUN_COMPONENT_RE.fullmatch(evaluation_owner_id) is None
        ):
            raise LaunchRefused(
                f"spec.stages.{stage}.evaluation_owner_id must be a safe "
                "1-128 character identifier"
            )
        evaluation_run_id = row["evaluation_run_id"]
        if evaluation_run_id != namespace.name:
            raise LaunchRefused(
                f"spec.stages.{stage}.evaluation_run_id must equal its "
                "no-clobber namespace basename"
            )
        frozen_eval_interval = _plain_int(
            row["frozen_eval_interval_updates"],
            name=f"spec.stages.{stage}.frozen_eval_interval_updates",
            minimum=1,
        )
        parent = namespace.parent
        try:
            parent_resolved = parent.resolve(strict=True)
        except OSError as exc:
            raise LaunchRefused(
                f"spec.stages.{stage}.namespace parent does not exist"
            ) from exc
        if parent_resolved != parent or not parent.is_dir():
            raise LaunchRefused(
                f"spec.stages.{stage}.namespace parent must be a real directory"
            )
        num_envs = _plain_int(
            row["num_envs"],
            name=f"spec.stages.{stage}.num_envs",
            minimum=1,
        )
        iterations = _plain_int(
            row["max_iterations"],
            name=f"spec.stages.{stage}.max_iterations",
            minimum=1,
        )
        save_interval = _plain_int(
            row["save_interval"],
            name=f"spec.stages.{stage}.save_interval",
            minimum=1,
        )
        if save_interval > iterations:
            raise LaunchRefused(
                f"spec.stages.{stage}.save_interval exceeds max_iterations"
            )
        if frozen_eval_interval > iterations:
            raise LaunchRefused(
                f"spec.stages.{stage}.frozen_eval_interval_updates exceeds "
                "max_iterations"
            )
        normalized[stage] = {
            **row,
            "namespace_path": namespace,
            "num_envs": num_envs,
            "max_iterations": iterations,
            "save_interval": save_interval,
            "evaluation_inbox_root": str(inbox_root),
            "evaluation_owner_id": evaluation_owner_id,
            "evaluation_run_id": evaluation_run_id,
            "frozen_eval_interval_updates": frozen_eval_interval,
        }
        namespaces.append(str(namespace))
    if len(namespaces) != len(set(namespaces)):
        raise LaunchRefused("smoke/canary/long namespaces must be distinct")
    smoke = normalized["smoke"]
    if (
        smoke["num_envs"],
        smoke["max_iterations"],
        smoke["save_interval"],
        smoke["frozen_eval_interval_updates"],
    ) != (1, 2, 1, 2):
        raise LaunchRefused(
            "smoke is fixed at exactly 1 env / 2 updates / save every update / "
            "one terminal evaluator construction point"
        )
    canary = normalized["canary"]
    if (
        canary["num_envs"] < 2
        or canary["max_iterations"] < 3
        or (
            canary["max_iterations"]
            // canary["frozen_eval_interval_updates"]
        )
        < action_set_contract["expected_n"]
    ):
        raise LaunchRefused(
            "canary must be larger than smoke and budget at least one frozen "
            "evaluation scheduling point per contracted action"
        )
    long = normalized["long"]
    if (
        long["num_envs"] < LONG_MIN_NUM_ENVS
        or long["max_iterations"] < LONG_MIN_ITERATIONS
        or long["save_interval"] > LONG_MAX_SAVE_INTERVAL
    ):
        raise LaunchRefused(
            "long is a full preregistered run: it requires at least "
            f"{LONG_MIN_NUM_ENVS} envs / {LONG_MIN_ITERATIONS} updates and "
            f"save_interval <= {LONG_MAX_SAVE_INTERVAL}"
        )
    if (
        long["num_envs"] < canary["num_envs"]
        or long["max_iterations"] <= canary["max_iterations"]
    ):
        raise LaunchRefused(
            "long must also use at least canary envs and strictly more updates"
        )
    if smoke["predecessor_receipt"] is not None:
        raise LaunchRefused("smoke predecessor_receipt must be null")
    return normalized


def _rsl_experiment_root(checkout: Path, experiment_name: str) -> Path:
    return (
        checkout
        / "hope_training/whole_body_tracking/logs/rsl_rl"
        / experiment_name
    )


def _stage_evaluation_runtime(
    stage_row: dict[str, Any],
    evaluator_identity: dict[str, Any],
) -> dict[str, Any]:
    heartbeat_path = (
        Path(stage_row["evaluation_inbox_root"])
        / "sidecar_status"
        / stage_row["evaluation_owner_id"]
        / stage_row["evaluation_run_id"]
        / "heartbeat.json"
    )
    return {
        "inbox_root": stage_row["evaluation_inbox_root"],
        "owner_id": stage_row["evaluation_owner_id"],
        "run_id": stage_row["evaluation_run_id"],
        "interval_updates": stage_row["frozen_eval_interval_updates"],
        "heartbeat_path": str(heartbeat_path),
        "heartbeat_contract": SIDECAR_HEARTBEAT_CONTRACT,
        "evaluator_v4_identity": evaluator_identity,
    }


def _validate_gpu_role(value: Any, *, role: str) -> dict[str, Any]:
    row = _exact_dict(
        value,
        (
            "index",
            "uuid",
            "owner",
            "lock_path",
            "boot_lock_path",
            "require_empty",
        ),
        name=f"spec.gpus.{role}",
    )
    index = _plain_int(
        row["index"], name=f"spec.gpus.{role}.index", minimum=0
    )
    required_index = 0 if role == "trainer" else 1
    if index != required_index:
        raise LaunchRefused(
            f"spec.gpus.{role}.index must be physical GPU {required_index}"
        )
    if (
        type(row["uuid"]) is not str
        or not row["uuid"].startswith("GPU-")
        or len(row["uuid"]) < 8
    ):
        raise LaunchRefused(
            f"spec.gpus.{role}.uuid must be an explicit GPU UUID"
        )
    owner = row["owner"]
    if (
        type(owner) is not str
        or not owner.strip()
        or owner != owner.strip()
        or owner.lower()
        in {"codex", "claude", "fable", "agent", "unassigned"}
    ):
        raise LaunchRefused(
            f"spec.gpus.{role}.owner must be an explicit human name"
        )
    lock_path = _absolute_normalized_path(
        row["lock_path"],
        name=f"spec.gpus.{role}.lock_path",
        must_exist=False,
    )
    expected_lock = Path(f"/tmp/hope_lean_queue_gpu{index}.lock")
    if lock_path != expected_lock:
        raise LaunchRefused(
            f"spec.gpus.{role}.lock_path must use the shared GPU-wide lock "
            f"{expected_lock}"
        )
    boot_lock = _absolute_normalized_path(
        row["boot_lock_path"],
        name=f"spec.gpus.{role}.boot_lock_path",
        must_exist=False,
    )
    if boot_lock != Path("/workspace/.kit_boot.lock"):
        raise LaunchRefused(
            f"spec.gpus.{role}.boot_lock_path must be the pod-wide "
            "/workspace/.kit_boot.lock"
        )
    if row["require_empty"] is not True:
        raise LaunchRefused(f"spec.gpus.{role}.require_empty must be true")
    return {
        "index": index,
        "uuid": row["uuid"],
        "owner": owner,
        "lock_path": str(lock_path),
        "boot_lock_path": str(boot_lock),
        "require_empty": True,
    }


def _require_real_directory(path: Path, *, name: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be inspected: {path}: {exc}") from exc
    if not stat.S_ISDIR(mode):
        raise LaunchRefused(f"{name} must be a real non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be resolved: {path}: {exc}") from exc
    if resolved != path:
        raise LaunchRefused(f"{name} must not resolve through symlinks")


def _validate_rsl_output_slot(
    checkout: Path, run_name: str, experiment_name: str
) -> None:
    """Refuse a reused trainer suffix without creating any output directory.

    ``train.py`` currently owns the wall-clock timestamp prefix and has no
    explicit ``log_dir`` seam.  The launcher therefore owns a dedicated
    experiment root and a never-reused namespace basename, then rechecks that
    no timestamped directory with this exact suffix already exists.  The
    trainer's actual timestamped path is bound by the signed stage receipt.
    """

    root = _rsl_experiment_root(checkout, experiment_name)
    if not os.path.lexists(root):
        return
    _assert_no_symlink_components(root, start=checkout, name="RSL output root")
    _require_real_directory(root, name="RSL output root")
    suffix = "_" + run_name
    try:
        children = list(root.iterdir())
    except OSError as exc:
        raise LaunchRefused(f"RSL output root cannot be enumerated: {exc}") from exc
    spent = sorted(child.name for child in children if child.name.endswith(suffix))
    if spent:
        raise LaunchRefused(
            "trainer output run_name is permanently spent under the dedicated "
            f"RSL root: {spent[0]}"
        )


def _validate_owner_receipt(
    pin: Any,
    *,
    stage: str,
    namespace: Path,
    source_commit: str,
    gpu_index: int,
    gpu_uuid: str,
    owner: str,
    lock_path: Path,
) -> str:
    path, digest = _verify_external_pin(
        pin, name=f"spec.stages.{stage}.gpu_owner_receipt"
    )
    row = _exact_dict(
        load_strict_json(path, name="GPU owner receipt"),
        (
            "schema_version",
            "kind",
            "owner",
            "gpu_index",
            "gpu_uuid",
            "lock_path",
            "stage",
            "namespace",
            "source_commit_sha",
        ),
        name="GPU owner receipt",
    )
    expected = {
        "schema_version": 1,
        "kind": "action_ball_gpu_owner",
        "owner": owner,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "lock_path": str(lock_path),
        "stage": stage,
        "namespace": str(namespace),
        "source_commit_sha": source_commit,
    }
    if row != expected:
        raise LaunchRefused(
            "GPU owner receipt does not exactly authorize this stage/namespace/commit"
        )
    return digest


def _verify_namespace_pin(
    namespace: Path,
    value: Any,
    *,
    name: str,
) -> tuple[Path, str]:
    row = _exact_dict(value, ("path", "sha256"), name=name)
    relative = row["path"]
    if (
        type(relative) is not str
        or not relative
        or Path(relative).is_absolute()
        or "\\" in relative
        or ":" in relative
        or "\x00" in relative
        or relative != Path(relative).as_posix()
        or any(part in ("", ".", "..") for part in Path(relative).parts)
    ):
        raise LaunchRefused(f"{name}.path must be namespace-relative")
    path = namespace / relative
    _assert_no_symlink_components(path, start=namespace, name=name)
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be inspected: {path}") from exc
    if not stat.S_ISREG(mode):
        raise LaunchRefused(f"{name} must be a regular file")
    expected = _sha256(row["sha256"], name=f"{name}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise LaunchRefused(
            f"{name} byte SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return path, actual


def _validate_trainer_output(
    value: Any,
    *,
    checkout: Path,
    completed_namespace: Path,
    expected_run_name: str,
    expected_experiment_name: str,
    expected_claim_sha: str,
    expected_ground_plant_sha: str,
    expected_reward_sha: str,
    expected_ppo_sha: str,
) -> dict[str, Any]:
    """Bind a signed result to the timestamped directory emitted by train.py."""

    row = _exact_dict(
        value,
        (
            "rsl_log_dir",
            "timestamp_prefix",
            "run_name",
            "launcher_log_sha256",
            "training_contract",
            "effective_reward_recipe",
        ),
        name="stage result trainer_output",
    )
    if row["run_name"] != expected_run_name:
        raise LaunchRefused("trainer_output.run_name differs from the claim namespace")
    timestamp_prefix = row["timestamp_prefix"]
    if (
        type(timestamp_prefix) is not str
        or RSL_TIMESTAMP_RE.fullmatch(timestamp_prefix) is None
    ):
        raise LaunchRefused(
            "trainer_output.timestamp_prefix must be YYYY-MM-DD_HH-MM-SS"
        )
    try:
        parsed_timestamp = _datetime.datetime.strptime(
            timestamp_prefix, "%Y-%m-%d_%H-%M-%S"
        )
    except ValueError as exc:
        raise LaunchRefused(
            "trainer_output.timestamp_prefix is not a real calendar timestamp"
        ) from exc
    if parsed_timestamp.strftime("%Y-%m-%d_%H-%M-%S") != timestamp_prefix:
        raise LaunchRefused("trainer_output.timestamp_prefix is not canonical")

    log_dir = _absolute_normalized_path(
        row["rsl_log_dir"],
        name="stage result trainer_output.rsl_log_dir",
        must_exist=True,
    )
    _assert_no_symlink_components(
        log_dir, start=checkout, name="stage result trainer output"
    )
    _require_real_directory(log_dir, name="stage result trainer output")
    expected_root = _rsl_experiment_root(checkout, expected_experiment_name)
    if log_dir.parent != expected_root:
        raise LaunchRefused(
            "trainer_output.rsl_log_dir is outside the dedicated ActionBall "
            "experiment root"
        )
    expected_basename = f"{timestamp_prefix}_{expected_run_name}"
    if log_dir.name != expected_basename:
        raise LaunchRefused(
            "trainer_output.rsl_log_dir does not bind its timestamp and run_name"
        )

    launcher_log = completed_namespace / "train.log"
    _assert_no_symlink_components(
        launcher_log, start=completed_namespace, name="launcher trainer log"
    )
    try:
        launcher_log_mode = launcher_log.lstat().st_mode
    except OSError as exc:
        raise LaunchRefused(
            f"launcher trainer log cannot be inspected: {launcher_log}"
        ) from exc
    if not stat.S_ISREG(launcher_log_mode):
        raise LaunchRefused("launcher trainer log must be a regular non-symlink file")
    expected_launcher_log_sha = _sha256(
        row["launcher_log_sha256"],
        name="trainer_output.launcher_log_sha256",
    )
    actual_launcher_log_sha = sha256_file(launcher_log)
    if actual_launcher_log_sha != expected_launcher_log_sha:
        raise LaunchRefused("launcher trainer log SHA-256 differs from stage receipt")
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LaunchRefused("launcher trainer log is not readable UTF-8") from exc
    prefix = (
        "[INFO] Task: HOPE-PingPong-ActionBall-AgibotA3-v0 | experiment: "
        f"{expected_experiment_name} | log: "
    )
    output_lines = [
        line[len(prefix) :]
        for line in log_text.splitlines()
        if line.startswith(prefix)
    ]
    if output_lines != [str(log_dir)]:
        raise LaunchRefused(
            "launcher log must contain exactly one trainer-emitted timestamped "
            "RSL output path matching trainer_output"
        )

    contract_pin = _exact_dict(
        row["training_contract"],
        ("path", "sha256"),
        name="trainer_output.training_contract",
    )
    reward_pin = _exact_dict(
        row["effective_reward_recipe"],
        ("path", "sha256"),
        name="trainer_output.effective_reward_recipe",
    )
    if contract_pin["path"] != "params/training_contract.json":
        raise LaunchRefused(
            "trainer_output.training_contract.path must be params/training_contract.json"
        )
    if reward_pin["path"] != "params/effective_reward_recipe.json":
        raise LaunchRefused(
            "trainer_output.effective_reward_recipe.path must be "
            "params/effective_reward_recipe.json"
        )
    contract_path, contract_file_sha = _verify_namespace_pin(
        log_dir,
        contract_pin,
        name="trainer hard contract",
    )
    reward_path, reward_file_sha = _verify_namespace_pin(
        log_dir,
        reward_pin,
        name="effective Reward receipt",
    )
    contract = load_strict_json(contract_path, name="trainer hard contract")
    if type(contract) is not dict:
        raise LaunchRefused("trainer hard contract must be a JSON object")
    reward = load_strict_json(reward_path, name="effective Reward receipt")
    if type(reward) is not dict:
        raise LaunchRefused("effective Reward receipt must be a JSON object")
    if contract.get("effective_reward_recipe") != reward:
        raise LaunchRefused(
            "effective Reward receipt differs from the trainer hard contract"
        )
    if reward.get("sha256") != expected_reward_sha:
        raise LaunchRefused(
            "trainer effective Reward SHA differs from the preregistered recipe"
        )
    action_ball = contract.get("action_ball_training")
    if (
        type(action_ball) is not dict
        or action_ball.get("effective_reward_recipe_sha256")
        != expected_reward_sha
    ):
        raise LaunchRefused(
            "trainer action_ball_training does not bind the effective Reward recipe"
        )
    ppo_recipe = contract.get("action_ball_ppo_runner_recipe")
    if type(ppo_recipe) is not dict:
        raise LaunchRefused(
            "trainer hard contract lacks action_ball_ppo_runner_recipe"
        )
    if canonical_sha256(ppo_recipe) != expected_ppo_sha:
        raise LaunchRefused(
            "trainer PPO recipe differs from the preregistered recipe"
        )
    ground_plant = contract.get("ground_plant")
    actual_ground_sha = (
        GROUND_PLANT_ABSENT_SHA256
        if ground_plant is None
        else canonical_sha256(ground_plant)
    )
    if actual_ground_sha != expected_ground_plant_sha:
        raise LaunchRefused(
            "trainer ground-plant contract differs from the preregistered recipe"
        )

    return {
        "rsl_log_dir": str(log_dir),
        "timestamp_prefix": timestamp_prefix,
        "run_name": expected_run_name,
        "launcher_log_sha256": actual_launcher_log_sha,
        "training_contract_file_sha256": contract_file_sha,
        "effective_reward_recipe_file_sha256": reward_file_sha,
        "training_launch_claim_sha256": expected_claim_sha,
    }


def _validate_prior_launch_claim(
    checkout: Path,
    namespace: Path,
    *,
    completed_stage: str,
    action_set_contract: dict[str, Any],
    source_commit: str,
    order: tuple[str, ...],
    manifest_sha: str,
    prototype_sha: str,
    admission_sha: str,
    evaluator_sha: str,
    sidecar_sha: str,
    drain_reset_sha: str,
    policy_sha: str,
    profile_pins_sha: str,
    launch_trust_spec_sha: str,
    launch_trust_root_sha: str,
    fitted_gate_sha: str,
    isaac_table_smoke_sha: str,
    prelaunch_safety_attestation_sha: str,
    expected_stage_budget: dict[str, int],
    expected_training_recipe: dict[str, Any],
    expected_isaac_python_runtime: dict[str, Any],
    expected_evaluation_runtime: dict[str, Any],
    expected_gpu_roles: dict[str, Any],
    expected_training_entrypoint: dict[str, Any],
    expected_claim_sha: str,
) -> None:
    claim_path = namespace / "launch_claim.json"
    _assert_no_symlink_components(
        claim_path, start=namespace, name="prior launch claim"
    )
    claim = _exact_dict(
        load_strict_json(claim_path, name="prior launch claim"),
        (
            "schema_version",
            "kind",
            "launch_claim_sha256",
            "canonical_payload",
            "argv",
            "confirmation_claim_sha256",
        ),
        name="prior launch claim",
    )
    if claim["schema_version"] != SCHEMA_VERSION or claim["kind"] != CLAIM_KIND:
        raise LaunchRefused("prior launch claim schema/kind is invalid")
    claim_sha = _sha256(
        claim["launch_claim_sha256"], name="prior launch claim SHA"
    )
    if (
        canonical_sha256(claim["canonical_payload"]) != claim_sha
        or claim_sha != expected_claim_sha
        or claim["confirmation_claim_sha256"] != claim_sha
    ):
        raise LaunchRefused(
            "prior launch claim canonical digest/confirmation does not match"
        )
    payload = claim["canonical_payload"]
    expected_scalars = {
        "stage": completed_stage,
        "namespace": str(namespace),
        "source_commit_sha": source_commit,
        "ordered_action_ids": list(order),
        "action_set_contract": action_set_contract,
        "policy_contract_sha256": policy_sha,
        "fitted_ball_profile_pins_sha256": profile_pins_sha,
        "fitted_ball_launch_trust_spec_sha256": launch_trust_spec_sha,
        "fitted_ball_launch_trust_root_sha256": launch_trust_root_sha,
        "fitted_ball_gate_receipt_sha256": fitted_gate_sha,
        "isaac_table_smoke_receipt_sha256": isaac_table_smoke_sha,
        "prelaunch_safety_attestation_sha256": (
            prelaunch_safety_attestation_sha
        ),
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise LaunchRefused(
                f"prior launch claim {key} differs from current lineage"
            )
    if payload.get("stage_budget") != expected_stage_budget:
        raise LaunchRefused(
            "prior launch claim stage_budget differs from the completed stage"
        )
    if payload.get("training_recipe") != expected_training_recipe:
        raise LaunchRefused(
            "prior launch claim seed/extra_overrides/ground/Reward/PPO recipe "
            "differs from the current spec"
        )
    if payload.get("training_recipe_sha256") != canonical_sha256(
        expected_training_recipe
    ):
        raise LaunchRefused(
            "prior launch claim training_recipe_sha256 differs from its recipe"
        )
    if payload.get("isaac_python_runtime") != expected_isaac_python_runtime:
        raise LaunchRefused(
            "prior launch claim Isaac Python runtime differs from the current spec"
        )
    if payload.get("frozen_evaluation_runtime") != expected_evaluation_runtime:
        raise LaunchRefused(
            "prior launch claim frozen-evaluation inbox/owner/run/interval "
            "identity differs from the completed stage"
        )
    if payload.get("gpus") != expected_gpu_roles:
        raise LaunchRefused(
            "prior launch claim trainer/evaluator GPU role identity differs "
            "from the completed stage"
        )
    if (
        payload.get("isolated_training_entrypoint")
        != expected_training_entrypoint
    ):
        raise LaunchRefused(
            "prior launch claim isolated training entrypoint drifted"
        )
    expected_pin_rows = {
        "manifest": manifest_sha,
        "prototype": prototype_sha,
    }
    for key, expected in expected_pin_rows.items():
        if (
            type(payload.get(key)) is not dict
            or payload[key].get("sha256") != expected
        ):
            raise LaunchRefused(
                f"prior launch claim {key} differs from current lineage"
            )
    if (
        type(payload.get("motion_admission_receipt")) is not dict
        or payload["motion_admission_receipt"].get("file_sha256")
        != admission_sha
        or type(payload.get("evaluator_launch_receipt")) is not dict
        or payload["evaluator_launch_receipt"].get("file_sha256")
        != evaluator_sha
        or type(payload.get("sidecar_launch_receipt")) is not dict
        or payload["sidecar_launch_receipt"].get("file_sha256")
        != sidecar_sha
        or type(payload.get("drain_reset_launch_receipt")) is not dict
        or payload["drain_reset_launch_receipt"].get("file_sha256")
        != drain_reset_sha
    ):
        raise LaunchRefused(
            "prior launch claim admission/evaluator/sidecar/drain differs from "
            "current lineage"
        )
    argv = claim["argv"]
    base_argv = payload.get("argv_without_launch_claim")
    base_contract = expected_training_entrypoint[
        "nosite_argv_contract"
    ]
    try:
        nosite = _load_source_module_without_bytecode(
            checkout / NOSITE_BOOTSTRAP_SOURCE,
            name="_action_ball_launcher_prior_claim_nosite_bootstrap",
            purpose="prior-claim no-site bootstrap",
        )
        base_command = nosite.validate_exact_nosite_argv(
            base_argv,
            expected_python=Path(expected_isaac_python_runtime["path"]),
            expected_bootstrap=base_contract["bootstrap"],
            expected_entrypoint=base_contract["entrypoint"],
            expected_import_roots=base_contract["import_roots"],
            expected_entrypoint_argv=base_contract["entrypoint_argv"],
            expected_contract_sha256=expected_training_entrypoint[
                "nosite_argv_contract_sha256"
            ],
            verify_live=True,
        )
        claim_command = nosite.validate_exact_nosite_argv(
            argv,
            expected_python=Path(expected_isaac_python_runtime["path"]),
            expected_bootstrap=base_contract["bootstrap"],
            expected_entrypoint=base_contract["entrypoint"],
            expected_import_roots=base_contract["import_roots"],
            expected_entrypoint_argv=[
                *base_command.contract["entrypoint_argv"],
                f"++training_launch_claim_sha256={claim_sha}",
            ],
            verify_live=True,
        )
    except Exception as exc:
        raise LaunchRefused(
            f"prior launch claim argv is not exactly self-bound: {exc}"
        ) from exc
    if claim_command.argv != tuple(argv):
        raise LaunchRefused(
            "prior launch claim argv canonicalization drifted"
        )


def _validate_stage_metrics(value: Any) -> None:
    row = _exact_dict(
        value,
        (
            "proposed_count",
            "solver_admitted_count",
            "solver_rejected_count",
            "solver_rejection_reason_counts",
            "attempt_count",
            "return_success_count",
            "policy_return_failure_count",
            "return_success_lcb",
            "policy_return_failure_rate",
            "unsafe_count",
            "table_hit_count",
            "fall_count",
            "hard_limit_count",
            "nan_count",
            "counter_violation_count",
            "domain_epoch_stale_count",
            "curriculum_counter_invariants_passed",
        ),
        name="stage result metrics",
    )
    counters: dict[str, int] = {}
    for key in (
        "proposed_count",
        "solver_admitted_count",
        "solver_rejected_count",
        "attempt_count",
        "return_success_count",
        "policy_return_failure_count",
        "unsafe_count",
        "table_hit_count",
        "fall_count",
        "hard_limit_count",
        "nan_count",
        "counter_violation_count",
        "domain_epoch_stale_count",
    ):
        counters[key] = _plain_int(
            row[key], name=f"stage result metrics.{key}", minimum=0
        )
    reasons = row["solver_rejection_reason_counts"]
    if type(reasons) is not dict or any(
        type(key) is not str
        or not key
        or type(count) is not int
        or count < 0
        for key, count in reasons.items()
    ):
        raise LaunchRefused(
            "stage result metrics solver_rejection_reason_counts is invalid"
        )
    if (
        counters["proposed_count"]
        != counters["solver_admitted_count"]
        + counters["solver_rejected_count"]
        or sum(reasons.values()) != counters["solver_rejected_count"]
        or counters["attempt_count"]
        != counters["return_success_count"]
        + counters["policy_return_failure_count"]
        or counters["attempt_count"] > counters["solver_admitted_count"]
    ):
        raise LaunchRefused("stage result counter invariants are false")
    _finite_number(
        row["return_success_lcb"],
        name="stage result metrics.return_success_lcb",
        minimum=0.0,
        maximum=1.0,
    )
    _finite_number(
        row["policy_return_failure_rate"],
        name="stage result metrics.policy_return_failure_rate",
        minimum=0.0,
        maximum=1.0,
    )
    if row["curriculum_counter_invariants_passed"] is not True:
        raise LaunchRefused(
            "stage result curriculum counter invariant gate did not pass"
        )
    for key in (
        "unsafe_count",
        "table_hit_count",
        "fall_count",
        "hard_limit_count",
        "nan_count",
        "counter_violation_count",
        "domain_epoch_stale_count",
    ):
        if counters[key] != 0:
            raise LaunchRefused(f"stage result metrics.{key} must be zero")


def _validate_predecessor_receipt(
    pin: Any,
    *,
    public_key: bytes,
    authority_file_sha: str,
    checkout: Path,
    completed_stage: str,
    completed_namespace: Path,
    completed_stage_budget: dict[str, int],
    action_set_contract: dict[str, Any],
    launch_profile: str,
    source_commit: str,
    order: tuple[str, ...],
    manifest_sha: str,
    prototype_sha: str,
    admission_sha: str,
    evaluator_sha: str,
    sidecar_sha: str,
    drain_reset_sha: str,
    policy_sha: str,
    profile_pins_sha: str,
    launch_trust_spec_sha: str,
    launch_trust_root_sha: str,
    fitted_gate_sha: str,
    isaac_table_smoke_sha: str,
    prelaunch_safety_attestation_sha: str,
    training_recipe: dict[str, Any],
    isaac_python_runtime: dict[str, Any],
    evaluation_runtime: dict[str, Any],
    gpu_roles: dict[str, Any],
    training_entrypoint: dict[str, Any],
) -> str:
    path, digest = _verify_external_pin(
        pin, name=f"{completed_stage} result receipt pin"
    )
    payload = _load_signed_payload(
        load_strict_json(path, name=f"{completed_stage} result receipt"),
        expected_kind="action_ball_signed_stage_result",
        public_key=public_key,
        name=f"{completed_stage} result receipt",
    )
    row = _exact_dict(
        payload,
        (
            "schema_version",
            "kind",
            "status",
            "completed_stage",
            "launch_profile",
            "source_commit_sha",
            "ordered_action_ids",
            "action_set_contract_sha256",
            "manifest_sha256",
            "prototype_sha256",
            "motion_admission_receipt_sha256",
            "evaluator_launch_receipt_sha256",
            "sidecar_launch_receipt_sha256",
            "drain_reset_launch_receipt_sha256",
            "policy_contract_sha256",
            "fitted_ball_profile_pins_sha256",
            "fitted_ball_launch_trust_spec_sha256",
            "fitted_ball_launch_trust_root_sha256",
            "fitted_ball_gate_receipt_sha256",
            "isaac_table_smoke_receipt_sha256",
            "prelaunch_safety_attestation_sha256",
            "stage_evaluator_authority_sha256",
            "namespace",
            "launch_claim_sha256",
            "stage_budget",
            "training_recipe_sha256",
            "isaac_python_runtime_sha256",
            "frozen_evaluation_runtime_sha256",
            "gpu_roles_sha256",
            "isolated_training_entrypoint_sha256",
            "trainer_output",
            "metrics_evidence",
            "checkpoint",
            "metrics",
        ),
        name=f"{completed_stage} result receipt",
    )
    _sha256(
        row["launch_claim_sha256"],
        name=f"{completed_stage} result receipt launch_claim_sha256",
    )
    expected = {
        "schema_version": 1,
        "kind": "action_ball_stage_result",
        "status": "passed",
        "completed_stage": completed_stage,
        "launch_profile": launch_profile,
        "source_commit_sha": source_commit,
        "ordered_action_ids": list(order),
        "action_set_contract_sha256": action_set_contract[
            "contract_sha256"
        ],
        "manifest_sha256": manifest_sha,
        "prototype_sha256": prototype_sha,
        "motion_admission_receipt_sha256": admission_sha,
        "evaluator_launch_receipt_sha256": evaluator_sha,
        "sidecar_launch_receipt_sha256": sidecar_sha,
        "drain_reset_launch_receipt_sha256": drain_reset_sha,
        "policy_contract_sha256": policy_sha,
        "fitted_ball_profile_pins_sha256": profile_pins_sha,
        "fitted_ball_launch_trust_spec_sha256": launch_trust_spec_sha,
        "fitted_ball_launch_trust_root_sha256": launch_trust_root_sha,
        "fitted_ball_gate_receipt_sha256": fitted_gate_sha,
        "isaac_table_smoke_receipt_sha256": isaac_table_smoke_sha,
        "prelaunch_safety_attestation_sha256": (
            prelaunch_safety_attestation_sha
        ),
        "stage_evaluator_authority_sha256": authority_file_sha,
        "namespace": str(completed_namespace),
        "launch_claim_sha256": row["launch_claim_sha256"],
        "stage_budget": completed_stage_budget,
        "training_recipe_sha256": canonical_sha256(training_recipe),
        "isaac_python_runtime_sha256": canonical_sha256(isaac_python_runtime),
        "frozen_evaluation_runtime_sha256": canonical_sha256(
            evaluation_runtime
        ),
        "gpu_roles_sha256": canonical_sha256(gpu_roles),
        "isolated_training_entrypoint_sha256": canonical_sha256(
            training_entrypoint
        ),
        "trainer_output": row["trainer_output"],
        "metrics_evidence": row["metrics_evidence"],
        "checkpoint": row["checkpoint"],
        "metrics": row["metrics"],
    }
    if row["training_recipe_sha256"] != expected["training_recipe_sha256"]:
        raise LaunchRefused(
            f"{completed_stage} result receipt training recipe differs from "
            "the current seed/extra_overrides/ground/Reward/PPO spec"
        )
    if (
        row["isaac_python_runtime_sha256"]
        != expected["isaac_python_runtime_sha256"]
    ):
        raise LaunchRefused(
            f"{completed_stage} result receipt Isaac Python runtime differs "
            "from the current spec"
        )
    if (
        row["frozen_evaluation_runtime_sha256"]
        != expected["frozen_evaluation_runtime_sha256"]
    ):
        raise LaunchRefused(
            f"{completed_stage} result receipt frozen-evaluation runtime "
            "differs from the completed stage"
        )
    if row["gpu_roles_sha256"] != expected["gpu_roles_sha256"]:
        raise LaunchRefused(
            f"{completed_stage} result receipt dual-GPU role binding drifted"
        )
    if (
        row["isolated_training_entrypoint_sha256"]
        != expected["isolated_training_entrypoint_sha256"]
    ):
        raise LaunchRefused(
            f"{completed_stage} result receipt isolated training entrypoint "
            "drifted"
        )
    if row != expected:
        raise LaunchRefused(
            f"{completed_stage} result receipt does not match this launch lineage"
        )
    _validate_prior_launch_claim(
        checkout,
        completed_namespace,
        completed_stage=completed_stage,
        action_set_contract=action_set_contract,
        source_commit=source_commit,
        order=order,
        manifest_sha=manifest_sha,
        prototype_sha=prototype_sha,
        admission_sha=admission_sha,
        evaluator_sha=evaluator_sha,
        sidecar_sha=sidecar_sha,
        drain_reset_sha=drain_reset_sha,
        policy_sha=policy_sha,
        profile_pins_sha=profile_pins_sha,
        launch_trust_spec_sha=launch_trust_spec_sha,
        launch_trust_root_sha=launch_trust_root_sha,
        fitted_gate_sha=fitted_gate_sha,
        isaac_table_smoke_sha=isaac_table_smoke_sha,
        prelaunch_safety_attestation_sha=prelaunch_safety_attestation_sha,
        expected_stage_budget=completed_stage_budget,
        expected_training_recipe=training_recipe,
        expected_isaac_python_runtime=isaac_python_runtime,
        expected_evaluation_runtime=evaluation_runtime,
        expected_gpu_roles=gpu_roles,
        expected_training_entrypoint=training_entrypoint,
        expected_claim_sha=row["launch_claim_sha256"],
    )
    _verify_namespace_pin(
        completed_namespace,
        row["metrics_evidence"],
        name=f"{completed_stage} metrics evidence",
    )
    trainer_output = _validate_trainer_output(
        row["trainer_output"],
        checkout=checkout,
        completed_namespace=completed_namespace,
        expected_run_name=completed_namespace.name,
        expected_experiment_name=action_set_contract["experiment_name"],
        expected_claim_sha=row["launch_claim_sha256"],
        expected_ground_plant_sha=training_recipe[
            "ground_plant_contract_sha256"
        ],
        expected_reward_sha=training_recipe[
            "effective_reward_recipe_sha256"
        ],
        expected_ppo_sha=training_recipe["ppo_recipe_sha256"],
    )
    rsl_log_dir = Path(trainer_output["rsl_log_dir"])
    checkpoint = _exact_dict(
        row["checkpoint"],
        ("path", "sha256", "finite", "exact_resume_passed"),
        name=f"{completed_stage} checkpoint",
    )
    if (
        type(checkpoint["path"]) is not str
        or re.fullmatch(r"model_[0-9]+\.pt", checkpoint["path"]) is None
    ):
        raise LaunchRefused(
            f"{completed_stage} checkpoint must be one root-level model_<N>.pt "
            "inside the bound trainer output"
        )
    _verify_namespace_pin(
        rsl_log_dir,
        {"path": checkpoint["path"], "sha256": checkpoint["sha256"]},
        name=f"{completed_stage} checkpoint",
    )
    if checkpoint["finite"] is not True or checkpoint["exact_resume_passed"] is not True:
        raise LaunchRefused(
            f"{completed_stage} checkpoint finite/exact-resume gate failed"
        )
    _validate_stage_metrics(row["metrics"])
    return digest


def _build_train_argv(
    *,
    checkout: Path,
    source_commit: str,
    isaac_python: Path,
    nosite_bootstrap: Path,
    nosite_bootstrap_sha256: str,
    nosite_import_roots: list[dict[str, Any]],
    action_set_contract: dict[str, Any],
    order: tuple[str, ...],
    bindings: list[dict[str, Any]],
    manifest_relative: str,
    manifest_sha: str,
    evaluator_relative: str,
    evaluator_file_sha: str,
    sidecar_receipt_relative: str,
    sidecar_receipt_file_sha: str,
    drain_receipt_relative: str,
    drain_receipt_file_sha: str,
    registry_relative: str,
    registry_sha: str,
    registry_alignment_sha: str,
    ready_sha: str,
    ready_fk_sha: str,
    promotion_relative: str,
    policy_sha: str,
    effective_reward_recipe_sha: str,
    seed: int,
    gpu_index: int,
    stage: str,
    stage_row: dict[str, Any],
    extra_overrides: list[str],
) -> tuple[list[str], dict[str, Any]]:
    train_script, train_relative, train_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        TRAIN_SOURCE,
        name="train.py",
    )
    entrypoint, entrypoint_relative, entrypoint_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        LAUNCHER_SOURCE,
        name="isolated training entrypoint",
    )
    import_root = (
        checkout
        / "hope_training/whole_body_tracking/source/whole_body_tracking"
    )
    _require_real_directory(import_root, name="isolated package import root")
    package_init, package_init_relative, package_init_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/__init__.py"
        ),
        name="whole_body_tracking package root",
    )
    del package_init
    motion_files = [
        str(checkout / binding["motion_path"]) for binding in bindings
    ]
    json_list = lambda values: json.dumps(  # noqa: E731
        values, separators=(",", ":"), ensure_ascii=False
    )
    identity = {
        "entrypoint_path": entrypoint_relative,
        "entrypoint_sha256": entrypoint_sha,
        "train_path": train_relative,
        "train_sha256": train_sha,
        "import_root": str(import_root),
        "package_init_path": package_init_relative,
        "package_init_sha256": package_init_sha,
        "source_commit_sha": source_commit,
    }
    entrypoint_argv = [
        "train-entrypoint",
        f"--expected-source-commit={source_commit}",
        f"--expected-entrypoint-sha256={entrypoint_sha}",
        f"--expected-train-sha256={train_sha}",
        f"--expected-import-root={import_root}",
        "--",
        f"task={TASK_PROFILE_ID}",
        "algo=ppo",
        "algo.policy.init_noise_std=0.02",
        "action_ball_shared_ready_bootstrap=true",
        "headless=true",
        "video=false",
        # CUDA_VISIBLE_DEVICES is rebuilt from the physical index by the
        # launcher, so the selected device is always logical cuda:0.
        "device=cuda:0",
        f"seed={seed}",
        f"num_envs={stage_row['num_envs']}",
        f"max_iterations={stage_row['max_iterations']}",
        f"algo.runner.save_interval={stage_row['save_interval']}",
        f"run_name={stage_row['namespace_path'].name}",
        (
            "task.experiment_name="
            f"{action_set_contract['experiment_name']}"
        ),
        (
            "expected_effective_reward_recipe_sha256="
            f"{effective_reward_recipe_sha}"
        ),
        (
            "task.actor_obs_contract="
            f"{action_set_contract['actor_obs_contract']}"
        ),
        (
            "task.rewards.full_body_mimic="
            f"{'true' if action_set_contract['scope'] == 'full' else 'false'}"
        ),
        f"motion_file={json_list(motion_files)}",
        f"task.racket.clip_names={json_list(list(order))}",
        "task.racket.target_mode=action_ball",
        f"task.racket.action_ball_manifest_path={manifest_relative}",
        f"task.racket.action_ball_manifest_sha256={manifest_sha}",
        f"task.racket.action_ball_policy_contract_sha256={policy_sha}",
        (
            "task.racket.action_ball_evaluator_launch_receipt_path="
            f"{evaluator_relative}"
        ),
        (
            "task.racket.action_ball_evaluator_launch_receipt_file_sha256="
            f"{evaluator_file_sha}"
        ),
        (
            "task.racket.action_ball_sidecar_launch_receipt_path="
            f"{sidecar_receipt_relative}"
        ),
        (
            "task.racket.action_ball_sidecar_launch_receipt_file_sha256="
            f"{sidecar_receipt_file_sha}"
        ),
        (
            "task.racket.action_ball_drain_reset_launch_receipt_path="
            f"{drain_receipt_relative}"
        ),
        (
            "task.racket.action_ball_drain_reset_launch_receipt_file_sha256="
            f"{drain_receipt_file_sha}"
        ),
        (
            "task.racket.action_ball_evaluation_inbox_root="
            f"{stage_row['evaluation_inbox_root']}"
        ),
        (
            "task.racket.action_ball_evaluation_owner_id="
            f"{stage_row['evaluation_owner_id']}"
        ),
        (
            "task.racket.action_ball_evaluation_run_id="
            f"{stage_row['evaluation_run_id']}"
        ),
        (
            "task.racket.action_ball_frozen_eval_interval_updates="
            f"{stage_row['frozen_eval_interval_updates']}"
        ),
        "task.racket.action_ball_diagnostic_unauthorized=false",
        "+task.racket.reference_guard_mode=metrics_only",
        f"task.racket.action_ball_seed={seed}",
        f"task.motion.canonical_registry_path={registry_relative}",
        f"task.motion.canonical_registry_repo_root={checkout}",
        f"task.motion.canonical_registry_sha256={registry_sha}",
        (
            "task.motion.canonical_registry_alignment_sha256="
            f"{registry_alignment_sha}"
        ),
        f"task.motion.canonical_ready_sha256={ready_sha}",
        f"task.motion.canonical_ready_fk_sha256={ready_fk_sha}",
        (
            "task.motion.canonical_promotion_certificate_path="
            f"{promotion_relative}"
        ),
        "task.racket.question_bank=",
        "task.racket.question_bank_allow_legacy=false",
        "task.racket.cq_anchor_bank=",
        "task.racket.exam_bank=",
        (
            "++training_launch_claim_path="
            f"{stage_row['namespace_path'] / 'launch_claim.json'}"
        ),
        *extra_overrides,
    ]
    module_name = "_action_ball_launcher_train_nosite_bootstrap"
    nosite = _load_source_module_without_bytecode(
        nosite_bootstrap,
        name=module_name,
        purpose="train no-site bootstrap",
    )
    try:
        command = nosite.build_exact_nosite_argv(
            python=isaac_python,
            bootstrap=nosite_bootstrap,
            bootstrap_sha256=nosite_bootstrap_sha256,
            entrypoint=entrypoint,
            entrypoint_sha256=entrypoint_sha,
            import_roots=nosite_import_roots,
            entrypoint_argv=entrypoint_argv,
        )
        nosite.validate_exact_nosite_argv(
            command.argv,
            expected_python=isaac_python,
            expected_bootstrap=command.contract["bootstrap"],
            expected_entrypoint=command.contract["entrypoint"],
            expected_import_roots=nosite_import_roots,
            expected_entrypoint_argv=entrypoint_argv,
            expected_contract_sha256=command.contract_sha256,
        )
    except Exception as exc:
        raise LaunchRefused(
            f"train no-site command construction failed: {exc}"
        ) from exc
    identity["nosite_argv_contract_sha256"] = command.contract_sha256
    identity["nosite_argv_contract"] = dict(command.contract)
    return list(command.argv), identity


def _build_sidecar_argv(
    *,
    checkout: Path,
    source_commit: str,
    isaac_python: Path,
    nosite_bootstrap: Path,
    nosite_bootstrap_sha256: str,
    nosite_import_roots: list[dict[str, Any]],
    sidecar_receipt_relative: str,
    stage_row: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    sidecar, _, sidecar_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        SIDECAR_CODE_SOURCE,
        name="formal frozen-evaluation sidecar",
    )
    entrypoint_argv = [
        "--inbox-root",
        stage_row["evaluation_inbox_root"],
        "--owner-id",
        stage_row["evaluation_owner_id"],
        "--run-id",
        stage_row["evaluation_run_id"],
        "--launch",
        str(checkout / sidecar_receipt_relative),
        "--heartbeat-interval-s",
        "5.0",
        "--request-deadline-s",
        "7200.0",
        "--backend",
        "formal",
        "--device",
        "cuda:0",
    ]
    module_name = "_action_ball_launcher_sidecar_nosite_bootstrap"
    nosite = _load_source_module_without_bytecode(
        nosite_bootstrap,
        name=module_name,
        purpose="sidecar no-site bootstrap",
    )
    try:
        command = nosite.build_exact_nosite_argv(
            python=isaac_python,
            bootstrap=nosite_bootstrap,
            bootstrap_sha256=nosite_bootstrap_sha256,
            entrypoint=sidecar,
            entrypoint_sha256=sidecar_sha,
            import_roots=nosite_import_roots,
            entrypoint_argv=entrypoint_argv,
        )
    except Exception as exc:
        raise LaunchRefused(
            f"sidecar no-site command construction failed: {exc}"
        ) from exc
    return list(command.argv), {
        "nosite_argv_contract_sha256": command.contract_sha256,
        "nosite_argv_contract": dict(command.contract),
    }


def prepare_launch_plan(spec_path_value: str | Path, stage: str) -> dict[str, Any]:
    """Validate a spec and return its immutable plan without side effects."""

    if stage not in STAGE_ORDER:
        raise LaunchRefused(
            f"stage must be one of {', '.join(STAGE_ORDER)}"
        )
    spec_path = _external_regular_file(
        str(spec_path_value), name="launch spec"
    )
    spec_file_sha = sha256_file(spec_path)
    spec = _exact_dict(
        load_strict_json(spec_path, name="launch spec"),
        _TOP_KEYS,
        name="launch spec",
    )
    if spec["schema_version"] != SCHEMA_VERSION or spec["kind"] != SPEC_KIND:
        raise LaunchRefused("launch spec schema_version/kind is invalid")
    checkout, source_commit = _verify_checkout(spec["source"])
    runtime_tool_identity = {
        "git": _trusted_system_executable("git"),
        "nvidia_smi_resolution": {
            "name": "nvidia-smi",
            "search_path": os.defpath,
            "caller_path_forbidden": True,
            "live_identity_bound_under_gpu_locks": True,
        },
    }
    action_set_contract, action_set_contract_source_sha = (
        _load_action_set_contract(
            checkout, source_commit, spec["action_set"]
        )
    )
    if spec["launch_profile"] != action_set_contract["profile_id"]:
        raise LaunchRefused(
            "launch_profile must equal the registered action-set profile"
        )
    order = tuple(action_set_contract["ordered_action_ids"])
    ordered_action_uids = tuple(action_set_contract["ordered_action_uids"])
    _, task_profile_relative, task_profile_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        TASK_PROFILE_SOURCE,
        name=f"immutable task profile {TASK_PROFILE_ID}",
    )
    task_profile_identity = {
        "profile_id": TASK_PROFILE_ID,
        "path": task_profile_relative,
        "sha256": task_profile_sha,
    }
    runtime_code_sha256: dict[str, str] = {}
    for relative in RUNTIME_CODE_SOURCES:
        _path, _relative, digest, _mode = _verify_repo_blob(
            checkout,
            source_commit,
            relative,
            name=f"runtime code {relative}",
            executable=(True if relative == KIT_LAUNCHER_SOURCE else None),
        )
        runtime_code_sha256[relative] = digest
    proposal_sampler_identity = _load_proposal_sampler_contract(
        checkout / PROPOSAL_SAMPLER_SOURCE,
        source_sha256=runtime_code_sha256[PROPOSAL_SAMPLER_SOURCE],
    )
    try:
        executing_launcher_sha = sha256_file(Path(__file__).resolve())
    except OSError as exc:
        raise LaunchRefused(
            f"executing launcher bytes cannot be inspected: {exc}"
        ) from exc
    if executing_launcher_sha != runtime_code_sha256[LAUNCHER_SOURCE]:
        raise LaunchRefused(
            "executing launcher bytes differ from the exact source commit"
        )
    if (
        runtime_code_sha256[ACTION_SET_CONTRACT_SOURCE]
        != action_set_contract_source_sha
    ):
        raise LaunchRefused(
            "action-set contract source identity changed during validation"
        )
    policy_sha = _sha256(
        spec["policy_contract_sha256"],
        name="spec.policy_contract_sha256",
    )
    inputs = _exact_dict(
        spec["inputs"],
        (
            "manifest",
            "prototype",
            "motion_admission_receipt",
            "evaluator_launch_receipt",
            "sidecar_launch_receipt",
            "drain_reset_launch_receipt",
            "canonical_registry",
            "promotion_certificate",
            "fitted_ball_profile_pins",
            "fitted_ball_launch_trust_spec",
            "fitted_ball_launch_trust_root",
            "fitted_ball_gate_receipt",
            "isaac_table_smoke_receipt",
            "stage_evaluator_authority",
            "prelaunch_safety_attestation",
        ),
        name="spec.inputs",
    )

    prototype_path, prototype_relative, prototype_sha = _verify_repo_pin(
        checkout,
        source_commit,
        inputs["prototype"],
        name="spec.inputs.prototype",
    )
    manifest_path, manifest_relative, manifest_sha = _verify_repo_pin(
        checkout,
        source_commit,
        inputs["manifest"],
        name="spec.inputs.manifest",
    )
    if (
        manifest_relative != action_set_contract["manifest_path"]
        or manifest_sha != action_set_contract["manifest_sha256"]
    ):
        raise LaunchRefused(
            "spec manifest pin differs from the code-owned action-set contract"
        )
    admission_path, admission_relative, admission_file_sha = _verify_repo_pin(
        checkout,
        source_commit,
        inputs["motion_admission_receipt"],
        name="spec.inputs.motion_admission_receipt",
    )
    evaluator_path, evaluator_relative, evaluator_file_sha = _verify_repo_pin(
        checkout,
        source_commit,
        inputs["evaluator_launch_receipt"],
        name="spec.inputs.evaluator_launch_receipt",
    )
    sidecar_receipt_path, sidecar_receipt_relative, sidecar_receipt_file_sha = (
        _verify_repo_pin(
            checkout,
            source_commit,
            inputs["sidecar_launch_receipt"],
            name="spec.inputs.sidecar_launch_receipt",
        )
    )
    drain_receipt_path, drain_receipt_relative, drain_receipt_file_sha = (
        _verify_repo_pin(
            checkout,
            source_commit,
            inputs["drain_reset_launch_receipt"],
            name="spec.inputs.drain_reset_launch_receipt",
        )
    )
    promotion_path, promotion_relative, promotion_sha = _verify_repo_pin(
        checkout,
        source_commit,
        inputs["promotion_certificate"],
        name="spec.inputs.promotion_certificate",
    )
    profile_pins_path, profile_pins_relative, profile_pins_sha = (
        _verify_repo_pin(
            checkout,
            source_commit,
            inputs["fitted_ball_profile_pins"],
            name="spec.inputs.fitted_ball_profile_pins",
        )
    )
    (
        launch_trust_spec_path,
        launch_trust_spec_relative,
        launch_trust_spec_sha,
    ) = _verify_repo_pin(
        checkout,
        source_commit,
        inputs["fitted_ball_launch_trust_spec"],
        name="spec.inputs.fitted_ball_launch_trust_spec",
    )
    (
        launch_trust_root_path,
        launch_trust_root_relative,
        launch_trust_root_sha,
    ) = _verify_repo_pin(
        checkout,
        source_commit,
        inputs["fitted_ball_launch_trust_root"],
        name="spec.inputs.fitted_ball_launch_trust_root",
    )
    fitted_gate_path, fitted_gate_sha = _verify_external_pin(
        inputs["fitted_ball_gate_receipt"],
        name="spec.inputs.fitted_ball_gate_receipt",
    )
    isaac_table_smoke_path, isaac_table_smoke_sha = _verify_external_pin(
        inputs["isaac_table_smoke_receipt"],
        name="spec.inputs.isaac_table_smoke_receipt",
    )
    authority_path, authority_relative, authority_file_sha = _verify_repo_pin(
        checkout,
        source_commit,
        inputs["stage_evaluator_authority"],
        name="spec.inputs.stage_evaluator_authority",
    )
    safety_attestation_path, safety_attestation_file_sha = _verify_external_pin(
        inputs["prelaunch_safety_attestation"],
        name="spec.inputs.prelaunch_safety_attestation",
    )
    registry_pin = _exact_dict(
        inputs["canonical_registry"],
        (
            "path",
            "sha256",
            "alignment_sha256",
            "canonical_ready_sha256",
            "canonical_ready_fk_sha256",
        ),
        name="spec.inputs.canonical_registry",
    )
    registry_sha = _sha256(
        registry_pin["sha256"],
        name="spec.inputs.canonical_registry.sha256",
    )
    registry_path, registry_relative, actual_registry_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        registry_pin["path"],
        name="spec.inputs.canonical_registry.path",
    )
    if actual_registry_sha != registry_sha:
        raise LaunchRefused("canonical registry byte SHA-256 mismatch")
    registry_alignment_sha = _sha256(
        registry_pin["alignment_sha256"],
        name="spec.inputs.canonical_registry.alignment_sha256",
    )
    ready_sha = _sha256(
        registry_pin["canonical_ready_sha256"],
        name="spec.inputs.canonical_registry.canonical_ready_sha256",
    )
    ready_fk_sha = _sha256(
        registry_pin["canonical_ready_fk_sha256"],
        name="spec.inputs.canonical_registry.canonical_ready_fk_sha256",
    )

    manifest = load_strict_json(manifest_path, name="ActionBall manifest")
    bindings, solver_sha, physics_sha = _validate_manifest(
        manifest,
        checkout=checkout,
        source_commit=source_commit,
        order=order,
        ordered_action_uids=ordered_action_uids,
        scope=action_set_contract["scope"],
        mobility_mode=action_set_contract["mobility_mode"],
        prototype_relative=prototype_relative,
        prototype_sha256=prototype_sha,
    )
    _validate_prototype(
        load_strict_json(prototype_path, name="stroke prototype"),
        order=order,
        bindings=bindings,
        scope=action_set_contract["scope"],
    )
    _validate_registry(
        load_strict_json(registry_path, name="canonical registry"),
        order=order,
        bindings=bindings,
        scope=action_set_contract["scope"],
        expected_ready_sha256=ready_sha,
        expected_ready_fk_sha256=ready_fk_sha,
    )
    admission_canonical_sha = _validate_admission_receipt(
        load_strict_json(admission_path, name="motion admission receipt"),
        order=order,
        bindings=bindings,
        scope=action_set_contract["scope"],
        mobility_mode=action_set_contract["mobility_mode"],
        registry_sha256=registry_sha,
        promotion_sha256=promotion_sha,
    )
    evaluator_canonical_sha, evaluator_identity = _validate_evaluator_receipt(
        load_strict_json(evaluator_path, name="evaluator launch receipt"),
        checkout=checkout,
        source_commit=source_commit,
        bindings=bindings,
        mobility_mode=action_set_contract["mobility_mode"],
        solver_sha256=solver_sha,
        policy_sha256=policy_sha,
    )
    (
        sidecar_canonical_sha,
        sidecar_launch_content_sha,
        sidecar_code_sha,
        sidecar_content,
    ) = _validate_sidecar_launch_receipt(
        load_strict_json(
            sidecar_receipt_path, name="sidecar launch receipt"
        ),
        checkout=checkout,
        source_commit=source_commit,
    )
    (
        drain_canonical_sha,
        drain_operational_identity,
    ) = _validate_drain_reset_launch_receipt(
        load_strict_json(
            drain_receipt_path, name="drain/reset launch receipt"
        ),
        checkout=checkout,
        source_commit=source_commit,
        bindings=bindings,
        mobility_mode=action_set_contract["mobility_mode"],
        evaluator_identity=evaluator_identity,
    )
    runtime_code_sha256[PROMOTION_TRUST_SOURCE] = _require_exact_trust(
        checkout,
        source_commit=source_commit,
        source_relative=PROMOTION_TRUST_SOURCE,
        variable=PROMOTION_TRUST_NAME,
        expected_digest=promotion_sha,
    )
    runtime_code_sha256[EVALUATOR_TRUST_SOURCE] = _require_exact_trust(
        checkout,
        source_commit=source_commit,
        source_relative=EVALUATOR_TRUST_SOURCE,
        variable=EVALUATOR_TRUST_NAME,
        expected_digest=evaluator_canonical_sha,
    )
    runtime_code_sha256[EVALUATION_INBOX_SOURCE] = _require_exact_trust(
        checkout,
        source_commit=source_commit,
        source_relative=EVALUATION_INBOX_SOURCE,
        variable=SIDECAR_CODE_TRUST_NAME,
        expected_digest=sidecar_code_sha,
    )
    inbox_source_sha = _require_exact_trust(
        checkout,
        source_commit=source_commit,
        source_relative=EVALUATION_INBOX_SOURCE,
        variable=SIDECAR_LAUNCH_TRUST_NAME,
        expected_digest=sidecar_launch_content_sha,
    )
    if inbox_source_sha != runtime_code_sha256[EVALUATION_INBOX_SOURCE]:
        raise LaunchRefused(
            "sidecar code and launch trust pins came from different inbox code"
        )
    runtime_code_sha256[SIDECAR_CODE_SOURCE] = sidecar_code_sha
    runtime_code_sha256[CURRICULUM_TRUST_SOURCE] = _require_exact_trust(
        checkout,
        source_commit=source_commit,
        source_relative=CURRICULUM_TRUST_SOURCE,
        variable=DRAIN_RESET_TRUST_NAME,
        expected_digest=drain_canonical_sha,
    )
    runtime_code_sha256[HOPE_COMMANDS_SOURCE] = (
        drain_operational_identity["runtime_source_sha256"]
    )
    # The MuJoCo fitted-ball lane and this Isaac lane must agree on the action
    # order bit for bit.  Nothing else compares them: this launcher pins the
    # fitted-ball gate receipt by path+SHA only (``_verify_external_pin``) and
    # never reads the order out of it, and ``FRESH_ORDER_SOURCE`` is not in
    # ``RUNTIME_CODE_SOURCES``.  Recording the sentinel's blob digest here is
    # how the launch spec self-reports that the comparison actually ran.
    runtime_code_sha256[FRESH_ORDER_SOURCE] = _require_fresh_order_sentinel(
        checkout, source_commit
    )
    authority_canonical_sha, stage_evaluator_public_key = (
        _validate_stage_evaluator_authority(
            load_strict_json(
                authority_path, name="frozen stage evaluator authority"
            ),
            checkout=checkout,
            source_commit=source_commit,
        )
    )
    prelaunch_safety_canonical_sha = (
        _validate_prelaunch_safety_attestation(
            load_strict_json(
                safety_attestation_path,
                name="prelaunch safety attestation",
            ),
            public_key=stage_evaluator_public_key,
            authority_file_sha=authority_file_sha,
            source_commit=source_commit,
            launch_profile=spec["launch_profile"],
            action_set_contract=action_set_contract,
            order=order,
            bindings=bindings,
            manifest_sha=manifest_sha,
            profile_pins_sha=profile_pins_sha,
            launch_trust_spec_sha=launch_trust_spec_sha,
            launch_trust_root_sha=launch_trust_root_sha,
            fitted_gate_sha=fitted_gate_sha,
            isaac_table_smoke_sha=isaac_table_smoke_sha,
        )
    )

    train = _exact_dict(
        spec["train"],
        (
            "isaac_python",
            "runtime_inventory",
            "seed",
            "extra_overrides",
            "ground_plant_contract_sha256",
            "effective_reward_recipe_sha256",
            "ppo_recipe_sha256",
        ),
        name="spec.train",
    )
    isaac_python, isaac_python_runtime = _validate_python_runtime(
        train["isaac_python"]
    )
    runtime_inventory_identity = _validate_runtime_inventory_receipt(
        train["runtime_inventory"],
        inventory_script=checkout / RUNTIME_INVENTORY_SOURCE,
        isaac_python=isaac_python,
        expected_version=isaac_python_runtime["version"],
        expected_cache_tag=isaac_python_runtime["cache_tag"],
        nosite_bootstrap_script=checkout / NOSITE_BOOTSTRAP_SOURCE,
        nosite_bootstrap_sha256=runtime_code_sha256[
            NOSITE_BOOTSTRAP_SOURCE
        ],
    )
    isaac_python_runtime = {
        **isaac_python_runtime,
        "runtime_inventory": runtime_inventory_identity,
    }
    seed = _plain_int(train["seed"], name="spec.train.seed", minimum=0)
    if seed >= (1 << 63):
        raise LaunchRefused("spec.train.seed must be < 2**63")
    extra_overrides = _validate_override_list(train["extra_overrides"])
    ground_plant_sha = _sha256(
        train["ground_plant_contract_sha256"],
        name="spec.train.ground_plant_contract_sha256",
    )
    effective_reward_recipe_sha = _sha256(
        train["effective_reward_recipe_sha256"],
        name="spec.train.effective_reward_recipe_sha256",
    )
    ppo_recipe_sha = _sha256(
        train["ppo_recipe_sha256"],
        name="spec.train.ppo_recipe_sha256",
    )
    training_recipe = {
        "seed": seed,
        "extra_overrides": extra_overrides,
        "task_profile": task_profile_identity,
        "ground_plant_contract_sha256": ground_plant_sha,
        "effective_reward_recipe_sha256": effective_reward_recipe_sha,
        "ppo_recipe_sha256": ppo_recipe_sha,
    }

    gpu_spec = _exact_dict(
        spec["gpus"], ("trainer", "evaluator"), name="spec.gpus"
    )
    gpus = {
        role: _validate_gpu_role(gpu_spec[role], role=role)
        for role in ("trainer", "evaluator")
    }
    if (
        gpus["trainer"]["uuid"] == gpus["evaluator"]["uuid"]
        or gpus["trainer"]["lock_path"] == gpus["evaluator"]["lock_path"]
    ):
        raise LaunchRefused(
            "trainer GPU0 and evaluator GPU1 must have distinct UUIDs and locks"
        )

    stages = _validate_stage_budgets(
        spec["stages"], action_set_contract
    )
    selected = stages[stage]
    selected_evaluation_runtime = _stage_evaluation_runtime(
        selected, evaluator_identity
    )
    namespace = selected["namespace_path"]
    if os.path.lexists(namespace):
        raise LaunchRefused(
            f"run namespace already exists and is permanently spent: {namespace}"
        )
    _validate_rsl_output_slot(
        checkout, namespace.name, action_set_contract["experiment_name"]
    )
    stage_gpu_identities: dict[str, dict[str, Any]] = {}
    for stage_name, stage_spec in stages.items():
        stage_gpu_identities[stage_name] = {}
        for role in ("trainer", "evaluator"):
            gpu = gpus[role]
            receipt_sha = _validate_owner_receipt(
                stage_spec[f"{role}_gpu_owner_receipt"],
                stage=stage_name,
                namespace=stage_spec["namespace_path"],
                source_commit=source_commit,
                gpu_index=gpu["index"],
                gpu_uuid=gpu["uuid"],
                owner=gpu["owner"],
                lock_path=Path(gpu["lock_path"]),
            )
            stage_gpu_identities[stage_name][role] = {
                **gpu,
                "owner_receipt_sha256": receipt_sha,
            }
    def _build_stage_train(
        stage_name: str,
    ) -> tuple[list[str], dict[str, Any]]:
        return _build_train_argv(
            checkout=checkout,
            source_commit=source_commit,
            isaac_python=isaac_python,
            nosite_bootstrap=checkout / NOSITE_BOOTSTRAP_SOURCE,
            nosite_bootstrap_sha256=runtime_code_sha256[
                NOSITE_BOOTSTRAP_SOURCE
            ],
            nosite_import_roots=runtime_inventory_identity[
                "import_roots"
            ],
            action_set_contract=action_set_contract,
            order=order,
            bindings=bindings,
            manifest_relative=manifest_relative,
            manifest_sha=manifest_sha,
            evaluator_relative=evaluator_relative,
            evaluator_file_sha=evaluator_file_sha,
            sidecar_receipt_relative=sidecar_receipt_relative,
            sidecar_receipt_file_sha=sidecar_receipt_file_sha,
            drain_receipt_relative=drain_receipt_relative,
            drain_receipt_file_sha=drain_receipt_file_sha,
            registry_relative=registry_relative,
            registry_sha=registry_sha,
            registry_alignment_sha=registry_alignment_sha,
            ready_sha=ready_sha,
            ready_fk_sha=ready_fk_sha,
            promotion_relative=promotion_relative,
            policy_sha=policy_sha,
            effective_reward_recipe_sha=effective_reward_recipe_sha,
            seed=seed,
            gpu_index=gpus["trainer"]["index"],
            stage=stage_name,
            stage_row=stages[stage_name],
            extra_overrides=extra_overrides,
        )

    argv_without_claim, isolated_training_entrypoint = _build_stage_train(stage)
    sidecar_argv, sidecar_nosite_identity = _build_sidecar_argv(
        checkout=checkout,
        source_commit=source_commit,
        isaac_python=isaac_python,
        nosite_bootstrap=checkout / NOSITE_BOOTSTRAP_SOURCE,
        nosite_bootstrap_sha256=runtime_code_sha256[
            NOSITE_BOOTSTRAP_SOURCE
        ],
        nosite_import_roots=runtime_inventory_identity[
            "import_roots"
        ],
        sidecar_receipt_relative=sidecar_receipt_relative,
        stage_row=selected,
    )
    predecessor_receipt_sha: str | None = None
    if stage != "smoke":
        previous = STAGE_ORDER[STAGE_ORDER.index(stage) - 1]
        _, previous_training_entrypoint = _build_stage_train(previous)
        if selected["predecessor_receipt"] is None:
            raise LaunchRefused(
                f"{stage} requires an exact passed {previous} receipt"
            )
        predecessor_receipt_sha = _validate_predecessor_receipt(
            selected["predecessor_receipt"],
            public_key=stage_evaluator_public_key,
            authority_file_sha=authority_file_sha,
            checkout=checkout,
            completed_stage=previous,
            completed_namespace=stages[previous]["namespace_path"],
            completed_stage_budget={
                "num_envs": stages[previous]["num_envs"],
                "max_iterations": stages[previous]["max_iterations"],
                "save_interval": stages[previous]["save_interval"],
            },
            action_set_contract=action_set_contract,
            launch_profile=spec["launch_profile"],
            source_commit=source_commit,
            order=order,
            manifest_sha=manifest_sha,
            prototype_sha=prototype_sha,
            admission_sha=admission_file_sha,
            evaluator_sha=evaluator_file_sha,
            sidecar_sha=sidecar_receipt_file_sha,
            drain_reset_sha=drain_receipt_file_sha,
            policy_sha=policy_sha,
            profile_pins_sha=profile_pins_sha,
            launch_trust_spec_sha=launch_trust_spec_sha,
            launch_trust_root_sha=launch_trust_root_sha,
            fitted_gate_sha=fitted_gate_sha,
            isaac_table_smoke_sha=isaac_table_smoke_sha,
            prelaunch_safety_attestation_sha=safety_attestation_file_sha,
            training_recipe=training_recipe,
            isaac_python_runtime=isaac_python_runtime,
            evaluation_runtime=_stage_evaluation_runtime(
                stages[previous], evaluator_identity
            ),
            gpu_roles=stage_gpu_identities[previous],
            training_entrypoint=previous_training_entrypoint,
        )

    canonical_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_ball_no_clobber_launch_payload_v3",
        "spec_path": str(spec_path),
        "spec_file_sha256": spec_file_sha,
        "launch_profile": spec["launch_profile"],
        "stage": stage,
        "source_checkout": str(checkout),
        "source_commit_sha": source_commit,
        "ordered_action_ids": list(order),
        "action_set_contract": action_set_contract,
        "manifest": {
            "path": manifest_relative,
            "sha256": manifest_sha,
        },
        "prototype": {
            "path": prototype_relative,
            "sha256": prototype_sha,
            "scope": action_set_contract["scope"],
        },
        "motion_admission_receipt": {
            "path": admission_relative,
            "file_sha256": admission_file_sha,
            "canonical_sha256": admission_canonical_sha,
        },
        "evaluator_launch_receipt": {
            "path": evaluator_relative,
            "file_sha256": evaluator_file_sha,
            "canonical_sha256": evaluator_canonical_sha,
        },
        "sidecar_launch_receipt": {
            "path": sidecar_receipt_relative,
            "file_sha256": sidecar_receipt_file_sha,
            "canonical_sha256": sidecar_canonical_sha,
            "content_sha256": sidecar_launch_content_sha,
            "sidecar_code_sha256": sidecar_code_sha,
            "backend_contract_sha256": sidecar_content[
                "backend_contract_sha256"
            ],
            "protocol_contract_sha256": sidecar_content[
                "protocol_contract_sha256"
            ],
            "policy_evaluation_contract_sha256": sidecar_content[
                "policy_evaluation_contract_sha256"
            ],
            "resolved_recipe_contract_sha256": sidecar_content[
                "resolved_recipe_contract_sha256"
            ],
            "runtime_identity_contract_sha256": sidecar_content[
                "runtime_identity_contract_sha256"
            ],
            "heartbeat_contract": sidecar_content[
                "heartbeat_contract"
            ],
        },
        "drain_reset_launch_receipt": {
            "path": drain_receipt_relative,
            "file_sha256": drain_receipt_file_sha,
            "canonical_sha256": drain_canonical_sha,
            "operational_identity": drain_operational_identity,
        },
        "canonical_registry": {
            "path": registry_relative,
            "sha256": registry_sha,
            "alignment_sha256": registry_alignment_sha,
            "canonical_ready_sha256": ready_sha,
            "canonical_ready_fk_sha256": ready_fk_sha,
        },
        "promotion_certificate": {
            "path": promotion_relative,
            "sha256": promotion_sha,
        },
        "fitted_ball_profile_pins": {
            "path": profile_pins_relative,
            "sha256": profile_pins_sha,
        },
        "fitted_ball_profile_pins_sha256": profile_pins_sha,
        "fitted_ball_launch_trust_spec": {
            "path": launch_trust_spec_relative,
            "sha256": launch_trust_spec_sha,
        },
        "fitted_ball_launch_trust_spec_sha256": launch_trust_spec_sha,
        "fitted_ball_launch_trust_root": {
            "path": launch_trust_root_relative,
            "sha256": launch_trust_root_sha,
        },
        "fitted_ball_launch_trust_root_sha256": launch_trust_root_sha,
        "fitted_ball_gate_receipt": {
            "path": str(fitted_gate_path),
            "sha256": fitted_gate_sha,
        },
        "fitted_ball_gate_receipt_sha256": fitted_gate_sha,
        "isaac_table_smoke_receipt": {
            "path": str(isaac_table_smoke_path),
            "sha256": isaac_table_smoke_sha,
        },
        "isaac_table_smoke_receipt_sha256": isaac_table_smoke_sha,
        "stage_evaluator_authority": {
            "path": authority_relative,
            "file_sha256": authority_file_sha,
            "canonical_sha256": authority_canonical_sha,
        },
        "prelaunch_safety_attestation": {
            "path": str(safety_attestation_path),
            "file_sha256": safety_attestation_file_sha,
            "canonical_sha256": prelaunch_safety_canonical_sha,
        },
        "prelaunch_safety_attestation_sha256": (
            safety_attestation_file_sha
        ),
        "runtime_code_sha256": dict(sorted(runtime_code_sha256.items())),
        "runtime_tool_identity": runtime_tool_identity,
        "proposal_sampler_contract_sha256": proposal_sampler_identity[
            "contract_sha256"
        ],
        "proposal_sampler": proposal_sampler_identity,
        "runtime_bootstrap": {
            "source_path": RUNTIME_BOOTSTRAP_SOURCE,
            "source_sha256": runtime_code_sha256[
                RUNTIME_BOOTSTRAP_SOURCE
            ],
        },
        "ppo_runner": {
            "source_path": PPO_RUNNER_SOURCE,
            "source_sha256": runtime_code_sha256[PPO_RUNNER_SOURCE],
        },
        "isaac_python_runtime": isaac_python_runtime,
        "isolated_training_entrypoint": isolated_training_entrypoint,
        "sidecar_nosite_execution": sidecar_nosite_identity,
        "training_recipe": training_recipe,
        "training_recipe_sha256": canonical_sha256(training_recipe),
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "policy_contract_sha256": policy_sha,
        "frozen_evaluation_runtime": selected_evaluation_runtime,
        "namespace": str(namespace),
        "gpus": stage_gpu_identities[stage],
        "stage_budget": {
            "num_envs": selected["num_envs"],
            "max_iterations": selected["max_iterations"],
            "save_interval": selected["save_interval"],
        },
        "predecessor_receipt_sha256": predecessor_receipt_sha,
        "argv_without_launch_claim": argv_without_claim,
        "sidecar_argv": sidecar_argv,
        "promotable": True,
        "fresh_start": True,
    }
    launch_claim_sha = canonical_sha256(canonical_payload)
    try:
        module_name = "_action_ball_launcher_claim_nosite_bootstrap"
        nosite = _load_source_module_without_bytecode(
            checkout / NOSITE_BOOTSTRAP_SOURCE,
            name=module_name,
            purpose="claim no-site bootstrap",
        )
        base_command = nosite.validate_exact_nosite_argv(
            argv_without_claim,
            expected_python=isaac_python,
            verify_live=True,
        )
        contract = base_command.contract
        claim_command = nosite.build_exact_nosite_argv(
            python=Path(argv_without_claim[0]),
            bootstrap=Path(contract["bootstrap"]["path"]),
            bootstrap_sha256=contract["bootstrap"]["sha256"],
            entrypoint=Path(contract["entrypoint"]["path"]),
            entrypoint_sha256=contract["entrypoint"]["sha256"],
            import_roots=contract["import_roots"],
            entrypoint_argv=[
                *contract["entrypoint_argv"],
                (
                    "++training_launch_claim_sha256="
                    f"{launch_claim_sha}"
                ),
            ],
        )
        argv = list(claim_command.argv)
    except Exception as exc:
        raise LaunchRefused(
            f"claim-bound no-site trainer argv construction failed: {exc}"
        ) from exc
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": launch_claim_sha,
        "canonical_payload": canonical_payload,
        "argv": argv,
        "confirmation_claim_sha256": launch_claim_sha,
    }


def acquire_gpu_lock(lock_path: Path) -> int:
    """Acquire one pre-existing shared GPU lifetime lock without mutation.

    A missing lock is not an invitation to manufacture a private scheduling
    universe.  Pod operations create the two well-known queue locks before
    launch; the launcher only opens those exact regular files, proves the
    pathname still names the opened inode, and then takes a non-blocking
    lifetime flock.
    """

    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before = lock_path.lstat()
    except OSError as exc:
        raise LaunchRefused(
            f"GPU lock must already exist and be inspectable: {lock_path}: {exc}"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise LaunchRefused(
            f"GPU lock must be a pre-existing regular non-symlink file: {lock_path}"
        )
    try:
        fd = os.open(lock_path, flags)
    except OSError as exc:
        raise LaunchRefused(
            f"cannot safely open GPU lock {lock_path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(fd)
        after = lock_path.lstat()
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(after.st_mode):
            raise LaunchRefused("GPU lock must be a regular file")
        if (
            (before.st_dev, before.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (after.st_dev, after.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise LaunchRefused(
                f"GPU lock pathname changed while opening: {lock_path}"
            )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LaunchRefused(
                f"GPU lifetime lock is already owned: {lock_path}"
            ) from exc
        locked = os.fstat(fd)
        try:
            locked_path = lock_path.lstat()
        except OSError as exc:
            raise LaunchRefused(
                f"GPU lock pathname vanished after flock: {lock_path}"
            ) from exc
        if (
            not stat.S_ISREG(locked.st_mode)
            or not stat.S_ISREG(locked_path.st_mode)
            or locked.st_nlink < 1
            or locked_path.st_nlink < 1
            or (locked.st_dev, locked.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (locked_path.st_dev, locked_path.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise LaunchRefused(
                f"GPU lock pathname changed before ownership was bound: {lock_path}"
            )
        return fd
    except BaseException:
        os.close(fd)
        raise


def _verify_live_gpu_empty(gpu_index: int, gpu_uuid: str) -> dict[str, Any]:
    nvidia_smi = _trusted_system_executable("nvidia-smi")
    probe_environment = {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
    }
    identity = subprocess.run(
        [
            nvidia_smi["path"],
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=probe_environment,
    )
    if identity.returncode != 0:
        raise LaunchRefused(
            f"nvidia-smi GPU identity query failed: {identity.stderr.strip()}"
        )
    observed: dict[int, str] = {}
    for line in identity.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            raise LaunchRefused(
                f"unparseable nvidia-smi GPU identity row: {line!r}"
            )
        observed[int(parts[0])] = parts[1]
    if observed.get(gpu_index) != gpu_uuid:
        raise LaunchRefused(
            f"GPU UUID mismatch for index {gpu_index}: "
            f"expected={gpu_uuid}, actual={observed.get(gpu_index)!r}"
        )
    occupancy = subprocess.run(
        [
            nvidia_smi["path"],
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=probe_environment,
    )
    if occupancy.returncode != 0:
        raise LaunchRefused(
            f"nvidia-smi compute query failed: {occupancy.stderr.strip()}"
        )
    rows = [line.strip() for line in occupancy.stdout.splitlines() if line.strip()]
    for line in rows:
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3 or not parts[0].startswith("GPU-") or not parts[1].isdigit():
            raise LaunchRefused(
                f"unparseable nvidia-smi compute row: {line!r}"
            )
        if parts[0] == gpu_uuid:
            raise LaunchRefused(
                f"target GPU is occupied by pid={parts[1]} process={parts[2]!r}"
            )
    return {
        "nvidia_smi": nvidia_smi,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "compute_process_count": 0,
    }


def _write_live_gpu_admission(
    plan: dict[str, Any],
    checks: dict[str, Any],
) -> str:
    payload = plan["canonical_payload"]
    namespace = Path(payload["namespace"])
    rows = _exact_dict(
        checks, ("trainer", "evaluator"), name="live GPU admission checks"
    )
    normalized: dict[str, Any] = {}
    binary_identity: dict[str, str] | None = None
    for role in ("trainer", "evaluator"):
        row = _exact_dict(
            rows[role],
            (
                "nvidia_smi",
                "gpu_index",
                "gpu_uuid",
                "compute_process_count",
            ),
            name=f"live GPU admission {role}",
        )
        tool = _exact_dict(
            row["nvidia_smi"],
            ("name", "requested_path", "path", "sha256"),
            name=f"live GPU admission {role} nvidia-smi",
        )
        _sha256(
            tool["sha256"],
            name=f"live GPU admission {role} nvidia-smi SHA",
        )
        expected_gpu = payload["gpus"][role]
        if (
            tool["name"] != "nvidia-smi"
            or type(tool["path"]) is not str
            or not Path(tool["path"]).is_absolute()
            or row["gpu_index"] != expected_gpu["index"]
            or row["gpu_uuid"] != expected_gpu["uuid"]
            or row["compute_process_count"] != 0
        ):
            raise LaunchRefused(
                f"live GPU admission {role} differs from the exact empty role"
            )
        if binary_identity is None:
            binary_identity = dict(tool)
        elif tool != binary_identity:
            raise LaunchRefused(
                "trainer/evaluator GPU probes used different nvidia-smi bytes"
            )
        normalized[role] = row
    assert binary_identity is not None
    receipt = {
        "schema_version": 1,
        "kind": "action_ball_live_gpu_admission",
        "admitted_utc": _utc_now(),
        "launch_claim_sha256": plan["launch_claim_sha256"],
        "source_commit_sha": payload["source_commit_sha"],
        "stage": payload["stage"],
        "namespace": str(namespace),
        "gpu_roles": payload["gpus"],
        "nvidia_smi": binary_identity,
        "checks": normalized,
    }
    path = namespace / "live_gpu_admission.json"
    _write_exclusive_json(path, receipt)
    document, digest = _read_canonical_namespace_json(
        path,
        namespace=namespace,
        name="live GPU admission receipt",
    )
    if document != receipt:
        raise LaunchRefused("live GPU admission receipt changed after write")
    return digest


def _write_exclusive_json(path: Path, value: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise LaunchRefused(
            f"exclusive receipt creation failed for {path}: {exc}"
        ) from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(_canonical_bytes(value))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_canonical_namespace_json_once(
    path: Path,
    *,
    namespace: Path,
    name: str,
) -> tuple[dict[str, Any], str]:
    """Read one stable, ordinary, canonical JSON receipt in our namespace."""

    _assert_no_symlink_components(path, start=namespace, name=name)
    try:
        before = path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be inspected: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise LaunchRefused(
            f"{name} must be a single-link regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be opened safely: {exc}") from exc
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
        opened = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after = path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"{name} vanished while reading") from exc
    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    stable_opened = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or opened.st_nlink != 1
        or after.st_nlink != 1
        or stable_before != stable_opened
        or stable_after != stable_opened
    ):
        raise LaunchRefused(f"{name} changed while reading")
    raw = b"".join(chunks)
    value = _load_strict_json_bytes(raw, name=name)
    if type(value) is not dict:
        raise LaunchRefused(f"{name} must be a plain JSON object")
    if raw != _canonical_bytes(value) + b"\n":
        raise LaunchRefused(f"{name} is not canonical JSON")
    return value, hashlib.sha256(raw).hexdigest()


def _read_canonical_namespace_json(
    path: Path,
    *,
    namespace: Path,
    name: str,
) -> tuple[dict[str, Any], str]:
    """Wait briefly for an O_EXCL producer to finish its visible file write."""

    last_error: LaunchRefused | None = None
    for _attempt in range(25):
        try:
            return _read_canonical_namespace_json_once(
                path,
                namespace=namespace,
                name=name,
            )
        except LaunchRefused as exc:
            detail = str(exc)
            if (
                "symlink" in detail
                or "escapes its trusted root" in detail
                or "regular non-symlink" in detail
            ):
                raise
            last_error = exc
            time.sleep(0.01)
    assert last_error is not None
    raise last_error


def _claim_namespace(plan: dict[str, Any]) -> Path:
    namespace = Path(plan["canonical_payload"]["namespace"])
    try:
        os.mkdir(namespace, 0o700)
    except FileExistsError as exc:
        raise LaunchRefused(
            f"run namespace already exists and is permanently spent: {namespace}"
        ) from exc
    except OSError as exc:
        raise LaunchRefused(
            f"cannot atomically claim run namespace {namespace}: {exc}"
        ) from exc
    try:
        _write_exclusive_json(namespace / "launch_claim.json", plan)
    except BaseException:
        # A partially claimed namespace is intentionally retained and spent.
        raise
    return namespace


def _load_exact_process_group(path: Path) -> Any:
    name = "_action_ball_launcher_exact_process_group"
    return _load_source_module_without_bytecode(
        path,
        name=name,
        purpose="the exact process-group helper",
        retain_in_sys_modules=True,
    )


def _load_proposal_sampler_contract(
    path: Path, *, source_sha256: str
) -> dict[str, Any]:
    """Load the committed stdlib-only sampler and recompute its public seal."""

    name = "_action_ball_launcher_proposal_sampler"
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise LaunchRefused("cannot load the exact proposal sampler source")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
        factory = getattr(
            module, "frozen_evaluation_proposal_sampler_contract", None
        )
        if not callable(factory):
            raise LaunchRefused(
                "proposal sampler lacks its frozen-evaluation contract factory"
            )
        value = factory()
    except LaunchRefused:
        raise
    except BaseException as exc:
        raise LaunchRefused(
            f"proposal sampler contract factory failed: {exc}"
        ) from exc
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        sys.modules.pop(name, None)
    row = _exact_dict(
        value,
        ("payload", "sha256"),
        name="proposal sampler contract",
    )
    payload = _exact_dict(
        row["payload"],
        (
            "schema_version",
            "kind",
            "random_access",
            "training_state_isolation",
            "sampling_core",
            "mixture",
            "frontier",
            "proposal_accounting",
            "sampling_schema_version",
            "arm_catalog_sha256",
            "draws_per_birth",
            "draws_per_sample",
            "implementation_source_sha256",
        ),
        name="proposal sampler contract payload",
    )
    digest = _sha256(row["sha256"], name="proposal sampler contract sha256")
    if (
        payload["schema_version"] != 1
        or payload["kind"]
        != "action_ball_frozen_evaluation_proposal_sampler"
        or payload["implementation_source_sha256"] != source_sha256
        or canonical_sha256(payload) != digest
    ):
        raise LaunchRefused(
            "proposal sampler contract is not canonically bound to its exact "
            "source bytes"
        )
    _sha256(
        payload["arm_catalog_sha256"],
        name="proposal sampler arm_catalog_sha256",
    )
    for field in ("draws_per_birth", "draws_per_sample"):
        _plain_int(
            payload[field],
            name=f"proposal sampler {field}",
            minimum=1,
        )
    for field in (
        "random_access",
        "training_state_isolation",
        "sampling_core",
        "mixture",
        "frontier",
        "proposal_accounting",
    ):
        if type(payload[field]) is not str or not payload[field]:
            raise LaunchRefused(
                f"proposal sampler contract {field} must be a non-empty string"
            )
    return {
        "source_path": PROPOSAL_SAMPLER_SOURCE,
        "source_sha256": source_sha256,
        "contract_sha256": digest,
        "contract_payload": payload,
    }


def _supervisor_proc_identity(
    *,
    pid: int,
    expected_executable: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Bind the still-gated supervisor wrapper to its exact Linux runtime."""

    proc = Path("/proc") / str(pid)
    exe = proc / "exe"
    cgroup = proc / "cgroup"
    try:
        first_target = os.readlink(exe)
        first_cgroup = cgroup.read_bytes()
        second_target = os.readlink(exe)
        second_cgroup = cgroup.read_bytes()
    except OSError as exc:
        raise LaunchRefused(
            "cannot bind gated supervisor /proc executable/cgroup identity"
        ) from exc
    if (
        first_target != second_target
        or first_cgroup != second_cgroup
        or not first_target
        or first_target.endswith(" (deleted)")
    ):
        raise LaunchRefused(
            "gated supervisor executable/cgroup identity changed while binding"
        )
    observed = Path(first_target)
    if not observed.is_absolute():
        raise LaunchRefused("gated supervisor /proc executable is not absolute")
    try:
        observed = observed.resolve(strict=True)
        expected = expected_executable.resolve(strict=True)
    except OSError as exc:
        raise LaunchRefused(
            "gated supervisor executable target cannot be resolved"
        ) from exc
    actual_sha = sha256_file(observed)
    if observed != expected or actual_sha != expected_sha256:
        raise LaunchRefused(
            "gated supervisor executable differs from the frozen Isaac Python"
        )
    return {
        "executable_path": str(observed),
        "executable_sha256": actual_sha,
        "cgroup_sha256": hashlib.sha256(first_cgroup).hexdigest(),
    }


def _validate_initial_sidecar_heartbeat(
    value: Any,
    *,
    payload: dict[str, Any],
    evaluator_pid: int,
) -> dict[str, Any]:
    row = _exact_dict(
        value,
        ("schema_version", "kind", "content", "content_sha256"),
        name="dual-GPU supervisor initial sidecar heartbeat",
    )
    if (
        row["schema_version"] != 1
        or row["kind"]
        != "whole_body_tracking.action_ball.formal_sidecar_heartbeat"
    ):
        raise LaunchRefused("initial sidecar heartbeat schema/kind is invalid")
    content = _exact_dict(
        row["content"],
        _SIDECAR_HEARTBEAT_CONTENT_KEYS,
        name="dual-GPU supervisor initial sidecar heartbeat content",
    )
    content_sha = _sha256(
        row["content_sha256"],
        name="initial sidecar heartbeat content_sha256",
    )
    if canonical_sha256(content) != content_sha:
        raise LaunchRefused("initial sidecar heartbeat content SHA drifted")
    expected_identity = {
        "owner_id": payload["frozen_evaluation_runtime"]["owner_id"],
        "run_id": payload["frozen_evaluation_runtime"]["run_id"],
        "pid": evaluator_pid,
        "sidecar_code_sha256": payload["sidecar_launch_receipt"][
            "sidecar_code_sha256"
        ],
        "launch_sha256": payload["sidecar_launch_receipt"][
            "content_sha256"
        ],
        "backend_contract_sha256": payload["sidecar_launch_receipt"][
            "backend_contract_sha256"
        ],
    }
    if any(content[key] != expected for key, expected in expected_identity.items()):
        raise LaunchRefused(
            "initial sidecar heartbeat identity differs from exact launch"
        )
    if (
        type(content["heartbeat_seq"]) is not int
        or content["heartbeat_seq"] < 0
        or content["phase"] not in {"ready", "waiting_for_request_or_ack"}
        or type(content["heartbeat_unix_ns"]) is not int
        or content["heartbeat_unix_ns"] < 1
        or type(content["heartbeat_monotonic_ns"]) is not int
        or content["heartbeat_monotonic_ns"] < 1
    ):
        raise LaunchRefused("initial sidecar heartbeat progress is invalid")
    now_ns = time.monotonic_ns()
    stale_ns = int(
        SIDECAR_HEARTBEAT_CONTRACT[
            "heartbeat_stale_after_seconds"
        ]
        * 1_000_000_000
    )
    if (
        content["heartbeat_monotonic_ns"] > now_ns
        or now_ns - content["heartbeat_monotonic_ns"] > stale_ns
    ):
        raise LaunchRefused("initial sidecar heartbeat is stale or from the future")
    if (
        content["request_seq"] is not None
        or content["request_sha256"] != ""
        or content["attempts_completed"] != 0
        or content["attempts_total"] != 0
        or content["request_started_unix_ns"] != 0
        or content["request_started_monotonic_ns"] != 0
        or content["request_deadline_unix_ns"] != 0
        or content["request_deadline_monotonic_ns"] != 0
        or content["error_type"] != ""
    ):
        raise LaunchRefused(
            "initial sidecar heartbeat retains active request state"
        )
    return row


def _validate_supervisor_ready_receipt(
    value: Any, *, plan: dict[str, Any], namespace: Path
) -> dict[str, Any]:
    payload = plan["canonical_payload"]
    row = _exact_dict(
        value,
        (
            "schema_version",
            "kind",
            "ready_utc",
            "claim_sha256",
            "source_commit_sha",
            "stage",
            "namespace",
            "gpu_roles",
            "sidecar_ready",
            "sidecar_heartbeat_initial",
            "trainer_learning_line",
            "processes",
            "logs",
        ),
        name="dual-GPU supervisor ready receipt",
    )
    expected_scalars = {
        "schema_version": 1,
        "kind": "action_ball_stage_supervisor_ready",
        "claim_sha256": plan["launch_claim_sha256"],
        "source_commit_sha": payload["source_commit_sha"],
        "stage": payload["stage"],
        "namespace": str(namespace),
        "gpu_roles": payload["gpus"],
    }
    for key, expected in expected_scalars.items():
        if row[key] != expected:
            raise LaunchRefused(
                f"dual-GPU supervisor ready receipt {key} is not claim-bound"
            )
    if type(row["ready_utc"]) is not str or not row["ready_utc"]:
        raise LaunchRefused("dual-GPU supervisor ready_utc is invalid")
    expected_sidecar_ready = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.formal_sidecar_ready",
        "owner_id": payload["frozen_evaluation_runtime"]["owner_id"],
        "run_id": payload["frozen_evaluation_runtime"]["run_id"],
        "backend": "formal",
        "device": "cuda:0",
        "launch_receipt_canonical_sha256": payload[
            "sidecar_launch_receipt"
        ]["content_sha256"],
    }
    if row["sidecar_ready"] != expected_sidecar_ready:
        raise LaunchRefused(
            "dual-GPU supervisor sidecar ready identity differs from the claim"
        )
    learning_line = row["trainer_learning_line"]
    if (
        type(learning_line) is not str
        or re.search(
            r"Learning iteration[ \t]+[0-9]+/[0-9]+", learning_line
        )
        is None
    ):
        raise LaunchRefused(
            "dual-GPU supervisor did not bind an exact trainer learning marker"
        )
    processes = _exact_dict(
        row["processes"],
        ("evaluator", "trainer"),
        name="dual-GPU supervisor ready processes",
    )
    for role in ("evaluator", "trainer"):
        snapshot = _exact_dict(
            processes[role],
            (
                "pid",
                "pgid",
                "starttime_ticks",
                "argv_sha256",
                "returncode",
                "leader_receipt",
                "leader_receipt_sha256",
                "term_receipt",
                "term_receipt_sha256",
                "kill_receipt",
                "kill_receipt_sha256",
            ),
            name=f"dual-GPU supervisor ready {role} process",
        )
        if (
            type(snapshot["pid"]) is not int
            or snapshot["pid"] <= 0
            or snapshot["pgid"] != snapshot["pid"]
            or type(snapshot["starttime_ticks"]) is not int
            or snapshot["starttime_ticks"] <= 0
            or snapshot["argv_sha256"]
            != canonical_sha256(
                (
                    payload["sidecar_argv"]
                    if role == "evaluator"
                    else plan["argv"]
                )
            )
            or snapshot["returncode"] is not None
            or snapshot["leader_receipt"]
            != str(namespace / f"{role}_leader_identity.json")
            or type(snapshot["leader_receipt_sha256"]) is not str
            or SHA256_RE.fullmatch(snapshot["leader_receipt_sha256"]) is None
            or snapshot["term_receipt"] != ""
            or snapshot["term_receipt_sha256"] != ""
            or snapshot["kill_receipt"] != ""
            or snapshot["kill_receipt_sha256"] != ""
        ):
            raise LaunchRefused(
                f"dual-GPU supervisor ready {role} process identity is invalid"
            )
        leader_document, leader_file_sha = _read_canonical_namespace_json(
            Path(snapshot["leader_receipt"]),
            namespace=namespace,
            name=f"dual-GPU supervisor ready {role} leader identity",
        )
        leader_row = _exact_dict(
            leader_document,
            ("schema_version", "kind", "leader"),
            name=f"dual-GPU supervisor ready {role} leader identity",
        )
        leader = _exact_dict(
            leader_row["leader"],
            ("pid", "pgid", "starttime_ticks"),
            name=f"dual-GPU supervisor ready {role} leader",
        )
        if (
            leader_row["schema_version"] != 1
            or leader_row["kind"] != "leader_identity"
            or leader
            != {
                "pid": snapshot["pid"],
                "pgid": snapshot["pgid"],
                "starttime_ticks": snapshot["starttime_ticks"],
            }
            or leader_file_sha != snapshot["leader_receipt_sha256"]
        ):
            raise LaunchRefused(
                f"dual-GPU supervisor ready {role} leader receipt drifted"
            )
    _validate_initial_sidecar_heartbeat(
        row["sidecar_heartbeat_initial"],
        payload=payload,
        evaluator_pid=processes["evaluator"]["pid"],
    )
    if row["logs"] != {
        "evaluator": str(namespace / "evaluator.log"),
        "trainer": str(namespace / "train.log"),
    }:
        raise LaunchRefused("dual-GPU supervisor log paths are not namespace-bound")
    return row


def _validate_supervisor_accept_ack(
    value: Any,
    *,
    launch_claim_sha256: str,
    supervisor_ready_sha256: str,
    accept_intent_sha256: str,
    live_gpu_admission_sha256: str,
) -> dict[str, Any]:
    row = _exact_dict(
        value,
        (
            "schema_version",
            "kind",
            "launch_claim_sha256",
            "supervisor_ready_sha256",
            "accept_intent_sha256",
            "live_gpu_admission_sha256",
        ),
        name="dual-GPU supervisor acceptance ack",
    )
    expected = {
        "schema_version": 1,
        "kind": "action_ball_stage_supervisor_accept_ack",
        "launch_claim_sha256": launch_claim_sha256,
        "supervisor_ready_sha256": supervisor_ready_sha256,
        "accept_intent_sha256": accept_intent_sha256,
        "live_gpu_admission_sha256": live_gpu_admission_sha256,
    }
    if row != expected:
        raise LaunchRefused(
            "dual-GPU supervisor acceptance ack is not "
            "claim/ready/intent-bound"
        )
    return row


def _validate_supervisor_launch_commit_ack(
    value: Any,
    *,
    launch_claim_sha256: str,
    supervisor_ready_sha256: str,
    accept_intent_sha256: str,
    supervisor_accept_ack_sha256: str,
    launch_accepted_sha256: str,
    live_gpu_admission_sha256: str,
    ready_processes: Any,
) -> dict[str, Any]:
    row = _exact_dict(
        value,
        (
            "schema_version",
            "kind",
            "launch_claim_sha256",
            "supervisor_ready_sha256",
            "accept_intent_sha256",
            "supervisor_accept_ack_sha256",
            "launch_accepted_sha256",
            "live_gpu_admission_sha256",
            "processes",
        ),
        name="dual-GPU supervisor launch commit ack",
    )
    expected = {
        "schema_version": 1,
        "kind": "action_ball_stage_supervisor_launch_commit_ack",
        "launch_claim_sha256": launch_claim_sha256,
        "supervisor_ready_sha256": supervisor_ready_sha256,
        "accept_intent_sha256": accept_intent_sha256,
        "supervisor_accept_ack_sha256": supervisor_accept_ack_sha256,
        "launch_accepted_sha256": launch_accepted_sha256,
        "live_gpu_admission_sha256": live_gpu_admission_sha256,
        "processes": ready_processes,
    }
    if row != expected:
        raise LaunchRefused(
            "dual-GPU supervisor launch commit ack is not bound to the exact "
            "claim/ready/intent/ack/accepted/process identities"
        )
    return row


def _cancel_and_reap_supervisor(
    *,
    process: subprocess.Popen[Any],
    exact: Any,
    leader_receipt: Path,
    namespace: Path,
    gate_write_fd: int,
    control_write_fd: int,
    gate_released: bool,
    gpus: dict[str, Any],
) -> None:
    """Cancel one exact supervisor and prove both GPU workloads are gone."""

    if gate_write_fd >= 0:
        os.close(gate_write_fd)
        gate_write_fd = -1
    if control_write_fd >= 0:
        try:
            os.write(control_write_fd, b"C")
        except (BrokenPipeError, OSError):
            # Exact process-group identity below is authoritative.  A closed
            # pipe merely means the supervisor already exited.
            pass
        finally:
            os.close(control_write_fd)
            control_write_fd = -1
    term_receipt = namespace / "supervisor_pre_term_identity.json"
    if process.poll() is None:
        try:
            exact.term_group(
                Path("/proc"), leader_receipt, term_receipt
            )
        except BaseException as exc:
            # Before gate release, EOF is sufficient to make the tiny wrapper
            # exit without executing the supervisor.  After release, inability
            # to address the exact group means closure is unknown and no
            # closed-failure receipt may be claimed.
            if gate_released and process.poll() is None:
                raise LaunchClosureUnknown(
                    "cannot address the exact live supervisor for cancellation"
                ) from exc
    if process.poll() is None:
        try:
            process.wait(timeout=180.0)
        except subprocess.TimeoutExpired as exc:
            # Do not SIGKILL the supervisor: its two children live in separate
            # exact groups, so killing the cleanup owner could orphan them.
            # The still-live supervisor retains its inherited GPU locks while
            # a human investigates a blocked cleanup.
            raise LaunchClosureUnknown(
                "cancelled supervisor did not close its exact child groups "
                "within 180 seconds; cleanup owner and locks remain live"
            ) from exc
    else:
        process.wait()
    if term_receipt.is_file():
        try:
            residual = exact.verify_residual(Path("/proc"), term_receipt)
        except BaseException as exc:
            raise LaunchClosureUnknown(
                "cannot prove the cancelled supervisor group is empty"
            ) from exc
        if residual:
            raise LaunchClosureUnknown(
                "cancelled supervisor process group still has residual members"
            )
    else:
        try:
            residual = exact.group_snapshot(Path("/proc"), process.pid)
        except BaseException as exc:
            raise LaunchClosureUnknown(
                "cannot prove the unreleased supervisor wrapper is gone"
            ) from exc
        if residual:
            raise LaunchClosureUnknown(
                "unreleased supervisor wrapper still has residual members"
            )
    try:
        for role in ("trainer", "evaluator"):
            gpu = gpus[role]
            _verify_live_gpu_empty(gpu["index"], gpu["uuid"])
    except LaunchRefused as exc:
        raise LaunchClosureUnknown(
            "supervisor exited but its two GPU workloads are not proven absent"
        ) from exc


def _start_stage_supervisor(
    plan: dict[str, Any],
    *,
    trainer_lock_fd: int,
    evaluator_lock_fd: int,
    live_gpu_admission_sha256: str,
) -> dict[str, Any]:
    payload = plan["canonical_payload"]
    checkout = Path(payload["source_checkout"])
    source_commit = payload["source_commit_sha"]
    supervisor, _, supervisor_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        STAGE_SUPERVISOR_SOURCE,
        name="dual-GPU stage supervisor",
    )
    exact_path, _, _, _ = _verify_repo_blob(
        checkout,
        source_commit,
        PROCESS_GROUP_SOURCE,
        name="exact supervisor process-group helper",
    )
    exact = _load_exact_process_group(exact_path)
    namespace = Path(payload["namespace"])
    live_gpu_admission_sha = _sha256(
        live_gpu_admission_sha256,
        name="live GPU admission SHA-256",
    )
    _live_gpu_document, observed_live_gpu_sha = (
        _read_canonical_namespace_json(
            namespace / "live_gpu_admission.json",
            namespace=namespace,
            name="live GPU admission receipt",
        )
    )
    if observed_live_gpu_sha != live_gpu_admission_sha:
        raise LaunchRefused("live GPU admission receipt SHA drifted")
    log_path = namespace / "supervisor.log"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        log_fd = os.open(log_path, flags, 0o600)
    except OSError as exc:
        raise LaunchRefused(
            f"cannot create no-clobber supervisor log: {exc}"
        ) from exc
    gate_program = (
        "import os,sys;"
        "fd=int(sys.argv[1]);"
        "token=os.read(fd,2);"
        "os.close(fd);"
        "command=sys.argv[2:];"
        "sys.exit(125) if token!=b'G' else os.execv(command[0],command)"
    )
    if hasattr(os, "pipe2"):
        gate_read_fd, gate_write_fd = os.pipe2(
            getattr(os, "O_CLOEXEC", 0)
        )
    else:
        gate_read_fd, gate_write_fd = os.pipe()
    if hasattr(os, "pipe2"):
        control_read_fd, control_write_fd = os.pipe2(
            getattr(os, "O_CLOEXEC", 0)
        )
    else:
        control_read_fd, control_write_fd = os.pipe()
    supervisor_args = [
        "run",
        "--claim",
        str(namespace / "launch_claim.json"),
        "--claim-sha256",
        plan["launch_claim_sha256"],
        "--trainer-lock-fd",
        str(trainer_lock_fd),
        "--evaluator-lock-fd",
        str(evaluator_lock_fd),
        "--launcher-control-fd",
        str(control_read_fd),
    ]
    bootstrap_path = checkout / NOSITE_BOOTSTRAP_SOURCE
    bootstrap_sha = payload["runtime_code_sha256"][
        NOSITE_BOOTSTRAP_SOURCE
    ]
    import_roots = payload["isaac_python_runtime"][
        "runtime_inventory"
    ]["import_roots"]
    module_name = "_action_ball_launcher_supervisor_nosite_bootstrap"
    nosite = _load_source_module_without_bytecode(
        bootstrap_path,
        name=module_name,
        purpose="supervisor no-site bootstrap",
    )
    try:
        supervisor_command = nosite.build_exact_nosite_argv(
            python=Path(plan["argv"][0]),
            bootstrap=bootstrap_path,
            bootstrap_sha256=bootstrap_sha,
            entrypoint=supervisor,
            entrypoint_sha256=supervisor_sha,
            import_roots=import_roots,
            entrypoint_argv=supervisor_args,
        )
        nosite.validate_exact_nosite_argv(
            supervisor_command.argv,
            expected_python=Path(plan["argv"][0]),
            expected_bootstrap=supervisor_command.contract["bootstrap"],
            expected_entrypoint=supervisor_command.contract["entrypoint"],
            expected_import_roots=import_roots,
            expected_entrypoint_argv=supervisor_args,
            expected_contract_sha256=supervisor_command.contract_sha256,
            verify_live=True,
        )
        command = list(supervisor_command.argv)
    except Exception as exc:
        raise LaunchRefused(
            f"supervisor no-site command construction failed: {exc}"
        ) from exc
    wrapper = [
        plan["argv"][0],
        "-I",
        "-B",
        "-S",
        "-c",
        gate_program,
        str(gate_read_fd),
        *command,
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _SANITIZED_ENV_ALLOWLIST
    }
    environment["PATH"] = os.defpath
    process: subprocess.Popen[Any] | None = None
    leader_receipt = namespace / "supervisor_leader_identity.json"
    gate_released = False
    accepted = False
    try:
        process = subprocess.Popen(
            wrapper,
            cwd=checkout / "hope_training/whole_body_tracking",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            pass_fds=(
                trainer_lock_fd,
                evaluator_lock_fd,
                gate_read_fd,
                control_read_fd,
            ),
            start_new_session=True,
        )
        os.close(log_fd)
        log_fd = -1
        os.close(gate_read_fd)
        gate_read_fd = -1
        os.close(control_read_fd)
        control_read_fd = -1
        leader_doc = exact.bind_leader(
            Path("/proc"),
            process.pid,
            process.pid,
            leader_receipt,
        )
        leader = leader_doc["leader"]
        executable_identity = _supervisor_proc_identity(
            pid=process.pid,
            expected_executable=Path(
                payload["isaac_python_runtime"]["resolved_path"]
            ),
            expected_sha256=payload["isaac_python_runtime"]["sha256"],
        )
        launch_record = {
            "schema_version": 2,
            "kind": "action_ball_stage_supervisor_bound_pending_release",
            "launch_claim_sha256": plan["launch_claim_sha256"],
            "source_commit_sha": source_commit,
            "stage": payload["stage"],
            "namespace": str(namespace),
            "leader": leader,
            **executable_identity,
            "leader_receipt": {
                "path": leader_receipt.name,
                "sha256": sha256_file(leader_receipt),
            },
            "live_gpu_admission_sha256": live_gpu_admission_sha,
            "command_sha256": canonical_sha256(command),
            "log_path": log_path.name,
        }
        _write_exclusive_json(
            namespace / "supervisor_started.json", launch_record
        )
        if os.write(gate_write_fd, b"G") != 1:
            raise LaunchRefused("could not release the exact supervisor start gate")
        gate_released = True
        os.close(gate_write_fd)
        gate_write_fd = -1
        _write_exclusive_json(
            namespace / "supervisor_released.json",
            {
                "schema_version": 1,
                "kind": "action_ball_stage_supervisor_gate_released",
                "launch_claim_sha256": plan["launch_claim_sha256"],
                "leader_receipt_sha256": launch_record[
                    "leader_receipt"
                ]["sha256"],
                "command_sha256": launch_record["command_sha256"],
            },
        )
        ready_path = namespace / "supervisor_ready.json"
        failed_path = namespace / "supervisor_failed.json"
        deadline = time.monotonic() + 1800.0
        while time.monotonic() < deadline:
            if failed_path.is_file():
                failure, _failure_sha = _read_canonical_namespace_json(
                    failed_path,
                    namespace=namespace,
                    name="dual-GPU supervisor failure receipt",
                )
                raise LaunchRefused(
                    "dual-GPU supervisor failed before acceptance: "
                    + json.dumps(failure, sort_keys=True)[-4000:]
                )
            if ready_path.is_file():
                ready_document, ready_sha = _read_canonical_namespace_json(
                    ready_path,
                    namespace=namespace,
                    name="dual-GPU supervisor ready receipt",
                )
                ready = _validate_supervisor_ready_receipt(
                    ready_document,
                    plan=plan,
                    namespace=namespace,
                )
                intent = {
                    "schema_version": 1,
                    "kind": "action_ball_launcher_accept_intent",
                    "launch_claim_sha256": plan["launch_claim_sha256"],
                    "supervisor_ready_sha256": ready_sha,
                    "live_gpu_admission_sha256": live_gpu_admission_sha,
                }
                intent_path = namespace / "launch_accept_intent.json"
                _write_exclusive_json(intent_path, intent)
                intent_sha = sha256_file(intent_path)
                if os.write(control_write_fd, b"A") != 1:
                    raise LaunchRefused(
                        "could not signal the exact supervisor acceptance intent"
                    )
                ack_path = namespace / "launch_accept_ack.json"
                ack_deadline = time.monotonic() + 120.0
                while time.monotonic() < ack_deadline:
                    if failed_path.is_file():
                        failure, _failure_sha = (
                            _read_canonical_namespace_json(
                                failed_path,
                                namespace=namespace,
                                name="dual-GPU supervisor failure receipt",
                            )
                        )
                        raise LaunchRefused(
                            "dual-GPU supervisor failed during acceptance: "
                            + json.dumps(failure, sort_keys=True)[-4000:]
                        )
                    if ack_path.is_file():
                        ack, ack_sha = _read_canonical_namespace_json(
                            ack_path,
                            namespace=namespace,
                            name="dual-GPU supervisor acceptance ack",
                        )
                        _validate_supervisor_accept_ack(
                            ack,
                            launch_claim_sha256=plan[
                                "launch_claim_sha256"
                            ],
                            supervisor_ready_sha256=ready_sha,
                            accept_intent_sha256=intent_sha,
                            live_gpu_admission_sha256=live_gpu_admission_sha,
                        )
                        accepted_receipt = {
                            "schema_version": 1,
                            "kind": "action_ball_launch_accepted",
                            "accepted_utc": _utc_now(),
                            "stage": payload["stage"],
                            "namespace": str(namespace),
                            "launch_claim_sha256": plan[
                                "launch_claim_sha256"
                            ],
                            "supervisor_ready": ready,
                            "accept_intent_sha256": intent_sha,
                            "supervisor_accept_ack_sha256": ack_sha,
                            "live_gpu_admission_sha256": (
                                live_gpu_admission_sha
                            ),
                        }
                        _write_exclusive_json(
                            namespace / "launch_accepted.json",
                            accepted_receipt,
                        )
                        accepted_path = namespace / "launch_accepted.json"
                        accepted_sha = sha256_file(accepted_path)
                        commit_ack_path = (
                            namespace / "launch_commit_ack.json"
                        )
                        commit_deadline = time.monotonic() + 120.0
                        while time.monotonic() < commit_deadline:
                            if failed_path.is_file():
                                failure, _failure_sha = (
                                    _read_canonical_namespace_json(
                                        failed_path,
                                        namespace=namespace,
                                        name=(
                                            "dual-GPU supervisor failure "
                                            "receipt"
                                        ),
                                    )
                                )
                                raise LaunchRefused(
                                    "dual-GPU supervisor failed before launch "
                                    "commit ack: "
                                    + json.dumps(
                                        failure, sort_keys=True
                                    )[-4000:]
                                )
                            if commit_ack_path.is_file():
                                commit_ack, _commit_ack_sha = (
                                    _read_canonical_namespace_json(
                                        commit_ack_path,
                                        namespace=namespace,
                                        name=(
                                            "dual-GPU supervisor launch "
                                            "commit ack"
                                        ),
                                    )
                                )
                                _validate_supervisor_launch_commit_ack(
                                    commit_ack,
                                    launch_claim_sha256=plan[
                                        "launch_claim_sha256"
                                    ],
                                    supervisor_ready_sha256=ready_sha,
                                    accept_intent_sha256=intent_sha,
                                    supervisor_accept_ack_sha256=ack_sha,
                                    launch_accepted_sha256=accepted_sha,
                                    live_gpu_admission_sha256=(
                                        live_gpu_admission_sha
                                    ),
                                    ready_processes=ready["processes"],
                                )
                                if process.poll() is not None:
                                    raise LaunchRefused(
                                        "dual-GPU supervisor exited while "
                                        "committing accepted launch"
                                    )
                                accepted = True
                                return accepted_receipt
                            returncode = process.poll()
                            if returncode is not None:
                                raise LaunchRefused(
                                    "dual-GPU supervisor exited before launch "
                                    f"commit ack: returncode={returncode}"
                                )
                            time.sleep(0.1)
                        raise LaunchRefused(
                            "dual-GPU supervisor did not commit-ack the exact "
                            "accepted launch within 120 seconds"
                        )
                    returncode = process.poll()
                    if returncode is not None:
                        raise LaunchRefused(
                            "dual-GPU supervisor exited before acceptance ack: "
                            f"returncode={returncode}"
                        )
                    time.sleep(0.1)
                raise LaunchRefused(
                    "dual-GPU supervisor did not acknowledge the exact "
                    "acceptance intent within 120 seconds"
                )
            returncode = process.poll()
            if returncode is not None:
                raise LaunchRefused(
                    "dual-GPU supervisor exited before ready: "
                    f"returncode={returncode}"
                )
            time.sleep(1.0)
        raise LaunchRefused(
            "dual-GPU supervisor did not prove sidecar-then-trainer readiness "
            "within 1800 seconds; cancelling the exact run"
        )
    except BaseException as original:
        if process is not None and not accepted:
            try:
                _cancel_and_reap_supervisor(
                    process=process,
                    exact=exact,
                    leader_receipt=leader_receipt,
                    namespace=namespace,
                    gate_write_fd=gate_write_fd,
                    control_write_fd=control_write_fd,
                    gate_released=gate_released,
                    gpus=payload["gpus"],
                )
                gate_write_fd = -1
                control_write_fd = -1
            except LaunchClosureUnknown:
                raise
            except BaseException as cleanup_exc:
                raise LaunchClosureUnknown(
                    "supervisor startup failed and exact closure could not be proven"
                ) from cleanup_exc
        raise original
    finally:
        for descriptor in (
            log_fd,
            gate_read_fd,
            gate_write_fd,
            control_read_fd,
            control_write_fd,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def launch_from_spec(
    spec_path: str | Path, stage: str, confirmation_claim_sha256: str
) -> dict[str, Any]:
    initial = prepare_launch_plan(spec_path, stage)
    confirmed = _sha256(
        confirmation_claim_sha256, name="confirmation claim SHA-256"
    )
    if confirmed != initial["launch_claim_sha256"]:
        raise LaunchRefused(
            "confirmation claim SHA-256 mismatch; expected "
            f"{initial['launch_claim_sha256']}"
        )
    gpu_rows = initial["canonical_payload"]["gpus"]
    lock_fds: dict[str, int] = {}
    namespace: Path | None = None
    try:
        for role in sorted(
            ("trainer", "evaluator"),
            key=lambda name: gpu_rows[name]["lock_path"],
        ):
            lock_fds[role] = acquire_gpu_lock(
                Path(gpu_rows[role]["lock_path"])
            )
        # All mutable static state is re-opened only after the shared lock.
        locked = prepare_launch_plan(spec_path, stage)
        if (
            locked["launch_claim_sha256"]
            != initial["launch_claim_sha256"]
        ):
            raise LaunchRefused(
                "launch plan changed between preflight and locked recheck"
            )
        locked_gpus = locked["canonical_payload"]["gpus"]
        live_gpu_checks: dict[str, Any] = {}
        for role in ("trainer", "evaluator"):
            gpu = locked_gpus[role]
            live_gpu_checks[role] = _verify_live_gpu_empty(
                gpu["index"], gpu["uuid"]
            )
        namespace = _claim_namespace(locked)
        try:
            live_gpu_admission_sha = _write_live_gpu_admission(
                locked, live_gpu_checks
            )
            accepted = _start_stage_supervisor(
                locked,
                trainer_lock_fd=lock_fds["trainer"],
                evaluator_lock_fd=lock_fds["evaluator"],
                live_gpu_admission_sha256=live_gpu_admission_sha,
            )
        except BaseException as exc:
            closure_unknown = isinstance(exc, LaunchClosureUnknown)
            _write_exclusive_json(
                namespace
                / (
                    "launch_unresolved.json"
                    if closure_unknown
                    else "launch_failed.json"
                ),
                {
                    "schema_version": 1,
                    "kind": (
                        "action_ball_launch_closure_unresolved"
                        if closure_unknown
                        else "action_ball_launch_failure"
                    ),
                    "failed_utc": _utc_now(),
                    "failure_class": type(exc).__name__,
                    "exception_type": type(exc).__name__,
                    "detail": str(exc)[-8000:],
                    "launch_claim_sha256": locked[
                        "launch_claim_sha256"
                    ],
                },
            )
            if isinstance(exc, (KeyboardInterrupt, SystemExit, LaunchRefused)):
                raise
            raise LaunchRefused(
                "dual-GPU supervisor raised an exception; namespace is "
                f"retained and spent: {namespace}: {exc}"
            ) from exc
        return accepted
    finally:
        # The exact training child inherited this descriptor through pass_fds.
        # Closing the parent copy therefore does not release the lifetime lock
        # until the child exits.
        for descriptor in lock_fds.values():
            os.close(descriptor)


def _run_isolated_train_entrypoint(args: argparse.Namespace) -> int:
    """Enter committed ``train.py`` with exactly one code-rooted import path."""

    source_commit = _commit(
        args.expected_source_commit,
        name="train-entrypoint expected source commit",
    )
    expected_entrypoint_sha = _sha256(
        args.expected_entrypoint_sha256,
        name="train-entrypoint expected entrypoint SHA",
    )
    expected_train_sha = _sha256(
        args.expected_train_sha256,
        name="train-entrypoint expected train.py SHA",
    )
    script = Path(__file__).resolve(strict=True)
    checkout = script.parents[3]
    verified_checkout, verified_commit = _verify_checkout(
        {"checkout": str(checkout), "commit_sha": source_commit}
    )
    if verified_checkout != checkout or verified_commit != source_commit:
        raise LaunchRefused("train-entrypoint checkout identity drifted")
    entrypoint, _, entrypoint_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        LAUNCHER_SOURCE,
        name="train-entrypoint code",
    )
    if entrypoint != script or entrypoint_sha != expected_entrypoint_sha:
        raise LaunchRefused("train-entrypoint code bytes drifted")
    train_script, _, train_sha, _ = _verify_repo_blob(
        checkout,
        source_commit,
        TRAIN_SOURCE,
        name="train-entrypoint train.py",
    )
    if train_sha != expected_train_sha:
        raise LaunchRefused("train-entrypoint train.py bytes drifted")
    expected_import_root = _absolute_normalized_path(
        args.expected_import_root,
        name="train-entrypoint expected import root",
        must_exist=True,
    )
    derived_import_root = (
        script.parents[1] / "source/whole_body_tracking"
    )
    if expected_import_root != derived_import_root:
        raise LaunchRefused(
            "train-entrypoint import root is not the exact checkout package root"
        )
    _require_real_directory(
        expected_import_root, name="train-entrypoint import root"
    )
    _verify_repo_blob(
        checkout,
        source_commit,
        (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/__init__.py"
        ),
        name="train-entrypoint package root",
    )
    sys.path.insert(0, str(expected_import_root))
    package_spec = importlib.util.find_spec("whole_body_tracking")
    if package_spec is None or package_spec.origin is None:
        raise LaunchRefused(
            "committed train-entrypoint could not resolve whole_body_tracking"
        )
    try:
        Path(package_spec.origin).resolve(strict=True).relative_to(
            expected_import_root
        )
    except (OSError, ValueError) as exc:
        raise LaunchRefused(
            "whole_body_tracking resolved outside the claimed import root"
        ) from exc
    train_argv = list(args.train_argv)
    if train_argv and train_argv[0] == "--":
        train_argv.pop(0)
    if not train_argv:
        raise LaunchRefused("train-entrypoint received no trainer arguments")
    sys.argv = [str(train_script), *train_argv]
    runpy.run_path(str(train_script), run_name="__main__")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan", help="static validation and exact argv rendering; no GPU side effects"
    )
    plan.add_argument("--spec", required=True)
    plan.add_argument("--stage", required=True, choices=STAGE_ORDER)
    launch = subparsers.add_parser(
        "launch", help="locked no-clobber launch after static validation"
    )
    launch.add_argument("--spec", required=True)
    launch.add_argument("--stage", required=True, choices=STAGE_ORDER)
    launch.add_argument("--confirm-claim-sha256", required=True)
    entrypoint = subparsers.add_parser(
        "train-entrypoint",
        help=argparse.SUPPRESS,
    )
    entrypoint.add_argument("--expected-source-commit", required=True)
    entrypoint.add_argument("--expected-entrypoint-sha256", required=True)
    entrypoint.add_argument("--expected-train-sha256", required=True)
    entrypoint.add_argument("--expected-import-root", required=True)
    entrypoint.add_argument("train_argv", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "train-entrypoint":
            return _run_isolated_train_entrypoint(args)
        if args.command == "plan":
            result = prepare_launch_plan(args.spec, args.stage)
        else:
            result = launch_from_spec(
                args.spec, args.stage, args.confirm_claim_sha256
            )
    except LaunchRefused as exc:
        print(f"ACTION_BALL_LAUNCH_REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
