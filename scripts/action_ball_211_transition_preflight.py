#!/usr/bin/env python3
"""Produce and validate the one-shot A211/C211 transition common-cut receipt.

The receipt proves only that, while the three physical-GPU coordination locks
were held exclusively, the legacy writers/reservations were drained, all three
GPUs were empty, and the four first-scale namespaces were absent.  It does not
claim atomicity after those locks are released.  Every launcher must therefore
retain its own immediate VendorV2 GPU admission and no-clobber checks.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
KIND = "action_ball_211_transition_preflight_v1"
MINIMUM_FREE_MEMORY_MIB = 8 * 1024
MAX_COMPUTE_PIDS = 2
WBT_RELATIVE = Path("hope_training/whole_body_tracking")
GPU_LOCK_PATHS = {
    0: Path("/tmp/hope_lean_queue_gpu0.lock"),
    1: Path("/tmp/hope_lean_queue_gpu1.lock"),
    2: Path("/tmp/hope_lean_queue_gpu2.lock"),
}
GPU_ROLES = {
    0: "A211_scale4096_pair",
    1: "C211_scale4096_pair",
    2: "reserved_for_mujoco",
}
A_EXPERIMENT_NAME = "agibot_a3_action_ball_a211_four_arm_diagnostic"
C_EXPERIMENT_NAME = "agibot_a3_action_ball_c211_diagnostic"
ALLOWED_CLAIM_KINDS = (
    "action_ball_a211_four_arm_diagnostic_claim_v2",
    "action_ball_c211_diagnostic_claim_v2",
)
# 2026-08-05 第二轴改版(第二次,exp §5.6.2d):探索包定死为四格共用的标准初始化 +
# sigma 1.0,第二轴换成本体感观测噪声开关,cell_id 随之改名。
# 卡角色未变:gpu0 = A 对,gpu1 = C 对,gpu2 留给 MuJoCo(本 preflight 仍要求它是空的)。
TARGET_SPECS = (
    (
        "a0",
        "A0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off",
        "A211",
        0,
        A_EXPERIMENT_NAME,
    ),
    (
        "a1",
        "A1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on",
        "A211",
        0,
        A_EXPERIMENT_NAME,
    ),
    (
        "c0",
        "C0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off",
        "C211",
        1,
        C_EXPERIMENT_NAME,
    ),
    (
        "c1",
        "C1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on",
        "C211",
        1,
        C_EXPERIMENT_NAME,
    ),
)
RUNTIME_SOURCE_SPECS = (
    (
        "transition_preflight",
        "scripts/action_ball_211_transition_preflight.py",
    ),
    (
        "vendor_v2_gpu_admission",
        "hope_training/whole_body_tracking/scripts/vendor_v2_gpu_admission.py",
    ),
    (
        "four_grid_contract",
        "hope_training/whole_body_tracking/scripts/action_ball_211_four_grid_contract.py",
    ),
    (
        "a211_launcher",
        "hope_training/whole_body_tracking/scripts/launch_action_ball_a211_four_arm_diagnostic.py",
    ),
    (
        "c211_launcher",
        "hope_training/whole_body_tracking/scripts/launch_action_ball_c211_diagnostic.py",
    ),
)
LEGACY_EXPERIMENT_NAMES = (
    "agibot_a3_action_ball_measured_vendor_v2_n1_diagnostic",
    "agibot_a3_action_ball_a225_four_arm_diagnostic",
    "agibot_a3_action_ball_c225_diagnostic",
    A_EXPERIMENT_NAME,
    C_EXPERIMENT_NAME,
)
WRITER_SOURCE_NAMES = (
    "launch_n1_vendor_baseline_diagnostic.py",
    "launch_n1_measured_vendor_v2_diagnostic.py",
    "launch_action_ball_a225_four_arm_diagnostic.py",
    "launch_action_ball_c225_diagnostic.py",
    "launch_action_ball_a211_four_arm_diagnostic.py",
    "launch_action_ball_c211_diagnostic.py",
)
GPU_RESERVATION_REGISTRY_SUFFIX = ".vendor_v2_reservations"
GPU_RESERVATION_KEYS = (
    "schema_version",
    "kind",
    "owner_pid",
    "owner_proc_starttime_ticks",
    "gpu_index",
    "gpu_uuid",
    "namespace",
    "checkout",
    "commit_sha",
    "launch_claim_sha256",
    "max_compute_pids",
    "minimum_free_memory_mib",
    "allow_vendor_v2_colocation",
)
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9][A-Za-z0-9-]*$")
BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
OBSERVED_AT_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"\.[0-9]{6}Z$"
)


class TransitionPreflightRefused(RuntimeError):
    """A fail-closed transition preflight refusal."""


PreflightRefused = TransitionPreflightRefused


def canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TransitionPreflightRefused(
            "transition receipt value is not canonical JSON"
        ) from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 of canonical JSON bytes (without newline)."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_dict(value: Any, keys: Sequence[str], *, name: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys):
        raise TransitionPreflightRefused("%s keys differ" % name)
    return dict(value)


def _sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise TransitionPreflightRefused("%s must be a lowercase SHA-256" % name)
    return value


def _commit_sha(value: Any) -> str:
    if type(value) is not str or COMMIT_RE.fullmatch(value) is None:
        raise TransitionPreflightRefused(
            "commit SHA must be a full lowercase 40-hex commit"
        )
    return value


def _normalized_absolute_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TransitionPreflightRefused("%s must be a path" % name)
    text = str(value)
    if (
        not text
        or "\x00" in text
        or "\n" in text
        or not os.path.isabs(text)
        or os.path.normpath(text) != text
    ):
        raise TransitionPreflightRefused(
            "%s must be a normalized absolute path" % name
        )
    return Path(text)


def _real_directory(path: Path, *, name: str) -> os.stat_result:
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        after = path.lstat()
    except OSError as exc:
        raise TransitionPreflightRefused("%s is not an existing directory" % name) from exc
    if (
        not stat.S_ISDIR(before.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or resolved != path
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
    ):
        raise TransitionPreflightRefused("%s must be one real directory" % name)
    return after


def _stable_regular_bytes(
    path: Path,
    *,
    name: str,
    maximum_bytes: int = 16 << 20,
    require_single_link: bool = True,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TransitionPreflightRefused("%s cannot be opened" % name) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (require_single_link and before.st_nlink != 1)
            or (require_single_link and resolved != path)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise TransitionPreflightRefused(
                "%s must be one bounded regular file" % name
            )
        chunks = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after_fd = os.fstat(descriptor)
    except OSError as exc:
        raise TransitionPreflightRefused("%s cannot be read stably" % name) from exc
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise TransitionPreflightRefused("%s disappeared while reading" % name) from exc
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    )
    if (
        len(raw) != before.st_size
        or identity
        != (
            after_fd.st_dev,
            after_fd.st_ino,
            after_fd.st_size,
            after_fd.st_mtime_ns,
            after_fd.st_nlink,
        )
        or identity
        != (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
            after_path.st_mtime_ns,
            after_path.st_nlink,
        )
    ):
        raise TransitionPreflightRefused("%s changed while reading" % name)
    return raw


def _strict_json_bytes(raw: bytes, *, name: str) -> Any:
    def _reject_constant(text: str) -> None:
        raise ValueError(text)

    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TransitionPreflightRefused("%s is not strict JSON" % name) from exc


def _stable_canonical_json(path: Path, *, name: str) -> Tuple[Dict[str, Any], str]:
    raw = _stable_regular_bytes(path, name=name)
    value = _strict_json_bytes(raw, name=name)
    if type(value) is not dict or raw != canonical_bytes(value) + b"\n":
        raise TransitionPreflightRefused(
            "%s must be canonical JSON plus one newline" % name
        )
    return dict(value), hashlib.sha256(raw).hexdigest()


def _trusted_binary(candidates: Sequence[str], *, name: str) -> Tuple[str, str]:
    for candidate in candidates:
        requested = Path(candidate)
        try:
            path = requested.resolve(strict=True)
            info = path.lstat()
        except OSError:
            continue
        if (
            stat.S_ISREG(info.st_mode)
            and os.access(str(path), os.X_OK)
        ):
            raw = _stable_regular_bytes(
                path,
                name=name,
                maximum_bytes=64 << 20,
                require_single_link=False,
            )
            return str(path), hashlib.sha256(raw).hexdigest()
    raise TransitionPreflightRefused("trusted %s binary is unavailable" % name)


def _run_git(checkout: Path, arguments: Sequence[str]) -> bytes:
    git, _digest = _trusted_binary(
        ("/usr/bin/git", "/usr/local/bin/git"), name="git"
    )
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LANG": "C",
        "LC_ALL": "C",
    }
    result = subprocess.run(
        [git, "-C", str(checkout)] + list(arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if result.returncode != 0:
        try:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
        except Exception:  # pragma: no cover - bytes decode is total with replace
            detail = ""
        raise TransitionPreflightRefused(
            "git %s failed: %s" % (" ".join(arguments), detail)
        )
    return result.stdout


def _script_checkout() -> Path:
    return Path(__file__).resolve(strict=True).parents[1]


def _source_document(checkout: Path, commit: str) -> Dict[str, Any]:
    _real_directory(checkout, name="source checkout")
    if _script_checkout() != checkout:
        raise TransitionPreflightRefused(
            "preflight executable must come from the exact source checkout"
        )
    try:
        root = _run_git(checkout, ("rev-parse", "--show-toplevel")).decode(
            "utf-8"
        ).strip()
        head = _run_git(checkout, ("rev-parse", "--verify", "HEAD^{commit}")).decode(
            "ascii"
        ).strip()
    except UnicodeError as exc:
        raise TransitionPreflightRefused("git source identity is not parseable") from exc
    if root != str(checkout) or head != commit:
        raise TransitionPreflightRefused(
            "source checkout HEAD does not equal the requested exact commit"
        )
    dirty = _run_git(
        checkout, ("status", "--porcelain=v1", "-z", "--untracked-files=all")
    )
    if dirty:
        raise TransitionPreflightRefused("source checkout is not clean")
    runtime_sources = {}
    for label, relative in RUNTIME_SOURCE_SPECS:
        tracked = _run_git(
            checkout, ("ls-files", "--error-unmatch", "--", relative)
        )
        try:
            tracked_text = tracked.decode("utf-8").strip()
        except UnicodeError as exc:
            raise TransitionPreflightRefused(
                "runtime source tracking result is not UTF-8"
            ) from exc
        if tracked_text != relative:
            raise TransitionPreflightRefused(
                "runtime source is not tracked exactly: %s" % relative
            )
        path = checkout / relative
        raw = _stable_regular_bytes(path, name="runtime source %s" % label)
        committed_raw = _run_git(
            checkout, ("cat-file", "blob", "%s:%s" % (commit, relative))
        )
        if raw != committed_raw:
            raise TransitionPreflightRefused(
                "runtime source bytes differ from the exact commit: %s" % relative
            )
        runtime_sources[label] = {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    final_head = _run_git(
        checkout, ("rev-parse", "--verify", "HEAD^{commit}")
    ).decode("ascii").strip()
    final_dirty = _run_git(
        checkout, ("status", "--porcelain=v1", "-z", "--untracked-files=all")
    )
    if final_head != commit or final_dirty:
        raise TransitionPreflightRefused(
            "source checkout changed while runtime sources were pinned"
        )
    return {
        "checkout": str(checkout),
        "commit_sha": commit,
        "clean": True,
        "runtime_sources": runtime_sources,
    }


def _boot_id() -> str:
    path = BOOT_ID_PATH
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TransitionPreflightRefused("Linux boot id cannot be opened") from exc
    try:
        opened = os.fstat(descriptor)
        raw = os.read(descriptor, 257)
        after_fd = os.fstat(descriptor)
    except OSError as exc:
        raise TransitionPreflightRefused("Linux boot id cannot be read") from exc
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise TransitionPreflightRefused(
            "Linux boot id disappeared while reading"
        ) from exc
    identity = (before.st_dev, before.st_ino, stat.S_IFMT(before.st_mode))
    if (
        not stat.S_ISREG(before.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(after_fd.st_mode)
        or not stat.S_ISREG(after_path.st_mode)
        or identity
        != (opened.st_dev, opened.st_ino, stat.S_IFMT(opened.st_mode))
        or identity
        != (after_fd.st_dev, after_fd.st_ino, stat.S_IFMT(after_fd.st_mode))
        or identity
        != (
            after_path.st_dev,
            after_path.st_ino,
            stat.S_IFMT(after_path.st_mode),
        )
        or not raw
        or len(raw) > 256
    ):
        raise TransitionPreflightRefused(
            "Linux boot id is not one stable bounded procfs file"
        )
    try:
        value = raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise TransitionPreflightRefused("Linux boot id is not ASCII") from exc
    if BOOT_ID_RE.fullmatch(value) is None:
        raise TransitionPreflightRefused("Linux boot id is invalid")
    return value


def _observed_at() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _acquire_exclusive_gpu_lock(index: int, path: Path) -> Tuple[int, Dict[str, Any]]:
    expected = Path("/tmp/hope_lean_queue_gpu%d.lock" % index)
    if GPU_LOCK_PATHS == {
        0: Path("/tmp/hope_lean_queue_gpu0.lock"),
        1: Path("/tmp/hope_lean_queue_gpu1.lock"),
        2: Path("/tmp/hope_lean_queue_gpu2.lock"),
    } and path != expected:
        raise TransitionPreflightRefused("GPU lock path differs for index %d" % index)
    try:
        before = path.lstat()
    except OSError as exc:
        raise TransitionPreflightRefused(
            "GPU %d coordination lock must already exist" % index
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise TransitionPreflightRefused(
            "GPU %d coordination lock is not a regular file" % index
        )
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TransitionPreflightRefused(
            "GPU %d coordination lock cannot be opened" % index
        ) from exc
    try:
        opened = os.fstat(descriptor)
        after = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise TransitionPreflightRefused(
                "GPU %d coordination lock pathname identity changed" % index
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TransitionPreflightRefused(
                "GPU %d coordination lock is already held" % index
            ) from exc
        os.set_inheritable(descriptor, False)
        identity = {
            "path": str(path),
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "exclusive": True,
        }
        _revalidate_open_gpu_lock(index, descriptor, identity)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_open_gpu_lock(
    index: int, descriptor: int, identity: Mapping[str, Any]
) -> None:
    path = GPU_LOCK_PATHS[index]
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
    except OSError as exc:
        raise TransitionPreflightRefused(
            "GPU %d held lock pathname cannot be revalidated" % index
        ) from exc
    expected = (identity["device"], identity["inode"])
    if (
        identity.get("path") != str(path)
        or identity.get("exclusive") is not True
        or not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (opened.st_dev, opened.st_ino) != expected
        or (current.st_dev, current.st_ino) != expected
        or os.get_inheritable(descriptor)
    ):
        raise TransitionPreflightRefused(
            "GPU %d held lock pathname identity changed" % index
        )


def _revalidate_held_gpu_locks(
    handles: Sequence[Tuple[int, int, Mapping[str, Any]]]
) -> None:
    if [row[0] for row in handles] != [0, 1, 2]:
        raise TransitionPreflightRefused(
            "held GPU lock order no longer covers 0, 1, and 2"
        )
    for index, descriptor, identity in handles:
        _revalidate_open_gpu_lock(index, descriptor, identity)


def _acquire_ordered_gpu_locks() -> List[Tuple[int, int, Dict[str, Any]]]:
    handles = []
    try:
        for index in (0, 1, 2):
            descriptor, identity = _acquire_exclusive_gpu_lock(
                index, GPU_LOCK_PATHS[index]
            )
            handles.append((index, descriptor, identity))
        _revalidate_held_gpu_locks(handles)
        return handles
    except BaseException:
        for _index, descriptor, _identity in reversed(handles):
            os.close(descriptor)
        raise


class _AdmissionBase:
    LaunchRefused = TransitionPreflightRefused
    WBT_RELATIVE = WBT_RELATIVE

    @staticmethod
    def _trusted_nvidia_smi() -> Tuple[str, str]:
        return _trusted_binary(
            ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"),
            name="nvidia-smi",
        )

    @staticmethod
    def _stable_regular_file(path: Path, *, name: str) -> os.stat_result:
        _stable_regular_bytes(path, name=name)
        return path.lstat()

    @staticmethod
    def _canonical_bytes(value: Any) -> bytes:
        return canonical_bytes(value)

    @staticmethod
    def _strict_json_bytes(raw: bytes, *, name: str) -> Any:
        return _strict_json_bytes(raw, name=name)

    @staticmethod
    def _absolute_path(value: Any, *, name: str, must_exist: bool = False) -> Path:
        path = _normalized_absolute_path(value, name=name)
        if must_exist:
            try:
                path.lstat()
            except OSError as exc:
                raise TransitionPreflightRefused("%s does not exist" % name) from exc
        return path

    @staticmethod
    def _sha256(value: Any, *, name: str) -> str:
        return _sha256(value, name=name)

    @staticmethod
    def sha256_file(path: Path) -> str:
        return hashlib.sha256(
            _stable_regular_bytes(path, name="SHA-256 input")
        ).hexdigest()

    @staticmethod
    def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
        _write_exclusive(path, value)


def _load_vendor_admission(checkout: Path) -> Any:
    path = checkout / dict(RUNTIME_SOURCE_SPECS)["vendor_v2_gpu_admission"]
    module_name = "_action_ball_211_transition_vendor_admission_%s" % os.getpid()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise TransitionPreflightRefused("cannot load VendorV2 GPU admission")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TransitionPreflightRefused(
            "cannot execute VendorV2 GPU admission"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    if (
        getattr(module, "MIN_VENDOR_V2_FREE_MEMORY_MIB", None)
        != MINIMUM_FREE_MEMORY_MIB
        or getattr(module, "GPU_RESERVATION_REGISTRY_SUFFIX", None)
        != GPU_RESERVATION_REGISTRY_SUFFIX
    ):
        raise TransitionPreflightRefused(
            "VendorV2 GPU admission constants differ from transition contract"
        )
    admission = module.VendorV2GPUAdmission(
        base=_AdmissionBase,
        schema_version=1,
        claim_kind="transition_preflight_never_accepts_a_live_claim",
        experiment_name="transition_preflight",
        colocation_spec_key="transition_colocation_prohibited",
        physical_ball_semantics="transition_preflight_only",
        runtime_source_paths=(),
        launcher_source="",
        admission_source="",
        exact_group_source="",
        exact_group=None,
        canonical_sha256=canonical_sha256,
        exact_dict=_exact_dict,
        validate_spec=lambda value, claimed: value,
        output_contract=lambda value: {},
        training_argv=lambda spec_value, bundle: [],
        physical_reservation_registry=True,
    )

    def _refuse_live_claim(*args: Any, **kwargs: Any) -> Any:
        raise TransitionPreflightRefused(
            "a live GPU reservation or runtime handoff exists"
        )

    admission._validate_namespace_claim = _refuse_live_claim
    return admission


def _verify_four_grid_authority(checkout: Path) -> None:
    path = checkout / dict(RUNTIME_SOURCE_SPECS)["four_grid_contract"]
    module_name = "_action_ball_211_transition_four_grid_%s" % os.getpid()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise TransitionPreflightRefused("cannot load four-grid authority")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TransitionPreflightRefused("cannot execute four-grid authority") from exc
    finally:
        sys.modules.pop(module_name, None)
    expected_cells = tuple(row[1] for row in TARGET_SPECS)
    if (
        tuple(getattr(module, "CELL_IDS", ())) != expected_cells
        or getattr(module, "FAMILY_CELL_IDS", None)
        != {
            "A211": expected_cells[:2],
            "C211": expected_cells[2:],
        }
    ):
        raise TransitionPreflightRefused(
            "transition targets differ from the code-owned four-grid authority"
        )


def _query_gpu(admission: Any, index: int, uuid: str) -> Mapping[str, Any]:
    return admission._query_gpu_processes(index, uuid)


def _physical_registry_snapshot(
    root: Path, *, gpu_index: int, gpu_uuid: str
) -> Dict[str, Any]:
    try:
        before = root.lstat()
    except FileNotFoundError:
        return {"present": False, "entries": []}
    except OSError as exc:
        raise TransitionPreflightRefused(
            "physical GPU reservation registry cannot be inspected"
        ) from exc
    try:
        resolved = root.resolve(strict=True)
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise TransitionPreflightRefused(
            "physical GPU reservation registry cannot be scanned"
        ) from exc
    if not stat.S_ISDIR(before.st_mode) or resolved != root:
        raise TransitionPreflightRefused(
            "physical GPU reservation registry is not one real directory"
        )
    rows = []
    for entry in sorted(entries, key=lambda value: value.name):
        document, file_sha = _stable_canonical_json(
            entry, name="physical GPU reservation entry"
        )
        reservation = _exact_dict(
            document,
            GPU_RESERVATION_KEYS,
            name="physical GPU reservation entry",
        )
        owner_pid = reservation["owner_pid"]
        owner_start = reservation["owner_proc_starttime_ticks"]
        observed_index = reservation["gpu_index"]
        observed_uuid = reservation["gpu_uuid"]
        claim_sha = _sha256(
            reservation["launch_claim_sha256"],
            name="physical GPU reservation claim SHA",
        )
        _commit_sha(reservation["commit_sha"])
        _normalized_absolute_path(
            reservation["namespace"],
            name="physical GPU reservation namespace",
        )
        _normalized_absolute_path(
            reservation["checkout"],
            name="physical GPU reservation checkout",
        )
        if (
            type(reservation["schema_version"]) is not int
            or isinstance(reservation["schema_version"], bool)
            or reservation["schema_version"] != 1
            or reservation["kind"]
            != "measured_vendor_v2_gpu_slot_reservation_v1"
            or type(owner_pid) is not int
            or isinstance(owner_pid, bool)
            or owner_pid <= 0
            or type(owner_start) is not int
            or isinstance(owner_start, bool)
            or owner_start <= 0
            or type(observed_index) is not int
            or isinstance(observed_index, bool)
            or observed_index != gpu_index
            or observed_uuid != gpu_uuid
            or entry.name != claim_sha + ".json"
            or reservation["max_compute_pids"] != MAX_COMPUTE_PIDS
            or reservation["minimum_free_memory_mib"]
            != MINIMUM_FREE_MEMORY_MIB
            or type(reservation["allow_vendor_v2_colocation"]) is not bool
        ):
            raise TransitionPreflightRefused(
                "physical reservation identity differs from its GPU lock registry"
            )
        rows.append(
            {
                "name": entry.name,
                "sha256": file_sha,
                "device": entry.lstat().st_dev,
                "inode": entry.lstat().st_ino,
            }
        )
    try:
        after = root.lstat()
    except OSError as exc:
        raise TransitionPreflightRefused(
            "physical GPU reservation registry changed while scanning"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise TransitionPreflightRefused(
            "physical GPU reservation registry changed while scanning"
        )
    return {
        "present": True,
        "device": before.st_dev,
        "inode": before.st_ino,
        "entries": rows,
    }


def _scan_live_reservations(
    admission: Any,
    *,
    checkout: Path,
    commit: str,
    gpu_index: int,
    gpu_uuid: str,
) -> List[Mapping[str, Any]]:
    lock_path = GPU_LOCK_PATHS[gpu_index]
    registry = lock_path.parent / (
        lock_path.name + GPU_RESERVATION_REGISTRY_SUFFIX
    )
    registry_before = _physical_registry_snapshot(
        registry, gpu_index=gpu_index, gpu_uuid=gpu_uuid
    )
    rows = list(
        admission._live_reservations(
            registry,
            checkout=checkout,
            commit=commit,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            proc_root=Path("/proc"),
        )
    )
    legacy_root = checkout / WBT_RELATIVE / "logs" / "rsl_rl"
    for experiment_name in LEGACY_EXPERIMENT_NAMES:
        rows.extend(
            admission._live_reservations(
                legacy_root / experiment_name,
                checkout=checkout,
                commit=commit,
                gpu_index=gpu_index,
                gpu_uuid=gpu_uuid,
                proc_root=Path("/proc"),
                namespace_audit_mode=True,
            )
        )
    registry_after = _physical_registry_snapshot(
        registry, gpu_index=gpu_index, gpu_uuid=gpu_uuid
    )
    if registry_after != registry_before:
        raise TransitionPreflightRefused(
            "physical GPU reservation registry changed during admission"
        )
    return rows


def _proc_starttime_if_present(pid: int) -> Optional[int]:
    path = Path("/proc") / str(pid) / "stat"
    try:
        raw = path.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise TransitionPreflightRefused(
            "cannot read process identity while scanning writers"
        ) from exc
    close = raw.rfind(")")
    fields = raw[close + 2 :].split() if close >= 0 else []
    if len(fields) <= 19 or not fields[19].isdigit() or int(fields[19]) <= 0:
        raise TransitionPreflightRefused(
            "unparseable process identity while scanning writers"
        )
    return int(fields[19])


def _scan_live_writers() -> List[Dict[str, Any]]:
    proc_root = Path("/proc")
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise TransitionPreflightRefused("cannot scan /proc for live writers") from exc
    rows = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        start_before = _proc_starttime_if_present(pid)
        if start_before is None:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise TransitionPreflightRefused(
                "cannot read live process cmdline while scanning writers"
            ) from exc
        if not raw:
            continue
        if not raw.endswith(b"\0"):
            raise TransitionPreflightRefused(
                "live process cmdline is not NUL terminated"
            )
        try:
            argv = [item.decode("utf-8") for item in raw[:-1].split(b"\0")]
        except UnicodeError as exc:
            raise TransitionPreflightRefused(
                "live process cmdline is not UTF-8"
            ) from exc
        start_after = _proc_starttime_if_present(pid)
        if start_after is None:
            continue
        if start_after != start_before:
            raise TransitionPreflightRefused(
                "process identity changed while scanning writers"
            )
        matched = sorted(
            {
                name
                for name in WRITER_SOURCE_NAMES
                if any(name in argument for argument in argv)
            }
        )
        if matched:
            rows.append(
                {
                    "pid": pid,
                    "proc_starttime_ticks": start_before,
                    "writer_sources": matched,
                    "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return sorted(rows, key=lambda row: row["pid"])


def _target_parent(checkout: Path, experiment_name: str) -> Path:
    return checkout / WBT_RELATIVE / "logs" / "rsl_rl" / experiment_name


def _refuse_existing_scale_claims(checkout: Path) -> None:
    for family, experiment_name, selector_key, allowed_cells, claim_kind in (
        (
            "A211",
            A_EXPERIMENT_NAME,
            "arm_id",
            tuple(row[1] for row in TARGET_SPECS[:2]),
            ALLOWED_CLAIM_KINDS[0],
        ),
        (
            "C211",
            C_EXPERIMENT_NAME,
            "recipe_id",
            tuple(row[1] for row in TARGET_SPECS[2:]),
            ALLOWED_CLAIM_KINDS[1],
        ),
    ):
        root = _target_parent(checkout, experiment_name)
        _real_directory(root, name="%s experiment root" % family)
        try:
            entries = tuple(root.iterdir())
        except OSError as exc:
            raise TransitionPreflightRefused(
                "%s experiment root cannot be scanned" % family
            ) from exc
        for namespace in entries:
            try:
                info = namespace.lstat()
            except OSError as exc:
                raise TransitionPreflightRefused(
                    "%s experiment namespace cannot be inspected" % family
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise TransitionPreflightRefused(
                    "%s experiment root contains a symlinked namespace" % family
                )
            if not stat.S_ISDIR(info.st_mode):
                continue
            claim_path = namespace / "launch_claim.json"
            try:
                claim_path.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise TransitionPreflightRefused(
                    "%s launch claim cannot be inspected" % family
                ) from exc
            claim, _claim_file_sha = _stable_canonical_json(
                claim_path, name="%s existing launch claim" % family
            )
            outer = _exact_dict(
                claim,
                (
                    "schema_version",
                    "kind",
                    "launch_claim_sha256",
                    "canonical_payload",
                ),
                name="%s existing launch claim" % family,
            )
            payload = outer["canonical_payload"]
            if (
                type(outer["schema_version"]) is not int
                or isinstance(outer["schema_version"], bool)
                or outer["schema_version"] != 2
                or outer["kind"] != claim_kind
                or type(payload) is not dict
                or outer["launch_claim_sha256"] != canonical_sha256(payload)
                or type(payload.get("spec")) is not dict
            ):
                raise TransitionPreflightRefused(
                    "%s experiment root contains an invalid launch claim" % family
                )
            launch_spec = payload["spec"]
            if launch_spec.get("stage") == "scale4096":
                selector = launch_spec.get(selector_key)
                if selector not in allowed_cells:
                    raise TransitionPreflightRefused(
                        "%s existing scale4096 claim has an unknown grid cell" % family
                    )
                raise TransitionPreflightRefused(
                    "%s scale4096 was already claimed before this preflight" % family
                )


def _absent(path: Path, *, name: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise TransitionPreflightRefused("cannot inspect %s" % name) from exc
    raise TransitionPreflightRefused("%s already exists" % name)


def _targets_document(
    checkout: Path,
    namespace_values: Mapping[str, Any],
    gpu_uuids: Mapping[int, str],
) -> List[Dict[str, Any]]:
    if type(namespace_values) is not dict or set(namespace_values) != {
        "a0",
        "a1",
        "c0",
        "c1",
    }:
        raise TransitionPreflightRefused(
            "exactly four named target namespaces are required"
        )
    rows = []
    seen = set()
    for short, cell_id, family, gpu_index, experiment_name in TARGET_SPECS:
        namespace = _normalized_absolute_path(
            namespace_values[short], name="%s target namespace" % short
        )
        parent = _target_parent(checkout, experiment_name)
        _real_directory(parent, name="%s experiment root" % family)
        if namespace.parent != parent or namespace.name in ("", ".", ".."):
            raise TransitionPreflightRefused(
                "%s target namespace has the wrong direct parent" % short
            )
        if str(namespace) in seen:
            raise TransitionPreflightRefused("target namespaces must be unique")
        seen.add(str(namespace))
        _absent(namespace, name="%s target namespace" % short)
        rows.append(
            {
                "cell_id": cell_id,
                "family": family,
                "gpu_index": gpu_index,
                "gpu_uuid": gpu_uuids[gpu_index],
                "namespace": str(namespace),
                "namespace_absent": True,
            }
        )
    return rows


def _gpu_row(
    *,
    index: int,
    uuid: str,
    lock: Mapping[str, Any],
    queried: Mapping[str, Any],
    reservations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if type(queried) is not dict:
        raise TransitionPreflightRefused("GPU %d query is not an object" % index)
    observed_index = queried.get("index")
    observed_uuid = queried.get("uuid")
    total = queried.get("total_memory_mib")
    free = queried.get("free_memory_mib")
    processes = queried.get("processes")
    if (
        type(observed_index) is not int
        or isinstance(observed_index, bool)
        or observed_index != index
        or observed_uuid != uuid
        or type(total) is not int
        or isinstance(total, bool)
        or total <= 0
        or type(free) is not int
        or isinstance(free, bool)
        or free < MINIMUM_FREE_MEMORY_MIB
        or free > total
        or type(processes) is not list
    ):
        raise TransitionPreflightRefused(
            "GPU %d UUID or memory observation differs" % index
        )
    if processes:
        raise TransitionPreflightRefused(
            "GPU %d has live compute processes at the transition cut" % index
        )
    if reservations:
        raise TransitionPreflightRefused(
            "GPU %d has live reservations at the transition cut" % index
        )
    return {
        "index": index,
        "uuid": uuid,
        "role": GPU_ROLES[index],
        "lock": dict(lock),
        "total_memory_mib": total,
        "free_memory_mib": free,
        "minimum_free_memory_mib": MINIMUM_FREE_MEMORY_MIB,
        "compute_processes": [],
        "live_reservations": [],
    }


def _validate_output_path(output: Any, targets: Mapping[str, Any]) -> Path:
    path = _normalized_absolute_path(output, name="transition receipt output")
    _real_directory(path.parent, name="transition receipt output parent")
    if str(path) in {str(value) for value in targets.values()}:
        raise TransitionPreflightRefused(
            "transition receipt output cannot be a target namespace"
        )
    _absent(path, name="transition receipt output")
    return path


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise TransitionPreflightRefused(
            "transition receipt output must be fresh"
        ) from exc
    opened = os.fstat(descriptor)
    created = True
    try:
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise TransitionPreflightRefused(
                "transition receipt output is not one regular file"
            )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short receipt write")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            current = path.lstat()
            if (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino):
                path.unlink()
                created = False
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    try:
        final = path.lstat()
        if (
            not stat.S_ISREG(final.st_mode)
            or (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino)
            or final.st_size != len(raw)
        ):
            raise TransitionPreflightRefused(
                "transition receipt output identity changed"
            )
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if created:
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == (
                    opened.st_dev,
                    opened.st_ino,
                ):
                    path.unlink()
            except OSError:
                pass
        raise
    return hashlib.sha256(raw).hexdigest()


def produce_receipt(
    *,
    checkout: Any,
    commit_sha: str,
    gpu_uuids: Mapping[int, str],
    namespaces: Mapping[str, Any],
    output: Any,
) -> Dict[str, Any]:
    """Produce one canonical no-clobber transition receipt."""

    checkout_path = _normalized_absolute_path(checkout, name="source checkout")
    commit = _commit_sha(commit_sha)
    if type(gpu_uuids) is not dict or set(gpu_uuids) != {0, 1, 2}:
        raise TransitionPreflightRefused("GPU UUIDs must cover indices 0, 1, and 2")
    normalized_uuids = {}
    for index in (0, 1, 2):
        uuid = gpu_uuids[index]
        if type(uuid) is not str or GPU_UUID_RE.fullmatch(uuid) is None:
            raise TransitionPreflightRefused("GPU %d UUID is invalid" % index)
        normalized_uuids[index] = uuid
    if len(set(normalized_uuids.values())) != 3:
        raise TransitionPreflightRefused("physical GPU UUIDs must be unique")
    output_path = _validate_output_path(output, namespaces)
    source_before = _source_document(checkout_path, commit)
    _verify_four_grid_authority(checkout_path)
    admission = _load_vendor_admission(checkout_path)
    locks = _acquire_ordered_gpu_locks()
    try:
        lock_by_index = {index: identity for index, _fd, identity in locks}
        source_at_cut = _source_document(checkout_path, commit)
        if source_at_cut != source_before:
            raise TransitionPreflightRefused(
                "source identity changed before the transition common cut"
            )
        _refuse_existing_scale_claims(checkout_path)
        boot_before = _boot_id()
        live_writers = _scan_live_writers()
        if live_writers:
            raise TransitionPreflightRefused(
                "legacy or grid launcher writer is live at the transition cut"
            )
        gpu_rows = []
        for index in (0, 1, 2):
            uuid = normalized_uuids[index]
            queried = _query_gpu(admission, index, uuid)
            reservations = _scan_live_reservations(
                admission,
                checkout=checkout_path,
                commit=commit,
                gpu_index=index,
                gpu_uuid=uuid,
            )
            gpu_rows.append(
                _gpu_row(
                    index=index,
                    uuid=uuid,
                    lock=lock_by_index[index],
                    queried=queried,
                    reservations=reservations,
                )
            )
        target_rows = _targets_document(
            checkout_path, namespaces, normalized_uuids
        )
        source_after_cut = _source_document(checkout_path, commit)
        if source_after_cut != source_at_cut:
            raise TransitionPreflightRefused(
                "source identity changed during the transition common cut"
            )
        boot_after = _boot_id()
        if boot_after != boot_before:
            raise TransitionPreflightRefused(
                "host rebooted during the transition common cut"
            )
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "status": "PASS",
            "diagnostic_unauthorized": True,
            "machine_verified": {
                "common_cut_before_first_scale4096": True,
                "legacy_live_or_pending_count": 0,
                "cross_time_atomicity_claimed": False,
                "cross_checkout_legacy_pending_completeness_claimed": False,
            },
            "host": {
                "boot_id": boot_before,
                "observed_at": _observed_at(),
            },
            "source": source_at_cut,
            "writer_policy": {
                "allowed_claim_kinds": list(ALLOWED_CLAIM_KINDS),
                "observed_live_writers": [],
            },
            "gpus": gpu_rows,
            "targets": target_rows,
        }
        document = dict(unsigned)
        document["content_sha256"] = canonical_sha256(unsigned)
        _revalidate_held_gpu_locks(locks)
        file_sha = _write_exclusive(output_path, document)
        _revalidate_held_gpu_locks(locks)
    finally:
        for _index, descriptor, _identity in reversed(locks):
            os.close(descriptor)
    return {
        "artifact": {"path": str(output_path), "sha256": file_sha},
        **document,
    }


def _validate_lock_identity(value: Any, *, index: int) -> Dict[str, Any]:
    row = _exact_dict(
        value,
        ("path", "device", "inode", "exclusive"),
        name="GPU %d lock receipt" % index,
    )
    path = _normalized_absolute_path(row["path"], name="GPU %d lock path" % index)
    if path != GPU_LOCK_PATHS[index]:
        raise TransitionPreflightRefused("GPU %d lock path differs" % index)
    if (
        type(row["device"]) is not int
        or isinstance(row["device"], bool)
        or row["device"] < 0
        or type(row["inode"]) is not int
        or isinstance(row["inode"], bool)
        or row["inode"] <= 0
        or row["exclusive"] is not True
    ):
        raise TransitionPreflightRefused("GPU %d lock identity is invalid" % index)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except OSError as exc:
        raise TransitionPreflightRefused(
            "GPU %d lock receipt cannot be re-opened" % index
        ) from exc
    expected = (row["device"], row["inode"])
    if (
        not stat.S_ISREG(before.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (before.st_dev, before.st_ino) != expected
        or (opened.st_dev, opened.st_ino) != expected
        or (after.st_dev, after.st_ino) != expected
    ):
        raise TransitionPreflightRefused("GPU %d lock inode changed" % index)
    return row


def _validate_document(
    document: Any, *, checkout: Path, commit: str
) -> Dict[str, Any]:
    row = _exact_dict(
        document,
        (
            "schema_version",
            "kind",
            "status",
            "diagnostic_unauthorized",
            "machine_verified",
            "host",
            "source",
            "writer_policy",
            "gpus",
            "targets",
            "content_sha256",
        ),
        name="transition preflight receipt",
    )
    if (
        type(row["schema_version"]) is not int
        or isinstance(row["schema_version"], bool)
        or row["schema_version"] != SCHEMA_VERSION
        or row["kind"] != KIND
        or row["status"] != "PASS"
        or row["diagnostic_unauthorized"] is not True
    ):
        raise TransitionPreflightRefused("transition preflight identity differs")
    machine = _exact_dict(
        row["machine_verified"],
        (
            "common_cut_before_first_scale4096",
            "legacy_live_or_pending_count",
            "cross_time_atomicity_claimed",
            "cross_checkout_legacy_pending_completeness_claimed",
        ),
        name="transition machine verification",
    )
    if (
        machine["common_cut_before_first_scale4096"] is not True
        or type(machine["legacy_live_or_pending_count"]) is not int
        or isinstance(machine["legacy_live_or_pending_count"], bool)
        or machine["legacy_live_or_pending_count"] != 0
        or machine["cross_time_atomicity_claimed"] is not False
        or machine[
            "cross_checkout_legacy_pending_completeness_claimed"
        ]
        is not False
    ):
        raise TransitionPreflightRefused(
            "transition machine verification semantics differ"
        )
    host = _exact_dict(
        row["host"], ("boot_id", "observed_at"), name="transition host"
    )
    if (
        type(host["boot_id"]) is not str
        or BOOT_ID_RE.fullmatch(host["boot_id"]) is None
        or type(host["observed_at"]) is not str
        or OBSERVED_AT_RE.fullmatch(host["observed_at"]) is None
        or host["boot_id"] != _boot_id()
    ):
        raise TransitionPreflightRefused(
            "transition receipt belongs to another or invalid host boot"
        )
    source = _exact_dict(
        row["source"],
        ("checkout", "commit_sha", "clean", "runtime_sources"),
        name="transition source",
    )
    current_source = _source_document(checkout, commit)
    if source["clean"] is not True or source != current_source:
        raise TransitionPreflightRefused(
            "transition source or runtime-source bytes differ"
        )
    policy = _exact_dict(
        row["writer_policy"],
        ("allowed_claim_kinds", "observed_live_writers"),
        name="transition writer policy",
    )
    if policy != {
        "allowed_claim_kinds": list(ALLOWED_CLAIM_KINDS),
        "observed_live_writers": [],
    }:
        raise TransitionPreflightRefused("transition writer policy differs")
    if type(row["gpus"]) is not list or len(row["gpus"]) != 3:
        raise TransitionPreflightRefused("transition GPU row count differs")
    gpu_uuids = {}
    for index, value in enumerate(row["gpus"]):
        gpu = _exact_dict(
            value,
            (
                "index",
                "uuid",
                "role",
                "lock",
                "total_memory_mib",
                "free_memory_mib",
                "minimum_free_memory_mib",
                "compute_processes",
                "live_reservations",
            ),
            name="transition GPU %d" % index,
        )
        total = gpu["total_memory_mib"]
        free = gpu["free_memory_mib"]
        if (
            type(gpu["index"]) is not int
            or isinstance(gpu["index"], bool)
            or gpu["index"] != index
            or type(gpu["uuid"]) is not str
            or GPU_UUID_RE.fullmatch(gpu["uuid"]) is None
            or gpu["role"] != GPU_ROLES[index]
            or type(total) is not int
            or isinstance(total, bool)
            or total <= 0
            or type(free) is not int
            or isinstance(free, bool)
            or free < MINIMUM_FREE_MEMORY_MIB
            or free > total
            or gpu["minimum_free_memory_mib"] != MINIMUM_FREE_MEMORY_MIB
            or gpu["compute_processes"] != []
            or gpu["live_reservations"] != []
        ):
            raise TransitionPreflightRefused(
                "transition GPU %d semantics differ" % index
            )
        _validate_lock_identity(gpu["lock"], index=index)
        gpu_uuids[index] = gpu["uuid"]
    if len(set(gpu_uuids.values())) != 3:
        raise TransitionPreflightRefused("transition GPU UUIDs are not unique")
    if type(row["targets"]) is not list or len(row["targets"]) != 4:
        raise TransitionPreflightRefused("transition target row count differs")
    seen_namespaces = set()
    for expected, value in zip(TARGET_SPECS, row["targets"]):
        short, cell_id, family, gpu_index, experiment_name = expected
        target = _exact_dict(
            value,
            (
                "cell_id",
                "family",
                "gpu_index",
                "gpu_uuid",
                "namespace",
                "namespace_absent",
            ),
            name="transition target %s" % short,
        )
        namespace = _normalized_absolute_path(
            target["namespace"], name="transition target %s namespace" % short
        )
        parent = _target_parent(checkout, experiment_name)
        _real_directory(parent, name="transition target %s parent" % short)
        if (
            target["cell_id"] != cell_id
            or target["family"] != family
            or type(target["gpu_index"]) is not int
            or isinstance(target["gpu_index"], bool)
            or target["gpu_index"] != gpu_index
            or target["gpu_uuid"] != gpu_uuids[gpu_index]
            or target["namespace_absent"] is not True
            or namespace.parent != parent
            or str(namespace) in seen_namespaces
        ):
            raise TransitionPreflightRefused(
                "transition target %s semantics differ" % short
            )
        seen_namespaces.add(str(namespace))
    content_sha = _sha256(
        row["content_sha256"], name="transition receipt content SHA"
    )
    unsigned = dict(row)
    unsigned.pop("content_sha256")
    if canonical_sha256(unsigned) != content_sha:
        raise TransitionPreflightRefused(
            "transition receipt content SHA differs"
        )
    return row


def validate_receipt(
    path: Any,
    expected_sha256: str,
    checkout: Any,
    commit: str,
) -> Dict[str, Any]:
    """Validate a receipt against this boot and the exact clean source checkout.

    This deliberately does not claim that the GPUs remain empty after the
    producer released its locks.  Launchers must still perform their immediate
    VendorV2 admission and no-clobber namespace operation.
    """

    receipt_path = _normalized_absolute_path(path, name="transition receipt")
    expected_sha = _sha256(expected_sha256, name="transition receipt file SHA")
    checkout_path = _normalized_absolute_path(checkout, name="source checkout")
    exact_commit = _commit_sha(commit)
    document, observed_sha = _stable_canonical_json(
        receipt_path, name="transition preflight receipt"
    )
    if observed_sha != expected_sha:
        raise TransitionPreflightRefused(
            "transition preflight receipt file SHA differs"
        )
    validated = _validate_document(
        document, checkout=checkout_path, commit=exact_commit
    )
    return {
        "artifact": {"path": str(receipt_path), "sha256": observed_sha},
        **validated,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--gpu0-uuid", required=True)
    parser.add_argument("--gpu1-uuid", required=True)
    parser.add_argument("--gpu2-uuid", required=True)
    parser.add_argument("--a0-namespace", required=True)
    parser.add_argument("--a1-namespace", required=True)
    parser.add_argument("--c0-namespace", required=True)
    parser.add_argument("--c1-namespace", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = produce_receipt(
        checkout=args.checkout,
        commit_sha=args.commit_sha,
        gpu_uuids={
            0: args.gpu0_uuid,
            1: args.gpu1_uuid,
            2: args.gpu2_uuid,
        },
        namespaces={
            "a0": args.a0_namespace,
            "a1": args.a1_namespace,
            "c0": args.c0_namespace,
            "c1": args.c1_namespace,
        },
        output=args.output,
    )
    print(
        json.dumps(
            {
                "path": result["artifact"]["path"],
                "sha256": result["artifact"]["sha256"],
                "content_sha256": result["content_sha256"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except TransitionPreflightRefused as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
