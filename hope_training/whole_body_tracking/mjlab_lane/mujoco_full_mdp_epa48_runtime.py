"""Bind Full-A to the exact project-built MuJoCo-Warp EPA48 wheel.

This module closes only runtime package identity.  It does not establish that
EPA48 fixes a collision, authorize training, or make a checkpoint resumable.
The returned identity is deliberately MuJoCo-Warp-only; the companion exact
RSL-RL wheel remains a process-local prerequisite checked here and by the
runner's existing RSL source/origin gate.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import os
from pathlib import Path
import stat
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = REPO_ROOT / "vendor_assets" / "mujoco_warp_epa48_1"
BUILD_RECEIPT_PATH = ARTIFACT_ROOT / "build_receipt.json"
EPA48_WHEEL_PATH = (
    ARTIFACT_ROOT
    / "wheelhouse"
    / "mujoco_warp-3.10.0.3+hope.epa48.1-py3-none-any.whl"
)
RSL3_WHEEL_PATH = (
    REPO_ROOT
    / "vendor_assets"
    / "rsl_rl_3_1_2"
    / "rsl_rl_lib-3.1.2-py3-none-any.whl"
)

EPA48_VERSION = "3.10.0.3+hope.epa48.1"
EPA48_TYPES_SHA256 = (
    "391e421eeede84389d6c7daeae39b19ce43132d29c11f7f3c328a50011c7a696"
)
EPA48_WHEEL_SHA256 = (
    "58f47b1c3b4249d82666f25d3a302ff5a215043a3d7a3b9445a5ca7ef15b561a"
)
BUILD_RECEIPT_SHA256 = (
    "336f6454296d3c062e26fb0c330d6dbca4b2fd0ad4e50f386f8a647db013e041"
)
RSL3_VERSION = "3.1.2"
RSL3_WHEEL_SHA256 = (
    "406867356b70920e99ed8fd12c5b3463a64895407cc3ed96c917fddb9bfae06d"
)

_RUNTIME_IDENTITY = {
    "schema_version": 1,
    "distribution": "mujoco-warp",
    "fork_id": "hope_mujoco_warp_epa48_v1",
    "version": EPA48_VERSION,
    "epa_horizon": 48,
    "types_py_sha256": EPA48_TYPES_SHA256,
    "wheel_sha256": EPA48_WHEEL_SHA256,
    "build_receipt_sha256": BUILD_RECEIPT_SHA256,
    "import_scope": "fresh_run_local_site",
}
_PRELOAD_PREFIXES = ("mujoco_warp", "rsl_rl", "mjlab")
_EXPECTED_DISTRIBUTIONS = (
    ("mujoco-warp", EPA48_VERSION, "mujoco_warp"),
    ("rsl-rl-lib", RSL3_VERSION, "rsl_rl"),
)


class RuntimeBindingError(RuntimeError):
    """The Full-A process cannot prove its exact runtime package identity."""


def expected_mujoco_warp_runtime_identity() -> dict:
    """Return the immutable build/runtime identity recorded by Full-A."""

    return dict(_RUNTIME_IDENTITY)


def _stable_regular_bytes(path: Path, label: str) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeBindingError(f"{label} is missing or not a regular file") from exc
    try:
        before = os.fstat(fd)
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        current = os.stat(path, follow_symlinks=False)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or not stable
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
            or path.resolve(strict=True) != path
        ):
            raise RuntimeBindingError(f"{label} is not one stable canonical file")
        return b"".join(chunks)
    except OSError as exc:
        raise RuntimeBindingError(f"cannot read stable {label}") from exc
    finally:
        os.close(fd)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_fresh_site_target(site: Path) -> None:
    if not isinstance(site, Path) or not site.is_absolute() or site.name in ("", ".", ".."):
        raise RuntimeBindingError("runtime site must be one absolute fresh path")
    if str(site) in sys.path:
        raise RuntimeBindingError("runtime site is already present in sys.path")
    if os.path.lexists(os.fspath(site)):
        raise RuntimeBindingError("runtime site already exists or is a symlink")
    try:
        parent_stat = site.parent.lstat()
        canonical_parent = site.parent.resolve(strict=True)
    except OSError as exc:
        raise RuntimeBindingError("runtime site parent is missing") from exc
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or canonical_parent != site.parent
    ):
        raise RuntimeBindingError("runtime site parent is not one canonical directory")


def _reject_preloaded_runtime() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in _PRELOAD_PREFIXES)
    )
    if loaded:
        raise RuntimeBindingError("Full-A runtime packages were preloaded: " + loaded[0])


def _verified_wheel_payloads() -> tuple[bytes, bytes]:
    # The fixed wheel SHAs are execution roots.  The receipt is reviewed
    # provenance evidence; runtime does not reconstruct the source build.
    receipt = _stable_regular_bytes(BUILD_RECEIPT_PATH, "EPA48 build receipt")
    epa48_wheel = _stable_regular_bytes(EPA48_WHEEL_PATH, "EPA48 wheel")
    rsl3_wheel = _stable_regular_bytes(RSL3_WHEEL_PATH, "RSL-RL 3 wheel")
    if _sha256(receipt) != BUILD_RECEIPT_SHA256:
        raise RuntimeBindingError("EPA48 build receipt SHA differs")
    if _sha256(epa48_wheel) != EPA48_WHEEL_SHA256:
        raise RuntimeBindingError("EPA48 wheel SHA differs")
    if _sha256(rsl3_wheel) != RSL3_WHEEL_SHA256:
        raise RuntimeBindingError("RSL-RL 3 wheel SHA differs")
    return rsl3_wheel, epa48_wheel


def _extract_exact_wheel(payload: bytes, site: Path, label: str) -> None:
    """Extract private bytes that already matched one fixed wheel SHA."""

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(site)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise RuntimeBindingError(f"cannot extract {label}") from exc


def _is_origin_under_site(value: object, site: Path) -> bool:
    raw = getattr(value, "__file__", None)
    if type(raw) is not str or not raw:
        return False
    try:
        Path(raw).resolve(strict=True).relative_to(site)
        return True
    except (OSError, ValueError):
        return False


def _distribution_root(distribution) -> Path:
    try:
        return Path(distribution.locate_file("")).resolve(strict=True)
    except (AttributeError, OSError, TypeError) as exc:
        raise RuntimeBindingError("runtime distribution root differs") from exc


def _require_site_candidates(site: Path) -> None:
    for name, expected_version, package in _EXPECTED_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeBindingError("runtime distribution is missing: " + name) from exc
        if (
            distribution.version != expected_version
            or _distribution_root(distribution) != site
        ):
            raise RuntimeBindingError("runtime distribution origin/version differs: " + name)
        spec = importlib.util.find_spec(package)
        origin = None if spec is None else spec.origin
        try:
            Path(origin or "").resolve(strict=True).relative_to(site)
        except (OSError, ValueError):
            raise RuntimeBindingError("runtime import candidate is foreign: " + package)


def _import_runtime_modules(_site: Path):
    package = importlib.import_module("mujoco_warp")
    types_module = importlib.import_module("mujoco_warp._src.types")
    return package, types_module


def _require_loaded_runtime(site: Path, package, types_module) -> None:
    types_path = Path(getattr(types_module, "__file__", ""))
    if (
        getattr(package, "__version__", None) != EPA48_VERSION
        or type(getattr(types_module, "MJ_MAX_EPAHORIZON", None)) is not int
        or types_module.MJ_MAX_EPAHORIZON != 48
        or not _is_origin_under_site(package, site)
        or not _is_origin_under_site(types_module, site)
        or _sha256(_stable_regular_bytes(
            types_path, "loaded EPA48 types.py"
        )) != EPA48_TYPES_SHA256
    ):
        raise RuntimeBindingError("loaded MuJoCo-Warp version/horizon/origin differs")
    foreign = sorted(
        name
        for name, module in sys.modules.items()
        if (name == "mujoco_warp" or name.startswith("mujoco_warp."))
        and not _is_origin_under_site(module, site)
    )
    if foreign:
        raise RuntimeBindingError("loaded MuJoCo-Warp module is foreign: " + foreign[0])


def bind_fresh_epa48_runtime_site(runtime_site: Path) -> dict:
    """Bind an exact dual-wheel site and return the EPA-only evidence identity."""

    _validate_fresh_site_target(runtime_site)
    _reject_preloaded_runtime()
    rsl3_wheel, epa48_wheel = _verified_wheel_payloads()

    try:
        os.mkdir(runtime_site, 0o700)
    except OSError as exc:
        raise RuntimeBindingError("cannot create fresh runtime site") from exc

    # Extract the already-hashed bytes, rather than reopening either wheel after
    # verification.  A failed site is deliberately left spent and cannot be reused.
    _extract_exact_wheel(rsl3_wheel, runtime_site, "RSL-RL 3 wheel")
    _extract_exact_wheel(epa48_wheel, runtime_site, "EPA48 wheel")

    prior_sys_path = list(sys.path)
    sys.path.insert(0, str(runtime_site))
    importlib.invalidate_caches()
    try:
        _require_site_candidates(runtime_site)
        package, types_module = _import_runtime_modules(runtime_site)
        _require_loaded_runtime(runtime_site, package, types_module)
    except Exception:
        sys.path[:] = prior_sys_path
        for name in tuple(sys.modules):
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in _PRELOAD_PREFIXES
            ):
                sys.modules.pop(name, None)
        importlib.invalidate_caches()
        raise
    return expected_mujoco_warp_runtime_identity()
