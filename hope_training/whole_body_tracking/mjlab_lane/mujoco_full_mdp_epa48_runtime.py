"""Bind Full-A to the exact project-built MuJoCo-Warp EPA48 wheel.

This module closes only runtime package identity.  It does not establish that
EPA48 fixes a collision, authorize training, or make a checkpoint resumable.
One cold pre-import verification reads the exact MuJoCo-Warp/RSL-RL wheel bytes
and measures the selected installed MJLab tree.  The runner can then consume
those already-verified bytes without hashing either wheel a second time.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
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
MJLAB_DISTRIBUTION = "mjlab"
MJLAB_VERSION = "1.5.3"
MJLAB_SELECTED_FILE_COUNT = 193
MJLAB_SELECTED_BYTE_COUNT = 1_399_177
MJLAB_SELECTED_TREE_SHA256 = (
    "88c9725d0416b4ac3e21f6752ad423c13ea3b8cfb9e23ca664f8aba146cec33d"
)
MJLAB_TASK_ENTRY_POINT_GROUP = "mjlab.tasks"

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
_RSL_RL_RUNTIME_IDENTITY = {
    "distribution": "rsl-rl-lib",
    "version": RSL3_VERSION,
    "wheel_sha256": RSL3_WHEEL_SHA256,
    "import_scope": "fresh_run_local_site",
}
_PRELOAD_PREFIXES = ("mujoco_warp", "rsl_rl", "mjlab")
_EXPECTED_DISTRIBUTIONS = (
    ("mujoco-warp", EPA48_VERSION, "mujoco_warp"),
    ("rsl-rl-lib", RSL3_VERSION, "rsl_rl"),
)


class RuntimeBindingError(RuntimeError):
    """The Full-A process cannot prove its exact runtime package identity."""


class _VerifiedRuntimeStackPreimport:
    """Opaque result of one complete cold runtime-stack verification."""

    __slots__ = ("_mjlab_identity", "_rsl3_wheel", "_epa48_wheel")

    def __init__(self, mjlab_identity: dict, rsl3_wheel: bytes, epa48_wheel: bytes):
        self._mjlab_identity = dict(mjlab_identity)
        self._rsl3_wheel = rsl3_wheel
        self._epa48_wheel = epa48_wheel


def expected_mujoco_warp_runtime_identity() -> dict:
    """Return the immutable build/runtime identity recorded by Full-A."""

    return dict(_RUNTIME_IDENTITY)


def expected_rsl_rl_runtime_identity() -> dict:
    """Return the path-free identity of the exact local RSL-RL wheel."""

    return dict(_RSL_RL_RUNTIME_IDENTITY)


def expected_mjlab_runtime_identity() -> dict:
    """Return the path-free identity of the selected installed MJLab tree."""

    return {
        "schema_version": 1,
        "distribution": MJLAB_DISTRIBUTION,
        "version": MJLAB_VERSION,
        "import_scope": "verified_venv_distribution",
        "selected_tree_scope": "mjlab/**/*.py+mjlab/scene/scene.xml",
        "selected_file_count": MJLAB_SELECTED_FILE_COUNT,
        "selected_byte_count": MJLAB_SELECTED_BYTE_COUNT,
        "selected_tree_sha256": MJLAB_SELECTED_TREE_SHA256,
        "mjlab_tasks_entry_point_count": 0,
    }


def _runtime_stack_identity(mjlab_identity: dict) -> dict:
    return {
        "schema_version": 1,
        "mujoco_warp": expected_mujoco_warp_runtime_identity(),
        "rsl_rl": expected_rsl_rl_runtime_identity(),
        "mjlab": dict(mjlab_identity),
    }


def expected_runtime_stack_identity() -> dict:
    """Return the exact path-free runtime-stack wire identity."""

    return _runtime_stack_identity(expected_mjlab_runtime_identity())


def verified_runtime_stack_identity(verification) -> dict:
    """Clone the identity produced by one successful cold verification."""

    if type(verification) is not _VerifiedRuntimeStackPreimport:
        raise RuntimeBindingError("runtime-stack pre-import verification is missing")
    return _runtime_stack_identity(verification._mjlab_identity)


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


def _stat_fingerprint(info: os.stat_result) -> tuple:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _require_canonical_directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeBindingError(f"{label} is missing") from exc
    if (
        not path.is_absolute()
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or resolved != path
    ):
        raise RuntimeBindingError(f"{label} is not one canonical directory")
    return resolved


def _stable_tree_file_bytes(path: Path, label: str) -> bytes:
    """Read one selected tree file without following an alias or a replacement."""

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeBindingError(f"{label} is missing, linked, or unreadable") from exc
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
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_fingerprint(before) != _stat_fingerprint(after)
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
            or path.resolve(strict=True) != path
        ):
            raise RuntimeBindingError(f"{label} is not one stable canonical file")
        return b"".join(chunks)
    except OSError as exc:
        raise RuntimeBindingError(f"cannot read stable {label}") from exc
    finally:
        os.close(fd)


def _enumerate_mjlab_selected_tree(
    distribution_root: Path, package_root: Path
) -> tuple:
    """Enumerate selected files without following any link inside MJLab."""

    selected = []
    pending = [package_root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise RuntimeBindingError("cannot enumerate installed MJLab tree") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeBindingError("installed MJLab tree changed during enumeration") from exc
            if stat.S_ISLNK(info.st_mode):
                raise RuntimeBindingError("installed MJLab tree contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                continue
            try:
                package_relative = path.relative_to(package_root).as_posix()
                distribution_relative = path.relative_to(distribution_root).as_posix()
            except ValueError as exc:
                raise RuntimeBindingError("installed MJLab entry escaped its package root") from exc
            is_selected = (
                package_relative.endswith(".py")
                or package_relative == "scene/scene.xml"
            )
            if not is_selected:
                continue
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeBindingError("selected MJLab entry is not a regular file")
            selected.append(
                (distribution_relative, path, _stat_fingerprint(info))
            )
    return tuple(sorted(selected, key=lambda item: item[0]))


def _measure_mjlab_selected_tree(
    distribution_root: Path, package_root: Path
) -> tuple:
    first = _enumerate_mjlab_selected_tree(distribution_root, package_root)
    items = []
    for relative, path, _fingerprint in first:
        payload = _stable_tree_file_bytes(path, "installed MJLab file " + relative)
        items.append([relative, len(payload), _sha256(payload)])
    second = _enumerate_mjlab_selected_tree(distribution_root, package_root)
    if first != second:
        raise RuntimeBindingError("installed MJLab tree changed across enumeration passes")
    encoded = json.dumps(
        items, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        len(items),
        sum(item[1] for item in items),
        _sha256(encoded),
        frozenset(path for _relative, path, _fingerprint in first),
    )


def _mjlab_task_entry_points() -> tuple:
    try:
        available = importlib.metadata.entry_points()
        if hasattr(available, "select"):
            selected = available.select(group=MJLAB_TASK_ENTRY_POINT_GROUP)
        elif isinstance(available, dict):
            selected = available.get(MJLAB_TASK_ENTRY_POINT_GROUP, ())
        else:
            selected = (
                item for item in available
                if getattr(item, "group", None) == MJLAB_TASK_ENTRY_POINT_GROUP
            )
        return tuple(selected)
    except Exception as exc:
        raise RuntimeBindingError("cannot enumerate MJLab task entry points") from exc


def _require_no_mjlab_task_entry_points() -> None:
    found = _mjlab_task_entry_points()
    if found:
        first = found[0]
        name = getattr(first, "name", "<unnamed>")
        raise RuntimeBindingError(
            "ambient mjlab.tasks entry point is forbidden: " + str(name)
        )


def _canonical_mjlab_roots() -> tuple:
    try:
        distribution = importlib.metadata.distribution(MJLAB_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeBindingError("MJLab runtime distribution is missing") from exc
    if distribution.version != MJLAB_VERSION:
        raise RuntimeBindingError("MJLab runtime distribution version differs")
    try:
        raw_distribution_root = Path(distribution.locate_file(""))
    except (AttributeError, TypeError) as exc:
        raise RuntimeBindingError("MJLab distribution root differs") from exc
    distribution_root = _require_canonical_directory(
        raw_distribution_root, "MJLab distribution root"
    )
    package_root = _require_canonical_directory(
        distribution_root / "mjlab", "MJLab package root"
    )
    try:
        spec = importlib.util.find_spec("mjlab")
    except (ImportError, AttributeError, ValueError) as exc:
        raise RuntimeBindingError("MJLab import candidate cannot be resolved") from exc
    origin = None if spec is None else spec.origin
    try:
        origin_path = Path(origin or "")
        resolved_origin = origin_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeBindingError("MJLab import candidate origin differs") from exc
    expected_origin = package_root / "__init__.py"
    if (
        spec is None
        or not origin_path.is_absolute()
        or origin_path != resolved_origin
        or resolved_origin != expected_origin
    ):
        raise RuntimeBindingError("MJLab import candidate origin differs")
    locations = getattr(spec, "submodule_search_locations", None)
    try:
        resolved_locations = tuple(
            Path(value).resolve(strict=True) for value in locations
        )
    except (OSError, TypeError) as exc:
        raise RuntimeBindingError("MJLab package search root differs") from exc
    if resolved_locations != (package_root,):
        raise RuntimeBindingError("MJLab package search root differs")
    return distribution_root, package_root


def _verify_mjlab_selected_tree(
    distribution_root: Path, package_root: Path
) -> tuple:
    count, byte_count, tree_sha256, selected_paths = _measure_mjlab_selected_tree(
        distribution_root, package_root
    )
    if count != MJLAB_SELECTED_FILE_COUNT:
        raise RuntimeBindingError("MJLab selected code-tree file count differs")
    if byte_count != MJLAB_SELECTED_BYTE_COUNT:
        raise RuntimeBindingError("MJLab selected code-tree byte count differs")
    if tree_sha256 != MJLAB_SELECTED_TREE_SHA256:
        raise RuntimeBindingError("MJLab selected code-tree SHA differs")
    return expected_mjlab_runtime_identity(), selected_paths


def verify_mjlab_runtime_preimport() -> dict:
    """Verify the exact ambient MJLab install before importing its package."""

    if any(name == "mjlab" or name.startswith("mjlab.") for name in sys.modules):
        raise RuntimeBindingError("MJLab runtime package was already imported")
    distribution_root, package_root = _canonical_mjlab_roots()
    _require_no_mjlab_task_entry_points()
    identity, _selected_paths = _verify_mjlab_selected_tree(
        distribution_root, package_root
    )
    return identity


def _require_loaded_mjlab_module_origins(
    package_root: Path, selected_paths: frozenset
) -> None:
    loaded = sorted(
        (name, module)
        for name, module in sys.modules.items()
        if name == "mjlab" or name.startswith("mjlab.")
    )
    if not loaded or loaded[0][0] != "mjlab":
        raise RuntimeBindingError("MJLab runtime package is not loaded")
    for name, module in loaded:
        raw_file = getattr(module, "__file__", None)
        if type(raw_file) is not str or not raw_file:
            raise RuntimeBindingError("loaded MJLab module has no file origin: " + name)
        try:
            file_path = Path(raw_file)
            resolved_file = file_path.resolve(strict=True)
            resolved_file.relative_to(package_root)
        except (OSError, ValueError):
            raise RuntimeBindingError("loaded MJLab module is foreign: " + name)
        if not file_path.is_absolute() or file_path != resolved_file:
            raise RuntimeBindingError("loaded MJLab module is foreign: " + name)
        if resolved_file not in selected_paths:
            raise RuntimeBindingError(
                "loaded MJLab module is outside the selected code tree: " + name
            )
        module_spec = getattr(module, "__spec__", None)
        spec_origin = getattr(module_spec, "origin", None)
        if type(spec_origin) is not str or spec_origin != raw_file:
            raise RuntimeBindingError("loaded MJLab module spec is foreign: " + name)


def verify_loaded_mjlab_runtime_modules() -> dict:
    """Re-verify the tree and prove every loaded MJLab module came from it."""

    distribution_root, package_root = _canonical_mjlab_roots()
    _require_no_mjlab_task_entry_points()
    identity, selected_paths = _verify_mjlab_selected_tree(
        distribution_root, package_root
    )
    _require_loaded_mjlab_module_origins(package_root, selected_paths)
    return identity


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


def verify_runtime_stack_preimport():
    """Measure every cold runtime input before importing a runtime package.

    The opaque result owns the exact wheel payloads that were hashed.  Passing
    it to :func:`bind_fresh_epa48_runtime_site` makes extraction consume those
    bytes directly instead of reopening either wheel.
    """

    _reject_preloaded_runtime()
    mjlab_identity = verify_mjlab_runtime_preimport()
    rsl3_wheel, epa48_wheel = _verified_wheel_payloads()
    return _VerifiedRuntimeStackPreimport(
        mjlab_identity, rsl3_wheel, epa48_wheel
    )


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


def bind_fresh_epa48_runtime_site(
    runtime_site: Path, *, preimport_verification=None
) -> dict:
    """Bind an exact dual-wheel site and return its complete runtime identity."""

    _validate_fresh_site_target(runtime_site)
    verification = (
        verify_runtime_stack_preimport()
        if preimport_verification is None
        else preimport_verification
    )
    if type(verification) is not _VerifiedRuntimeStackPreimport:
        raise RuntimeBindingError("runtime-stack pre-import verification is missing")
    _reject_preloaded_runtime()
    rsl3_wheel = verification._rsl3_wheel
    epa48_wheel = verification._epa48_wheel

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
    return verified_runtime_stack_identity(verification)
