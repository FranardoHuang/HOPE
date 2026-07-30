#!/usr/bin/env python3
"""Mint and verify the exact Python/IsaacLab runtime used by action-ball runs.

The receipt is deliberately independent of Isaac Lab and of the training
package.  ``mint`` executes the requested Python only for a small isolated
stdlib probe, inventories the resulting import closure, proves that the
IsaacLab checkout is clean, and publishes one immutable JSON receipt.
``verify`` repeats the complete inventory from the paths in that receipt and
requires byte-for-byte equality.

This is an identity tool, not an installer.  It never repairs an environment,
changes a checkout, follows a receipt symlink, or overwrites an artifact.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import urllib.parse
import urllib.request


SCHEMA_VERSION = 2
RECEIPT_KIND = "action_ball_runtime_inventory_v2"
PROBE_SENTINEL = "ACTION_BALL_RUNTIME_INVENTORY_PROBE_V2:"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
HERE = Path(__file__).resolve().parent
INVENTORY_ENTRYPOINT = Path(__file__).resolve()
NOSITE_BOOTSTRAP = HERE / "action_ball_python_nosite_bootstrap.py"

# The order is part of the contract.  Preferred distribution names are only
# used to obtain a version without importing GPU/runtime packages.
MODULE_CONTRACT: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("isaacsim", ("isaacsim",)),
    ("torch", ("torch",)),
    ("isaaclab", ("isaaclab",)),
    ("hydra", ("hydra-core",)),
    ("omegaconf", ("omegaconf",)),
    ("packaging", ("packaging",)),
    ("numpy", ("numpy",)),
    ("warp", ("warp-lang", "warp")),
    ("gymnasium", ("gymnasium",)),
    ("rsl_rl", ("rsl-rl-lib", "rsl-rl", "rsl_rl")),
    (
        "whole_body_tracking",
        ("whole_body_tracking", "whole-body-tracking"),
    ),
)
MODULE_NAMES = tuple(item[0] for item in MODULE_CONTRACT)
OPTIONAL_DISTRIBUTION_NAMES = ("tensordict",)
CRITICAL_RECORD_WITNESSES = ("carb", "omni")
NORMALIZED_DISTRIBUTION_RE = re.compile(r"[-_.]+")
TOP_LEVEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_LSTAT_KEYS = {
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
}
_ANCESTOR_LSTAT_KEYS = {
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
}


class RuntimeInventoryError(RuntimeError):
    """The live runtime cannot satisfy the frozen inventory contract."""


def _load_nosite_bootstrap():
    """Load the sibling stdlib bootstrap without relying on ``sys.path``."""

    spec = importlib.util.spec_from_file_location(
        "_action_ball_python_nosite_bootstrap_inventory",
        NOSITE_BOOTSTRAP,
    )
    if spec is None or spec.loader is None:
        raise RuntimeInventoryError(
            "cannot construct the no-site bootstrap module loader"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeInventoryError(
            "cannot load the no-site bootstrap source"
        ) from exc
    return module


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _bytes_snapshot(raw: bytes) -> Dict[str, Any]:
    return {
        "byte_count": len(raw),
        "sha256": _sha256_bytes(raw),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
    }


def _validate_bytes_snapshot(value: Any, label: str) -> bytes:
    row = _require_mapping(value, label)
    _require_exact_keys(
        row, {"byte_count", "sha256", "raw_base64"}, label
    )
    count = _require_plain_int(row["byte_count"], "%s.byte_count" % label)
    digest = _require_sha256(row["sha256"], "%s.sha256" % label)
    encoded = row["raw_base64"]
    if type(encoded) is not str:
        raise RuntimeInventoryError("%s.raw_base64 must be a string" % label)
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeInventoryError(
            "%s.raw_base64 is not canonical base64" % label
        ) from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise RuntimeInventoryError(
            "%s.raw_base64 is not canonical base64" % label
        )
    if len(raw) != count or _sha256_bytes(raw) != digest:
        raise RuntimeInventoryError(
            "%s bytes differ from byte_count/sha256" % label
        )
    return raw


def _reject_duplicate_pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeInventoryError("receipt JSON contains duplicate key: %s" % key)
        result[key] = value
    return result


def _strict_json_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeInventoryError("%s is not UTF-8" % label) from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeInventoryError(
                    "%s contains forbidden JSON constant %s" % (label, value)
                )
            ),
        )
    except RuntimeInventoryError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeInventoryError("%s is not strict JSON" % label) from exc


def _require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    actual = set(value.keys())
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise RuntimeInventoryError(
            "%s keys differ (missing=%s extra=%s)" % (label, missing, extra)
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeInventoryError("%s must be an object" % label)
    return value


def _require_plain_int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeInventoryError("%s must be an integer >= %d" % (label, minimum))
    return value


def _require_nonempty_line(value: Any, label: str, maximum: int = 4096) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise RuntimeInventoryError("%s must be one bounded non-empty line" % label)
    return value


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise RuntimeInventoryError("%s must be a lowercase SHA-256" % label)
    return value


def _normalized_distribution_name(value: str) -> str:
    return NORMALIZED_DISTRIBUTION_RE.sub("-", value).lower()


def _canonical_absolute_path(value: Any, label: str) -> Path:
    text = _require_nonempty_line(value, label, maximum=16384)
    if not os.path.isabs(text):
        raise RuntimeInventoryError("%s must be absolute" % label)
    normalized = os.path.normpath(text)
    if text != normalized:
        raise RuntimeInventoryError("%s must already be lexically normalized" % label)
    return Path(text)


def _lstat_dict(info: os.stat_result) -> Dict[str, int]:
    return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "st_mode": int(info.st_mode),
        "st_nlink": int(info.st_nlink),
        "st_uid": int(info.st_uid),
        "st_gid": int(info.st_gid),
        "st_size": int(info.st_size),
        "st_mtime_ns": int(info.st_mtime_ns),
        "st_ctime_ns": int(info.st_ctime_ns),
    }


def _ancestor_lstat_dict(info: os.stat_result) -> Dict[str, int]:
    # Ancestor directory mtime/ctime/size/nlink legitimately change when an
    # unrelated process creates a sibling.  Bind the path-walk identity and
    # ownership instead; bind symlink text separately below.
    return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "st_mode": int(info.st_mode),
        "st_uid": int(info.st_uid),
        "st_gid": int(info.st_gid),
    }


def _kind_from_mode(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "regular"
    return "other"


def _same_lstat(left: os.stat_result, right: os.stat_result) -> bool:
    return _lstat_dict(left) == _lstat_dict(right)


def _ancestor_paths(path: Path) -> List[Path]:
    parent = path.parent
    if not parent.is_absolute():
        raise RuntimeInventoryError("internal error: ancestor path is not absolute")
    rows: List[Path] = [Path(parent.anchor)]
    current = Path(parent.anchor)
    for part in parent.parts[1:]:
        current = current / part
        rows.append(current)
    return rows


def _ancestor_snapshots(path: Path) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for ancestor in _ancestor_paths(path):
        try:
            info = ancestor.lstat()
        except OSError as exc:
            raise RuntimeInventoryError(
                "ancestor is missing while inventorying %s: %s" % (path, ancestor)
            ) from exc
        kind = _kind_from_mode(info.st_mode)
        if kind not in {"directory", "symlink"}:
            raise RuntimeInventoryError(
                "ancestor is neither directory nor symlink: %s" % ancestor
            )
        row: Dict[str, Any] = {
            "path": str(ancestor),
            "kind": kind,
            "lstat": _ancestor_lstat_dict(info),
        }
        if kind == "symlink":
            try:
                row["link_text"] = os.readlink(str(ancestor))
            except OSError as exc:
                raise RuntimeInventoryError(
                    "cannot read ancestor symlink: %s" % ancestor
                ) from exc
        result.append(row)
    return result


def _open_regular_snapshot(
    path: Path,
    label: str,
    *,
    executable: bool = False,
    include_raw: bool = False,
) -> Dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeInventoryError("O_NOFOLLOW is required on this platform")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeInventoryError("%s is not an openable regular file: %s" % (label, path)) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeInventoryError("%s is not regular: %s" % (label, path))
        if executable and before.st_mode & 0o111 == 0:
            raise RuntimeInventoryError("%s is not executable: %s" % (label, path))
        chunks: List[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if not _same_lstat(before, after):
            raise RuntimeInventoryError("%s changed while being read: %s" % (label, path))
    finally:
        os.close(descriptor)
    try:
        final_lstat = path.lstat()
    except OSError as exc:
        raise RuntimeInventoryError("%s disappeared after read: %s" % (label, path)) from exc
    if not _same_lstat(before, final_lstat):
        raise RuntimeInventoryError("%s path identity changed after read: %s" % (label, path))
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeInventoryError("%s byte count differs from stat size: %s" % (label, path))
    result = {
        "path": str(path),
        "lstat": _lstat_dict(before),
        "ancestors": _ancestor_snapshots(path),
        "byte_count": len(raw),
        "sha256": _sha256_bytes(raw),
    }
    if include_raw:
        result["raw_base64"] = base64.b64encode(raw).decode("ascii")
    return result


def _python_chain_snapshot(requested: Path) -> Tuple[List[Dict[str, Any]], Path]:
    current = requested
    seen: set = set()
    rows: List[Dict[str, Any]] = []
    for index in range(64):
        text = str(current)
        if text in seen:
            raise RuntimeInventoryError("Python symlink chain contains a cycle")
        seen.add(text)
        try:
            info = current.lstat()
        except OSError as exc:
            raise RuntimeInventoryError("Python chain entry is missing: %s" % current) from exc
        kind = _kind_from_mode(info.st_mode)
        if kind == "symlink":
            try:
                link_text = os.readlink(str(current))
            except OSError as exc:
                raise RuntimeInventoryError(
                    "cannot read Python symlink: %s" % current
                ) from exc
            if not link_text or "\x00" in link_text:
                raise RuntimeInventoryError("Python symlink has invalid link text: %s" % current)
            rows.append(
                {
                    "index": index,
                    "kind": "symlink",
                    "path": str(current),
                    "link_text": link_text,
                    "lstat": _lstat_dict(info),
                    "ancestors": _ancestor_snapshots(current),
                }
            )
            target = Path(link_text)
            if target.is_absolute():
                next_text = os.path.normpath(link_text)
            else:
                next_text = os.path.normpath(str(current.parent / target))
            if not os.path.isabs(next_text):
                raise RuntimeInventoryError("Python symlink did not resolve to an absolute path")
            current = Path(next_text)
            continue
        if kind != "regular":
            raise RuntimeInventoryError(
                "Python chain must terminate in a regular executable: %s" % current
            )
        file_row = _open_regular_snapshot(current, "Python executable", executable=True)
        rows.append(
            {
                "index": index,
                "kind": "regular",
                **file_row,
            }
        )
        if index == 0:
            raise RuntimeInventoryError("--python must name a symlink, not a regular file")
        return rows, current
    raise RuntimeInventoryError("Python symlink chain exceeds 64 entries")


_PROBE_PROGRAM = r"""
import base64
import importlib.metadata
import importlib.util
import json
import os
import sys
import urllib.parse
import urllib.request
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement

contract = [
    ("isaacsim", ("isaacsim",)),
    ("torch", ("torch",)),
    ("isaaclab", ("isaaclab",)),
    ("hydra", ("hydra-core",)),
    ("omegaconf", ("omegaconf",)),
    ("packaging", ("packaging",)),
    ("numpy", ("numpy",)),
    ("warp", ("warp-lang", "warp")),
    ("gymnasium", ("gymnasium",)),
    ("rsl_rl", ("rsl-rl-lib", "rsl-rl", "rsl_rl")),
    ("whole_body_tracking", ("whole_body_tracking", "whole-body-tracking")),
]
optional_distribution_names = ("tensordict",)
outer_execution = ACTION_BALL_NOSITE_ATTESTATION

def normalized_distribution_name(value):
    result = value.lower()
    for needle in ("-", "_", "."):
        result = result.replace(needle, "-")
    while "--" in result:
        result = result.replace("--", "-")
    return result

sys_path = []
for value in sys.path:
    candidate = value if value else os.getcwd()
    sys_path.append(os.path.normpath(os.path.abspath(candidate)))

site_paths = set()
for row in outer_execution["import_roots"]:
    candidate = os.path.normpath(os.path.abspath(row["path"]))
    if not os.path.isdir(candidate):
        raise RuntimeError("explicit import root disappeared: " + candidate)
    site_paths.add(candidate)

installed_distribution_rows = list(importlib.metadata.distributions())
if hasattr(importlib.metadata, "packages_distributions"):
    package_map = importlib.metadata.packages_distributions()
else:
    package_map = {}
    for installed in installed_distribution_rows:
        installed_name = str(installed.metadata.get("Name") or "")
        top_level_text = installed.read_text("top_level.txt")
        if not installed_name or top_level_text is None:
            continue
        for line in top_level_text.splitlines():
            top_level = line.strip()
            if top_level:
                package_map.setdefault(top_level, []).append(installed_name)

def distribution_identity(distribution):
    metadata_candidate = getattr(distribution, "_path", None)
    if metadata_candidate is None:
        raise RuntimeError("distribution metadata path is unavailable")
    metadata_path = os.path.normpath(
        os.path.abspath(os.fspath(metadata_candidate))
    )
    actual_name = str(distribution.metadata.get("Name") or "")
    version = str(distribution.version)
    if not actual_name or "\n" in actual_name or "\r" in actual_name:
        raise RuntimeError("distribution name is invalid: " + metadata_path)
    if not version or "\n" in version or "\r" in version:
        raise RuntimeError("distribution version is invalid: " + actual_name)
    return actual_name, version, metadata_path

installed_by_name = {}
distribution_by_path = {}
for installed in installed_distribution_rows:
    actual_name, version, metadata_path = distribution_identity(installed)
    distribution_by_path[metadata_path] = installed
    normalized = normalized_distribution_name(actual_name)
    installed_by_name.setdefault(normalized, {})[metadata_path] = installed

def resolve_distribution(name):
    normalized = normalized_distribution_name(name)
    candidates = installed_by_name.get(normalized, {})
    if not candidates:
        raise importlib.metadata.PackageNotFoundError(name)
    if len(candidates) != 1:
        raise RuntimeError(
            "distribution name resolves ambiguously: " + name
        )
    return next(iter(candidates.values()))

def describe_distribution(distribution, module_hints=()):
    actual_name, version, metadata_path = distribution_identity(distribution)
    direct_url_path = None
    project_path = None
    editable = False
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is not None:
        direct_url_path = os.path.join(metadata_path, "direct_url.json")
        try:
            direct_url = json.loads(direct_url_text)
        except (TypeError, ValueError):
            raise RuntimeError(
                "distribution direct_url.json is invalid: " + actual_name
            )
        dir_info = direct_url.get("dir_info")
        editable = (
            isinstance(dir_info, dict)
            and dir_info.get("editable") is True
        )
        if editable:
            parsed = urllib.parse.urlsplit(direct_url.get("url", ""))
            if (
                parsed.scheme != "file"
                or parsed.netloc not in ("", "localhost")
                or parsed.query
                or parsed.fragment
            ):
                raise RuntimeError(
                    "editable distribution URL is not local file: "
                    + actual_name
                )
            decoded = urllib.parse.unquote(parsed.path)
            project_path = os.path.normpath(
                os.path.abspath(urllib.request.url2pathname(decoded))
            )

    top_level_names = []
    top_level_text = distribution.read_text("top_level.txt")
    if top_level_text is not None:
        for line in top_level_text.splitlines():
            value = line.strip()
            if value and value not in top_level_names:
                top_level_names.append(value)
    normalized_actual = normalized_distribution_name(actual_name)
    for top_level, distribution_names in package_map.items():
        if normalized_actual in {
            normalized_distribution_name(value)
            for value in distribution_names
        } and top_level not in top_level_names:
            top_level_names.append(top_level)
    for module_hint in module_hints:
        top_level = module_hint.split(".", 1)[0]
        mapped_names = {
            normalized_distribution_name(value)
            for value in package_map.get(top_level, ())
        }
        if (
            normalized_actual in mapped_names
            and top_level not in top_level_names
        ):
            top_level_names.append(top_level)
    return {
        "name": actual_name,
        "version": version,
        "metadata_path": metadata_path,
        "editable": editable,
        "project_path": project_path,
        "direct_url_path": direct_url_path,
        "top_level_names": sorted(top_level_names),
    }

modules = []
root_paths = set()
root_module_hints = {}
for module_name, preferred in contract:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin in (None, "built-in", "frozen"):
        raise RuntimeError("module has no regular origin candidate: " + module_name)
    origin = os.path.normpath(os.path.abspath(spec.origin))
    names = []
    for candidate in tuple(preferred) + tuple(sorted(package_map.get(module_name, ()))):
        if candidate not in names:
            names.append(candidate)
    distributions = []
    seen_metadata_paths = set()
    for candidate in names:
        try:
            distribution = resolve_distribution(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
        descriptor = describe_distribution(distribution, (module_name,))
        metadata_path = descriptor["metadata_path"]
        if metadata_path in seen_metadata_paths:
            continue
        seen_metadata_paths.add(metadata_path)
        root_paths.add(metadata_path)
        root_module_hints.setdefault(metadata_path, set()).add(module_name)
        distributions.append(descriptor)
    if not distributions:
        raise RuntimeError("module has no installed distribution version: " + module_name)
    preferred_normalized = {normalized_distribution_name(value) for value in preferred}
    preferred_rows = [
        row
        for row in distributions
        if normalized_distribution_name(row["name"]) in preferred_normalized
    ]
    chosen = preferred_rows[0] if preferred_rows else distributions[0]
    if not chosen["version"] or "\n" in chosen["version"] or "\r" in chosen["version"]:
        raise RuntimeError("module version is invalid: " + module_name)
    modules.append({
        "name": module_name,
        "version": chosen["version"],
        "version_source": "distribution:" + chosen["name"],
        "distributions": sorted(distributions, key=lambda row: (row["metadata_path"], row["name"])),
        "origin_path": origin,
    })

optional_distributions = []
for optional_name in optional_distribution_names:
    candidates = installed_by_name.get(
        normalized_distribution_name(optional_name), {}
    )
    if len(candidates) > 1:
        raise RuntimeError(
            "optional distribution resolves ambiguously: " + optional_name
        )
    if not candidates:
        optional_distributions.append(
            {
                "name": optional_name,
                "present": False,
                "version": None,
                "metadata_path": None,
            }
        )
        continue
    optional = next(iter(candidates.values()))
    optional_name_actual, optional_version, optional_path = (
        distribution_identity(optional)
    )
    root_paths.add(optional_path)
    optional_distributions.append(
        {
            "name": optional_name_actual,
            "present": True,
            "version": optional_version,
            "metadata_path": optional_path,
        }
    )

marker_environment = {
    key: str(value)
    for key, value in sorted(default_environment().items())
}
requested_extras = {path: set() for path in root_paths}
processed_extras = {}
queue = sorted(root_paths)
dependency_edges = {}
while queue:
    parent_path = queue.pop(0)
    extras = set(requested_extras[parent_path])
    if processed_extras.get(parent_path) == extras:
        continue
    processed_extras[parent_path] = extras
    parent = distribution_by_path.get(parent_path)
    if parent is None:
        raise RuntimeError(
            "resolved distribution disappeared: " + parent_path
        )
    parent_name, _parent_version, _parent_path = distribution_identity(parent)
    contexts = [""] + sorted(extras)
    for requirement_text in parent.metadata.get_all("Requires-Dist") or []:
        if (
            not requirement_text
            or "\n" in requirement_text
            or "\r" in requirement_text
        ):
            raise RuntimeError(
                "distribution has an invalid Requires-Dist: " + parent_name
            )
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement as exc:
            raise RuntimeError(
                "distribution has an unparseable Requires-Dist: "
                + parent_name
            ) from exc
        active = requirement.marker is None
        if requirement.marker is not None:
            active = any(
                requirement.marker.evaluate(
                    dict(marker_environment, extra=extra)
                )
                for extra in contexts
            )
        if not active:
            continue
        dependency = resolve_distribution(requirement.name)
        dependency_name, dependency_version, dependency_path = (
            distribution_identity(dependency)
        )
        if requirement.specifier and not requirement.specifier.contains(
            dependency_version, prereleases=True
        ):
            raise RuntimeError(
                "installed dependency version violates Requires-Dist: "
                + requirement_text
            )
        if requirement.url is not None:
            dependency_direct_url_text = dependency.read_text(
                "direct_url.json"
            )
            if dependency_direct_url_text is None:
                raise RuntimeError(
                    "URL dependency has no direct_url.json: "
                    + requirement_text
                )
            try:
                dependency_direct_url = json.loads(
                    dependency_direct_url_text
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "URL dependency direct_url.json is invalid: "
                    + requirement_text
                ) from exc
            if dependency_direct_url.get("url") != requirement.url:
                raise RuntimeError(
                    "URL dependency differs from Requires-Dist: "
                    + requirement_text
                )
        distribution_by_path[dependency_path] = dependency
        requested = requested_extras.setdefault(dependency_path, set())
        before = set(requested)
        requested.update(requirement.extras)
        if (
            dependency_path not in processed_extras
            or requested != before
        ) and dependency_path not in queue:
            queue.append(dependency_path)
            queue.sort()
        edge_key = (
            parent_path,
            requirement_text,
            dependency_path,
            dependency_name,
            dependency_version,
        )
        dependency_edges[edge_key] = {
            "from_metadata_path": parent_path,
            "requirement": requirement_text,
            "to_metadata_path": dependency_path,
            "to_name": dependency_name,
            "to_version": dependency_version,
        }

resolved_distributions = []
for metadata_path in sorted(requested_extras):
    distribution = distribution_by_path.get(metadata_path)
    if distribution is None:
        raise RuntimeError(
            "dependency distribution disappeared: " + metadata_path
        )
    resolved_distributions.append(
        describe_distribution(
            distribution,
            tuple(sorted(root_module_hints.get(metadata_path, set()))),
        )
    )

payload = {
    "implementation": sys.implementation.name,
    "version": ".".join(str(item) for item in sys.version_info[:3]),
    "cache_tag": sys.implementation.cache_tag,
    "executable": os.path.normpath(os.path.abspath(sys.executable)),
    "prefix": os.path.normpath(os.path.abspath(sys.prefix)),
    "base_prefix": os.path.normpath(os.path.abspath(sys.base_prefix)),
    "sys_path": sys_path,
    "site_package_paths": sorted(site_paths),
    "modules": modules,
    "marker_environment": marker_environment,
    "resolved_distributions": resolved_distributions,
    "dependency_edges": [
        dependency_edges[key] for key in sorted(dependency_edges)
    ],
    "optional_distributions": optional_distributions,
    "no_site_execution": {
        "outer": outer_execution,
        "inner": {
            "flags": {
                "isolated": bool(sys.flags.isolated),
                "no_site": bool(sys.flags.no_site),
                "no_user_site": bool(sys.flags.no_user_site),
                "ignore_environment": bool(sys.flags.ignore_environment),
                "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
                "optimize": int(sys.flags.optimize),
            },
            "site_module_loaded": "site" in sys.modules,
            "pth_files_executed": False,
            "sys_path": sys_path,
        },
    },
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
print("ACTION_BALL_RUNTIME_INVENTORY_PROBE_V2:" + base64.b64encode(raw).decode("ascii"))
"""


def _emit_nosite_probe() -> int:
    attestation = globals().get("ACTION_BALL_NOSITE_ATTESTATION")
    if type(attestation) is not dict:
        raise RuntimeInventoryError(
            "runtime probe requires the exact no-site bootstrap attestation"
        )
    scope = {
        "ACTION_BALL_NOSITE_ATTESTATION": attestation,
        "__name__": "_action_ball_runtime_inventory_probe",
    }
    try:
        exec(compile(_PROBE_PROGRAM, "<action-ball-runtime-probe>", "exec"), scope, scope)
    except Exception as exc:
        raise RuntimeInventoryError(
            "no-site runtime probe failed: %s" % exc
        ) from exc
    return 0


def _run_probe(
    requested: Path,
    import_roots: Sequence[Mapping[str, Any]],
    *,
    bootstrap_path: Path = NOSITE_BOOTSTRAP,
    entrypoint_path: Path = INVENTORY_ENTRYPOINT,
) -> Mapping[str, Any]:
    nosite = _load_nosite_bootstrap()
    bootstrap_binding = nosite.bind_regular_file(
        bootstrap_path, label="inventory no-site bootstrap"
    )
    entrypoint_binding = nosite.bind_regular_file(
        entrypoint_path, label="inventory probe entrypoint"
    )
    try:
        command = nosite.build_exact_nosite_argv(
            python=requested,
            bootstrap=bootstrap_path,
            bootstrap_sha256=bootstrap_binding["sha256"],
            entrypoint=entrypoint_path,
            entrypoint_sha256=entrypoint_binding["sha256"],
            import_roots=import_roots,
            entrypoint_argv=["_probe"],
            verify_import_roots=False,
        )
        nosite.validate_exact_nosite_argv(
            command.argv,
            expected_python=requested,
            expected_bootstrap=bootstrap_binding,
            expected_entrypoint=entrypoint_binding,
            expected_import_roots=import_roots,
            expected_entrypoint_argv=["_probe"],
            expected_contract_sha256=command.contract_sha256,
            verify_live=False,
        )
    except Exception as exc:
        raise RuntimeInventoryError(
            "cannot build the exact no-site runtime probe command"
        ) from exc
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONUSERBASE",
        "PYTHONPYCACHEPREFIX",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            list(command.argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeInventoryError("isolated Python runtime probe failed to execute") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeInventoryError(
            "isolated Python runtime probe exited %d: %s"
            % (completed.returncode, stderr)
        )
    output_lines = completed.stdout.decode(
        "utf-8", errors="replace"
    ).splitlines()
    if len(output_lines) != 1 or not output_lines[0].startswith(
        PROBE_SENTINEL
    ):
        raise RuntimeInventoryError(
            "isolated Python probe stdout is not the one exact payload line"
        )
    candidates = [output_lines[0][len(PROBE_SENTINEL) :]]
    try:
        raw = base64.b64decode(candidates[0].encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeInventoryError("isolated Python probe payload is not canonical base64") from exc
    payload = _strict_json_loads(raw, "isolated Python probe payload")
    return _require_mapping(payload, "isolated Python probe payload")


def _validate_nosite_probe_execution(
    value: Any, requested: Path
) -> List[str]:
    execution = _require_mapping(value, "probe.no_site_execution")
    _require_exact_keys(
        execution, {"outer", "inner"}, "probe.no_site_execution"
    )
    outer = _require_mapping(
        execution["outer"], "probe.no_site_execution.outer"
    )
    _require_exact_keys(
        outer,
        {
            "schema_version",
            "kind",
            "argv_contract_sha256",
            "bootstrap",
            "entrypoint",
            "import_roots",
            "flags",
            "site_module_loaded_before_entrypoint",
            "pth_files_executed",
            "sys_path_before_import_roots",
            "sys_path_after_import_roots",
        },
        "probe.no_site_execution.outer",
    )
    if (
        outer["schema_version"] != 1
        or outer["kind"] != "action_ball_python_nosite_execution_v1"
    ):
        raise RuntimeInventoryError(
            "probe outer no-site execution schema/kind is unsupported"
        )
    contract_sha = _require_sha256(
        outer["argv_contract_sha256"],
        "probe outer argv_contract_sha256",
    )

    bindings = {}
    for name, expected_path in (
        ("bootstrap", NOSITE_BOOTSTRAP),
        ("entrypoint", INVENTORY_ENTRYPOINT),
    ):
        row = _require_mapping(
            outer[name], "probe outer %s" % name
        )
        _require_exact_keys(
            row,
            {"path", "byte_count", "sha256"},
            "probe outer %s" % name,
        )
        path = _canonical_absolute_path(
            row["path"], "probe outer %s.path" % name
        )
        if path != expected_path:
            raise RuntimeInventoryError(
                "probe outer %s path differs from the fixed source" % name
            )
        count = _require_plain_int(
            row["byte_count"], "probe outer %s.byte_count" % name
        )
        digest = _require_sha256(
            row["sha256"], "probe outer %s.sha256" % name
        )
        bindings[name] = {
            "path": str(path),
            "byte_count": count,
            "sha256": digest,
        }

    raw_roots = outer["import_roots"]
    if not isinstance(raw_roots, list) or not raw_roots:
        raise RuntimeInventoryError(
            "probe outer import_roots must be non-empty"
        )
    roots = []
    root_paths = []
    for index, raw_root in enumerate(raw_roots):
        root = _require_mapping(
            raw_root, "probe outer import_roots[%d]" % index
        )
        _require_exact_keys(
            root,
            {
                "path",
                "tree_sha256",
                "file_count",
                "total_size_bytes",
            },
            "probe outer import root",
        )
        path = str(
            _canonical_absolute_path(
                root["path"], "probe outer import root path"
            )
        )
        if path in root_paths:
            raise RuntimeInventoryError(
                "probe outer import roots must be unique"
            )
        root_paths.append(path)
        roots.append(
            {
                "path": path,
                "tree_sha256": _require_sha256(
                    root["tree_sha256"],
                    "probe outer import root tree_sha256",
                ),
                "file_count": _require_plain_int(
                    root["file_count"],
                    "probe outer import root file_count",
                ),
                "total_size_bytes": _require_plain_int(
                    root["total_size_bytes"],
                    "probe outer import root total_size_bytes",
                ),
            }
        )

    expected_flags = {
        "isolated": True,
        "no_site": True,
        "no_user_site": True,
        "ignore_environment": True,
        "dont_write_bytecode": True,
        "optimize": 0,
    }
    flags = _require_mapping(
        outer["flags"], "probe outer no-site flags"
    )
    _require_exact_keys(
        flags, expected_flags.keys(), "probe outer no-site flags"
    )
    if dict(flags) != expected_flags:
        raise RuntimeInventoryError(
            "probe outer interpreter flags are not exact -I -B -S"
        )
    if (
        outer["site_module_loaded_before_entrypoint"] is not False
        or outer["pth_files_executed"] is not False
    ):
        raise RuntimeInventoryError(
            "probe outer execution imported site or executed .pth"
        )

    def normalized_sys_path(raw: Any, label: str) -> List[str]:
        if not isinstance(raw, list):
            raise RuntimeInventoryError("%s must be a list" % label)
        return [
            str(_canonical_absolute_path(item, "%s[]" % label))
            for item in raw
        ]

    before = normalized_sys_path(
        outer["sys_path_before_import_roots"],
        "probe outer sys_path_before_import_roots",
    )
    after = normalized_sys_path(
        outer["sys_path_after_import_roots"],
        "probe outer sys_path_after_import_roots",
    )
    if len(after) < len(root_paths) or after[-len(root_paths) :] != root_paths:
        raise RuntimeInventoryError(
            "probe outer import roots are not the exact sys.path suffix"
        )
    if after[: len(after) - len(root_paths)] != before:
        raise RuntimeInventoryError(
            "probe outer sys.path changed outside explicit import roots"
        )
    if any(path in before for path in root_paths):
        raise RuntimeInventoryError(
            "probe import root was exposed before no-site bootstrap"
        )

    inner = _require_mapping(
        execution["inner"], "probe.no_site_execution.inner"
    )
    _require_exact_keys(
        inner,
        {"flags", "site_module_loaded", "pth_files_executed", "sys_path"},
        "probe.no_site_execution.inner",
    )
    inner_flags = _require_mapping(
        inner["flags"], "probe inner no-site flags"
    )
    _require_exact_keys(
        inner_flags, expected_flags.keys(), "probe inner no-site flags"
    )
    if dict(inner_flags) != expected_flags:
        raise RuntimeInventoryError(
            "probe inner interpreter flags are not exact -I -B -S"
        )
    if (
        inner["site_module_loaded"] is not False
        or inner["pth_files_executed"] is not False
    ):
        raise RuntimeInventoryError(
            "probe inner execution imported site or executed .pth"
        )
    if normalized_sys_path(
        inner["sys_path"], "probe inner sys_path"
    ) != after:
        raise RuntimeInventoryError(
            "probe inner sys.path differs from outer root installation"
        )

    argv_contract = {
        "schema_version": 1,
        "kind": "action_ball_python_nosite_argv_contract_v1",
        "bootstrap": bindings["bootstrap"],
        "entrypoint": bindings["entrypoint"],
        "import_roots": roots,
        "entrypoint_argv": ["_probe"],
    }
    if _sha256_bytes(_canonical_json_bytes(argv_contract)) != contract_sha:
        raise RuntimeInventoryError(
            "probe outer argv contract SHA differs from canonical rebuild"
        )
    return root_paths


def _validate_probe_payload(payload: Mapping[str, Any], requested: Path) -> None:
    _require_exact_keys(
        payload,
        {
            "implementation",
            "version",
            "cache_tag",
            "executable",
            "prefix",
            "base_prefix",
            "sys_path",
            "site_package_paths",
            "modules",
            "marker_environment",
            "resolved_distributions",
            "dependency_edges",
            "optional_distributions",
            "no_site_execution",
        },
        "probe",
    )
    _require_nonempty_line(payload["implementation"], "probe.implementation", 128)
    _require_nonempty_line(payload["version"], "probe.version", 128)
    _require_nonempty_line(payload["cache_tag"], "probe.cache_tag", 128)
    executable = _canonical_absolute_path(payload["executable"], "probe.executable")
    try:
        requested_chain, _resolved_requested = _python_chain_snapshot(
            requested
        )
    except (OSError, RuntimeInventoryError) as exc:
        raise RuntimeInventoryError(
            "requested Python chain cannot be resolved while validating the probe"
        ) from exc
    allowed_executables = {
        _canonical_absolute_path(row["path"], "requested Python chain path")
        for row in requested_chain
    }
    try:
        executable_resolved = executable.resolve(strict=True)
    except OSError as exc:
        raise RuntimeInventoryError(
            "probe sys.executable cannot be resolved"
        ) from exc
    if (
        executable not in allowed_executables
        and executable_resolved
        != _resolved_requested.resolve(strict=True)
    ):
        raise RuntimeInventoryError(
            "probe sys.executable is neither an exact member of the "
            "requested Python chain nor an alias of its bound executable"
        )
    _canonical_absolute_path(payload["prefix"], "probe.prefix")
    _canonical_absolute_path(payload["base_prefix"], "probe.base_prefix")
    explicit_root_paths = _validate_nosite_probe_execution(
        payload["no_site_execution"], requested
    )
    for label in ("sys_path", "site_package_paths"):
        rows = payload[label]
        if not isinstance(rows, list):
            raise RuntimeInventoryError("probe.%s must be a list" % label)
        parsed = [
            str(_canonical_absolute_path(row, "probe.%s[]" % label)) for row in rows
        ]
        if label == "site_package_paths":
            if not parsed:
                raise RuntimeInventoryError("probe must expose at least one site-packages path")
            if parsed != sorted(set(parsed)):
                raise RuntimeInventoryError(
                    "probe.site_package_paths must be sorted and unique"
                )
            if parsed != sorted(explicit_root_paths):
                raise RuntimeInventoryError(
                    "probe.site_package_paths differ from explicit no-site roots"
                )
    if payload["sys_path"] != payload["no_site_execution"]["inner"]["sys_path"]:
        raise RuntimeInventoryError(
            "probe.sys_path differs from inner no-site sys.path"
        )
    modules = payload["modules"]
    if not isinstance(modules, list) or len(modules) != len(MODULE_NAMES):
        raise RuntimeInventoryError("probe.modules must contain the fixed module set")
    if [row.get("name") if isinstance(row, dict) else None for row in modules] != list(
        MODULE_NAMES
    ):
        raise RuntimeInventoryError("probe.modules order differs from the fixed contract")
    for index, raw_row in enumerate(modules):
        row = _require_mapping(raw_row, "probe.modules[%d]" % index)
        _require_exact_keys(
            row,
            {
                "name",
                "version",
                "version_source",
                "distributions",
                "origin_path",
            },
            "probe.modules[%d]" % index,
        )
        _require_nonempty_line(row["name"], "probe module name", 128)
        _require_nonempty_line(row["version"], "probe module version", 256)
        _require_nonempty_line(row["version_source"], "probe module version source", 256)
        _canonical_absolute_path(row["origin_path"], "probe module origin")
        distributions = row["distributions"]
        if not isinstance(distributions, list) or not distributions:
            raise RuntimeInventoryError("probe module distributions must be non-empty")
        seen_paths = set()
        parsed_order = []
        for dist_index, raw_dist in enumerate(distributions):
            dist = _require_mapping(
                raw_dist, "probe module distribution[%d]" % dist_index
            )
            _require_exact_keys(
                dist,
                {
                    "name",
                    "version",
                    "metadata_path",
                    "editable",
                    "project_path",
                    "direct_url_path",
                    "top_level_names",
                },
                "probe module distribution",
            )
            name = _require_nonempty_line(dist["name"], "distribution name", 256)
            _require_nonempty_line(dist["version"], "distribution version", 256)
            metadata_path = _canonical_absolute_path(
                dist["metadata_path"], "distribution metadata_path"
            )
            if str(metadata_path) in seen_paths:
                raise RuntimeInventoryError(
                    "probe module distribution metadata path is duplicated"
                )
            seen_paths.add(str(metadata_path))
            parsed_order.append((str(metadata_path), name))
            if type(dist["editable"]) is not bool:
                raise RuntimeInventoryError("distribution editable must be boolean")
            project_path = dist["project_path"]
            direct_url_path = dist["direct_url_path"]
            if dist["editable"]:
                _canonical_absolute_path(
                    project_path, "editable distribution project_path"
                )
                direct = _canonical_absolute_path(
                    direct_url_path, "editable distribution direct_url_path"
                )
                if direct.parent != metadata_path:
                    raise RuntimeInventoryError(
                        "editable direct_url.json must be in metadata_path"
                    )
            elif project_path is not None:
                raise RuntimeInventoryError(
                    "non-editable distribution cannot carry project_path"
                )
            elif direct_url_path is not None:
                direct = _canonical_absolute_path(
                    direct_url_path, "distribution direct_url_path"
                )
                if direct.parent != metadata_path:
                    raise RuntimeInventoryError(
                        "distribution direct_url.json must be in metadata_path"
                    )
            top_levels = dist["top_level_names"]
            if not isinstance(top_levels, list):
                raise RuntimeInventoryError(
                    "distribution top_level_names must be a list"
                )
            if top_levels != sorted(set(top_levels)):
                raise RuntimeInventoryError(
                    "distribution top_level_names must be sorted unique"
                )
            for top_level in top_levels:
                value = _require_nonempty_line(
                    top_level, "distribution top-level name", 256
                )
                if TOP_LEVEL_NAME_RE.fullmatch(value) is None:
                    raise RuntimeInventoryError(
                        "distribution top-level name is unsafe"
                    )
        if parsed_order != sorted(parsed_order):
            raise RuntimeInventoryError(
                "probe module distributions must be sorted by metadata path/name"
            )
        version_source = "distribution:" + row["distributions"][0]["name"]
        if not any(
            row["version_source"] == "distribution:" + dist["name"]
            and row["version"] == dist["version"]
            for dist in row["distributions"]
        ):
            raise RuntimeInventoryError(
                "module version_source/version does not name one resolved distribution"
            )

    marker_environment = _require_mapping(
        payload["marker_environment"], "probe.marker_environment"
    )
    if not marker_environment:
        raise RuntimeInventoryError(
            "probe.marker_environment must be non-empty"
        )
    for key, value in marker_environment.items():
        _require_nonempty_line(key, "marker environment key", 256)
        _require_nonempty_line(value, "marker environment value", 4096)
    for required_key in (
        "python_version",
        "python_full_version",
        "sys_platform",
        "platform_machine",
    ):
        if required_key not in marker_environment:
            raise RuntimeInventoryError(
                "probe.marker_environment is missing %s" % required_key
            )

    resolved = payload["resolved_distributions"]
    if not isinstance(resolved, list) or not resolved:
        raise RuntimeInventoryError(
            "probe.resolved_distributions must be a non-empty list"
        )
    resolved_by_path = {}
    resolved_order = []
    for index, raw_dist in enumerate(resolved):
        dist = _require_mapping(
            raw_dist, "probe.resolved_distributions[%d]" % index
        )
        _require_exact_keys(
            dist,
            {
                "name",
                "version",
                "metadata_path",
                "editable",
                "project_path",
                "direct_url_path",
                "top_level_names",
            },
            "probe resolved distribution",
        )
        name = _require_nonempty_line(
            dist["name"], "resolved distribution name", 256
        )
        _require_nonempty_line(
            dist["version"], "resolved distribution version", 256
        )
        metadata_path = _canonical_absolute_path(
            dist["metadata_path"], "resolved distribution metadata_path"
        )
        if str(metadata_path) in resolved_by_path:
            raise RuntimeInventoryError(
                "resolved distribution metadata_path is duplicated"
            )
        if type(dist["editable"]) is not bool:
            raise RuntimeInventoryError(
                "resolved distribution editable must be boolean"
            )
        if dist["editable"]:
            _canonical_absolute_path(
                dist["project_path"],
                "resolved editable distribution project_path",
            )
            direct = _canonical_absolute_path(
                dist["direct_url_path"],
                "resolved editable distribution direct_url_path",
            )
            if direct.parent != metadata_path:
                raise RuntimeInventoryError(
                    "resolved editable direct_url is outside metadata_path"
                )
        elif dist["project_path"] is not None:
            raise RuntimeInventoryError(
                "resolved non-editable distribution carries project_path"
            )
        elif dist["direct_url_path"] is not None:
            direct = _canonical_absolute_path(
                dist["direct_url_path"],
                "resolved distribution direct_url_path",
            )
            if direct.parent != metadata_path:
                raise RuntimeInventoryError(
                    "resolved direct_url is outside metadata_path"
                )
        top_levels = dist["top_level_names"]
        if (
            not isinstance(top_levels, list)
            or top_levels != sorted(set(top_levels))
        ):
            raise RuntimeInventoryError(
                "resolved distribution top_level_names must be sorted unique"
            )
        for top_level in top_levels:
            value = _require_nonempty_line(
                top_level, "resolved distribution top-level name", 256
            )
            if TOP_LEVEL_NAME_RE.fullmatch(value) is None:
                raise RuntimeInventoryError(
                    "resolved distribution top-level name is unsafe"
                )
        resolved_by_path[str(metadata_path)] = dist
        resolved_order.append((str(metadata_path), name))
    if resolved_order != sorted(resolved_order):
        raise RuntimeInventoryError(
            "resolved distributions must be sorted by metadata_path/name"
        )

    optional_distributions = payload["optional_distributions"]
    if (
        not isinstance(optional_distributions, list)
        or len(optional_distributions) != len(OPTIONAL_DISTRIBUTION_NAMES)
    ):
        raise RuntimeInventoryError(
            "probe.optional_distributions differs from fixed contract"
        )
    optional_roots = set()
    for expected_name, raw_optional in zip(
        OPTIONAL_DISTRIBUTION_NAMES, optional_distributions
    ):
        optional = _require_mapping(
            raw_optional, "probe optional distribution"
        )
        _require_exact_keys(
            optional,
            {"name", "present", "version", "metadata_path"},
            "probe optional distribution",
        )
        name = _require_nonempty_line(
            optional["name"], "optional distribution name", 256
        )
        if _normalized_distribution_name(name) != (
            _normalized_distribution_name(expected_name)
        ):
            raise RuntimeInventoryError(
                "probe optional distribution order/name differs"
            )
        if type(optional["present"]) is not bool:
            raise RuntimeInventoryError(
                "optional distribution present must be boolean"
            )
        if optional["present"]:
            _require_nonempty_line(
                optional["version"], "optional distribution version", 256
            )
            metadata_path = str(
                _canonical_absolute_path(
                    optional["metadata_path"],
                    "optional distribution metadata_path",
                )
            )
            if metadata_path not in resolved_by_path:
                raise RuntimeInventoryError(
                    "present optional distribution is absent from closure"
                )
            optional_roots.add(metadata_path)
        elif (
            optional["version"] is not None
            or optional["metadata_path"] is not None
        ):
            raise RuntimeInventoryError(
                "absent optional distribution carries an identity"
            )

    root_paths = set(optional_roots)
    for module in modules:
        for dist in module["distributions"]:
            metadata_path = dist["metadata_path"]
            resolved_dist = resolved_by_path.get(metadata_path)
            if resolved_dist is None:
                raise RuntimeInventoryError(
                    "module distribution is absent from resolved closure"
                )
            if _canonical_json_bytes(dist) != _canonical_json_bytes(
                resolved_dist
            ):
                raise RuntimeInventoryError(
                    "module distribution differs from resolved closure"
                )
            root_paths.add(metadata_path)

    edges = payload["dependency_edges"]
    if not isinstance(edges, list):
        raise RuntimeInventoryError(
            "probe.dependency_edges must be a list"
        )
    edge_order = []
    adjacency = {}
    for index, raw_edge in enumerate(edges):
        edge = _require_mapping(
            raw_edge, "probe.dependency_edges[%d]" % index
        )
        _require_exact_keys(
            edge,
            {
                "from_metadata_path",
                "requirement",
                "to_metadata_path",
                "to_name",
                "to_version",
            },
            "probe dependency edge",
        )
        source = str(
            _canonical_absolute_path(
                edge["from_metadata_path"], "dependency edge source"
            )
        )
        target = str(
            _canonical_absolute_path(
                edge["to_metadata_path"], "dependency edge target"
            )
        )
        requirement = _require_nonempty_line(
            edge["requirement"], "dependency requirement", 16384
        )
        to_name = _require_nonempty_line(
            edge["to_name"], "dependency target name", 256
        )
        to_version = _require_nonempty_line(
            edge["to_version"], "dependency target version", 256
        )
        if source not in resolved_by_path or target not in resolved_by_path:
            raise RuntimeInventoryError(
                "dependency edge leaves the resolved distribution closure"
            )
        target_row = resolved_by_path[target]
        if (
            target_row["name"] != to_name
            or target_row["version"] != to_version
        ):
            raise RuntimeInventoryError(
                "dependency edge target identity is inconsistent"
            )
        identity = (source, requirement, target, to_name, to_version)
        edge_order.append(identity)
        adjacency.setdefault(source, set()).add(target)
    if edge_order != sorted(set(edge_order)):
        raise RuntimeInventoryError(
            "dependency edges must be sorted unique"
        )
    reachable = set(root_paths)
    pending = list(sorted(root_paths))
    while pending:
        source = pending.pop(0)
        for target in sorted(adjacency.get(source, ())):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    if reachable != set(resolved_by_path):
        raise RuntimeInventoryError(
            "resolved distribution closure contains an unreachable node"
        )


def _site_packages_snapshot(path: Path) -> Dict[str, Any]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeInventoryError("site-packages directory is missing: %s" % path) from exc
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeInventoryError("site-packages path is not a real directory: %s" % path)
    try:
        first_names = sorted(
            entry.name
            for entry in os.scandir(str(path))
            if entry.name.endswith(".pth")
        )
    except OSError as exc:
        raise RuntimeInventoryError("cannot enumerate .pth files in %s" % path) from exc
    files = [
        _open_regular_snapshot(
            path / name, "site-packages .pth", include_raw=True
        )
        for name in first_names
    ]
    try:
        second_names = sorted(
            entry.name
            for entry in os.scandir(str(path))
            if entry.name.endswith(".pth")
        )
        after = path.lstat()
    except OSError as exc:
        raise RuntimeInventoryError("site-packages changed during inventory: %s" % path) from exc
    if first_names != second_names or not _same_lstat(before, after):
        raise RuntimeInventoryError("site-packages changed during inventory: %s" % path)
    return {
        "path": str(path),
        "lstat": _lstat_dict(before),
        "ancestors": _ancestor_snapshots(path),
        "pth_files": files,
    }


def _raw_snapshot_bytes(value: Mapping[str, Any], label: str) -> bytes:
    encoded = value.get("raw_base64")
    if type(encoded) is not str:
        raise RuntimeInventoryError("%s is missing raw_base64" % label)
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeInventoryError("%s raw_base64 is invalid" % label) from exc
    if len(raw) != value.get("byte_count"):
        raise RuntimeInventoryError("%s raw bytes differ from byte_count" % label)
    if _sha256_bytes(raw) != value.get("sha256"):
        raise RuntimeInventoryError("%s raw bytes differ from sha256" % label)
    return raw


def _light_regular_snapshot(path: Path, label: str) -> Dict[str, Any]:
    full = _open_regular_snapshot(path, label)
    return {
        "path": full["path"],
        "byte_count": full["byte_count"],
        "sha256": full["sha256"],
    }


def _record_absolute_path(metadata_path: Path, record_path: str) -> Path:
    if (
        not record_path
        or "\x00" in record_path
        or "\n" in record_path
        or "\r" in record_path
        or "\\" in record_path
    ):
        raise RuntimeInventoryError("distribution RECORD contains an unsafe path")
    pure = PurePosixPath(record_path)
    if pure.is_absolute() or str(pure) != record_path:
        raise RuntimeInventoryError(
            "distribution RECORD path must be canonical relative POSIX"
        )
    candidate = os.path.normpath(
        os.path.join(str(metadata_path.parent), *pure.parts)
    )
    if not os.path.isabs(candidate):
        raise RuntimeInventoryError(
            "distribution RECORD path did not resolve to an absolute path"
        )
    return Path(candidate)


def _declared_record_sha256(value: str, label: str) -> Optional[str]:
    if value == "":
        return None
    if "=" not in value:
        raise RuntimeInventoryError("%s RECORD hash is malformed" % label)
    algorithm, encoded = value.split("=", 1)
    if algorithm != "sha256" or not encoded:
        raise RuntimeInventoryError(
            "%s RECORD hash must use sha256" % label
        )
    padding = "=" * ((4 - len(encoded) % 4) % 4)
    try:
        digest = base64.b64decode(
            (encoded + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeInventoryError(
            "%s RECORD sha256 is not canonical base64url" % label
        ) from exc
    if len(digest) != hashlib.sha256().digest_size:
        raise RuntimeInventoryError(
            "%s RECORD sha256 has the wrong length" % label
        )
    canonical = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    if canonical != encoded:
        raise RuntimeInventoryError(
            "%s RECORD sha256 is not canonical base64url" % label
        )
    return digest.hex()


def _noneditable_distribution_snapshot(
    descriptor: Mapping[str, Any],
) -> Dict[str, Any]:
    name = descriptor["name"]
    metadata_path = _canonical_absolute_path(
        descriptor["metadata_path"], "distribution metadata path"
    )
    try:
        metadata_info = metadata_path.lstat()
    except OSError as exc:
        raise RuntimeInventoryError(
            "distribution metadata directory is missing: %s" % metadata_path
        ) from exc
    if not stat.S_ISDIR(metadata_info.st_mode):
        raise RuntimeInventoryError(
            "distribution metadata path must be a real directory: %s"
            % metadata_path
        )
    if not metadata_path.name.endswith(".dist-info"):
        raise RuntimeInventoryError(
            "distribution metadata path must be a .dist-info directory: %s"
            % metadata_path
        )
    metadata_file_path = metadata_path / "METADATA"
    wheel_file_path = metadata_path / "WHEEL"
    record_path = metadata_path / "RECORD"
    metadata_file = _open_regular_snapshot(
        metadata_file_path,
        "distribution %s METADATA" % name,
        include_raw=True,
    )
    wheel_file = _open_regular_snapshot(
        wheel_file_path,
        "distribution %s WHEEL" % name,
        include_raw=True,
    )
    record = _open_regular_snapshot(
        record_path,
        "distribution %s RECORD" % name,
        include_raw=True,
    )
    record_raw = _raw_snapshot_bytes(record, "distribution RECORD")
    try:
        record_text = record_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeInventoryError(
            "distribution RECORD is not UTF-8: %s" % name
        ) from exc
    try:
        parsed_rows = list(
            csv.reader(io.StringIO(record_text, newline=""), strict=True)
        )
    except (csv.Error, ValueError) as exc:
        raise RuntimeInventoryError(
            "distribution RECORD is not strict CSV: %s" % name
        ) from exc
    if not parsed_rows:
        raise RuntimeInventoryError("distribution RECORD is empty: %s" % name)

    files: List[Dict[str, Any]] = []
    seen_record_paths = set()
    seen_absolute_paths = set()
    for index, parsed in enumerate(parsed_rows):
        label = "distribution %s RECORD row %d" % (name, index)
        if len(parsed) != 3:
            raise RuntimeInventoryError("%s must have exactly three columns" % label)
        record_name, declared_hash, declared_size = parsed
        if record_name in seen_record_paths:
            raise RuntimeInventoryError("%s path is duplicated" % label)
        seen_record_paths.add(record_name)
        absolute = _record_absolute_path(metadata_path, record_name)
        if str(absolute) in seen_absolute_paths:
            raise RuntimeInventoryError(
                "%s aliases another RECORD path" % label
            )
        seen_absolute_paths.add(str(absolute))
        actual = _light_regular_snapshot(
            absolute, "%s file" % label
        )
        expected_sha256 = _declared_record_sha256(declared_hash, label)
        if expected_sha256 is not None and expected_sha256 != actual["sha256"]:
            raise RuntimeInventoryError(
                "%s sha256 differs from RECORD" % label
            )
        if declared_size == "":
            parsed_size = None
        else:
            if not declared_size.isdigit() or (
                len(declared_size) > 1 and declared_size.startswith("0")
            ):
                raise RuntimeInventoryError(
                    "%s size is not a canonical non-negative integer" % label
                )
            parsed_size = int(declared_size)
            if parsed_size != actual["byte_count"]:
                raise RuntimeInventoryError(
                    "%s byte count differs from RECORD" % label
                )
        files.append(
            {
                "record_path": record_name,
                "record_hash": declared_hash,
                "record_size": parsed_size,
                "path": actual["path"],
                "byte_count": actual["byte_count"],
                "sha256": actual["sha256"],
            }
        )

    if str(record_path) not in seen_absolute_paths:
        raise RuntimeInventoryError(
            "distribution RECORD must list its own bytes: %s" % name
        )
    files.sort(key=lambda row: row["record_path"])
    files_by_path = {row["path"]: row for row in files}
    for required_path, required_label in (
        (metadata_file_path, "METADATA"),
        (wheel_file_path, "WHEEL"),
    ):
        required_row = files_by_path.get(str(required_path))
        if required_row is None:
            raise RuntimeInventoryError(
                "distribution RECORD does not list %s: %s"
                % (required_label, name)
            )
        if (
            not required_row["record_hash"]
            or required_row["record_size"] is None
        ):
            raise RuntimeInventoryError(
                "distribution RECORD must hash and size %s: %s"
                % (required_label, name)
            )

    owned_roots = {str(metadata_path)}
    base = metadata_path.parent
    for top_level in descriptor["top_level_names"]:
        candidate = base / top_level
        module_file = base / (top_level + ".py")
        if candidate.exists() or candidate.is_symlink():
            owned_roots.add(str(candidate))
        elif module_file.exists() or module_file.is_symlink():
            owned_roots.add(str(module_file))
        else:
            raise RuntimeInventoryError(
                "distribution top-level root is missing: %s" % top_level
            )
    for file_row in files:
        absolute = Path(file_row["path"])
        try:
            relative = absolute.relative_to(base)
        except ValueError:
            continue
        if not relative.parts:
            continue
        root = base / relative.parts[0]
        if root == base:
            continue
        owned_roots.add(str(root))

    return {
        "name": name,
        "version": descriptor["version"],
        "metadata_path": str(metadata_path),
        "editable": False,
        "top_level_names": list(descriptor["top_level_names"]),
        "metadata_lstat": _lstat_dict(metadata_info),
        "metadata_ancestors": _ancestor_snapshots(metadata_path),
        "metadata_file": metadata_file,
        "wheel_file": wheel_file,
        "record": record,
        "files": files,
        "owned_roots": sorted(owned_roots),
    }


def _iter_owned_regular_files(root: Path) -> Iterable[Path]:
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise RuntimeInventoryError(
            "distribution owned root is missing: %s" % root
        ) from exc
    if stat.S_ISLNK(root_info.st_mode):
        raise RuntimeInventoryError(
            "distribution owned root cannot be a symlink: %s" % root
        )
    if stat.S_ISREG(root_info.st_mode):
        yield root
        return
    if not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeInventoryError(
            "distribution owned root has unsupported type: %s" % root
        )
    for directory, names, files in os.walk(str(root), topdown=True, followlinks=False):
        names[:] = sorted(names)
        for name in names:
            child = Path(directory) / name
            try:
                child_info = child.lstat()
            except OSError as exc:
                raise RuntimeInventoryError(
                    "distribution owned directory changed during scan: %s"
                    % child
                ) from exc
            if stat.S_ISLNK(child_info.st_mode):
                raise RuntimeInventoryError(
                    "distribution owned tree contains a directory symlink: %s"
                    % child
                )
        for name in sorted(files):
            child = Path(directory) / name
            try:
                child_info = child.lstat()
            except OSError as exc:
                raise RuntimeInventoryError(
                    "distribution owned file changed during scan: %s" % child
                ) from exc
            if not stat.S_ISREG(child_info.st_mode):
                raise RuntimeInventoryError(
                    "distribution owned tree contains a non-regular file: %s"
                    % child
                )
            yield child


def _reject_unlisted_distribution_files(
    distributions: Sequence[Mapping[str, Any]],
) -> None:
    allowed = {
        row["path"]
        for distribution in distributions
        if "files" in distribution
        for row in distribution["files"]
    }
    roots = sorted(
        {
            root
            for distribution in distributions
            if "owned_roots" in distribution
            for root in distribution["owned_roots"]
        }
    )
    observed = set()
    for root_text in roots:
        for path in _iter_owned_regular_files(Path(root_text)):
            observed.add(str(path))
    extra = sorted(observed - allowed)
    if extra:
        raise RuntimeInventoryError(
            "non-editable distribution contains files absent from RECORD: %s"
            % extra[:8]
        )
    missing_from_scan = sorted(
        path
        for path in allowed
        if any(
            path == root or path.startswith(root + os.sep)
            for root in roots
        )
        and path not in observed
    )
    if missing_from_scan:
        raise RuntimeInventoryError(
            "RECORD file is missing from distribution owned roots: %s"
            % missing_from_scan[:8]
        )


def _git(
    checkout: Path, arguments: Sequence[str], label: str, *, binary: bool = False
) -> Any:
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        raise RuntimeInventoryError("git executable is unavailable")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GIT_", "LD_", "DYLD_"))
        and key != "XDG_CONFIG_HOME"
    }
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_COUNT"] = "0"
    try:
        completed = subprocess.run(
            [
                executable,
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "diff.external=",
                "-C",
                str(checkout),
            ]
            + list(arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeInventoryError("git %s failed to execute" % label) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeInventoryError("git %s failed: %s" % (label, stderr))
    if binary:
        return completed.stdout
    try:
        return completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeInventoryError("git %s output is not ASCII" % label) from exc


def _verify_git_worktree_blobs(
    checkout: Path, tracked_tree_raw: bytes, label: str
) -> None:
    for raw_entry in tracked_tree_raw.split(b"\x00"):
        if not raw_entry:
            continue
        try:
            metadata, path_raw = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
            path_text = path_raw.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeInventoryError(
                "%s tracked tree output is malformed or non-UTF-8" % label
            ) from exc
        if object_type != b"blob" or re.fullmatch(
            rb"[0-9a-f]{40}", object_id
        ) is None:
            raise RuntimeInventoryError(
                "%s tracked tree contains a non-blob or malformed object"
                % label
            )
        pure = PurePosixPath(path_text)
        if (
            pure.is_absolute()
            or str(pure) != path_text
            or ".." in pure.parts
            or not pure.parts
        ):
            raise RuntimeInventoryError(
                "%s tracked tree contains an unsafe path" % label
            )
        current = checkout
        for part in pure.parts[:-1]:
            current = current / part
            try:
                info = current.lstat()
            except OSError as exc:
                raise RuntimeInventoryError(
                    "%s tracked parent is missing: %s" % (label, current)
                ) from exc
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeInventoryError(
                    "%s tracked parent is not a real directory: %s"
                    % (label, current)
                )
        path = checkout.joinpath(*pure.parts)
        if mode in (b"100644", b"100755"):
            snapshot = _open_regular_snapshot(
                path, "%s tracked file" % label, include_raw=True
            )
            raw = _raw_snapshot_bytes(
                snapshot, "%s tracked file" % label
            )
            is_executable = bool(snapshot["lstat"]["st_mode"] & 0o111)
            if is_executable != (mode == b"100755"):
                raise RuntimeInventoryError(
                    "%s tracked executable mode differs from HEAD: %s"
                    % (label, path)
                )
        elif mode == b"120000":
            try:
                info = path.lstat()
                link_text = os.readlink(str(path))
            except OSError as exc:
                raise RuntimeInventoryError(
                    "%s tracked symlink cannot be read: %s" % (label, path)
                ) from exc
            if not stat.S_ISLNK(info.st_mode):
                raise RuntimeInventoryError(
                    "%s tracked symlink differs from HEAD: %s" % (label, path)
                )
            raw = os.fsencode(link_text)
        else:
            raise RuntimeInventoryError(
                "%s tracked tree contains unsupported mode %s"
                % (label, mode.decode("ascii", errors="replace"))
            )
        header = b"blob " + str(len(raw)).encode("ascii") + b"\x00"
        if hashlib.sha1(header + raw).hexdigest().encode("ascii") != object_id:
            raise RuntimeInventoryError(
                "%s worktree bytes differ from HEAD: %s" % (label, path)
            )


def _git_checkout_snapshot(checkout: Path, label: str) -> Dict[str, Any]:
    try:
        before = checkout.lstat()
    except OSError as exc:
        raise RuntimeInventoryError(
            "%s checkout is missing: %s" % (label, checkout)
        ) from exc
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeInventoryError("%s checkout must be a real directory" % label)
    toplevel = _git(checkout, ("rev-parse", "--show-toplevel"), "toplevel")
    if Path(toplevel) != checkout:
        raise RuntimeInventoryError(
            "%s path is not the exact Git toplevel: %s" % (label, checkout)
        )
    commit = _git(
        checkout, ("rev-parse", "--verify", "HEAD^{commit}"), "commit"
    )
    tree = _git(checkout, ("rev-parse", "--verify", "HEAD^{tree}"), "tree")
    head_ref = _git(
        checkout,
        ("rev-parse", "--symbolic-full-name", "HEAD"),
        "HEAD ref",
    )
    _require_nonempty_line(head_ref, "%s HEAD ref" % label, 16384)
    if GIT_OBJECT_RE.fullmatch(commit) is None:
        raise RuntimeInventoryError(
            "%s HEAD is not a SHA-1 commit object" % label
        )
    if GIT_OBJECT_RE.fullmatch(tree) is None:
        raise RuntimeInventoryError("%s HEAD tree is not a SHA-1 object" % label)
    tracked_tree_raw = _git(
        checkout,
        ("ls-tree", "-r", "-z", "--full-tree", "HEAD"),
        "tracked tree",
        binary=True,
    )
    _verify_git_worktree_blobs(checkout, tracked_tree_raw, label)
    index_flags_raw = _git(
        checkout,
        ("ls-files", "-v", "-z"),
        "index flags",
        binary=True,
    )
    for raw_entry in index_flags_raw.split(b"\x00"):
        if not raw_entry:
            continue
        if len(raw_entry) < 3 or raw_entry[:2] != b"H ":
            raise RuntimeInventoryError(
                "%s index contains assume-unchanged, skip-worktree, or non-normal flags"
                % label
            )
    status_raw = _git(
        checkout,
        (
            "status",
            "--porcelain=v2",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        "clean status",
        binary=True,
    )
    if status_raw != b"":
        raise RuntimeInventoryError("%s checkout is not exactly clean" % label)
    remote_v_raw = _git(
        checkout, ("remote", "-v"), "remotes", binary=True
    )
    if not remote_v_raw:
        raise RuntimeInventoryError("%s checkout has no bound Git remote" % label)
    remote_config_raw = _git(
        checkout,
        (
            "config",
            "--local",
            "--null",
            "--get-regexp",
            r"^remote\..*\.(url|pushurl|fetch)$",
        ),
        "remote config",
        binary=True,
    )
    if not remote_config_raw:
        raise RuntimeInventoryError(
            "%s checkout has no bound remote URL/refspec config" % label
        )
    try:
        after = checkout.lstat()
    except OSError as exc:
        raise RuntimeInventoryError("%s checkout disappeared" % label) from exc
    if not _same_lstat(before, after):
        raise RuntimeInventoryError(
            "%s checkout identity changed during inventory" % label
        )
    return {
        "path": str(checkout),
        "lstat": _lstat_dict(before),
        "ancestors": _ancestor_snapshots(checkout),
        "commit": commit,
        "tree": tree,
        "head_ref": head_ref,
        "tracked_tree": _bytes_snapshot(tracked_tree_raw),
        "index_flags": _bytes_snapshot(index_flags_raw),
        "status_porcelain_v2": _bytes_snapshot(status_raw),
        "remote_v": _bytes_snapshot(remote_v_raw),
        "remote_config": _bytes_snapshot(remote_config_raw),
        "clean": True,
    }


def _isaaclab_snapshot(checkout: Path) -> Dict[str, Any]:
    return _git_checkout_snapshot(checkout, "IsaacLab")


def _editable_distribution_snapshot(
    descriptor: Mapping[str, Any],
    origins: Sequence[Path],
    site_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    name = descriptor["name"]
    metadata_path = _canonical_absolute_path(
        descriptor["metadata_path"], "editable distribution metadata path"
    )
    try:
        metadata_info = metadata_path.lstat()
    except OSError as exc:
        raise RuntimeInventoryError(
            "editable distribution metadata is missing: %s" % metadata_path
        ) from exc
    if not stat.S_ISDIR(metadata_info.st_mode):
        raise RuntimeInventoryError(
            "editable distribution metadata path must be a real directory"
        )
    project_path = _canonical_absolute_path(
        descriptor["project_path"], "editable distribution project path"
    )
    if not project_path.exists():
        raise RuntimeInventoryError(
            "editable distribution project path is missing: %s" % project_path
        )
    toplevel_text = _git(
        project_path, ("rev-parse", "--show-toplevel"), "editable toplevel"
    )
    git_root = _canonical_absolute_path(
        toplevel_text, "editable distribution Git root"
    )
    checkout_before = _git_checkout_snapshot(
        git_root, "editable distribution %s" % name
    )

    origin_repo_paths = []
    for origin in origins:
        try:
            relative = origin.relative_to(git_root)
        except ValueError as exc:
            raise RuntimeInventoryError(
                "editable module origin is outside its Git root: %s" % origin
            ) from exc
        relative_text = relative.as_posix()
        if not relative_text or relative_text.startswith("../"):
            raise RuntimeInventoryError(
                "editable module origin has an unsafe Git-relative path"
            )
        _git(
            git_root,
            ("ls-files", "--error-unmatch", "--", relative_text),
            "editable tracked origin",
        )
        origin_repo_paths.append(relative_text)

    direct_url_path = _canonical_absolute_path(
        descriptor["direct_url_path"], "editable direct_url.json"
    )
    if direct_url_path != metadata_path / "direct_url.json":
        raise RuntimeInventoryError(
            "editable direct_url.json does not belong to its metadata directory"
        )
    direct_url = _open_regular_snapshot(
        direct_url_path,
        "editable distribution direct_url.json",
        include_raw=True,
    )
    direct_url_raw = _raw_snapshot_bytes(
        direct_url, "editable distribution direct_url.json"
    )
    direct_url_document = _require_mapping(
        _strict_json_loads(
            direct_url_raw, "editable distribution direct_url.json"
        ),
        "editable distribution direct_url.json",
    )
    _require_exact_keys(
        direct_url_document,
        {"url", "dir_info"},
        "editable distribution direct_url.json",
    )
    direct_url_text = _require_nonempty_line(
        direct_url_document.get("url"),
        "editable distribution direct_url.json url",
        16384,
    )
    dir_info = _require_mapping(
        direct_url_document.get("dir_info"),
        "editable distribution direct_url.json dir_info",
    )
    _require_exact_keys(
        dir_info,
        {"editable"},
        "editable distribution direct_url.json dir_info",
    )
    if dir_info["editable"] is not True:
        raise RuntimeInventoryError(
            "editable distribution direct_url.json must assert editable=true"
        )
    parsed_url = urllib.parse.urlsplit(direct_url_text)
    if (
        parsed_url.scheme != "file"
        or parsed_url.netloc not in ("", "localhost")
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise RuntimeInventoryError(
            "editable distribution direct_url.json must use a local file URL"
        )
    decoded_path = urllib.request.url2pathname(
        urllib.parse.unquote(parsed_url.path)
    )
    direct_project_path = _canonical_absolute_path(
        os.path.normpath(os.path.abspath(decoded_path)),
        "editable distribution direct_url.json project path",
    )
    if direct_project_path != project_path:
        raise RuntimeInventoryError(
            "editable distribution direct_url.json differs from project_path"
        )
    install_descriptor = dict(descriptor)
    # The editable source tree is owned by Git.  Its wheel-side RECORD owns
    # only the generated loader/.pth/dist-info artifacts, so package
    # top-level names must not redirect that RECORD scan into the source tree.
    install_descriptor["top_level_names"] = []
    install_closure = _noneditable_distribution_snapshot(install_descriptor)

    normalized = _normalized_distribution_name(name)
    selected_pth_paths = []
    for site_row in site_rows:
        for pth in site_row["pth_files"]:
            raw = _raw_snapshot_bytes(pth, "site-packages .pth")
            basename_normalized = _normalized_distribution_name(Path(pth["path"]).name)
            if (
                normalized in basename_normalized
                or str(project_path).encode("utf-8") in raw
                or str(git_root).encode("utf-8") in raw
            ):
                selected_pth_paths.append(pth["path"])
    selected_pth_paths = sorted(set(selected_pth_paths))
    if not selected_pth_paths:
        raise RuntimeInventoryError(
            "editable distribution has no exact .pth binding: %s" % name
        )

    checkout_after = _git_checkout_snapshot(
        git_root, "editable distribution %s" % name
    )
    if _canonical_json_bytes(checkout_before) != _canonical_json_bytes(
        checkout_after
    ):
        raise RuntimeInventoryError(
            "editable distribution Git checkout changed during inventory"
        )
    return {
        "name": name,
        "version": descriptor["version"],
        "metadata_path": str(metadata_path),
        "editable": True,
        "top_level_names": list(descriptor["top_level_names"]),
        "metadata_lstat": _lstat_dict(metadata_info),
        "metadata_ancestors": _ancestor_snapshots(metadata_path),
        "project_path": str(project_path),
        "direct_url": direct_url,
        "pth_paths": selected_pth_paths,
        "origin_repo_paths": sorted(set(origin_repo_paths)),
        "git_checkout": checkout_before,
        "metadata_file": install_closure["metadata_file"],
        "wheel_file": install_closure["wheel_file"],
        "record": install_closure["record"],
        "files": install_closure["files"],
        "owned_roots": install_closure["owned_roots"],
    }


def build_content(
    python: Path,
    isaaclab_checkout: Path,
    import_roots: Sequence[Path],
) -> Dict[str, Any]:
    """Build one live content document; no output file is created."""

    requested = _canonical_absolute_path(str(python), "--python")
    checkout = _canonical_absolute_path(str(isaaclab_checkout), "--isaaclab-checkout")
    nosite = _load_nosite_bootstrap()
    try:
        import_root_bindings = nosite.bind_import_roots(
            [
                _canonical_absolute_path(
                    str(path), "--import-root"
                )
                for path in import_roots
            ]
        )
    except Exception as exc:
        raise RuntimeInventoryError(
            "cannot bind the explicit no-site import roots"
        ) from exc

    chain_before, resolved = _python_chain_snapshot(requested)
    pyvenv_path = requested.parents[1] / "pyvenv.cfg"
    pyvenv_before = _open_regular_snapshot(
        pyvenv_path, "pyvenv.cfg", include_raw=True
    )

    probe = _run_probe(requested, import_root_bindings)
    _validate_probe_payload(probe, requested)

    site_rows = [
        _site_packages_snapshot(_canonical_absolute_path(path, "site-packages path"))
        for path in probe["site_package_paths"]
    ]

    module_rows: List[Dict[str, Any]] = []
    descriptors: Dict[str, Dict[str, Any]] = {}
    descriptor_origins: Dict[str, List[Path]] = {}
    for raw_descriptor in probe["resolved_distributions"]:
        descriptor = dict(raw_descriptor)
        metadata_path = str(
            _canonical_absolute_path(
                descriptor["metadata_path"],
                "resolved distribution metadata path",
            )
        )
        descriptor["metadata_path"] = metadata_path
        descriptors[metadata_path] = descriptor
        descriptor_origins[metadata_path] = []
    for raw_module in probe["modules"]:
        module = dict(raw_module)
        origin_path = _canonical_absolute_path(
            module.pop("origin_path"), "probe module origin"
        )
        module["origin"] = _open_regular_snapshot(
            origin_path, "module %s origin" % module["name"]
        )
        references = []
        for raw_descriptor in module["distributions"]:
            descriptor = dict(raw_descriptor)
            metadata_path = str(
                _canonical_absolute_path(
                    descriptor["metadata_path"], "distribution metadata path"
                )
            )
            descriptor["metadata_path"] = metadata_path
            existing = descriptors.get(metadata_path)
            if existing is None or _canonical_json_bytes(
                existing
            ) != _canonical_json_bytes(descriptor):
                raise RuntimeInventoryError(
                    "module distribution differs from resolved identity"
                )
            descriptor_origins[metadata_path].append(origin_path)
            references.append(
                {
                    "name": descriptor["name"],
                    "version": descriptor["version"],
                    "metadata_path": metadata_path,
                }
            )
        module["distributions"] = sorted(
            references, key=lambda row: (row["metadata_path"], row["name"])
        )
        module_rows.append(module)

    def snapshot_distributions() -> List[Dict[str, Any]]:
        result = []
        for metadata_path in sorted(descriptors):
            descriptor = descriptors[metadata_path]
            if descriptor["editable"]:
                row = _editable_distribution_snapshot(
                    descriptor,
                    descriptor_origins[metadata_path],
                    site_rows,
                )
            else:
                row = _noneditable_distribution_snapshot(descriptor)
            result.append(row)
        _reject_unlisted_distribution_files(result)
        return result

    distribution_rows = snapshot_distributions()
    distribution_by_path = {
        row["metadata_path"]: row for row in distribution_rows
    }
    critical_witness_rows = []
    for witness in CRITICAL_RECORD_WITNESSES:
        candidates = []
        for distribution in distribution_rows:
            for file_row in distribution["files"]:
                record_path = file_row["record_path"]
                parts = PurePosixPath(record_path).parts
                if any(part.endswith(".dist-info") for part in parts):
                    continue
                if witness in record_path.lower():
                    candidates.append(
                        (
                            distribution["metadata_path"],
                            record_path,
                            file_row,
                        )
                    )
        if not candidates:
            raise RuntimeInventoryError(
                "resolved distribution RECORD closure has no %s runtime witness"
                % witness
            )
        metadata_path, record_path, file_row = sorted(
            candidates, key=lambda row: (row[0], row[1])
        )[0]
        critical_witness_rows.append(
            {
                "witness": witness,
                "distribution_metadata_path": metadata_path,
                "record_path": record_path,
                "path": file_row["path"],
                "byte_count": file_row["byte_count"],
                "sha256": file_row["sha256"],
            }
        )
    for module in module_rows:
        origin_path = module["origin"]["path"]
        closed = False
        for reference in module["distributions"]:
            distribution = distribution_by_path[reference["metadata_path"]]
            if distribution["editable"]:
                git_root = Path(distribution["git_checkout"]["path"])
                try:
                    relative = Path(origin_path).relative_to(git_root).as_posix()
                except ValueError:
                    continue
                if relative in distribution["origin_repo_paths"]:
                    closed = True
                    break
            elif any(
                file_row["path"] == origin_path
                for file_row in distribution["files"]
            ):
                closed = True
                break
        if not closed:
            raise RuntimeInventoryError(
                "module origin is not closed by any resolved distribution: %s"
                % module["name"]
            )

    chain_after, resolved_after = _python_chain_snapshot(requested)
    pyvenv_after = _open_regular_snapshot(
        pyvenv_path, "pyvenv.cfg", include_raw=True
    )
    if (
        resolved_after != resolved
        or _canonical_json_bytes(chain_after) != _canonical_json_bytes(chain_before)
        or _canonical_json_bytes(pyvenv_after) != _canonical_json_bytes(pyvenv_before)
    ):
        raise RuntimeInventoryError("Python runtime changed during inventory")

    isaaclab_before = _isaaclab_snapshot(checkout)
    isaaclab_after = _isaaclab_snapshot(checkout)
    if _canonical_json_bytes(isaaclab_before) != _canonical_json_bytes(isaaclab_after):
        raise RuntimeInventoryError("IsaacLab checkout changed during inventory")
    distribution_rows_after = snapshot_distributions()
    if _canonical_json_bytes(distribution_rows) != _canonical_json_bytes(
        distribution_rows_after
    ):
        raise RuntimeInventoryError(
            "installed distribution closure changed during inventory"
        )
    try:
        import_root_bindings_after = nosite.bind_import_roots(
            [Path(row["path"]) for row in import_root_bindings]
        )
    except Exception as exc:
        raise RuntimeInventoryError(
            "explicit no-site import roots changed during inventory"
        ) from exc
    if import_root_bindings_after != import_root_bindings:
        raise RuntimeInventoryError(
            "explicit no-site import roots changed during inventory"
        )

    probe_record = {
        "implementation": probe["implementation"],
        "version": probe["version"],
        "cache_tag": probe["cache_tag"],
        "executable": probe["executable"],
        "prefix": probe["prefix"],
        "base_prefix": probe["base_prefix"],
        "sys_path": probe["sys_path"],
        "site_package_paths": probe["site_package_paths"],
        "modules": module_rows,
        "marker_environment": probe["marker_environment"],
        "resolved_distributions": [
            {
                "name": descriptor["name"],
                "version": descriptor["version"],
                "metadata_path": descriptor["metadata_path"],
            }
            for descriptor in probe["resolved_distributions"]
        ],
        "dependency_edges": probe["dependency_edges"],
        "optional_distributions": probe["optional_distributions"],
        "no_site_execution": probe["no_site_execution"],
    }
    return {
        "python": {
            "requested_path": str(requested),
            "resolved_path": str(resolved),
            "symlink_chain": chain_before,
            "pyvenv_cfg": pyvenv_before,
            "probe": probe_record,
            "site_packages": site_rows,
            "distributions": distribution_rows,
            "critical_record_witnesses": critical_witness_rows,
        },
        "isaaclab_checkout": isaaclab_before,
    }


def build_receipt(
    python: Path,
    isaaclab_checkout: Path,
    import_roots: Sequence[Path],
) -> Dict[str, Any]:
    content = build_content(python, isaaclab_checkout, import_roots)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "content": content,
        "content_sha256": _sha256_bytes(_canonical_json_bytes(content)),
    }


def _validate_lstat(value: Any, label: str) -> None:
    row = _require_mapping(value, label)
    _require_exact_keys(row, _LSTAT_KEYS, label)
    for key in sorted(_LSTAT_KEYS):
        _require_plain_int(row[key], "%s.%s" % (label, key))


def _validate_ancestor_lstat(value: Any, label: str) -> None:
    row = _require_mapping(value, label)
    _require_exact_keys(row, _ANCESTOR_LSTAT_KEYS, label)
    for key in sorted(_ANCESTOR_LSTAT_KEYS):
        _require_plain_int(row[key], "%s.%s" % (label, key))


def _validate_ancestor_list(value: Any, path: Path, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise RuntimeInventoryError("%s must be a non-empty list" % label)
    expected_paths = [str(item) for item in _ancestor_paths(path)]
    actual_paths: List[str] = []
    for index, raw_row in enumerate(value):
        row = _require_mapping(raw_row, "%s[%d]" % (label, index))
        kind = row.get("kind")
        keys = {"path", "kind", "lstat", "link_text"} if kind == "symlink" else {
            "path",
            "kind",
            "lstat",
        }
        _require_exact_keys(row, keys, "%s[%d]" % (label, index))
        ancestor = _canonical_absolute_path(
            row["path"], "%s[%d].path" % (label, index)
        )
        actual_paths.append(str(ancestor))
        if kind not in {"directory", "symlink"}:
            raise RuntimeInventoryError("%s ancestor kind is invalid" % label)
        _validate_ancestor_lstat(
            row["lstat"], "%s[%d].lstat" % (label, index)
        )
        if _kind_from_mode(row["lstat"]["st_mode"]) != kind:
            raise RuntimeInventoryError("%s ancestor kind disagrees with mode" % label)
        if kind == "symlink":
            _require_nonempty_line(
                row["link_text"], "%s[%d].link_text" % (label, index), 16384
            )
    if actual_paths != expected_paths:
        raise RuntimeInventoryError("%s does not bind the exact ancestor chain" % label)


def _validate_regular_file_row(value: Any, label: str) -> Path:
    row = _require_mapping(value, label)
    _require_exact_keys(
        row,
        {"path", "lstat", "ancestors", "byte_count", "sha256"},
        label,
    )
    path = _canonical_absolute_path(row["path"], "%s.path" % label)
    _validate_lstat(row["lstat"], "%s.lstat" % label)
    if _kind_from_mode(row["lstat"]["st_mode"]) != "regular":
        raise RuntimeInventoryError("%s lstat is not regular" % label)
    _validate_ancestor_list(row["ancestors"], path, "%s.ancestors" % label)
    count = _require_plain_int(row["byte_count"], "%s.byte_count" % label)
    if count != row["lstat"]["st_size"]:
        raise RuntimeInventoryError("%s byte_count differs from lstat size" % label)
    _require_sha256(row["sha256"], "%s.sha256" % label)
    return path


def _validate_raw_regular_file_row(value: Any, label: str) -> Path:
    row = _require_mapping(value, label)
    _require_exact_keys(
        row,
        {
            "path",
            "lstat",
            "ancestors",
            "byte_count",
            "sha256",
            "raw_base64",
        },
        label,
    )
    reduced = dict(row)
    reduced.pop("raw_base64")
    path = _validate_regular_file_row(reduced, label)
    _raw_snapshot_bytes(row, label)
    return path


def _validate_git_checkout_row(value: Any, label: str) -> Path:
    checkout = _require_mapping(value, label)
    _require_exact_keys(
        checkout,
        {
            "path",
            "lstat",
            "ancestors",
            "commit",
            "tree",
            "head_ref",
            "tracked_tree",
            "index_flags",
            "status_porcelain_v2",
            "remote_v",
            "remote_config",
            "clean",
        },
        label,
    )
    path = _canonical_absolute_path(checkout["path"], "%s.path" % label)
    _validate_lstat(checkout["lstat"], "%s.lstat" % label)
    if _kind_from_mode(checkout["lstat"]["st_mode"]) != "directory":
        raise RuntimeInventoryError("%s lstat is not directory" % label)
    _validate_ancestor_list(checkout["ancestors"], path, "%s.ancestors" % label)
    for key in ("commit", "tree"):
        if (
            type(checkout[key]) is not str
            or GIT_OBJECT_RE.fullmatch(checkout[key]) is None
        ):
            raise RuntimeInventoryError(
                "%s.%s must be a SHA-1 object" % (label, key)
            )
    _require_nonempty_line(checkout["head_ref"], "%s.head_ref" % label, 16384)
    tracked_tree = _validate_bytes_snapshot(
        checkout["tracked_tree"], "%s.tracked_tree" % label
    )
    if not tracked_tree:
        raise RuntimeInventoryError("%s tracked tree cannot be empty" % label)
    index_flags = _validate_bytes_snapshot(
        checkout["index_flags"], "%s.index_flags" % label
    )
    if not index_flags:
        raise RuntimeInventoryError("%s index flags cannot be empty" % label)
    status = _validate_bytes_snapshot(
        checkout["status_porcelain_v2"],
        "%s.status_porcelain_v2" % label,
    )
    if status:
        raise RuntimeInventoryError("%s status must be exactly clean" % label)
    if not _validate_bytes_snapshot(
        checkout["remote_v"], "%s.remote_v" % label
    ):
        raise RuntimeInventoryError("%s remote_v cannot be empty" % label)
    if not _validate_bytes_snapshot(
        checkout["remote_config"], "%s.remote_config" % label
    ):
        raise RuntimeInventoryError("%s remote_config cannot be empty" % label)
    if checkout["clean"] is not True:
        raise RuntimeInventoryError("%s must assert exact clean state" % label)
    return path


def _validate_distribution_record(
    distribution: Mapping[str, Any],
    metadata_path: Path,
    label: str,
) -> None:
    record = _require_mapping(distribution["record"], "%s.record" % label)
    record_path = _validate_raw_regular_file_row(record, "%s.record" % label)
    if record_path != metadata_path / "RECORD":
        raise RuntimeInventoryError("%s RECORD path is inconsistent" % label)
    raw = _raw_snapshot_bytes(record, "%s.record" % label)
    try:
        text = raw.decode("utf-8")
        parsed_rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise RuntimeInventoryError("%s RECORD bytes are invalid" % label) from exc
    if not parsed_rows:
        raise RuntimeInventoryError("%s RECORD cannot be empty" % label)

    files = distribution["files"]
    if not isinstance(files, list) or not files:
        raise RuntimeInventoryError("%s files must be a non-empty list" % label)
    parsed_files = []
    seen_record_paths = set()
    seen_paths = set()
    for index, raw_file in enumerate(files):
        file_row = _require_mapping(
            raw_file, "%s.files[%d]" % (label, index)
        )
        _require_exact_keys(
            file_row,
            {
                "record_path",
                "record_hash",
                "record_size",
                "path",
                "byte_count",
                "sha256",
            },
            "%s.files[%d]" % (label, index),
        )
        record_name = _require_nonempty_line(
            file_row["record_path"], "distribution record_path", 16384
        )
        expected_path = _record_absolute_path(metadata_path, record_name)
        path = _canonical_absolute_path(
            file_row["path"], "distribution file path"
        )
        if path != expected_path:
            raise RuntimeInventoryError(
                "%s file path differs from RECORD resolution" % label
            )
        if record_name in seen_record_paths or str(path) in seen_paths:
            raise RuntimeInventoryError("%s files contain a duplicate" % label)
        seen_record_paths.add(record_name)
        seen_paths.add(str(path))
        declared_hash = file_row["record_hash"]
        if type(declared_hash) is not str:
            raise RuntimeInventoryError(
                "%s record_hash must be a string" % label
            )
        declared_sha = _declared_record_sha256(declared_hash, label)
        actual_sha = _require_sha256(
            file_row["sha256"], "%s file sha256" % label
        )
        if declared_sha is not None and declared_sha != actual_sha:
            raise RuntimeInventoryError(
                "%s declared RECORD hash differs from computed sha256" % label
            )
        byte_count = _require_plain_int(
            file_row["byte_count"], "%s file byte_count" % label
        )
        record_size = file_row["record_size"]
        if record_size is not None:
            record_size = _require_plain_int(
                record_size, "%s file record_size" % label
            )
            if record_size != byte_count:
                raise RuntimeInventoryError(
                    "%s RECORD size differs from byte_count" % label
                )
        parsed_files.append(
            (
                record_name,
                declared_hash,
                "" if record_size is None else str(record_size),
            )
        )
    if [row[0] for row in parsed_files] != sorted(seen_record_paths):
        raise RuntimeInventoryError(
            "%s files must be sorted by RECORD path" % label
        )
    if sorted(tuple(row) for row in parsed_rows) != parsed_files:
        raise RuntimeInventoryError(
            "%s parsed RECORD rows differ from files closure" % label
        )
    if str(record_path) not in seen_paths:
        raise RuntimeInventoryError("%s RECORD must list itself" % label)
    files_by_path = {
        file_row["path"]: file_row for file_row in files
    }
    for key, basename in (
        ("metadata_file", "METADATA"),
        ("wheel_file", "WHEEL"),
    ):
        required_path = _validate_raw_regular_file_row(
            distribution[key], "%s.%s" % (label, key)
        )
        expected_path = metadata_path / basename
        if required_path != expected_path:
            raise RuntimeInventoryError(
                "%s %s path is inconsistent" % (label, basename)
            )
        closure_row = files_by_path.get(str(required_path))
        if closure_row is None:
            raise RuntimeInventoryError(
                "%s RECORD closure does not contain %s" % (label, basename)
            )
        if (
            not closure_row["record_hash"]
            or closure_row["record_size"] is None
            or closure_row["sha256"] != distribution[key]["sha256"]
            or closure_row["byte_count"] != distribution[key]["byte_count"]
        ):
            raise RuntimeInventoryError(
                "%s %s differs from its RECORD closure" % (label, basename)
            )

    owned_roots = distribution["owned_roots"]
    if not isinstance(owned_roots, list) or not owned_roots:
        raise RuntimeInventoryError(
            "%s owned_roots must be a non-empty list" % label
        )
    parsed_roots = [
        str(_canonical_absolute_path(value, "%s owned root" % label))
        for value in owned_roots
    ]
    if parsed_roots != sorted(set(parsed_roots)):
        raise RuntimeInventoryError(
            "%s owned_roots must be sorted unique" % label
        )
    if str(metadata_path) not in parsed_roots:
        raise RuntimeInventoryError(
            "%s owned_roots must include metadata_path" % label
        )


def _validate_distribution_row(value: Any, label: str) -> Mapping[str, Any]:
    row = _require_mapping(value, label)
    editable = row.get("editable")
    common = {
        "name",
        "version",
        "metadata_path",
        "editable",
        "top_level_names",
        "metadata_lstat",
        "metadata_ancestors",
        "metadata_file",
        "wheel_file",
        "record",
        "files",
        "owned_roots",
    }
    expected = (
        common
        | {
            "project_path",
            "direct_url",
            "pth_paths",
            "origin_repo_paths",
            "git_checkout",
        }
        if editable is True
        else common
    )
    _require_exact_keys(row, expected, label)
    _require_nonempty_line(row["name"], "%s.name" % label, 256)
    _require_nonempty_line(row["version"], "%s.version" % label, 256)
    metadata_path = _canonical_absolute_path(
        row["metadata_path"], "%s.metadata_path" % label
    )
    if not metadata_path.name.endswith(".dist-info"):
        raise RuntimeInventoryError(
            "%s metadata_path must name a .dist-info directory" % label
        )
    _validate_lstat(row["metadata_lstat"], "%s.metadata_lstat" % label)
    if _kind_from_mode(row["metadata_lstat"]["st_mode"]) != "directory":
        raise RuntimeInventoryError("%s metadata is not a directory" % label)
    _validate_ancestor_list(
        row["metadata_ancestors"],
        metadata_path,
        "%s.metadata_ancestors" % label,
    )
    if type(editable) is not bool:
        raise RuntimeInventoryError("%s.editable must be boolean" % label)
    top_levels = row["top_level_names"]
    if not isinstance(top_levels, list) or top_levels != sorted(set(top_levels)):
        raise RuntimeInventoryError(
            "%s top_level_names must be sorted unique" % label
        )
    for value in top_levels:
        top_level = _require_nonempty_line(
            value, "%s top-level name" % label, 256
        )
        if TOP_LEVEL_NAME_RE.fullmatch(top_level) is None:
            raise RuntimeInventoryError("%s top-level name is unsafe" % label)
    _validate_distribution_record(row, metadata_path, label)
    if editable:
        project_path = _canonical_absolute_path(
            row["project_path"], "%s.project_path" % label
        )
        direct_url = _validate_raw_regular_file_row(
            row["direct_url"], "%s.direct_url" % label
        )
        if direct_url != metadata_path / "direct_url.json":
            raise RuntimeInventoryError(
                "%s direct_url path is inconsistent" % label
            )
        direct_url_document = _require_mapping(
            _strict_json_loads(
                _raw_snapshot_bytes(row["direct_url"], "%s.direct_url" % label),
                "%s.direct_url" % label,
            ),
            "%s.direct_url" % label,
        )
        _require_exact_keys(
            direct_url_document,
            {"url", "dir_info"},
            "%s.direct_url" % label,
        )
        dir_info = _require_mapping(
            direct_url_document.get("dir_info"), "%s.direct_url.dir_info" % label
        )
        _require_exact_keys(
            dir_info, {"editable"}, "%s.direct_url.dir_info" % label
        )
        if dir_info["editable"] is not True:
            raise RuntimeInventoryError(
                "%s direct_url must assert editable=true" % label
            )
        url_text = _require_nonempty_line(
            direct_url_document.get("url"), "%s.direct_url.url" % label, 16384
        )
        parsed_url = urllib.parse.urlsplit(url_text)
        if (
            parsed_url.scheme != "file"
            or parsed_url.netloc not in ("", "localhost")
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise RuntimeInventoryError(
                "%s direct_url must use a local file URL" % label
            )
        decoded_path = urllib.request.url2pathname(
            urllib.parse.unquote(parsed_url.path)
        )
        url_project_path = _canonical_absolute_path(
            os.path.normpath(os.path.abspath(decoded_path)),
            "%s direct_url project path" % label,
        )
        if url_project_path != project_path:
            raise RuntimeInventoryError(
                "%s direct_url differs from project_path" % label
            )
        git_root = _validate_git_checkout_row(
            row["git_checkout"], "%s.git_checkout" % label
        )
        try:
            project_path.relative_to(git_root)
        except ValueError as exc:
            raise RuntimeInventoryError(
                "%s project_path is outside Git checkout" % label
            ) from exc
        pth_paths = row["pth_paths"]
        if not isinstance(pth_paths, list) or not pth_paths:
            raise RuntimeInventoryError(
                "%s pth_paths must be non-empty" % label
            )
        parsed_pth = [
            str(_canonical_absolute_path(value, "%s pth path" % label))
            for value in pth_paths
        ]
        if parsed_pth != sorted(set(parsed_pth)):
            raise RuntimeInventoryError(
                "%s pth_paths must be sorted unique" % label
            )
        repo_paths = row["origin_repo_paths"]
        if not isinstance(repo_paths, list):
            raise RuntimeInventoryError(
                "%s origin_repo_paths must be a list" % label
            )
        if repo_paths != sorted(set(repo_paths)):
            raise RuntimeInventoryError(
                "%s origin_repo_paths must be sorted unique" % label
            )
        for repo_path in repo_paths:
            value = _require_nonempty_line(
                repo_path, "%s origin repo path" % label, 16384
            )
            pure = PurePosixPath(value)
            if pure.is_absolute() or str(pure) != value or ".." in pure.parts:
                raise RuntimeInventoryError(
                    "%s origin repo path is unsafe" % label
                )
    return row


def _validate_content_schema(content_value: Any) -> Mapping[str, Any]:
    content = _require_mapping(content_value, "content")
    _require_exact_keys(content, {"python", "isaaclab_checkout"}, "content")
    python = _require_mapping(content["python"], "content.python")
    _require_exact_keys(
        python,
        {
            "requested_path",
            "resolved_path",
            "symlink_chain",
            "pyvenv_cfg",
            "probe",
            "site_packages",
            "distributions",
            "critical_record_witnesses",
        },
        "content.python",
    )
    requested = _canonical_absolute_path(
        python["requested_path"], "content.python.requested_path"
    )
    resolved = _canonical_absolute_path(
        python["resolved_path"], "content.python.resolved_path"
    )
    chain = python["symlink_chain"]
    if not isinstance(chain, list) or len(chain) < 2:
        raise RuntimeInventoryError("Python symlink_chain must contain a link and final file")
    prior_path: Path = requested
    seen = set()
    for index, raw_row in enumerate(chain):
        row = _require_mapping(raw_row, "Python chain[%d]" % index)
        kind = row.get("kind")
        keys = (
            {"index", "kind", "path", "link_text", "lstat", "ancestors"}
            if kind == "symlink"
            else {
                "index",
                "kind",
                "path",
                "lstat",
                "ancestors",
                "byte_count",
                "sha256",
            }
        )
        _require_exact_keys(row, keys, "Python chain[%d]" % index)
        if row["index"] != index:
            raise RuntimeInventoryError("Python chain indices are not contiguous")
        path = _canonical_absolute_path(row["path"], "Python chain path")
        if path != prior_path or str(path) in seen:
            raise RuntimeInventoryError("Python chain path/order is inconsistent")
        seen.add(str(path))
        _validate_lstat(row["lstat"], "Python chain lstat")
        if _kind_from_mode(row["lstat"]["st_mode"]) != kind:
            raise RuntimeInventoryError("Python chain kind disagrees with lstat mode")
        _validate_ancestor_list(row["ancestors"], path, "Python chain ancestors")
        if kind == "symlink":
            link_text = _require_nonempty_line(
                row["link_text"], "Python chain link_text", 16384
            )
            target = Path(link_text)
            prior_path = Path(
                os.path.normpath(
                    link_text if target.is_absolute() else str(path.parent / target)
                )
            )
        elif kind == "regular":
            if index != len(chain) - 1:
                raise RuntimeInventoryError("Python regular entry must end the chain")
            count = _require_plain_int(row["byte_count"], "Python executable byte_count")
            if count != row["lstat"]["st_size"]:
                raise RuntimeInventoryError(
                    "Python executable byte_count differs from lstat"
                )
            _require_sha256(row["sha256"], "Python executable sha256")
            if path != resolved:
                raise RuntimeInventoryError("Python resolved_path differs from chain end")
        else:
            raise RuntimeInventoryError("Python chain kind must be symlink or regular")
    if chain[0]["kind"] != "symlink" or chain[-1]["kind"] != "regular":
        raise RuntimeInventoryError("Python chain must start at a link and end in regular")

    pyvenv_path = _validate_raw_regular_file_row(
        python["pyvenv_cfg"], "content.python.pyvenv_cfg"
    )
    if pyvenv_path != requested.parents[1] / "pyvenv.cfg":
        raise RuntimeInventoryError("pyvenv.cfg is not rooted at the requested venv")

    probe = _require_mapping(python["probe"], "content.python.probe")
    _require_exact_keys(
        probe,
        {
            "implementation",
            "version",
            "cache_tag",
            "executable",
            "prefix",
            "base_prefix",
            "sys_path",
            "site_package_paths",
            "modules",
            "marker_environment",
            "resolved_distributions",
            "dependency_edges",
            "optional_distributions",
            "no_site_execution",
        },
        "content.python.probe",
    )
    _require_nonempty_line(probe["implementation"], "probe implementation", 128)
    _require_nonempty_line(probe["version"], "probe version", 128)
    _require_nonempty_line(probe["cache_tag"], "probe cache_tag", 128)
    probe_executable = _canonical_absolute_path(
        probe["executable"], "probe executable"
    )
    allowed_probe_executables = {
        _canonical_absolute_path(row["path"], "Python chain executable")
        for row in chain
    }
    try:
        probe_executable_resolved = probe_executable.resolve(strict=True)
    except OSError as exc:
        raise RuntimeInventoryError(
            "probe executable cannot be resolved"
        ) from exc
    if (
        probe_executable not in allowed_probe_executables
        and probe_executable_resolved != resolved.resolve(strict=True)
    ):
        raise RuntimeInventoryError(
            "probe executable is neither a member nor a resolved alias of "
            "the bound Python chain"
        )
    _canonical_absolute_path(probe["prefix"], "probe prefix")
    _canonical_absolute_path(probe["base_prefix"], "probe base_prefix")
    explicit_root_paths = _validate_nosite_probe_execution(
        probe["no_site_execution"], requested
    )
    sys_path = probe["sys_path"]
    if not isinstance(sys_path, list):
        raise RuntimeInventoryError("probe sys_path must be a list")
    for value in sys_path:
        _canonical_absolute_path(value, "probe sys_path entry")
    site_paths = probe["site_package_paths"]
    if not isinstance(site_paths, list) or not site_paths:
        raise RuntimeInventoryError("probe site_package_paths must be non-empty")
    parsed_site_paths = [
        str(_canonical_absolute_path(value, "probe site_package_path"))
        for value in site_paths
    ]
    if parsed_site_paths != sorted(set(parsed_site_paths)):
        raise RuntimeInventoryError("probe site_package_paths must be sorted unique")
    if parsed_site_paths != sorted(explicit_root_paths):
        raise RuntimeInventoryError(
            "probe site_package_paths differ from explicit no-site roots"
        )
    if probe["sys_path"] != probe["no_site_execution"]["inner"]["sys_path"]:
        raise RuntimeInventoryError(
            "probe sys_path differs from inner no-site sys.path"
        )

    modules = probe["modules"]
    if not isinstance(modules, list) or len(modules) != len(MODULE_NAMES):
        raise RuntimeInventoryError("probe modules differ from fixed module count")
    if [row.get("name") if isinstance(row, dict) else None for row in modules] != list(
        MODULE_NAMES
    ):
        raise RuntimeInventoryError("probe modules differ from fixed module order")
    for index, raw_module in enumerate(modules):
        module = _require_mapping(raw_module, "probe module[%d]" % index)
        _require_exact_keys(
            module,
            {
                "name",
                "version",
                "version_source",
                "distributions",
                "origin",
            },
            "probe module[%d]" % index,
        )
        _require_nonempty_line(module["name"], "module name", 128)
        _require_nonempty_line(module["version"], "module version", 256)
        _require_nonempty_line(module["version_source"], "module version source", 256)
        distributions = module["distributions"]
        if not isinstance(distributions, list) or not distributions:
            raise RuntimeInventoryError("module distributions must be non-empty")
        seen_dist = set()
        parsed_dist_order = []
        for raw_dist in distributions:
            dist = _require_mapping(raw_dist, "module distribution")
            _require_exact_keys(
                dist,
                {"name", "version", "metadata_path"},
                "module distribution",
            )
            name = _require_nonempty_line(dist["name"], "distribution name", 256)
            _require_nonempty_line(dist["version"], "distribution version", 256)
            metadata_path = _canonical_absolute_path(
                dist["metadata_path"], "module distribution metadata_path"
            )
            identity = (str(metadata_path), name)
            if identity in seen_dist:
                raise RuntimeInventoryError(
                    "module distribution identities must be unique"
                )
            seen_dist.add(identity)
            parsed_dist_order.append(identity)
        if parsed_dist_order != sorted(parsed_dist_order):
            raise RuntimeInventoryError(
                "module distributions must be sorted by metadata_path/name"
            )
        if not any(
            module["version_source"] == "distribution:" + dist["name"]
            and module["version"] == dist["version"]
            for dist in distributions
        ):
            raise RuntimeInventoryError(
                "module version_source/version has no distribution reference"
            )
        _validate_regular_file_row(module["origin"], "module origin")

    marker_environment = _require_mapping(
        probe["marker_environment"], "probe marker_environment"
    )
    if not marker_environment:
        raise RuntimeInventoryError("probe marker_environment must be non-empty")
    for key, value in marker_environment.items():
        _require_nonempty_line(key, "probe marker key", 256)
        _require_nonempty_line(value, "probe marker value", 4096)
    probe_resolved = probe["resolved_distributions"]
    if not isinstance(probe_resolved, list) or not probe_resolved:
        raise RuntimeInventoryError(
            "probe resolved_distributions must be non-empty"
        )
    probe_resolved_by_path = {}
    probe_resolved_order = []
    for raw_reference in probe_resolved:
        reference = _require_mapping(
            raw_reference, "probe resolved distribution"
        )
        _require_exact_keys(
            reference,
            {"name", "version", "metadata_path"},
            "probe resolved distribution",
        )
        name = _require_nonempty_line(
            reference["name"], "probe resolved name", 256
        )
        _require_nonempty_line(
            reference["version"], "probe resolved version", 256
        )
        metadata_path = str(
            _canonical_absolute_path(
                reference["metadata_path"],
                "probe resolved metadata_path",
            )
        )
        if metadata_path in probe_resolved_by_path:
            raise RuntimeInventoryError(
                "probe resolved metadata_path is duplicated"
            )
        probe_resolved_by_path[metadata_path] = reference
        probe_resolved_order.append((metadata_path, name))
    if probe_resolved_order != sorted(probe_resolved_order):
        raise RuntimeInventoryError(
            "probe resolved distributions must be sorted"
        )
    dependency_edges = probe["dependency_edges"]
    if not isinstance(dependency_edges, list):
        raise RuntimeInventoryError("probe dependency_edges must be a list")
    edge_order = []
    adjacency = {}
    for raw_edge in dependency_edges:
        edge = _require_mapping(raw_edge, "probe dependency edge")
        _require_exact_keys(
            edge,
            {
                "from_metadata_path",
                "requirement",
                "to_metadata_path",
                "to_name",
                "to_version",
            },
            "probe dependency edge",
        )
        source = str(
            _canonical_absolute_path(
                edge["from_metadata_path"], "probe dependency source"
            )
        )
        target = str(
            _canonical_absolute_path(
                edge["to_metadata_path"], "probe dependency target"
            )
        )
        requirement = _require_nonempty_line(
            edge["requirement"], "probe dependency requirement", 16384
        )
        to_name = _require_nonempty_line(
            edge["to_name"], "probe dependency target name", 256
        )
        to_version = _require_nonempty_line(
            edge["to_version"], "probe dependency target version", 256
        )
        target_row = probe_resolved_by_path.get(target)
        if source not in probe_resolved_by_path or target_row is None:
            raise RuntimeInventoryError(
                "probe dependency edge leaves resolved closure"
            )
        if (
            target_row["name"] != to_name
            or target_row["version"] != to_version
        ):
            raise RuntimeInventoryError(
                "probe dependency target identity differs"
            )
        identity = (source, requirement, target, to_name, to_version)
        edge_order.append(identity)
        adjacency.setdefault(source, set()).add(target)
    if edge_order != sorted(set(edge_order)):
        raise RuntimeInventoryError(
            "probe dependency_edges must be sorted unique"
        )
    optional_distributions = probe["optional_distributions"]
    if (
        not isinstance(optional_distributions, list)
        or len(optional_distributions) != len(OPTIONAL_DISTRIBUTION_NAMES)
    ):
        raise RuntimeInventoryError(
            "probe optional_distributions differs from fixed contract"
        )
    optional_roots = set()
    for expected_name, raw_optional in zip(
        OPTIONAL_DISTRIBUTION_NAMES, optional_distributions
    ):
        optional = _require_mapping(
            raw_optional, "probe optional distribution"
        )
        _require_exact_keys(
            optional,
            {"name", "present", "version", "metadata_path"},
            "probe optional distribution",
        )
        name = _require_nonempty_line(
            optional["name"], "probe optional name", 256
        )
        if _normalized_distribution_name(name) != (
            _normalized_distribution_name(expected_name)
        ):
            raise RuntimeInventoryError(
                "probe optional distribution name/order differs"
            )
        if type(optional["present"]) is not bool:
            raise RuntimeInventoryError(
                "probe optional present must be boolean"
            )
        if optional["present"]:
            _require_nonempty_line(
                optional["version"], "probe optional version", 256
            )
            metadata_path = str(
                _canonical_absolute_path(
                    optional["metadata_path"],
                    "probe optional metadata_path",
                )
            )
            reference = probe_resolved_by_path.get(metadata_path)
            if (
                reference is None
                or reference["name"] != optional["name"]
                or reference["version"] != optional["version"]
            ):
                raise RuntimeInventoryError(
                    "present optional distribution differs from closure"
                )
            optional_roots.add(metadata_path)
        elif (
            optional["version"] is not None
            or optional["metadata_path"] is not None
        ):
            raise RuntimeInventoryError(
                "absent optional distribution carries identity"
            )

    root_paths = set(optional_roots) | {
        reference["metadata_path"]
        for module in modules
        for reference in module["distributions"]
    }
    reachable = set(root_paths)
    pending = sorted(root_paths)
    while pending:
        source = pending.pop(0)
        for target in sorted(adjacency.get(source, ())):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    if reachable != set(probe_resolved_by_path):
        raise RuntimeInventoryError(
            "probe resolved dependency closure is not root-reachable"
        )

    site_packages = python["site_packages"]
    if not isinstance(site_packages, list) or len(site_packages) != len(site_paths):
        raise RuntimeInventoryError("site_packages rows differ from probed paths")
    actual_site_paths: List[str] = []
    for index, raw_site in enumerate(site_packages):
        site_row = _require_mapping(raw_site, "site_packages[%d]" % index)
        _require_exact_keys(
            site_row, {"path", "lstat", "ancestors", "pth_files"}, "site_packages row"
        )
        path = _canonical_absolute_path(site_row["path"], "site_packages path")
        actual_site_paths.append(str(path))
        _validate_lstat(site_row["lstat"], "site_packages lstat")
        if _kind_from_mode(site_row["lstat"]["st_mode"]) != "directory":
            raise RuntimeInventoryError("site_packages row is not a directory")
        _validate_ancestor_list(site_row["ancestors"], path, "site_packages ancestors")
        pth_files = site_row["pth_files"]
        if not isinstance(pth_files, list):
            raise RuntimeInventoryError("site_packages pth_files must be a list")
        pth_paths: List[str] = []
        for raw_file in pth_files:
            file_path = _validate_raw_regular_file_row(
                raw_file, "site-packages .pth"
            )
            if file_path.parent != path or not file_path.name.endswith(".pth"):
                raise RuntimeInventoryError(".pth file is outside its site-packages directory")
            pth_paths.append(str(file_path))
        if pth_paths != sorted(set(pth_paths)):
            raise RuntimeInventoryError(".pth files must be sorted and unique")
    if actual_site_paths != parsed_site_paths:
        raise RuntimeInventoryError("site_packages rows differ from probe path order")

    raw_distribution_rows = python["distributions"]
    if (
        not isinstance(raw_distribution_rows, list)
        or not raw_distribution_rows
    ):
        raise RuntimeInventoryError(
            "content.python.distributions must be a non-empty list"
        )
    distribution_rows = []
    distribution_by_path = {}
    distribution_order = []
    for index, raw_distribution in enumerate(raw_distribution_rows):
        distribution = _validate_distribution_row(
            raw_distribution, "python.distributions[%d]" % index
        )
        metadata_path = distribution["metadata_path"]
        identity = (metadata_path, distribution["name"])
        if metadata_path in distribution_by_path:
            raise RuntimeInventoryError(
                "python distributions contain duplicate metadata_path"
            )
        distribution_by_path[metadata_path] = distribution
        distribution_order.append(identity)
        distribution_rows.append(distribution)
    if distribution_order != sorted(distribution_order):
        raise RuntimeInventoryError(
            "python distributions must be sorted by metadata_path/name"
        )
    if set(distribution_by_path) != set(probe_resolved_by_path):
        raise RuntimeInventoryError(
            "distribution snapshots differ from the resolved dependency graph"
        )
    for metadata_path, distribution in distribution_by_path.items():
        reference = probe_resolved_by_path[metadata_path]
        if (
            distribution["name"] != reference["name"]
            or distribution["version"] != reference["version"]
        ):
            raise RuntimeInventoryError(
                "distribution snapshot identity differs from probe graph"
            )

    critical_witnesses = python["critical_record_witnesses"]
    if (
        not isinstance(critical_witnesses, list)
        or len(critical_witnesses) != len(CRITICAL_RECORD_WITNESSES)
    ):
        raise RuntimeInventoryError(
            "critical_record_witnesses must cover the fixed witness set"
        )
    if [
        row.get("witness") if isinstance(row, dict) else None
        for row in critical_witnesses
    ] != list(CRITICAL_RECORD_WITNESSES):
        raise RuntimeInventoryError(
            "critical_record_witnesses order differs from fixed contract"
        )
    for raw_witness in critical_witnesses:
        witness = _require_mapping(
            raw_witness, "critical RECORD witness"
        )
        _require_exact_keys(
            witness,
            {
                "witness",
                "distribution_metadata_path",
                "record_path",
                "path",
                "byte_count",
                "sha256",
            },
            "critical RECORD witness",
        )
        token = _require_nonempty_line(
            witness["witness"], "critical witness token", 64
        )
        metadata_path = str(
            _canonical_absolute_path(
                witness["distribution_metadata_path"],
                "critical witness distribution metadata_path",
            )
        )
        distribution = distribution_by_path.get(metadata_path)
        if distribution is None:
            raise RuntimeInventoryError(
                "critical witness distribution is absent"
            )
        record_path = _require_nonempty_line(
            witness["record_path"], "critical witness record_path", 16384
        )
        path = str(
            _canonical_absolute_path(
                witness["path"], "critical witness path"
            )
        )
        byte_count = _require_plain_int(
            witness["byte_count"], "critical witness byte_count"
        )
        digest = _require_sha256(
            witness["sha256"], "critical witness sha256"
        )
        if token not in record_path.lower():
            raise RuntimeInventoryError(
                "critical witness token is absent from RECORD path"
            )
        if any(
            part.endswith(".dist-info")
            for part in PurePosixPath(record_path).parts
        ):
            raise RuntimeInventoryError(
                "critical witness cannot be a dist-info metadata file"
            )
        if not any(
            row["record_path"] == record_path
            and row["path"] == path
            and row["byte_count"] == byte_count
            and row["sha256"] == digest
            for row in distribution["files"]
        ):
            raise RuntimeInventoryError(
                "critical witness differs from its RECORD closure"
            )

    all_pth_paths = {
        row["path"]
        for site_row in site_packages
        for row in site_row["pth_files"]
    }
    for distribution in distribution_rows:
        if distribution["editable"]:
            if not set(distribution["pth_paths"]).issubset(all_pth_paths):
                raise RuntimeInventoryError(
                    "editable distribution references an unbound .pth file"
                )

    for module in modules:
        origin_path = module["origin"]["path"]
        closed = False
        for reference in module["distributions"]:
            distribution = distribution_by_path.get(reference["metadata_path"])
            if distribution is None:
                raise RuntimeInventoryError(
                    "module references an absent distribution closure"
                )
            if (
                reference["name"] != distribution["name"]
                or reference["version"] != distribution["version"]
            ):
                raise RuntimeInventoryError(
                    "module distribution reference differs from its closure"
                )
            if distribution["editable"]:
                git_root = Path(distribution["git_checkout"]["path"])
                try:
                    relative = Path(origin_path).relative_to(
                        git_root
                    ).as_posix()
                except ValueError:
                    continue
                if relative in distribution["origin_repo_paths"]:
                    closed = True
                    break
            elif any(
                file_row["path"] == origin_path
                for file_row in distribution["files"]
            ):
                closed = True
                break
        if not closed:
            raise RuntimeInventoryError(
                "module origin is not closed by its distribution references"
            )

    _validate_git_checkout_row(
        content["isaaclab_checkout"], "isaaclab_checkout"
    )
    return content


def validate_receipt_document(value: Any) -> Mapping[str, Any]:
    receipt = _require_mapping(value, "receipt")
    _require_exact_keys(
        receipt, {"schema_version", "kind", "content", "content_sha256"}, "receipt"
    )
    if receipt["schema_version"] != SCHEMA_VERSION:
        raise RuntimeInventoryError("runtime inventory schema_version is unsupported")
    if receipt["kind"] != RECEIPT_KIND:
        raise RuntimeInventoryError("runtime inventory kind is unsupported")
    content = _validate_content_schema(receipt["content"])
    expected_digest = _sha256_bytes(_canonical_json_bytes(content))
    if _require_sha256(receipt["content_sha256"], "content_sha256") != expected_digest:
        raise RuntimeInventoryError("runtime inventory content_sha256 mismatch")
    return receipt


def _read_receipt(path: Path) -> Tuple[Mapping[str, Any], bytes]:
    canonical = _canonical_absolute_path(str(path), "--receipt")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeInventoryError("O_NOFOLLOW is required on this platform")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(canonical), flags)
    except OSError as exc:
        raise RuntimeInventoryError("cannot open receipt without following links") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeInventoryError("receipt is not a regular file")
        if stat.S_IMODE(before.st_mode) != 0o600:
            raise RuntimeInventoryError("receipt mode must remain exactly 0600")
        chunks: List[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if not _same_lstat(before, after):
            raise RuntimeInventoryError("receipt changed while being read")
    finally:
        os.close(descriptor)
    try:
        final_lstat = canonical.lstat()
    except OSError as exc:
        raise RuntimeInventoryError(
            "receipt disappeared after it was read"
        ) from exc
    if not _same_lstat(before, final_lstat):
        raise RuntimeInventoryError(
            "receipt path identity changed after it was read"
        )
    raw = b"".join(chunks)
    document = _strict_json_loads(raw, "runtime inventory receipt")
    receipt = validate_receipt_document(document)
    expected_raw = _canonical_json_bytes(receipt) + b"\n"
    if raw != expected_raw:
        raise RuntimeInventoryError("runtime inventory receipt is not canonical JSON")
    return receipt, raw


def _exclusive_write(path: Path, raw: bytes) -> None:
    output = _canonical_absolute_path(str(path), "--output")
    parent = output.parent
    try:
        parent_info = parent.lstat()
    except OSError as exc:
        raise RuntimeInventoryError("output parent does not exist") from exc
    if not stat.S_ISDIR(parent_info.st_mode):
        raise RuntimeInventoryError("output parent must be a real directory")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeInventoryError("O_NOFOLLOW is required on this platform")
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
    )
    try:
        parent_fd = os.open(str(parent), parent_flags)
    except OSError as exc:
        raise RuntimeInventoryError("cannot open output parent without following links") from exc
    descriptor = -1
    try:
        opened_parent = os.fstat(parent_fd)
        if (
            opened_parent.st_dev != parent_info.st_dev
            or opened_parent.st_ino != parent_info.st_ino
        ):
            raise RuntimeInventoryError("output parent identity changed before publish")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | os.O_NOFOLLOW
        )
        try:
            descriptor = os.open(output.name, flags, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            raise RuntimeInventoryError(
                "output must be absent and creatable with O_EXCL"
            ) from exc
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise RuntimeInventoryError("short write while publishing receipt")
            written += count
        os.fsync(descriptor)
        published = os.fstat(descriptor)
        if not stat.S_ISREG(published.st_mode) or stat.S_IMODE(published.st_mode) != 0o600:
            raise RuntimeInventoryError("published receipt is not regular mode 0600")
        os.close(descriptor)
        descriptor = -1
        os.fsync(parent_fd)
        try:
            final_info = output.lstat()
        except OSError as exc:
            raise RuntimeInventoryError("published receipt disappeared") from exc
        if (
            final_info.st_dev != published.st_dev
            or final_info.st_ino != published.st_ino
            or not stat.S_ISREG(final_info.st_mode)
            or stat.S_IMODE(final_info.st_mode) != 0o600
        ):
            raise RuntimeInventoryError("published receipt path identity changed")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def mint_receipt(
    python: Path,
    isaaclab_checkout: Path,
    output: Path,
    import_roots: Sequence[Path],
) -> Mapping[str, Any]:
    receipt = build_receipt(python, isaaclab_checkout, import_roots)
    validate_receipt_document(receipt)
    output_path = _canonical_absolute_path(str(output), "--output")
    # Creating a file changes its parent directory's lstat metadata.  Refuse an
    # output parent that is itself among the identities frozen above; otherwise
    # minting would invalidate its own receipt.  A dedicated artifact directory
    # is therefore not merely convention: it is enforced by the producer.
    bound_directories = set()

    def collect_bound_directories(value: Any) -> None:
        if isinstance(value, dict):
            ancestors = value.get("ancestors")
            if isinstance(ancestors, list):
                for row in ancestors:
                    if isinstance(row, dict) and type(row.get("path")) is str:
                        bound_directories.add(row["path"])
            if (
                type(value.get("path")) is str
                and isinstance(value.get("lstat"), dict)
                and _kind_from_mode(value["lstat"].get("st_mode", 0)) == "directory"
            ):
                bound_directories.add(value["path"])
            for child in value.values():
                collect_bound_directories(child)
        elif isinstance(value, list):
            for child in value:
                collect_bound_directories(child)

    collect_bound_directories(receipt["content"])
    if str(output_path.parent) in bound_directories:
        raise RuntimeInventoryError(
            "--output parent is part of the inventoried runtime; use a dedicated external directory"
        )
    raw = _canonical_json_bytes(receipt) + b"\n"
    _exclusive_write(output_path, raw)
    read_back, read_raw = _read_receipt(output_path)
    if read_raw != raw or _canonical_json_bytes(read_back) != _canonical_json_bytes(receipt):
        raise RuntimeInventoryError("published runtime inventory failed exact readback")
    return receipt


def verify_receipt(path: Path) -> Mapping[str, Any]:
    receipt, _raw = _read_receipt(path)
    content = receipt["content"]
    python_path = Path(content["python"]["requested_path"])
    checkout_path = Path(content["isaaclab_checkout"]["path"])
    outer_execution = content["python"]["probe"]["no_site_execution"][
        "outer"
    ]
    import_roots = [
        Path(row["path"]) for row in outer_execution["import_roots"]
    ]
    live_content = build_content(
        python_path, checkout_path, import_roots
    )
    if _canonical_json_bytes(live_content) != _canonical_json_bytes(content):
        raise RuntimeInventoryError(
            "live runtime inventory differs from the frozen receipt"
        )
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mint or verify an exact action-ball runtime inventory."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    mint = subparsers.add_parser(
        "mint", help="inventory one runtime and create one no-clobber receipt"
    )
    mint.add_argument("--python", required=True, type=Path)
    mint.add_argument("--isaaclab-checkout", required=True, type=Path)
    mint.add_argument(
        "--import-root",
        required=True,
        action="append",
        type=Path,
        help=(
            "one explicit import root; repeat in the exact sys.path order "
            "(site/.pth discovery is forbidden)"
        ),
    )
    mint.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser(
        "verify", help="recompute the complete inventory from one receipt"
    )
    verify.add_argument("--receipt", required=True, type=Path)
    subparsers.add_parser(
        "_probe",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "mint":
            receipt = mint_receipt(
                args.python,
                args.isaaclab_checkout,
                args.output,
                args.import_root,
            )
            result = {
                "ok": True,
                "kind": receipt["kind"],
                "content_sha256": receipt["content_sha256"],
                "receipt_path": str(_canonical_absolute_path(str(args.output), "--output")),
                "receipt_sha256": _sha256_bytes(
                    _canonical_json_bytes(receipt) + b"\n"
                ),
            }
        elif args.command == "verify":
            receipt = verify_receipt(args.receipt)
            result = {
                "ok": True,
                "kind": receipt["kind"],
                "content_sha256": receipt["content_sha256"],
                "receipt_path": str(
                    _canonical_absolute_path(str(args.receipt), "--receipt")
                ),
                "receipt_sha256": _sha256_bytes(
                    _canonical_json_bytes(receipt) + b"\n"
                ),
            }
        elif args.command == "_probe":
            return _emit_nosite_probe()
        else:  # pragma: no cover - argparse owns the command set
            raise RuntimeInventoryError("unknown command")
    except RuntimeInventoryError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    print(_canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
