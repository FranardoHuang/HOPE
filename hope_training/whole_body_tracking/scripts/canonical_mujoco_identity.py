#!/usr/bin/env python3
"""Pre-registered, fail-closed identity for one exact MuJoCo model.

This module closes the gap between a top-level MJCF hash and the compiled model
that is actually used by playback, path conversion, or dynamics checks.  The
public verifier accepts an *existing expected manifest file*.  It never offers
an API that turns the model observed at runtime into its own expected value.

The verified identity binds:

* the root MJCF bytes;
* a strict source closure containing XML includes and every supported external
  mesh, texture, hfield, or skin file;
* the complete serialized MJB bytes and size;
* the MuJoCo core library/version plus platform semantics;
* the compiled model name, dimensions, object-name order, and address tables.

The source closure is scanned both before and after compilation.  Absolute
dependency paths, path traversal outside the MJCF directory, symlink
components, include cycles, duplicate JSON keys, and unsupported ``file=``
elements fail closed.

The MJB is streamed through a temporary file and deleted.  It is never copied
into the repository.  Verification is model identity evidence only and grants
no training, deployment, or hardware authorization.  The internally held
MuJoCo model remains mutable, so it is not exposed as a public field.  The
receipt is point-in-time evidence; consumers must use
``consume_verified_model()`` and mutate only mjData.

Threat boundary: the process, imported official MuJoCo wheel, and filesystem
are trusted while verification runs.  Pre/post source scans catch persistent
drift but are not an fd-anchored snapshot against a hostile concurrent ABA
writer.  The wheel's unique core-library file is pinned, but this portable
contract does not prove its process mapping through platform-specific APIs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


IDENTITY_SCHEMA_VERSION = 1
IDENTITY_TYPE = "exact_mujoco_identity_v1"
EXPECTED_MANIFEST_TYPE = "pre_registered_exact_mujoco_identity_v1"
SOURCE_CLOSURE_TYPE = "strict_mjcf_external_source_closure_v1"
MODEL_CONTRACT_TYPE = "compiled_mujoco_name_dimension_address_contract_v1"
TOOLCHAIN_CONTRACT_TYPE = "mujoco_compiler_toolchain_contract_v1"
VERIFICATION_STATUS = "PASS_PRE_REGISTERED_EXACT_MUJOCO_IDENTITY"

SUPPORTED_EXTERNAL_FILE_TAGS = frozenset(
    {"mesh", "texture", "hfield", "skin"}
)
_TEXTURE_FILE_ATTRIBUTES = frozenset(
    {
        "file",
        "fileright",
        "fileleft",
        "fileup",
        "filedown",
        "filefront",
        "fileback",
    }
)
_EXTERNAL_FILE_ATTRIBUTES = _TEXTURE_FILE_ATTRIBUTES
_DIRECTORY_ATTRIBUTE_FOR_TAG = {
    "mesh": "meshdir",
    "texture": "texturedir",
    "hfield": "meshdir",
    "skin": "meshdir",
}
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_XML_BYTES = 16 * 1024 * 1024
_MAX_CLOSURE_MEMBERS = 4096
_MAX_CLOSURE_BYTES = 4 * 1024 * 1024 * 1024

_TOP_LEVEL_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "manifest_type",
        "identity_type",
        "root_filename",
        "expected",
        "authorization",
    }
)
_EXPECTED_KEYS = frozenset(
    {
        "root_mjcf_sha256",
        "source_closure_sha256",
        "source_member_count",
        "source_total_bytes",
        "compiled_mjb_sha256",
        "compiled_mjb_size_bytes",
        "compiler_toolchain_sha256",
        "model_contract_sha256",
        "portable_identity_sha256",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {"training_authorized", "deployment_authorized", "hardware_authorized"}
)


class ExactMujocoIdentityError(ValueError):
    """Expected identity, source closure, or compiled model did not match."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class StableFileReceipt:
    path: Path
    size_bytes: int
    sha256: str
    payload: bytes | None = None


@dataclass(frozen=True)
class SourceClosure:
    root_path: Path
    root_filename: str
    root_mjcf_sha256: str
    closure_sha256: str
    member_count: int
    total_bytes: int
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ExpectedMujocoIdentity:
    manifest_path: Path
    manifest_sha256: str
    root_filename: str
    root_mjcf_sha256: str
    source_closure_sha256: str
    source_member_count: int
    source_total_bytes: int
    compiled_mjb_sha256: str
    compiled_mjb_size_bytes: int
    compiler_toolchain_sha256: str
    model_contract_sha256: str
    portable_identity_sha256: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedMujocoIdentity:
    """Point-in-time identity evidence plus an explicitly re-checkable model."""

    portable_identity_sha256: str
    verification_receipt_sha256: str
    receipt: Mapping[str, Any]
    _model: Any = field(repr=False, compare=False)
    _mujoco_module: Any = field(repr=False, compare=False)
    _xml_model_name: str = field(repr=False, compare=False)
    _expected_compiled_mjb_sha256: str = field(
        repr=False, compare=False
    )
    _expected_compiled_mjb_size_bytes: int = field(
        repr=False, compare=False
    )
    _expected_model_contract_sha256: str = field(
        repr=False, compare=False
    )

    def assert_model_unchanged(self) -> None:
        """Fail unless the live model still matches the verified point in time."""

        unused_contract, contract_before = (
            _model_name_dimension_address_contract(
                self._mujoco_module,
                self._model,
                xml_model_name=self._xml_model_name,
            )
        )
        del unused_contract
        mjb_sha, mjb_size = _serialize_mjb(
            self._mujoco_module, self._model
        )
        unused_contract, contract_after = (
            _model_name_dimension_address_contract(
                self._mujoco_module,
                self._model,
                xml_model_name=self._xml_model_name,
            )
        )
        del unused_contract
        if contract_before != contract_after:
            raise ExactMujocoIdentityError(
                "live MuJoCo model changed while it was being re-checked"
            )
        mismatches: dict[str, dict[str, Any]] = {}
        checks = {
            "compiled_mjb_sha256": (
                self._expected_compiled_mjb_sha256,
                mjb_sha,
            ),
            "compiled_mjb_size_bytes": (
                self._expected_compiled_mjb_size_bytes,
                mjb_size,
            ),
            "model_contract_sha256": (
                self._expected_model_contract_sha256,
                contract_after,
            ),
        }
        for key, (expected, actual) in checks.items():
            if expected != actual:
                mismatches[key] = {
                    "expected": expected,
                    "actual": actual,
                }
        if mismatches:
            raise ExactMujocoIdentityError(
                "live MuJoCo model changed after exact identity verification",
                report=_deep_freeze(
                    {
                        "status": "FAIL_LIVE_MUJOCO_MODEL_DRIFT",
                        "mismatches": mismatches,
                        "training_authorized": False,
                        "deployment_authorized": False,
                        "hardware_authorized": False,
                    }
                ),
            )

    def consume_verified_model(self, consumer: Any) -> Any:
        """Run one trusted consumer between exact pre/post model checks.

        The callback must not retain the model reference and must mutate only
        mjData.  A post-check catches changes made during the callback.
        """

        if not callable(consumer):
            raise TypeError("consumer must be callable")
        self.assert_model_unchanged()
        try:
            return consumer(self._model)
        finally:
            self.assert_model_unchanged()


@dataclass(frozen=True)
class _CurrentIdentity:
    model: Any
    source_closure: SourceClosure
    compiled_mjb_sha256: str
    compiled_mjb_size_bytes: int
    compiler_toolchain: Mapping[str, Any]
    compiler_toolchain_sha256: str
    model_contract: Mapping[str, Any]
    model_contract_sha256: str
    runtime_provenance: Mapping[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _mutable_json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _mutable_json_copy(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_mutable_json_copy(item) for item in value]
    return value


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ExactMujocoIdentityError(
            f"{label} must be exactly 64 lowercase SHA-256 hex digits"
        )
    return value


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactMujocoIdentityError(
            f"{label} must be a non-negative integer"
        )
    return int(value)


def _require_exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactMujocoIdentityError(f"{label} must be a JSON object")
    actual = frozenset(value)
    if actual != expected or any(not isinstance(key, str) for key in value):
        raise ExactMujocoIdentityError(
            f"{label} keys changed: expected={sorted(expected)}, "
            f"actual={sorted(actual)}"
        )
    return value


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExactMujocoIdentityError(
                f"expected manifest contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _normalized_absolute(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _strict_real_directory(path: os.PathLike[str] | str, label: str) -> Path:
    absolute = _normalized_absolute(path)
    try:
        resolved = absolute.resolve(strict=True)
        row = os.lstat(absolute)
    except OSError as exc:
        raise ExactMujocoIdentityError(f"cannot resolve {label}: {exc}") from exc
    if absolute != resolved or stat.S_ISLNK(row.st_mode) or not resolved.is_dir():
        raise ExactMujocoIdentityError(
            f"{label} must be a real directory with no symlink component: "
            f"{absolute}"
        )
    return absolute


def _inside_root(path: Path, allowed_root: Path, label: str) -> None:
    try:
        common = Path(os.path.commonpath((str(path), str(allowed_root))))
    except ValueError as exc:
        raise ExactMujocoIdentityError(
            f"{label} does not share the MJCF source root"
        ) from exc
    if common != allowed_root:
        raise ExactMujocoIdentityError(
            f"{label} escapes the MJCF source root: {path}"
        )


def _strict_real_file(
    path: os.PathLike[str] | str,
    *,
    allowed_root: Path,
    label: str,
) -> Path:
    absolute = _normalized_absolute(path)
    _inside_root(absolute, allowed_root, label)
    try:
        resolved = absolute.resolve(strict=True)
        row = os.lstat(absolute)
    except OSError as exc:
        raise ExactMujocoIdentityError(f"cannot resolve {label}: {exc}") from exc
    if absolute != resolved or stat.S_ISLNK(row.st_mode):
        raise ExactMujocoIdentityError(
            f"{label} must have no symlink component: {absolute}"
        )
    _inside_root(resolved, allowed_root, label)
    if not stat.S_ISREG(row.st_mode):
        raise ExactMujocoIdentityError(
            f"{label} must be a regular file: {absolute}"
        )
    return absolute


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _read_stable_file(
    path: os.PathLike[str] | str,
    *,
    allowed_root: Path,
    label: str,
    retain_payload: bool,
    maximum_payload_bytes: int | None = None,
) -> StableFileReceipt:
    checked = _strict_real_file(path, allowed_root=allowed_root, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(checked, flags)
    except OSError as exc:
        raise ExactMujocoIdentityError(f"cannot open {label}: {exc}") from exc
    payload_parts: list[bytes] | None = [] if retain_payload else None
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExactMujocoIdentityError(
                f"{label} changed away from a regular file"
            )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if (
                maximum_payload_bytes is not None
                and total > maximum_payload_bytes
            ):
                raise ExactMujocoIdentityError(
                    f"{label} exceeds {maximum_payload_bytes} bytes"
                )
            digest.update(block)
            if payload_parts is not None:
                payload_parts.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.stat(checked, follow_symlinks=False)
    except OSError as exc:
        raise ExactMujocoIdentityError(
            f"{label} disappeared after reading: {exc}"
        ) from exc
    if (
        not _same_file_state(before, after)
        or not _same_file_state(after, path_after)
        or total != after.st_size
    ):
        raise ExactMujocoIdentityError(
            f"{label} changed while it was being hashed"
        )
    _strict_real_file(checked, allowed_root=allowed_root, label=label)
    return StableFileReceipt(
        path=checked,
        size_bytes=total,
        sha256=digest.hexdigest(),
        payload=(
            None if payload_parts is None else b"".join(payload_parts)
        ),
    )


def _path_label(path: Path, source_root: Path) -> str:
    try:
        label = path.relative_to(source_root).as_posix()
    except ValueError as exc:  # pragma: no cover - guarded by _inside_root
        raise ExactMujocoIdentityError(
            f"closure member escapes source root: {path}"
        ) from exc
    if not label or label == "." or label.startswith("../"):
        raise ExactMujocoIdentityError(
            f"invalid closure-relative path {label!r}"
        )
    return label


def _safe_relative_reference(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
        raise ExactMujocoIdentityError(
            f"{label} must be one non-empty trimmed relative path"
        )
    if "\x00" in raw or "\\" in raw:
        raise ExactMujocoIdentityError(
            f"{label} contains a NUL or platform-ambiguous backslash"
        )
    value = Path(raw)
    if value.is_absolute() or value.anchor:
        raise ExactMujocoIdentityError(f"{label} must not be absolute")
    return raw


def _local_tag(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _parse_xml(receipt: StableFileReceipt, label: str) -> ET.Element:
    payload = receipt.payload
    if payload is None:  # pragma: no cover - construction invariant
        raise ExactMujocoIdentityError("XML payload was not retained")
    upper = payload.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ExactMujocoIdentityError(
            f"{label} contains a forbidden DTD/entity declaration"
        )
    try:
        return ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ExactMujocoIdentityError(
            f"cannot parse {label}: {exc}"
        ) from exc


def scan_mjcf_source_closure(
    mjcf_path: os.PathLike[str] | str,
) -> SourceClosure:
    """Return the strict root/include/external-file source closure."""

    root_absolute = _normalized_absolute(mjcf_path)
    source_root = _strict_real_directory(
        root_absolute.parent, "MJCF source directory"
    )
    root_path = _strict_real_file(
        root_absolute, allowed_root=source_root, label="root MJCF"
    )
    xml_rows: dict[Path, tuple[StableFileReceipt, ET.Element]] = {}
    include_edges: list[dict[str, str]] = []
    visiting: set[Path] = set()

    def walk_xml(path: Path) -> None:
        checked = _strict_real_file(
            path, allowed_root=source_root, label="MJCF XML"
        )
        if checked in visiting:
            raise ExactMujocoIdentityError(
                f"MJCF include cycle detected at {_path_label(checked, source_root)}"
            )
        if checked in xml_rows:
            return
        if len(xml_rows) >= _MAX_CLOSURE_MEMBERS:
            raise ExactMujocoIdentityError(
                "MJCF closure exceeds the member limit"
            )
        visiting.add(checked)
        file_receipt = _read_stable_file(
            checked,
            allowed_root=source_root,
            label="MJCF XML",
            retain_payload=True,
            maximum_payload_bytes=_MAX_XML_BYTES,
        )
        xml_root = _parse_xml(
            file_receipt, f"MJCF XML {_path_label(checked, source_root)}"
        )
        xml_rows[checked] = (file_receipt, xml_root)
        for element in xml_root.iter():
            if _local_tag(element) != "include":
                continue
            raw = _safe_relative_reference(
                element.attrib.get("file"),
                "MJCF include file",
            )
            # MuJoCo resolves every include, including a nested include,
            # relative to the directory of the main MJCF.
            child = _normalized_absolute(source_root / raw)
            child = _strict_real_file(
                child, allowed_root=source_root, label="MJCF include"
            )
            if child in xml_rows and child not in visiting:
                raise ExactMujocoIdentityError(
                    "MuJoCo permits each included XML file at most once: "
                    f"{_path_label(child, source_root)}"
                )
            include_edges.append(
                {
                    "declaring_xml": _path_label(checked, source_root),
                    "raw_file": raw,
                    "resolved_path": _path_label(child, source_root),
                }
            )
            walk_xml(child)
        visiting.remove(checked)

    walk_xml(root_path)
    root_xml = xml_rows[root_path][1]
    if _local_tag(root_xml) != "mujoco":
        raise ExactMujocoIdentityError(
            f"root MJCF element must be 'mujoco', got {_local_tag(root_xml)!r}"
        )
    model_name = root_xml.attrib.get("model")
    if (
        not isinstance(model_name, str)
        or not model_name
        or model_name.strip() != model_name
    ):
        raise ExactMujocoIdentityError(
            "root MJCF must declare one non-empty trimmed model name"
        )

    root_compilers = [
        element
        for element in root_xml.iter()
        if _local_tag(element) == "compiler"
    ]
    included_compilers = [
        (_path_label(path, source_root), element)
        for path, (_, xml_root) in xml_rows.items()
        if path != root_path
        for element in xml_root.iter()
        if _local_tag(element) == "compiler"
    ]
    if len(root_compilers) > 1 or included_compilers:
        raise ExactMujocoIdentityError(
            "strict closure supports at most one compiler element in the root "
            "MJCF and none in included XML"
        )
    compiler = root_compilers[0] if root_compilers else None
    raw_directories = {
        key: (
            ""
            if compiler is None
            else str(compiler.attrib.get(key, ""))
        )
        for key in ("assetdir", "meshdir", "texturedir")
    }
    for key, raw in raw_directories.items():
        if raw:
            _safe_relative_reference(raw, f"compiler {key}")
    raw_strippath = (
        "false"
        if compiler is None
        else str(compiler.attrib.get("strippath", "false"))
    )
    if raw_strippath not in {"true", "false"}:
        raise ExactMujocoIdentityError(
            "compiler strippath must be exactly 'true' or 'false'"
        )
    strip_path = raw_strippath == "true"
    effective_directories = {
        "assetdir": raw_directories["assetdir"],
        "meshdir": (
            raw_directories["meshdir"] or raw_directories["assetdir"]
        ),
        "texturedir": (
            raw_directories["texturedir"] or raw_directories["assetdir"]
        ),
    }
    for key, raw in effective_directories.items():
        directory = _normalized_absolute(source_root / raw)
        _inside_root(directory, source_root, f"compiler {key}")
        if raw:
            _strict_real_directory(directory, f"compiler {key}")

    member_kinds: dict[Path, set[str]] = {
        path: {"xml_root" if path == root_path else "xml_include"}
        for path in xml_rows
    }
    external_references: list[dict[str, str]] = []
    for declaring_path, (_, xml_root) in xml_rows.items():
        for element in xml_root.iter():
            file_attributes = sorted(
                (
                    attribute,
                    raw_value,
                )
                for attribute, raw_value in element.attrib.items()
                if attribute in _EXTERNAL_FILE_ATTRIBUTES
            )
            if not file_attributes:
                continue
            tag = _local_tag(element)
            if tag == "include":
                unexpected = [
                    attribute
                    for attribute, unused_value in file_attributes
                    if attribute != "file"
                ]
                if unexpected:
                    raise ExactMujocoIdentityError(
                        "include has unsupported external-file attributes: "
                        f"{unexpected}"
                    )
                continue
            if tag not in SUPPORTED_EXTERNAL_FILE_TAGS:
                raise ExactMujocoIdentityError(
                    f"unsupported file-bearing MJCF element {tag!r} in "
                    f"{_path_label(declaring_path, source_root)}"
                )
            allowed_attributes = (
                _TEXTURE_FILE_ATTRIBUTES
                if tag == "texture"
                else frozenset({"file"})
            )
            unexpected = [
                attribute
                for attribute, unused_value in file_attributes
                if attribute not in allowed_attributes
            ]
            if unexpected:
                raise ExactMujocoIdentityError(
                    f"{tag} has unsupported external-file attributes: "
                    f"{unexpected}"
                )
            for attribute, raw_file in file_attributes:
                raw = _safe_relative_reference(
                    raw_file, f"{tag} {attribute} attribute"
                )
                effective_file = Path(raw).name if strip_path else raw
                directory_key = _DIRECTORY_ATTRIBUTE_FOR_TAG[tag]
                prefix = effective_directories[directory_key]
                dependency = _normalized_absolute(
                    source_root / prefix / effective_file
                )
                dependency = _strict_real_file(
                    dependency,
                    allowed_root=source_root,
                    label=f"MJCF {tag} {attribute} dependency",
                )
                member_kinds.setdefault(dependency, set()).add(
                    f"{tag}:{attribute}"
                )
                external_references.append(
                    {
                        "attribute": attribute,
                        "declaring_xml": _path_label(
                            declaring_path, source_root
                        ),
                        "element": tag,
                        "raw_file": raw,
                        "effective_file": effective_file,
                        "resolved_path": _path_label(
                            dependency, source_root
                        ),
                    }
                )

    if len(member_kinds) > _MAX_CLOSURE_MEMBERS:
        raise ExactMujocoIdentityError(
            "MJCF closure exceeds the member limit"
        )
    members: list[dict[str, Any]] = []
    total_bytes = 0
    root_sha: str | None = None
    for path in sorted(member_kinds, key=lambda value: _path_label(value, source_root)):
        if path in xml_rows:
            file_receipt = xml_rows[path][0]
        else:
            file_receipt = _read_stable_file(
                path,
                allowed_root=source_root,
                label="MJCF external dependency",
                retain_payload=False,
            )
        total_bytes += file_receipt.size_bytes
        if total_bytes > _MAX_CLOSURE_BYTES:
            raise ExactMujocoIdentityError(
                "MJCF closure exceeds the byte limit"
            )
        if path == root_path:
            root_sha = file_receipt.sha256
        members.append(
            {
                "kinds": sorted(member_kinds[path]),
                "path": _path_label(path, source_root),
                "size_bytes": file_receipt.size_bytes,
                "sha256": file_receipt.sha256,
            }
        )
    if root_sha is None:  # pragma: no cover - construction invariant
        raise ExactMujocoIdentityError("root MJCF left its own closure")
    receipt = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "closure_type": SOURCE_CLOSURE_TYPE,
        "root_filename": _path_label(root_path, source_root),
        "xml_model_name": model_name,
        "compiler_directories": {
            "declared": raw_directories,
            "effective": effective_directories,
            "strippath": strip_path,
        },
        "members": members,
        "include_edges": sorted(
            include_edges,
            key=lambda row: (
                row["declaring_xml"],
                row["resolved_path"],
                row["raw_file"],
            ),
        ),
        "external_references": sorted(
            external_references,
            key=lambda row: (
                row["declaring_xml"],
                row["element"],
                row["attribute"],
                row["resolved_path"],
                row["raw_file"],
                row["effective_file"],
            ),
        ),
        "member_count": len(members),
        "total_bytes": total_bytes,
    }
    closure_sha = _digest_bytes(_canonical_json_bytes(receipt))
    return SourceClosure(
        root_path=root_path,
        root_filename=_path_label(root_path, source_root),
        root_mjcf_sha256=root_sha,
        closure_sha256=closure_sha,
        member_count=len(members),
        total_bytes=total_bytes,
        receipt=_deep_freeze(receipt),
    )


def _find_core_mujoco_library(mujoco_module: Any) -> StableFileReceipt:
    module_file_raw = getattr(mujoco_module, "__file__", None)
    if not isinstance(module_file_raw, str) or not module_file_raw:
        raise ExactMujocoIdentityError(
            "MuJoCo module does not expose a package file path"
        )
    module_file = _normalized_absolute(module_file_raw)
    package_root = _strict_real_directory(
        module_file.parent, "MuJoCo package directory"
    )
    _strict_real_file(
        module_file,
        allowed_root=package_root,
        label="MuJoCo package __init__",
    )
    candidates = sorted(
        path
        for path in package_root.iterdir()
        if path.is_file()
        and (
            path.name.startswith("libmujoco.so")
            or path.name.startswith("libmujoco.")
            and path.suffix == ".dylib"
            or path.name.lower() in {"mujoco.dll", "libmujoco.dll"}
        )
    )
    if len(candidates) != 1:
        raise ExactMujocoIdentityError(
            "MuJoCo package must expose exactly one core shared library; "
            f"found {[path.name for path in candidates]}"
        )
    return _read_stable_file(
        candidates[0],
        allowed_root=package_root,
        label="MuJoCo core shared library",
        retain_payload=False,
    )


def _compiler_toolchain_contract(
    mujoco_module: Any, model: Any
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    py_version = getattr(mujoco_module, "__version__", None)
    if not isinstance(py_version, str) or not py_version:
        raise ExactMujocoIdentityError(
            "MuJoCo module lacks a non-empty __version__"
        )
    try:
        version_integer = int(mujoco_module.mj_version())
        version_string = str(mujoco_module.mj_versionString())
    except Exception as exc:
        raise ExactMujocoIdentityError(
            f"cannot query MuJoCo core version: {exc}"
        ) from exc
    if (
        version_integer <= 0
        or not version_string
        or version_string != py_version
    ):
        raise ExactMujocoIdentityError(
            "MuJoCo Python/core versions disagree: "
            f"python={py_version!r}, core={version_string!r}/{version_integer}"
        )
    active_plugins = int(getattr(model, "nplugin", 0))
    if active_plugins != 0:
        raise ExactMujocoIdentityError(
            "active MuJoCo plugins require a versioned plugin-library resolver; "
            f"current strict contract supports nplugin=0, got {active_plugins}"
        )
    core = _find_core_mujoco_library(mujoco_module)
    libc_name, libc_version = platform.libc_ver()
    contract = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "contract_type": TOOLCHAIN_CONTRACT_TYPE,
        "mujoco_python_version": py_version,
        "mj_version": version_integer,
        "mj_version_string": version_string,
        "core_library": {
            "filename": core.path.name,
            "size_bytes": core.size_bytes,
            "sha256": core.sha256,
            "resolution": "unique_file_in_imported_mujoco_package",
            "process_mapping_proven": False,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "libc": [libc_name, libc_version],
        },
        "active_model_plugins": [],
    }
    digest = _digest_bytes(_canonical_json_bytes(contract))
    runtime = {
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "numpy_version": str(np.__version__),
        "kernel_release": platform.release(),
        "python_cache_tag": str(sys.implementation.cache_tag),
    }
    return contract, digest, runtime


def _object_names(
    mujoco_module: Any,
    model: Any,
    *,
    count: int,
    object_type: Any,
    label: str,
) -> list[str | None]:
    names: list[str | None] = []
    seen: set[str] = set()
    for index in range(count):
        raw = mujoco_module.mj_id2name(model, object_type, index)
        if raw is None:
            names.append(None)
            continue
        name = str(raw)
        if not name:
            raise ExactMujocoIdentityError(
                f"compiled {label} name at index {index} is empty"
            )
        if name in seen:
            raise ExactMujocoIdentityError(
                f"compiled model has duplicate named {label} {name!r}"
            )
        seen.add(name)
        names.append(name)
    return names


def _integer_array(
    model: Any, name: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    if not hasattr(model, name):
        raise ExactMujocoIdentityError(
            f"compiled model lacks required address array {name}"
        )
    raw = np.asarray(getattr(model, name))
    if raw.shape != expected_shape or raw.dtype.kind not in "iu":
        raise ExactMujocoIdentityError(
            f"compiled model {name} must be integer {expected_shape}, "
            f"got {raw.dtype}/{raw.shape}"
        )
    return np.ascontiguousarray(raw, dtype=np.int64)


def _model_name_dimension_address_contract(
    mujoco_module: Any,
    model: Any,
    *,
    xml_model_name: str,
) -> tuple[dict[str, Any], str]:
    dimension_names = (
        "nq",
        "nv",
        "njnt",
        "nbody",
        "ngeom",
        "nsite",
        "nu",
        "nmesh",
        "nplugin",
    )
    dimensions: dict[str, int] = {}
    for name in dimension_names:
        if not hasattr(model, name):
            raise ExactMujocoIdentityError(
                f"compiled model lacks dimension {name}"
            )
        value = int(getattr(model, name))
        if value < 0:
            raise ExactMujocoIdentityError(
                f"compiled model dimension {name} is negative"
            )
        dimensions[name] = value
    if dimensions["nq"] < 1 or dimensions["nv"] < 1:
        raise ExactMujocoIdentityError(
            "compiled model must have positive nq/nv"
        )
    jnt_type = _integer_array(
        model, "jnt_type", (dimensions["njnt"],)
    )
    jnt_bodyid = _integer_array(
        model, "jnt_bodyid", (dimensions["njnt"],)
    )
    jnt_qposadr = _integer_array(
        model, "jnt_qposadr", (dimensions["njnt"],)
    )
    jnt_dofadr = _integer_array(
        model, "jnt_dofadr", (dimensions["njnt"],)
    )
    body_parentid = _integer_array(
        model, "body_parentid", (dimensions["nbody"],)
    )
    site_bodyid = _integer_array(
        model, "site_bodyid", (dimensions["nsite"],)
    )
    actuator_trnid = _integer_array(
        model, "actuator_trnid", (dimensions["nu"], 2)
    )
    if np.any(jnt_qposadr < 0) or np.any(
        jnt_qposadr >= dimensions["nq"]
    ):
        raise ExactMujocoIdentityError(
            "compiled joint qpos address leaves nq"
        )
    if np.any(jnt_dofadr < 0) or np.any(
        jnt_dofadr >= dimensions["nv"]
    ):
        raise ExactMujocoIdentityError(
            "compiled joint dof address leaves nv"
        )
    free_type = int(mujoco_module.mjtJoint.mjJNT_FREE)
    free_ids = np.flatnonzero(jnt_type == free_type)
    if len(free_ids) != 1:
        raise ExactMujocoIdentityError(
            f"compiled model must have exactly one free root, got "
            f"{free_ids.tolist()}"
        )
    free_id = int(free_ids[0])
    if (
        int(jnt_qposadr[free_id]) != 0
        or int(jnt_dofadr[free_id]) != 0
    ):
        raise ExactMujocoIdentityError(
            "compiled free root must start at qpos/dof address zero"
        )

    object_names = {
        "body": _object_names(
            mujoco_module,
            model,
            count=dimensions["nbody"],
            object_type=mujoco_module.mjtObj.mjOBJ_BODY,
            label="body",
        ),
        "joint": _object_names(
            mujoco_module,
            model,
            count=dimensions["njnt"],
            object_type=mujoco_module.mjtObj.mjOBJ_JOINT,
            label="joint",
        ),
        "geom": _object_names(
            mujoco_module,
            model,
            count=dimensions["ngeom"],
            object_type=mujoco_module.mjtObj.mjOBJ_GEOM,
            label="geom",
        ),
        "site": _object_names(
            mujoco_module,
            model,
            count=dimensions["nsite"],
            object_type=mujoco_module.mjtObj.mjOBJ_SITE,
            label="site",
        ),
        "actuator": _object_names(
            mujoco_module,
            model,
            count=dimensions["nu"],
            object_type=mujoco_module.mjtObj.mjOBJ_ACTUATOR,
            label="actuator",
        ),
    }
    contract = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "contract_type": MODEL_CONTRACT_TYPE,
        "xml_model_name": xml_model_name,
        "dimensions": dimensions,
        "object_names": object_names,
        "addresses": {
            "jnt_type": jnt_type.tolist(),
            "jnt_bodyid": jnt_bodyid.tolist(),
            "jnt_qposadr": jnt_qposadr.tolist(),
            "jnt_dofadr": jnt_dofadr.tolist(),
            "body_parentid": body_parentid.tolist(),
            "site_bodyid": site_bodyid.tolist(),
            "actuator_trnid": actuator_trnid.tolist(),
            "free_joint_id": free_id,
        },
    }
    return contract, _digest_bytes(_canonical_json_bytes(contract))


def _serialize_mjb(
    mujoco_module: Any, model: Any
) -> tuple[str, int]:
    # macOS commonly exposes /var as an alias of /private/var.  That alias is
    # harmless for this private scratch area and must not weaken the strict
    # no-symlink policy used for MJCF sources or expected manifests.  Resolve
    # the operating-system temp root first, then create a mode-0700 staging
    # directory whose contents cannot be swapped by another user.
    temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix="exact-mujoco-identity-",
            dir=str(temp_root),
        )
    )
    stage = _strict_real_directory(stage, "temporary MJB directory")
    path = stage / "compiled-model.mjb"
    try:
        saved = False
        for arguments in (
            (model, str(path), None, 0),
            (model, str(path)),
        ):
            try:
                mujoco_module.mj_saveModel(*arguments)
                saved = True
                break
            except TypeError:
                continue
        if not saved:
            raise ExactMujocoIdentityError(
                "MuJoCo binding exposes no supported mj_saveModel signature"
            )
        receipt = _read_stable_file(
            path,
            allowed_root=stage,
            label="compiled temporary MJB",
            retain_payload=False,
        )
        if receipt.size_bytes <= 0:
            raise ExactMujocoIdentityError(
                "compiled MJB is empty"
            )
        return receipt.sha256, receipt.size_bytes
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        try:
            stage.rmdir()
        except FileNotFoundError:
            pass


def _compute_current_identity(
    mujoco_module: Any,
    mjcf_path: os.PathLike[str] | str,
) -> _CurrentIdentity:
    before = scan_mjcf_source_closure(mjcf_path)
    try:
        model = mujoco_module.MjModel.from_xml_path(str(before.root_path))
    except Exception as exc:
        raise ExactMujocoIdentityError(
            f"MuJoCo cannot compile the pinned source closure: {exc}"
        ) from exc
    after = scan_mjcf_source_closure(mjcf_path)
    if (
        before.closure_sha256 != after.closure_sha256
        or before.receipt != after.receipt
    ):
        raise ExactMujocoIdentityError(
            "MJCF source closure changed while MuJoCo compiled it"
        )
    toolchain, toolchain_sha, runtime = _compiler_toolchain_contract(
        mujoco_module, model
    )
    xml_model_name = str(before.receipt["xml_model_name"])
    model_contract, model_contract_sha = (
        _model_name_dimension_address_contract(
            mujoco_module,
            model,
            xml_model_name=xml_model_name,
        )
    )
    mjb_sha, mjb_size = _serialize_mjb(mujoco_module, model)
    model_contract_after, model_contract_sha_after = (
        _model_name_dimension_address_contract(
            mujoco_module,
            model,
            xml_model_name=xml_model_name,
        )
    )
    if (
        model_contract_sha != model_contract_sha_after
        or model_contract != model_contract_after
    ):
        raise ExactMujocoIdentityError(
            "compiled MuJoCo model changed while its MJB was serialized"
        )
    return _CurrentIdentity(
        model=model,
        source_closure=before,
        compiled_mjb_sha256=mjb_sha,
        compiled_mjb_size_bytes=mjb_size,
        compiler_toolchain=_deep_freeze(toolchain),
        compiler_toolchain_sha256=toolchain_sha,
        model_contract=_deep_freeze(model_contract),
        model_contract_sha256=model_contract_sha,
        runtime_provenance=_deep_freeze(runtime),
    )


def _portable_identity_payload(
    *,
    root_filename: str,
    root_mjcf_sha256: str,
    source_closure_sha256: str,
    source_member_count: int,
    source_total_bytes: int,
    compiled_mjb_sha256: str,
    compiled_mjb_size_bytes: int,
    compiler_toolchain_sha256: str,
    model_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "identity_type": IDENTITY_TYPE,
        "root_filename": root_filename,
        "root_mjcf_sha256": root_mjcf_sha256,
        "source_closure_sha256": source_closure_sha256,
        "source_member_count": source_member_count,
        "source_total_bytes": source_total_bytes,
        "compiled_mjb_sha256": compiled_mjb_sha256,
        "compiled_mjb_size_bytes": compiled_mjb_size_bytes,
        "compiler_toolchain_sha256": compiler_toolchain_sha256,
        "model_contract_sha256": model_contract_sha256,
    }


def load_expected_identity_manifest(
    path: os.PathLike[str] | str,
    *,
    trusted_manifest_sha256: str,
) -> ExpectedMujocoIdentity:
    """Load an externally pinned expected manifest, never create one."""

    trusted_manifest_sha256 = _require_digest(
        trusted_manifest_sha256, "trusted_manifest_sha256"
    )
    manifest_absolute = _normalized_absolute(path)
    manifest_parent = _strict_real_directory(
        manifest_absolute.parent, "expected-manifest directory"
    )
    receipt = _read_stable_file(
        manifest_absolute,
        allowed_root=manifest_parent,
        label="pre-registered expected manifest",
        retain_payload=True,
        maximum_payload_bytes=_MAX_MANIFEST_BYTES,
    )
    if receipt.sha256 != trusted_manifest_sha256:
        raise ExactMujocoIdentityError(
            "expected manifest does not match its external trust anchor: "
            f"expected={trusted_manifest_sha256}, actual={receipt.sha256}"
        )
    payload = receipt.payload
    if payload is None:  # pragma: no cover - construction invariant
        raise ExactMujocoIdentityError("manifest payload was not retained")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExactMujocoIdentityError(
                    f"expected manifest contains non-finite JSON {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactMujocoIdentityError(
            f"cannot parse expected manifest JSON: {exc}"
        ) from exc
    top = _require_exact_keys(
        decoded, _TOP_LEVEL_MANIFEST_KEYS, "expected manifest"
    )
    if (
        top["schema_version"] != IDENTITY_SCHEMA_VERSION
        or isinstance(top["schema_version"], bool)
    ):
        raise ExactMujocoIdentityError(
            f"expected manifest schema_version must be "
            f"{IDENTITY_SCHEMA_VERSION}"
        )
    if top["manifest_type"] != EXPECTED_MANIFEST_TYPE:
        raise ExactMujocoIdentityError(
            f"expected manifest_type must be {EXPECTED_MANIFEST_TYPE!r}"
        )
    if top["identity_type"] != IDENTITY_TYPE:
        raise ExactMujocoIdentityError(
            f"expected identity_type must be {IDENTITY_TYPE!r}"
        )
    root_filename = top["root_filename"]
    if (
        not isinstance(root_filename, str)
        or not root_filename
        or root_filename != Path(root_filename).name
        or "\\" in root_filename
    ):
        raise ExactMujocoIdentityError(
            "expected root_filename must be one basename"
        )
    authorization = _require_exact_keys(
        top["authorization"], _AUTHORIZATION_KEYS, "authorization"
    )
    if any(value is not False for value in authorization.values()):
        raise ExactMujocoIdentityError(
            "identity manifest cannot authorize training, deployment, or hardware"
        )
    expected = _require_exact_keys(
        top["expected"], _EXPECTED_KEYS, "expected values"
    )
    root_sha = _require_digest(
        expected["root_mjcf_sha256"], "root_mjcf_sha256"
    )
    closure_sha = _require_digest(
        expected["source_closure_sha256"], "source_closure_sha256"
    )
    member_count = _require_nonnegative_integer(
        expected["source_member_count"], "source_member_count"
    )
    total_bytes = _require_nonnegative_integer(
        expected["source_total_bytes"], "source_total_bytes"
    )
    mjb_sha = _require_digest(
        expected["compiled_mjb_sha256"], "compiled_mjb_sha256"
    )
    mjb_size = _require_nonnegative_integer(
        expected["compiled_mjb_size_bytes"],
        "compiled_mjb_size_bytes",
    )
    toolchain_sha = _require_digest(
        expected["compiler_toolchain_sha256"],
        "compiler_toolchain_sha256",
    )
    model_sha = _require_digest(
        expected["model_contract_sha256"], "model_contract_sha256"
    )
    portable_sha = _require_digest(
        expected["portable_identity_sha256"],
        "portable_identity_sha256",
    )
    if member_count < 1 or total_bytes < 1 or mjb_size < 1:
        raise ExactMujocoIdentityError(
            "expected source/MJB counts and sizes must be positive"
        )
    portable = _portable_identity_payload(
        root_filename=root_filename,
        root_mjcf_sha256=root_sha,
        source_closure_sha256=closure_sha,
        source_member_count=member_count,
        source_total_bytes=total_bytes,
        compiled_mjb_sha256=mjb_sha,
        compiled_mjb_size_bytes=mjb_size,
        compiler_toolchain_sha256=toolchain_sha,
        model_contract_sha256=model_sha,
    )
    recomputed = _digest_bytes(_canonical_json_bytes(portable))
    if recomputed != portable_sha:
        raise ExactMujocoIdentityError(
            "portable_identity_sha256 does not match expected component pins"
        )
    return ExpectedMujocoIdentity(
        manifest_path=receipt.path,
        manifest_sha256=receipt.sha256,
        root_filename=root_filename,
        root_mjcf_sha256=root_sha,
        source_closure_sha256=closure_sha,
        source_member_count=member_count,
        source_total_bytes=total_bytes,
        compiled_mjb_sha256=mjb_sha,
        compiled_mjb_size_bytes=mjb_size,
        compiler_toolchain_sha256=toolchain_sha,
        model_contract_sha256=model_sha,
        portable_identity_sha256=portable_sha,
        raw=_deep_freeze(top),
    )


def _mismatch_report(
    expected: ExpectedMujocoIdentity,
    current: _CurrentIdentity,
) -> dict[str, Any]:
    actual = {
        "root_filename": current.source_closure.root_filename,
        "root_mjcf_sha256": current.source_closure.root_mjcf_sha256,
        "source_closure_sha256": current.source_closure.closure_sha256,
        "source_member_count": current.source_closure.member_count,
        "source_total_bytes": current.source_closure.total_bytes,
        "compiled_mjb_sha256": current.compiled_mjb_sha256,
        "compiled_mjb_size_bytes": current.compiled_mjb_size_bytes,
        "compiler_toolchain_sha256": current.compiler_toolchain_sha256,
        "model_contract_sha256": current.model_contract_sha256,
    }
    wanted = {
        key: getattr(expected, key)
        for key in actual
    }
    return {
        "status": "FAIL_PRE_REGISTERED_IDENTITY_MISMATCH",
        "mismatches": {
            key: {"expected": wanted[key], "actual": actual[key]}
            for key in actual
            if wanted[key] != actual[key]
        },
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }


def _verify_exact_mujoco_identity_with_module(
    *,
    mjcf_path: os.PathLike[str] | str,
    expected_manifest_path: os.PathLike[str] | str,
    trusted_expected_manifest_sha256: str,
    mujoco_module: Any,
) -> VerifiedMujocoIdentity:
    """Private dependency-injected verifier used by pure fixture tests."""

    expected = load_expected_identity_manifest(
        expected_manifest_path,
        trusted_manifest_sha256=trusted_expected_manifest_sha256,
    )
    if Path(mjcf_path).name != expected.root_filename:
        raise ExactMujocoIdentityError(
            f"MJCF basename {Path(mjcf_path).name!r} does not match "
            f"pre-registered {expected.root_filename!r}"
        )
    current = _compute_current_identity(mujoco_module, mjcf_path)
    mismatch = _mismatch_report(expected, current)
    if mismatch["mismatches"]:
        raise ExactMujocoIdentityError(
            "runtime model does not match the pre-registered exact identity",
            report=_deep_freeze(mismatch),
        )
    portable = _portable_identity_payload(
        root_filename=current.source_closure.root_filename,
        root_mjcf_sha256=current.source_closure.root_mjcf_sha256,
        source_closure_sha256=current.source_closure.closure_sha256,
        source_member_count=current.source_closure.member_count,
        source_total_bytes=current.source_closure.total_bytes,
        compiled_mjb_sha256=current.compiled_mjb_sha256,
        compiled_mjb_size_bytes=current.compiled_mjb_size_bytes,
        compiler_toolchain_sha256=current.compiler_toolchain_sha256,
        model_contract_sha256=current.model_contract_sha256,
    )
    portable_sha = _digest_bytes(_canonical_json_bytes(portable))
    if portable_sha != expected.portable_identity_sha256:
        raise ExactMujocoIdentityError(
            "verified components do not reproduce the pre-registered "
            "portable identity"
        )
    receipt_without_digest = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "identity_type": IDENTITY_TYPE,
        "status": VERIFICATION_STATUS,
        "portable_identity": portable,
        "portable_identity_sha256": portable_sha,
        "expected_manifest": {
            "filename": expected.manifest_path.name,
            "sha256": expected.manifest_sha256,
        },
        "source_closure": _mutable_json_copy(
            current.source_closure.receipt
        ),
        "compiler_toolchain": _mutable_json_copy(
            current.compiler_toolchain
        ),
        "model_contract": _mutable_json_copy(current.model_contract),
        "runtime_provenance": _mutable_json_copy(
            current.runtime_provenance
        ),
        "compiled_mjb": {
            "sha256": current.compiled_mjb_sha256,
            "size_bytes": current.compiled_mjb_size_bytes,
            "stored_in_repository": False,
        },
        "claims": {
            "pre_registered_expected_values_used": True,
            "runtime_values_promoted_to_expected": False,
            "external_manifest_sha256_matched": True,
            "source_closure_checked_before_and_after_compile": True,
            "model_contract_checked_before_and_after_mjb": True,
            "complete_compiled_mjb_bound": True,
            "live_model_identity_is_point_in_time": True,
            "live_model_exposed_directly": False,
            "live_model_recheck_method": "assert_model_unchanged",
            "live_model_consumer_method": "consume_verified_model",
            "trusted_process_and_filesystem_required": True,
            "hostile_concurrent_source_writer_resistant": False,
            "core_library_process_mapping_proven": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
    }
    receipt_sha = _digest_bytes(
        _canonical_json_bytes(receipt_without_digest)
    )
    receipt = dict(receipt_without_digest)
    receipt["verification_receipt_sha256"] = receipt_sha
    return VerifiedMujocoIdentity(
        portable_identity_sha256=portable_sha,
        verification_receipt_sha256=receipt_sha,
        receipt=_deep_freeze(receipt),
        _model=current.model,
        _mujoco_module=mujoco_module,
        _xml_model_name=str(current.source_closure.receipt["xml_model_name"]),
        _expected_compiled_mjb_sha256=current.compiled_mjb_sha256,
        _expected_compiled_mjb_size_bytes=current.compiled_mjb_size_bytes,
        _expected_model_contract_sha256=current.model_contract_sha256,
    )


def verify_exact_mujoco_identity(
    *,
    mjcf_path: os.PathLike[str] | str,
    expected_manifest_path: os.PathLike[str] | str,
    trusted_expected_manifest_sha256: str,
) -> VerifiedMujocoIdentity:
    """Compile with the installed MuJoCo and verify an externally pinned manifest."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ExactMujocoIdentityError(
            "MuJoCo Python is required for exact identity verification"
        ) from exc
    return _verify_exact_mujoco_identity_with_module(
        mjcf_path=mjcf_path,
        expected_manifest_path=expected_manifest_path,
        trusted_expected_manifest_sha256=(
            trusted_expected_manifest_sha256
        ),
        mujoco_module=mujoco,
    )


__all__ = [
    "EXPECTED_MANIFEST_TYPE",
    "ExactMujocoIdentityError",
    "ExpectedMujocoIdentity",
    "IDENTITY_SCHEMA_VERSION",
    "IDENTITY_TYPE",
    "SOURCE_CLOSURE_TYPE",
    "SourceClosure",
    "VerifiedMujocoIdentity",
    "load_expected_identity_manifest",
    "scan_mjcf_source_closure",
    "verify_exact_mujoco_identity",
]
