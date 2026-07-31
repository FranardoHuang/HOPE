#!/usr/bin/env python3
"""Install one live A3-vendor contract and derive its required identity.

The live schema-3 training contract is produced by the real Isaac runtime.  It
is therefore an input, not something this host-only producer may reconstruct.
This tool validates those exact bytes against the code-owned action registry
and vendor authority validator, installs them byte-for-byte at the registry's
fixed contract path, and derives the matching required-identity document from
the verified per-joint plant.

Both outputs are built and validated before publication.  Their fixed targets
are reserved with ``O_EXCL``/``O_NOFOLLOW`` and form one transaction: a
reservation, write, or fsync failure removes every output reserved by this
invocation.  No operator-selected output path or expected digest is accepted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


_THIS_FILE = Path(__file__).resolve()
_SCRIPTS_REPO_PATH = PurePosixPath(
    "hope_training/whole_body_tracking/scripts"
)
PRODUCER_REPO_PATH = (
    _SCRIPTS_REPO_PATH / "materialize_a3_vendor_required_identity.py"
).as_posix()
ACTION_REGISTRY_REPO_PATH = (
    _SCRIPTS_REPO_PATH / "a3_vendor_action_registry.py"
).as_posix()
AUTHORITY_REPO_PATH = (
    _SCRIPTS_REPO_PATH / "materialize_a3_vendor_runtime_authority.py"
).as_posix()


def _load_sibling(module_name: str, filename: str):
    path = _THIS_FILE.with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"cannot load required sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ORIGINAL_DONT_WRITE_BYTECODE = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    _REGISTRY = _load_sibling(
        "_hope_a3_vendor_required_identity_registry",
        "a3_vendor_action_registry.py",
    )
    _AUTHORITY = _load_sibling(
        "_hope_a3_vendor_required_identity_authority",
        "materialize_a3_vendor_runtime_authority.py",
    )
finally:
    sys.dont_write_bytecode = _ORIGINAL_DONT_WRITE_BYTECODE


SCHEMA_VERSION = 1
KIND = "a3_vendor_runtime_training_contract_required_identity_v1"
AUTHORITY = (
    "Agibot A3 exact deploy nominal plus vendor training settings delivered "
    "2026-07-31"
)
SOURCE_COMMIT_BINDING = "launcher_selected_clean_commit"
ACTION_SCALE_RULE = "0.25 * base_effort_limit / base_stiffness"
EXACT_DEPLOY_NOMINAL_GROUP_COUNT = 12
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class VendorRequiredIdentityError(RuntimeError):
    """Raised when exact live-contract installation cannot be proven."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_identity_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_json(raw: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise VendorRequiredIdentityError(
            f"{name} is not strict UTF-8 JSON: {exc}"
        ) from exc
    if type(value) is not dict:
        raise VendorRequiredIdentityError(f"{name} must be one JSON object")
    return value


def _git(
    repo_root: Path, arguments: Sequence[str], *, text: bool = False
) -> bytes | str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=text,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise VendorRequiredIdentityError(
            f"git {' '.join(arguments)} failed: {str(detail).strip()}"
        ) from exc


def _real_repo_root(value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise VendorRequiredIdentityError("repo root must be normalized absolute")
    requested = Path(os.path.abspath(candidate))
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise VendorRequiredIdentityError(f"cannot resolve repo root: {exc}") from exc
    if candidate != requested or requested != resolved or not resolved.is_dir():
        raise VendorRequiredIdentityError(
            "repo root must be one real directory without symlink components"
        )
    return resolved


def _resolve_full_commit(repo_root: Path, value: object) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise VendorRequiredIdentityError(
            "source commit must be one full 40-character lowercase Git commit"
        )
    resolved = str(
        _git(
            repo_root,
            ["rev-parse", "--verify", f"{value}^{{commit}}"],
            text=True,
        )
    ).strip()
    if resolved != value:
        raise VendorRequiredIdentityError(
            f"source commit resolved unexpectedly: {resolved!r} != {value!r}"
        )
    return resolved


def _require_clean_head(repo_root: Path, source_commit: str) -> None:
    head = str(
        _git(repo_root, ["rev-parse", "--verify", "HEAD^{commit}"], text=True)
    ).strip()
    if head != source_commit:
        raise VendorRequiredIdentityError(
            f"producer requires HEAD={source_commit}, got {head}"
        )
    status_bytes = bytes(
        _git(
            repo_root,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        )
    )
    if status_bytes:
        first = status_bytes.decode("utf-8", "replace").splitlines()[0]
        raise VendorRequiredIdentityError(
            f"producer requires a completely clean checkout; found {first!r}"
        )


def _git_blob(repo_root: Path, commit: str, relative: str) -> bytes:
    return bytes(_git(repo_root, ["show", f"{commit}:{relative}"]))


def _normalized_repo_path(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise VendorRequiredIdentityError(f"{name} must be repo-relative text")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise VendorRequiredIdentityError(
            f"{name} must be one normalized repo-relative POSIX path"
        )
    return value


def _tracked_source_pin(
    repo_root: Path, source_commit: str, relative_value: object, *, name: str
) -> dict[str, str]:
    relative = _normalized_repo_path(relative_value, name=name)
    requested = repo_root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise VendorRequiredIdentityError(f"cannot resolve {name}: {exc}") from exc
    if requested != resolved or resolved.is_symlink() or not resolved.is_file():
        raise VendorRequiredIdentityError(
            f"{name} must be one real regular file without symlink components"
        )
    committed = _git_blob(repo_root, source_commit, relative)
    current = resolved.read_bytes()
    if current != committed:
        raise VendorRequiredIdentityError(
            f"{name} differs between source commit and worktree"
        )
    return {"path": relative, "sha256": _sha256_bytes(committed)}


def _fixed_output_target(
    repo_root: Path, relative_value: object, *, name: str
) -> tuple[Path, Path, str]:
    relative = _normalized_repo_path(relative_value, name=name)
    requested = repo_root.joinpath(*PurePosixPath(relative).parts)
    parent_input = requested.parent
    try:
        parent = parent_input.resolve(strict=True)
    except OSError as exc:
        raise VendorRequiredIdentityError(
            f"cannot resolve {name} parent: {exc}"
        ) from exc
    if (
        parent_input != parent
        or not parent.is_dir()
        or not requested.name
        or os.path.lexists(requested)
    ):
        raise VendorRequiredIdentityError(
            f"{name} parent must be real and target must not exist"
        )
    return requested, parent, relative


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_live_file(
    value: str | Path,
    *,
    forbidden_targets: Sequence[Path],
) -> tuple[Path, bytes]:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise VendorRequiredIdentityError(
            "live training contract path must be normalized absolute"
        )
    requested = Path(os.path.abspath(candidate))
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise VendorRequiredIdentityError(
            f"cannot resolve live training contract: {exc}"
        ) from exc
    if candidate != requested or requested != resolved or resolved.is_symlink():
        raise VendorRequiredIdentityError(
            "live training contract must have no symlink or non-normal component"
        )
    if resolved in forbidden_targets:
        raise VendorRequiredIdentityError(
            "live training contract must differ from both fixed output targets"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise VendorRequiredIdentityError(
            f"cannot open live training contract: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VendorRequiredIdentityError(
                "live training contract must be one regular file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise VendorRequiredIdentityError(
                    "live training contract changed size while being read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise VendorRequiredIdentityError(
                "live training contract grew while being read"
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.lstat(resolved)
    except OSError as exc:
        raise VendorRequiredIdentityError(
            f"cannot restat live training contract: {exc}"
        ) from exc
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(path_after)
        or not stat.S_ISREG(path_after.st_mode)
    ):
        raise VendorRequiredIdentityError(
            "live training contract identity changed while being read"
        )
    return resolved, b"".join(chunks)


def _validate_pretty_live_bytes(raw: bytes) -> dict[str, object]:
    document = _strict_json(raw, name="live training contract")
    try:
        expected = (
            json.dumps(
                document,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - strict loader narrows
        raise VendorRequiredIdentityError(
            f"live training contract cannot be rendered canonically: {exc}"
        ) from exc
    if raw != expected:
        raise VendorRequiredIdentityError(
            "live training contract bytes are not the exact sorted/indented "
            "train.py representation"
        )
    return document


class _Cfg:
    def __init__(self, **kwargs: object):
        self.__dict__.update(kwargs)


class _ArticulationCfg(_Cfg):
    class InitialStateCfg(_Cfg):
        pass


def _resolve_joint_value(value: object, joint: str, *, name: str) -> float:
    if type(value) is dict:
        matches = [
            candidate
            for pattern, candidate in value.items()
            if type(pattern) is str and re.fullmatch(pattern, joint)
        ]
        if len(matches) != 1:
            raise VendorRequiredIdentityError(
                f"exact robot source {name} matched {joint!r} {len(matches)} times"
            )
        value = matches[0]
    if (
        isinstance(value, bool)
        or type(value) not in (int, float)
        or not math.isfinite(float(value))
    ):
        raise VendorRequiredIdentityError(
            f"exact robot source {name} for {joint!r} is not finite numeric"
        )
    return float(value)


def _canonical_deploy_joint_values(robot_source: bytes) -> dict[str, dict[str, float]]:
    """Execute only the reviewed actuator declaration and scale derivation.

    Importing the robot module would require Isaac Lab.  The tracked source has
    one declarative ``AGIBOT_A3_CFG`` assignment followed by a typed empty scale
    map and its derivation loop.  Selecting only those three AST nodes preserves
    the exact deploy constants without importing simulator code or duplicating
    another nominal table in this producer.
    """

    try:
        tree = ast.parse(robot_source.decode("utf-8"), filename="agibot_a3.py")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise VendorRequiredIdentityError(
            f"cannot parse exact robot actuator source: {exc}"
        ) from exc
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "AGIBOT_A3_CFG"
            for target in node.targets
        ):
            selected.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "AGIBOT_A3_ACTION_SCALE"
        ):
            selected.append(node)
        elif isinstance(node, ast.For) and any(
            isinstance(child, ast.Name)
            and child.id == "AGIBOT_A3_ACTION_SCALE"
            for child in ast.walk(node)
        ):
            selected.append(node)
    if len(selected) != 3:
        raise VendorRequiredIdentityError(
            "exact robot source must contain one actuator cfg, scale map, and "
            "scale derivation loop"
        )
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "ArticulationCfg": _ArticulationCfg,
        "ImplicitActuatorCfg": _Cfg,
        "_make_agibot_a3_spawn_cfg": lambda: _Cfg(),
    }
    try:
        exec(compile(module, "agibot_a3.py", "exec"), namespace)
        cfg = namespace["AGIBOT_A3_CFG"]
        action_scale = namespace["AGIBOT_A3_ACTION_SCALE"]
    except Exception as exc:
        raise VendorRequiredIdentityError(
            f"cannot evaluate exact robot actuator declaration: {exc}"
        ) from exc
    actuators = getattr(cfg, "actuators", None)
    if type(actuators) is not dict or type(action_scale) is not dict:
        raise VendorRequiredIdentityError(
            "exact robot actuator declaration did not produce mappings"
        )

    result: dict[str, dict[str, float]] = {}
    for joint in _AUTHORITY.RUNTIME_JOINT_NAMES:
        matching = []
        for actuator in actuators.values():
            expressions = getattr(actuator, "joint_names_expr", None)
            if type(expressions) is list and any(
                type(pattern) is str and re.fullmatch(pattern, joint)
                for pattern in expressions
            ):
                matching.append(actuator)
        if len(matching) != 1:
            raise VendorRequiredIdentityError(
                f"exact robot actuator source matched {joint!r} {len(matching)} times"
            )
        actuator = matching[0]
        result[joint] = {
            "joint_stiffness": _resolve_joint_value(
                actuator.stiffness, joint, name="stiffness"
            ),
            "joint_damping": _resolve_joint_value(
                actuator.damping, joint, name="damping"
            ),
            "joint_effort_limits": _resolve_joint_value(
                actuator.effort_limit_sim, joint, name="effort_limit_sim"
            ),
            "joint_armature": _resolve_joint_value(
                actuator.armature, joint, name="armature"
            ),
            "action_scale": _resolve_joint_value(
                action_scale, joint, name="AGIBOT_A3_ACTION_SCALE"
            ),
        }
    return result


def _group_verified_joint_values(
    verified: Mapping[str, Any],
    *,
    canonical_values: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    values = verified.get("vendor_joint_values")
    if type(values) is not dict:
        raise VendorRequiredIdentityError(
            "vendor validator did not return per-joint values"
        )
    fields = (
        "joint_stiffness",
        "joint_damping",
        "joint_effort_limits",
        "joint_armature",
        "action_scale",
    )
    groups: list[dict[str, object]] = []
    by_tuple: dict[tuple[float, ...], dict[str, object]] = {}
    for joint in _AUTHORITY.RUNTIME_JOINT_NAMES:
        row = values.get(joint)
        if type(row) is not dict or set(row) != set(fields):
            raise VendorRequiredIdentityError(
                f"vendor validator returned malformed values for {joint!r}"
            )
        canonical = canonical_values.get(joint)
        if type(canonical) is not dict or set(canonical) != set(fields):
            raise VendorRequiredIdentityError(
                f"exact robot source returned malformed values for {joint!r}"
            )
        for field in fields:
            # The authority has already checked each live value against its
            # validator-owned binary32 expectation at 1e-7.  This second,
            # intentionally wider comparison only relates that accepted
            # binary32 value back to the source literal (118.2 is serialized
            # as 118.19999694824219); it is not an alternative admission gate.
            if not math.isclose(
                float(row[field]),
                float(canonical[field]),
                rel_tol=0.0,
                abs_tol=1.0e-5,
            ):
                raise VendorRequiredIdentityError(
                    f"authority/live value {joint}.{field} does not match the "
                    "exact deploy source"
                )
        # The authority deliberately accepts binary32 serialization noise.
        # Required identity records the exact source nominal, not those live
        # round-trip residues, so one tolerated residue cannot create a 13th
        # nominal group.
        key = tuple(float(canonical[field]) for field in fields)
        group = by_tuple.get(key)
        if group is None:
            group = {
                "joints": [],
                "stiffness": key[0],
                "damping": key[1],
                "effort_limit": key[2],
                "armature": key[3],
                "action_scale": key[4],
            }
            by_tuple[key] = group
            groups.append(group)
        joints = group["joints"]
        if type(joints) is not list:  # pragma: no cover - construction invariant
            raise AssertionError("joint group was not a list")
        joints.append(joint)
    flattened = [joint for group in groups for joint in group["joints"]]
    if len(flattened) != 31 or set(flattened) != set(_AUTHORITY.RUNTIME_JOINT_NAMES):
        raise AssertionError("stable vendor grouping changed joint coverage")
    if len(groups) != EXACT_DEPLOY_NOMINAL_GROUP_COUNT:
        raise VendorRequiredIdentityError(
            "exact deploy source no longer forms the reviewed 12 nominal groups"
        )
    return groups


def _build_required_identity(
    *,
    source_commit: str,
    source_pins: Mapping[str, Mapping[str, str]],
    verified: Mapping[str, Any],
    canonical_values: Mapping[str, Mapping[str, float]],
    contract_sha256: str,
    action_id: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "materialized",
        "authority": AUTHORITY,
        "source_commit_binding": SOURCE_COMMIT_BINDING,
        "sources": {key: dict(value) for key, value in source_pins.items()},
        "robot_action_contract": {
            "runtime_dof_count": 31,
            "vendor_body_dof_count": 29,
            "legacy_head_dof_count": 2,
            "action_scale_rule": ACTION_SCALE_RULE,
            "groups": _group_verified_joint_values(
                verified, canonical_values=canonical_values
            ),
        },
        "runtime_materialization": {
            "required_training_contract_schema_version": 3,
            "training_contract_sha256": contract_sha256,
            "required_dynamic_ready_actions": [action_id],
            "required_nominal_hold_verdict": "PASS",
            "note": (
                "Materialized from the exact validated live A3 vendor contract "
                f"at source commit {source_commit}; installed contract SHA-256 "
                f"{contract_sha256}."
            ),
        },
    }


def _reserve_outputs(
    targets: Sequence[tuple[Path, Path, str]],
) -> list[tuple[int, int, Path]]:
    reserved: list[tuple[int, int, Path]] = []
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for requested, parent, _relative in targets:
            parent_fd = os.open(parent, parent_flags)
            try:
                descriptor = os.open(
                    requested.name, flags, 0o444, dir_fd=parent_fd
                )
            except BaseException:
                os.close(parent_fd)
                raise
            reserved.append((descriptor, parent_fd, requested))
            os.fchmod(descriptor, 0o444)
        return reserved
    except BaseException as exc:
        for descriptor, parent_fd, requested in reversed(reserved):
            try:
                os.close(descriptor)
            finally:
                try:
                    os.unlink(requested.name, dir_fd=parent_fd)
                    try:
                        os.fsync(parent_fd)
                    except OSError:
                        pass
                finally:
                    os.close(parent_fd)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        raise VendorRequiredIdentityError(
            f"cannot reserve all no-clobber outputs: {exc}"
        ) from exc


def _rollback_reserved(
    reserved: Sequence[tuple[int, int, Path]],
) -> None:
    for descriptor, _parent_fd, _requested in reserved:
        try:
            os.close(descriptor)
        except OSError:
            pass
    for _descriptor, parent_fd, requested in reversed(reserved):
        try:
            os.unlink(requested.name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
    for _descriptor, parent_fd, _requested in reserved:
        try:
            os.close(parent_fd)
        except OSError:
            pass


def _publish_reserved(
    reserved: Sequence[tuple[int, int, Path]], payloads: Sequence[bytes]
) -> None:
    try:
        if len(reserved) != len(payloads):  # pragma: no cover - internal invariant
            raise AssertionError("reservation/payload count mismatch")
        for (descriptor, _parent_fd, _requested), payload in zip(
            reserved, payloads
        ):
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("exclusive write made no progress")
                written += count
            os.fsync(descriptor)
        seen_parent_fds: set[int] = set()
        for _descriptor, parent_fd, _requested in reserved:
            if parent_fd not in seen_parent_fds:
                os.fsync(parent_fd)
                seen_parent_fds.add(parent_fd)
    except BaseException as exc:
        _rollback_reserved(reserved)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        raise VendorRequiredIdentityError(
            f"atomic output publication failed and was rolled back: {exc}"
        ) from exc
    else:
        for descriptor, parent_fd, _requested in reserved:
            os.close(descriptor)
            os.close(parent_fd)


def materialize_a3_vendor_required_identity(
    *,
    repo_root: str | Path,
    source_commit: str,
    action_id: str,
    live_training_contract: str | Path,
) -> dict[str, object]:
    root = _real_repo_root(repo_root)
    commit = _resolve_full_commit(root, source_commit)
    _require_clean_head(root, commit)

    try:
        config = _REGISTRY.get_action_config(action_id)
    except _REGISTRY.VendorActionRegistryError as exc:
        raise VendorRequiredIdentityError(str(exc)) from exc
    if (
        config.runtime_contract.sha256 is not None
        or config.required_identity_manifest.sha256 is not None
    ):
        raise VendorRequiredIdentityError(
            f"vendor action {config.action_id!r} already has a materialized "
            "runtime-contract/required-identity lineage"
        )

    producer_pin = _tracked_source_pin(
        root, commit, PRODUCER_REPO_PATH, name="required-identity producer"
    )
    if _THIS_FILE != root / PRODUCER_REPO_PATH:
        raise VendorRequiredIdentityError(
            "running producer is not the exact selected repo source"
        )
    registry_pin = _tracked_source_pin(
        root, commit, ACTION_REGISTRY_REPO_PATH, name="A3 vendor action registry"
    )
    if Path(_REGISTRY.__file__).resolve(strict=True) != root / registry_pin["path"]:
        raise VendorRequiredIdentityError(
            "imported action registry is not the exact selected repo source"
        )
    authority_pin = _tracked_source_pin(
        root, commit, AUTHORITY_REPO_PATH, name="vendor runtime authority validator"
    )
    if Path(_AUTHORITY.__file__).resolve(strict=True) != root / authority_pin["path"]:
        raise VendorRequiredIdentityError(
            "imported authority validator is not the exact selected repo source"
        )
    if (
        Path(_AUTHORITY._REGISTRY.__file__).resolve(strict=True)
        != root / ACTION_REGISTRY_REPO_PATH
    ):
        raise VendorRequiredIdentityError(
            "authority validator imported a different action registry source"
        )
    # Keep the local variables alive as explicit evidence that these sources
    # were checked, without embedding the producer into its own output seal.
    if not producer_pin["sha256"] or not registry_pin["sha256"]:
        raise AssertionError("tracked producer/registry pins cannot be empty")

    contract_target = _fixed_output_target(
        root, config.runtime_contract.path, name="runtime contract output"
    )
    identity_target = _fixed_output_target(
        root,
        config.required_identity_manifest.path,
        name="required identity output",
    )
    if contract_target[0] == identity_target[0]:
        raise VendorRequiredIdentityError(
            "runtime contract and required identity fixed paths must differ"
        )

    _live_path, live_bytes = _read_stable_live_file(
        live_training_contract,
        forbidden_targets=(contract_target[0], identity_target[0]),
    )
    document = _validate_pretty_live_bytes(live_bytes)
    try:
        plant = _AUTHORITY._canonical_runtime_plant_identity(document)
        verified = _AUTHORITY._verified_vendor_runtime(
            document,
            stable_motion_sha256=config.stable_motion.sha256,
            action_id=config.action_id,
        )
    except _AUTHORITY.VendorRuntimeAuthorityError as exc:
        raise VendorRequiredIdentityError(
            f"live training contract failed vendor authority validation: {exc}"
        ) from exc
    if (
        plant.get("joint_names") != list(_AUTHORITY.RUNTIME_JOINT_NAMES)
        or verified.get("action_id") != config.action_id
    ):
        raise VendorRequiredIdentityError(
            "vendor plant/action validation returned an inconsistent identity"
        )

    source_roles = {
        "robot_config": _AUTHORITY.ROBOT_SOURCE_REPO_PATH,
        "task_profile": _AUTHORITY.VENDOR_TASK_REPO_PATH,
        "training_contract_builder": (
            _AUTHORITY.TRAINING_CONTRACT_SOURCE_REPO_PATH
        ),
        "training_entrypoint": _AUTHORITY.TRAIN_SOURCE_REPO_PATH,
    }
    source_pins = {
        role: _tracked_source_pin(root, commit, relative, name=role)
        for role, relative in source_roles.items()
    }
    robot_source_bytes = (
        root / source_pins["robot_config"]["path"]
    ).read_bytes()
    canonical_values = _canonical_deploy_joint_values(robot_source_bytes)
    contract_sha = _sha256_bytes(live_bytes)
    identity = _build_required_identity(
        source_commit=commit,
        source_pins=source_pins,
        verified=verified,
        canonical_values=canonical_values,
        contract_sha256=contract_sha,
        action_id=config.action_id,
    )
    identity_bytes = _canonical_identity_bytes(identity)
    # Reopen our own bytes through the strict loader before any target exists.
    if _strict_json(identity_bytes, name="required identity") != identity:
        raise AssertionError("required identity canonical roundtrip changed data")
    identity_sha = _sha256_bytes(identity_bytes)

    reserved = _reserve_outputs((contract_target, identity_target))
    _publish_reserved(reserved, (live_bytes, identity_bytes))
    return {
        "action_id": config.action_id,
        "source_commit": commit,
        "runtime_contract": {
            "path": contract_target[2],
            "sha256": contract_sha,
        },
        "required_identity": {
            "path": identity_target[2],
            "sha256": identity_sha,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--action-id",
        required=True,
        choices=tuple(sorted(_REGISTRY.ALLOWED_ACTION_IDS)),
    )
    parser.add_argument("--live-training-contract", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = materialize_a3_vendor_required_identity(
            repo_root=args.repo_root,
            source_commit=args.source_commit,
            action_id=args.action_id,
            live_training_contract=args.live_training_contract,
        )
    except VendorRequiredIdentityError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
