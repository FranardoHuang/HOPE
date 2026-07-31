#!/usr/bin/env python3
"""Repin one code-authorized vendor identity-bootstrap manifest without a bundle.

This producer exists only to break the Stage-A identity materialization cycle.
It does not build or bless a dynamic-ready candidate, contact admission, or a
training bundle.  Instead it reopens one already reviewed stable-v2 N=1
manifest/prototype and one formal profile-pins blob from the same exact Git
commit, then permits only the provenance/profile fields enumerated below to
change.

All three outputs are fully constructed and validated before any byte is
written.  Their final paths are reserved with ``O_EXCL`` before the first
write, so a spent target causes a zero-byte rollback and no partial sibling
publication.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

_ORIGINAL_DONT_WRITE_BYTECODE = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import a3_vendor_action_registry as _ACTION_REGISTRY
finally:
    sys.dont_write_bytecode = _ORIGINAL_DONT_WRITE_BYTECODE


SCHEMA_VERSION = 1
KIND = "agibot_a3_vendor_identity_manifest_repin_receipt_v1"
PURPOSE = "identity_bootstrap_repin"
ACTION_ID = _ACTION_REGISTRY.DEFAULT_ACTION_ID
MOBILITY_MODE = "no_move"
_DEFAULT_ACTION = _ACTION_REGISTRY.get_action_config(ACTION_ID)
STABLE_MOTION_PATH = _DEFAULT_ACTION.stable_motion.path
STABLE_MOTION_SHA256 = _DEFAULT_ACTION.stable_motion.sha256
STABLE_SOURCE_MANIFEST_PATH = _DEFAULT_ACTION.stable_source_manifest.path
STABLE_SOURCE_MANIFEST_SHA256 = _DEFAULT_ACTION.stable_source_manifest.sha256
STABLE_SOURCE_PROTOTYPE_PATH = _DEFAULT_ACTION.stable_source_prototype.path
STABLE_SOURCE_PROTOTYPE_SHA256 = _DEFAULT_ACTION.stable_source_prototype.sha256
SCRIPTS_RELATIVE = PurePosixPath(
    "hope_training/whole_body_tracking/scripts"
)
ACTION_REGISTRY_RELATIVE = (
    SCRIPTS_RELATIVE / "a3_vendor_action_registry.py"
).as_posix()
PRODUCER_RELATIVE = (
    SCRIPTS_RELATIVE / "materialize_a3_vendor_identity_manifest.py"
).as_posix()
PROFILE_PRODUCER_RELATIVE = (
    SCRIPTS_RELATIVE / "pin_action_ball_profile_contracts.py"
).as_posix()
MDP_RELATIVE = PurePosixPath(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
MANIFEST_MODULE_RELATIVE = (
    MDP_RELATIVE / "action_ball_manifest.py"
).as_posix()
SOLVER_SOURCE_NAMES = (
    "hope_commands.py",
    "continuous_questions.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "virtual_ball.py",
    "counter_rally.py",
    "counter_rally_torch.py",
)
PROFILE_TOP_LEVEL_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "source_authority",
        "cfg",
        "geometry",
        "venue_yaml",
        "venue_yaml_sha256",
        "planes",
        "solver_implementation_source_sha256",
        "contact_geometry",
        "counter_rally",
        "physics_profile_sha256",
        "solver_profile_sha256",
        "physics_payload",
        "solver_payload",
    )
)
PROFILE_KIND = "whole_body_tracking.action_ball.profile_pins"
PROFILE_AUTHORITY = {
    "schema_version": 1,
    "authority": "external_exact_commit_subset_blob_map_v1",
    "commit_binding": "external_preexec_immutable_launch_capsule_v1",
    "embedded_commit": False,
}
PROTOTYPE_ALLOWED_CHANGES = (
    "provenance.producer",
    "provenance.producer_source_sha256",
    "provenance.profile_pins",
)
MANIFEST_ALLOWED_CHANGES = (
    "manifest_id",
    "solver_profile_sha256",
    "physics_profile_sha256",
    "prototype",
    "notes",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class IdentityManifestRepinError(RuntimeError):
    """Raised when identity-bootstrap repinning cannot be proven exact."""


def _action_config(action_id: object) -> _ACTION_REGISTRY.VendorActionConfig:
    try:
        return _ACTION_REGISTRY.get_action_config(action_id)
    except _ACTION_REGISTRY.VendorActionRegistryError as exc:
        raise IdentityManifestRepinError(str(exc)) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
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


def _canonical_ascii_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256_bytes(raw)


def _strict_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_json_bytes(raw: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IdentityManifestRepinError(f"{name} is not strict UTF-8 JSON: {exc}") from exc
    if type(value) is not dict:
        raise IdentityManifestRepinError(f"{name} must be one JSON object")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise IdentityManifestRepinError(f"{name} must be one lowercase SHA-256")
    return value


def _git(repo_root: Path, arguments: Sequence[str], *, text: bool = False) -> bytes | str:
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
        raise IdentityManifestRepinError(
            f"git {' '.join(arguments)} failed: {str(detail).strip()}"
        ) from exc


def _resolve_commit(repo_root: Path, value: str) -> str:
    result = str(
        _git(repo_root, ["rev-parse", "--verify", f"{value}^{{commit}}"], text=True)
    ).strip()
    if _COMMIT_RE.fullmatch(result) is None:
        raise IdentityManifestRepinError(
            f"source commit did not resolve to one full commit: {result!r}"
        )
    return result


def _require_clean_checkout(repo_root: Path) -> None:
    status = bytes(
        _git(
            repo_root,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        )
    )
    if status:
        first = status.decode("utf-8", "replace").splitlines()[0]
        raise IdentityManifestRepinError(
            f"producer requires a completely clean checkout; found {first!r}"
        )


def _git_blob(repo_root: Path, commit: str, relative: str) -> bytes:
    return bytes(_git(repo_root, ["show", f"{commit}:{relative}"]))


def _repo_file(
    repo_root: Path, value: str | Path, *, name: str
) -> tuple[Path, str]:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    requested = Path(os.path.abspath(candidate))
    try:
        resolved = requested.resolve(strict=True)
        relative = resolved.relative_to(repo_root).as_posix()
    except (OSError, ValueError) as exc:
        raise IdentityManifestRepinError(
            f"{name} must be one real regular file inside repo root: {exc}"
        ) from exc
    pure = PurePosixPath(relative)
    if (
        requested != resolved
        or resolved.is_symlink()
        or not resolved.is_file()
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise IdentityManifestRepinError(
            f"{name} must have no symlink or non-normal path component"
        )
    return resolved, relative


def _tracked_exact_blob(
    repo_root: Path,
    source_commit: str,
    value: str | Path,
    expected_sha256: object,
    *,
    name: str,
) -> tuple[Path, str, bytes, str]:
    path, relative = _repo_file(repo_root, value, name=name)
    expected = _require_sha256(expected_sha256, name=f"expected {name} SHA-256")
    raw = path.read_bytes()
    committed = _git_blob(repo_root, source_commit, relative)
    worktree_sha = _sha256_bytes(raw)
    commit_sha = _sha256_bytes(committed)
    if raw != committed or worktree_sha != expected or commit_sha != expected:
        raise IdentityManifestRepinError(
            f"{name} differs across expected/commit/worktree: "
            f"expected={expected}, commit={commit_sha}, worktree={worktree_sha}"
        )
    return path, relative, raw, expected


def _tracked_current_source(
    repo_root: Path, source_commit: str, relative: str, *, name: str
) -> tuple[Path, str]:
    path, normalized = _repo_file(repo_root, relative, name=name)
    if normalized != relative:
        raise IdentityManifestRepinError(f"{name} path is not canonical")
    raw = path.read_bytes()
    committed = _git_blob(repo_root, source_commit, relative)
    if raw != committed:
        raise IdentityManifestRepinError(
            f"{name} differs between source commit and worktree"
        )
    return path, _sha256_bytes(raw)


def _load_manifest_module(repo_root: Path, source_commit: str):
    path, _sha = _tracked_current_source(
        repo_root,
        source_commit,
        MANIFEST_MODULE_RELATIVE,
        name="ActionBall manifest contract",
    )
    module_name = "_a3_vendor_identity_manifest_contract"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise IdentityManifestRepinError("cannot load ActionBall manifest contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise IdentityManifestRepinError(
            f"cannot execute ActionBall manifest contract: {exc}"
        ) from exc
    return module


def _validate_profile_pins(
    *,
    repo_root: Path,
    source_commit: str,
    path_value: str | Path,
    expected_sha256: object,
) -> tuple[dict[str, object], dict[str, str]]:
    _path, relative, raw, actual_sha = _tracked_exact_blob(
        repo_root,
        source_commit,
        path_value,
        expected_sha256,
        name="profile pins",
    )
    profile = _strict_json_bytes(raw, name="profile pins")
    if set(profile) != PROFILE_TOP_LEVEL_KEYS:
        raise IdentityManifestRepinError(
            "profile pins top-level keys differ from the formal producer contract"
        )
    if profile.get("schema_version") != 1 or profile.get("kind") != PROFILE_KIND:
        raise IdentityManifestRepinError("profile pins schema/kind mismatch")

    profile_producer, _profile_producer_sha = _tracked_current_source(
        repo_root,
        source_commit,
        PROFILE_PRODUCER_RELATIVE,
        name="formal profile-pins producer",
    )
    try:
        reproduced = subprocess.run(
            [
                sys.executable,
                str(profile_producer),
                "--repo-root",
                str(repo_root),
                "--source-rev",
                source_commit,
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise IdentityManifestRepinError(
            "formal profile-pins reproduction failed: "
            f"{str(detail).strip()}"
        ) from exc
    if reproduced != raw:
        raise IdentityManifestRepinError(
            "tracked profile pins are not the exact formal pinner output for "
            "the selected source commit"
        )

    physics_payload = profile.get("physics_payload")
    solver_payload = profile.get("solver_payload")
    if type(physics_payload) is not dict or type(solver_payload) is not dict:
        raise IdentityManifestRepinError(
            "profile physics_payload and solver_payload must be objects"
        )
    physics_sha = _require_sha256(
        profile.get("physics_profile_sha256"), name="physics profile SHA-256"
    )
    solver_sha = _require_sha256(
        profile.get("solver_profile_sha256"), name="solver profile SHA-256"
    )
    if _canonical_ascii_sha256(physics_payload) != physics_sha:
        raise IdentityManifestRepinError(
            "physics profile SHA does not seal its canonical payload"
        )
    if _canonical_ascii_sha256(solver_payload) != solver_sha:
        raise IdentityManifestRepinError(
            "solver profile SHA does not seal its canonical payload"
        )
    if solver_payload.get("physics_profile_sha256") != physics_sha:
        raise IdentityManifestRepinError(
            "solver payload does not reference the exact physics profile"
        )

    source_map: dict[str, str] = {}
    for filename in SOLVER_SOURCE_NAMES:
        relative_source = (MDP_RELATIVE / filename).as_posix()
        _source_path, source_sha = _tracked_current_source(
            repo_root,
            source_commit,
            relative_source,
            name=f"solver source {filename}",
        )
        source_map[filename] = source_sha
    if profile.get("solver_implementation_source_sha256") != source_map:
        raise IdentityManifestRepinError(
            "profile solver source map differs from the exact source commit"
        )
    if solver_payload.get("implementation_source_sha256") != source_map:
        raise IdentityManifestRepinError(
            "solver payload source map differs from the exact source commit"
        )
    authority = dict(PROFILE_AUTHORITY)
    authority["source_blob_map_sha256"] = _canonical_ascii_sha256(source_map)
    if profile.get("source_authority") != authority:
        raise IdentityManifestRepinError(
            "profile source_authority does not seal the exact seven-source map"
        )

    geometry = profile.get("contact_geometry")
    if type(geometry) is not dict or set(geometry) != {"payload", "sha256"}:
        raise IdentityManifestRepinError("profile contact_geometry is malformed")
    geometry_sha = _require_sha256(
        geometry.get("sha256"), name="contact geometry payload SHA-256"
    )
    if _canonical_ascii_sha256(geometry.get("payload")) != geometry_sha:
        raise IdentityManifestRepinError(
            "contact geometry SHA does not seal its canonical payload"
        )
    if solver_payload.get("contact_geometry") != geometry:
        raise IdentityManifestRepinError(
            "solver payload contact geometry differs from profile pins"
        )

    venue_relative = profile.get("venue_yaml")
    venue_sha = _require_sha256(
        profile.get("venue_yaml_sha256"), name="venue YAML SHA-256"
    )
    if type(venue_relative) is not str:
        raise IdentityManifestRepinError("profile venue_yaml must be repo-relative")
    _venue_path, normalized_venue, _venue_raw, _venue_actual = (
        _tracked_exact_blob(
            repo_root,
            source_commit,
            venue_relative,
            venue_sha,
            name="venue physics YAML",
        )
    )
    venue_source = physics_payload.get("venue_source")
    if venue_source != {
        "path": normalized_venue,
        "file_sha256": venue_sha,
    }:
        raise IdentityManifestRepinError(
            "physics payload venue source differs from the exact source-commit blob"
        )

    counter_rally = profile.get("counter_rally")
    solver_counter = solver_payload.get("counter_rally")
    if type(counter_rally) is not dict or type(solver_counter) is not dict:
        raise IdentityManifestRepinError("counter-rally profile binding is missing")
    objective_sha = _require_sha256(
        counter_rally.get("objective_profile_sha256"),
        name="counter-rally objective SHA-256",
    )
    counter_venue_sha = _require_sha256(
        counter_rally.get("venue_physics_sha256"),
        name="counter-rally venue physics SHA-256",
    )
    if _canonical_ascii_sha256(counter_rally.get("objective_profile")) != objective_sha:
        raise IdentityManifestRepinError(
            "counter-rally objective SHA does not seal its payload"
        )
    if (
        _canonical_ascii_sha256(counter_rally.get("venue_physics"))
        != counter_venue_sha
    ):
        raise IdentityManifestRepinError(
            "counter-rally venue SHA does not seal its payload"
        )
    if (
        solver_counter.get("objective_profile_sha256") != objective_sha
        or solver_counter.get("venue_physics_sha256") != counter_venue_sha
    ):
        raise IdentityManifestRepinError(
            "solver payload counter-rally binding differs from profile pins"
        )

    return profile, {
        "path": relative,
        "sha256": actual_sha,
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "geometry_payload_sha256": geometry_sha,
    }


def _validate_source_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    path_value: str | Path,
    expected_sha256: object,
    manifest_module,
    action_config: _ACTION_REGISTRY.VendorActionConfig,
) -> tuple[dict[str, object], dict[str, str], dict[str, object], dict[str, str]]:
    path, relative, raw, actual_sha = _tracked_exact_blob(
        repo_root,
        source_commit,
        path_value,
        expected_sha256,
        name="source manifest",
    )
    stable_manifest = _ACTION_REGISTRY.stable_pin(
        action_config.stable_source_manifest
    )
    stable_prototype = _ACTION_REGISTRY.stable_pin(
        action_config.stable_source_prototype
    )
    stable_motion = _ACTION_REGISTRY.stable_pin(action_config.stable_motion)
    if (
        relative != stable_manifest["path"]
        or actual_sha != stable_manifest["sha256"]
    ):
        raise IdentityManifestRepinError(
            "source manifest is not the code-owned stable-v2 manifest blob "
            f"for action {action_config.action_id!r}"
        )
    source_document = _strict_json_bytes(raw, name="source manifest")
    try:
        loaded = manifest_module.load_action_ball_manifest(
            path,
            expected_sha256=actual_sha,
            verify_referenced_assets=True,
            repo_root=repo_root,
        )
    except Exception as exc:
        raise IdentityManifestRepinError(
            f"source manifest failed the formal strict loader: {exc}"
        ) from exc
    manifest = loaded.manifest
    if (
        manifest.schema_version != 3
        or manifest.mobility_mode != MOBILITY_MODE
        or tuple(manifest.action_order) != (action_config.action_id,)
        or len(manifest.actions) != 1
        or manifest.actions[0].action_id != action_config.action_id
        or manifest.prototype.scope != action_config.scope
        or manifest.actions[0].motion_path != stable_motion["path"]
        or manifest.actions[0].motion_sha256 != stable_motion["sha256"]
    ):
        raise IdentityManifestRepinError(
            "source manifest is not exact schema-3 "
            f"{action_config.action_id}/{MOBILITY_MODE}/stable-v2"
        )
    source_mapping = manifest.to_mapping()
    if _canonical_bytes(source_mapping) != raw:
        raise IdentityManifestRepinError(
            "source manifest is not canonical ActionBall JSON plus newline"
        )

    prototype_binding = source_mapping.get("prototype")
    if type(prototype_binding) is not dict:
        raise IdentityManifestRepinError("source manifest prototype binding is missing")
    if prototype_binding != {
        "path": stable_prototype["path"],
        "sha256": stable_prototype["sha256"],
        "scope": action_config.scope,
    }:
        raise IdentityManifestRepinError(
            "source manifest does not bind the code-owned stable-v2 prototype"
        )
    prototype_path, prototype_relative, prototype_raw, prototype_sha = (
        _tracked_exact_blob(
            repo_root,
            source_commit,
            prototype_binding.get("path"),
            prototype_binding.get("sha256"),
            name="source prototype",
        )
    )
    prototype_document = _strict_json_bytes(
        prototype_raw, name="source prototype"
    )
    motion = source_mapping["actions"][0]
    _tracked_exact_blob(
        repo_root,
        source_commit,
        motion["motion_path"],
        motion["motion_sha256"],
        name="stable-v2 motion",
    )
    return (
        source_mapping,
        {"path": relative, "sha256": actual_sha},
        prototype_document,
        {"path": prototype_relative, "sha256": prototype_sha},
    )


def _without_prototype_allowed(document: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(document))
    provenance = result.get("provenance")
    if type(provenance) is not dict:
        raise IdentityManifestRepinError("prototype provenance must be an object")
    for key in ("producer", "producer_source_sha256", "profile_pins"):
        provenance.pop(key, None)
    return result


def _without_manifest_allowed(document: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(document))
    for key in MANIFEST_ALLOWED_CHANGES:
        result.pop(key, None)
    return result


def _output_target(
    repo_root: Path, value: str | Path, *, name: str
) -> tuple[Path, Path, str]:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    requested = Path(os.path.abspath(candidate))
    parent_input = requested.parent
    try:
        parent = parent_input.resolve(strict=True)
        relative = requested.relative_to(repo_root).as_posix()
    except (OSError, ValueError) as exc:
        raise IdentityManifestRepinError(
            f"{name} must be one leaf inside a real repo directory: {exc}"
        ) from exc
    if (
        parent_input != parent
        or not parent.is_dir()
        or not requested.name
        or os.path.lexists(requested)
    ):
        raise IdentityManifestRepinError(
            f"{name} parent must be real and target must not exist"
        )
    return requested, parent, relative


def _reserve_outputs(
    targets: Sequence[tuple[Path, Path, str]],
) -> list[tuple[int, int, Path]]:
    reserved: list[tuple[int, int, Path]] = []
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        for requested, parent, _relative in targets:
            parent_fd = os.open(parent, parent_flags)
            try:
                fd = os.open(requested.name, flags, 0o444, dir_fd=parent_fd)
            except Exception:
                os.close(parent_fd)
                raise
            reserved.append((fd, parent_fd, requested))
        return reserved
    except Exception as exc:
        for fd, parent_fd, requested in reversed(reserved):
            try:
                os.close(fd)
            finally:
                try:
                    os.unlink(requested.name, dir_fd=parent_fd)
                finally:
                    os.close(parent_fd)
        raise IdentityManifestRepinError(
            f"cannot reserve all no-clobber outputs: {exc}"
        ) from exc


def _publish_reserved(
    reserved: Sequence[tuple[int, int, Path]], payloads: Sequence[bytes]
) -> None:
    try:
        for (fd, _parent_fd, _requested), payload in zip(reserved, payloads):
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(fd, view[written:])
                if count <= 0:
                    raise OSError("exclusive write made no progress")
                written += count
            os.fsync(fd)
        for _fd, parent_fd, _requested in reserved:
            os.fsync(parent_fd)
    except OSError as exc:
        raise IdentityManifestRepinError(
            f"reserved output publication failed: {exc}"
        ) from exc
    finally:
        for fd, parent_fd, _requested in reserved:
            try:
                os.close(fd)
            finally:
                os.close(parent_fd)


def materialize_a3_vendor_identity_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    source_manifest: str | Path,
    expected_source_manifest_sha256: str,
    profile_pins: str | Path,
    expected_profile_pins_sha256: str,
    prototype_output: str | Path,
    manifest_output: str | Path,
    receipt_output: str | Path,
    action_id: str = ACTION_ID,
) -> dict[str, object]:
    action_config = _action_config(action_id)
    if action_config.identity_repin_producer.path != PRODUCER_RELATIVE:
        raise IdentityManifestRepinError(
            f"action {action_config.action_id!r} plans identity output from "
            "a different producer path"
        )
    root = Path(repo_root).resolve(strict=True)
    commit = _resolve_commit(root, source_commit)
    head = _resolve_commit(root, "HEAD")
    if head != commit:
        raise IdentityManifestRepinError(
            f"producer requires HEAD={commit}, got {head}"
        )
    _require_clean_checkout(root)
    action_registry_path, _action_registry_file_sha = _tracked_current_source(
        root,
        commit,
        ACTION_REGISTRY_RELATIVE,
        name="A3 vendor action registry",
    )
    if Path(_ACTION_REGISTRY.__file__).resolve(strict=True) != action_registry_path:
        raise IdentityManifestRepinError(
            "imported A3 vendor action registry is not the exact selected repo source"
        )
    action_registry_pin = dict(
        _ACTION_REGISTRY.action_source_registry_pin(action_config)
    )
    producer_path, producer_sha = _tracked_current_source(
        root, commit, PRODUCER_RELATIVE, name="identity manifest producer"
    )
    if Path(__file__).resolve(strict=True) != producer_path:
        raise IdentityManifestRepinError(
            "running producer is not the exact selected repo source"
        )
    manifest_module = _load_manifest_module(root, commit)
    profile, profile_pin = _validate_profile_pins(
        repo_root=root,
        source_commit=commit,
        path_value=profile_pins,
        expected_sha256=expected_profile_pins_sha256,
    )
    (
        source_mapping,
        source_manifest_pin,
        source_prototype,
        source_prototype_pin,
    ) = _validate_source_manifest(
        repo_root=root,
        source_commit=commit,
        path_value=source_manifest,
        expected_sha256=expected_source_manifest_sha256,
        manifest_module=manifest_module,
        action_config=action_config,
    )
    objective_sha = _canonical_ascii_sha256(
        source_mapping.get("counter_rally_objective")
    )
    if (
        profile.get("counter_rally", {}).get("objective_profile_sha256")
        != objective_sha
    ):
        raise IdentityManifestRepinError(
            "source manifest counter-rally objective differs from formal profile pins"
        )
    source_velocity_contract = source_prototype.get("velocity_contract")
    if (
        type(source_velocity_contract) is not dict
        or source_velocity_contract.get("geometry_source_sha256")
        != profile.get("contact_geometry", {}).get("sha256")
        or source_prototype.get("provenance", {}).get(
            "geometry_source_file_sha256"
        )
        != profile.get("solver_implementation_source_sha256", {}).get(
            "racket_contact_geometry.py"
        )
    ):
        raise IdentityManifestRepinError(
            "source prototype geometry source differs from formal profile pins"
        )

    prototype_target = _output_target(root, prototype_output, name="prototype output")
    manifest_target = _output_target(root, manifest_output, name="manifest output")
    receipt_target = _output_target(root, receipt_output, name="receipt output")
    target_paths = [prototype_target[0], manifest_target[0], receipt_target[0]]
    if len(set(target_paths)) != 3:
        raise IdentityManifestRepinError("prototype/manifest/receipt outputs must differ")

    new_prototype = deepcopy(source_prototype)
    provenance = new_prototype.get("provenance")
    if type(provenance) is not dict:
        raise IdentityManifestRepinError("source prototype provenance is malformed")
    provenance["profile_pins"] = dict(profile_pin)
    provenance["producer"] = Path(PRODUCER_RELATIVE).name
    provenance["producer_source_sha256"] = producer_sha
    if _without_prototype_allowed(new_prototype) != _without_prototype_allowed(
        source_prototype
    ):
        raise AssertionError("prototype changed outside its explicit allowlist")
    if (
        new_prototype.get("scopes") != source_prototype.get("scopes")
        or new_prototype.get("derived_sha256")
        != source_prototype.get("derived_sha256")
        or provenance.get("profile_pins", {}).get("solver_profile_sha256")
        != profile_pin["solver_profile_sha256"]
    ):
        raise IdentityManifestRepinError(
            "prototype scopes/derived hash/profile provenance did not remain exact"
        )
    prototype_bytes = _canonical_bytes(new_prototype)
    prototype_sha = _sha256_bytes(prototype_bytes)

    new_manifest = deepcopy(source_mapping)
    old_manifest_id = new_manifest.get("manifest_id")
    if type(old_manifest_id) is not str:
        raise IdentityManifestRepinError("source manifest_id is malformed")
    new_manifest["manifest_id"] = (
        f"{old_manifest_id}__a3_vendor_identity_repin_"
        f"{profile_pin['solver_profile_sha256'][:12]}"
    )
    new_manifest["solver_profile_sha256"] = profile_pin[
        "solver_profile_sha256"
    ]
    new_manifest["physics_profile_sha256"] = profile_pin[
        "physics_profile_sha256"
    ]
    new_manifest["prototype"] = {
        "path": prototype_target[2],
        "sha256": prototype_sha,
        "scope": "upper",
    }
    source_notes = new_manifest.get("notes")
    if type(source_notes) is not str:
        raise IdentityManifestRepinError("source manifest notes are malformed")
    new_manifest["notes"] = (
        f"{source_notes.rstrip()} Identity-bootstrap-only profile repin from "
        f"source manifest SHA-256 {source_manifest_pin['sha256']} and source "
        f"prototype SHA-256 {source_prototype_pin['sha256']} to formal profile "
        f"pins SHA-256 {profile_pin['sha256']} at source commit {commit}. "
        "Action UID, motion, ball profile, counter-rally objective, and prototype "
        "scope geometry are unchanged. This is not a formal bundle, contact "
        "admission, deployment, or hardware authorization."
    )
    if _without_manifest_allowed(new_manifest) != _without_manifest_allowed(
        source_mapping
    ):
        raise AssertionError("manifest changed outside its explicit allowlist")
    try:
        validated_manifest = manifest_module.ActionBallManifest.from_mapping(
            new_manifest
        )
        manifest_bytes = manifest_module.canonical_manifest_bytes(
            validated_manifest
        )
    except Exception as exc:
        raise IdentityManifestRepinError(
            f"repinned manifest failed strict ActionBall roundtrip: {exc}"
        ) from exc
    roundtrip_mapping = validated_manifest.to_mapping()
    if roundtrip_mapping != new_manifest:
        raise IdentityManifestRepinError(
            "repinned manifest strict roundtrip changed its mapping"
        )
    manifest_sha = _sha256_bytes(manifest_bytes)

    source_action = source_mapping["actions"][0]
    new_action = new_manifest["actions"][0]
    invariants = {
        "action_order_unchanged": new_manifest["action_order"]
        == source_mapping["action_order"],
        "action_uid_unchanged": new_action["action_uid"]
        == source_action["action_uid"],
        "motion_binding_unchanged": (
            new_action["motion_path"], new_action["motion_sha256"]
        )
        == (source_action["motion_path"], source_action["motion_sha256"]),
        "ball_profile_unchanged": new_action["ball_profile"]
        == source_action["ball_profile"],
        "counter_rally_objective_unchanged": new_manifest.get(
            "counter_rally_objective"
        )
        == source_mapping.get("counter_rally_objective"),
        "prototype_scopes_geometry_unchanged": new_prototype.get("scopes")
        == source_prototype.get("scopes"),
        "prototype_motion_provenance_unchanged": provenance.get("motion")
        == source_prototype.get("provenance", {}).get("motion"),
        "prototype_source_manifest_provenance_unchanged": provenance.get(
            "source_manifest"
        )
        == source_prototype.get("provenance", {}).get("source_manifest"),
        "only_allowlisted_fields_changed": True,
    }
    if not all(value is True for value in invariants.values()):
        failed = sorted(key for key, value in invariants.items() if value is not True)
        raise IdentityManifestRepinError(
            f"identity repin invariant failed: {failed}"
        )

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "purpose": PURPOSE,
        "source_commit": commit,
        "inputs": {
            "source_manifest": source_manifest_pin,
            "source_prototype": source_prototype_pin,
            "profile_pins": profile_pin,
            "producer": {
                "path": PRODUCER_RELATIVE,
                "sha256": producer_sha,
            },
            "action_registry": action_registry_pin,
        },
        "outputs": {
            "prototype": {
                "path": prototype_target[2],
                "sha256": prototype_sha,
            },
            "manifest": {
                "path": manifest_target[2],
                "sha256": manifest_sha,
            },
        },
        "allowed_changes": {
            "prototype": list(PROTOTYPE_ALLOWED_CHANGES),
            "manifest": list(MANIFEST_ALLOWED_CHANGES),
        },
        "invariants": invariants,
        "authorization": {
            "identity_bootstrap_repin": True,
            "formal_bundle": False,
            "contact_admission": False,
            "dynamic_ready": False,
            "training": False,
            "deployment": False,
            "hardware": False,
        },
    }
    receipt_bytes = _canonical_bytes(receipt)
    receipt_sha = _sha256_bytes(receipt_bytes)

    reserved = _reserve_outputs(
        (prototype_target, manifest_target, receipt_target)
    )
    _publish_reserved(
        reserved, (prototype_bytes, manifest_bytes, receipt_bytes)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "action_id": action_config.action_id,
        "source_commit": commit,
        "action_registry_source_identity_sha256": action_registry_pin[
            "source_identity_sha256"
        ],
        "producer": {
            "path": PRODUCER_RELATIVE,
            "sha256": producer_sha,
        },
        "profile_pins_sha256": profile_pin["sha256"],
        "solver_profile_sha256": profile_pin["solver_profile_sha256"],
        "physics_profile_sha256": profile_pin["physics_profile_sha256"],
        "prototype": {
            "path": prototype_target[2],
            "sha256": prototype_sha,
        },
        "manifest": {
            "path": manifest_target[2],
            "sha256": manifest_sha,
        },
        "receipt": {
            "path": receipt_target[2],
            "sha256": receipt_sha,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-id",
        default=ACTION_ID,
        choices=tuple(sorted(_ACTION_REGISTRY.ALLOWED_ACTION_IDS)),
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--profile-pins", required=True)
    parser.add_argument("--expected-profile-pins-sha256", required=True)
    parser.add_argument("--prototype-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--receipt-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = materialize_a3_vendor_identity_manifest(
            action_id=args.action_id,
            repo_root=Path(args.repo_root),
            source_commit=args.source_commit,
            source_manifest=args.source_manifest,
            expected_source_manifest_sha256=(
                args.expected_source_manifest_sha256
            ),
            profile_pins=args.profile_pins,
            expected_profile_pins_sha256=args.expected_profile_pins_sha256,
            prototype_output=args.prototype_output,
            manifest_output=args.manifest_output,
            receipt_output=args.receipt_output,
        )
    except IdentityManifestRepinError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
