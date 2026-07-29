#!/usr/bin/env python3
"""Materialize the code-trusted fresh-N5 launch closure without self-authorizing it.

The tool deliberately has four phases:

``sidecar``
    Build the frozen-evaluation sidecar receipt and emit the two inbox trust
    digests that a reviewer must pin in source.

``physical-gate``
    Preserve the strict training manifest byte-for-byte while materializing a
    disposable, no-clobber copy with the already-solved physical ball/task
    overlay required by the teacher/policy fitted-ball Gates.

``formal``
    Refuse to run until those two sidecar digests are the exact singleton
    literals in ``action_ball_evaluation_inbox.py``.  Then build the selected
    upper registry, fresh-N5 promotion certificate, static motion-admission
    receipt, V4 evaluator receipt, drain/reset receipt, and a proposal for the
    remaining three code-owned trust sets.

``verify``
    Re-open every produced byte, require all four code-owned trust sources to
    contain exactly the proposed singleton values, and re-run the strict
    registry and promotion-evidence validators.

No phase edits a trust source.  Outputs are published only into a new
repository-relative directory; an existing target is never reused.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from dataclasses import fields
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


ACTION_ORDER = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
BANK_ORDER = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
    "v12_forehand_block",
)
RETIRED_ACTIONS = frozenset(("fh_loop", "fh_block_syn"))
MOBILITY_MODE = "no_move"
SCOPE = "upper"

SIDECAR_KIND = "fresh_n5_sidecar_materialization_v1"
FORMAL_SPEC_KIND = "fresh_n5_formal_launch_materialization_spec_v1"
FORMAL_RECEIPT_KIND = "fresh_n5_formal_launch_materialization_receipt_v1"
PHYSICAL_GATE_RECEIPT_KIND = (
    "fresh_n5_disposable_physical_gate_manifest_materialization_v1"
)
ACTION_SET_PROFILE = "fresh_upper_nomove_n5_v3"
TEACHER_GATE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate.py"
)
TRUST_PROPOSAL_KIND = "fresh_n5_code_trust_pin_proposal_v1"

SIDECAR_CODE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_frozen_eval_sidecar.py"
)
PROMOTION_TRUST_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "canonical_motion_admission.py"
)
EVALUATOR_TRUST_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_evaluation.py"
)
INBOX_SOURCE = (
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
DRAIN_TRUST_NAME = "TRUSTED_DRAIN_RESET_LAUNCH_RECEIPT_SHA256"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class MaterializationError(RuntimeError):
    """A requested artifact closure is incomplete, stale, or unsafe."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise MaterializationError(
            f"{label} must be one lowercase SHA-256"
        )
    return value


def _exact_dict(
    value: object, keys: Sequence[str], label: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise MaterializationError(f"{label} must be an object")
    expected = frozenset(keys)
    actual = frozenset(value)
    if actual != expected:
        raise MaterializationError(
            f"{label} keys changed: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return dict(value)


def _strict_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MaterializationError(f"cannot read {label}: {exc}") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MaterializationError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MaterializationError(
                    f"{label} contains non-finite token {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"{label} is not strict JSON: {exc}") from exc
    if type(value) is not dict:
        raise MaterializationError(f"{label} must contain one JSON object")
    return value, raw


def _repo_root(value: os.PathLike[str] | str) -> Path:
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise MaterializationError(
            f"cannot resolve repository root {value}: {exc}"
        ) from exc
    if not root.is_dir() or not (root / ".git").exists():
        raise MaterializationError(
            "repository root must be an existing Git worktree"
        )
    return root


def _relative_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or ":" in value
        or "\x00" in value
    ):
        raise MaterializationError(
            f"{label} must be a non-empty POSIX repository-relative path"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(
        part in ("", ".", "..") for part in pure.parts
    ):
        raise MaterializationError(
            f"{label} must be a normalized repository-relative path"
        )
    return pure.as_posix()


def _assert_no_symlink_components(
    root: Path,
    candidate: Path,
    label: str,
    *,
    include_leaf: bool,
) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise MaterializationError(f"{label} escapes the repository") from exc
    components = relative.parts if include_leaf else relative.parts[:-1]
    current = root
    for component in components:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise MaterializationError(
                f"{label} path component cannot be inspected: {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise MaterializationError(
                f"{label} must not contain a symlink: {current}"
            )


def _repo_member(
    root: Path,
    value: object,
    label: str,
    *,
    must_exist: bool = True,
) -> tuple[str, Path]:
    relative = _relative_text(value, label)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    _assert_no_symlink_components(
        root,
        candidate,
        label,
        include_leaf=must_exist,
    )
    try:
        resolved = (
            candidate.resolve(strict=True)
            if must_exist
            else candidate.parent.resolve(strict=True) / candidate.name
        )
    except OSError as exc:
        raise MaterializationError(f"cannot resolve {label}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MaterializationError(f"{label} escapes the repository") from exc
    if must_exist:
        try:
            stat_result = resolved.stat()
        except OSError as exc:
            raise MaterializationError(f"cannot stat {label}: {exc}") from exc
        if not resolved.is_file() or stat_result.st_nlink < 1:
            raise MaterializationError(f"{label} must be a regular file")
    return relative, resolved


def _pin(
    root: Path,
    value: object,
    label: str,
) -> dict[str, str]:
    row = _exact_dict(value, ("path", "sha256"), label)
    relative, path = _repo_member(root, row["path"], f"{label}.path")
    expected = _sha(row["sha256"], f"{label}.sha256")
    actual = _sha256_bytes(path.read_bytes())
    if actual != expected:
        raise MaterializationError(
            f"{label} bytes drifted: expected={expected}, actual={actual}"
        )
    return {"path": relative, "sha256": actual}


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _native_rename_noreplace(source: Path, destination: Path) -> bool:
    """Atomically install a directory without ever replacing the target."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(str(source))
    destination_bytes = os.fsencode(str(destination))
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        renameat2 = library.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,
        )
        if result == 0:
            return True
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(str(destination))
        if error not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise OSError(error, os.strerror(error), str(destination))
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        renamex_np = library.renamex_np
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
        if result == 0:
            return True
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(str(destination))
        if error not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise OSError(error, os.strerror(error), str(destination))
    return False


def _write_new_directory(
    root: Path,
    relative: str,
    documents: Mapping[str, object],
    *,
    prepublish: Optional[Any] = None,
) -> Path:
    relative, target = _repo_member(
        root, relative, "output directory", must_exist=False
    )
    del relative
    if target.exists() or target.is_symlink():
        raise MaterializationError(
            f"output directory already exists: {target}"
        )
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.staging-",
                dir=str(target.parent),
            )
        )
    except OSError as exc:
        raise MaterializationError(
            f"cannot create output staging directory beside {target}: {exc}"
        ) from exc
    published = False
    try:
        for name, document in documents.items():
            if (
                type(name) is not str
                or not name
                or "/" in name
                or "\\" in name
            ):
                raise MaterializationError(
                    f"unsafe output filename {name!r}"
                )
            path = staging / name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(path, flags, 0o444)
            try:
                payload = (
                    document
                    if isinstance(document, bytes)
                    else _json_bytes(document)
                )
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    descriptor = -1
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        os.chmod(staging, 0o755)
        _fsync_directory(staging)
        if prepublish is not None:
            prepublish(staging, target)
        try:
            installed = _native_rename_noreplace(staging, target)
        except FileExistsError as exc:
            raise MaterializationError(
                f"output directory already exists: {target}"
            ) from exc
        if not installed:
            raise MaterializationError(
                "platform lacks atomic directory no-replace rename; "
                "refusing a partial-visible publication"
            )
        published = True
        _fsync_directory(target.parent)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return target


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MaterializationError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise MaterializationError(
            f"cannot load contract module {path}: {exc}"
        ) from exc
    return module


def _runtime_modules(root: Path) -> tuple[Any, Any, Any]:
    mdp = (
        root
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp"
    )
    curriculum = _load_module(
        mdp / "action_ball_curriculum.py",
        "_fresh_n5_action_ball_curriculum",
    )
    missing = object()
    previous = sys.modules.get("action_ball_curriculum", missing)
    sys.modules["action_ball_curriculum"] = curriculum
    try:
        evaluation = _load_module(
            mdp / "action_ball_evaluation.py",
            "_fresh_n5_action_ball_evaluation",
        )
    finally:
        if previous is missing:
            sys.modules.pop("action_ball_curriculum", None)
        else:
            sys.modules["action_ball_curriculum"] = previous
    inbox = _load_module(
        mdp / "action_ball_evaluation_inbox.py",
        "_fresh_n5_action_ball_evaluation_inbox",
    )
    return curriculum, evaluation, inbox


def _registry_module(root: Path) -> Any:
    scripts = root / "hope_training/whole_body_tracking/scripts"
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(scripts))
        return _load_module(
            scripts / "canonical_motion_registry.py",
            "_fresh_n5_canonical_motion_registry",
        )
    finally:
        sys.path[:] = old_path


def _teacher_gate_module(root: Path) -> Any:
    scripts = root / "hope_training/whole_body_tracking/scripts"
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(scripts))
        return _load_module(
            root / TEACHER_GATE_SOURCE,
            "_fresh_n5_teacher_physical_gate",
        )
    finally:
        sys.path[:] = old_path


def _literal_trust_set(path: Path, variable: str) -> frozenset[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise MaterializationError(
            f"cannot parse trust source {path}: {exc}"
        ) from exc
    values: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
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
        raise MaterializationError(
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
        raise MaterializationError(
            f"{variable} must be frozenset(<literal strings>)"
        )
    try:
        raw = () if not expression.args else ast.literal_eval(expression.args[0])
    except (ValueError, SyntaxError) as exc:
        raise MaterializationError(
            f"{variable} must contain literal strings"
        ) from exc
    if type(raw) not in (tuple, list, set, frozenset):
        raise MaterializationError(
            f"{variable} literal must be a sequence/set"
        )
    values_set = frozenset(raw)
    if len(values_set) != len(raw) or any(
        type(item) is not str or _SHA256.fullmatch(item) is None
        for item in raw
    ):
        raise MaterializationError(
            f"{variable} must contain unique lowercase SHA-256 strings"
        )
    return values_set


def _require_singleton_trust(
    root: Path, source: str, variable: str, expected: str
) -> None:
    _, path = _repo_member(root, source, f"{variable} source")
    actual = _literal_trust_set(path, variable)
    wanted = frozenset((_sha(expected, variable),))
    if actual != wanted:
        raise MaterializationError(
            f"{variable} must equal the exact singleton {sorted(wanted)}; "
            f"actual={sorted(actual)}"
        )


def _load_manifest(
    root: Path,
    pin_value: object,
    *,
    expected_prototype_pin_value: Optional[object] = None,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    pin = _pin(root, pin_value, "formal spec manifest")
    _, path = _repo_member(root, pin["path"], "formal manifest path")
    manifest, _ = _strict_json(path, "ActionBall manifest")
    if (
        manifest.get("schema_version") != 3
        or manifest.get("mobility_mode") != MOBILITY_MODE
        or manifest.get("action_order") != list(ACTION_ORDER)
    ):
        raise MaterializationError(
            "manifest must be schema-v3 exact fresh upper/no-move N5"
        )
    prototype = _exact_dict(
        manifest.get("prototype"),
        ("path", "sha256", "scope"),
        "manifest.prototype",
    )
    prototype_pin = _pin(
        root,
        {"path": prototype["path"], "sha256": prototype["sha256"]},
        "manifest prototype",
    )
    if prototype["scope"] != SCOPE:
        raise MaterializationError(
            "manifest prototype must bind scope=upper"
        )
    if expected_prototype_pin_value is not None:
        expected_prototype_pin = _pin(
            root,
            expected_prototype_pin_value,
            "formal spec prototype",
        )
        if prototype_pin != expected_prototype_pin:
            raise MaterializationError(
                "manifest prototype path/SHA differs from the formal spec pin"
            )
    actions = manifest.get("actions")
    if type(actions) is not list or len(actions) != len(ACTION_ORDER):
        raise MaterializationError(
            "manifest.actions must contain exactly five ordered rows"
        )
    bindings: list[dict[str, Any]] = []
    seen_uids: set[int] = set()
    for index, (action, action_id) in enumerate(zip(actions, ACTION_ORDER)):
        if type(action) is not dict or action.get("action_id") != action_id:
            raise MaterializationError(
                f"manifest action[{index}] does not preserve fresh-N5 order"
            )
        if action_id in RETIRED_ACTIONS:
            raise MaterializationError("retired action entered fresh N5")
        uid = action.get("action_uid")
        if (
            type(uid) is not int
            or uid < 1
            or uid > (1 << 53) - 1
            or uid in seen_uids
        ):
            raise MaterializationError(
                f"manifest action[{index}].action_uid is invalid/duplicated"
            )
        seen_uids.add(uid)
        motion_relative, motion_path = _repo_member(
            root,
            action.get("motion_path"),
            f"manifest action[{index}].motion_path",
        )
        motion_sha = _sha(
            action.get("motion_sha256"),
            f"manifest action[{index}].motion_sha256",
        )
        if _sha256_bytes(motion_path.read_bytes()) != motion_sha:
            raise MaterializationError(
                f"manifest action[{index}] motion bytes drifted"
            )
        family = action.get("family")
        if family not in ("forehand", "backhand"):
            raise MaterializationError(
                f"manifest action[{index}].family is invalid"
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
    _sha(manifest.get("solver_profile_sha256"), "manifest solver profile")
    _sha(manifest.get("physics_profile_sha256"), "manifest physics profile")
    return manifest, pin, bindings


def _physical_bundle(
    root: Path,
    pin_value: object,
    *,
    base_manifest_pin: Mapping[str, str],
    bindings: Sequence[Mapping[str, Any]],
    prototype: Mapping[str, Any],
    solver_sha256: str,
    physics_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    pin = _pin(root, pin_value, "physical task bundle")
    _, path = _repo_member(root, pin["path"], "physical task bundle path")
    bundle, _ = _strict_json(path, "physical task bundle")
    bundle = _exact_dict(
        bundle,
        (
            "schema_version",
            "artifact_type",
            "base_manifest",
            "batch",
            "prototype",
            "profile_pins",
            "action_order",
            "selector_executed",
            "action_identity_frozen",
            "action_switching_allowed",
            "mobility_mode",
            "base_task_frame",
            "gate_materialization_fields",
            "materialization_contract",
            "actions",
            "content_sha256",
        ),
        "physical task bundle",
    )
    declared_content = _sha(
        bundle["content_sha256"], "physical task bundle content_sha256"
    )
    unsigned = dict(bundle)
    del unsigned["content_sha256"]
    if canonical_sha256(unsigned) != declared_content:
        raise MaterializationError(
            "physical task bundle content seal is false"
        )
    base = _exact_dict(
        bundle["base_manifest"],
        ("path", "raw_sha256", "schema_version", "strict_training_input"),
        "physical task bundle base_manifest",
    )
    if (
        base["path"] != base_manifest_pin["path"]
        or base["raw_sha256"] != base_manifest_pin["sha256"]
        or base["schema_version"] != 3
        or base["strict_training_input"] is not True
    ):
        raise MaterializationError(
            "physical task bundle binds a different strict base manifest"
        )
    if (
        bundle["schema_version"] != 1
        or bundle["artifact_type"] != "fresh_n5_physical_task_bundle_v1"
        or bundle["action_order"] != list(ACTION_ORDER)
        or bundle["selector_executed"] is not False
        or bundle["action_identity_frozen"] is not True
        or bundle["action_switching_allowed"] is not False
        or bundle["mobility_mode"] != MOBILITY_MODE
        or bundle["base_task_frame"]
        != "relative_about_actual_episode_spawn"
    ):
        raise MaterializationError(
            "physical task bundle is not the frozen-action no-move contract"
        )
    materialization = _exact_dict(
        bundle["materialization_contract"],
        (
            "training_consumer",
            "fitted_gate_consumer",
            "current_inline_fitted_gate_support",
            "downstream_gap",
            "required_external_inputs_not_synthesized",
        ),
        "physical task bundle materialization_contract",
    )
    if (
        materialization["training_consumer"] != "consume base_manifest only"
        or materialization["current_inline_fitted_gate_support"] is not False
    ):
        raise MaterializationError(
            "physical task bundle no longer preserves the strict training input"
        )
    required_external = materialization[
        "required_external_inputs_not_synthesized"
    ]
    if (
        type(required_external) is not list
        or required_external
        != [
            "per-action compiler_candidate_pre_admission_v1 evidence",
            "formal source-receipt trust root bound to a clean commit",
            "clean committed runtime/source/data closure",
        ]
    ):
        raise MaterializationError(
            "physical task bundle external-input disclosure drifted"
        )
    batch = _exact_dict(
        bundle["batch"],
        ("path", "sha256"),
        "physical task bundle batch",
    )
    _pin(root, batch, "physical task bundle batch")
    bundle_prototype = _exact_dict(
        bundle["prototype"],
        ("path", "sha256", "scope"),
        "physical task bundle prototype",
    )
    if bundle_prototype != dict(prototype):
        raise MaterializationError(
            "physical task bundle prototype differs from strict manifest"
        )
    _pin(
        root,
        {
            "path": bundle_prototype["path"],
            "sha256": bundle_prototype["sha256"],
        },
        "physical task bundle prototype",
    )
    profile_pins = _exact_dict(
        bundle["profile_pins"],
        (
            "path",
            "sha256",
            "solver_profile_sha256",
            "physics_profile_sha256",
            "geometry_source_sha256",
        ),
        "physical task bundle profile_pins",
    )
    if (
        profile_pins["solver_profile_sha256"] != solver_sha256
        or profile_pins["physics_profile_sha256"] != physics_sha256
    ):
        raise MaterializationError(
            "physical task bundle solver/physics identity differs from manifest"
        )
    _sha(
        profile_pins["geometry_source_sha256"],
        "physical task bundle geometry source",
    )
    _pin(
        root,
        {
            "path": profile_pins["path"],
            "sha256": profile_pins["sha256"],
        },
        "physical task bundle profile pins",
    )
    gate_fields = _exact_dict(
        bundle["gate_materialization_fields"],
        ("racket_geometry_contract", "physical_contact_contract"),
        "physical task bundle gate_materialization_fields",
    )
    geometry_contract = _exact_dict(
        gate_fields["racket_geometry_contract"],
        (
            "schema_version",
            "semantics",
            "ball_target_point",
            "site_target_mapping",
            "face_velocity_mapping",
            "source_path",
            "source_sha256",
            "geometry_source_sha256",
        ),
        "physical task bundle racket_geometry_contract",
    )
    if (
        geometry_contract["schema_version"] != 2
        or geometry_contract["semantics"] != "exact_face_contact_v2"
        or geometry_contract["geometry_source_sha256"]
        != profile_pins["geometry_source_sha256"]
    ):
        raise MaterializationError(
            "physical task bundle racket geometry identity drifted"
        )
    _pin(
        root,
        {
            "path": geometry_contract["source_path"],
            "sha256": geometry_contract["source_sha256"],
        },
        "physical task bundle racket geometry source",
    )
    contact_contract = gate_fields["physical_contact_contract"]
    if (
        type(contact_contract) is not dict
        or contact_contract.get("schema_version") != 2
        or not contact_contract
    ):
        raise MaterializationError(
            "physical task bundle physical_contact_contract must be schema v2"
        )
    actions = bundle["actions"]
    if type(actions) is not list or len(actions) != len(bindings):
        raise MaterializationError(
            "physical task bundle must contain five ordered actions"
        )
    for index, (raw, binding) in enumerate(zip(actions, bindings)):
        row = _exact_dict(
            raw,
            (
                "action_id",
                "action_uid",
                "motion_sha256",
                "physical_ball_launch",
                "physical_task_binding",
            ),
            f"physical task bundle actions[{index}]",
        )
        if (
            row["action_id"] != binding["motion_id"]
            or row["action_uid"] != binding["action_uid"]
            or row["motion_sha256"] != binding["motion_sha256"]
            or type(row["physical_ball_launch"]) is not dict
            or not row["physical_ball_launch"]
            or type(row["physical_task_binding"]) is not dict
            or not row["physical_task_binding"]
        ):
            raise MaterializationError(
                f"physical task bundle actions[{index}] identity/payload drifted"
            )
        task = _exact_dict(
            row["physical_task_binding"],
            (
                "schema_version",
                "authority",
                "action_id",
                "action_uid",
                "motion_sha256",
                "ball_profile_sha256",
                "solver_profile_sha256",
                "physics_profile_sha256",
                "solver_implementation_source_sha256",
                "solver_execution_receipt_path",
                "solver_execution_receipt_sha256",
                "solver_execution_identity",
                "solver_execution_identity_sha256",
                "selector_executed",
                "action_identity_frozen",
                "cases",
                "cases_sha256",
            ),
            f"physical task bundle actions[{index}].physical_task_binding",
        )
        execution = _exact_dict(
            task["solver_execution_identity"],
            (
                "artifact_type",
                "execution_id",
                "executed_before_gate",
                "solver_replayed_exact",
                "selector_executed",
                "action_identity_frozen",
                "action_switching_allowed",
                "hardware_authorized",
            ),
            (
                f"physical task bundle actions[{index}]"
                ".solver_execution_identity"
            ),
        )
        expected_execution_id = (
            f"fresh-n5:{base_manifest_pin['sha256']}:{binding['motion_id']}"
        )
        if (
            task["schema_version"] != 1
            or task.get("authority")
            != "pre_registered_frozen_action_ball_solver_receipt_v1"
            or task.get("action_id") != binding["motion_id"]
            or task.get("action_uid") != binding["action_uid"]
            or task.get("motion_sha256") != binding["motion_sha256"]
            or task.get("solver_profile_sha256") != solver_sha256
            or task.get("physics_profile_sha256") != physics_sha256
            or task.get("selector_executed") is not False
            or task.get("action_identity_frozen") is not True
            or type(execution) is not dict
            or execution.get("artifact_type")
            != "frozen_ball_to_task_solver_execution_v1"
            or execution.get("execution_id") != expected_execution_id
            or execution.get("executed_before_gate") is not True
            or execution.get("solver_replayed_exact") is not True
            or execution.get("selector_executed") is not False
            or execution.get("action_identity_frozen") is not True
            or execution.get("action_switching_allowed") is not False
            or execution.get("hardware_authorized") is not False
            or task.get("solver_execution_identity_sha256")
            != canonical_sha256(execution)
            or type(task.get("cases")) is not list
            or not task.get("cases")
            or task.get("cases_sha256")
            != canonical_sha256(task.get("cases"))
        ):
            raise MaterializationError(
                f"physical task bundle actions[{index}] solver receipt "
                "does not bind the exact strict base/action identity"
            )
        receipt_pin = _pin(
            root,
            {
                "path": task.get("solver_execution_receipt_path"),
                "sha256": task.get("solver_execution_receipt_sha256"),
            },
            f"physical task bundle actions[{index}] solver receipt",
        )
        _, receipt_path = _repo_member(
            root,
            receipt_pin["path"],
            f"physical task bundle actions[{index}] solver receipt path",
        )
        receipt, _ = _strict_json(
            receipt_path,
            f"physical task bundle actions[{index}] solver receipt",
        )
        if (
            receipt.get("solver_execution_identity") != execution
            or receipt.get("action_identity")
            != {
                "action_id": binding["motion_id"],
                "action_uid": binding["action_uid"],
                "motion_sha256": binding["motion_sha256"],
            }
            or receipt.get("cases") != task.get("cases")
            or receipt.get("receipt_payload_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_payload_sha256"
                }
            )
        ):
            raise MaterializationError(
                f"physical task bundle actions[{index}] external solver "
                "receipt identity drifted"
            )
    return bundle, pin


def _candidate_document(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "fresh_n5_compiler_candidate_registry_entry_v1",
        "motion_id": binding["motion_id"],
        "scope": SCOPE,
        "npz_path": binding["motion_path"],
        "npz_sha256": binding["motion_sha256"],
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }


def _physical_gate_documents(
    root: Path,
    *,
    base_manifest_pin_value: object,
    bundle_pin_value: object,
    base_build_manifest_pin_value: object,
    append_build_manifest_pin_value: object,
    base_bank_gate_pin_value: object,
    append_bank_gate_pin_value: object,
    output_dir_relative: str,
) -> dict[str, object]:
    manifest, manifest_pin, bindings = _load_manifest(
        root, base_manifest_pin_value
    )
    if (
        "racket_geometry_contract" in manifest
        or "physical_contact_contract" in manifest
        or any(
            "physical_ball_launch" in row
            or "physical_task_binding" in row
            or "admission" in row
            for row in manifest["actions"]
        )
    ):
        raise MaterializationError(
            "strict training manifest already contains gate-only inline extras"
        )
    bundle, bundle_pin = _physical_bundle(
        root,
        bundle_pin_value,
        base_manifest_pin=manifest_pin,
        bindings=bindings,
        prototype=manifest["prototype"],
        solver_sha256=manifest["solver_profile_sha256"],
        physics_sha256=manifest["physics_profile_sha256"],
    )
    base_build = _pin(
        root, base_build_manifest_pin_value, "physical gate base build manifest"
    )
    append_build = _pin(
        root,
        append_build_manifest_pin_value,
        "physical gate append build manifest",
    )
    base_gate = _pin(
        root, base_bank_gate_pin_value, "physical gate base bank report"
    )
    append_gate = _pin(
        root, append_bank_gate_pin_value, "physical gate append bank report"
    )
    output_dir_relative = _relative_text(
        output_dir_relative, "physical gate output directory"
    )
    documents: dict[str, object] = {}
    candidate_pins: dict[str, dict[str, str]] = {}
    for binding in bindings:
        filename = (
            f"physical_gate_candidate__{binding['motion_id']}.json"
        )
        document = _candidate_document(binding)
        documents[filename] = document
        candidate_pins[binding["motion_id"]] = {
            "path": f"{output_dir_relative}/{filename}",
            "sha256": _sha256_bytes(_json_bytes(document)),
        }
    overlay_by_id = {
        row["action_id"]: row for row in bundle["actions"]
    }
    physical = json.loads(json.dumps(manifest))
    gate_fields = bundle["gate_materialization_fields"]
    physical["racket_geometry_contract"] = gate_fields[
        "racket_geometry_contract"
    ]
    physical["physical_contact_contract"] = gate_fields[
        "physical_contact_contract"
    ]
    for action in physical["actions"]:
        action_id = action["action_id"]
        overlay = overlay_by_id[action_id]
        is_append = action_id in ("v12_forehand_block", "fh_loop_high")
        compiler = append_build if is_append else base_build
        bank_gate = append_gate if is_append else base_gate
        action["physical_ball_launch"] = overlay["physical_ball_launch"]
        action["physical_task_binding"] = overlay[
            "physical_task_binding"
        ]
        action["admission"] = {
            "evidence_stage": "compiler_candidate_pre_admission_v1",
            "publication_class": "compiler_candidate",
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "scope": SCOPE,
            "registry_entry_path": candidate_pins[action_id]["path"],
            "registry_entry_sha256": candidate_pins[action_id]["sha256"],
            "compiler_manifest_path": compiler["path"],
            "compiler_manifest_sha256": compiler["sha256"],
            "bank_gate_report_path": bank_gate["path"],
            "bank_gate_report_sha256": bank_gate["sha256"],
        }
    physical_filename = "physical_gate_manifest.json"
    physical_pin = {
        "path": f"{output_dir_relative}/{physical_filename}",
        "sha256": _sha256_bytes(_json_bytes(physical)),
    }
    receipt = {
        "schema_version": 1,
        "kind": PHYSICAL_GATE_RECEIPT_KIND,
        "strict_training_manifest": manifest_pin,
        "physical_task_bundle": bundle_pin,
        "physical_gate_manifest": physical_pin,
        "candidate_entries": [
            {
                "action_id": action_id,
                **candidate_pins[action_id],
            }
            for action_id in ACTION_ORDER
        ],
        "compiler_manifests": {
            "base": base_build,
            "append": append_build,
        },
        "bank_gate_reports": {
            "base": base_gate,
            "append": append_gate,
        },
        "action_order": list(ACTION_ORDER),
        "strict_training_manifest_preserved": True,
        "inline_manifest_gate_only": True,
        "selector_executed": False,
        "authorization_granted": False,
    }
    documents[physical_filename] = physical
    documents["physical_gate_materialization_receipt.json"] = receipt
    return documents


def _sidecar_documents(root: Path) -> dict[str, object]:
    _, evaluation, inbox = _runtime_modules(root)
    del evaluation
    _, code_path = _repo_member(
        root, SIDECAR_CODE_SOURCE, "frozen sidecar code"
    )
    code_sha = _sha256_bytes(code_path.read_bytes())
    receipt = inbox.build_sidecar_launch_document(
        sidecar_code_sha256=code_sha,
        backend_contract_sha256=inbox.FORMAL_ISAAC_BACKEND_CONTRACT_SHA256,
    )
    inbox.validate_sidecar_launch_document(
        receipt,
        actual_sidecar_code_sha256=code_sha,
        backend_contract_sha256=inbox.FORMAL_ISAAC_BACKEND_CONTRACT_SHA256,
        require_trust=False,
    )
    proposal = {
        "schema_version": 1,
        "kind": TRUST_PROPOSAL_KIND,
        "phase": "sidecar",
        "pins": {
            SIDECAR_CODE_TRUST_NAME: code_sha,
            SIDECAR_LAUNCH_TRUST_NAME: receipt["content_sha256"],
        },
        "source": INBOX_SOURCE,
        "next_phase_requires_exact_singletons": True,
        "authorization_granted": False,
    }
    materialization = {
        "schema_version": 1,
        "kind": SIDECAR_KIND,
        "sidecar_code": {
            "path": SIDECAR_CODE_SOURCE,
            "sha256": code_sha,
        },
        "sidecar_launch_receipt": {
            "filename": "sidecar_launch_receipt.json",
            "file_sha256": _sha256_bytes(_json_bytes(receipt)),
            "canonical_sha256": canonical_sha256(receipt),
            "content_sha256": receipt["content_sha256"],
        },
        "trust_pin_proposal": {
            "filename": "sidecar_trust_pin_proposal.json",
            "file_sha256": _sha256_bytes(_json_bytes(proposal)),
        },
        "authorization_granted": False,
    }
    return {
        "sidecar_launch_receipt.json": receipt,
        "sidecar_trust_pin_proposal.json": proposal,
        "sidecar_materialization_receipt.json": materialization,
    }


def _sidecar_input(
    root: Path, value: object
) -> tuple[dict[str, Any], dict[str, str]]:
    pin = _pin(root, value, "sidecar launch receipt")
    _, path = _repo_member(root, pin["path"], "sidecar receipt path")
    document, _ = _strict_json(path, "sidecar launch receipt")
    _, _, inbox = _runtime_modules(root)
    _, code_path = _repo_member(
        root, SIDECAR_CODE_SOURCE, "frozen sidecar code"
    )
    code_sha = _sha256_bytes(code_path.read_bytes())
    inbox.validate_sidecar_launch_document(
        document,
        actual_sidecar_code_sha256=code_sha,
        backend_contract_sha256=inbox.FORMAL_ISAAC_BACKEND_CONTRACT_SHA256,
        require_trust=False,
    )
    _require_singleton_trust(
        root, INBOX_SOURCE, SIDECAR_CODE_TRUST_NAME, code_sha
    )
    _require_singleton_trust(
        root,
        INBOX_SOURCE,
        SIDECAR_LAUNCH_TRUST_NAME,
        document["content_sha256"],
    )
    return document, pin


def _normalize_discriminated(
    root: Path,
    value: object,
    label: str,
    kinds: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    rows = _exact_dict(value, tuple(kinds), label)
    result: dict[str, dict[str, str]] = {}
    for discriminator, expected_kind in kinds.items():
        row = _exact_dict(
            rows[discriminator],
            ("kind", "path", "sha256"),
            f"{label}.{discriminator}",
        )
        if row["kind"] != expected_kind:
            raise MaterializationError(
                f"{label}.{discriminator}.kind changed"
            )
        pin = _pin(
            root,
            {"path": row["path"], "sha256": row["sha256"]},
            f"{label}.{discriminator}",
        )
        result[discriminator] = {"kind": expected_kind, **pin}
    return result


def _normalize_fitted_capsule(
    root: Path, value: object
) -> dict[str, Any]:
    row = _exact_dict(
        value,
        ("path", "sha256", "retained_capsule_receipt"),
        "promotion.mujoco_fitted_ball_receipt",
    )
    formal = _pin(
        root,
        {"path": row["path"], "sha256": row["sha256"]},
        "promotion MuJoCo fitted receipt",
    )
    retained = _pin(
        root,
        row["retained_capsule_receipt"],
        "promotion retained capsule receipt",
    )
    return {**formal, "retained_capsule_receipt": retained}


def _registry_document(
    registry_spec: object,
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = _exact_dict(
        registry_spec,
        ("bank_id", "canonical_ready", "canonical_ready_fk", "entries"),
        "formal spec registry",
    )
    bank_id = row["bank_id"]
    if type(bank_id) is not str or _SLUG.fullmatch(bank_id) is None:
        raise MaterializationError("registry.bank_id must be a normalized slug")
    entries = row["entries"]
    if type(entries) is not list or len(entries) != len(ACTION_ORDER):
        raise MaterializationError(
            "registry.entries must contain exactly five complete rows"
        )
    for index, (entry, binding) in enumerate(zip(entries, bindings)):
        if type(entry) is not dict:
            raise MaterializationError(
                f"registry.entries[{index}] must be an object"
            )
        if (
            entry.get("motion_id") != binding["motion_id"]
            or entry.get("scope") != SCOPE
            or entry.get("npz_path") != binding["motion_path"]
            or entry.get("npz_sha256") != binding["motion_sha256"]
            or entry.get("family") != binding["family"]
            or entry.get("training_authorized") is not True
            or entry.get("deployment_authorized") is not False
            or entry.get("hardware_authorized") is not False
            or entry.get("publication_class") != "training_adopted"
        ):
            raise MaterializationError(
                f"registry.entries[{index}] does not exactly match the "
                "training-adopted manifest binding"
            )
    return {
        "schema_version": 2,
        "bank_id": bank_id,
        "scope": SCOPE,
        "canonical_ready_path": row["canonical_ready"]["path"],
        "canonical_ready_sha256": row["canonical_ready"]["sha256"],
        "canonical_ready_fk_path": row["canonical_ready_fk"]["path"],
        "canonical_ready_fk_sha256": row["canonical_ready_fk"]["sha256"],
        "motion_ids": list(ACTION_ORDER),
        "entries": entries,
    }


def _stage_registry(
    root: Path, document: Mapping[str, Any]
) -> tuple[Path, Any, Any]:
    registry_module = _registry_module(root)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".fresh-n5-registry-", dir=root)
    )
    path = staging_root / "canonical_registry.json"
    path.write_bytes(_json_bytes(document))
    digest = _sha256_bytes(path.read_bytes())
    try:
        registry = registry_module.load_canonical_motion_bank_registry(
            path,
            repo_root=root,
            expected_registry_sha256=digest,
        )
        generic = registry_module.bank_promotion_binding(
            registry, authorization_purpose="training"
        )
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return path, registry_module, (registry, generic)


def _fresh_binding(
    registry_module: Any,
    generic: Any,
    promotion: Mapping[str, Any],
) -> Any:
    admission = registry_module.motion_admission
    bank_hashes = promotion["bank_npz_sha256"]
    if (
        type(bank_hashes) is not list
        or len(bank_hashes) != 14
        or any(_SHA256.fullmatch(value or "") is None for value in bank_hashes)
        or len(set(bank_hashes)) != 14
    ):
        raise MaterializationError(
            "promotion.bank_npz_sha256 must contain fourteen distinct hashes"
        )
    if promotion["bank_motion_ids"] != list(BANK_ORDER):
        raise MaterializationError(
            "promotion.bank_motion_ids must preserve exact base-five+append order"
        )
    base_manifest = promotion["base_build_manifest"]
    append_manifest = promotion["append_build_manifest"]
    generic_kwargs = {
        field.name: getattr(generic, field.name)
        for field in fields(type(generic))
    }
    return admission.FreshN5BankPromotionBinding(
        **generic_kwargs,
        base_bank_id=promotion["base_bank_id"],
        bank_motion_ids=tuple(promotion["bank_motion_ids"]),
        bank_npz_sha256=tuple(bank_hashes),
        base_build_manifest_sha256=base_manifest["sha256"],
        append_build_manifest_sha256=append_manifest["sha256"],
        base_bank_gate_report_sha256=promotion["bank_gate_reports"][
            "base"
        ]["sha256"],
        append_bank_gate_report_sha256=promotion["bank_gate_reports"][
            "append"
        ]["sha256"],
        base_swept_clearance_receipt_sha256=promotion[
            "continuous_swept_clearance_receipts"
        ]["base"]["sha256"],
        append_swept_clearance_receipt_sha256=promotion[
            "continuous_swept_clearance_receipts"
        ]["append"]["sha256"],
        mujoco_fitted_ball_receipt_sha256=promotion[
            "mujoco_fitted_ball_receipt"
        ]["sha256"],
        mujoco_fitted_ball_capsule_receipt_sha256=promotion[
            "mujoco_fitted_ball_receipt"
        ]["retained_capsule_receipt"]["sha256"],
        isaac_table_filtered_smoke_receipt_sha256=promotion[
            "isaac_table_filtered_smoke_receipt"
        ]["sha256"],
    )


def _validate_physical_gate_closure(
    root: Path,
    value: object,
    *,
    manifest: Mapping[str, Any],
    manifest_pin: Mapping[str, str],
    bindings: Sequence[Mapping[str, Any]],
    promotion: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    row = _exact_dict(
        value,
        ("bundle", "manifest", "materialization_receipt"),
        "formal spec physical_gate",
    )
    bundle, bundle_pin = _physical_bundle(
        root,
        row["bundle"],
        base_manifest_pin=manifest_pin,
        bindings=bindings,
        prototype=manifest["prototype"],
        solver_sha256=str(manifest["solver_profile_sha256"]),
        physics_sha256=str(manifest["physics_profile_sha256"]),
    )
    gate_manifest_pin = _pin(
        root, row["manifest"], "formal physical-gate manifest"
    )
    gate_receipt_pin = _pin(
        root,
        row["materialization_receipt"],
        "formal physical-gate materialization receipt",
    )
    _, gate_path = _repo_member(
        root, gate_manifest_pin["path"], "formal physical-gate manifest path"
    )
    gate_manifest, _ = _strict_json(
        gate_path, "formal physical-gate manifest"
    )
    _, receipt_path = _repo_member(
        root,
        gate_receipt_pin["path"],
        "formal physical-gate materialization receipt path",
    )
    receipt, _ = _strict_json(
        receipt_path, "formal physical-gate materialization receipt"
    )
    receipt = _exact_dict(
        receipt,
        (
            "schema_version",
            "kind",
            "strict_training_manifest",
            "physical_task_bundle",
            "physical_gate_manifest",
            "candidate_entries",
            "compiler_manifests",
            "bank_gate_reports",
            "action_order",
            "strict_training_manifest_preserved",
            "inline_manifest_gate_only",
            "selector_executed",
            "authorization_granted",
        ),
        "formal physical-gate materialization receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != PHYSICAL_GATE_RECEIPT_KIND
        or receipt["strict_training_manifest"] != manifest_pin
        or receipt["physical_task_bundle"] != bundle_pin
        or receipt["physical_gate_manifest"] != gate_manifest_pin
        or receipt["action_order"] != list(ACTION_ORDER)
        or receipt["strict_training_manifest_preserved"] is not True
        or receipt["inline_manifest_gate_only"] is not True
        or receipt["selector_executed"] is not False
        or receipt["authorization_granted"] is not False
        or receipt["compiler_manifests"]
        != {
            "base": promotion["base_build_manifest"],
            "append": promotion["append_build_manifest"],
        }
        or receipt["bank_gate_reports"]
        != {
            "base": {
                "path": promotion["bank_gate_reports"]["base"]["path"],
                "sha256": promotion["bank_gate_reports"]["base"]["sha256"],
            },
            "append": {
                "path": promotion["bank_gate_reports"]["append"]["path"],
                "sha256": promotion["bank_gate_reports"]["append"]["sha256"],
            },
        }
    ):
        raise MaterializationError(
            "physical-gate materialization receipt crossbinding drifted"
        )
    candidate_rows = receipt["candidate_entries"]
    if type(candidate_rows) is not list or len(candidate_rows) != len(bindings):
        raise MaterializationError(
            "physical-gate candidate entry list must contain five rows"
        )
    candidates: dict[str, dict[str, str]] = {}
    for index, (raw, binding) in enumerate(
        zip(candidate_rows, bindings)
    ):
        candidate = _exact_dict(
            raw,
            ("action_id", "path", "sha256"),
            f"physical-gate candidate_entries[{index}]",
        )
        if candidate["action_id"] != binding["motion_id"]:
            raise MaterializationError(
                f"physical-gate candidate_entries[{index}] order drifted"
            )
        pin = _pin(
            root,
            {"path": candidate["path"], "sha256": candidate["sha256"]},
            f"physical-gate candidate_entries[{index}]",
        )
        _, candidate_path = _repo_member(
            root,
            pin["path"],
            f"physical-gate candidate_entries[{index}] path",
        )
        document, _ = _strict_json(
            candidate_path,
            f"physical-gate candidate_entries[{index}]",
        )
        if document != _candidate_document(binding):
            raise MaterializationError(
                f"physical-gate candidate_entries[{index}] bytes are not "
                "the exact compiler-candidate projection"
            )
        candidates[binding["motion_id"]] = pin
    if (
        gate_manifest.get("schema_version") != 3
        or gate_manifest.get("action_order") != list(ACTION_ORDER)
        or gate_manifest.get("mobility_mode") != MOBILITY_MODE
    ):
        raise MaterializationError(
            "disposable physical-gate manifest identity changed"
        )
    gate_actions = gate_manifest.get("actions")
    base_actions = manifest.get("actions")
    if (
        type(gate_actions) is not list
        or type(base_actions) is not list
        or len(gate_actions) != len(bindings)
    ):
        raise MaterializationError(
            "disposable physical-gate manifest action matrix changed"
        )
    overlay_by_id = {
        action["action_id"]: action for action in bundle["actions"]
    }
    for index, (gate_action, base_action, binding) in enumerate(
        zip(gate_actions, base_actions, bindings)
    ):
        if type(gate_action) is not dict:
            raise MaterializationError(
                f"physical-gate manifest actions[{index}] must be an object"
            )
        stripped = dict(gate_action)
        launch = stripped.pop("physical_ball_launch", None)
        task = stripped.pop("physical_task_binding", None)
        admission = stripped.pop("admission", None)
        if stripped != base_action:
            raise MaterializationError(
                f"physical-gate manifest actions[{index}] modified strict "
                "training fields"
            )
        overlay = overlay_by_id[binding["motion_id"]]
        if (
            launch != overlay["physical_ball_launch"]
            or task != overlay["physical_task_binding"]
        ):
            raise MaterializationError(
                f"physical-gate manifest actions[{index}] overlay payload drifted"
            )
        is_append = binding["motion_id"] in (
            "v12_forehand_block",
            "fh_loop_high",
        )
        compiler = (
            promotion["append_build_manifest"]
            if is_append
            else promotion["base_build_manifest"]
        )
        gate = (
            promotion["bank_gate_reports"]["append"]
            if is_append
            else promotion["bank_gate_reports"]["base"]
        )
        expected_admission = {
            "evidence_stage": "compiler_candidate_pre_admission_v1",
            "publication_class": "compiler_candidate",
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "scope": SCOPE,
            "registry_entry_path": candidates[binding["motion_id"]]["path"],
            "registry_entry_sha256": candidates[binding["motion_id"]][
                "sha256"
            ],
            "compiler_manifest_path": compiler["path"],
            "compiler_manifest_sha256": compiler["sha256"],
            "bank_gate_report_path": gate["path"],
            "bank_gate_report_sha256": gate["sha256"],
        }
        if admission != expected_admission:
            raise MaterializationError(
                f"physical-gate manifest actions[{index}] candidate admission "
                "crossbinding drifted"
            )
    top_without_actions = dict(gate_manifest)
    del top_without_actions["actions"]
    gate_geometry = top_without_actions.pop(
        "racket_geometry_contract", None
    )
    gate_contact = top_without_actions.pop(
        "physical_contact_contract", None
    )
    expected_top = dict(manifest)
    del expected_top["actions"]
    if (
        top_without_actions != expected_top
        or gate_geometry
        != bundle["gate_materialization_fields"][
            "racket_geometry_contract"
        ]
        or gate_contact
        != bundle["gate_materialization_fields"][
            "physical_contact_contract"
        ]
    ):
        raise MaterializationError(
            "disposable physical-gate manifest modified strict top-level fields"
        )
    return {
        "bundle": bundle_pin,
        "manifest": gate_manifest_pin,
        "materialization_receipt": gate_receipt_pin,
    }


def _formal_documents(
    root: Path,
    spec_path: Path,
    sidecar_document: Mapping[str, Any],
    sidecar_pin: Mapping[str, str],
) -> dict[str, object]:
    spec, spec_raw = _strict_json(spec_path, "formal materialization spec")
    spec = _exact_dict(
        spec,
        (
            "schema_version",
            "kind",
            "manifest",
            "prototype",
            "physical_gate",
            "registry",
            "promotion",
            "evaluator",
            "drain",
        ),
        "formal materialization spec",
    )
    if spec["schema_version"] != 1 or spec["kind"] != FORMAL_SPEC_KIND:
        raise MaterializationError(
            "formal materialization spec schema/kind changed"
        )
    manifest, manifest_pin, bindings = _load_manifest(
        root,
        spec["manifest"],
        expected_prototype_pin_value=spec["prototype"],
    )
    registry_doc = _registry_document(spec["registry"], bindings)
    stage_path, registry_module, loaded = _stage_registry(root, registry_doc)
    registry, generic = loaded
    try:
        promotion_raw = _exact_dict(
            spec["promotion"],
            (
                "base_bank_id",
                "bank_motion_ids",
                "bank_npz_sha256",
                "base_build_manifest",
                "append_build_manifest",
                "bank_gate_reports",
                "continuous_swept_clearance_receipts",
                "mujoco_fitted_ball_receipt",
                "isaac_table_filtered_smoke_receipt",
            ),
            "formal spec promotion",
        )
        if (
            type(promotion_raw["base_bank_id"]) is not str
            or _SLUG.fullmatch(promotion_raw["base_bank_id"]) is None
            or promotion_raw["base_bank_id"] == registry.bank_id
        ):
            raise MaterializationError(
                "promotion.base_bank_id must be a distinct normalized slug"
            )
        promotion = {
            "base_bank_id": promotion_raw["base_bank_id"],
            "bank_motion_ids": promotion_raw["bank_motion_ids"],
            "bank_npz_sha256": promotion_raw["bank_npz_sha256"],
            "base_build_manifest": _pin(
                root,
                promotion_raw["base_build_manifest"],
                "promotion base build manifest",
            ),
            "append_build_manifest": _pin(
                root,
                promotion_raw["append_build_manifest"],
                "promotion append build manifest",
            ),
            "bank_gate_reports": _normalize_discriminated(
                root,
                promotion_raw["bank_gate_reports"],
                "promotion bank gate reports",
                {
                    "base": "canonical_base_five_full_replay",
                    "append": "fresh_n5_append_suffix",
                },
            ),
            "continuous_swept_clearance_receipts": _normalize_discriminated(
                root,
                promotion_raw["continuous_swept_clearance_receipts"],
                "promotion swept receipts",
                {
                    "base": "canonical_base_five",
                    "append": "fresh_n5_append_suffix",
                },
            ),
            "mujoco_fitted_ball_receipt": _normalize_fitted_capsule(
                root, promotion_raw["mujoco_fitted_ball_receipt"]
            ),
            "isaac_table_filtered_smoke_receipt": _pin(
                root,
                promotion_raw["isaac_table_filtered_smoke_receipt"],
                "promotion Isaac table smoke receipt",
            ),
        }
        physical_gate = _validate_physical_gate_closure(
            root,
            spec["physical_gate"],
            manifest=manifest,
            manifest_pin=manifest_pin,
            bindings=bindings,
            promotion=promotion,
        )
        fresh = _fresh_binding(registry_module, generic, promotion)
        admission = registry_module.motion_admission
        certificate = {
            "schema_version": 3,
            "certificate_type": (
                "canonical-motion-fresh-n5-append-swept-promotion-v3"
            ),
            **admission._binding_document(fresh),
            "bank_gate_reports": promotion["bank_gate_reports"],
            "continuous_swept_clearance_receipts": promotion[
                "continuous_swept_clearance_receipts"
            ],
            "mujoco_fitted_ball_receipt": promotion[
                "mujoco_fitted_ball_receipt"
            ],
            "isaac_table_filtered_smoke_receipt": promotion[
                "isaac_table_filtered_smoke_receipt"
            ],
        }
        # These are the same evidence checks performed again by the trusted
        # admission capability after the certificate digest is reviewed.
        admission._validate_fresh_n5_bank_closure(
            certificate, binding=fresh, repo_root=root
        )
        fitted_identity = admission._validate_fresh_n5_fitted_ball_receipt(
            certificate["mujoco_fitted_ball_receipt"],
            binding=fresh,
            repo_root=root,
        )
        admission._validate_fresh_n5_isaac_table_smoke_receipt(
            certificate["isaac_table_filtered_smoke_receipt"],
            binding=fresh,
            repo_root=root,
            expected_identity=fitted_identity,
        )
        certificate_file_sha = _sha256_bytes(_json_bytes(certificate))

        curriculum, evaluation, inbox = _runtime_modules(root)
        evaluator_spec = _exact_dict(
            spec["evaluator"],
            (
                "policy_contract_sha256",
                "curriculum_contract_sha256",
                "profile_order",
                "scheduler_contract_sha256",
                "sampler_sha256",
            ),
            "formal spec evaluator",
        )
        profile_rows = evaluator_spec["profile_order"]
        if type(profile_rows) is not list or len(profile_rows) != len(bindings):
            raise MaterializationError(
                "evaluator.profile_order must contain exactly five rows"
            )
        profile_keys = []
        for index, (profile, binding) in enumerate(
            zip(profile_rows, bindings)
        ):
            profile = _exact_dict(
                profile,
                ("action_uid", "profile_sha256", "mobility"),
                f"evaluator.profile_order[{index}]",
            )
            if (
                profile["action_uid"] != binding["action_uid"]
                or profile["mobility"] != MOBILITY_MODE
            ):
                raise MaterializationError(
                    f"evaluator.profile_order[{index}] identity drifted"
                )
            _sha(
                profile["profile_sha256"],
                f"evaluator.profile_order[{index}].profile_sha256",
            )
            profile_keys.append(curriculum.ActionProfileKey(**profile))
        solver_sha = _sha(
            manifest["solver_profile_sha256"], "manifest solver profile"
        )
        evaluator = evaluation.launch_receipt_document_v4(
            curriculum_contract_sha256=_sha(
                evaluator_spec["curriculum_contract_sha256"],
                "evaluator curriculum contract",
            ),
            profile_order=tuple(profile_keys),
            arm_catalog_sha256=curriculum.ARM_CATALOG_SHA256,
            scheduler_contract_sha256=_sha(
                evaluator_spec["scheduler_contract_sha256"],
                "evaluator scheduler contract",
            ),
            sampler_sha256=_sha(
                evaluator_spec["sampler_sha256"],
                "evaluator sampler contract",
            ),
            solver_sha256=solver_sha,
            policy_contract_sha256=_sha(
                evaluator_spec["policy_contract_sha256"],
                "evaluator policy contract",
            ),
            attempt_source_contract_sha256=(
                inbox.FROZEN_EVALUATION_INBOX_ATTEMPT_SOURCE_CONTRACT_SHA256
            ),
            attempt_source_path=INBOX_SOURCE,
            attempt_source_sha256=_sha256_bytes(
                (root / INBOX_SOURCE).read_bytes()
            ),
        )
        drain_spec = _exact_dict(
            spec["drain"],
            (
                "runtime_source_contract_sha256",
                "broker_contract_sha256",
                "attempt_pool_contract_sha256",
                "task_receipt_pool_contract_sha256",
                "env_reset_contract_sha256",
            ),
            "formal spec drain",
        )
        drain = curriculum.drain_reset_launch_receipt_document(
            curriculum_contract_sha256=evaluator[
                "curriculum_contract_sha256"
            ],
            profile_order=tuple(profile_keys),
            arm_catalog_sha256=curriculum.ARM_CATALOG_SHA256,
            scheduler_contract_sha256=evaluator[
                "scheduler_contract_sha256"
            ],
            sampler_sha256=evaluator["sampler_sha256"],
            solver_sha256=evaluator["solver_sha256"],
            policy_contract_sha256=evaluator["policy_contract_sha256"],
            runtime_source_contract_sha256=_sha(
                drain_spec["runtime_source_contract_sha256"],
                "drain runtime source contract",
            ),
            runtime_source_path=HOPE_COMMANDS_SOURCE,
            runtime_source_sha256=_sha256_bytes(
                (root / HOPE_COMMANDS_SOURCE).read_bytes()
            ),
            broker_contract_sha256=_sha(
                drain_spec["broker_contract_sha256"],
                "drain broker contract",
            ),
            attempt_pool_contract_sha256=_sha(
                drain_spec["attempt_pool_contract_sha256"],
                "drain attempt-pool contract",
            ),
            task_receipt_pool_contract_sha256=_sha(
                drain_spec["task_receipt_pool_contract_sha256"],
                "drain task-receipt-pool contract",
            ),
            env_reset_contract_sha256=_sha(
                drain_spec["env_reset_contract_sha256"],
                "drain env-reset contract",
            ),
        )
        admission_unsigned = {
            "schema_version": 1,
            "kind": "action_ball_static_motion_admission_launch",
            "authorization_purpose": "training",
            "scope": SCOPE,
            "mobility_mode": MOBILITY_MODE,
            "ordered_action_ids": list(ACTION_ORDER),
            "registry_sha256": registry.registry_sha256,
            "promotion_certificate_sha256": certificate_file_sha,
            "motion_rows": [
                {
                    "motion_id": row["motion_id"],
                    "action_uid": row["action_uid"],
                    "motion_path": row["motion_path"],
                    "motion_sha256": row["motion_sha256"],
                }
                for row in bindings
            ],
        }
        static_admission = {
            **admission_unsigned,
            "canonical_sha256": canonical_sha256(admission_unsigned),
        }
        evaluator_canonical = evaluation._canonical_sha256(evaluator)
        drain_canonical = curriculum._canonical_sha256(drain)
        _, sidecar_code_path = _repo_member(
            root, SIDECAR_CODE_SOURCE, "sidecar code"
        )
        proposal = {
            "schema_version": 1,
            "kind": TRUST_PROPOSAL_KIND,
            "phase": "formal",
            "pins": {
                PROMOTION_TRUST_NAME: certificate_file_sha,
                EVALUATOR_TRUST_NAME: evaluator_canonical,
                SIDECAR_CODE_TRUST_NAME: _sha256_bytes(
                    sidecar_code_path.read_bytes()
                ),
                SIDECAR_LAUNCH_TRUST_NAME: sidecar_document[
                    "content_sha256"
                ],
                DRAIN_TRUST_NAME: drain_canonical,
            },
            "manifest_bindings": {
                "strict_training_manifest": manifest_pin,
                "prototype": dict(manifest["prototype"]),
                "physical_task_bundle": physical_gate["bundle"],
                "disposable_physical_gate_manifest": physical_gate[
                    "manifest"
                ],
                "physical_gate_materialization_receipt": physical_gate[
                    "materialization_receipt"
                ],
            },
            "sources": {
                PROMOTION_TRUST_NAME: PROMOTION_TRUST_SOURCE,
                EVALUATOR_TRUST_NAME: EVALUATOR_TRUST_SOURCE,
                SIDECAR_CODE_TRUST_NAME: INBOX_SOURCE,
                SIDECAR_LAUNCH_TRUST_NAME: INBOX_SOURCE,
                DRAIN_TRUST_NAME: CURRICULUM_TRUST_SOURCE,
            },
            "authorization_granted": False,
        }
        spec_relative = spec_path.relative_to(root).as_posix()
        outputs: dict[str, object] = {
            "canonical_registry.json": registry_doc,
            "promotion_certificate.json": certificate,
            "motion_admission_receipt.json": static_admission,
            "evaluator_launch_receipt.json": evaluator,
            "drain_reset_launch_receipt.json": drain,
            "formal_trust_pin_proposal.json": proposal,
        }
        output_pins = {
            name: {
                "file_sha256": _sha256_bytes(_json_bytes(document)),
                "canonical_sha256": canonical_sha256(document),
            }
            for name, document in outputs.items()
        }
        receipt = {
            "schema_version": 1,
            "kind": FORMAL_RECEIPT_KIND,
            "materialization_spec": {
                "path": spec_relative,
                "sha256": _sha256_bytes(spec_raw),
            },
            "manifest": manifest_pin,
            "prototype": dict(manifest["prototype"]),
            "physical_gate": physical_gate,
            "sidecar_launch_receipt": dict(sidecar_pin),
            "action_order": list(ACTION_ORDER),
            "scope": SCOPE,
            "mobility_mode": MOBILITY_MODE,
            "registry_alignment_sha256": generic.alignment_sha256,
            "outputs": output_pins,
            "authorization_granted": False,
        }
        outputs["formal_materialization_receipt.json"] = receipt
        return outputs
    finally:
        shutil.rmtree(stage_path.parent, ignore_errors=True)


def _sidecar_command(args: argparse.Namespace) -> int:
    root = _repo_root(args.repo_root)
    documents = _sidecar_documents(root)
    target = _write_new_directory(root, args.out_dir, documents)
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase": "sidecar",
                "output_dir": str(target),
                "next": (
                    "review and pin the two exact inbox singleton digests, "
                    "materialize/run the physical fitted Gates, then run formal"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_staged_physical_gate(
    root: Path,
    staging: Path,
    target: Path,
) -> None:
    """Run the same deep physical validator before atomic publication."""

    receipt, receipt_raw = _strict_json(
        staging / "physical_gate_materialization_receipt.json",
        "staged physical-gate materialization receipt",
    )
    physical, physical_raw = _strict_json(
        staging / "physical_gate_manifest.json",
        "staged physical-gate manifest",
    )
    strict_pin = _pin(
        root,
        receipt.get("strict_training_manifest"),
        "staged physical-gate strict manifest",
    )
    physical_binding = _exact_dict(
        receipt.get("physical_gate_manifest"),
        ("path", "sha256"),
        "staged physical-gate manifest binding",
    )
    physical_pin = {
        "path": _relative_text(
            physical_binding["path"],
            "staged physical-gate manifest binding.path",
        ),
        "sha256": _sha(
            physical_binding["sha256"],
            "staged physical-gate manifest binding.sha256",
        ),
    }
    expected_target_relative = target.relative_to(root).as_posix()
    if (
        physical_pin["path"]
        != f"{expected_target_relative}/physical_gate_manifest.json"
        or physical_pin["sha256"] != _sha256_bytes(physical_raw)
    ):
        raise MaterializationError(
            "staged physical-gate manifest bytes/path differ from its "
            "publication binding"
        )
    _, strict_path = _repo_member(
        root, strict_pin["path"], "staged strict manifest path"
    )
    strict, strict_raw = _strict_json(
        strict_path, "staged strict training manifest"
    )
    gate = _teacher_gate_module(root)
    try:
        trusted = gate.action_set_contract.load_contract_from_source(
            gate.ACTION_SET_CONTRACT_SOURCE_PATH.read_bytes(),
            ACTION_SET_PROFILE,
        )
        gate.action_set_contract.verify_manifest_identity(
            trusted, strict, strict_raw
        )
        if (
            trusted["manifest_path"] != strict_pin["path"]
            or trusted["manifest_sha256"] != strict_pin["sha256"]
        ):
            raise MaterializationError(
                "code-owned action-set contract does not bind the strict "
                "manifest path/SHA"
            )
        gate.validate_physical_materialization_receipt(
            receipt,
            strict_manifest_pin=strict_pin,
            physical_manifest_pin=physical_pin,
            trusted_action_set=trusted,
        )
        overrides: dict[str, Path] = {}
        for index, row in enumerate(receipt.get("candidate_entries", ())):
            if not isinstance(row, Mapping):
                raise MaterializationError(
                    f"candidate_entries[{index}] is not an object"
                )
            logical = _relative_text(
                row.get("path"), f"candidate_entries[{index}].path"
            )
            candidate = staging / PurePosixPath(logical).name
            overrides[logical] = candidate
        gate.validate_physical_manifest(
            physical,
            trusted_action_set=trusted,
            repo_file_overrides=overrides,
        )
    except MaterializationError:
        raise
    except Exception as exc:
        raise MaterializationError(
            f"teacher physical prepublication validation failed: {exc}"
        ) from exc
    if _sha256_bytes(receipt_raw) != _sha256_bytes(
        (staging / "physical_gate_materialization_receipt.json").read_bytes()
    ):
        raise MaterializationError(
            "staged physical materialization receipt changed during validation"
        )


def _physical_gate_command(args: argparse.Namespace) -> int:
    root = _repo_root(args.repo_root)
    output_relative = _relative_text(
        args.out_dir, "physical gate output directory"
    )
    documents = _physical_gate_documents(
        root,
        base_manifest_pin_value={
            "path": args.base_manifest,
            "sha256": args.base_manifest_sha256,
        },
        bundle_pin_value={
            "path": args.physical_task_bundle,
            "sha256": args.physical_task_bundle_sha256,
        },
        base_build_manifest_pin_value={
            "path": args.base_build_manifest,
            "sha256": args.base_build_manifest_sha256,
        },
        append_build_manifest_pin_value={
            "path": args.append_build_manifest,
            "sha256": args.append_build_manifest_sha256,
        },
        base_bank_gate_pin_value={
            "path": args.base_bank_gate,
            "sha256": args.base_bank_gate_sha256,
        },
        append_bank_gate_pin_value={
            "path": args.append_bank_gate,
            "sha256": args.append_bank_gate_sha256,
        },
        output_dir_relative=output_relative,
    )
    target = _write_new_directory(
        root,
        output_relative,
        documents,
        prepublish=lambda staging, target: _validate_staged_physical_gate(
            root, staging, target
        ),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase": "physical-gate",
                "output_dir": str(target),
                "strict_training_manifest": (
                    "preserved; continue using the original strict manifest "
                    "for training"
                ),
                "next": (
                    "run teacher/policy fitted-ball Gates only against "
                    "physical_gate_manifest.json, then bind their receipts in "
                    "the formal spec"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _formal_command(args: argparse.Namespace) -> int:
    root = _repo_root(args.repo_root)
    spec_relative, spec_path = _repo_member(
        root, args.spec, "formal materialization spec"
    )
    del spec_relative
    sidecar_document, sidecar_pin = _sidecar_input(
        root,
        {"path": args.sidecar_receipt, "sha256": args.sidecar_receipt_sha256},
    )
    documents = _formal_documents(
        root, spec_path, sidecar_document, sidecar_pin
    )
    target = _write_new_directory(root, args.out_dir, documents)
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase": "formal",
                "output_dir": str(target),
                "next": (
                    "review/pin promotion, evaluator, and drain singleton "
                    "digests, commit, then run verify"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    root = _repo_root(args.repo_root)
    output_relative = _relative_text(args.artifact_dir, "artifact directory")
    output_dir = root.joinpath(*PurePosixPath(output_relative).parts)
    try:
        output_dir = output_dir.resolve(strict=True)
    except OSError as exc:
        raise MaterializationError(
            f"cannot resolve artifact directory: {exc}"
        ) from exc
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise MaterializationError("artifact directory escapes repository") from exc
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise MaterializationError(
            "artifact directory must be a real repository directory"
        )
    receipt_path = output_dir / "formal_materialization_receipt.json"
    receipt, _ = _strict_json(
        receipt_path, "formal materialization receipt"
    )
    receipt = _exact_dict(
        receipt,
        (
            "schema_version",
            "kind",
            "materialization_spec",
            "manifest",
            "prototype",
            "physical_gate",
            "sidecar_launch_receipt",
            "action_order",
            "scope",
            "mobility_mode",
            "registry_alignment_sha256",
            "outputs",
            "authorization_granted",
        ),
        "formal materialization receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != FORMAL_RECEIPT_KIND
        or receipt["action_order"] != list(ACTION_ORDER)
        or receipt["scope"] != SCOPE
        or receipt["mobility_mode"] != MOBILITY_MODE
        or receipt["authorization_granted"] is not False
    ):
        raise MaterializationError(
            "formal materialization receipt identity changed"
        )
    outputs = receipt["outputs"]
    if type(outputs) is not dict:
        raise MaterializationError(
            "formal materialization receipt outputs must be an object"
        )
    loaded: dict[str, dict[str, Any]] = {}
    for filename, pin in outputs.items():
        if (
            type(filename) is not str
            or "/" in filename
            or "\\" in filename
        ):
            raise MaterializationError("formal output filename is unsafe")
        pin = _exact_dict(
            pin,
            ("file_sha256", "canonical_sha256"),
            f"formal output {filename}",
        )
        document, raw = _strict_json(
            output_dir / filename, f"formal output {filename}"
        )
        if (
            _sha256_bytes(raw)
            != _sha(pin["file_sha256"], f"{filename} file SHA")
            or canonical_sha256(document)
            != _sha(pin["canonical_sha256"], f"{filename} canonical SHA")
        ):
            raise MaterializationError(
                f"formal output {filename} bytes/canonical hash drifted"
            )
        loaded[filename] = document
    proposal = loaded["formal_trust_pin_proposal.json"]
    proposal = _exact_dict(
        proposal,
        (
            "schema_version",
            "kind",
            "phase",
            "pins",
            "manifest_bindings",
            "sources",
            "authorization_granted",
        ),
        "formal trust proposal",
    )
    pins = proposal["pins"]
    if (
        proposal["schema_version"] != 1
        or proposal["kind"] != TRUST_PROPOSAL_KIND
        or proposal["phase"] != "formal"
        or proposal["authorization_granted"] is not False
        or type(pins) is not dict
    ):
        raise MaterializationError("formal trust proposal pins changed")
    expected_manifest_bindings = {
        "strict_training_manifest": receipt["manifest"],
        "prototype": receipt["prototype"],
        "physical_task_bundle": receipt["physical_gate"]["bundle"],
        "disposable_physical_gate_manifest": receipt["physical_gate"][
            "manifest"
        ],
        "physical_gate_materialization_receipt": receipt["physical_gate"][
            "materialization_receipt"
        ],
    }
    if proposal["manifest_bindings"] != expected_manifest_bindings:
        raise MaterializationError(
            "formal trust proposal manifest crossbindings drifted"
        )
    for variable, source in (
        (PROMOTION_TRUST_NAME, PROMOTION_TRUST_SOURCE),
        (EVALUATOR_TRUST_NAME, EVALUATOR_TRUST_SOURCE),
        (SIDECAR_CODE_TRUST_NAME, INBOX_SOURCE),
        (SIDECAR_LAUNCH_TRUST_NAME, INBOX_SOURCE),
        (DRAIN_TRUST_NAME, CURRICULUM_TRUST_SOURCE),
    ):
        _require_singleton_trust(
            root,
            source,
            variable,
            _sha(pins.get(variable), f"trust proposal {variable}"),
        )
    admission_document = loaded["motion_admission_receipt.json"]
    declared = admission_document.get("canonical_sha256")
    unsigned = dict(admission_document)
    unsigned.pop("canonical_sha256", None)
    if declared != canonical_sha256(unsigned):
        raise MaterializationError(
            "static motion-admission canonical SHA drifted"
        )
    if (
        admission_document.get("promotion_certificate_sha256")
        != _sha256_bytes(
            (output_dir / "promotion_certificate.json").read_bytes()
        )
    ):
        raise MaterializationError(
            "static admission does not bind promotion certificate bytes"
        )
    registry_module = _registry_module(root)
    registry_sha = _sha256_bytes(
        (output_dir / "canonical_registry.json").read_bytes()
    )
    registry = registry_module.load_canonical_motion_bank_registry(
        output_dir / "canonical_registry.json",
        repo_root=root,
        expected_registry_sha256=registry_sha,
    )
    if (
        registry.motion_ids != ACTION_ORDER
        or registry.scope != SCOPE
        or registry_module._alignment_sha256(registry)
        != receipt["registry_alignment_sha256"]
    ):
        raise MaterializationError(
            "verified registry order/scope/alignment changed"
        )
    # Rebuild from the pinned spec and sidecar receipt.  This repeats the full
    # base/append, fitted-ball, and Isaac promotion-evidence validation.
    spec_pin = _pin(root, receipt["materialization_spec"], "verified spec")
    _, spec_path = _repo_member(root, spec_pin["path"], "verified spec path")
    sidecar_pin = _pin(
        root,
        receipt["sidecar_launch_receipt"],
        "verified sidecar receipt",
    )
    _, sidecar_path = _repo_member(
        root, sidecar_pin["path"], "verified sidecar path"
    )
    sidecar_document, _ = _strict_json(
        sidecar_path, "verified sidecar receipt"
    )
    rebuilt = _formal_documents(
        root, spec_path, sidecar_document, sidecar_pin
    )
    for filename, expected in rebuilt.items():
        if filename == "formal_materialization_receipt.json":
            expected = {
                **expected,
                "outputs": receipt["outputs"],
            }
        actual = (
            receipt
            if filename == "formal_materialization_receipt.json"
            else loaded[filename]
        )
        if actual != expected:
            raise MaterializationError(
                f"formal output {filename} is not reproducible"
            )
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase": "verify",
                "artifact_dir": str(output_dir),
                "launch_authority": (
                    "trust closure verified; launcher plan remains the "
                    "separate final authority"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sidecar = subparsers.add_parser(
        "sidecar", help="materialize sidecar receipt and inbox trust proposal"
    )
    sidecar.add_argument("--repo-root", required=True)
    sidecar.add_argument("--out-dir", required=True)
    sidecar.set_defaults(handler=_sidecar_command)

    physical_gate = subparsers.add_parser(
        "physical-gate",
        help=(
            "materialize a disposable fitted-gate manifest while preserving "
            "the strict training manifest"
        ),
    )
    physical_gate.add_argument("--repo-root", required=True)
    physical_gate.add_argument("--base-manifest", required=True)
    physical_gate.add_argument("--base-manifest-sha256", required=True)
    physical_gate.add_argument("--physical-task-bundle", required=True)
    physical_gate.add_argument(
        "--physical-task-bundle-sha256", required=True
    )
    physical_gate.add_argument("--base-build-manifest", required=True)
    physical_gate.add_argument(
        "--base-build-manifest-sha256", required=True
    )
    physical_gate.add_argument("--append-build-manifest", required=True)
    physical_gate.add_argument(
        "--append-build-manifest-sha256", required=True
    )
    physical_gate.add_argument("--base-bank-gate", required=True)
    physical_gate.add_argument("--base-bank-gate-sha256", required=True)
    physical_gate.add_argument("--append-bank-gate", required=True)
    physical_gate.add_argument("--append-bank-gate-sha256", required=True)
    physical_gate.add_argument("--out-dir", required=True)
    physical_gate.set_defaults(handler=_physical_gate_command)

    formal = subparsers.add_parser(
        "formal",
        help=(
            "materialize registry/promotion/admission/evaluator/drain after "
            "sidecar singleton pins"
        ),
    )
    formal.add_argument("--repo-root", required=True)
    formal.add_argument("--spec", required=True)
    formal.add_argument("--sidecar-receipt", required=True)
    formal.add_argument("--sidecar-receipt-sha256", required=True)
    formal.add_argument("--out-dir", required=True)
    formal.set_defaults(handler=_formal_command)

    verify = subparsers.add_parser(
        "verify", help="re-open artifacts and require exact code trust pins"
    )
    verify.add_argument("--repo-root", required=True)
    verify.add_argument("--artifact-dir", required=True)
    verify.set_defaults(handler=_verify_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except MaterializationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
