#!/usr/bin/env python3
"""Materialize a commit-required, C211-only direct-ball lineage.

The producer has two input modes.  It can consume an already tracked canonical
C211 bundle, or it can deterministically build that bundle from the exact
tracked source pins.  In both modes every source byte is checked against the
given clean full commit and every bundle pin is cross-checked.  Publication is
canonical and no-clobber; it never authorizes a launch.  The existing C211
launcher remains the final authority after the bundle and lineage are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER_FILE = SCRIPT_DIR / "launch_action_ball_c211_diagnostic.py"
SHA256_HEX = frozenset("0123456789abcdef")

BUNDLE_KEYS = (
    "schema_version",
    "kind",
    "diagnostic_unauthorized",
    "action_id",
    "action_uid",
    "teacher_id",
    "actor_contract",
    "actor_width",
    "critic_contract",
    "critic_width",
    "trainability_contract",
    "actor_normalizer_identity",
    "critic_normalizer_identity",
    "target_source",
    "question_source",
    "question_rng",
    "target_recipe",
    "curriculum_scope",
    "target_validity_mask",
    "incoming_ball_fields",
    "reset_inverse_solve",
    "online_solver_calls",
    "online_lm_calls",
    "motion",
    "action_manifest",
    "initial_center_task_receipt",
    "dynamic_ready_artifact",
    "dynamic_ready_nominal_receipt",
    "teacher_frame0_artifact",
    "dr_l0_manifest",
)

class MaterializationError(RuntimeError):
    """A C211 source, bundle, or publication boundary was invalid."""


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "_c211_lineage_materializer_launcher", LAUNCHER_FILE
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot import C211 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_L = _load_launcher()


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MaterializationError("value is not canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise MaterializationError("%s must be one lowercase SHA-256" % name)
    return value


def _commit(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in SHA256_HEX for character in value)
    ):
        raise MaterializationError(
            "%s must be a 40-character lowercase Git commit" % name
        )
    return value


def _relative(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise MaterializationError(
            "%s must be a non-empty POSIX relative path" % name
        )
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(
        part in ("", ".", "..") for part in candidate.parts
    ):
        raise MaterializationError("%s must be a normalized relative path" % name)
    return candidate.as_posix()


def _regular(path: Path, *, name: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise MaterializationError("cannot inspect %s: %s" % (name, exc)) from exc
    if not stat.S_ISREG(info.st_mode):
        raise MaterializationError("%s must be a regular non-symlink file" % name)
    if path.resolve(strict=True) != path:
        raise MaterializationError("%s must not resolve through a symlink" % name)


def _strict_json(
    raw: bytes, *, name: str, require_canonical_bytes: bool = False
) -> dict[str, Any]:
    def unique(rows):
        output = {}
        for key, value in rows:
            if key in output:
                raise MaterializationError(
                    "%s contains duplicate key %r" % (name, key)
                )
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MaterializationError(
                    "%s contains non-finite %s" % (name, token)
                )
            ),
        )
    except MaterializationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError("%s is not strict UTF-8 JSON" % name) from exc
    if type(value) is not dict:
        raise MaterializationError("%s root must be an object" % name)
    if require_canonical_bytes and raw != canonical_bytes(value) + b"\n":
        raise MaterializationError(
            "%s must be canonical JSON plus newline" % name
        )
    return value


def _git(root: Path, args: Sequence[str], *, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=text,
    )


def _repo_root(value: str) -> Path:
    root = Path(value).resolve(strict=True)
    result = _git(root, ("rev-parse", "--show-toplevel"))
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        raise MaterializationError("--repo-root must be the Git worktree root")
    return root


def _pin(path: str, digest: str, *, name: str) -> dict[str, str]:
    return {
        "path": _relative(path, name=name + ".path"),
        "sha256": _sha(digest, name=name + ".sha256"),
    }


def _verify_clean_source(
    root: Path, source_commit: str, *, allowed_outputs: Sequence[str]
) -> None:
    head = _git(root, ("rev-parse", "--verify", "HEAD"))
    if head.returncode or head.stdout.strip() != source_commit:
        raise MaterializationError("--source-commit must equal the checkout HEAD")
    allowed = frozenset(
        _relative(path, name="output") for path in allowed_outputs
    )
    status = _git(
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    )
    if status.returncode:
        raise MaterializationError("cannot inspect source cleanliness")
    for entry in status.stdout.split("\x00"):
        if not entry:
            continue
        if len(entry) < 4:
            raise MaterializationError("cannot parse source cleanliness")
        code = entry[:2]
        relative = entry[3:]
        # Outputs are commit-required products, but they must remain untracked
        # while this producer is allowed to validate or reproduce them.  In
        # particular, accepting an index addition here would let an ``AM``
        # state carry different staged bytes than the correct working-tree
        # publication; the next plain commit would then publish the wrong
        # artifact.  Requiring ``??`` makes the index boundary unambiguous.
        if relative not in allowed or code != "??":
            raise MaterializationError("source checkout is dirty: %s" % relative)


def _tracked_file(
    root: Path,
    *,
    path: str,
    digest: str,
    name: str,
    source_commit: str,
) -> tuple[dict[str, str], Path]:
    pin = _pin(path, digest, name=name)
    candidate = root / pin["path"]
    _regular(candidate, name=name)
    stage = _git(root, ("ls-files", "--stage", "--", pin["path"]))
    rows = stage.stdout.splitlines() if stage.returncode == 0 else []
    if len(rows) != 1:
        raise MaterializationError("%s is not one tracked Git blob" % name)
    parts = rows[0].split(None, 3)
    if len(parts) != 4 or parts[0] not in ("100644", "100755"):
        raise MaterializationError("%s must be a normal tracked Git blob" % name)
    committed = _git(
        root, ("show", source_commit + ":" + pin["path"]), text=False
    )
    committed_sha = (
        hashlib.sha256(committed.stdout).hexdigest()
        if committed.returncode == 0
        else ""
    )
    observed_sha = sha256_file(candidate)
    if committed_sha != pin["sha256"] or observed_sha != pin["sha256"]:
        raise MaterializationError(
            "%s SHA differs: pin=%s commit=%s worktree=%s"
            % (name, pin["sha256"], committed_sha, observed_sha)
        )
    return pin, candidate


def _tracked_json(
    root: Path,
    *,
    path: str,
    digest: str,
    name: str,
    source_commit: str,
    require_canonical_bytes: bool = False,
) -> tuple[dict[str, str], Path, dict[str, Any]]:
    pin, candidate = _tracked_file(
        root,
        path=path,
        digest=digest,
        name=name,
        source_commit=source_commit,
    )
    document = _strict_json(
        candidate.read_bytes(),
        name=name,
        require_canonical_bytes=require_canonical_bytes,
    )
    return pin, candidate, document


def _launcher_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (_L.LaunchRefused, _L._FRAME0.LaunchRefused) as exc:
        raise MaterializationError(str(exc)) from exc


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _require_seal(document: Mapping[str, Any], *, name: str) -> str:
    value = _sha(document.get("content_sha256"), name=name + ".content_sha256")
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    if canonical_sha256(unsigned) != value:
        raise MaterializationError("%s content seal differs" % name)
    return value


def _expected_bundle(
    pins: Mapping[str, Mapping[str, str]],
    *,
    dr_l0_manifest: Mapping[str, str],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 4,
        "kind": _L.C211_BUNDLE_KIND,
        "diagnostic_unauthorized": True,
        "action_id": _L.ACTION_ID,
        "action_uid": _L.ACTION_UID,
        "teacher_id": _L.TEACHER_ID,
        "actor_contract": _L.ACTOR_CONTRACT,
        "actor_width": _L.ACTOR_WIDTH,
        "critic_contract": _L.CRITIC_CONTRACT,
        "critic_width": _L.CRITIC_WIDTH,
        "trainability_contract": _L.TRAINABILITY_CONTRACT,
        "actor_normalizer_identity": _L.ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": _L.CRITIC_NORMALIZER_IDENTITY,
        "target_source": _L.TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "question_rng": _L._question_rng_contract(),
        "target_recipe": _L.TARGET_RECIPE,
        "curriculum_scope": _L._curriculum_scope_contract(),
        "target_validity_mask": list(_L.TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(_L.INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "motion": dict(pins["motion"]),
        "action_manifest": dict(pins["action_manifest"]),
        "initial_center_task_receipt": dict(
            pins["initial_center_task_receipt"]
        ),
        "dynamic_ready_artifact": dict(pins["dynamic_ready_artifact"]),
        "dynamic_ready_nominal_receipt": dict(
            pins["dynamic_ready_nominal_receipt"]
        ),
        "teacher_frame0_artifact": dict(pins["teacher_frame0_artifact"]),
        "dr_l0_manifest": dict(dr_l0_manifest),
    }
    return _sealed(unsigned)


def _validate_manifest(
    manifest: Mapping[str, Any], *, motion: Mapping[str, str]
) -> None:
    actions = manifest.get("actions")
    action = actions[0] if type(actions) is list and len(actions) == 1 else None
    if (
        manifest.get("schema_version") != 3
        or manifest.get("action_order") != [_L.ACTION_ID]
        or manifest.get("mobility_mode") != "no_move"
        or type(action) is not dict
        or action.get("action_id") != _L.ACTION_ID
        or action.get("action_uid") != _L.ACTION_UID
        or action.get("motion_path") != motion["path"]
        or action.get("motion_sha256") != motion["sha256"]
    ):
        raise MaterializationError("action manifest C211 closure differs")


def _validate_dynamic_ready(
    *,
    dynamic: Mapping[str, Any],
    nominal: Mapping[str, Any],
    motion: Mapping[str, str],
    dynamic_pin: Mapping[str, str],
    nominal_pin: Mapping[str, str],
    teacher_artifact: Mapping[str, Any],
    initial_center_receipt: Mapping[str, Any],
    initial_center_receipt_pin: Mapping[str, str],
    action_manifest: Mapping[str, Any],
    action_manifest_pin: Mapping[str, str],
    motion_path: Path,
) -> dict[str, Any]:
    dynamic_content_sha = _require_seal(dynamic, name="dynamic-ready artifact")
    nominal_content_sha = _require_seal(
        nominal, name="dynamic-ready raw nominal receipt"
    )
    if (
        dynamic.get("kind") != "agibot_a3_action_dynamic_ready_candidate_v2"
        or dynamic.get("action_id") != _L.ACTION_ID
        or nominal.get("action_id") != _L.ACTION_ID
        or nominal.get("motion_sha256") != motion["sha256"]
        or nominal.get("verdict") != "PASS"
    ):
        raise MaterializationError("dynamic-ready C211 closure differs")
    teacher = _launcher_call(
        _L._FRAME0._validate_teacher_frame0_artifact,
        teacher_artifact,
        motion_path=motion_path,
        motion_sha256=motion["sha256"],
    )
    timing = _launcher_call(
        _L._FRAME0._initial_center_timing_authority,
        receipt=initial_center_receipt,
        receipt_pin=initial_center_receipt_pin,
        action_manifest=action_manifest,
        action_manifest_pin=action_manifest_pin,
        motion_sha256=motion["sha256"],
        family="C",
    )
    authority = _launcher_call(
        _L._FRAME0._split_ready_reset_wait_semantics,
        dynamic=dynamic,
        nominal=nominal,
        dynamic_pin=dynamic_pin,
        nominal_pin=nominal_pin,
        teacher_frame0=teacher["frame0"],
        motion_sha256=motion["sha256"],
        initial_center_timing_authority=timing,
    )
    return {
        "dynamic_ready_content_sha256": dynamic_content_sha,
        "nominal_receipt_content_sha256": nominal_content_sha,
        "teacher_frame0_artifact_content_sha256": teacher["content_sha256"],
        "initial_center_timing_claim_sha256": timing["claim_sha256"],
        "split_ready_reset_wait_claim_sha256": authority["claim_sha256"],
    }



def _validate_bundle(
    bundle: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    _launcher_call(_L._assert_c211_only, bundle, name="C211 bundle")
    normalized = _launcher_call(
        _L._sealed_row, bundle, BUNDLE_KEYS, name="C211 bundle"
    )
    if normalized != expected:
        raise MaterializationError(
            "C211 bundle semantics or explicit source pin closure differs"
        )


def _destination(
    root: Path, relative: str, raw: bytes, *, source_commit: str, name: str
) -> tuple[dict[str, str], Path, bool]:
    relative = _relative(relative, name=name)
    ignored = _git(root, ("check-ignore", "-q", "--no-index", "--", relative))
    if ignored.returncode == 0:
        raise MaterializationError("%s must not be Git-ignored" % name)
    if ignored.returncode not in (0, 1):
        raise MaterializationError("cannot inspect %s ignore policy" % name)
    committed = _git(
        root, ("cat-file", "-e", source_commit + ":" + relative)
    )
    if committed.returncode == 0:
        raise MaterializationError(
            "%s already belongs to --source-commit" % name
        )

    output = root / relative
    parent = root
    missing_parent = False
    for component in PurePosixPath(relative).parts[:-1]:
        parent = parent / component
        if missing_parent:
            continue
        if parent.exists() or parent.is_symlink():
            try:
                info = parent.lstat()
            except OSError as exc:  # pragma: no cover - race protection
                raise MaterializationError(
                    "cannot inspect %s parent" % name
                ) from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise MaterializationError(
                    "%s parent must not traverse a symlink" % name
                )
        else:
            # Destination planning is read-only.  Parent directories are
            # created only after every source, digest, cross-output, and
            # no-clobber check has succeeded.
            missing_parent = True
    if not missing_parent and parent.resolve(strict=True) != parent:
        raise MaterializationError(
            "%s parent must not traverse a symlink" % name
        )
    if output.exists() or output.is_symlink():
        _regular(output, name=name)
        observed = output.read_bytes()
        if observed != raw:
            raise MaterializationError(
                "no-clobber %s already exists with different bytes: %s"
                % (name, output)
            )
        return (
            {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()},
            output,
            False,
        )
    return (
        {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()},
        output,
        True,
    )


def _prepare_parent(root: Path, output: Path, *, name: str) -> Path:
    """Create an already-validated destination parent without symlink traversal."""

    try:
        relative = output.relative_to(root)
    except ValueError as exc:  # pragma: no cover - _relative already prevents it
        raise MaterializationError("%s escaped the repository" % name) from exc
    parent = root
    for component in relative.parts[:-1]:
        parent = parent / component
        if not parent.exists() and not parent.is_symlink():
            try:
                parent.mkdir()
            except FileExistsError:
                # A concurrent claimant is inspected below instead of being
                # trusted as the directory we intended to create.
                pass
            except OSError as exc:
                raise MaterializationError(
                    "cannot create %s parent: %s" % (name, exc)
                ) from exc
        try:
            info = parent.lstat()
        except OSError as exc:  # pragma: no cover - race protection
            raise MaterializationError(
                "cannot inspect %s parent" % name
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise MaterializationError(
                "%s parent must not traverse a symlink" % name
            )
    if parent.resolve(strict=True) != parent:
        raise MaterializationError(
            "%s parent must not traverse a symlink" % name
        )
    return parent


def _publish(
    root: Path, output: Path, raw: bytes, *, create: bool, name: str
) -> None:
    if not create:
        return
    parent = _prepare_parent(root, output, name=name)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".c211-lineage-", dir=str(parent)
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.is_symlink() or not output.is_file() or output.read_bytes() != raw:
                raise MaterializationError(
                    "no-clobber %s was concurrently claimed" % name
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    _regular(output, name=name)


def _lineage(
    bundle: Mapping[str, str],
    pins: Mapping[str, Mapping[str, str]],
    *,
    dr_l0_manifest: Mapping[str, str],
):
    value = {
        "schema_version": 4,
        "kind": _L.LINEAGE_KIND,
        "actor_contract": _L.ACTOR_CONTRACT,
        "actor_width": _L.ACTOR_WIDTH,
        "critic_contract": _L.CRITIC_CONTRACT,
        "critic_width": _L.CRITIC_WIDTH,
        "trainability_contract": _L.TRAINABILITY_CONTRACT,
        "actor_normalizer_identity": _L.ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": _L.CRITIC_NORMALIZER_IDENTITY,
        "task_profile": _L.TASK_PROFILE_ID,
        "gym_task": _L.GYM_TASK_ID,
        "target_semantics": _L.TARGET_SEMANTICS,
        "curriculum_scope": _L._curriculum_scope_contract(),
        "target_source": _L.TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "question_rng": _L._question_rng_contract(),
        "target_recipe": _L.TARGET_RECIPE,
        "target_validity_mask": list(_L.TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(_L.INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "action_id": _L.ACTION_ID,
        "action_uid": _L.ACTION_UID,
        "teacher_id": _L.TEACHER_ID,
        "seed": 0,
        "bundle": dict(bundle),
        "motion": dict(pins["motion"]),
        "action_manifest": dict(pins["action_manifest"]),
        "initial_center_task_receipt": dict(
            pins["initial_center_task_receipt"]
        ),
        "dynamic_ready_artifact": dict(pins["dynamic_ready_artifact"]),
        "dynamic_ready_nominal_receipt": dict(
            pins["dynamic_ready_nominal_receipt"]
        ),
        "teacher_frame0_artifact": dict(pins["teacher_frame0_artifact"]),
        "dr_l0_manifest": dict(dr_l0_manifest),
    }
    _launcher_call(_L._assert_c211_only, value, name="C211 lineage")
    return value


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root(args.repo_root)
    source_commit = _commit(args.source_commit, name="source_commit")
    resolved = _git(
        root, ("rev-parse", "--verify", source_commit + "^{commit}")
    )
    if resolved.returncode or resolved.stdout.strip() != source_commit:
        raise MaterializationError("source_commit is not an exact Git commit")
    generated_bundle = args.bundle_output is not None
    if generated_bundle and args.expected_bundle_sha256 is not None:
        _sha(args.expected_bundle_sha256, name="expected bundle SHA-256")
    if not generated_bundle and args.expected_bundle_sha256 is None:
        raise MaterializationError(
            "--expected-bundle-sha256 is required with --bundle-path"
        )
    allowed_outputs = [args.output]
    if generated_bundle:
        allowed_outputs.append(args.bundle_output)
    _verify_clean_source(
        root, source_commit, allowed_outputs=allowed_outputs
    )
    _launcher_call(_L._verify_c211_runtime_authorities, root)

    pins = {}
    paths = {}
    documents = {}
    sources = (
        (
            "action_manifest",
            args.action_manifest_path,
            args.expected_action_manifest_sha256,
            "action manifest",
        ),
        (
            "initial_center_task_receipt",
            args.initial_center_task_receipt_path,
            args.expected_initial_center_task_receipt_sha256,
            "initial-center C task receipt",
        ),
        (
            "dynamic_ready_artifact",
            args.dynamic_ready_artifact_path,
            args.expected_dynamic_ready_artifact_sha256,
            "dynamic-ready artifact",
        ),
        (
            "dynamic_ready_nominal_receipt",
            args.dynamic_ready_nominal_receipt_path,
            args.expected_dynamic_ready_nominal_receipt_sha256,
            "dynamic-ready raw nominal receipt",
        ),
        (
            "teacher_frame0_artifact",
            args.teacher_frame0_artifact_path,
            args.expected_teacher_frame0_artifact_sha256,
            "teacher-frame0 artifact",
        ),
    )
    for key, path, digest, name in sources:
        pin, resolved_path, document = _tracked_json(
            root,
            path=path,
            digest=digest,
            name=name,
            source_commit=source_commit,
        )
        pins[key] = pin
        paths[key] = resolved_path
        documents[key] = document
    pins["motion"], paths["motion"] = _tracked_file(
        root,
        path=args.motion_path,
        digest=args.expected_motion_sha256,
        name="motion",
        source_commit=source_commit,
    )
    if (
        pins["dynamic_ready_artifact"]["sha256"]
        != _L._FRAME0.SPLIT_READY_DYNAMIC_ARTIFACT_SHA256
        or pins["dynamic_ready_nominal_receipt"]["sha256"]
        != _L._FRAME0.SPLIT_READY_NOMINAL_HOLD_SHA256
        or pins["teacher_frame0_artifact"]["sha256"]
        != _L._FRAME0.SPLIT_READY_TEACHER_FRAME0_ARTIFACT_SHA256
    ):
        raise MaterializationError("split-ready authority bytes differ")

    _validate_manifest(documents["action_manifest"], motion=pins["motion"])
    dynamic_semantics = _validate_dynamic_ready(
        dynamic=documents["dynamic_ready_artifact"],
        nominal=documents["dynamic_ready_nominal_receipt"],
        motion=pins["motion"],
        dynamic_pin=pins["dynamic_ready_artifact"],
        nominal_pin=pins["dynamic_ready_nominal_receipt"],
        teacher_artifact=documents["teacher_frame0_artifact"],
        initial_center_receipt=documents["initial_center_task_receipt"],
        initial_center_receipt_pin=pins["initial_center_task_receipt"],
        action_manifest=documents["action_manifest"],
        action_manifest_pin=pins["action_manifest"],
        motion_path=paths["motion"],
    )
    dr_l0_manifest = _launcher_call(
        _L._FRAME0._dr_l0_manifest_binding,
        root,
        source_commit,
        family="C",
        task_profile=_L.TASK_PROFILE_ID,
    )
    expected_bundle = _expected_bundle(
        pins, dr_l0_manifest=dr_l0_manifest
    )
    bundle_raw = canonical_bytes(expected_bundle) + b"\n"

    destinations = []
    if generated_bundle:
        bundle_pin, bundle_path, bundle_create = _destination(
            root,
            args.bundle_output,
            bundle_raw,
            source_commit=source_commit,
            name="bundle output",
        )
        if (
            args.expected_bundle_sha256 is not None
            and bundle_pin["sha256"] != args.expected_bundle_sha256
        ):
            raise MaterializationError("generated bundle SHA differs from expected")
        destinations.append(
            (bundle_path, bundle_raw, bundle_create, "bundle output")
        )
    else:
        bundle_pin, _bundle_path, bundle_document = _tracked_json(
            root,
            path=args.bundle_path,
            digest=args.expected_bundle_sha256,
            name="C211 bundle",
            source_commit=source_commit,
            require_canonical_bytes=True,
        )
        _validate_bundle(bundle_document, expected_bundle)

    lineage = _lineage(
        bundle_pin, pins, dr_l0_manifest=dr_l0_manifest
    )
    lineage_raw = canonical_bytes(lineage) + b"\n"
    lineage_pin, lineage_path, lineage_create = _destination(
        root,
        args.output,
        lineage_raw,
        source_commit=source_commit,
        name="lineage output",
    )
    if bundle_pin["path"] == lineage_pin["path"]:
        raise MaterializationError("bundle and lineage outputs must differ")
    destinations.append(
        (lineage_path, lineage_raw, lineage_create, "lineage output")
    )
    for output, raw, create, name in destinations:
        _publish(root, output, raw, create=create, name=name)

    semantic = {
        "bundle_content_sha256": expected_bundle["content_sha256"],
        "target_source": _L.TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "question_rng_sha256": canonical_sha256(_L._question_rng_contract()),
        "curriculum_scope_sha256": canonical_sha256(
            _L._curriculum_scope_contract()
        ),
        "dr_l0_contract_sha256": dr_l0_manifest["contract_sha256"],
        **dynamic_semantics,
    }
    return {
        "status": "MATERIALIZED_COMMIT_REQUIRED",
        "diagnostic_unauthorized": True,
        "launch_authorized": False,
        "source_commit": source_commit,
        "bundle_mode": "materialized" if generated_bundle else "tracked",
        "bundle": bundle_pin,
        "bundle_content_sha256": expected_bundle["content_sha256"],
        "lineage": lineage_pin,
        "lineage_content_sha256": canonical_sha256(lineage),
        "semantic_sha256": canonical_sha256(semantic),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit", required=True)
    bundle = parser.add_mutually_exclusive_group(required=True)
    bundle.add_argument("--bundle-path")
    bundle.add_argument("--bundle-output")
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--motion-path", required=True)
    parser.add_argument("--expected-motion-sha256", required=True)
    parser.add_argument("--action-manifest-path", required=True)
    parser.add_argument("--expected-action-manifest-sha256", required=True)
    parser.add_argument("--initial-center-task-receipt-path", required=True)
    parser.add_argument(
        "--expected-initial-center-task-receipt-sha256", required=True
    )
    parser.add_argument("--dynamic-ready-artifact-path", required=True)
    parser.add_argument(
        "--expected-dynamic-ready-artifact-sha256", required=True
    )
    parser.add_argument(
        "--dynamic-ready-nominal-receipt-path",
        "--dynamic-ready-raw-nominal-receipt-path",
        dest="dynamic_ready_nominal_receipt_path",
        required=True,
    )
    parser.add_argument(
        "--expected-dynamic-ready-nominal-receipt-sha256",
        "--expected-dynamic-ready-raw-nominal-receipt-sha256",
        dest="expected_dynamic_ready_nominal_receipt_sha256",
        required=True,
    )
    parser.add_argument("--teacher-frame0-artifact-path", required=True)
    parser.add_argument(
        "--expected-teacher-frame0-artifact-sha256", required=True
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize(args)
    except MaterializationError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
