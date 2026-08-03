#!/usr/bin/env python3
"""Build a commit-required canonical lineage for one A225 fixed question.

This producer is deliberately not a launcher.  It validates the exact fresh
bundle/tape/manifest closure and atomically writes one canonical lineage JSON.
The A225 launcher remains the final authority: until this output and every pin
inside it are committed, its tracked-file gate rejects them.
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
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER_FILE = SCRIPT_DIR / "launch_action_ball_a225_four_arm_diagnostic.py"
SHA256_HEX = frozenset("0123456789abcdef")


class MaterializationError(RuntimeError):
    """An A225 lineage input or publication boundary was invalid."""


def _load_launcher():
    spec = importlib.util.spec_from_file_location("_a225_lineage_launcher", LAUNCHER_FILE)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot import A225 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_L = _load_launcher()


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
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
    if type(value) is not str or len(value) != 64 or any(c not in SHA256_HEX for c in value):
        raise MaterializationError("%s must be one lowercase SHA-256" % name)
    return value


def _commit(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 40 or any(c not in SHA256_HEX for c in value):
        raise MaterializationError("%s must be a 40-character lowercase Git commit" % name)
    return value


def _relative(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise MaterializationError("%s must be a non-empty POSIX relative path" % name)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise MaterializationError("%s must be a normalized relative path" % name)
    return path.as_posix()


def _regular(path: Path, *, name: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise MaterializationError("cannot inspect %s: %s" % (name, exc)) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise MaterializationError("%s must be a regular non-symlink file" % name)


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def unique(rows):
        output = {}
        for key, value in rows:
            if key in output:
                raise MaterializationError("%s contains duplicate key %r" % (name, key))
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MaterializationError("%s contains non-finite %s" % (name, token))
            ),
        )
    except MaterializationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError("%s is not strict UTF-8 JSON" % name) from exc
    if type(value) is not dict:
        raise MaterializationError("%s root must be an object" % name)
    if raw != canonical_bytes(value) + b"\n":
        raise MaterializationError("%s must be canonical JSON plus newline" % name)
    return value


def _repo_root(value: str) -> Path:
    root = Path(value).resolve(strict=True)
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        raise MaterializationError("--repo-root must be the Git worktree root")
    return root


def _git(root: Path, args: Sequence[str], *, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=False,
        capture_output=True, text=text,
    )


def _pin(path: str, digest: str, *, name: str) -> dict[str, str]:
    return {"path": _relative(path, name=name + ".path"), "sha256": _sha(digest, name=name + ".sha256")}


def _input(
    root: Path, *, path: str, digest: str, name: str, explicit: bool, source_commit: str
) -> tuple[dict[str, str], dict[str, Any], str]:
    pin = _pin(path, digest, name=name)
    candidate = root / pin["path"]
    _regular(candidate, name=name)
    observed = sha256_file(candidate)
    if observed != pin["sha256"]:
        raise MaterializationError("%s file SHA differs" % name)
    if not explicit:
        tracked = _git(root, ("ls-files", "--stage", "--", pin["path"]))
        if tracked.returncode or not tracked.stdout.strip():
            raise MaterializationError("%s is not tracked; use its explicit input flag" % name)
        committed = _git(root, ("show", source_commit + ":" + pin["path"]), text=False)
        if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != pin["sha256"]:
            raise MaterializationError("%s differs from --source-commit" % name)
    document = _strict_json(candidate.read_bytes(), name=name)
    return pin, document, canonical_sha256(document)


def _tracked_input(root: Path, *, path: str, digest: str, name: str, source_commit: str) -> tuple[dict[str, str], dict[str, Any], str]:
    return _input(
        root, path=path, digest=digest, name=name, explicit=False,
        source_commit=source_commit,
    )


def _tracked_file(root: Path, *, path: str, digest: str, name: str, source_commit: str) -> dict[str, str]:
    pin = _pin(path, digest, name=name)
    candidate = root / pin["path"]
    _regular(candidate, name=name)
    if sha256_file(candidate) != pin["sha256"]:
        raise MaterializationError("%s file SHA differs" % name)
    tracked = _git(root, ("ls-files", "--stage", "--", pin["path"]))
    if tracked.returncode or not tracked.stdout.strip():
        raise MaterializationError("%s is not tracked" % name)
    committed = _git(root, ("show", source_commit + ":" + pin["path"]), text=False)
    if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != pin["sha256"]:
        raise MaterializationError("%s differs from --source-commit" % name)
    return pin


def _require_seal(document: Mapping[str, Any], key: str, *, name: str) -> str:
    seal = _sha(document.get(key), name=name + "." + key)
    unsigned = dict(document)
    unsigned.pop(key)
    if canonical_sha256(unsigned) != seal:
        raise MaterializationError("%s %s is not reproducible" % (name, key))
    return seal


def _bundle_semantics(bundle: Mapping[str, Any], tape: Mapping[str, str], motion: Mapping[str, str]):
    if bundle.get("schema_version") != 1 or bundle.get("artifact_type") != "measured_action_ball_n1_diagnostic_bundle_v1":
        raise MaterializationError("bundle kind/schema differs")
    action_id = bundle.get("action_id")
    teacher_id = bundle.get("measured_uid")
    action_uid = bundle.get("action_uid")
    if type(action_id) is not str or type(teacher_id) is not str or type(action_uid) is not int:
        raise MaterializationError("bundle action identity is malformed")
    if bundle.get("target_recipe") != "current_lm":
        raise MaterializationError("bundle must use current_lm")
    validity = bundle.get("target_validity")
    if type(validity) is not dict or validity.get("order") != ["position", "velocity", "face"] or validity.get("mask") != [True, True, True]:
        raise MaterializationError("bundle target validity differs")
    if bundle.get("immutable_tape") != tape or bundle.get("motion") != motion:
        raise MaterializationError("bundle tape/motion pin differs")
    claims = bundle.get("claims")
    runtime = bundle.get("runtime_contract")
    if type(claims) is not dict or claims.get("diagnostic_unauthorized") is not True:
        raise MaterializationError("bundle is not diagnostic_unauthorized")
    if type(runtime) is not dict or runtime.get("physical_ball_semantics") != _L.PHYSICAL_BALL_SEMANTICS or runtime.get("reset_inverse_solve") is not False or runtime.get("target_source") != "immutable_tape":
        raise MaterializationError("bundle runtime semantics differ")
    return action_id, action_uid, teacher_id


def _manifest_semantics(manifest: Mapping[str, Any], *, action_id: str, action_uid: int, motion: Mapping[str, str]) -> tuple[str, str]:
    if manifest.get("schema_version") != 3 or manifest.get("action_order") != [action_id] or manifest.get("mobility_mode") != "no_move":
        raise MaterializationError("action manifest fixed-N1 identity differs")
    actions = manifest.get("actions")
    if type(actions) is not list or len(actions) != 1 or type(actions[0]) is not dict:
        raise MaterializationError("action manifest must contain exactly one action")
    action = actions[0]
    if action.get("action_id") != action_id or action.get("action_uid") != action_uid or action.get("motion_path") != motion["path"] or action.get("motion_sha256") != motion["sha256"]:
        raise MaterializationError("action manifest action pin differs")
    return (
        _sha(manifest.get("solver_profile_sha256"), name="manifest solver_profile_sha256"),
        _sha(manifest.get("physics_profile_sha256"), name="manifest physics_profile_sha256"),
    )


def _tape_semantics(tape: Mapping[str, Any], *, action_uid: int, motion: Mapping[str, str], physics_sha: str) -> str:
    if tape.get("schema_version") != 1 or tape.get("kind") != "action_ball_n1_immutable_single_question_tape" or tape.get("diagnostic_unauthorized") is not True:
        raise MaterializationError("immutable tape kind/schema/diagnostic status differs")
    seal = _require_seal(tape, "canonical_sha256", name="immutable tape")
    question = tape.get("question")
    if type(question) is not dict or question.get("action_uid") != action_uid or question.get("motion_sha256") != motion["sha256"] or question.get("physics_sha256") != physics_sha:
        raise MaterializationError("immutable tape question semantics differ")
    return seal


def _dynamic_semantics(document: Mapping[str, Any], *, action_id: str, motion: Mapping[str, str], nominal: bool) -> str:
    if document.get("action_id") != action_id or document.get("motion_sha256", motion["sha256"]) != motion["sha256"]:
        raise MaterializationError("dynamic-ready input action/motion differs")
    if nominal:
        if document.get("verdict") != "PASS":
            raise MaterializationError("nominal-hold receipt verdict differs")
    else:
        if document.get("kind") != "agibot_a3_action_dynamic_ready_candidate_v2":
            raise MaterializationError("dynamic-ready artifact kind differs")
        teacher = document.get("teacher_reference")
        if type(teacher) is not dict or teacher.get("motion_sha256") != motion["sha256"]:
            raise MaterializationError("dynamic-ready teacher motion differs")
    return _require_seal(document, "content_sha256", name="dynamic-ready input")


def _write_new(root: Path, relative: str, raw: bytes) -> dict[str, str]:
    relative = _relative(relative, name="output")
    check = _git(root, ("check-ignore", "-q", "--no-index", "--", relative))
    if check.returncode == 0:
        raise MaterializationError("output must not be Git-ignored")
    if check.returncode not in (0, 1):
        raise MaterializationError("cannot inspect output ignore policy")
    output = root / relative
    if output.exists() or output.is_symlink():
        raise MaterializationError("no-clobber output already exists: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise MaterializationError("output parent escaped repo root") from exc
    if parent != output.parent:
        raise MaterializationError("output parent must not traverse a symlink")
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".a225-lineage-", dir=str(parent))
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:  # pragma: no cover - race protection
            raise MaterializationError("no-clobber output already exists: %s" % output) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    _regular(output, name="output")
    return {"path": relative, "sha256": sha256_file(output)}


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root(args.repo_root)
    source_commit = _commit(args.source_commit, name="source_commit")
    if _git(root, ("rev-parse", "--verify", source_commit + "^{commit}")).returncode:
        raise MaterializationError("source_commit is not a Git commit")
    tape_pin, tape, tape_content_sha = _input(
        root, path=args.immutable_tape_path, digest=args.expected_immutable_tape_sha256,
        name="immutable tape", explicit=args.immutable_tape_explicit, source_commit=source_commit,
    )
    bundle_pin, bundle, bundle_content_sha = _input(
        root, path=args.bundle_path, digest=args.expected_bundle_sha256,
        name="bundle", explicit=args.bundle_explicit, source_commit=source_commit,
    )
    manifest_pin, manifest, manifest_content_sha = _input(
        root, path=args.action_manifest_path, digest=args.expected_action_manifest_sha256,
        name="action manifest", explicit=args.action_manifest_explicit, source_commit=source_commit,
    )
    motion = _tracked_file(
        root, path=args.motion_path, digest=args.expected_motion_sha256,
        name="motion", source_commit=source_commit,
    )
    dynamic_ready, dynamic_doc, _dynamic_doc_content = _tracked_input(
        root, path=args.dynamic_ready_artifact_path, digest=args.expected_dynamic_ready_artifact_sha256,
        name="dynamic-ready artifact", source_commit=source_commit,
    )
    nominal_hold, hold_doc, _hold_doc_content = _tracked_input(
        root, path=args.dynamic_ready_nominal_receipt_path, digest=args.expected_dynamic_ready_nominal_receipt_sha256,
        name="dynamic-ready nominal-hold receipt", source_commit=source_commit,
    )
    action_id, action_uid, teacher_id = _bundle_semantics(bundle, tape_pin, motion)
    solver_sha, physics_sha = _manifest_semantics(manifest, action_id=action_id, action_uid=action_uid, motion=motion)
    tape_semantic_sha = _tape_semantics(tape, action_uid=action_uid, motion=motion, physics_sha=physics_sha)
    dynamic_content_sha = _dynamic_semantics(dynamic_doc, action_id=action_id, motion=motion, nominal=False)
    hold_content_sha = _dynamic_semantics(hold_doc, action_id=action_id, motion=motion, nominal=True)
    lineage = {
        "schema_version": 1, "kind": _L.LINEAGE_KIND,
        "actor_contract": _L.ACTOR_CONTRACT, "actor_width": _L.ACTOR_WIDTH,
        "critic_contract": _L.CRITIC_CONTRACT, "critic_width": _L.CRITIC_WIDTH,
        "task_profile": _L.TASK_PROFILE_ID, "gym_task": _L.GYM_TASK_ID,
        "target_semantics": _L.TARGET_SEMANTICS, "action_id": action_id,
        "teacher_id": teacher_id, "seed": 0, "bundle": bundle_pin,
        "motion": motion, "immutable_tape": tape_pin,
        "action_manifest": manifest_pin, "dynamic_ready_artifact": dynamic_ready,
        "dynamic_ready_nominal_receipt": nominal_hold,
    }
    lineage_pin = _write_new(root, args.output, canonical_bytes(lineage) + b"\n")
    semantic = {
        "bundle_content_sha256": bundle_content_sha,
        "tape_canonical_sha256": tape_semantic_sha,
        "manifest_content_sha256": manifest_content_sha,
        "solver_profile_sha256": solver_sha, "physics_profile_sha256": physics_sha,
        "dynamic_ready_content_sha256": dynamic_content_sha,
        "nominal_hold_content_sha256": hold_content_sha,
    }
    return {
        "status": "MATERIALIZED_COMMIT_REQUIRED", "diagnostic_unauthorized": True,
        "launch_authorized": False, "lineage": lineage_pin,
        "lineage_content_sha256": canonical_sha256(lineage),
        "semantic_sha256": canonical_sha256(semantic), "source_commit": source_commit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit", required=True)
    for name in ("bundle", "immutable-tape", "action-manifest"):
        parser.add_argument("--%s-path" % name, required=True)
        parser.add_argument("--expected-%s-sha256" % name, required=True)
        parser.add_argument("--%s-explicit" % name, action="store_true")
    parser.add_argument("--motion-path", required=True)
    parser.add_argument("--expected-motion-sha256", required=True)
    parser.add_argument("--dynamic-ready-artifact-path", required=True)
    parser.add_argument("--expected-dynamic-ready-artifact-sha256", required=True)
    parser.add_argument("--dynamic-ready-nominal-receipt-path", required=True)
    parser.add_argument("--expected-dynamic-ready-nominal-receipt-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
