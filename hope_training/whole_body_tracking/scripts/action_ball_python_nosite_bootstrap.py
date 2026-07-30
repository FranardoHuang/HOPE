#!/usr/bin/env python3
"""Exact, stdlib-only, no-site Python entrypoint bootstrap.

The public builder emits one canonical command shape::

    python -I -B -S -c <TRAMPOLINE> \
        <bootstrap-path> <bootstrap-sha256> \
        <argv-contract-sha256> <canonical-contract-base64>

The trampoline opens the bootstrap with ``O_NOFOLLOW``, hashes the descriptor
bytes, checks that the path still names the same regular file, and only then
compiles and executes those bytes.  The bootstrap repeats that discipline for
the requested entrypoint, hashes every regular file below each preregistered
import root, appends the roots directly to ``sys.path``, and executes the
entrypoint.  It never imports ``site``, calls ``site.addsitedir``, or interprets
``.pth`` files.

This module is intentionally Python 3.8 and standard-library only so the same
bytes can be used by the launcher, supervisor, inventory, evaluator, and
exact-resume verifier before any project or GPU package is importable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple


CONTRACT_SCHEMA_VERSION = 1
CONTRACT_KIND = "action_ball_python_nosite_argv_contract_v1"
ATTESTATION_KIND = "action_ball_python_nosite_execution_v1"
SHA256_CHARS = frozenset("0123456789abcdef")

# This source is an exact argv token.  Keep it self-contained: it runs before
# the bootstrap file has been trusted and therefore may use only interpreter
# builtins and explicitly imported stdlib modules.
TRAMPOLINE_SOURCE = r"""import hashlib,os,stat,sys
if not (sys.flags.isolated and sys.flags.no_site and sys.flags.no_user_site and sys.flags.ignore_environment and sys.flags.dont_write_bytecode and sys.flags.optimize == 0):
    raise SystemExit("ACTION_BALL_NOSITE: interpreter flags are not exact")
if len(sys.argv) != 5:
    raise SystemExit("ACTION_BALL_NOSITE: trampoline argv is not exact")
p,e,c,b=sys.argv[1:]
if not os.path.isabs(p) or os.path.normpath(p) != p or len(e) != 64 or any(x not in "0123456789abcdef" for x in e):
    raise SystemExit("ACTION_BALL_NOSITE: bootstrap binding is invalid")
f=os.O_RDONLY|getattr(os,"O_CLOEXEC",0)
if not hasattr(os,"O_NOFOLLOW"):
    raise SystemExit("ACTION_BALL_NOSITE: O_NOFOLLOW is unavailable")
d=os.open(p,f|os.O_NOFOLLOW)
try:
    s0=os.fstat(d)
    if not stat.S_ISREG(s0.st_mode):
        raise SystemExit("ACTION_BALL_NOSITE: bootstrap is not regular")
    q=[]
    while True:
        x=os.read(d,1048576)
        if not x:
            break
        q.append(x)
    s1=os.fstat(d)
finally:
    os.close(d)
r=b"".join(q)
s2=os.lstat(p)
if (s0.st_dev,s0.st_ino,s0.st_mode,s0.st_size,s0.st_mtime_ns)!=(s1.st_dev,s1.st_ino,s1.st_mode,s1.st_size,s1.st_mtime_ns) or (s0.st_dev,s0.st_ino,s0.st_mode,s0.st_size,s0.st_mtime_ns)!=(s2.st_dev,s2.st_ino,s2.st_mode,s2.st_size,s2.st_mtime_ns) or len(r)!=s0.st_size or hashlib.sha256(r).hexdigest()!=e:
    raise SystemExit("ACTION_BALL_NOSITE: bootstrap bytes or identity drifted")
g={"__name__":"__main__","__file__":p,"__package__":None,"__spec__":None}
sys.argv=[p,e,c,b]
exec(compile(r,p,"exec"),g,g)
"""


class NoSiteBootstrapError(RuntimeError):
    """An exact no-site command or live byte binding is invalid."""


class ExactNoSiteCommand(NamedTuple):
    """Canonical builder/validator result."""

    argv: Tuple[str, ...]
    contract: Mapping[str, Any]
    contract_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the canonical JSON representation used by the argv contract."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NoSiteBootstrapError("argv contract is not canonical JSON") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in SHA256_CHARS for character in value)
    ):
        raise NoSiteBootstrapError("%s must be a lowercase SHA-256" % label)
    return value


def _require_line(value: Any, label: str, maximum: int = 65536) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise NoSiteBootstrapError("%s must be one bounded non-empty line" % label)
    return value


def _canonical_absolute(value: Any, label: str) -> Path:
    text = _require_line(value, label, maximum=16384)
    if not os.path.isabs(text) or os.path.normpath(text) != text:
        raise NoSiteBootstrapError(
            "%s must be an absolute lexically-normalized path" % label
        )
    return Path(text)


def _reject_duplicate_pairs(
    pairs: Iterable[Tuple[str, Any]]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NoSiteBootstrapError(
                "argv contract JSON contains duplicate key %s" % key
            )
        result[key] = value
    return result


def _strict_json(raw: bytes) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                NoSiteBootstrapError(
                    "argv contract contains forbidden constant %s" % token
                )
            ),
        )
    except NoSiteBootstrapError:
        raise
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise NoSiteBootstrapError("argv contract is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise NoSiteBootstrapError("argv contract root must be an object")
    if canonical_json_bytes(value) != raw:
        raise NoSiteBootstrapError("argv contract JSON is not canonical")
    return value


def _stable_regular_bytes(path: Path, label: str) -> Tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise NoSiteBootstrapError("O_NOFOLLOW is required")
    try:
        descriptor = os.open(str(path), flags | os.O_NOFOLLOW)
    except OSError as exc:
        raise NoSiteBootstrapError(
            "%s is not an openable no-follow regular file: %s" % (label, path)
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise NoSiteBootstrapError("%s is not a regular file" % label)
        chunks: List[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        final = os.lstat(str(path))
    except OSError as exc:
        raise NoSiteBootstrapError("%s disappeared after read" % label) from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    if identity_before != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ) or identity_before != (
        final.st_dev,
        final.st_ino,
        final.st_mode,
        final.st_size,
        final.st_mtime_ns,
    ):
        raise NoSiteBootstrapError("%s identity changed while being read" % label)
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise NoSiteBootstrapError("%s size changed while being read" % label)
    return raw, before


def bind_regular_file(path: Path, *, label: str = "source") -> Dict[str, Any]:
    """Return an exact path/size/SHA binding for one no-follow regular file."""

    canonical = _canonical_absolute(str(path), label)
    raw, _info = _stable_regular_bytes(canonical, label)
    return {
        "path": str(canonical),
        "byte_count": len(raw),
        "sha256": _sha256(raw),
    }


def _assert_directory_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = os.lstat(str(current))
        except OSError as exc:
            raise NoSiteBootstrapError(
                "%s path component is missing: %s" % (label, current)
            ) from exc
        if not stat.S_ISDIR(info.st_mode):
            raise NoSiteBootstrapError(
                "%s path contains a symlink or non-directory: %s"
                % (label, current)
            )


def bind_import_root(path: Path, *, label: str = "import root") -> Dict[str, Any]:
    """Hash one symlink-free regular-file tree without executing ``.pth``."""

    root = _canonical_absolute(str(path), label)
    _assert_directory_components(root, label)
    initial = os.lstat(str(root))
    if not stat.S_ISDIR(initial.st_mode):
        raise NoSiteBootstrapError("%s is not a real directory" % label)
    pending = [root]
    rows: List[Dict[str, Any]] = []
    directory_identities: Dict[str, Tuple[int, int, int]] = {}
    while pending:
        current = pending.pop()
        current_info = os.lstat(str(current))
        if not stat.S_ISDIR(current_info.st_mode):
            raise NoSiteBootstrapError(
                "%s tree directory drifted: %s" % (label, current)
            )
        directory_identities[str(current)] = (
            int(current_info.st_dev),
            int(current_info.st_ino),
            int(current_info.st_mtime_ns),
        )
        try:
            with os.scandir(str(current)) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise NoSiteBootstrapError(
                "cannot scan %s directory %s" % (label, current)
            ) from exc
        for entry in entries:
            member = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise NoSiteBootstrapError(
                    "cannot stat %s member %s" % (label, member)
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise NoSiteBootstrapError(
                    "symlink is forbidden in %s: %s" % (label, member)
                )
            if stat.S_ISDIR(info.st_mode):
                pending.append(member)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise NoSiteBootstrapError(
                    "special file is forbidden in %s: %s" % (label, member)
                )
            raw, opened_info = _stable_regular_bytes(member, "%s member" % label)
            if (
                opened_info.st_dev != info.st_dev
                or opened_info.st_ino != info.st_ino
                or opened_info.st_size != info.st_size
            ):
                raise NoSiteBootstrapError(
                    "%s member identity drifted: %s" % (label, member)
                )
            rows.append(
                {
                    "path": member.relative_to(root).as_posix(),
                    "byte_count": len(raw),
                    "sha256": _sha256(raw),
                }
            )
    for directory, expected in directory_identities.items():
        info = os.lstat(directory)
        if (
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_mtime_ns),
        ) != expected:
            raise NoSiteBootstrapError(
                "%s directory changed while hashing: %s" % (label, directory)
            )
    rows.sort(key=lambda row: row["path"])
    tree_document = {
        "schema_version": 1,
        "kind": "action_ball_symlink_free_import_tree_v1",
        "files": rows,
    }
    return {
        "path": str(root),
        "tree_sha256": _sha256(canonical_json_bytes(tree_document)),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["byte_count"]) for row in rows),
    }


def bind_import_roots(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Bind an ordered, non-empty list of unique import roots."""

    if not paths:
        raise NoSiteBootstrapError("at least one explicit import root is required")
    result: List[Dict[str, Any]] = []
    seen = set()
    for index, path in enumerate(paths):
        row = bind_import_root(path, label="import root %d" % index)
        if row["path"] in seen:
            raise NoSiteBootstrapError("import roots must be unique")
        candidate = Path(row["path"])
        for prior in result:
            prior_path = Path(prior["path"])
            if (
                candidate in prior_path.parents
                or prior_path in candidate.parents
            ):
                raise NoSiteBootstrapError(
                    "import roots must not overlap or nest"
                )
        seen.add(row["path"])
        result.append(row)
    return result


def _normalize_import_root_bindings(
    values: Sequence[Mapping[str, Any]],
    *,
    verify_live: bool,
) -> List[Dict[str, Any]]:
    if not isinstance(values, (list, tuple)) or not values:
        raise NoSiteBootstrapError("import_roots must be a non-empty sequence")
    result: List[Dict[str, Any]] = []
    seen = set()
    for index, raw in enumerate(values):
        if type(raw) is not dict or set(raw) != {
            "path",
            "tree_sha256",
            "file_count",
            "total_size_bytes",
        }:
            raise NoSiteBootstrapError(
                "import_roots[%d] keys are not exact" % index
            )
        path = _canonical_absolute(raw["path"], "import_roots[%d].path" % index)
        digest = _require_sha256(
            raw["tree_sha256"], "import_roots[%d].tree_sha256" % index
        )
        file_count = raw["file_count"]
        total_size = raw["total_size_bytes"]
        if type(file_count) is not int or file_count < 0:
            raise NoSiteBootstrapError("import root file_count is invalid")
        if type(total_size) is not int or total_size < 0:
            raise NoSiteBootstrapError("import root total_size_bytes is invalid")
        if str(path) in seen:
            raise NoSiteBootstrapError("import roots must be unique")
        for prior in result:
            prior_path = Path(prior["path"])
            if path in prior_path.parents or prior_path in path.parents:
                raise NoSiteBootstrapError(
                    "import roots must not overlap or nest"
                )
        seen.add(str(path))
        normalized = {
            "path": str(path),
            "tree_sha256": digest,
            "file_count": file_count,
            "total_size_bytes": total_size,
        }
        if verify_live:
            live = bind_import_root(path, label="import root %d" % index)
            if live != normalized:
                raise NoSiteBootstrapError(
                    "import root %d differs from its exact binding" % index
                )
        result.append(normalized)
    return result


def _normalize_contract(
    value: Mapping[str, Any], *, verify_live: bool
) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "kind",
        "bootstrap",
        "entrypoint",
        "import_roots",
        "entrypoint_argv",
    }:
        raise NoSiteBootstrapError("argv contract keys are not exact")
    if (
        value["schema_version"] != CONTRACT_SCHEMA_VERSION
        or value["kind"] != CONTRACT_KIND
    ):
        raise NoSiteBootstrapError("argv contract schema/kind is unsupported")
    bindings = {}
    for name in ("bootstrap", "entrypoint"):
        raw = value[name]
        if type(raw) is not dict or set(raw) != {
            "path",
            "byte_count",
            "sha256",
        }:
            raise NoSiteBootstrapError("%s binding keys are not exact" % name)
        path = _canonical_absolute(raw["path"], "%s.path" % name)
        count = raw["byte_count"]
        if type(count) is not int or count < 0:
            raise NoSiteBootstrapError("%s.byte_count is invalid" % name)
        digest = _require_sha256(raw["sha256"], "%s.sha256" % name)
        normalized = {
            "path": str(path),
            "byte_count": count,
            "sha256": digest,
        }
        if verify_live:
            live = bind_regular_file(path, label=name)
            if live != normalized:
                raise NoSiteBootstrapError(
                    "%s source differs from its exact binding" % name
                )
        bindings[name] = normalized
    roots = _normalize_import_root_bindings(
        value["import_roots"], verify_live=verify_live
    )
    raw_argv = value["entrypoint_argv"]
    if not isinstance(raw_argv, list):
        raise NoSiteBootstrapError("entrypoint_argv must be a list")
    entrypoint_argv = [
        _require_line(item, "entrypoint_argv[]") for item in raw_argv
    ]
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "bootstrap": bindings["bootstrap"],
        "entrypoint": bindings["entrypoint"],
        "import_roots": roots,
        "entrypoint_argv": entrypoint_argv,
    }


def build_exact_nosite_argv(
    *,
    python: Path,
    bootstrap: Path,
    bootstrap_sha256: str,
    entrypoint: Path,
    entrypoint_sha256: str,
    import_roots: Sequence[Mapping[str, Any]],
    entrypoint_argv: Sequence[str],
    verify_import_roots: bool = True,
) -> ExactNoSiteCommand:
    """Build one exact ``-I -B -S -c`` command and its canonical SHA."""

    python_path = _canonical_absolute(str(python), "python")
    bootstrap_binding = bind_regular_file(bootstrap, label="bootstrap")
    entrypoint_binding = bind_regular_file(entrypoint, label="entrypoint")
    if bootstrap_binding["sha256"] != _require_sha256(
        bootstrap_sha256, "bootstrap_sha256"
    ):
        raise NoSiteBootstrapError("bootstrap source SHA differs")
    if entrypoint_binding["sha256"] != _require_sha256(
        entrypoint_sha256, "entrypoint_sha256"
    ):
        raise NoSiteBootstrapError("entrypoint source SHA differs")
    if type(verify_import_roots) is not bool:
        raise NoSiteBootstrapError("verify_import_roots must be boolean")
    contract = _normalize_contract(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "kind": CONTRACT_KIND,
            "bootstrap": bootstrap_binding,
            "entrypoint": entrypoint_binding,
            "import_roots": list(import_roots),
            "entrypoint_argv": list(entrypoint_argv),
        },
        verify_live=verify_import_roots,
    )
    raw = canonical_json_bytes(contract)
    contract_sha256 = _sha256(raw)
    encoded = base64.b64encode(raw).decode("ascii")
    argv = (
        str(python_path),
        "-I",
        "-B",
        "-S",
        "-c",
        TRAMPOLINE_SOURCE,
        bootstrap_binding["path"],
        bootstrap_binding["sha256"],
        contract_sha256,
        encoded,
    )
    return ExactNoSiteCommand(
        argv=argv,
        contract=contract,
        contract_sha256=contract_sha256,
    )


def validate_exact_nosite_argv(
    argv: Sequence[str],
    *,
    expected_python: Optional[Path] = None,
    expected_bootstrap: Optional[Mapping[str, Any]] = None,
    expected_entrypoint: Optional[Mapping[str, Any]] = None,
    expected_import_roots: Optional[Sequence[Mapping[str, Any]]] = None,
    expected_entrypoint_argv: Optional[Sequence[str]] = None,
    expected_contract_sha256: Optional[str] = None,
    verify_live: bool = True,
) -> ExactNoSiteCommand:
    """Strictly parse and optionally live-verify one canonical command.

    Any reordered flag/root, duplicate root, extra token, source drift, or
    entrypoint argument drift is rejected.
    """

    if not isinstance(argv, (list, tuple)) or len(argv) != 10:
        raise NoSiteBootstrapError("no-site argv must contain exactly 10 tokens")
    tokens = list(argv)
    python_path = _canonical_absolute(tokens[0], "argv python")
    if tokens[1:6] != ["-I", "-B", "-S", "-c", TRAMPOLINE_SOURCE]:
        raise NoSiteBootstrapError("no-site argv flags/trampoline are not exact")
    bootstrap_path = _canonical_absolute(tokens[6], "argv bootstrap path")
    bootstrap_sha = _require_sha256(tokens[7], "argv bootstrap SHA")
    contract_sha = _require_sha256(tokens[8], "argv contract SHA")
    encoded = _require_line(tokens[9], "argv contract base64", maximum=16 * 1024 * 1024)
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise NoSiteBootstrapError("argv contract is not canonical base64") from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise NoSiteBootstrapError("argv contract is not canonical base64")
    if _sha256(raw) != contract_sha:
        raise NoSiteBootstrapError("argv contract SHA differs")
    contract = _normalize_contract(_strict_json(raw), verify_live=verify_live)
    if (
        contract["bootstrap"]["path"] != str(bootstrap_path)
        or contract["bootstrap"]["sha256"] != bootstrap_sha
    ):
        raise NoSiteBootstrapError(
            "trampoline bootstrap binding differs from argv contract"
        )
    if expected_python is not None and python_path != _canonical_absolute(
        str(expected_python), "expected_python"
    ):
        raise NoSiteBootstrapError("argv Python differs from expected")
    for label, expected in (
        ("bootstrap", expected_bootstrap),
        ("entrypoint", expected_entrypoint),
    ):
        if expected is not None and dict(expected) != contract[label]:
            raise NoSiteBootstrapError(
                "argv %s binding differs from expected" % label
            )
    if expected_import_roots is not None:
        normalized_expected = _normalize_import_root_bindings(
            expected_import_roots, verify_live=verify_live
        )
        if normalized_expected != contract["import_roots"]:
            raise NoSiteBootstrapError(
                "argv import root order/bindings differ from expected"
            )
    if expected_entrypoint_argv is not None:
        normalized_args = [
            _require_line(value, "expected_entrypoint_argv[]")
            for value in expected_entrypoint_argv
        ]
        if normalized_args != contract["entrypoint_argv"]:
            raise NoSiteBootstrapError(
                "argv entrypoint arguments differ from expected"
            )
    if expected_contract_sha256 is not None and contract_sha != _require_sha256(
        expected_contract_sha256, "expected_contract_sha256"
    ):
        raise NoSiteBootstrapError("argv contract SHA differs from expected")
    canonical = build_exact_nosite_argv(
        python=python_path,
        bootstrap=Path(contract["bootstrap"]["path"]),
        bootstrap_sha256=contract["bootstrap"]["sha256"],
        entrypoint=Path(contract["entrypoint"]["path"]),
        entrypoint_sha256=contract["entrypoint"]["sha256"],
        import_roots=contract["import_roots"],
        entrypoint_argv=contract["entrypoint_argv"],
    ) if verify_live else ExactNoSiteCommand(
        argv=tuple(tokens),
        contract=contract,
        contract_sha256=contract_sha,
    )
    if tuple(tokens) != canonical.argv:
        raise NoSiteBootstrapError("no-site argv is not its canonical rebuild")
    return canonical


def _execution_flags() -> Dict[str, Any]:
    return {
        "isolated": bool(sys.flags.isolated),
        "no_site": bool(sys.flags.no_site),
        "no_user_site": bool(sys.flags.no_user_site),
        "ignore_environment": bool(sys.flags.ignore_environment),
        "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        "optimize": int(sys.flags.optimize),
    }


def _require_exact_flags() -> Dict[str, Any]:
    flags = _execution_flags()
    if flags != {
        "isolated": True,
        "no_site": True,
        "no_user_site": True,
        "ignore_environment": True,
        "dont_write_bytecode": True,
        "optimize": 0,
    }:
        raise NoSiteBootstrapError(
            "bootstrap requires exact python -I -B -S with optimize=0"
        )
    return flags


def _execute_from_trampoline(argv: Sequence[str]) -> None:
    if len(argv) != 4:
        raise NoSiteBootstrapError("bootstrap trampoline argv is not exact")
    bootstrap_path = _canonical_absolute(argv[0], "bootstrap __file__")
    bootstrap_sha = _require_sha256(argv[1], "bootstrap trampoline SHA")
    contract_sha = _require_sha256(argv[2], "argv contract SHA")
    encoded = _require_line(
        argv[3], "argv contract base64", maximum=16 * 1024 * 1024
    )
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise NoSiteBootstrapError("argv contract is not canonical base64") from exc
    if (
        base64.b64encode(raw).decode("ascii") != encoded
        or _sha256(raw) != contract_sha
    ):
        raise NoSiteBootstrapError("argv contract base64/SHA differs")
    contract = _normalize_contract(_strict_json(raw), verify_live=True)
    if (
        contract["bootstrap"]["path"] != str(bootstrap_path)
        or contract["bootstrap"]["sha256"] != bootstrap_sha
    ):
        raise NoSiteBootstrapError(
            "executed bootstrap differs from argv contract"
        )
    flags = _require_exact_flags()
    if "site" in sys.modules:
        raise NoSiteBootstrapError("site was imported before no-site entrypoint")
    initial_sys_path = [
        os.path.normpath(os.path.abspath(value))
        for value in sys.path
        if value
    ]
    if any(
        part in {"site-packages", "dist-packages"}
        for value in initial_sys_path
        for part in Path(value).parts
    ):
        raise NoSiteBootstrapError(
            "implicit site/dist-packages path exists before root install"
        )
    root_paths = [row["path"] for row in contract["import_roots"]]
    if any(path in initial_sys_path for path in root_paths):
        raise NoSiteBootstrapError(
            "preregistered import root was exposed before bootstrap"
        )
    sys.path.extend(root_paths)
    entrypoint_path = Path(contract["entrypoint"]["path"])
    entrypoint_raw, _entrypoint_info = _stable_regular_bytes(
        entrypoint_path, "entrypoint"
    )
    if (
        len(entrypoint_raw) != contract["entrypoint"]["byte_count"]
        or _sha256(entrypoint_raw) != contract["entrypoint"]["sha256"]
    ):
        raise NoSiteBootstrapError("entrypoint bytes differ from argv contract")
    attestation = {
        "schema_version": 1,
        "kind": ATTESTATION_KIND,
        "argv_contract_sha256": contract_sha,
        "bootstrap": dict(contract["bootstrap"]),
        "entrypoint": dict(contract["entrypoint"]),
        "import_roots": [dict(row) for row in contract["import_roots"]],
        "flags": flags,
        "site_module_loaded_before_entrypoint": False,
        "pth_files_executed": False,
        "sys_path_before_import_roots": initial_sys_path,
        "sys_path_after_import_roots": list(sys.path),
    }
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(entrypoint_path),
        "__package__": None,
        "__spec__": None,
        "ACTION_BALL_NOSITE_ATTESTATION": attestation,
    }
    sys.argv = [str(entrypoint_path)] + list(contract["entrypoint_argv"])
    exec(
        compile(entrypoint_raw, str(entrypoint_path), "exec"),
        globals_dict,
        globals_dict,
    )


if __name__ == "__main__":
    try:
        _execute_from_trampoline(sys.argv)
    except NoSiteBootstrapError as exc:
        # stderr only: successful bootstraps never pollute the entrypoint's
        # stdout protocol.
        print("ACTION_BALL_NOSITE_ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
