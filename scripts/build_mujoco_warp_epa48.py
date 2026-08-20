#!/usr/bin/env python3
"""Offline build/verify for the narrow project-owned MuJoCo-Warp EPA48 fork.

It neither downloads nor installs. It proves the pinned sdist and two-file
patch, builds one ignored wheel, and verifies its version and EPA constant.
Physics acceptance is deliberately outside this supply-chain tool.
"""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
import email.policy
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = REPO_ROOT / "configs/mujoco_warp_epa48_20260821/PROVENANCE.json"
RECEIPT_NAME = "build_receipt.json"
RECEIPT_SCHEMA_VERSION = 4

class ForkError(RuntimeError):
    pass

def _atomic_rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish one directory without replacing any target."""
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise ForkError("atomic no-replace rename is unavailable on this macOS host")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = rename(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise ForkError("atomic no-replace renameat2 is unavailable on this Linux host")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = rename(-100, source_bytes, -100, target_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif os.name == "nt":
        try:
            os.rename(source, target)  # Windows rename already refuses an existing target.
        except FileExistsError as exc:
            raise ForkError(f"refusing to replace publish target {target}") from exc
        return
    else:
        raise ForkError(f"atomic no-replace directory publish is unsupported on {sys.platform}")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise ForkError(f"refusing to replace publish target {target}")
    raise ForkError(f"atomic no-replace publish failed for {target}: {os.strerror(error)}")

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForkError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ForkError(f"JSON root is not an object: {path}")
    return value

def _load_provenance(path: Path = PROVENANCE_PATH) -> Dict[str, Any]:
    value = _read_json(path)
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1 or value.get("fork_id") != "hope_mujoco_warp_epa48_v1":
        raise ForkError("unexpected provenance identity")
    build = value.get("build", {})
    if build.get("network_allowed") is not False or build.get("dependency_install_allowed") is not False:
        raise ForkError("build must remain no-network and no-install")
    return value

def _write_json_x(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise ForkError(f"refusing to overwrite {path}") from exc

def _safe_extract(sdist: Path, destination: Path, top_level: str) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(str(sdist), "r:gz") as archive:
            members = archive.getmembers()
            seen = set()
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    not path.parts
                    or path.is_absolute()
                    or ".." in path.parts
                    or path.parts[0] != top_level
                    or member.name in seen
                    or not (member.isfile() or member.isdir())
                ):
                    raise ForkError(f"unsafe sdist member: {member.name!r}")
                seen.add(member.name)
            if sys.version_info >= (3, 12):
                archive.extractall(str(destination), members=members, filter="data")
            else:
                archive.extractall(str(destination), members=members)
    except (OSError, tarfile.TarError) as exc:
        raise ForkError(f"cannot extract sdist: {exc}") from exc
    root = destination / top_level
    if not root.is_dir() or root.is_symlink():
        raise ForkError(f"missing real source root {root}")
    return root

def _inventory(root: Path) -> Dict[str, str]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ForkError(f"source symlink is not allowed: {path}")
        if path.is_file():
            out[path.relative_to(root).as_posix()] = _sha256_file(path)
    return out

def _package_payload(source: Path) -> Dict[str, str]:
    root = source / "mujoco_warp"
    if not root.is_dir() or root.is_symlink():
        raise ForkError("patched source has no real mujoco_warp package")
    payload = {f"mujoco_warp/{name}": digest for name, digest in _inventory(root).items()}
    if "mujoco_warp/__init__.py" not in payload or "mujoco_warp/_src/types.py" not in payload:
        raise ForkError("patched source package payload is incomplete")
    return payload

def _manifest_sha256(payload: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(payload.items()):
        digest.update(name.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    return digest.hexdigest()

def _manifest_record(payload: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "file_count": len(payload),
        "manifest_sha256": _manifest_sha256(payload),
    }

def _check_input(path: Path, contract: Mapping[str, Any], kind: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ForkError(f"{kind} must be an existing regular file: {path}")
    if "filename" in contract and path.name != contract["filename"]:
        raise ForkError(f"{kind} filename mismatch: {path.name}")
    actual = _sha256_file(path)
    if actual != contract["sha256"]:
        raise ForkError(f"{kind} SHA mismatch: {actual} != {contract['sha256']}")

def _prepare_source(sdist: Path, destination: Path, provenance: Mapping[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    sdist_contract = provenance["upstream"]["sdist"]
    patch_contract = provenance["fork"]["patch"]
    patch_path = PROVENANCE_PATH.parent / patch_contract["path"]
    _check_input(sdist, sdist_contract, "sdist")
    _check_input(patch_path, patch_contract, "patch")
    source = _safe_extract(sdist, destination, sdist_contract["top_level_directory"])
    before = _inventory(source)
    package_before = _package_payload(source)
    allowed = provenance["fork"]["allowed_source_changes"]
    for name, contract in allowed.items():
        if before.get(name) != contract["before_sha256"]:
            raise ForkError(f"unexpected upstream bytes for {name}")

    patch_tool = shutil.which("patch")
    if not patch_tool:
        raise ForkError("required host tool `patch` is missing")
    result = subprocess.run(
        [patch_tool, "-p1", "-f", "-N", "-F", "0", "-i", str(patch_path)],
        cwd=str(source),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ForkError(f"tracked patch failed: {(result.stdout + result.stderr).strip()}")

    after = _inventory(source)
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    if set(before) != set(after) or changed != sorted(allowed):
        raise ForkError(f"patch changed {changed}; allowed={sorted(allowed)}")
    for name, contract in allowed.items():
        if after[name] != contract["after_sha256"]:
            raise ForkError(f"unexpected patched bytes for {name}")
    payload = _package_payload(source)
    package_changed = sorted(
        name for name in set(package_before) | set(payload)
        if package_before.get(name) != payload.get(name)
    )
    if package_changed != ["mujoco_warp/_src/types.py"]:
        raise ForkError(f"runtime package delta is not the one-file EPA constant: {package_changed}")
    return source, {
        "sdist": {
            "filename": sdist_contract["filename"],
            "sha256": sdist_contract["sha256"],
            "top_level_directory": sdist_contract["top_level_directory"],
        },
        "patch": {
            "path": patch_contract["path"],
            "sha256": patch_contract["sha256"],
        },
        "changed_files": changed,
        "package_changed_files": package_changed,
        "package_before": _manifest_record(package_before),
        "package_after": _manifest_record(payload),
    }

def _builder_identity(python: Path) -> Dict[str, Any]:
    """Return reported build-environment telemetry.

    These values help reproduce a build, but the source/wheel byte verifier is
    the authority.  A receipt cannot independently authenticate the process
    that wrote its own environment fields.
    """
    code = r'''
import importlib.metadata as md, json, sys
from pathlib import Path
out = {"reported_executable": sys.executable,
       "resolved_executable": str(Path(sys.executable).resolve()),
       "python_version": sys.version}
for name in ("pip", "setuptools", "wheel"):
  dist = md.distribution(name)
  out[name] = {"version": dist.version,
               "root": str(Path(dist.locate_file("")).resolve())}
print(json.dumps(out, sort_keys=True))
'''
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ForkError(
            "caller Python must already contain pip, setuptools and wheel; "
            f"nothing will be installed: {(result.stdout + result.stderr).strip()}"
        )
    try:
        value = json.loads(result.stdout.splitlines()[-1])
        resolved = Path(value["resolved_executable"])
    except (IndexError, KeyError, json.JSONDecodeError) as exc:
        raise ForkError("cannot parse caller Python identity") from exc
    if not resolved.is_file():
        raise ForkError("caller Python does not resolve to a regular file")
    value["executable_sha256"] = _sha256_file(resolved)
    return value

def _build_wheel(source: Path, wheelhouse: Path, python: Path) -> Tuple[Path, Dict[str, Any]]:
    python = python.absolute()
    identity = _builder_identity(python)
    wheelhouse.mkdir(parents=True, exist_ok=False)
    flags = ["--no-index", "--no-deps", "--no-build-isolation", "--no-cache-dir"]
    command = [str(python), "-m", "pip", "wheel", *flags, "--wheel-dir", str(wheelhouse), str(source)]
    environment = os.environ.copy()
    environment.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ForkError(f"offline wheel build failed: {(result.stdout + result.stderr).strip()}")
    wheels = list(wheelhouse.glob("*.whl"))
    if len(wheels) != 1:
        raise ForkError(f"build produced {len(wheels)} wheels, expected one")
    return wheels[0], {"caller": identity, "pip_flags": flags}

def _verify_wheel(wheel: Path, provenance: Mapping[str, Any], source: Path) -> Dict[str, Any]:
    fork = provenance["fork"]
    if not wheel.is_file() or wheel.is_symlink():
        raise ForkError("wheel must be a regular non-symlink file")
    if wheel.name != fork["expected_wheel_filename"]:
        raise ForkError(f"wheel filename mismatch: {wheel.name}")
    payload = _package_payload(source)
    dist_info = f"mujoco_warp-{fork['version']}.dist-info"
    metadata_files = {
        f"{dist_info}/licenses/AUTHORS": source / "AUTHORS",
        f"{dist_info}/licenses/LICENSE": source / "LICENSE",
        f"{dist_info}/entry_points.txt": source / "mujoco_warp.egg-info/entry_points.txt",
        f"{dist_info}/top_level.txt": source / "mujoco_warp.egg-info/top_level.txt",
    }
    metadata_name = f"{dist_info}/METADATA"
    wheel_name = f"{dist_info}/WHEEL"
    record_name = f"{dist_info}/RECORD"
    expected_names = set(payload) | set(metadata_files) | {metadata_name, wheel_name, record_name}
    try:
        with zipfile.ZipFile(str(wheel)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ForkError("wheel contains duplicate ZIP entries")
            for info in infos:
                path = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    not info.filename
                    or info.filename.endswith("/")
                    or "\\" in info.filename
                    or "\0" in info.filename
                    or path.is_absolute()
                    or ".." in path.parts
                    or path.as_posix() != info.filename
                    or stat.S_ISLNK(mode)
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                ):
                    raise ForkError(f"unsafe wheel member: {info.filename!r}")
            actual_names = set(names)
            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)[:5]
                extra = sorted(actual_names - expected_names)[:5]
                raise ForkError(f"wheel payload set mismatch: missing={missing}, extra={extra}")
            evidence = {}
            for name in names:
                data = archive.read(name)
                evidence[name] = (hashlib.sha256(data).hexdigest(), len(data))
            changed = sorted(name for name, digest in payload.items() if evidence[name][0] != digest)
            if changed:
                raise ForkError(f"wheel package bytes differ from patched source: {changed[:5]}")
            for name, path in metadata_files.items():
                if evidence[name][0] != _sha256_file(path):
                    raise ForkError(f"wheel metadata payload differs from source: {name}")
            metadata = email.message_from_bytes(archive.read(metadata_name), policy=email.policy.default)
            if str(metadata["Name"]).replace("_", "-").lower() != fork["distribution"]:
                raise ForkError(f"wheel Name mismatch: {metadata['Name']}")
            if str(metadata["Version"]) != fork["version"]:
                raise ForkError(f"wheel Version mismatch: {metadata['Version']}")
            source_metadata = email.message_from_bytes(
                (source / "PKG-INFO").read_bytes(), policy=email.policy.default
            )
            if metadata.get_all("Requires-Dist", []) != source_metadata.get_all("Requires-Dist", []):
                raise ForkError("wheel dependency metadata differs from pinned source")
            wheel_metadata = email.message_from_bytes(archive.read(wheel_name), policy=email.policy.default)
            if (
                str(wheel_metadata["Root-Is-Purelib"]).lower() != "true"
                or wheel_metadata.get_all("Tag", []) != ["py3-none-any"]
            ):
                raise ForkError("wheel platform contract mismatch")
            rows = list(csv.reader(archive.read(record_name).decode("utf-8").splitlines()))
            if any(len(row) != 3 for row in rows) or len(rows) != len({row[0] for row in rows}):
                raise ForkError("wheel RECORD is malformed or duplicate")
            record = {row[0]: row[1:] for row in rows}
            if set(record) != expected_names or record[record_name] != ["", ""]:
                raise ForkError("wheel RECORD entry set mismatch")
            for name in expected_names - {record_name}:
                encoded = base64.urlsafe_b64encode(bytes.fromhex(evidence[name][0])).rstrip(b"=").decode()
                if record[name] != [f"sha256={encoded}", str(evidence[name][1])]:
                    raise ForkError(f"wheel RECORD evidence mismatch: {name}")
            types_data = archive.read("mujoco_warp/_src/types.py")
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError, RuntimeError) as exc:
        raise ForkError(f"cannot inspect wheel: {exc}") from exc
    expected_types = fork["allowed_source_changes"]["mujoco_warp/_src/types.py"]["after_sha256"]
    if hashlib.sha256(types_data).hexdigest() != expected_types:
        raise ForkError("wheel types.py differs from the exact patched bytes")
    if types_data.count(b"MJ_MAX_EPAHORIZON = 48") != 1:
        raise ForkError("wheel does not contain exact EPA horizon 48")
    return {
        "filename": wheel.name,
        "bytes": wheel.stat().st_size,
        "sha256": _sha256_file(wheel),
        "distribution": fork["distribution"],
        "version": fork["version"],
        "epa_horizon": 48,
        "types_py_sha256": expected_types,
    }

def _validate_target(artifact_root: Path, provenance: Mapping[str, Any]) -> None:
    vendor = REPO_ROOT / "vendor_assets"
    expected = REPO_ROOT / provenance["build"]["wheel_output_root"]
    if not vendor.is_dir() or vendor.is_symlink():
        raise ForkError("vendor_assets must be a real ignored directory; see setup_local_sync.md")
    if artifact_root.absolute() != expected.absolute():
        raise ForkError(f"artifact root must be {expected}")
    ignored = subprocess.run(["git", "check-ignore", "-q", str(vendor)], cwd=str(REPO_ROOT))
    if ignored.returncode:
        raise ForkError("vendor_assets is not ignored")
    if artifact_root.exists() or artifact_root.is_symlink():
        raise ForkError(f"refusing to overwrite {artifact_root}")

def _validate_receipt(
    receipt: Mapping[str, Any], provenance: Mapping[str, Any], source: Mapping[str, Any], wheel: Mapping[str, Any]
) -> None:
    required = {"schema_version", "verdict", "fork_id", "source", "build", "wheel"}
    if (
        set(receipt) != required
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
    ):
        raise ForkError("receipt schema mismatch")
    if receipt.get("verdict") != "PASS_BUILD_CHAIN_ONLY" or receipt.get("fork_id") != provenance["fork_id"]:
        raise ForkError("receipt identity mismatch")
    if receipt.get("source") != source or receipt.get("wheel") != wheel:
        raise ForkError("receipt source or wheel evidence mismatch")
    build = receipt.get("build")
    flags = ["--no-index", "--no-deps", "--no-build-isolation", "--no-cache-dir"]
    if not isinstance(build, dict) or set(build) != {"caller", "pip_flags"} or build.get("pip_flags") != flags:
        raise ForkError("receipt build contract mismatch")
    caller = build.get("caller")
    caller_keys = {"reported_executable", "resolved_executable", "python_version", "pip", "setuptools", "wheel", "executable_sha256"}
    if not isinstance(caller, dict) or set(caller) != caller_keys:
        raise ForkError("receipt builder identity mismatch")
    sha = caller.get("executable_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ForkError("receipt builder executable SHA mismatch")
    if any(not isinstance(caller.get(key), str) or not caller[key] for key in caller_keys - {"pip", "setuptools", "wheel", "executable_sha256"}):
        raise ForkError("receipt builder strings are missing")
    for name in ("pip", "setuptools", "wheel"):
        item = caller.get(name)
        if not isinstance(item, dict) or set(item) != {"version", "root"} or not all(isinstance(item[key], str) and item[key] for key in item):
            raise ForkError(f"receipt {name} identity mismatch")

def build_artifact(sdist: Path, artifact_root: Path, python: Path) -> Path:
    provenance = _load_provenance()
    _validate_target(artifact_root, provenance)
    stage = Path(tempfile.mkdtemp(prefix=f".{artifact_root.name}.partial.", dir=str(artifact_root.parent)))
    try:
        with tempfile.TemporaryDirectory(prefix="hope-mjwarp-epa48-source.") as temp:
            source, _ = _prepare_source(sdist.absolute(), Path(temp) / "source", provenance)
            # ``_build_wheel`` preserves a venv entry path; resolving
            # ``venv/bin/python`` would silently escape the selected prefix.
            wheel, build_evidence = _build_wheel(source, stage / "wheelhouse", python)
        # The build backend may mutate its input tree.  Reconstruct the pinned
        # source independently so a mutated build tree cannot certify itself.
        with tempfile.TemporaryDirectory(prefix="hope-mjwarp-epa48-verify-source.") as temp:
            verify_source, source_evidence = _prepare_source(
                sdist.absolute(), Path(temp) / "source", provenance
            )
            wheel_evidence = _verify_wheel(wheel, provenance, verify_source)
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "verdict": "PASS_BUILD_CHAIN_ONLY",
            "fork_id": provenance["fork_id"],
            "source": source_evidence,
            "build": build_evidence,
            "wheel": wheel_evidence,
        }
        _write_json_x(stage / RECEIPT_NAME, receipt)
        _validate_receipt(_read_json(stage / RECEIPT_NAME), provenance, source_evidence, wheel_evidence)
        if artifact_root.exists() or artifact_root.is_symlink():
            raise ForkError(f"publish target appeared during build: {artifact_root}")
        _atomic_rename_noreplace(stage, artifact_root)
    except Exception:
        print(f"partial build preserved at {stage}", file=sys.stderr)
        raise
    return artifact_root

def verify_artifact(sdist: Path, artifact_root: Path) -> Dict[str, Any]:
    provenance = _load_provenance()
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ForkError("artifact root must be a real directory")
    if set(path.name for path in artifact_root.iterdir()) != {RECEIPT_NAME, "wheelhouse"}:
        raise ForkError("artifact root must contain exactly receipt and wheelhouse")
    wheelhouse = artifact_root / "wheelhouse"
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise ForkError("artifact wheelhouse must be a real directory")
    receipt = _read_json(artifact_root / RECEIPT_NAME)
    if (artifact_root / RECEIPT_NAME).is_symlink():
        raise ForkError("artifact receipt must not be a symlink")
    entries = list(wheelhouse.iterdir())
    wheels = [path for path in entries if path.suffix == ".whl"]
    if len(entries) != 1 or len(wheels) != 1 or wheels[0].is_symlink():
        raise ForkError("artifact wheelhouse must contain exactly one regular wheel")
    with tempfile.TemporaryDirectory(prefix="hope-mjwarp-epa48-verify.") as temp:
        source, source_evidence = _prepare_source(sdist.absolute(), Path(temp) / "source", provenance)
        wheel_evidence = _verify_wheel(wheels[0], provenance, source)
    _validate_receipt(receipt, provenance, source_evidence, wheel_evidence)
    return receipt

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--sdist", type=Path, required=True)
    build.add_argument("--output-root", type=Path, default=REPO_ROOT / "vendor_assets/mujoco_warp_epa48_1")
    build.add_argument("--python", type=Path, default=Path(sys.executable))
    verify = commands.add_parser("verify")
    verify.add_argument("--sdist", type=Path, required=True)
    verify.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "vendor_assets/mujoco_warp_epa48_1")
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            root = build_artifact(args.sdist, args.output_root, args.python)
            receipt = _read_json(root / RECEIPT_NAME)
        else:
            receipt = verify_artifact(args.sdist, args.artifact_root)
    except ForkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
