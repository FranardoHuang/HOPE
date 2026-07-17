#!/usr/bin/env python3
"""Run the four registered d=12 ready-to-strike ladder cells, fail closed.

This is a host/CPU-only screening runner.  It never opens SSH, sends a signal,
retries a failed cell, starts a trainer, or addresses a robot.  Dry-run is the
default.  Execute mode needs an explicit token and an absolute, nonexistent
result root; every input snapshot and result is published with O_EXCL.

Stage 2 is intentionally downstream of the historical Stage-1 attestation.
The activation document must bind the exact receipt bytes and keep the honest
``screening-only`` formal claims.  A missing or edited receipt therefore blocks
before a namespace or subprocess is created.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
import zipfile
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA_VERSION = 1
CONFIRM_TOKEN = "RUN_READY_TO_STRIKE_STAGE2_ONCE"
CHILD_TIMEOUT_S = 3600
EXPECTED_ACTIVATION_ID = "ready_to_strike_join_ladder_stage2_v8_20260717"
EXPECTED_EXPERIMENT_ID = "ready_to_strike_join_ladder_20260717"
EXPECTED_QUEUE_SHA256 = "cfa112f799dab9af33914fdfb5bfff90d21b4692e38b16a4627393936a527b8b"
EXPECTED_PREREG_COMMIT = "8d74025e88fee832fae0ac2f672ec0eb9b2d3d5a"
EXPECTED_EVIDENCE_STATUS = "historical_stage1_attested_screening_only"
EXPECTED_STAGE2_NAMESPACE = (
    "/workspace/codexschema/ready_to_strike_0p5_20260717/"
    "join_ladder_stage2_d12_v8_canonical_interpreter"
)
EXPECTED_PRIOR_ATTEMPT = {
    "namespace": (
        "/workspace/codexschema/ready_to_strike_0p5_20260717/"
        "join_ladder_stage2_d12_v2_float32_producer"
    ),
    "summary_path": (
        "/workspace/codexschema/ready_to_strike_0p5_20260717/"
        "join_ladder_stage2_d12_v2_float32_producer/stage2_summary.json"
    ),
    "summary_sha256": "6910db2826654123c576afa67b9c2e873c4785c2bd095b2f61abb26d5f1f1476",
    "runner_sha256": "049295e63e6f786cdb6aeb9ae8fe1d30d8418f8aad8253587fce52d76f44b9c5",
    "activation_sha256": "8742aadff796218f170fede3f6e386e54314e086740f4ad82b9242f52667ab10",
    "failure_class": "prior_v2_topp_rc1_no_timing",
    "automatic_retry": False,
}

EXPECTED_PRIOR_V1_SUMMARY_SHA256 = (
    "f92e6b8b30844ba366c0bc901aacdb0f040e61f961678bd2290d833b8ac63c0e"
)
EXPECTED_MJCF_MODEL_TREE_OID = "0870b9bf9eff29473b02cc9e363cbf084dda9048"
EXPECTED_MJCF_CLOSURE_FILE_COUNT = 75
EXPECTED_MJCF_CLOSURE_TOTAL_BYTES = 14127373
EXPECTED_MJCF_CLOSURE_MANIFEST_SHA256 = (
    "e0381752eab46013c08559b331abb261beaa88a207a3c2f1155ab00857b962de"
)
EXPECTED_TOPP_RUNTIME = {
    "interpreter": {
        "path": "/workspace/hope_mjeval_venv/bin/python",
        "canonical_realpath": "/usr/bin/python3.12",
        "binary_sha256": (
            "1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5"
        ),
        "python_version": "3.12.3",
        "venv_prefix": "/workspace/hope_mjeval_venv",
    },
    "packages": {
        "numpy": {
            "version": "2.5.0",
            "module_sha256": (
                "09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11"
            ),
            "metadata_sha256": (
                "981cedfa033b69d5a8e153e42cf5f26f7027dd0b3701c37d8d8a2b83e8315d48"
            ),
            "record_sha256": (
                "ad8472357bd1a24f7c0e38ec421a28a07560bb948d27db25fa61cf7ff62f8a9a"
            ),
            "wheel_sha256": (
                "ac90a994678616346852fc495a154e351832198199a0e921c9e8f4b372b28e82"
            ),
            "record_file_count": 1339,
            "record_total_bytes": 68935821,
            "record_manifest_sha256": (
                "a07476342bb248770c73f8ab4ecf07036ad9a84fe325c9606afe837dfaaf8ea8"
            ),
            "record_native_elf_count": 22,
            "record_unhashed_row_count": 407,
        },
        "mujoco": {
            "version": "3.10.0",
            "module_sha256": (
                "c734d493d95933f4414633325491e8e6658670455a3c94981a6c1d26600d43e1"
            ),
            "metadata_sha256": (
                "e37572aef23253626ac77d51bba1eaee630f69037c315e8f715fc56758f7fe3f"
            ),
            "record_sha256": (
                "1dbbdae72fe8522bdf8d6640d7a4059948229a491ffbbac3bd964e40019c9f76"
            ),
            "wheel_sha256": (
                "b40ad1e4df54976de0a343902219f27fbed3bc20eb34efbef032b05c5a0f93e9"
            ),
            "record_file_count": 324,
            "record_total_bytes": 60705862,
            "record_manifest_sha256": (
                "726014ea93041792bdc179cdf65f2595d6ed9e6c3e94037bb40bae585cabe62a"
            ),
            "record_native_elf_count": 22,
            "record_unhashed_row_count": 91,
        },
    },
    "dynamic_dependencies": {
        "ldd_path": "/usr/bin/ldd",
        "ldd_sha256": (
            "429938a30ba5d51f4cdba476e8f8f8b1595d51b14a665ab6edf642454ff662ea"
        ),
        "readelf_path": "/usr/bin/x86_64-linux-gnu-readelf",
        "readelf_sha256": (
            "6d54602a1ee13f1214973086bd60efe2dae4363f8f5ab7516eaaf3e259dca90e"
        ),
        "allowed_virtual_dependencies": ["linux-vdso.so.1"],
        "elf_input_count": 38,
        "resolved_file_count": 17,
        "edge_count": 167,
        "manifest_sha256": (
            "088ea1213da73d9149eb624f87211d4b1cc64c0f2fa4f2bc788e875582ae5982"
        ),
    },
    "mjcf_model": {
        "loader": "mujoco.MjModel.from_xml_path",
        "nq": 38,
        "nv": 37,
        "nbody": 33,
        "ngeom": 79,
        "nmesh": 74,
    },
}
EXPECTED_PRIOR_CANDIDATES = {
    "fh_rf_d12": ("a6c181f1b29b7e683a2efa70414f908c0896d110b21721c39565e3641a4eeb17",
                  "7c8e1f3a5184829d66e48f33e2ed93dbe93c044b2b4feea1dd921f2dddd9fb1a"),
    "fh_rb_d12": ("ac3089ed72492eb92a4bdb63c218070af9303fa7fb4ec6df909f7e406ea13c6a",
                  "9970770e897b9464f258888e645bd45f6de8cebdfc640816e194ad713a20a535"),
    "bh_rf_d12": ("c892336ee0363e0867535be9fc892a071c49ae3af338412bcc090f06d66c6c64",
                  "f7686ef8dad9709eecf9009d276b90b2a2d04ae72836c93585aa95d4ad2afbfb"),
    "bh_rb_d12": ("d9ce654c861d343be8fd6ed81ac40a15fda9b95d6bf2969bacdb936697e68643",
                  "e504637a42bf1c26d6100d5a682974a5e950c0a18aeeb10c120754a87cce1790"),
}
EXPECTED_STAGE1_CELLS = {
    "fh_rf_d17": ("forehand", "forehand", 17),
    "fh_rb_d06": ("forehand", "backhand", 6),
    "fh_rb_d17": ("forehand", "backhand", 17),
    "bh_rf_d17": ("backhand", "forehand", 17),
    "bh_rb_d06": ("backhand", "backhand", 6),
    "bh_rb_d17": ("backhand", "backhand", 17),
}
EXPECTED_STAGE2_CELLS = {
    "fh_rf_d12": ("forehand", "forehand", 12),
    "fh_rb_d12": ("forehand", "backhand", 12),
    "bh_rf_d12": ("backhand", "forehand", 12),
    "bh_rb_d12": ("backhand", "backhand", 12),
}
EXPECTED_OBSERVATIONS = {
    "fh_rf_d06": ("forehand", "forehand", 6, "b0350f4d8db600344651135673afc99c95d43a2c27432d131a8ed8a08253ac4d", "c16c4dc74f9ac254ab22a4987ad23ad3bc45fc462040a1a37f8eb98c213474ed", 0.64),
    "fh_rf_d17": ("forehand", "forehand", 17, "ff11d738dfe8fe629fbb796f322c51a47017918643d8da55f697cbf137ad9122", "5d939ec781858919611f0191447d0699499c1e3ccffc38c9d3b33f07ab371e60", 1.28),
    "fh_rb_d06": ("forehand", "backhand", 6, "469ef828855151854f425dda66928454395d6ab84098d2de469dc05dd73e1113", "71964c47d6c340d4f2268394b4009fa295cbb46c4095f52996f4ca10134e4b37", 0.70),
    "fh_rb_d17": ("forehand", "backhand", 17, "733e63b5c7cad7406d551f210f9ea1ddd5b8b4b5701ee2a93a972148fe0e91a1", "fe76360795483cdd4397d903eeeef085f82021dce5f7e8b1ced446de94f5512a", 1.54),
    "bh_rf_d06": ("backhand", "forehand", 6, "0f017dc58918aa1e80846d4dfec81668efbb3fecabf546727750a8c2717dfc89", "0d0e45e336d3aa02e3bb781fd252a5b2b5453d4e49caa6049b3b1d0c33f0c720", 0.94),
    "bh_rf_d17": ("backhand", "forehand", 17, "21fc80a57b214f78f9e4a8e9e6afa062e290b28bf876f7da34d2d8009fae9046", "0b9243d023173a2710fa5bd7b4ba27d97d9216e61b0bf16803caed41ab18d333", 1.94),
    "bh_rb_d06": ("backhand", "backhand", 6, "18b27e4b530f1145f0e8fd8cea50902c8559d2d037e0cbc023bd73d0b4187eec", "5482539bb4302a18b0868239a39ddf1e7ebcbff29e96d5223f8ba68050df8810", 0.78),
    "bh_rb_d17": ("backhand", "backhand", 17, "a842a87d0557dfc6e814e09ff466336ed14e294885304935316bc94d3410725e", "e8e7adac56ba5225061437a03cc26e1cb671b8b6bb099c5f0d034dc5c6dd1d6f", 1.42),
}

RUNTIME_RELATIVE_PATHS = {
    "topp_sha256": Path("hope_training/whole_body_tracking/scripts/topp_mintime.py"),
    "mjcf_sha256": Path("agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"),
    "urdf_sha256": Path("agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"),
    "body_order_sha256": Path("configs/a3_runtime_body_order.txt"),
}
TOPP_CLOSURE_PATHS = (
    Path("hope_training/whole_body_tracking/scripts/topp_mintime.py"),
    Path("hope_training/whole_body_tracking/scripts/synthesize_timing.py"),
    Path("hope_training/whole_body_tracking/scripts/synthesize_timing_v2.py"),
    Path("hope_training/whole_body_tracking/scripts/audit_motion_npz.py"),
    Path("hope_training/whole_body_tracking/scripts/csv_to_npz_mujoco.py"),
    Path("hope_training/whole_body_tracking/scripts/racket_geometry_contract.py"),
    Path("hope_training/whole_body_tracking/scripts/motion_kinematics_contract.py"),
    Path("hope_training/whole_body_tracking/scripts/hope_frame_utils.py"),
    Path("hope_training/whole_body_tracking/scripts/a3_joint_order_contract.py"),
    Path("scripts/feasibility_oracle.py"),
)
TOPP_CERTIFICATE_KEYS = {
    "tool", "algorithm_scope", "search_objective", "generated_utc", "verdict",
    "direction", "chosen_scale", "feasible_reason", "budget", "acceptance",
    "durations", "oracle_before", "oracle_after", "kin", "source", "answer",
    "baseline_law", "stretch", "output", "timing_bound", "fidelity",
    "outer_trace", "inner_trace_best", "files", "budget_provenance",
    "runtime_provenance",
}
TOPP_TOOL = "topp_mintime.py v3 (unified-budget min-time bidirectional retiming)"
TOPP_ALGORITHM_SCOPE = (
    "heuristic upper bound within the sampled gamma ladder plus greedy local repair; "
    "not strict TOPP and not a global minimum proof"
)


class Stage2Error(ValueError):
    """Activation, evidence, namespace, subprocess, or result contract failed."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    payload: bytes
    mode: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage2Error(message)


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False,
                       separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _reject_constant(value: str) -> None:
    raise Stage2Error(f"JSON contains non-finite constant {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), parse_constant=_reject_constant,
                          object_pairs_hook=_pairs_no_duplicates)
    except Stage2Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2Error(f"cannot parse {label}: {exc}") from exc


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    actual = set(value)
    _require(actual == expected,
             f"{label} keys changed: missing={sorted(expected - actual)} "
             f"unexpected={sorted(actual - expected)}")
    return value


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    _require(type(value) in (int, float), f"{label} must be a number, not bool")
    number = float(value)
    _require(math.isfinite(number), f"{label} must be finite")
    if minimum is not None:
        _require(number >= minimum, f"{label} must be >= {minimum}")
    return number


def _ensure_no_symlink_components(path: Path, label: str,
                                  *, leaf_may_be_missing: bool = False) -> None:
    absolute = _absolute(path)
    current = Path(absolute.parts[0])
    for index, part in enumerate(absolute.parts[1:], start=1):
        current /= part
        leaf = index == len(absolute.parts) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            if leaf and leaf_may_be_missing:
                return
            raise Stage2Error(f"{label} component is missing: {current}")
        _require(not stat.S_ISLNK(info.st_mode), f"{label} contains symlink: {current}")


def _read_snapshot(path: Path | str, label: str) -> Snapshot:
    absolute = _absolute(path)
    _ensure_no_symlink_components(absolute, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(absolute, flags)
    except OSError as exc:
        raise Stage2Error(f"cannot open {label}: {absolute}: {exc}") from exc
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size,
                           before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns, after.st_ctime_ns)
        _require(identity_before == identity_after, f"{label} changed while reading")
        payload = b"".join(chunks)
        _require(len(payload) == before.st_size, f"{label} short read")
        return Snapshot(absolute, payload, stat.S_IMODE(before.st_mode))
    finally:
        os.close(fd)


_TOPP_RUNTIME_PROBE = r"""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import mujoco
import numpy

def package_record(name, module):
    distribution = metadata.distribution(name)
    files = list(distribution.files or ())
    selected = {}
    for leaf in ("METADATA", "RECORD", "WHEEL"):
        matches = [entry for entry in files if str(entry).endswith(".dist-info/" + leaf)]
        if len(matches) != 1:
            raise RuntimeError(f"{name} has {len(matches)} {leaf} records")
        selected[leaf.lower()] = str(Path(distribution.locate_file(matches[0])).resolve())
    return {
        "version": distribution.version,
        "module": str(Path(module.__file__).resolve()),
        **selected,
    }

print(json.dumps({
    "python_version": ".".join(str(value) for value in sys.version_info[:3]),
    "executable": str(Path(sys.executable).absolute()),
    "prefix": str(Path(sys.prefix).resolve()),
    "packages": {
        "numpy": package_record("numpy", numpy),
        "mujoco": package_record("mujoco", mujoco),
    },
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""

_MJCF_PREFLIGHT_PROBE = r"""
import json
import os
from pathlib import Path
import sys
import mujoco

model = mujoco.MjModel.from_xml_path(sys.argv[1])
candidate_paths = set()
for module in tuple(sys.modules.values()):
    path = getattr(module, "__file__", None)
    if isinstance(path, str) and os.path.isabs(path):
        candidate_paths.add(path)
with open("/proc/self/maps", "r", encoding="utf-8") as stream:
    for line in stream:
        fields = line.rstrip("\n").split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            candidate_paths.add(fields[5])
loaded_elf_paths = set()
for raw_path in sorted(candidate_paths):
    if raw_path.endswith(" (deleted)"):
        loaded_elf_paths.add(raw_path)
        continue
    try:
        with open(raw_path, "rb") as stream:
            if stream.read(4) == b"\x7fELF":
                loaded_elf_paths.add(os.path.realpath(raw_path))
    except OSError:
        loaded_elf_paths.add(raw_path + " (unreadable)")
print(json.dumps({
    "nq": int(model.nq),
    "nv": int(model.nv),
    "nbody": int(model.nbody),
    "ngeom": int(model.ngeom),
    "nmesh": int(model.nmesh),
    "loaded_elf_paths": sorted(loaded_elf_paths),
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


def _topp_runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    for key in tuple(env):
        if key.startswith("LD_"):
            env.pop(key, None)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    return env


def _run_runtime_command(command: Sequence[str], *, cwd: Path,
                         env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=cwd, env=dict(env), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, check=False, timeout=120,
    )


def _verify_distribution_record(*, package_name: str, version: str,
                                site_packages: Path, venv_root: Path,
                                record_snapshot: Snapshot) -> tuple[
                                    dict[str, Any], dict[str, Snapshot]
                                ]:
    """Verify and content-address every file named by a wheel RECORD."""
    try:
        text = record_snapshot.payload.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise Stage2Error(f"TOPP runtime {package_name} RECORD is invalid: {exc}") from exc
    _require(rows, f"TOPP runtime {package_name} RECORD is empty")
    record_relative = f"{package_name}-{version}.dist-info/RECORD"
    seen: set[str] = set()
    manifest: list[dict[str, Any]] = []
    native_elfs: dict[str, Snapshot] = {}
    self_rows = 0
    explicitly_bound_unhashed_rows = 0
    destinations: set[str] = set()
    for index, row in enumerate(rows, start=1):
        _require(len(row) == 3,
                 f"TOPP runtime {package_name} RECORD row {index} must have three fields")
        raw_path, hash_field, size_field = row
        _require(raw_path != "" and "\\" not in raw_path and "\0" not in raw_path,
                 f"TOPP runtime {package_name} RECORD row {index} has invalid path")
        relative = PurePosixPath(raw_path)
        _require(not relative.is_absolute() and all(part not in ("", ".") for part in relative.parts),
                 f"TOPP runtime {package_name} RECORD row {index} contains absolute or empty path")
        normalized = relative.as_posix()
        _require(raw_path == normalized,
                 f"TOPP runtime {package_name} RECORD row {index} has noncanonical path")
        _require(normalized not in seen,
                 f"TOPP runtime {package_name} RECORD has duplicate path {normalized}")
        seen.add(normalized)
        path = _absolute(os.path.normpath(os.path.join(site_packages, *relative.parts)))
        try:
            destination_relative = path.relative_to(venv_root).as_posix()
        except ValueError as exc:
            raise Stage2Error(
                f"TOPP runtime {package_name} RECORD row {index} escapes the fixed venv"
            ) from exc
        _require(str(path) not in destinations,
                 f"TOPP runtime {package_name} RECORD has duplicate canonical destination {path}")
        destinations.add(str(path))
        snapshot = _read_snapshot(path, f"TOPP runtime {package_name} RECORD {normalized}")
        if normalized == record_relative:
            self_rows += 1
            _require(hash_field == "" and size_field == "",
                     f"TOPP runtime {package_name} RECORD self row must use its bound empty fields")
            _require(snapshot.payload == record_snapshot.payload,
                     f"TOPP runtime {package_name} RECORD self row bytes changed")
            expected_sha = record_snapshot.sha256
        else:
            if hash_field == "" and size_field == "":
                explicitly_bound_unhashed_rows += 1
                expected_sha = snapshot.sha256
                manifest.append({
                    "path": normalized, "venv_relative_path": destination_relative,
                    "bytes": len(snapshot.payload), "sha256": snapshot.sha256,
                    "record_hash_present": False,
                })
                if snapshot.payload.startswith(b"\x7fELF"):
                    native_elfs[str(path)] = snapshot
                continue
            _require(hash_field != "" and size_field != "",
                     f"TOPP runtime {package_name} RECORD row {normalized} has partial hash or size")
            _require(hash_field.startswith("sha256=") and hash_field.count("=") == 1,
                     f"TOPP runtime {package_name} RECORD row {normalized} uses unsupported hash")
            encoded = hash_field.split("=", 1)[1]
            _require(encoded != "", f"TOPP runtime {package_name} RECORD row {normalized} has empty hash")
            try:
                expected_digest = base64.b64decode(
                    encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
                expected_size = int(size_field, 10)
            except (ValueError, base64.binascii.Error) as exc:
                raise Stage2Error(
                    f"TOPP runtime {package_name} RECORD row {normalized} is malformed"
                ) from exc
            _require(len(expected_digest) == 32 and expected_size >= 0
                     and str(expected_size) == size_field,
                     f"TOPP runtime {package_name} RECORD row {normalized} has noncanonical hash or size")
            canonical_digest = base64.urlsafe_b64encode(
                expected_digest).rstrip(b"=").decode("ascii")
            _require(encoded == canonical_digest,
                     f"TOPP runtime {package_name} RECORD row {normalized} has noncanonical hash or size")
            _require(len(snapshot.payload) == expected_size,
                     f"TOPP runtime {package_name} RECORD row {normalized} size changed")
            expected_sha = expected_digest.hex()
            _require(snapshot.sha256 == expected_sha,
                     f"TOPP runtime {package_name} RECORD row {normalized} SHA changed")
        manifest.append({
            "path": normalized, "venv_relative_path": destination_relative,
            "bytes": len(snapshot.payload), "sha256": snapshot.sha256,
            "record_hash_present": normalized != record_relative,
        })
        if snapshot.payload.startswith(b"\x7fELF"):
            native_elfs[str(path)] = snapshot
    _require(self_rows == 1,
             f"TOPP runtime {package_name} RECORD must contain exactly one self row")
    manifest.sort(key=lambda row: row["path"])
    return ({
        "record_sha256": record_snapshot.sha256,
        "file_count": len(manifest),
        "total_bytes": sum(row["bytes"] for row in manifest),
        "verified_manifest_sha256": _sha256(_canonical_json(manifest)),
        "native_elf_count": len(native_elfs),
        "explicitly_bound_unhashed_row_count": explicitly_bound_unhashed_rows,
        "record_self_row_empty_and_explicitly_bound": True,
    }, native_elfs)


def _run_ldd_command(command: Sequence[str], *, cwd: Path,
                     env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=cwd, env=dict(env), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, check=False, timeout=60,
    )


def _run_readelf_command(command: Sequence[str], *, cwd: Path,
                         env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=cwd, env=dict(env), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, check=False, timeout=60,
    )


def _dynamic_dependency_env() -> dict[str, str]:
    env = _topp_runtime_env()
    env["LC_ALL"] = "C"
    return env


def _parse_ldd_output(payload: str, *, source: Path) -> tuple[
    list[tuple[str, str]], list[str], list[str]
]:
    resolved: list[tuple[str, str]] = []
    virtual: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=>" in line:
            soname, remainder = (part.strip() for part in line.split("=>", 1))
            _require(soname != "" and soname not in seen,
                     f"ldd returned duplicate or empty dependency for {source}")
            seen.add(soname)
            if remainder == "not found":
                unresolved.append(soname)
                continue
            _require(not remainder.startswith("not found"),
                     f"ldd dependency {soname} has malformed not-found output for {source}")
            reported = remainder.rsplit(" (", 1)[0].strip()
            _require(Path(reported).is_absolute(),
                     f"ldd dependency {soname} is not absolute for {source}")
            resolved.append((soname, reported))
            continue
        token = line.split(" ", 1)[0]
        _require(token not in seen,
                 f"ldd returned duplicate dependency {token} for {source}")
        seen.add(token)
        if token == "linux-vdso.so.1":
            virtual.append(token)
            continue
        _require(Path(token).is_absolute(),
                 f"ldd returned unparsed output for {source}: {line}")
        resolved.append((Path(token).name, token))
    _require(resolved or unresolved,
             f"ldd returned no resolved or unresolved dependencies for {source}")
    return resolved, virtual, unresolved


def _verify_static_elf_has_no_needed(*, source: Path, readelf_path: Path) -> dict[str, Any]:
    command = [str(readelf_path), "-d", str(source)]
    try:
        completed = _run_readelf_command(
            command, cwd=source.parent, env=_dynamic_dependency_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Stage2Error(f"readelf failed to run for {source}: {exc}") from exc
    _require(completed.returncode == 0,
             f"readelf failed rc={completed.returncode} for {source}: {completed.stderr.strip()}")
    _require(completed.stderr == "", f"readelf wrote stderr for {source}")
    lines = completed.stdout.splitlines()
    if lines and lines[0] == "":
        lines = lines[1:]
    dynamic_entries: list[str] = []
    if lines == ["There is no dynamic section in this file."]:
        section_kind = "no_dynamic_section"
    else:
        _require(lines and re.fullmatch(
            r"Dynamic section at offset 0x[0-9a-fA-F]+ contains [0-9]+ entries:",
            lines[0]) is not None,
            f"readelf dynamic-section header is malformed for {source}")
        expected_count = int(lines[0].rsplit(" ", 2)[1])
        _require(len(lines) >= 2 and lines[1].split() == ["Tag", "Type", "Name/Value"],
                 f"readelf dynamic-section columns are malformed for {source}")
        for line in lines[2:]:
            if line == "":
                continue
            match = re.fullmatch(
                r"\s*0x[0-9a-fA-F]+ \(([A-Z0-9_]+)\)\s+.*", line)
            _require(match is not None,
                     f"readelf dynamic-section row is malformed for {source}: {line}")
            dynamic_entries.append(match.group(1))
        _require(len(dynamic_entries) == expected_count,
                 f"readelf dynamic-section entry count changed for {source}")
        section_kind = "dynamic_section_without_needed"
    _require("NEEDED" not in dynamic_entries,
             f"ldd claimed static but readelf found NEEDED for {source}")
    return {
        "argv": command, "returncode": completed.returncode,
        "stdout_sha256": _sha256(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256(completed.stderr.encode("utf-8")),
        "section_kind": section_kind,
        "dynamic_entry_count": len(dynamic_entries), "needed_count": 0,
    }


def _observe_dynamic_dependency_closure(
    *, elf_inputs: Mapping[str, Snapshot], ldd_path: Path | str,
    readelf_path: Path | str,
    allowed_virtual_dependencies: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ldd_path = _absolute(ldd_path)
    ldd_snapshot = _read_snapshot(ldd_path, "reviewed ldd tool")
    readelf_path = _absolute(readelf_path)
    readelf_snapshot = _read_snapshot(readelf_path, "reviewed readelf tool")
    allowed_virtual = list(allowed_virtual_dependencies)
    _require(allowed_virtual == ["linux-vdso.so.1"],
             "TOPP dynamic dependency virtual allowlist changed")
    sources: list[dict[str, Any]] = []
    dependency_files: dict[str, dict[str, Any]] = {}
    edge_count = 0
    loaded_by_soname: dict[str, list[str]] = {}
    for loaded_path in elf_inputs:
        canonical = str(_absolute(loaded_path))
        loaded_by_soname.setdefault(Path(canonical).name, []).append(canonical)
    for source_path_text, source_snapshot in sorted(elf_inputs.items()):
        source_path = _absolute(source_path_text)
        _require(source_snapshot.payload.startswith(b"\x7fELF"),
                 f"dynamic dependency input is not ELF: {source_path}")
        current_source = _read_snapshot(
            source_path, f"unchanged loaded ELF before ldd {source_path}")
        _require(current_source.payload == source_snapshot.payload,
                 f"loaded ELF changed before ldd: {source_path}")
        command = [str(ldd_path), str(source_path)]
        try:
            completed = _run_ldd_command(
                command, cwd=source_path.parent, env=_dynamic_dependency_env())
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise Stage2Error(f"ldd failed to run for {source_path}: {exc}") from exc
        _require(completed.returncode == 0,
                 f"ldd failed rc={completed.returncode} for {source_path}: {completed.stderr.strip()}")
        _require(completed.stderr == "", f"ldd wrote stderr for {source_path}")
        static_verification: dict[str, Any] | None = None
        if completed.stdout == "\tstatically linked\n":
            dependencies: list[tuple[str, str]] = []
            virtual: list[str] = []
            unresolved: list[str] = []
            linkage_kind = "static_no_dependencies"
            static_verification = _verify_static_elf_has_no_needed(
                source=source_path, readelf_path=readelf_path)
        else:
            dependencies, virtual, unresolved = _parse_ldd_output(
                completed.stdout, source=source_path)
            linkage_kind = "dynamic"
        current_source = _read_snapshot(
            source_path, f"unchanged loaded ELF after ldd {source_path}")
        _require(current_source.payload == source_snapshot.payload,
                 f"loaded ELF changed while running ldd: {source_path}")
        _require(set(virtual) <= set(allowed_virtual),
                 f"ldd returned unreviewed virtual dependency for {source_path}")
        edges: list[dict[str, str]] = []
        resolved_dependencies = [
            (soname, reported, "ldd_absolute")
            for soname, reported in dependencies
        ]
        for soname in unresolved:
            matches = loaded_by_soname.get(soname, [])
            _require(len(matches) == 1,
                     f"ldd unresolved dependency {soname} for {source_path} "
                     "does not uniquely match an actual loaded ELF")
            resolved_dependencies.append(
                (soname, matches[0], "actual_loaded_unique_soname"))
        for soname, reported, resolution_kind in resolved_dependencies:
            resolved_path = _absolute(os.path.realpath(reported))
            snapshot = _read_snapshot(resolved_path, f"ldd dependency {soname}")
            row = {
                "path": str(resolved_path), "bytes": len(snapshot.payload),
                "sha256": snapshot.sha256,
            }
            existing = dependency_files.setdefault(str(resolved_path), row)
            _require(existing == row,
                     f"dynamic dependency file changed while collecting: {resolved_path}")
            edges.append({
                "soname": soname, "resolved_path": str(resolved_path),
                "resolution_kind": resolution_kind,
            })
        edges.sort(key=lambda row: (
            row["soname"], row["resolved_path"], row["resolution_kind"]))
        edge_count += len(edges)
        sources.append({
            "path": str(source_path), "bytes": len(source_snapshot.payload),
            "sha256": source_snapshot.sha256, "dependencies": edges,
            "virtual_dependencies": sorted(virtual),
            "linkage_kind": linkage_kind,
            "static_verification": static_verification,
        })
    dependency_rows = sorted(dependency_files.values(), key=lambda row: row["path"])
    manifest = {
        "ldd": {"path": str(ldd_path), "sha256": ldd_snapshot.sha256},
        "readelf": {"path": str(readelf_path), "sha256": readelf_snapshot.sha256},
        "sources": sources,
        "resolved_files": dependency_rows,
    }
    current_ldd = _read_snapshot(ldd_path, "unchanged reviewed ldd tool")
    _require(current_ldd.payload == ldd_snapshot.payload,
             "reviewed ldd tool changed while collecting dependencies")
    current_readelf = _read_snapshot(readelf_path, "unchanged reviewed readelf tool")
    _require(current_readelf.payload == readelf_snapshot.payload,
             "reviewed readelf tool changed while collecting dependencies")
    manifest_sha = _sha256(_canonical_json(manifest))
    receipt = {
        "ldd": manifest["ldd"],
        "readelf": manifest["readelf"],
        "elf_input_count": len(sources),
        "resolved_file_count": len(dependency_rows),
        "edge_count": edge_count,
        "manifest_sha256": manifest_sha,
        "allowed_virtual_dependencies": list(allowed_virtual),
    }
    return receipt, manifest


def _collect_dynamic_dependency_closure(
    *, elf_inputs: Mapping[str, Snapshot], contract: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _exact_keys(contract, {
        "ldd_path", "ldd_sha256", "readelf_path", "readelf_sha256",
        "allowed_virtual_dependencies",
        "elf_input_count", "resolved_file_count", "edge_count", "manifest_sha256",
    }, "TOPP dynamic dependency contract")
    ldd_snapshot = _read_snapshot(contract["ldd_path"], "reviewed ldd tool")
    _require(ldd_snapshot.sha256 == contract["ldd_sha256"],
             "reviewed ldd tool SHA changed")
    readelf_snapshot = _read_snapshot(contract["readelf_path"], "reviewed readelf tool")
    _require(readelf_snapshot.sha256 == contract["readelf_sha256"],
             "reviewed readelf tool SHA changed")
    receipt, _manifest = _observe_dynamic_dependency_closure(
        elf_inputs=elf_inputs, ldd_path=contract["ldd_path"],
        readelf_path=contract["readelf_path"],
        allowed_virtual_dependencies=contract["allowed_virtual_dependencies"],
    )
    _require(receipt["ldd"]["sha256"] == contract["ldd_sha256"],
             "reviewed ldd tool SHA changed")
    _require(receipt["readelf"]["sha256"] == contract["readelf_sha256"],
             "reviewed readelf tool SHA changed")
    _require(receipt["elf_input_count"] == contract["elf_input_count"],
             "TOPP dynamic dependency ELF input count changed")
    _require(receipt["resolved_file_count"] == contract["resolved_file_count"],
             "TOPP dynamic dependency resolved file count changed")
    _require(receipt["edge_count"] == contract["edge_count"],
             "TOPP dynamic dependency edge count changed")
    _require(receipt["manifest_sha256"] == contract["manifest_sha256"],
             "TOPP dynamic dependency manifest changed")
    return receipt


def _inspect_topp_runtime(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Snapshot]]:
    """Validate the exact interpreter and package files used by TOPP.

    The venv entry-point symlink text is deliberately evidence, not identity:
    relative and absolute links can name the same final interpreter.  Identity
    is the canonical final path plus its bytes, Python version, venv prefix,
    distribution RECORD closures, and native ELF closure.  The outer symlink
    and final target are nevertheless snapshotted on both sides of the whole
    inspection so a concurrent retarget or binary replacement fails closed.
    """
    _require(contract == EXPECTED_TOPP_RUNTIME, "TOPP runtime contract changed")
    interpreter_contract = _exact_keys(contract["interpreter"], {
        "path", "canonical_realpath", "binary_sha256", "python_version",
        "venv_prefix",
    }, "TOPP runtime interpreter")
    interpreter = _absolute(interpreter_contract["path"])
    _ensure_no_symlink_components(interpreter.parent, "TOPP interpreter parent")

    def identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (info.st_dev, info.st_ino, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)

    expected_realpath = _absolute(interpreter_contract["canonical_realpath"])
    _require(Path(interpreter_contract["canonical_realpath"]).is_absolute()
             and _absolute(os.path.realpath(expected_realpath)) == expected_realpath,
             "TOPP interpreter canonical realpath contract is not canonical")
    try:
        link_info_before = interpreter.lstat()
        observed_link_target = os.readlink(interpreter)
        canonical_before = _absolute(os.path.realpath(interpreter))
        link_info_before_confirm = interpreter.lstat()
        observed_link_target_confirm = os.readlink(interpreter)
    except (FileNotFoundError, OSError) as exc:
        raise Stage2Error(f"cannot inspect TOPP interpreter symlink: {exc}") from exc
    _require(stat.S_ISLNK(link_info_before.st_mode),
             "TOPP interpreter must be a symlink")
    _require(identity(link_info_before) == identity(link_info_before_confirm)
             and observed_link_target == observed_link_target_confirm,
             "TOPP interpreter symlink changed while taking the initial snapshot")
    _require(canonical_before == expected_realpath,
             "TOPP interpreter canonical realpath changed")
    try:
        target_info_before = expected_realpath.lstat()
    except OSError as exc:
        raise Stage2Error(f"cannot inspect TOPP interpreter target: {exc}") from exc
    target_snapshot = _read_snapshot(expected_realpath, "TOPP interpreter target")
    _require(target_snapshot.sha256 == interpreter_contract["binary_sha256"],
             "TOPP interpreter binary SHA changed")
    command = [str(interpreter), "-I", "-B", "-c", _TOPP_RUNTIME_PROBE]
    try:
        completed = _run_runtime_command(
            command, cwd=interpreter.parent, env=_topp_runtime_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Stage2Error(f"TOPP runtime probe could not run: {exc}") from exc
    _require(completed.returncode == 0,
             f"TOPP runtime probe failed rc={completed.returncode}: {completed.stderr.strip()}")
    _require(completed.stderr == "", "TOPP runtime probe wrote stderr")
    probe = _exact_keys(_load_json(completed.stdout.encode("utf-8"), "TOPP runtime probe"), {
        "python_version", "executable", "prefix", "packages",
    }, "TOPP runtime probe")
    _require(probe["python_version"] == interpreter_contract["python_version"],
             "TOPP Python version changed")
    _require(_absolute(probe["executable"]) == interpreter,
             "TOPP probe ran under the wrong interpreter")
    venv_root = _absolute(interpreter_contract["venv_prefix"])
    _require(Path(interpreter_contract["venv_prefix"]).is_absolute()
             and venv_root == interpreter.parent.parent,
             "TOPP interpreter venv prefix contract changed")
    _require(_absolute(probe["prefix"]) == venv_root,
             "TOPP interpreter does not use the expected venv prefix")
    site_packages = venv_root / "lib" / "python3.12" / "site-packages"
    expected_paths = {
        "numpy": {
            "module": site_packages / "numpy" / "__init__.py",
            "metadata": site_packages / "numpy-2.5.0.dist-info" / "METADATA",
            "record": site_packages / "numpy-2.5.0.dist-info" / "RECORD",
            "wheel": site_packages / "numpy-2.5.0.dist-info" / "WHEEL",
        },
        "mujoco": {
            "module": site_packages / "mujoco" / "__init__.py",
            "metadata": site_packages / "mujoco-3.10.0.dist-info" / "METADATA",
            "record": site_packages / "mujoco-3.10.0.dist-info" / "RECORD",
            "wheel": site_packages / "mujoco-3.10.0.dist-info" / "WHEEL",
        },
    }
    packages = _exact_keys(probe["packages"], {"numpy", "mujoco"},
                           "TOPP runtime packages")
    package_receipts: dict[str, Any] = {}
    snapshots = {"topp_runtime:interpreter_target": target_snapshot}
    for package_name in ("numpy", "mujoco"):
        package_contract = _exact_keys(contract["packages"][package_name], {
            "version", "module_sha256", "metadata_sha256", "record_sha256",
            "wheel_sha256", "record_file_count", "record_total_bytes",
            "record_manifest_sha256", "record_native_elf_count",
            "record_unhashed_row_count",
        }, f"TOPP runtime {package_name} contract")
        package_probe = _exact_keys(packages[package_name], {
            "version", "module", "metadata", "record", "wheel",
        }, f"TOPP runtime {package_name} probe")
        _require(package_probe["version"] == package_contract["version"],
                 f"TOPP runtime {package_name} version changed")
        receipt_files: dict[str, Any] = {}
        for label in ("module", "metadata", "record", "wheel"):
            path = _absolute(package_probe[label])
            _require(path == expected_paths[package_name][label],
                     f"TOPP runtime {package_name} {label} path changed")
            snapshot = _read_snapshot(path, f"TOPP runtime {package_name} {label}")
            expected_sha = package_contract[f"{label}_sha256"]
            _require(snapshot.sha256 == expected_sha,
                     f"TOPP runtime {package_name} {label} SHA changed")
            snapshots[f"topp_runtime:{package_name}:{label}"] = snapshot
            receipt_files[label] = {"path": str(path), "sha256": snapshot.sha256}
        package_receipts[package_name] = {
            "version": package_probe["version"], "files": receipt_files,
        }
        record_receipt, _package_elfs = _verify_distribution_record(
            package_name=package_name,
            version=package_probe["version"],
            site_packages=site_packages,
            venv_root=venv_root,
            record_snapshot=snapshots[f"topp_runtime:{package_name}:record"],
        )
        _require(record_receipt["file_count"] == package_contract["record_file_count"],
                 f"TOPP runtime {package_name} RECORD file count changed")
        _require(record_receipt["total_bytes"] == package_contract["record_total_bytes"],
                 f"TOPP runtime {package_name} RECORD total bytes changed")
        _require(record_receipt["verified_manifest_sha256"]
                 == package_contract["record_manifest_sha256"],
                 f"TOPP runtime {package_name} RECORD manifest changed")
        _require(record_receipt["native_elf_count"]
                 == package_contract["record_native_elf_count"],
                 f"TOPP runtime {package_name} RECORD native ELF count changed")
        _require(record_receipt["explicitly_bound_unhashed_row_count"]
                 == package_contract["record_unhashed_row_count"],
                 f"TOPP runtime {package_name} RECORD unhashed row count changed")
        package_receipts[package_name]["record_closure"] = record_receipt

    try:
        link_info_after = interpreter.lstat()
        observed_link_target_after = os.readlink(interpreter)
        canonical_after = _absolute(os.path.realpath(interpreter))
        target_info_after = expected_realpath.lstat()
        target_snapshot_after = _read_snapshot(
            expected_realpath, "TOPP interpreter target after runtime inspection")
    except OSError as exc:
        raise Stage2Error(f"cannot re-inspect TOPP interpreter runtime: {exc}") from exc
    _require(identity(link_info_before) == identity(link_info_after)
             and observed_link_target == observed_link_target_after,
             "TOPP interpreter symlink changed while inspecting")
    _require(canonical_after == canonical_before == expected_realpath,
             "TOPP interpreter canonical realpath changed while inspecting")
    _require(identity(target_info_before) == identity(target_info_after)
             and target_snapshot_after.payload == target_snapshot.payload
             and target_snapshot_after.mode == target_snapshot.mode,
             "TOPP interpreter binary changed while inspecting")
    receipt = {
        "interpreter": {
            "path": str(interpreter),
            "observed_symlink_target": observed_link_target,
            "canonical_realpath": str(expected_realpath),
            "binary_sha256": target_snapshot.sha256,
            "python_version": probe["python_version"],
            "venv_prefix": str(venv_root),
            "symlink_identity": {
                "device": link_info_before.st_dev, "inode": link_info_before.st_ino,
                "size": link_info_before.st_size,
                "mtime_ns": link_info_before.st_mtime_ns,
                "ctime_ns": link_info_before.st_ctime_ns,
            },
            "binary_identity": {
                "device": target_info_before.st_dev, "inode": target_info_before.st_ino,
                "size": target_info_before.st_size,
                "mtime_ns": target_info_before.st_mtime_ns,
                "ctime_ns": target_info_before.st_ctime_ns,
            },
        },
        "packages": package_receipts,
        "probe_argv": command,
        "probe_rc": completed.returncode,
        "probe_stdout_sha256": _sha256(completed.stdout.encode("utf-8")),
        "pythonpath_removed": True,
        "pythonhome_removed": True,
    }
    return receipt, snapshots


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        _require(not stat.S_ISLNK(info.st_mode),
                 f"MJCF preflight tree contains symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            rows.append({"path": relative, "kind": "directory",
                         "mode": stat.S_IMODE(info.st_mode)})
        else:
            _require(stat.S_ISREG(info.st_mode),
                     f"MJCF preflight tree contains special file: {relative}")
            snapshot = _read_snapshot(path, f"MJCF preflight file {relative}")
            rows.append({"path": relative, "kind": "file", "mode": snapshot.mode,
                         "bytes": len(snapshot.payload), "sha256": snapshot.sha256})
    return rows


def _observe_mjcf_runtime_preflight(*, runtime_snapshot_root: Path,
                                    contract: Mapping[str, Any]) -> tuple[
                                        dict[str, Any], dict[str, Snapshot]
                                    ]:
    """Load the exact MJCF and return the actually mapped ELF set."""
    interpreter = _absolute(contract["interpreter"]["path"])
    mjcf = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["mjcf_sha256"]
    _require(mjcf.is_file(), "O_EXCL MJCF snapshot is missing")
    before = _tree_manifest(runtime_snapshot_root)
    command = [str(interpreter), "-I", "-B", "-c", _MJCF_PREFLIGHT_PROBE, str(mjcf)]
    try:
        completed = _run_runtime_command(
            command, cwd=runtime_snapshot_root, env=_topp_runtime_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Stage2Error(f"MJCF runtime preflight could not run: {exc}") from exc
    _require(completed.returncode == 0,
             f"MJCF runtime preflight failed rc={completed.returncode}: {completed.stderr.strip()}")
    _require(completed.stderr == "", "MJCF runtime preflight wrote stderr")
    probe = _exact_keys(
        _load_json(completed.stdout.encode("utf-8"), "MJCF runtime preflight"),
        {"nq", "nv", "nbody", "ngeom", "nmesh", "loaded_elf_paths"},
        "MJCF runtime preflight result",
    )
    dimensions = {key: probe[key] for key in ("nq", "nv", "nbody", "ngeom", "nmesh")}
    expected = _exact_keys(contract["mjcf_model"], {
        "loader", "nq", "nv", "nbody", "ngeom", "nmesh",
    }, "MJCF runtime model contract")
    _require(expected["loader"] == "mujoco.MjModel.from_xml_path",
             "MJCF runtime loader changed")
    _require(dimensions == {key: expected[key] for key in dimensions},
             "MJCF runtime model dimensions changed")
    loaded_paths = probe["loaded_elf_paths"]
    _require(isinstance(loaded_paths, list) and loaded_paths
             and all(isinstance(path, str) and path for path in loaded_paths),
             "MJCF runtime loaded ELF path list is malformed or empty")
    _require(loaded_paths == sorted(set(loaded_paths)),
             "MJCF runtime loaded ELF paths are duplicate or noncanonical")
    loaded_elfs: dict[str, Snapshot] = {}
    venv_root = _absolute(contract["interpreter"]["path"]).parent.parent
    interpreter_target = _absolute(contract["interpreter"]["canonical_realpath"])
    for raw_path in loaded_paths:
        _require(not raw_path.endswith(" (deleted)")
                 and not raw_path.endswith(" (unreadable)")
                 and not raw_path.startswith("["),
                 f"MJCF runtime has deleted, unreadable, or anonymous ELF mapping: {raw_path}")
        path = _absolute(raw_path)
        _require(Path(raw_path).is_absolute(),
                 f"MJCF runtime loaded ELF path is not absolute: {raw_path}")
        resolved = _absolute(os.path.realpath(path))
        _require(path == resolved,
                 f"MJCF runtime loaded ELF path is not canonical: {raw_path}")
        snapshot = _read_snapshot(resolved, f"MJCF runtime loaded ELF {resolved}")
        _require(snapshot.payload.startswith(b"\x7fELF"),
                 f"MJCF runtime reported a non-ELF mapping: {resolved}")
        _require(str(resolved) not in loaded_elfs,
                 f"MJCF runtime loaded ELF resolves twice: {resolved}")
        loaded_elfs[str(resolved)] = snapshot
    _require(str(interpreter_target) in loaded_elfs,
             "MJCF runtime did not report the fixed interpreter ELF")
    _require(any(path.startswith(str(venv_root / "lib" / "python3.12" / "site-packages" / "numpy") + "/")
                 for path in loaded_elfs),
             "MJCF runtime did not load a NumPy native ELF")
    _require(any(path.startswith(str(venv_root / "lib" / "python3.12" / "site-packages" / "mujoco") + "/")
                 for path in loaded_elfs),
             "MJCF runtime did not load a MuJoCo native ELF")
    _require(any(Path(path).name.startswith("libmujoco.so") for path in loaded_elfs),
             "MJCF runtime did not load libmujoco")
    after = _tree_manifest(runtime_snapshot_root)
    _require(after == before,
             "MJCF runtime preflight created or changed snapshot files")
    return ({
        "loader": expected["loader"], "argv": command,
        "returncode": completed.returncode, "dimensions": dict(dimensions),
        "loaded_elf_paths": sorted(loaded_elfs),
        "loaded_elf_count": len(loaded_elfs),
        "stdout_sha256": _sha256(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256(completed.stderr.encode("utf-8")),
        "snapshot_tree_before_sha256": _sha256(_canonical_json(before)),
        "snapshot_tree_after_sha256": _sha256(_canonical_json(after)),
        "output_files_created": [],
    }, loaded_elfs)


def _preflight_mjcf_runtime(*, runtime_snapshot_root: Path,
                            contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact MJCF behavior and its actually loaded native closure."""
    receipt, loaded_elfs = _observe_mjcf_runtime_preflight(
        runtime_snapshot_root=runtime_snapshot_root, contract=contract)
    receipt["dynamic_dependencies"] = _collect_dynamic_dependency_closure(
        elf_inputs=loaded_elfs, contract=contract["dynamic_dependencies"])
    return receipt


def _git_blob_oid(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _safe_repo_relative(value: str, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and value != "", f"{label} must be non-empty")
    _require("\\" not in value and "\0" not in value,
             f"{label} contains a forbidden separator or NUL")
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"{label} must be relative")
    _require(all(part not in ("", ".", "..") for part in path.parts),
             f"{label} contains traversal or empty components")
    return path


def _run_git_readonly(runtime_root: Path, arguments: Sequence[str], label: str) -> bytes:
    env = dict(os.environ)
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
    })
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "--no-optional-locks", "-C",
             str(runtime_root), *arguments],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, env=env, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Stage2Error(f"cannot run read-only git {label}: {exc}") from exc
    _require(completed.returncode == 0,
             f"read-only git {label} failed rc={completed.returncode}: "
             f"{completed.stderr.decode('utf-8', errors='replace').strip()}")
    return completed.stdout


def _collect_mjcf_mesh_closure(*, runtime_root: Path, checkout_commit: str,
                               mjcf_relative: Path,
                               mjcf_snapshot: Snapshot) -> tuple[dict[str, Any], dict[str, Snapshot]]:
    """Bind every MJCF-referenced mesh to the exact runtime Git commit."""
    _require(isinstance(checkout_commit, str) and len(checkout_commit) == 40
             and all(char in "0123456789abcdef" for char in checkout_commit),
             "runtime checkout commit is malformed")
    head = _run_git_readonly(runtime_root, ["rev-parse", "--verify", "HEAD^{commit}"],
                             "HEAD verification").decode("ascii", errors="strict").strip()
    _require(head == checkout_commit, "runtime checkout HEAD differs from frozen commit")
    _require(b"<!DOCTYPE" not in mjcf_snapshot.payload
             and b"<!ENTITY" not in mjcf_snapshot.payload,
             "MJCF external or internal entity declarations are forbidden")
    try:
        xml_root = ET.fromstring(mjcf_snapshot.payload)
    except ET.ParseError as exc:
        raise Stage2Error(f"cannot parse frozen MJCF XML: {exc}") from exc
    elements = list(xml_root.iter())
    _require(all(isinstance(element.tag, str) and "}" not in element.tag
                 for element in elements), "MJCF namespaces are not supported")
    compilers = [element for element in elements if element.tag == "compiler"]
    _require(len(compilers) == 1, "MJCF must contain exactly one compiler element")
    meshdir_raw = compilers[0].get("meshdir")
    _require(meshdir_raw == "meshes", "MJCF compiler meshdir changed from meshes")
    meshdir = _safe_repo_relative(meshdir_raw, "MJCF compiler meshdir")
    for element in elements:
        if "file" in element.attrib:
            _require(element.tag == "mesh",
                     f"unsupported external MJCF file reference on <{element.tag}>")
    files: list[PurePosixPath] = []
    for element in elements:
        if element.tag != "mesh":
            continue
        file_value = element.get("file")
        _require(file_value is not None, "MJCF mesh without file is unsupported")
        files.append(_safe_repo_relative(file_value, "MJCF mesh file"))
    _require(files, "MJCF contains no file-backed meshes")
    _require(len(files) == len(set(files)), "MJCF contains duplicate mesh file references")
    model_dir = PurePosixPath(mjcf_relative.as_posix()).parent
    tree_oid = _run_git_readonly(
        runtime_root,
        ["rev-parse", f"{checkout_commit}:{model_dir.as_posix()}"],
        "MJCF model-root tree binding",
    ).decode("ascii", errors="strict").strip()
    _require(tree_oid == EXPECTED_MJCF_MODEL_TREE_OID,
             "MJCF model-root Git tree differs from the frozen tree")
    mesh_repo_paths = [model_dir / meshdir / value for value in sorted(files, key=str)]
    repo_paths = [PurePosixPath(mjcf_relative.as_posix()), *mesh_repo_paths]
    tree = _run_git_readonly(
        runtime_root,
        ["ls-tree", "-rz", "--full-tree", checkout_commit, "--",
         *(path.as_posix() for path in repo_paths)],
        "MJCF mesh tree binding",
    )
    entries: dict[str, tuple[str, str, str]] = {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, oid = metadata.decode("ascii").split(" ")
            path_text = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise Stage2Error("cannot parse git ls-tree MJCF closure output") from exc
        _require(path_text not in entries, "duplicate path in git MJCF closure output")
        entries[path_text] = (mode, object_type, oid)
    _require(set(entries) == {path.as_posix() for path in repo_paths},
             "Git tree does not contain the complete MJCF mesh closure")
    snapshots: dict[str, Snapshot] = {}
    closure_rows: list[dict[str, Any]] = []
    git_rows: list[dict[str, str]] = []
    for relative in repo_paths:
        relative_text = relative.as_posix()
        mode, object_type, oid = entries[relative_text]
        _require(mode == "100644" and object_type == "blob" and len(oid) == 40,
                 f"MJCF mesh has unsupported Git entry: {relative_text}")
        object_payload = _run_git_readonly(runtime_root, ["cat-file", "blob", oid],
                                           f"MJCF object {relative_text}")
        _require(_git_blob_oid(object_payload) == oid,
                 f"Git returned corrupt MJCF blob bytes: {relative_text}")
        working = _read_snapshot(runtime_root / relative_text,
                                 f"working MJCF closure {relative_text}")
        _require(working.payload == object_payload,
                 f"working MJCF file differs from frozen Git object: {relative_text}")
        object_snapshot = Snapshot(working.path, object_payload, working.mode)
        if relative != PurePosixPath(mjcf_relative.as_posix()):
            snapshots[f"runtime:{relative_text}"] = object_snapshot
        else:
            _require(object_payload == mjcf_snapshot.payload,
                     "queue-bound MJCF bytes differ from frozen Git object")
        model_relative = relative.relative_to(model_dir).as_posix()
        closure_rows.append({
            "path": model_relative,
            "bytes": len(object_payload),
            "sha256": _sha256(object_payload),
        })
        git_rows.append({"path": model_relative, "git_blob_sha1": oid})
    closure_rows.sort(key=lambda row: row["path"])
    git_rows.sort(key=lambda row: row["path"])
    manifest_payload = json.dumps(
        closure_rows, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    manifest_sha = _sha256(manifest_payload)
    total_bytes = sum(row["bytes"] for row in closure_rows)
    _require(len(closure_rows) == EXPECTED_MJCF_CLOSURE_FILE_COUNT,
             "MJCF closure file count changed")
    _require(total_bytes == EXPECTED_MJCF_CLOSURE_TOTAL_BYTES,
             "MJCF closure total bytes changed")
    _require(manifest_sha == EXPECTED_MJCF_CLOSURE_MANIFEST_SHA256,
             "MJCF closure manifest changed")
    return ({
        "checkout_commit": checkout_commit,
        "model_root_git_tree_oid": tree_oid,
        "mjcf_relative_path": mjcf_relative.as_posix(),
        "compiler_meshdir": meshdir.as_posix(),
        "file_count": len(closure_rows),
        "mesh_count": len(mesh_repo_paths),
        "total_bytes": total_bytes,
        "mesh_manifest_sha256": manifest_sha,
        "files": closure_rows,
        "git_blob_manifest_sha256": _sha256(_canonical_json(git_rows)),
        "git_blobs": git_rows,
    }, snapshots)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    _ensure_no_symlink_components(path.parent, "output parent")
    _ensure_no_symlink_components(path, "output", leaf_may_be_missing=True)
    _require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, mode)
    except OSError as exc:
        raise Stage2Error(f"cannot publish no-clobber output {path}: {exc}") from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_queue(queue: Any) -> Mapping[str, Any]:
    queue = _exact_keys(queue, {
        "schema_version", "experiment_id", "created_utc", "human_owner", "executor",
        "purpose", "runtime", "assets", "fixed_contract",
        "observed_baseline_not_to_rerun", "staged_cells", "derived_cell_rule",
        "acceptance", "selection",
    }, "queue")
    _require(queue["schema_version"] == 1, "queue schema changed")
    _require(queue["experiment_id"] == EXPECTED_EXPERIMENT_ID, "queue experiment changed")
    runtime = _exact_keys(queue["runtime"], {
        "checkout_path", "checkout_commit", "generator_source_commit", "generator_sha256",
        "topp_sha256", "mjcf_sha256", "urdf_sha256", "body_order_sha256",
    }, "queue.runtime")
    _require(Path(runtime["checkout_path"]).is_absolute(), "runtime checkout must be absolute")
    for key in (*RUNTIME_RELATIVE_PATHS, "generator_sha256"):
        _require(_is_sha256(runtime[key]), f"runtime {key} SHA is malformed")
    assets = _exact_keys(queue["assets"], {"forehand", "backhand"}, "queue.assets")
    for name, asset in assets.items():
        asset = _exact_keys(asset, {"path", "sha256", "contact_frame"}, f"asset {name}")
        _require(Path(asset["path"]).is_absolute(), f"asset {name} path must be absolute")
        _require(_is_sha256(asset["sha256"]), f"asset {name} SHA is malformed")
        _require(type(asset["contact_frame"]) is int and asset["contact_frame"] > 17,
                 f"asset {name} contact frame is invalid")
    fixed_expected = {
        "fps": 50, "ready_frame": 0, "ready_velocity": "bitwise_zero",
        "hold_frames": 4, "output_contact_frame": 25,
        "protected_precontact_seconds": 0.1, "delta_plus_blend_intervals": 22,
        "topp_objective": "runup", "body_mode": "fk", "automatic_retry": False,
        "gpu_or_trainer_signals": False, "training_authorized": False,
        "deployment_authorized": False, "hardware_authorized": False,
    }
    _require(queue["fixed_contract"] == fixed_expected, "queue fixed contract changed")
    _require(queue["derived_cell_rule"] == {
        "join_frame": "contact_frame - delta", "blend_intervals": "22 - delta",
        "output_contact_frame": 25,
    }, "queue derived-cell rule changed")
    staged = _exact_keys(queue["staged_cells"], {
        "stage1_endpoint_factorial", "stage2_midpoint_rule", "stage3_refinement_rule",
    }, "queue.staged_cells")
    stage1 = staged["stage1_endpoint_factorial"]
    _require(isinstance(stage1, list) and len(stage1) == 6, "queue must register six Stage1 cells")
    actual_stage1: dict[str, tuple[str, str, int]] = {}
    for row in stage1:
        row = _exact_keys(row, {"cell_id", "action", "ready_source", "delta"}, "Stage1 cell")
        _require(row["cell_id"] not in actual_stage1, "duplicate Stage1 cell")
        actual_stage1[row["cell_id"]] = (row["action"], row["ready_source"], row["delta"])
    _require(actual_stage1 == EXPECTED_STAGE1_CELLS, "Stage1 cell registry changed")
    midpoint = staged["stage2_midpoint_rule"]
    _require(isinstance(midpoint, Mapping) and midpoint.get("delta") == 12,
             "Stage2 midpoint rule changed")
    return queue


def _validate_activation(activation: Any, queue: Mapping[str, Any], *,
                         runner_sha256: str) -> tuple[Mapping[str, Any], list[dict[str, Any]], Path, str]:
    activation = _exact_keys(activation, {
        "schema_version", "activation_id", "created_utc", "parent_queue",
        "stage1_namespace", "observations", "decision", "evidence_status",
        "required_attestation_receipt", "required_attestation_receipt_sha256",
        "stage2_runner", "stage2_namespace", "launch_authorized",
        "authorized_stage2_cells", "prior_failed_attempt",
        "topp_runtime", "runtime_authority",
    }, "activation")
    _require(activation["schema_version"] == 1, "activation schema changed")
    _require(activation["activation_id"] == EXPECTED_ACTIVATION_ID, "unknown activation id")
    _require(activation["parent_queue"] == {
        "path": "configs/ready_to_strike_join_ladder_20260717.yaml",
        "sha256": EXPECTED_QUEUE_SHA256,
        "main_commit": EXPECTED_PREREG_COMMIT,
    }, "activation parent queue changed")
    _require(activation["evidence_status"] == EXPECTED_EVIDENCE_STATUS,
             "Stage1 evidence is not historically attested screening-only")
    _require(activation["launch_authorized"] is True, "Stage2 launch is not authorized")
    _require(activation["stage2_runner"] == {
        "path": "scripts/run_ready_to_strike_join_ladder_stage2.py",
        "sha256": runner_sha256,
    }, "activation does not bind the executing Stage2 runner bytes")
    stage2_namespace = _absolute(activation["stage2_namespace"])
    _require(str(stage2_namespace) == EXPECTED_STAGE2_NAMESPACE,
             "Stage2 namespace differs from the source-pinned one-shot namespace")
    _require(activation["prior_failed_attempt"] == EXPECTED_PRIOR_ATTEMPT,
             "prior failed attempt binding changed")
    _require(activation["topp_runtime"] == EXPECTED_TOPP_RUNTIME,
             "activation TOPP runtime contract changed")
    runtime_authority = {
        "cpu_only": True, "automatic_retry": False, "trainer_signal": False,
        "robot_command": False, "training_authorized": False,
        "deployment_authorized": False,
    }
    _require(activation["runtime_authority"] == runtime_authority,
             "activation runtime authority changed")
    expected_decision = {
        "ready_by_side_crossover": True, "forehand_prefers": "forehand",
        "backhand_prefers": "backhand",
        "delta17_strictly_worse_than_delta6_all_four_ready_by_side_pairs": True,
        "activate_both_ready_sources_at_midpoint": True,
        "shared_ready_not_yet_selected": True,
    }
    _require(activation["decision"] == expected_decision, "activation decision changed")
    observations = activation["observations"]
    _require(isinstance(observations, list) and len(observations) == 8,
             "activation must contain eight Stage1 observations")
    observed: dict[str, tuple[str, str, int, str, str, float]] = {}
    for row in observations:
        row = _exact_keys(row, {"cell_id", "action", "ready_source", "delta",
                                "candidate_sha256", "topp_certificate_sha256",
                                "start_to_contact_s"}, "activation observation")
        identity = (row["action"], row["ready_source"], row["delta"],
                    row["candidate_sha256"], row["topp_certificate_sha256"],
                    _finite(row["start_to_contact_s"], "observation timing", minimum=0.0))
        _require(row["cell_id"] not in observed, "duplicate activation observation")
        observed[row["cell_id"]] = identity
    _require(observed == EXPECTED_OBSERVATIONS, "activation Stage1 observations changed")
    cells = activation["authorized_stage2_cells"]
    _require(isinstance(cells, list) and len(cells) == 4,
             "activation must contain four Stage2 cells")
    actual_cells: dict[str, tuple[str, str, int]] = {}
    normalized: list[dict[str, Any]] = []
    for row in cells:
        row = _exact_keys(row, {"cell_id", "action", "ready_source", "delta"}, "Stage2 cell")
        identity = (row["action"], row["ready_source"], row["delta"])
        _require(row["cell_id"] not in actual_cells, "duplicate Stage2 cell")
        actual_cells[row["cell_id"]] = identity
        normalized.append(dict(row))
    _require(actual_cells == EXPECTED_STAGE2_CELLS, "Stage2 cell set changed")
    receipt_path = _absolute(activation["required_attestation_receipt"])
    stage1_root = _absolute(activation["stage1_namespace"])
    _require(receipt_path == stage1_root / "stage1_historical_attestation.json",
             "attestation receipt path is not inside the declared Stage1 namespace")
    receipt_sha = activation["required_attestation_receipt_sha256"]
    _require(_is_sha256(receipt_sha), "attestation receipt SHA is missing or malformed")
    return activation, normalized, receipt_path, receipt_sha


def _validate_stage1_receipt(document: Any, *, receipt_sha: str,
                             receipt_payload: bytes, queue: Mapping[str, Any],
                             activation: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(_sha256(receipt_payload) == receipt_sha, "Stage1 receipt bytes do not match activation SHA")
    receipt = _exact_keys(document, {
        "schema_version", "artifact_kind", "status", "experiment_id", "inputs",
        "cells", "attestor", "formal_claims", "runtime_authority",
    }, "Stage1 receipt")
    _require(receipt["schema_version"] == 1, "Stage1 receipt schema changed")
    _require(receipt["artifact_kind"] == "ready_to_strike_stage1_historical_attestation",
             "Stage1 receipt kind changed")
    _require(receipt["status"] == "historical_evidence_attested_no_runtime_authority",
             "Stage1 receipt status changed")
    _require(receipt["experiment_id"] == EXPECTED_EXPERIMENT_ID,
             "Stage1 receipt experiment changed")
    inputs = _exact_keys(receipt["inputs"], {
        "root", "queue_path", "queue_sha256", "generator_sha256", "summary_sha256",
        "runtime_source_commit",
    }, "Stage1 receipt inputs")
    _require(inputs["root"] == activation["stage1_namespace"], "Stage1 receipt root is misbound")
    _require(inputs["queue_sha256"] == EXPECTED_QUEUE_SHA256, "Stage1 receipt queue is misbound")
    _require(inputs["generator_sha256"] == queue["runtime"]["generator_sha256"],
             "Stage1 receipt generator is misbound")
    _require(inputs["runtime_source_commit"] == queue["runtime"]["checkout_commit"],
             "Stage1 receipt runtime commit is misbound")
    _require(_is_sha256(inputs["summary_sha256"]), "Stage1 summary SHA is malformed")
    _require(receipt["formal_claims"] == {
        "physics_replay_exact": False, "source_closure_exact": False,
        "mjcf_closure_exact": False, "screening_activation_evidence_only": True,
    }, "Stage1 receipt formal claims changed")
    _require(receipt["runtime_authority"] == {
        "read_only_historical": True, "ssh": False, "process_signal": False,
        "automatic_retry": False, "simulator": False, "trainer": False,
        "deployment": False, "robot_command": False,
    }, "Stage1 receipt runtime authority changed")
    attestor = _exact_keys(receipt["attestor"], {
        "source_sha256", "launch_snapshot", "launch_snapshot_sha256",
        "source_and_inputs_unchanged_before_publish",
    }, "Stage1 receipt attestor")
    _require(_is_sha256(attestor["source_sha256"]), "attestor source SHA is malformed")
    _require(attestor["source_and_inputs_unchanged_before_publish"] is True,
             "Stage1 inputs were not stable before receipt publication")
    _require(attestor["launch_snapshot_sha256"] == _sha256(_canonical_json(attestor["launch_snapshot"])),
             "Stage1 attestor launch snapshot is misbound")
    observation_by_id = {row["cell_id"]: row for row in activation["observations"]}
    cells = receipt["cells"]
    _require(isinstance(cells, list) and len(cells) == 6, "Stage1 receipt must bind six cells")
    actual: dict[str, tuple[str, str, int]] = {}
    for row in cells:
        row = _exact_keys(row, {
            "cell_id", "action", "ready_source", "delta", "join_frame",
            "blend_intervals", "candidate_sha256", "generator_contract_sha256",
            "output_sha256", "certificate_sha256", "candidate_start_to_contact_s",
            "within_0p5_s", "doses",
        }, "Stage1 receipt cell")
        cell_id = row["cell_id"]
        _require(cell_id in EXPECTED_STAGE1_CELLS and cell_id not in actual,
                 f"unexpected or duplicate Stage1 receipt cell {cell_id!r}")
        identity = (row["action"], row["ready_source"], row["delta"])
        _require(identity == EXPECTED_STAGE1_CELLS[cell_id], f"Stage1 receipt cell {cell_id} changed")
        actual[cell_id] = identity
        observation = observation_by_id[cell_id]
        _require(row["candidate_sha256"] == observation["candidate_sha256"],
                 f"Stage1 receipt candidate SHA for {cell_id} is misbound")
        _require(row["certificate_sha256"] == observation["topp_certificate_sha256"],
                 f"Stage1 receipt certificate SHA for {cell_id} is misbound")
        for key in ("generator_contract_sha256", "output_sha256"):
            _require(_is_sha256(row[key]), f"Stage1 receipt {cell_id}.{key} is malformed")
        contact = queue["assets"][row["action"]]["contact_frame"]
        _require(row["join_frame"] == contact - row["delta"], "Stage1 join frame is misbound")
        _require(row["blend_intervals"] == 22 - row["delta"], "Stage1 blend is misbound")
        timing = _finite(row["candidate_start_to_contact_s"], "Stage1 receipt timing", minimum=0.0)
        _require(type(row["within_0p5_s"]) is bool and row["within_0p5_s"] == (timing <= 0.5),
                 "Stage1 within-0.5 flag is misbound")
        doses = _exact_keys(row["doses"], {"cop", "friction", "torque"}, "Stage1 doses")
        for name, value in doses.items():
            _finite(value, f"Stage1 {cell_id} {name} dose", minimum=0.0)
    _require(actual == EXPECTED_STAGE1_CELLS, "Stage1 receipt cell set is incomplete")
    return receipt


def _validate_prior_failure(document: Any, *, payload: bytes,
                            expected_receipt_sha: str) -> Mapping[str, Any]:
    prior = EXPECTED_PRIOR_ATTEMPT
    _require(_sha256(payload) == prior["summary_sha256"],
             "prior Stage2 failure summary bytes changed")
    summary = _exact_keys(document, {
        "schema_version", "artifact_kind", "status", "activation_sha256",
        "queue_sha256", "stage1_receipt_sha256", "runner_sha256",
        "prior_failed_attempt_summary_sha256",
        "runtime_snapshot_shas", "asset_snapshot_shas", "rows",
        "screening_acceptance", "input_stability_errors", "formal_claims",
        "runtime_authority", "automatic_retry", "reviewed_child_timeout_s",
        "trainer_or_robot_signals",
    }, "prior Stage2 failure summary")
    _require(summary["schema_version"] == 1, "prior Stage2 schema changed")
    _require(summary["artifact_kind"] == "ready_to_strike_join_ladder_stage2_screening_result",
             "prior Stage2 artifact kind changed")
    _require(summary["status"] == "stage2_terminal_failure_no_retry",
             "prior Stage2 status changed")
    _require(summary["activation_sha256"] == prior["activation_sha256"],
             "prior Stage2 activation is misbound")
    _require(summary["runner_sha256"] == prior["runner_sha256"],
             "prior Stage2 runner is misbound")
    _require(summary["queue_sha256"] == EXPECTED_QUEUE_SHA256,
             "prior Stage2 queue is misbound")
    _require(summary["stage1_receipt_sha256"] == expected_receipt_sha,
             "prior Stage2 receipt is misbound")
    _require(summary["prior_failed_attempt_summary_sha256"]
             == EXPECTED_PRIOR_V1_SUMMARY_SHA256,
             "prior Stage2 does not bind the frozen V1 failure summary")
    for label in ("runtime_snapshot_shas", "asset_snapshot_shas"):
        values = summary[label]
        _require(isinstance(values, Mapping) and values,
                 f"prior Stage2 {label} is empty")
        _require(all(isinstance(key, str) and _is_sha256(value)
                     for key, value in values.items()),
                 f"prior Stage2 {label} is malformed")
    rows = summary["rows"]
    _require(isinstance(rows, list) and len(rows) == 4,
             "prior Stage2 row count changed")
    actual: dict[str, tuple[str, str, int]] = {}
    for row in rows:
        row = _exact_keys(row, {
            "cell_id", "action", "ready_source", "delta", "join_frame",
            "blend_intervals", "generator_rc", "candidate_sha256",
            "generator_contract_sha256", "frames", "phase", "joint_path_l2",
            "joint_curvature_l2", "max_joint_step_rad", "topp_rc",
        }, "prior Stage2 row")
        cell_id = row["cell_id"]
        _require(cell_id in EXPECTED_STAGE2_CELLS and cell_id not in actual,
                 f"unexpected or duplicate prior Stage2 cell {cell_id!r}")
        identity = (row["action"], row["ready_source"], row["delta"])
        _require(identity == EXPECTED_STAGE2_CELLS[cell_id],
                 f"prior Stage2 cell {cell_id} changed")
        actual[cell_id] = identity
        _require(row["generator_rc"] == 0 and row["blend_intervals"] == 10,
                 f"prior Stage2 cell {cell_id} did not finish generation")
        expected_join = (66 if row["action"] == "forehand" else 45) - 12
        _require(row["join_frame"] == expected_join,
                 f"prior Stage2 cell {cell_id} join changed")
        expected_candidate, expected_contract = EXPECTED_PRIOR_CANDIDATES[cell_id]
        _require(row["candidate_sha256"] == expected_candidate
                 and row["generator_contract_sha256"] == expected_contract,
                 f"prior Stage2 cell {cell_id} candidate lineage changed")
        _require(type(row["frames"]) is int and row["frames"] > 25,
                 f"prior Stage2 cell {cell_id} frame count is invalid")
        for key in ("phase", "joint_path_l2", "joint_curvature_l2",
                    "max_joint_step_rad"):
            _finite(row[key], f"prior Stage2 {cell_id} {key}", minimum=0.0)
        _require(row["topp_rc"] == 1,
                 f"prior Stage2 cell {cell_id} did not fail in TOPP as recorded")
    _require(actual == EXPECTED_STAGE2_CELLS, "prior Stage2 cell set changed")
    _require(summary["screening_acceptance"] == {
        "at_or_below_0p5_cells": [], "timing_by_cell_s": {},
        "shared_ready_two_side_at_or_below_0p5": {
            "backhand": False, "forehand": False,
        },
        "any_shared_ready_pass": False,
    }, "prior Stage2 screening outcome changed")
    _require(summary["input_stability_errors"] == [],
             "prior Stage2 inputs were unstable")
    _require(summary["formal_claims"] == {
        "physics_replay_exact": False, "source_closure_exact": False,
        "mjcf_closure_exact": False, "screening_evidence_only": True,
        "strict_global_minimum_proven": False,
    }, "prior Stage2 formal claims changed")
    _require(summary["runtime_authority"] == {
        "cpu_only": True, "automatic_retry": False, "trainer_signal": False,
        "robot_command": False, "training_authorized": False,
        "deployment_authorized": False,
    }, "prior Stage2 runtime authority changed")
    _require(summary["automatic_retry"] is False
             and summary["reviewed_child_timeout_s"] == 3600
             and summary["trainer_or_robot_signals"] == [],
             "prior Stage2 retry/signal record changed")
    return summary


def _load_npz(payload: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)), f"{label} contains duplicate ZIP entries")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise Stage2Error(f"cannot parse {label}: {exc}") from exc
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number):
            _require(bool(np.all(np.isfinite(value))), f"{label}.{key} contains non-finite values")
    return arrays


def _scalar_unicode(array: np.ndarray, label: str) -> str:
    value = np.asarray(array)
    _require(value.shape == () and value.dtype.kind == "U" and not value.dtype.hasobject,
             f"{label} must be one canonical Unicode scalar")
    return str(value.item())


def _validate_schema2(arrays: Mapping[str, np.ndarray], *, label: str,
                      body_order: Sequence[str], allow_migration: bool,
                      gradient_contract: str) -> int:
    time_keys = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                 "body_lin_vel_w", "body_ang_vel_w"}
    metadata_keys = {"fps", "kinematics_schema_version", "body_pos_point",
                     "body_lin_vel_point", "body_names"}
    migration_keys = {"kinematics_migration_source_sha256",
                      "kinematics_migration_source_point", "kinematics_migration_tool"}
    actual = set(arrays)
    core = time_keys | metadata_keys
    allowed = {frozenset(core)}
    if allow_migration:
        allowed.add(frozenset(core | migration_keys))
    _require(frozenset(actual) in allowed,
             f"{label} schema keys changed: {sorted(actual)}")
    fps = np.asarray(arrays["fps"])
    schema = np.asarray(arrays["kinematics_schema_version"])
    _require(fps.dtype == np.int64 and fps.shape == (1,) and int(fps[0]) == 50,
             f"{label}.fps changed")
    _require(schema.dtype == np.int64 and schema.shape == (1,) and int(schema[0]) == 2,
             f"{label}.kinematics_schema_version changed")
    _require(_scalar_unicode(arrays["body_pos_point"], f"{label}.body_pos_point") == "link_origin",
             f"{label}.body_pos_point changed")
    _require(_scalar_unicode(arrays["body_lin_vel_point"], f"{label}.body_lin_vel_point") == "center_of_mass",
             f"{label}.body_lin_vel_point changed")
    names = np.asarray(arrays["body_names"])
    _require(names.dtype.kind == "U" and names.shape == (len(body_order),)
             and tuple(str(value) for value in names.tolist()) == tuple(body_order),
             f"{label}.body_names differs from runtime order")
    q = np.asarray(arrays["joint_pos"])
    _require(q.dtype == np.float32 and q.ndim == 2 and q.shape[0] > 26 and q.shape[1] == 31,
             f"{label}.joint_pos shape/dtype changed")
    frames, bodies = q.shape[0], len(body_order)
    expected_shapes = {
        "joint_pos": (frames, 31), "joint_vel": (frames, 31),
        "body_pos_w": (frames, bodies, 3), "body_quat_w": (frames, bodies, 4),
        "body_lin_vel_w": (frames, bodies, 3), "body_ang_vel_w": (frames, bodies, 3),
    }
    for key, shape in expected_shapes.items():
        value = np.asarray(arrays[key])
        _require(value.dtype == np.float32 and value.shape == shape,
                 f"{label}.{key} shape/dtype changed")
    _require(gradient_contract in {"float32_producer", "float64_workspace"},
             f"unknown gradient contract {gradient_contract!r}")
    gradient_input = q if gradient_contract == "float32_producer" else q.astype(np.float64)
    expected_velocity = np.gradient(gradient_input, 1.0 / 50.0, axis=0).astype(np.float32)
    _require(np.array_equal(np.asarray(arrays["joint_vel"]), expected_velocity),
             f"{label}.joint_vel is not the canonical position gradient")
    quaternion_norm = np.linalg.norm(np.asarray(arrays["body_quat_w"], dtype=np.float64), axis=-1)
    _require(float(np.max(np.abs(quaternion_norm - 1.0))) <= 2.0e-5,
             f"{label}.body_quat_w is not normalized")
    if migration_keys <= actual:
        _require(_is_sha256(_scalar_unicode(arrays["kinematics_migration_source_sha256"],
                                             f"{label}.migration SHA")),
                 f"{label} migration SHA is malformed")
        _require(_scalar_unicode(arrays["kinematics_migration_source_point"],
                                 f"{label}.migration point") in {"link_origin", "center_of_mass"},
                 f"{label} migration point changed")
        _require(_scalar_unicode(arrays["kinematics_migration_tool"],
                                 f"{label}.migration tool") == "migrate_motion_kinematics.py/v2",
                 f"{label} migration tool changed")
    return frames


def _validate_candidate(candidate: Snapshot, contract: Snapshot, *, cell: Mapping[str, Any],
                        queue: Mapping[str, Any], generator_path: Path,
                        asset_paths: Mapping[str, Path],
                        body_order: Sequence[str]) -> dict[str, Any]:
    arrays = _load_npz(candidate.payload, f"{cell['cell_id']} candidate")
    frames = _validate_schema2(arrays, label=f"{cell['cell_id']} candidate",
                               body_order=body_order, allow_migration=True,
                               gradient_contract="float32_producer")
    q = np.asarray(arrays["joint_pos"])
    qd = np.asarray(arrays["joint_vel"])
    for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        _require(np.array_equal(np.asarray(arrays[key])[:3], np.zeros_like(arrays[key][:3])),
                 f"candidate {key} does not start bitwise zero")
    document = _exact_keys(_load_json(contract.payload, "candidate contract"), {
        "schema_version", "artifact_kind", "status", "inputs", "tool", "request",
        "synthesis", "proof", "output", "authorization", "required_next_gates",
        "explicit_non_claims",
    }, "candidate contract")
    _require(document["schema_version"] == 1, "candidate contract schema changed")
    inputs = _exact_keys(document["inputs"], {
        "source_schema2_npz", "shared_ready_schema2_npz", "shared_ready_frame",
    }, "candidate inputs")
    _require(inputs["shared_ready_frame"] == 0, "candidate ready frame changed")
    for evidence, asset_name, evidence_label in (
        (inputs["source_schema2_npz"], cell["action"], "source"),
        (inputs["shared_ready_schema2_npz"], cell["ready_source"], "ready"),
    ):
        evidence = _exact_keys(evidence, {
            "path", "bytes", "sha256", "device", "inode", "mtime_ns", "ctime_ns",
        }, f"candidate {evidence_label} evidence")
        asset = queue["assets"][asset_name]
        _require(evidence["path"] == str(asset_paths[asset_name])
                 and evidence["sha256"] == asset["sha256"],
                 f"candidate {evidence_label} asset is misbound")
    proof = document["proof"]
    required_proof = {
        "fps": 50, "source_contact_frame": queue["assets"][cell["action"]]["contact_frame"],
        "source_join_frame": queue["assets"][cell["action"]]["contact_frame"] - 12,
        "output_contact_frame": 25, "protected_frames_before_contact": 5,
        "protected_window_bitwise_equal": True,
        "pose_and_body_velocity_source_suffix_bitwise_equal": True,
        "frame0_shared_ready_pose_bitwise_equal": True,
        "ready_source_velocity_channels_ignored": True,
        "ready_velocity_definition": "explicit_bitwise_zero",
        "initial_zero_velocity_frames": 3,
        "joint_position_continuous_quintic_endpoint_c2": True,
        "finite": True, "contact_time_from_frame0_s": 0.5,
    }
    for key, value in required_proof.items():
        _require(proof.get(key) == value, f"candidate proof {key} changed")
    _require(document["request"] == {
        "source_contact_frame": queue["assets"][cell["action"]]["contact_frame"],
        "source_join_frame": queue["assets"][cell["action"]]["contact_frame"] - 12,
        "ready_hold_frames": 4, "quintic_blend_intervals": 10,
        "protected_precontact_seconds": 0.1,
    }, "candidate request changed")
    _require(document["tool"].get("path") == str(generator_path)
             and document["tool"].get("sha256") == queue["runtime"]["generator_sha256"],
             "candidate generator evidence is misbound")
    _require(document["output"]["npz"].get("sha256") == candidate.sha256,
             "candidate contract does not bind candidate bytes")
    authorization = document["authorization"]
    _require(authorization.get("training_authorized") is False
             and authorization.get("deployment_authorized") is False
             and authorization.get("hardware_authorized") is False,
             "candidate authorization escaped screening")
    segment = np.diff(q[:26].astype(np.float64), axis=0)
    second = np.diff(q[:26].astype(np.float64), n=2, axis=0)
    return {
        "candidate_sha256": candidate.sha256,
        "generator_contract_sha256": contract.sha256,
        "frames": frames,
        "phase": 25.0 / float(frames - 1),
        "joint_path_l2": float(np.linalg.norm(segment, axis=1).sum()),
        "joint_curvature_l2": float(np.linalg.norm(second, axis=1).sum()),
        "max_joint_step_rad": float(np.abs(segment).max()),
    }


def _validate_topp(certificate: Snapshot, output: Snapshot, markdown: Snapshot, *,
                   candidate: Snapshot, candidate_info: Mapping[str, Any],
                   queue: Mapping[str, Any], paths: Mapping[str, Path],
                   body_order: Sequence[str]) -> dict[str, Any]:
    document = _exact_keys(_load_json(certificate.payload, "TOPP certificate"),
                           TOPP_CERTIFICATE_KEYS, "TOPP certificate")
    _require(document["tool"] == TOPP_TOOL, "TOPP tool/schema identity changed")
    _require(document["algorithm_scope"] == TOPP_ALGORITHM_SCOPE,
             "TOPP algorithm-scope honesty changed")
    _require(document["search_objective"] == "runup", "TOPP objective changed")
    files = _exact_keys(document["files"], {"input", "output", "report_path", "markdown_path"},
                       "TOPP files")
    _require(files["input"].get("sha256") == candidate.sha256, "TOPP input is misbound")
    _require(files["output"].get("sha256") == output.sha256, "TOPP output is misbound")
    _require(files["report_path"] == str(certificate.path), "TOPP report path is misbound")
    _require(files["markdown_path"] == str(markdown.path), "TOPP markdown path is misbound")
    arrays = _load_npz(output.payload, "TOPP output")
    output_frames = _validate_schema2(arrays, label="TOPP output", body_order=body_order,
                                      allow_migration=False,
                                      gradient_contract="float64_workspace")
    candidate_arrays = _load_npz(candidate.payload, "TOPP candidate input")
    acceptance = _exact_keys(document["acceptance"], {
        "cop_dose_final", "fric_dose_final", "tau_dose_final", "within_budget",
        "kin_out_window_clean", "kin_lock_window_clean", "kinematic_hard_limits_clean",
    }, "TOPP acceptance")
    for key in ("within_budget", "kin_out_window_clean", "kin_lock_window_clean",
                "kinematic_hard_limits_clean"):
        _require(acceptance[key] is True, f"TOPP {key} failed")
    source_doc = _exact_keys(document["source"], {
        "frames", "fps", "contact_frame", "phase", "runup_s", "duration_s",
        "clean_blade_speed_mps", "mean_abs_acc",
    }, "TOPP source")
    output_doc = _exact_keys(document["output"], {
        "frames", "fps", "contact_frame", "phase_out", "runup_s", "duration_s",
        "runup_change_x", "duration_change_x", "wait_s", "body_mode", "mean_abs_acc",
    }, "TOPP output")
    _require(output_doc.get("body_mode") == "fk", "TOPP did not use production FK")
    _require(output_doc.get("frames") == output_frames and output_doc.get("fps") == 50.0,
             "TOPP output frame/fps tuple is misbound")
    _require(source_doc["frames"] == candidate_info["frames"]
             and source_doc["fps"] == 50 and source_doc["contact_frame"] == 25
             and math.isclose(_finite(source_doc["phase"], "TOPP source phase"),
                              candidate_info["phase"], rel_tol=0.0, abs_tol=1e-12),
             "TOPP source tuple is misbound")
    source_runup = _finite(source_doc["runup_s"], "TOPP source runup", minimum=0.0)
    _require(math.isclose(source_runup,
                          round(source_doc["contact_frame"] / source_doc["fps"], 4),
                          rel_tol=0.0, abs_tol=5.1e-5),
             "TOPP source runup disagrees with contact frame/fps")
    output_contact = output_doc["contact_frame"]
    _require(type(output_contact) is int and 0 <= output_contact < output_frames,
             "TOPP output contact frame is invalid")
    _require(math.isclose(_finite(output_doc["phase_out"], "TOPP output phase"),
                          output_contact / float(output_frames - 1),
                          rel_tol=0.0, abs_tol=5.1e-7), "TOPP output phase is misbound")
    output_runup = _finite(output_doc["runup_s"], "TOPP output runup", minimum=0.0)
    _require(math.isclose(output_runup,
                          round(output_contact / float(output_doc["fps"]), 4),
                          rel_tol=0.0, abs_tol=5.1e-5),
             "TOPP output runup disagrees with contact frame/fps")
    _require(np.array_equal(np.asarray(arrays["joint_pos"])[output_contact],
                            np.asarray(candidate_arrays["joint_pos"])[25]),
             "TOPP output contact row differs from candidate")
    fidelity = _exact_keys(document["fidelity"], {
        "contact_row_bitwise", "blade_speed_clean_out_mps", "blade_speed_dev_frac",
        "face_normal_diff_deg", "first_frame_max_joint_vel",
    }, "TOPP fidelity")
    _require(fidelity["contact_row_bitwise"] is True, "TOPP contact row changed")
    _require(_finite(fidelity["blade_speed_dev_frac"], "TOPP blade-speed deviation", minimum=0.0) <= 0.02,
             "TOPP blade-speed deviation exceeds 2 percent")
    _require(_finite(fidelity["first_frame_max_joint_vel"], "TOPP first velocity", minimum=0.0) == 0.0,
             "TOPP output does not start at rest")
    _require(float(np.max(np.abs(np.asarray(arrays["joint_vel"])[0]))) == 0.0,
             "TOPP NPZ does not start at rest")
    timing = _exact_keys(document["timing_bound"], {
        "candidate_start_to_contact_s", "bound_semantics", "strict_global_minimum_proven",
    }, "TOPP timing bound")
    timing_s = _finite(timing["candidate_start_to_contact_s"], "TOPP timing", minimum=0.0)
    _require(timing["strict_global_minimum_proven"] is False, "TOPP falsely claims global minimum")
    _require(math.isclose(timing_s, output_runup, rel_tol=0.0, abs_tol=1e-12),
             "TOPP timing disagrees with output")
    provenance = document["runtime_provenance"]
    for name in ("mjcf", "urdf", "body_order"):
        _require(provenance[name].get("sha256") == queue["runtime"][f"{name}_sha256"],
                 f"TOPP {name} evidence is misbound")
    _require(provenance["tool"]["topp_mintime"].get("sha256") == queue["runtime"]["topp_sha256"],
             "TOPP tool evidence is misbound")
    budget_provenance = _exact_keys(document["budget_provenance"],
                                    {"clips", "scale", "envelope"},
                                    "TOPP budget provenance")
    _require(_finite(budget_provenance["scale"], "TOPP budget scale", minimum=0.0) == 1.5,
             "TOPP budget scale changed from the source-pinned default 1.5")
    _require(isinstance(budget_provenance["envelope"], list)
             and budget_provenance["envelope"], "TOPP budget envelope is empty")
    for index, value in enumerate(budget_provenance["envelope"]):
        _finite(value, f"TOPP budget envelope {index}", minimum=0.0)
    budget_clips = budget_provenance["clips"]
    _require(isinstance(budget_clips, list) and len(budget_clips) == 2,
             "TOPP budget clip count changed")
    for evidence, name in zip(budget_clips, ("forehand", "backhand")):
        _require(evidence.get("sha256") == queue["assets"][name]["sha256"],
                 f"TOPP budget clip {name} is misbound")
    doses = {
        "cop": _finite(acceptance["cop_dose_final"], "CoP dose", minimum=0.0),
        "friction": _finite(acceptance["fric_dose_final"], "friction dose", minimum=0.0),
        "torque": _finite(acceptance["tau_dose_final"], "torque dose", minimum=0.0),
    }
    budget = _exact_keys(document["budget"], {
        "cop_gate", "fric_gate", "tau_gate", "vel_limit_frac", "kin_vel_target",
        "kin_acc_target", "note",
    }, "TOPP budget")
    gates = {
        "cop": _finite(budget["cop_gate"], "CoP gate", minimum=0.0),
        "friction": _finite(budget["fric_gate"], "friction gate", minimum=0.0),
        "torque": _finite(budget["tau_gate"], "torque gate", minimum=0.0),
    }
    for name in doses:
        _require(doses[name] <= gates[name] + 5e-5,
                 f"TOPP {name} dose exceeds its recorded gate")
    return {
        "output_sha256": output.sha256,
        "topp_certificate_sha256": certificate.sha256,
        "topp_markdown_sha256": markdown.sha256,
        "candidate_start_to_contact_s": timing_s,
        "within_0p5_s": timing_s <= 0.5,
        "doses": doses,
    }


def _collect_prior_v2_inputs(*, activation: Mapping[str, Any],
                             queue: Mapping[str, Any],
                             body_order: Sequence[str]) -> tuple[
                                 dict[str, dict[str, Any]], dict[str, Snapshot]
                             ]:
    prior_root = _absolute(activation["prior_failed_attempt"]["namespace"])
    snapshots: dict[str, Snapshot] = {}
    prior_assets: dict[str, Path] = {}
    for name in ("forehand", "backhand"):
        path = prior_root / f"snapshots/assets/{name}.npz"
        snapshot = _read_snapshot(path, f"prior V2 {name} asset")
        _require(snapshot.sha256 == queue["assets"][name]["sha256"],
                 f"prior V2 {name} asset SHA changed")
        snapshots[f"prior:asset:{name}"] = snapshot
        prior_assets[name] = path
    records: dict[str, dict[str, Any]] = {}
    generator_path = prior_root / "snapshots/build_ready_to_strike_motion.py"
    for cell_id, (action, ready_source, delta) in EXPECTED_STAGE2_CELLS.items():
        candidate_path = prior_root / cell_id / "candidate.npz"
        contract_path = prior_root / cell_id / "candidate.contract.json"
        candidate = _read_snapshot(candidate_path, f"prior V2 {cell_id} candidate")
        contract = _read_snapshot(contract_path, f"prior V2 {cell_id} contract")
        expected_candidate, expected_contract = EXPECTED_PRIOR_CANDIDATES[cell_id]
        _require(candidate.sha256 == expected_candidate
                 and contract.sha256 == expected_contract,
                 f"prior V2 {cell_id} candidate or contract SHA changed: "
                 f"candidate={candidate.sha256} expected={expected_candidate}; "
                 f"contract={contract.sha256} expected={expected_contract}")
        cell = {"cell_id": cell_id, "action": action,
                "ready_source": ready_source, "delta": delta}
        info = _validate_candidate(
            candidate, contract, cell=cell, queue=queue,
            generator_path=generator_path, asset_paths=prior_assets,
            body_order=body_order,
        )
        _require(info["candidate_sha256"] == expected_candidate
                 and info["generator_contract_sha256"] == expected_contract,
                 f"prior V2 {cell_id} candidate validation lineage changed")
        records[cell_id] = {
            "candidate": candidate, "contract": contract, "info": info,
        }
        snapshots[f"prior:candidate:{cell_id}"] = candidate
        snapshots[f"prior:contract:{cell_id}"] = contract
    return records, snapshots


def _validate_inputs(*, activation_path: Path | str, queue_path: Path | str,
                     root: Path | str, runner_source: Path | str | None = None) -> dict[str, Any]:
    root = _absolute(root)
    _require(root.is_absolute(), "Stage2 root must be absolute")
    _ensure_no_symlink_components(root.parent, "Stage2 root parent")
    _ensure_no_symlink_components(root, "Stage2 root", leaf_may_be_missing=True)
    _require(not root.exists() and not root.is_symlink(), f"Stage2 root already exists: {root}")
    source_snapshot = _read_snapshot(runner_source or Path(__file__), "Stage2 runner source")
    activation_snapshot = _read_snapshot(activation_path, "activation")
    queue_snapshot = _read_snapshot(queue_path, "queue")
    _require(queue_snapshot.sha256 == EXPECTED_QUEUE_SHA256, "queue bytes differ from preregistration")
    queue = _validate_queue(_load_json(queue_snapshot.payload, "queue"))
    activation, cells, receipt_path, receipt_sha = _validate_activation(
        _load_json(activation_snapshot.payload, "activation"), queue,
        runner_sha256=source_snapshot.sha256)
    _require(root == _absolute(activation["stage2_namespace"]),
             "requested root differs from the one-shot Stage2 activation namespace")
    receipt_snapshot = _read_snapshot(receipt_path, "Stage1 attestation receipt")
    receipt = _validate_stage1_receipt(
        _load_json(receipt_snapshot.payload, "Stage1 receipt"),
        receipt_sha=receipt_sha, receipt_payload=receipt_snapshot.payload,
        queue=queue, activation=activation)
    prior_summary_snapshot = _read_snapshot(
        activation["prior_failed_attempt"]["summary_path"],
        "prior Stage2 failure summary",
    )
    prior_summary = _validate_prior_failure(
        _load_json(prior_summary_snapshot.payload, "prior Stage2 failure summary"),
        payload=prior_summary_snapshot.payload,
        expected_receipt_sha=receipt_sha,
    )
    runtime_root = _absolute(queue["runtime"]["checkout_path"])
    _require(runtime_root.is_dir(), "runtime checkout is missing")
    snapshots: dict[str, Snapshot] = {
        "runner": source_snapshot, "activation": activation_snapshot,
        "queue": queue_snapshot, "receipt": receipt_snapshot,
        "prior_summary": prior_summary_snapshot,
    }
    topp_runtime_receipt, topp_runtime_snapshots = _inspect_topp_runtime(
        activation["topp_runtime"])
    snapshots.update(topp_runtime_snapshots)
    for key, relative in RUNTIME_RELATIVE_PATHS.items():
        snapshot = _read_snapshot(runtime_root / relative, f"runtime {key}")
        _require(snapshot.sha256 == queue["runtime"][key], f"runtime {key} SHA changed")
        snapshots[f"runtime:{relative}"] = snapshot
    for relative in TOPP_CLOSURE_PATHS:
        key = f"runtime:{relative}"
        if key not in snapshots:
            snapshots[key] = _read_snapshot(runtime_root / relative, f"TOPP closure {relative}")
    mjcf_relative = RUNTIME_RELATIVE_PATHS["mjcf_sha256"]
    mjcf_closure, mesh_snapshots = _collect_mjcf_mesh_closure(
        runtime_root=runtime_root,
        checkout_commit=queue["runtime"]["checkout_commit"],
        mjcf_relative=mjcf_relative,
        mjcf_snapshot=snapshots[f"runtime:{mjcf_relative}"],
    )
    snapshots.update(mesh_snapshots)
    for name, asset in queue["assets"].items():
        snapshot = _read_snapshot(asset["path"], f"asset {name}")
        _require(snapshot.sha256 == asset["sha256"], f"asset {name} SHA changed")
        snapshots[f"asset:{name}"] = snapshot
    current_prior_runtime = {
        key.split(":", 1)[1]: value.sha256
        for key, value in snapshots.items()
        if key.startswith("runtime:")
        and Path(key.split(":", 1)[1]) in (
            set(TOPP_CLOSURE_PATHS) | set(RUNTIME_RELATIVE_PATHS.values())
        )
    }
    _require(prior_summary["runtime_snapshot_shas"] == current_prior_runtime,
             "prior V2 runtime inputs differ from the frozen current TOPP closure")
    current_assets = {
        name: snapshots[f"asset:{name}"].sha256 for name in ("forehand", "backhand")
    }
    _require(prior_summary["asset_snapshot_shas"] == current_assets,
             "prior V2 assets differ from the frozen current assets")
    try:
        body_order = tuple(
            line.strip()
            for line in snapshots[f"runtime:{RUNTIME_RELATIVE_PATHS['body_order_sha256']}"]
            .payload.decode("utf-8").splitlines()
            if line.strip()
        )
    except UnicodeDecodeError as exc:
        raise Stage2Error("runtime body order is not UTF-8") from exc
    _require(body_order and len(body_order) == len(set(body_order)),
             "runtime body order must contain unique non-empty names")
    prior_candidate_records, prior_input_snapshots = _collect_prior_v2_inputs(
        activation=activation, queue=queue, body_order=body_order,
    )
    snapshots.update(prior_input_snapshots)
    return {
        "root": root, "queue": queue, "activation": activation, "receipt": receipt,
        "prior_summary": prior_summary,
        "cells": cells, "snapshots": snapshots, "runtime_root": runtime_root,
        "body_order": body_order, "mjcf_asset_closure": mjcf_closure,
        "prior_candidate_records": prior_candidate_records,
        "topp_runtime_receipt": topp_runtime_receipt,
    }


def _materialize_runtime_snapshot(*, snapshots: Mapping[str, Snapshot],
                                  destination: Path) -> None:
    destination.mkdir(mode=0o700, parents=True)
    for key, snapshot in snapshots.items():
        if not key.startswith("runtime:"):
            continue
        relative = Path(key.split(":", 1)[1])
        output = destination / relative
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_exclusive(output, snapshot.payload)


def _verify_materialized_snapshot_tree(
    *, snapshots: Mapping[str, Snapshot], destination: Path,
    key_prefix: str, expected_tree: Sequence[Mapping[str, Any]], label: str,
) -> dict[str, Any]:
    """Re-read every copied input and reject added, missing, or changed tree entries."""
    file_shas: dict[str, str] = {}
    for key, original in sorted(snapshots.items()):
        if not key.startswith(key_prefix):
            continue
        relative = key.removeprefix(key_prefix)
        relative_path = PurePosixPath(relative)
        _require(relative != "" and not relative_path.is_absolute()
                 and all(part not in ("", ".", "..") for part in relative_path.parts),
                 f"{label} snapshot key is malformed: {key}")
        current = _read_snapshot(
            destination.joinpath(*relative_path.parts), f"unchanged {label} {relative}")
        _require(current.payload == original.payload,
                 f"{label} changed during Stage2: {relative}")
        file_shas[relative] = current.sha256
    _require(file_shas, f"{label} snapshot set is empty")
    current_tree = _tree_manifest(destination)
    expected_rows = [dict(row) for row in expected_tree]
    _require(current_tree == expected_rows,
             f"{label} tree changed during Stage2")
    return {
        "file_count": len(file_shas),
        "file_shas": file_shas,
        "tree_manifest_sha256": _sha256(_canonical_json(current_tree)),
    }


def plan_stage2(*, activation_path: Path | str, queue_path: Path | str,
                root: Path | str, runner_source: Path | str | None = None) -> dict[str, Any]:
    context = _validate_inputs(activation_path=activation_path, queue_path=queue_path,
                               root=root, runner_source=runner_source)
    queue = context["queue"]
    plans: list[dict[str, Any]] = []
    for cell in context["cells"]:
        contact = queue["assets"][cell["action"]]["contact_frame"]
        plans.append({**cell, "join_frame": contact - 12, "blend_intervals": 10,
                      "output_contact_frame": 25})
    with tempfile.TemporaryDirectory(prefix="ready-stage2-v8-mjcf-") as temporary:
        runtime_snapshot_root = Path(os.path.realpath(temporary)) / "runtime"
        _materialize_runtime_snapshot(
            snapshots=context["snapshots"], destination=runtime_snapshot_root)
        mjcf_preflight = _preflight_mjcf_runtime(
            runtime_snapshot_root=runtime_snapshot_root,
            contract=context["activation"]["topp_runtime"],
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run_passed_no_namespace_created",
        "activation_id": EXPECTED_ACTIVATION_ID,
        "root": str(context["root"]),
        "cells": plans,
        "receipt_sha256": context["snapshots"]["receipt"].sha256,
        "runner_sha256": context["snapshots"]["runner"].sha256,
        "mjcf_mesh_count": context["mjcf_asset_closure"]["mesh_count"],
        "mjcf_mesh_manifest_sha256": (
            context["mjcf_asset_closure"]["mesh_manifest_sha256"]
        ),
        "topp_runtime_receipt": context["topp_runtime_receipt"],
        "mjcf_runtime_preflight": mjcf_preflight,
        "runtime_authority": context["activation"]["runtime_authority"],
    }


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _default_command_runner(command: Sequence[str], *, cwd: Path,
                            env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=cwd, env=dict(env), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, check=False,
                          timeout=CHILD_TIMEOUT_S)


def _run_stage2_impl(*, activation_path: Path | str, queue_path: Path | str,
                     root: Path | str, execute: bool = False,
                     confirm: str | None = None,
                     runner_source: Path | str | None = None,
                     command_runner: CommandRunner = _default_command_runner,
                     _ownership: dict[str, bool] | None = None) -> dict[str, Any]:
    if not execute:
        _require(confirm is None, "--confirm is only valid with --execute")
        return plan_stage2(activation_path=activation_path, queue_path=queue_path,
                           root=root, runner_source=runner_source)
    _require(confirm == CONFIRM_TOKEN, f"--execute requires --confirm {CONFIRM_TOKEN}")
    context = _validate_inputs(activation_path=activation_path, queue_path=queue_path,
                               root=root, runner_source=runner_source)
    root_path: Path = context["root"]
    root_path.mkdir(mode=0o700)
    if _ownership is not None:
        _ownership["created"] = True
    snapshots: Mapping[str, Snapshot] = context["snapshots"]
    snapshot_root = root_path / "snapshots"
    snapshot_root.mkdir(mode=0o700)
    snapshot_destinations: dict[str, Path] = {
        "runner": snapshot_root / "run_ready_to_strike_join_ladder_stage2.py",
        "activation": snapshot_root / "activation.json",
        "queue": snapshot_root / "queue.json",
        "receipt": snapshot_root / "stage1_historical_attestation.json",
        "prior_summary": snapshot_root / "prior_stage2_failure_summary.json",
    }
    for key in ("runner", "activation", "queue", "receipt", "prior_summary"):
        _write_exclusive(snapshot_destinations[key], snapshots[key].payload)
    runtime_snapshot_root = snapshot_root / "runtime"
    _materialize_runtime_snapshot(
        snapshots=snapshots, destination=runtime_snapshot_root)
    runtime_snapshot_tree = _tree_manifest(runtime_snapshot_root)
    mjcf_preflight = _preflight_mjcf_runtime(
        runtime_snapshot_root=runtime_snapshot_root,
        contract=context["activation"]["topp_runtime"],
    )
    _write_exclusive(
        snapshot_root / "topp_runtime_receipt.json",
        _canonical_json(context["topp_runtime_receipt"]),
    )
    _write_exclusive(
        snapshot_root / "mjcf_runtime_preflight.json",
        _canonical_json(mjcf_preflight),
    )
    prior_snapshot_root = snapshot_root / "prior_v2"
    prior_snapshot_root.mkdir(mode=0o700)
    for key, snapshot in snapshots.items():
        if not key.startswith("prior:"):
            continue
        destination = prior_snapshot_root / key.removeprefix("prior:").replace(":", "__")
        _write_exclusive(destination, snapshot.payload)
    asset_snapshot_root = snapshot_root / "assets"
    asset_snapshot_root.mkdir(mode=0o700)
    asset_snapshot_paths: dict[str, Path] = {}
    for name in ("forehand", "backhand"):
        destination = asset_snapshot_root / f"{name}.npz"
        _write_exclusive(destination, snapshots[f"asset:{name}"].payload)
        asset_snapshot_paths[name] = destination
    materialized_asset_snapshots = {
        f"asset:{name}.npz": snapshots[f"asset:{name}"]
        for name in ("forehand", "backhand")
    }
    asset_snapshot_tree = _tree_manifest(asset_snapshot_root)

    queue: Mapping[str, Any] = context["queue"]
    runtime_root: Path = context["runtime_root"]
    topp = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["topp_sha256"]
    mjcf = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["mjcf_sha256"]
    urdf = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["urdf_sha256"]
    body_order = runtime_snapshot_root / RUNTIME_RELATIVE_PATHS["body_order_sha256"]
    env = _topp_runtime_env()
    rows: list[dict[str, Any]] = []
    candidate_records: dict[str, tuple[Path, Snapshot, dict[str, Any]]] = {}
    for cell in context["cells"]:
        cell_root = root_path / cell["cell_id"]
        cell_root.mkdir(mode=0o700)
        action = queue["assets"][cell["action"]]
        join = action["contact_frame"] - 12
        candidate_path = cell_root / "candidate.npz"
        contract_path = cell_root / "candidate.contract.json"
        prior_record = context["prior_candidate_records"][cell["cell_id"]]
        candidate_snapshot: Snapshot = prior_record["candidate"]
        contract_snapshot: Snapshot = prior_record["contract"]
        info: dict[str, Any] = dict(prior_record["info"])
        _write_exclusive(candidate_path, candidate_snapshot.payload)
        _write_exclusive(contract_path, contract_snapshot.payload)
        row: dict[str, Any] = {
            **cell, "join_frame": join, "blend_intervals": 10,
            "generator_rc": 0, "generator_reused_from_prior_v2": True, **info,
        }
        rows.append(row)
        candidate_records[cell["cell_id"]] = (candidate_path, candidate_snapshot, info)

    def run_topp(cell: Mapping[str, Any], row: dict[str, Any]) -> tuple[
        str, subprocess.CompletedProcess[str] | None, str | None, list[str]
    ]:
        candidate_path, _candidate_snapshot, info = candidate_records[cell["cell_id"]]
        topp_root = root_path / cell["cell_id"] / "topp"
        topp_root.mkdir(mode=0o700)
        command = [
            context["activation"]["topp_runtime"]["interpreter"]["path"],
            "-I", "-B", str(topp), "--input", str(candidate_path),
            "--phase", format(info["phase"], ".17g"), "--objective", "runup",
            "--budget-clips", str(asset_snapshot_paths["forehand"]),
            str(asset_snapshot_paths["backhand"]), "--mjcf", str(mjcf),
            "--urdf", str(urdf), "--body-order", str(body_order), "--body-mode", "fk",
            "--output", str(topp_root / "motion.npz"),
            "--report", str(topp_root / "certificate.json"),
            "--md", str(topp_root / "certificate.md"),
        ]
        try:
            completed = command_runner(
                command, cwd=runtime_root / "hope_training/whole_body_tracking", env=env)
        except Exception as exc:
            return (cell["cell_id"], None,
                    f"topp_spawn_or_wait_failed:{type(exc).__name__}:{exc}", command)
        return cell["cell_id"], completed, None, command

    row_by_id = {row["cell_id"]: row for row in rows}
    futures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for cell in context["cells"]:
            if cell["cell_id"] in candidate_records:
                futures.append(pool.submit(run_topp, cell, row_by_id[cell["cell_id"]]))
        for future in as_completed(futures):
            cell_id, completed, terminal_error, command = future.result()
            topp_root = root_path / cell_id / "topp"
            row = row_by_id[cell_id]
            row["topp_argv"] = command
            if terminal_error is not None or completed is None:
                row["terminal_error"] = terminal_error or "TOPP child failed without result"
                continue
            log_payload = completed.stdout.encode("utf-8")
            _write_exclusive(topp_root / "run.log", log_payload)
            row["topp_log_sha256"] = _sha256(log_payload)
            row["topp_rc"] = completed.returncode
            if completed.returncode != 0:
                continue
            try:
                certificate = _read_snapshot(topp_root / "certificate.json", f"{cell_id} certificate")
                output = _read_snapshot(topp_root / "motion.npz", f"{cell_id} TOPP output")
                markdown = _read_snapshot(topp_root / "certificate.md", f"{cell_id} TOPP markdown")
                candidate_path, candidate_snapshot, info = candidate_records[cell_id]
                row.update(_validate_topp(
                    certificate, output, markdown, candidate=candidate_snapshot,
                    candidate_info=info, queue=queue,
                    paths={"candidate": candidate_path, "topp": topp, "mjcf": mjcf,
                           "urdf": urdf, "body_order": body_order},
                    body_order=context["body_order"],
                ))
            except Stage2Error as exc:
                row["terminal_error"] = f"topp_validation_failed:{exc}"

    input_stability_errors: list[str] = []
    for key, original in snapshots.items():
        try:
            current = _read_snapshot(original.path, f"unchanged input {key}")
            _require(current.payload == original.payload,
                     f"input changed during Stage2: {original.path}")
        except Stage2Error as exc:
            input_stability_errors.append(str(exc))
    try:
        final_runtime_receipt, _unused_runtime_snapshots = _inspect_topp_runtime(
            context["activation"]["topp_runtime"])
        _require(final_runtime_receipt == context["topp_runtime_receipt"],
                 "TOPP runtime changed during Stage2")
    except Stage2Error as exc:
        input_stability_errors.append(str(exc))
    materialized_runtime_postcheck: dict[str, Any] | None = None
    try:
        materialized_runtime_postcheck = _verify_materialized_snapshot_tree(
            snapshots=snapshots, destination=runtime_snapshot_root,
            key_prefix="runtime:", expected_tree=runtime_snapshot_tree,
            label="materialized runtime snapshot",
        )
    except Stage2Error as exc:
        input_stability_errors.append(str(exc))
    materialized_asset_postcheck: dict[str, Any] | None = None
    try:
        materialized_asset_postcheck = _verify_materialized_snapshot_tree(
            snapshots=materialized_asset_snapshots, destination=asset_snapshot_root,
            key_prefix="asset:", expected_tree=asset_snapshot_tree,
            label="materialized asset snapshot",
        )
    except Stage2Error as exc:
        input_stability_errors.append(str(exc))
    mjcf_postflight: dict[str, Any] | None = None
    try:
        mjcf_postflight = _preflight_mjcf_runtime(
            runtime_snapshot_root=runtime_snapshot_root,
            contract=context["activation"]["topp_runtime"],
        )
        _require(mjcf_postflight == mjcf_preflight,
                 "MJCF runtime or actual-loaded dynamic closure changed during Stage2")
    except Stage2Error as exc:
        input_stability_errors.append(str(exc))
    if mjcf_postflight is not None:
        _write_exclusive(
            snapshot_root / "mjcf_runtime_postflight.json",
            _canonical_json(mjcf_postflight),
        )
    execution_snapshot_stable = (
        materialized_runtime_postcheck is not None
        and materialized_asset_postcheck is not None
        and mjcf_postflight == mjcf_preflight
    )
    execution_complete = (
        len(rows) == 4 and not input_stability_errors
        and all(row.get("generator_rc") == 0 and row.get("topp_rc") == 0
                and "topp_certificate_sha256" in row and "terminal_error" not in row
                for row in rows)
    )
    accepted_cells = sorted(
        row["cell_id"] for row in rows if row.get("within_0p5_s") is True
    )
    timing_by_cell = {
        row["cell_id"]: row.get("candidate_start_to_contact_s") for row in rows
        if "candidate_start_to_contact_s" in row
    }
    shared_ready_pass = {
        ready: all(
            row_by_id[f"{side}_{'rf' if ready == 'forehand' else 'rb'}_d12"].get("within_0p5_s") is True
            for side in ("fh", "bh")
        )
        for ready in ("forehand", "backhand")
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "ready_to_strike_join_ladder_stage2_screening_result",
        "status": ("stage2_execution_complete_no_retry"
                   if execution_complete else "stage2_terminal_failure_no_retry"),
        "activation_sha256": snapshots["activation"].sha256,
        "queue_sha256": snapshots["queue"].sha256,
        "stage1_receipt_sha256": snapshots["receipt"].sha256,
        "prior_failed_attempt_summary_sha256": snapshots["prior_summary"].sha256,
        "runner_sha256": snapshots["runner"].sha256,
        "runtime_snapshot_shas": {
            key.split(":", 1)[1]: value.sha256
            for key, value in sorted(snapshots.items()) if key.startswith("runtime:")
        },
        "asset_snapshot_shas": {
            key.split(":", 1)[1]: value.sha256
            for key, value in sorted(snapshots.items()) if key.startswith("asset:")
        },
        "prior_v2_snapshot_shas": {
            key.removeprefix("prior:"): value.sha256
            for key, value in sorted(snapshots.items()) if key.startswith("prior:")
        },
        "prior_diagnostic_logs_consumed": False,
        "prior_v2_timing_available": False,
        "mjcf_asset_closure": context["mjcf_asset_closure"],
        "topp_runtime_receipt": context["topp_runtime_receipt"],
        "mjcf_runtime_preflight": mjcf_preflight,
        "mjcf_runtime_postflight": mjcf_postflight,
        "execution_snapshot_stability": {
            "runtime": materialized_runtime_postcheck,
            "assets": materialized_asset_postcheck,
            "postflight_matches_preflight": mjcf_postflight == mjcf_preflight,
        },
        "rows": rows,
        "screening_acceptance": {
            "at_or_below_0p5_cells": accepted_cells,
            "timing_by_cell_s": timing_by_cell,
            "shared_ready_two_side_at_or_below_0p5": shared_ready_pass,
            "any_shared_ready_pass": any(shared_ready_pass.values()),
        },
        "input_stability_errors": input_stability_errors,
        "formal_claims": {
            "physics_replay_exact": False, "source_closure_exact": False,
            "mjcf_closure_exact": execution_complete and execution_snapshot_stable,
            "screening_evidence_only": True,
            "strict_global_minimum_proven": False,
        },
        "runtime_authority": context["activation"]["runtime_authority"],
        "generator_commands_executed": 0,
        "automatic_retry": False,
        "reviewed_child_timeout_s": CHILD_TIMEOUT_S,
        "trainer_or_robot_signals": [],
    }
    _write_exclusive(root_path / "stage2_summary.json", _canonical_json(summary))
    _require(execution_complete,
             "one or more Stage2 cells failed; evidence preserved and no retry attempted")
    return summary


def run_stage2(*, activation_path: Path | str, queue_path: Path | str,
               root: Path | str, execute: bool = False, confirm: str | None = None,
               runner_source: Path | str | None = None,
               command_runner: CommandRunner = _default_command_runner) -> dict[str, Any]:
    """Run once and preserve an honest terminal marker after namespace creation."""
    ownership = {"created": False}
    try:
        return _run_stage2_impl(
            activation_path=activation_path, queue_path=queue_path, root=root,
            execute=execute, confirm=confirm, runner_source=runner_source,
            command_runner=command_runner, _ownership=ownership,
        )
    except Exception as exc:
        root_path = _absolute(root)
        summary_path = root_path / "stage2_summary.json"
        terminal_path = root_path / "stage2_unexpected_terminal_failure.json"
        if (ownership["created"] and root_path.is_dir()
                and not summary_path.exists() and not terminal_path.exists()):
            terminal = {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "ready_to_strike_join_ladder_stage2_terminal_failure",
                "status": "stage2_unexpected_terminal_failure_no_retry",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "automatic_retry": False,
                "reviewed_child_timeout_s": CHILD_TIMEOUT_S,
                "trainer_or_robot_signals": [],
            }
            try:
                _write_exclusive(terminal_path, _canonical_json(terminal))
            except Exception:
                pass
        if isinstance(exc, Stage2Error):
            raise
        raise Stage2Error(
            f"unexpected Stage2 failure after evidence preservation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="Absolute, nonexistent no-clobber Stage2 result namespace")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_stage2(
            activation_path=args.activation, queue_path=args.queue, root=args.root,
            execute=args.execute, confirm=args.confirm, runner_source=Path(__file__),
        )
    except Stage2Error as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
