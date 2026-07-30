#!/usr/bin/env python3
"""Independent Isaac worker for the action-ball frozen-evaluation inbox.

The CLI exposes only the formal headless Isaac backend.  It launches Kit before
importing Isaac modules, validates its code-pinned launch receipt, emits one
machine-readable READY line, then remains resident and serves append-only
requests.  The small deterministic fake below is import-only test scaffolding:
there is no CLI route that can publish its rows as production policy evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Dict, Mapping, Optional


SCRIPT_PATH = Path(__file__).resolve()
WBT_ROOT = SCRIPT_PATH.parents[1]
INBOX_MODULE_PATH = (
    WBT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_evaluation_inbox.py"
)


def _load_inbox_protocol():
    name = "action_ball_evaluation_inbox"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    before = INBOX_MODULE_PATH.lstat()
    if (
        not INBOX_MODULE_PATH.is_file()
        or INBOX_MODULE_PATH.is_symlink()
        or before.st_nlink != 1
    ):
        raise RuntimeError(
            "action-ball evaluation inbox source is not one regular file"
        )
    descriptor = os.open(
        str(INBOX_MODULE_PATH),
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = INBOX_MODULE_PATH.lstat()
    signature = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if (
        signature(before) != signature(opened)
        or signature(before) != signature(after)
        or signature(before) != signature(final)
    ):
        raise RuntimeError(
            "action-ball evaluation inbox source changed while loading"
        )
    try:
        code = compile(
            b"".join(chunks),
            str(INBOX_MODULE_PATH),
            "exec",
            dont_inherit=True,
        )
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(
            "cannot compile action-ball evaluation inbox source"
        ) from exc
    module = type(sys)(name)
    module.__file__ = str(INBOX_MODULE_PATH)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


inbox_protocol = _load_inbox_protocol()

TASK_ID = "HOPE-PingPong-ActionBall-AgibotA3-v0"
READY_KIND = "whole_body_tracking.action_ball.formal_sidecar_ready"
HEARTBEAT_KIND = (
    "whole_body_tracking.action_ball.formal_sidecar_heartbeat"
)
HEARTBEAT_SCHEMA_VERSION = 1
FORMAL_ISAAC_BACKEND_CONTRACT_SHA256 = (
    inbox_protocol.FORMAL_ISAAC_BACKEND_CONTRACT_SHA256
)
RUNTIME_INVENTORY_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_runtime_inventory.py"
)
RUNTIME_INVENTORY_VERIFICATION_KIND = (
    "action_ball_runtime_inventory_live_verification"
)
_INVENTORY_STDIN_WRAPPER = (
    "import sys\n"
    "_raw=sys.stdin.buffer.read()\n"
    "_path=sys.argv.pop(1)\n"
    "_globals={'__name__':'__main__','__file__':_path,'__package__':None}\n"
    "exec(compile(_raw,_path,'exec',dont_inherit=True,optimize=0),_globals)\n"
)
_LIVE_RUNTIME_INVENTORY_CACHE: Dict[str, Dict[str, object]] = {}
_LIVE_RUNTIME_INVENTORY_CACHE_LOCK = threading.Lock()

_CPU_FAKE_BACKEND_DOCUMENT = {
    "schema_version": 1,
    "kind": "action_ball_frozen_eval_cpu_fake_backend",
    "purpose": "transport-tests-only-not-policy-evidence",
    "sampling": "deterministic 20/60/20 center/interior/frontier",
    "outcomes": (
        "deterministic raw signal fixtures with physics-invalid and "
        "solver-rejected accounted separately"
    ),
}
CPU_FAKE_BACKEND_CONTRACT_SHA256 = inbox_protocol.canonical_sha256(
    _CPU_FAKE_BACKEND_DOCUMENT
)


def _runtime_inventory_content_sha256(value: object) -> str:
    """Match runtime_inventory.py's UTF-8, ensure_ascii=False contract."""

    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _strict_read_runtime_inventory(
    receipt: Mapping[str, object],
) -> Dict[str, object]:
    """Read the inventory tool's canonical UTF-8 JSON, not inbox ASCII JSON."""

    raw = inbox_protocol.read_artifact_receipt_bytes(
        receipt,
        label="claim-bound runtime inventory",
    )

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(
                    "runtime inventory contains a duplicate JSON key"
                )
            result[key] = value
        return result

    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "runtime inventory is not strict UTF-8 JSON"
        ) from exc
    if type(document) is not dict:
        raise RuntimeError("runtime inventory is not a JSON object")
    expected = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if raw != expected:
        raise RuntimeError(
            "runtime inventory is not canonical UTF-8 JSON"
        )
    return document


def _preimport_claim_payload(
    claim_document: Mapping[str, object],
) -> Dict[str, object]:
    """Validate enough of the claim using only the pinned inbox protocol.

    The full trainer/bootstrap validator runs after the live Python inventory
    passes.  Keeping this pre-import validator stdlib-only prevents an
    editable/.pth/module-origin drift from importing the very package whose
    identity is about to be checked.
    """

    if type(claim_document) is not dict or set(claim_document) != {
        "schema_version",
        "kind",
        "launch_claim_sha256",
        "canonical_payload",
        "argv",
        "confirmation_claim_sha256",
    }:
        raise RuntimeError(
            "exact-resume launch claim envelope is not exact"
        )
    payload = claim_document["canonical_payload"]
    claim_sha256 = claim_document["launch_claim_sha256"]
    argv_without_claim = (
        payload.get("argv_without_launch_claim")
        if type(payload) is dict
        else None
    )
    raw_namespace = (
        payload.get("namespace") if type(payload) is dict else None
    )
    namespace = (
        Path(raw_namespace)
        if type(raw_namespace) is str
        else None
    )
    expected_claim_path_override = (
        "++training_launch_claim_path={}".format(
            namespace / "launch_claim.json"
        )
        if namespace is not None
        else None
    )
    normalized_claim_bindings = []
    if type(argv_without_claim) is list:
        for token in argv_without_claim:
            if type(token) is not str:
                normalized_claim_bindings.append((None, token))
                continue
            key = token.split("=", 1)[0].lstrip("+~")
            if key in (
                "training_launch_claim_path",
                "training_launch_claim_sha256",
            ):
                normalized_claim_bindings.append((key, token))
    path_bindings = [
        token
        for key, token in normalized_claim_bindings
        if key == "training_launch_claim_path"
    ]
    sha_bindings = [
        token
        for key, token in normalized_claim_bindings
        if key == "training_launch_claim_sha256"
    ]
    if (
        claim_document["schema_version"] != 3
        or claim_document["kind"]
        != "action_ball_no_clobber_launch_claim_v3"
        or type(payload) is not dict
        or type(claim_sha256) is not str
        or len(claim_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in claim_sha256
        )
        or inbox_protocol.canonical_sha256(payload) != claim_sha256
        or claim_document["confirmation_claim_sha256"] != claim_sha256
        or type(claim_document["argv"]) is not list
        or type(argv_without_claim) is not list
        or any(type(token) is not str for token in argv_without_claim)
        or any(
            type(token) is not str
            for token in claim_document["argv"]
        )
        or namespace is None
        or not namespace.is_absolute()
        or not namespace.name
        or ".." in namespace.parts
        or os.path.normpath(raw_namespace)
        != raw_namespace.rstrip(os.sep)
        or path_bindings != [expected_claim_path_override]
        or sha_bindings
        or claim_document["argv"]
        != [
            *argv_without_claim,
            "++training_launch_claim_sha256={}".format(claim_sha256),
        ]
    ):
        raise RuntimeError(
            "exact-resume launch claim canonical binding is invalid"
        )
    return dict(payload)


def _live_verify_runtime_inventory(
    *,
    claim_payload: Mapping[str, object],
) -> Dict[str, object]:
    """Re-run the claim-pinned runtime inventory before project imports."""

    raw_root = claim_payload.get("source_checkout")
    runtime_code = claim_payload.get("runtime_code_sha256")
    runtime = claim_payload.get("isaac_python_runtime")
    inventory_identity = (
        runtime.get("runtime_inventory")
        if type(runtime) is dict
        else None
    )
    if (
        type(raw_root) is not str
        or not raw_root
        or type(runtime_code) is not dict
        or type(inventory_identity) is not dict
        or set(inventory_identity)
        != {"path", "file_sha256", "content_sha256", "kind"}
    ):
        raise RuntimeError(
            "launch claim lacks an exact live runtime inventory binding"
        )
    root = Path(raw_root)
    if (
        not root.is_absolute()
        or ".." in root.parts
        or os.path.normpath(raw_root) != raw_root.rstrip(os.sep)
        or root.resolve(strict=True) != root
        or not root.is_dir()
    ):
        raise RuntimeError(
            "launch claim source checkout is not one real absolute root"
        )
    script_path = root / RUNTIME_INVENTORY_SOURCE
    script_receipt = inbox_protocol.artifact_receipt(script_path)
    script_bytes = inbox_protocol.read_artifact_receipt_bytes(
        script_receipt,
        label="claim-bound runtime inventory verifier",
    )
    expected_script_sha256 = runtime_code.get(
        RUNTIME_INVENTORY_SOURCE
    )
    if script_receipt["sha256"] != expected_script_sha256:
        raise RuntimeError(
            "live runtime inventory verifier differs from the launch claim"
        )
    inventory_receipt = inbox_protocol.artifact_receipt(
        inventory_identity["path"]
    )
    if (
        inventory_receipt["path"] != inventory_identity["path"]
        or inventory_receipt["sha256"]
        != inventory_identity["file_sha256"]
    ):
        raise RuntimeError(
            "live runtime inventory artifact differs from the launch claim"
        )
    inventory_document = _strict_read_runtime_inventory(
        inventory_receipt
    )
    if (
        type(inventory_document) is not dict
        or set(inventory_document)
        != {"schema_version", "kind", "content", "content_sha256"}
        or inventory_document["schema_version"] != 1
        or inventory_document["kind"]
        != inventory_identity["kind"]
        or inventory_document["kind"]
        != "action_ball_runtime_inventory_v1"
        or type(inventory_document["content"]) is not dict
        or _runtime_inventory_content_sha256(
            inventory_document["content"]
        )
        != inventory_document["content_sha256"]
        or inventory_document["content_sha256"]
        != inventory_identity["content_sha256"]
    ):
        raise RuntimeError(
            "claim-bound runtime inventory document is invalid"
        )
    python_identity = inventory_document["content"].get("python")
    requested_interpreter = (
        python_identity.get("requested_path")
        if type(python_identity) is dict
        else None
    )
    current_interpreter = os.path.normpath(
        os.path.abspath(sys.executable)
    )
    if (
        type(requested_interpreter) is not str
        or current_interpreter != requested_interpreter
        or runtime.get("path") != current_interpreter
    ):
        raise RuntimeError(
            "sidecar interpreter differs from the inventoried Python"
        )
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONUSERBASE",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            [
                current_interpreter,
                "-I",
                "-B",
                "-c",
                _INVENTORY_STDIN_WRAPPER,
                str(script_path),
                "verify",
                "--receipt",
                inventory_receipt["path"],
            ],
            cwd=os.sep,
            input=script_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "live runtime inventory verifier could not execute"
        ) from exc
    try:
        post_script_receipt = inbox_protocol.artifact_receipt(
            script_path
        )
        post_inventory_receipt = inbox_protocol.artifact_receipt(
            inventory_identity["path"]
        )
    except Exception as exc:
        raise RuntimeError(
            "live runtime inventory verifier/artifact changed during "
            "execution"
        ) from exc
    if (
        post_script_receipt != script_receipt
        or post_inventory_receipt != inventory_receipt
    ):
        raise RuntimeError(
            "live runtime inventory verifier/artifact changed during "
            "execution"
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "live runtime inventory verification failed: {}".format(
                completed.stderr.decode(
                    "utf-8", errors="replace"
                )[-4000:]
            )
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise RuntimeError(
            "live runtime inventory verifier emitted no unique result"
        )
    try:
        result = json.loads(
            lines[0].decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "live runtime inventory result is not strict JSON"
        ) from exc
    expected_result = {
        "ok": True,
        "kind": inventory_document["kind"],
        "content_sha256": inventory_document["content_sha256"],
        "receipt_path": inventory_receipt["path"],
        "receipt_sha256": inventory_receipt["sha256"],
    }
    if (
        result != expected_result
        or lines[0]
        != json.dumps(
            expected_result,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ):
        raise RuntimeError(
            "live runtime inventory result differs from its exact binding"
        )
    content = {
        "schema_version": 1,
        "kind": RUNTIME_INVENTORY_VERIFICATION_KIND,
        "verifier_source": script_receipt,
        "inventory_artifact": inventory_receipt,
        "inventory_content_sha256": inventory_document[
            "content_sha256"
        ],
        "current_interpreter": current_interpreter,
        "verification_result": result,
    }
    return {
        "schema_version": 1,
        "kind": RUNTIME_INVENTORY_VERIFICATION_KIND,
        "content": content,
        "content_sha256": inbox_protocol.canonical_sha256(content),
    }


def _live_inventory_cache_key(
    claim_payload: Mapping[str, object],
) -> str:
    return inbox_protocol.canonical_sha256(
        {
            "schema_version": 1,
            "claim_payload_sha256": inbox_protocol.canonical_sha256(
                claim_payload
            ),
            "current_interpreter": os.path.normpath(
                os.path.abspath(sys.executable)
            ),
        }
    )


def _validate_preimport_live_inventory_verification(
    value: Mapping[str, object],
    *,
    claim_payload: Mapping[str, object],
) -> Dict[str, object]:
    """Deeply bind a verifier-produced proof before project imports."""

    try:
        proof = json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "pre-import live runtime inventory proof is not JSON"
        ) from exc
    if (
        type(proof) is not dict
        or set(proof)
        != {"schema_version", "kind", "content", "content_sha256"}
        or proof["schema_version"] != 1
        or proof["kind"] != RUNTIME_INVENTORY_VERIFICATION_KIND
        or type(proof["content"]) is not dict
        or set(proof["content"])
        != {
            "schema_version",
            "kind",
            "verifier_source",
            "inventory_artifact",
            "inventory_content_sha256",
            "current_interpreter",
            "verification_result",
        }
        or proof["content_sha256"]
        != inbox_protocol.canonical_sha256(proof["content"])
    ):
        raise RuntimeError(
            "pre-import live runtime inventory proof is invalid"
        )
    content = proof["content"]
    raw_root = claim_payload.get("source_checkout")
    runtime_code = claim_payload.get("runtime_code_sha256")
    python_runtime = claim_payload.get("isaac_python_runtime")
    inventory_identity = (
        python_runtime.get("runtime_inventory")
        if type(python_runtime) is dict
        else None
    )
    if (
        type(raw_root) is not str
        or type(runtime_code) is not dict
        or type(inventory_identity) is not dict
        or set(inventory_identity)
        != {"path", "file_sha256", "content_sha256", "kind"}
    ):
        raise RuntimeError(
            "launch claim lacks live inventory identity"
        )
    root = Path(raw_root)
    if (
        not root.is_absolute()
        or ".." in root.parts
        or os.path.normpath(raw_root) != raw_root.rstrip(os.sep)
        or root.resolve(strict=True) != root
        or not root.is_dir()
    ):
        raise RuntimeError(
            "launch claim source root is not real/absolute"
        )
    script_path = root / RUNTIME_INVENTORY_SOURCE
    verifier_receipt = inbox_protocol.artifact_receipt(script_path)
    inventory_receipt = inbox_protocol.artifact_receipt(
        inventory_identity.get("path")
    )
    inventory_document = _strict_read_runtime_inventory(
        inventory_receipt
    )
    current_interpreter = os.path.normpath(
        os.path.abspath(sys.executable)
    )
    expected_result = {
        "ok": True,
        "kind": "action_ball_runtime_inventory_v1",
        "content_sha256": inventory_identity.get("content_sha256"),
        "receipt_path": inventory_identity.get("path"),
        "receipt_sha256": inventory_identity.get("file_sha256"),
    }
    if (
        content["schema_version"] != 1
        or content["kind"] != RUNTIME_INVENTORY_VERIFICATION_KIND
        or verifier_receipt["sha256"]
        != runtime_code.get(RUNTIME_INVENTORY_SOURCE)
        or content["verifier_source"] != verifier_receipt
        or inventory_receipt["path"]
        != inventory_identity.get("path")
        or inventory_receipt["sha256"]
        != inventory_identity.get("file_sha256")
        or content["inventory_artifact"] != inventory_receipt
        or type(inventory_document) is not dict
        or inventory_document.get("schema_version") != 1
        or inventory_document.get("kind")
        != "action_ball_runtime_inventory_v1"
        or inventory_identity.get("kind")
        != "action_ball_runtime_inventory_v1"
        or type(inventory_document.get("content")) is not dict
        or inventory_document.get("content_sha256")
        != _runtime_inventory_content_sha256(
            inventory_document["content"]
        )
        or inventory_document.get("content_sha256")
        != inventory_identity.get("content_sha256")
        or content["inventory_content_sha256"]
        != inventory_identity.get("content_sha256")
        or content["current_interpreter"] != current_interpreter
        or python_runtime.get("path") != current_interpreter
        or type(inventory_document["content"].get("python")) is not dict
        or inventory_document["content"]["python"].get(
            "requested_path"
        )
        != current_interpreter
        or content["verification_result"] != expected_result
    ):
        raise RuntimeError(
            "pre-import live runtime inventory proof differs from claim/live "
            "artifacts"
        )
    return proof


def _install_live_inventory_cache(
    *,
    claim_payload: Mapping[str, object],
    proof: Mapping[str, object],
) -> Dict[str, object]:
    validated = _validate_preimport_live_inventory_verification(
        proof,
        claim_payload=claim_payload,
    )
    cache_key = _live_inventory_cache_key(claim_payload)
    with _LIVE_RUNTIME_INVENTORY_CACHE_LOCK:
        existing = _LIVE_RUNTIME_INVENTORY_CACHE.get(cache_key)
        if existing is None:
            _LIVE_RUNTIME_INVENTORY_CACHE[cache_key] = validated
            existing = validated
        elif existing != validated:
            raise RuntimeError(
                "live runtime inventory cache already binds another proof"
            )
        return json.loads(
            json.dumps(
                existing,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )


def _live_verify_runtime_inventory_cached(
    *,
    claim_payload: Mapping[str, object],
) -> Dict[str, object]:
    """Verify once per exact process/claim, then safely reopen cache inputs.

    The exact-resume verifier restores the source checkpoint and its immediate
    no-step roundtrip in the same process.  Re-running the recursive runtime
    inventory can hash tens of gigabytes twice even though both restores use
    the already-imported module closure.  The second call may reuse only the
    first canonical proof, keyed by the entire claim payload and interpreter,
    after rehashing the pinned verifier and inventory receipt artifacts.
    """

    cache_key = _live_inventory_cache_key(claim_payload)
    with _LIVE_RUNTIME_INVENTORY_CACHE_LOCK:
        cached = _LIVE_RUNTIME_INVENTORY_CACHE.get(cache_key)
    if cached is not None:
        return _validate_preimport_live_inventory_verification(
            cached,
            claim_payload=claim_payload,
        )
    verified = _live_verify_runtime_inventory(
        claim_payload=claim_payload
    )
    return _install_live_inventory_cache(
        claim_payload=claim_payload,
        proof=verified,
    )


@dataclass
class ExactResumeRuntime:
    """One no-step exact Isaac restore owned by the shared runtime factory."""

    wrapped_env: object
    runner: object
    construction_receipt: Dict[str, object]
    _owner: object = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._owner._close_environment()


class SidecarProgressPublisher:
    """Atomically publish watchdog-visible sidecar/request progress."""

    _PHASES = frozenset(
        (
            "starting",
            "ready",
            "waiting_for_request_or_ack",
            "request_accepted",
            "runtime_building",
            "evaluating",
            "validating_evidence",
            "evidence_published",
            "request_failed",
            "stopping",
        )
    )

    def __init__(
        self,
        *,
        inbox_root: object,
        owner_id: str,
        run_id: str,
        launch_sha256: str,
        interval_s: float,
    ) -> None:
        if (
            type(interval_s) not in (int, float)
            or not 0.25 <= float(interval_s) <= 60.0
        ):
            raise RuntimeError(
                "sidecar heartbeat interval must be in [0.25, 60] s"
            )
        for name, value in (
            ("owner_id", owner_id),
            ("run_id", run_id),
        ):
            if (
                type(value) is not str
                or not value
                or len(value) > 128
                or not value[0].isalnum()
                or any(
                    character
                    not in (
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        "abcdefghijklmnopqrstuvwxyz"
                        "0123456789_.-"
                    )
                    for character in value
                )
            ):
                raise RuntimeError(
                    "{} is not a safe inbox identifier".format(name)
                )
        self._root = Path(inbox_root)
        if not self._root.is_absolute():
            raise RuntimeError(
                "sidecar heartbeat requires an absolute inbox root"
            )
        queue = inbox_protocol.EvaluationInbox(self._root)
        queue.initialize()
        self._root = queue.root
        directory = (
            self._root
            / "sidecar_status"
            / owner_id
            / run_id
        )
        directory.mkdir(parents=True, exist_ok=True)
        for candidate in (
            self._root / "sidecar_status",
            self._root / "sidecar_status" / owner_id,
            directory,
        ):
            if candidate.is_symlink() or not candidate.is_dir():
                raise RuntimeError(
                    "sidecar heartbeat namespace is not a real directory"
                )
        self.path = directory / "heartbeat.json"
        self._owner_id = owner_id
        self._run_id = run_id
        self._launch_sha256 = launch_sha256
        self._interval_s = float(interval_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._heartbeat_seq = -1
        self._phase = "starting"
        self._request_seq = None
        self._request_sha256 = ""
        self._attempts_completed = 0
        self._attempts_total = 0
        self._request_started_unix_ns = 0
        self._request_started_monotonic_ns = 0
        self._request_deadline_unix_ns = 0
        self._request_deadline_monotonic_ns = 0
        self._error_type = ""

    @property
    def request_deadline_monotonic_ns(self) -> int:
        with self._lock:
            return self._request_deadline_monotonic_ns

    def _document_locked(self) -> Dict[str, object]:
        # This sequence proves that the atomic pointer is advancing.  It is
        # deliberately not named "progress": actual request work is measured
        # only by attempts_completed/attempts_total.
        self._heartbeat_seq += 1
        content = {
            "owner_id": self._owner_id,
            "run_id": self._run_id,
            "pid": os.getpid(),
            "sidecar_code_sha256": sidecar_code_sha256(),
            "launch_sha256": self._launch_sha256,
            "backend_contract_sha256": (
                FORMAL_ISAAC_BACKEND_CONTRACT_SHA256
            ),
            "heartbeat_seq": self._heartbeat_seq,
            "phase": self._phase,
            "request_seq": self._request_seq,
            "request_sha256": self._request_sha256,
            "attempts_completed": self._attempts_completed,
            "attempts_total": self._attempts_total,
            "request_started_unix_ns": (
                self._request_started_unix_ns
            ),
            "request_started_monotonic_ns": (
                self._request_started_monotonic_ns
            ),
            "request_deadline_unix_ns": (
                self._request_deadline_unix_ns
            ),
            "request_deadline_monotonic_ns": (
                self._request_deadline_monotonic_ns
            ),
            "heartbeat_unix_ns": time.time_ns(),
            "heartbeat_monotonic_ns": time.monotonic_ns(),
            "error_type": self._error_type,
        }
        return {
            "schema_version": HEARTBEAT_SCHEMA_VERSION,
            "kind": HEARTBEAT_KIND,
            "content": content,
            "content_sha256": inbox_protocol.canonical_sha256(
                content
            ),
        }

    def _atomic_write(self, document: Mapping[str, object]) -> None:
        encoded = (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        temporary = self.path.with_name(
            ".heartbeat.{}.{}.tmp".format(
                os.getpid(), document["content"]["heartbeat_seq"]
            )
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(str(temporary), flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if self.path.exists() or self.path.is_symlink():
                status = self.path.lstat()
                if (
                    self.path.is_symlink()
                    or not self.path.is_file()
                    or status.st_nlink != 1
                ):
                    raise RuntimeError(
                        "sidecar heartbeat target is not a single-link "
                        "regular file"
                    )
            os.replace(str(temporary), str(self.path))
            directory_fd = os.open(
                str(self.path.parent),
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def publish(
        self,
        phase: str,
        *,
        attempts_completed: Optional[int] = None,
        error_type: str = "",
    ) -> None:
        if phase not in self._PHASES:
            raise RuntimeError(
                "unknown sidecar heartbeat phase {!r}".format(phase)
            )
        with self._lock:
            if attempts_completed is not None:
                if (
                    type(attempts_completed) is not int
                    or attempts_completed
                    < self._attempts_completed
                    or attempts_completed > self._attempts_total
                ):
                    raise RuntimeError(
                        "sidecar attempt progress is invalid"
                    )
                self._attempts_completed = attempts_completed
            self._phase = phase
            self._error_type = error_type
            document = self._document_locked()
            self._atomic_write(document)

    def begin_request(
        self,
        *,
        request_seq: int,
        request_sha256: str,
        attempts_total: int,
        deadline_s: float,
    ) -> None:
        if (
            type(request_seq) is not int
            or request_seq < 0
            or type(request_sha256) is not str
            or len(request_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in request_sha256
            )
            or type(attempts_total) is not int
            or attempts_total < 1
            or type(deadline_s) not in (int, float)
            or not 1.0 <= float(deadline_s) <= 86400.0
        ):
            raise RuntimeError(
                "sidecar request progress/deadline is invalid"
            )
        now_unix = time.time_ns()
        now_monotonic = time.monotonic_ns()
        duration_ns = int(float(deadline_s) * 1_000_000_000)
        with self._lock:
            self._request_seq = request_seq
            self._request_sha256 = request_sha256
            self._attempts_completed = 0
            self._attempts_total = attempts_total
            self._request_started_unix_ns = now_unix
            self._request_started_monotonic_ns = now_monotonic
            self._request_deadline_unix_ns = now_unix + duration_ns
            self._request_deadline_monotonic_ns = (
                now_monotonic + duration_ns
            )
            self._phase = "request_accepted"
            self._error_type = ""
            document = self._document_locked()
            self._atomic_write(document)

    def waiting_for_request_or_ack(self) -> None:
        """Publish an idle state with no stale active-request deadline."""

        with self._lock:
            self._request_seq = None
            self._request_sha256 = ""
            self._attempts_completed = 0
            self._attempts_total = 0
            self._request_started_unix_ns = 0
            self._request_started_monotonic_ns = 0
            self._request_deadline_unix_ns = 0
            self._request_deadline_monotonic_ns = 0
            self._phase = "waiting_for_request_or_ack"
            self._error_type = ""
            document = self._document_locked()
            self._atomic_write(document)

    def assert_before_deadline(self) -> None:
        deadline = self.request_deadline_monotonic_ns
        if deadline and time.monotonic_ns() > deadline:
            raise RuntimeError(
                "formal sidecar request exceeded its pre-registered "
                "deadline"
            )

    def _pulse_loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                with self._lock:
                    document = self._document_locked()
                    self._atomic_write(document)
            except Exception as exc:
                self._error = exc
                self._stop.set()
                return

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(
                "sidecar heartbeat publisher was already started"
            )
        self.publish("starting")
        self._thread = threading.Thread(
            target=self._pulse_loop,
            name="action-ball-sidecar-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "sidecar heartbeat publisher failed"
            ) from self._error

    def stop(self) -> None:
        try:
            # Make the terminal heartbeat schema unambiguous: "stopping" is
            # always idle, never a stale active request that a watchdog could
            # mistake for deadline-bearing work.
            with self._lock:
                self._request_seq = None
                self._request_sha256 = ""
                self._attempts_completed = 0
                self._attempts_total = 0
                self._request_started_unix_ns = 0
                self._request_started_monotonic_ns = 0
                self._request_deadline_unix_ns = 0
                self._request_deadline_monotonic_ns = 0
                self._phase = "stopping"
                self._error_type = ""
                document = self._document_locked()
                self._atomic_write(document)
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=max(1.0, self._interval_s * 2.0))


def sidecar_code_sha256() -> str:
    return hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()


def _digest(*parts: object) -> str:
    return hashlib.sha256(
        ":".join(str(part) for part in parts).encode("ascii")
    ).hexdigest()


def _empty_terminal_signals() -> Dict[str, bool]:
    return {
        "infrastructure_invalid": False,
        "joint_actual_limit": False,
        "joint_qdes_limit": False,
        "fall": False,
        "table_hit": False,
        "collision": False,
        "legal_return": False,
    }


class DeterministicCpuFakeEvaluator:
    """Import-only deterministic fixture for host transport tests."""

    backend_contract_sha256 = CPU_FAKE_BACKEND_CONTRACT_SHA256

    @staticmethod
    def _attempt(
        *,
        request_sha256: str,
        proposal_sampler_contract_sha256: str,
        target: Mapping[str, object],
        allocation: Mapping[str, object],
        offset: int,
    ) -> Dict[str, object]:
        role = allocation["role"]
        seed = allocation["seed_start"] + offset
        sample_id = allocation["sample_start"] + offset
        birth_id = allocation["birth_start"] + offset
        sample_receipt = _digest(
            "sample", request_sha256, role, sample_id
        )
        birth_receipt = _digest(
            "birth", request_sha256, role, birth_id
        )
        proposal_receipt = _digest(
            "proposal",
            proposal_sampler_contract_sha256,
            request_sha256,
            role,
            seed,
            sample_id,
            birth_id,
        )
        slot = offset % 5
        if slot == 0:
            stratum = "center"
            frontier_arm = ""
        elif slot == 4:
            stratum = "frontier"
            frontier_arm = (
                target["selected_arm_key"] or "time_to_contact_lower"
            )
        else:
            stratum = "interior"
            frontier_arm = ""
        signals = _empty_terminal_signals()
        common = {
            "proposal_offset": offset,
            "seed": seed,
            "sample_id": sample_id,
            "birth_id": birth_id,
            "sample_receipt_sha256": sample_receipt,
            "birth_receipt_sha256": birth_receipt,
            "proposal_sampler_contract_sha256": (
                proposal_sampler_contract_sha256
            ),
            "proposal_receipt_sha256": proposal_receipt,
            "sampling_stratum": stratum,
            "frontier_arm": frontier_arm,
        }
        if offset % 37 == 0:
            return {
                **common,
                "solver_disposition": "physics_invalid",
                "reject_reason": "incoming_ball_physics_invalid",
                "task_receipt_sha256": "",
                "installed": False,
                "started": False,
                "closed": False,
                "terminal_signals": signals,
            }
        if offset % 41 == 0:
            return {
                **common,
                "solver_disposition": "rejected",
                "reject_reason": "ball_to_task_geometry_unreachable",
                "task_receipt_sha256": "",
                "installed": False,
                "started": False,
                "closed": False,
                "terminal_signals": signals,
            }
        task_receipt = _digest(
            "task", request_sha256, role, offset
        )
        if offset % 113 == 0:
            signals["infrastructure_invalid"] = True
            return {
                **common,
                "solver_disposition": "admitted",
                "reject_reason": "",
                "task_receipt_sha256": task_receipt,
                "installed": True,
                "started": True,
                "closed": False,
                "terminal_signals": signals,
            }
        if offset % 97 == 0:
            signals["table_hit"] = True
        elif offset % 89 == 0:
            signals["joint_actual_limit"] = True
        elif offset % 13 != 0:
            signals["legal_return"] = True
        return {
            **common,
            "solver_disposition": "admitted",
            "reject_reason": "",
            "task_receipt_sha256": task_receipt,
            "installed": True,
            "started": True,
            "closed": True,
            "terminal_signals": signals,
        }

    def evaluate(
        self, request_document: Mapping[str, object]
    ) -> Dict[str, object]:
        request = inbox_protocol.validate_request_document(
            request_document
        )
        request_sha256 = request_document["content_sha256"]
        result = {}
        for allocation in request["windows"]:
            result[allocation["role"]] = [
                self._attempt(
                    request_sha256=request_sha256,
                    proposal_sampler_contract_sha256=request[
                        "bindings"
                    ]["proposal_sampler_contract_sha256"],
                    target=request["target"],
                    allocation=allocation,
                    offset=offset,
                )
                for offset in range(allocation["proposal_count"])
            ]
        return result


class FormalIsaacEvaluator:
    """Reconstruct and execute each request in an independent Isaac env."""

    backend_contract_sha256 = FORMAL_ISAAC_BACKEND_CONTRACT_SHA256
    runtime_hook_name = "action_ball_frozen_evaluator_execute_v1"

    def __init__(
        self,
        *,
        device: str,
        progress: Optional[SidecarProgressPublisher] = None,
    ) -> None:
        if device != "cuda:0":
            raise RuntimeError(
                "formal sidecar device must be local cuda:0; physical GPU "
                "ownership is established by CUDA_VISIBLE_DEVICES"
            )
        self.device = device
        self._progress = progress
        self._active_env = None
        self._active_runner = None
        self._preflight_runtime()

    def _preflight_runtime(self) -> None:
        """Import the real task only after AppLauncher has started Kit."""

        import gymnasium as gym
        import torch

        import whole_body_tracking  # noqa: F401
        import whole_body_tracking.tasks  # noqa: F401
        from whole_body_tracking.tasks.tracking.mdp.hope_commands import (
            RacketTargetCommand,
        )

        if TASK_ID not in gym.registry:
            raise RuntimeError(
                "formal ActionBall task is not registered in this runtime"
            )
        if not callable(
            getattr(RacketTargetCommand, self.runtime_hook_name, None)
        ):
            raise RuntimeError(
                "formal ActionBall evaluator runtime hook is absent"
            )
        if (
            getattr(
                RacketTargetCommand,
                "ACTION_BALL_FROZEN_EVALUATOR_RUNTIME_V1_READY",
                None,
            )
            is not True
        ):
            raise RuntimeError(
                "formal ActionBall proposal-level Isaac injection runtime "
                "is not enabled"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "formal ActionBall evaluator requires CUDA"
            )
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "formal sidecar must see exactly one evaluator-owned CUDA "
                "device"
            )

    @staticmethod
    def _artifact_path(
        bindings: Mapping[str, object], name: str
    ) -> Path:
        receipt = bindings[name]
        inbox_protocol.verify_artifact_receipt(
            receipt, label="bindings.{}".format(name)
        )
        path = Path(receipt["path"])
        if not path.is_absolute():
            raise RuntimeError(
                "formal sidecar artifact paths must be absolute"
            )
        return path.resolve(strict=True)

    @staticmethod
    def _normalizer_payload(runner: object, name: str) -> dict:
        if not hasattr(runner, name):
            raise RuntimeError(
                "{} is absent from the reconstructed runner".format(name)
            )
        normalizer = getattr(runner, name)
        if normalizer is None:
            return {"enabled": False}
        state_dict = getattr(normalizer, "state_dict", None)
        if not callable(state_dict):
            raise RuntimeError(
                "{} lacks deterministic state_dict()".format(name)
            )
        return {"enabled": True, "state": state_dict()}

    @staticmethod
    def _critic_normalizer_name(runner: object) -> str:
        """Resolve the RSL-RL critic normalizer without silent omission."""

        has_privileged = hasattr(
            runner, "privileged_obs_normalizer"
        )
        has_legacy = hasattr(runner, "critic_obs_normalizer")
        if has_privileged and has_legacy:
            privileged = getattr(
                runner, "privileged_obs_normalizer"
            )
            legacy = getattr(runner, "critic_obs_normalizer")
            if (
                privileged is not None
                and legacy is not None
                and privileged is not legacy
            ):
                raise RuntimeError(
                    "runner exposes two different critic normalizers"
                )
            return "privileged_obs_normalizer"
        if has_privileged:
            return "privileged_obs_normalizer"
        if has_legacy:
            return "critic_obs_normalizer"
        raise RuntimeError(
            "reconstructed runner exposes no critic observation normalizer"
        )

    def _validate_runtime_identity(
        self,
        request: Mapping[str, object],
    ) -> dict:
        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_frozen_eval_identity as runtime_identity,
        )

        content = inbox_protocol.validate_request_document(request)
        bindings = content["bindings"]
        training_contract_path = self._artifact_path(
            bindings, "training_contract"
        )
        env_pickle_path = self._artifact_path(
            bindings, "environment_config_pickle"
        )
        agent_pickle_path = self._artifact_path(
            bindings, "agent_config_pickle"
        )
        identity_path = self._artifact_path(
            bindings, "runtime_identity"
        )
        identity = inbox_protocol.strict_read_json(
            identity_path,
            label="formal sidecar runtime identity",
        )
        identity_content = identity.get("content")
        if type(identity_content) is not dict:
            raise RuntimeError(
                "formal sidecar runtime identity has no content object"
            )
        runtime_identity.validate_runtime_identity_document(
            identity,
            repo_root=identity_content["source"]["repo_root"],
            task_id=TASK_ID,
            training_launch_claim_sha256=bindings[
                "training_launch_claim_sha256"
            ],
            training_contract_path=training_contract_path,
            environment_config_pickle_path=env_pickle_path,
            agent_config_pickle_path=agent_pickle_path,
        )
        return {
            "training_contract_path": training_contract_path,
            "environment_config_pickle_path": env_pickle_path,
            "agent_config_pickle_path": agent_pickle_path,
            "identity_path": identity_path,
        }

    @staticmethod
    def _validate_training_contract_bindings(
        contract: Mapping[str, object],
        bindings: Mapping[str, object],
    ) -> None:
        """Cross-bind request identities to the saved training contract."""

        if type(contract) is not dict:
            raise RuntimeError(
                "formal sidecar training contract must be a JSON object"
            )
        try:
            ppo_sha256 = contract[
                "action_ball_ppo_runner_recipe"
            ]["sha256"]
            reward_sha256 = contract[
                "effective_reward_recipe"
            ]["sha256"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "formal sidecar training contract lacks PPO/Reward "
                "identity"
            ) from exc
        if (
            ppo_sha256 != bindings["ppo_recipe_sha256"]
            or reward_sha256 != bindings["reward_sha256"]
        ):
            raise RuntimeError(
                "request PPO/Reward identity differs from the saved "
                "training contract"
            )

    @staticmethod
    def _validate_action_ball_term_identity(
        term: object,
        request: Mapping[str, object],
    ) -> None:
        """Prove the live command term is the request's exact arbitrary-N run."""

        hard_contract_fn = getattr(
            term, "action_ball_hard_contract", None
        )
        if not callable(hard_contract_fn):
            raise RuntimeError(
                "formal evaluator term lacks action_ball_hard_contract()"
            )
        hard = hard_contract_fn()
        if type(hard) is not dict:
            raise RuntimeError(
                "formal evaluator term returned no hard contract"
            )
        bindings = request["bindings"]
        target = request["target"]
        expected_order = list(bindings["action_order"])
        if (
            hard.get("kind")
            != (
                "whole_body_tracking.RacketTargetCommand."
                "action_ball_hard_contract"
            )
            or hard.get("action_uids") != expected_order
            or hard.get("mobility_mode")
            != target["mobility_mode"]
        ):
            raise RuntimeError(
                "live ActionBall action order/mobility differs from request"
            )
        manifest = hard.get("manifest")
        profiles = hard.get("profiles")
        solver = hard.get("solver")
        physics = hard.get("physics")
        curriculum = hard.get("curriculum")
        hard_bindings = hard.get("bindings")
        if not all(
            type(value) is dict
            for value in (
                manifest,
                profiles,
                solver,
                physics,
                curriculum,
            )
        ) or type(hard_bindings) is not list:
            raise RuntimeError(
                "live ActionBall hard contract is structurally incomplete"
            )
        if (
            manifest.get("file_sha256")
            != bindings["manifest_sha256"]
            or profiles.get("sampler_contract_sha256")
            != bindings["sampler_sha256"]
            or profiles.get(
                "frozen_evaluation_proposal_sampler_contract_sha256"
            )
            != bindings["proposal_sampler_contract_sha256"]
            or profiles.get("adapter_contract_sha256")
            != bindings["curriculum_sha256"]
            or solver.get("sha256") != bindings["solver_sha256"]
            or physics.get("sha256") != bindings["physics_sha256"]
            or curriculum.get("policy_contract_sha256")
            != bindings["policy_contract_sha256"]
        ):
            raise RuntimeError(
                "live ActionBall manifest/sampler/solver/physics/"
                "curriculum identity differs from request"
            )
        if len(hard_bindings) != len(expected_order):
            raise RuntimeError(
                "live ActionBall binding count differs from request"
            )
        request_actions = bindings["actions"]
        observed_profiles = profiles.get("profile_sha256")
        if (
            type(observed_profiles) is not list
            or len(observed_profiles) != len(expected_order)
        ):
            raise RuntimeError(
                "live ActionBall profile order is incomplete"
            )
        target_slots = [
            index
            for index, uid in enumerate(expected_order)
            if uid == target["action_uid"]
        ]
        if (
            len(target_slots) != 1
            or observed_profiles[target_slots[0]]
            != target["profile_sha256"]
        ):
            raise RuntimeError(
                "request target profile does not match its frozen action slot"
            )
        for index, (hard_row, request_row) in enumerate(
            zip(hard_bindings, request_actions)
        ):
            if (
                type(hard_row) is not dict
                or hard_row.get("action_uid") != expected_order[index]
                or hard_row.get("action_slot") != index
                or hard_row.get("profile_sha256")
                != observed_profiles[index]
                or request_row["action_uid"] != expected_order[index]
                or request_row["motion"]["sha256"]
                != hard_row.get("motion_sha256")
            ):
                raise RuntimeError(
                    "live ActionBall ordered motion/profile binding "
                    "differs from request"
                )

    @staticmethod
    def _validate_proposal_receipt_bindings(
        *,
        request: Mapping[str, object],
        attempts_by_role: object,
    ) -> None:
        """Cross-bind every returned proposal to the frozen eval sampler.

        The training sampler and the stateless frozen-evaluation proposal
        sampler are distinct contracts.  Treating the training sampler SHA as
        a substitute would make the heldout transcript irreproducible.
        """

        if type(attempts_by_role) is not dict:
            raise RuntimeError(
                "formal evaluator runtime hook returned no attempt mapping"
            )
        expected_contract = request["bindings"][
            "proposal_sampler_contract_sha256"
        ]
        expected_roles = {
            window["role"]: int(window["proposal_count"])
            for window in request["windows"]
        }
        if set(attempts_by_role) != set(expected_roles):
            raise RuntimeError(
                "formal evaluator returned wrong evidence roles"
            )
        observed_receipts = set()
        for role, expected_count in expected_roles.items():
            rows = attempts_by_role[role]
            if type(rows) is not list or len(rows) != expected_count:
                raise RuntimeError(
                    "formal evaluator returned wrong {} attempt count".format(
                        role
                    )
                )
            for offset, row in enumerate(rows):
                if (
                    type(row) is not dict
                    or row.get("proposal_sampler_contract_sha256")
                    != expected_contract
                ):
                    raise RuntimeError(
                        "formal {} proposal {} is not bound to the frozen "
                        "evaluation sampler".format(role, offset)
                    )
                receipt = row.get("proposal_receipt_sha256")
                if (
                    type(receipt) is not str
                    or len(receipt) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in receipt
                    )
                    or receipt in observed_receipts
                ):
                    raise RuntimeError(
                        "formal {} proposal {} has an invalid or repeated "
                        "proposal receipt".format(role, offset)
                    )
                observed_receipts.add(receipt)

    def _close_environment(self) -> None:
        env = self._active_env
        self._active_runner = None
        self._active_env = None
        if env is not None:
            env.close()

    @staticmethod
    def _assert_finite_tensor_tree(
        value: object, *, label: str
    ) -> None:
        """Reject any non-finite model/normalizer tensor before inference."""

        import torch

        if torch.is_tensor(value):
            if (
                torch.is_floating_point(value)
                and not bool(torch.isfinite(value).all())
            ):
                raise RuntimeError(
                    "{} contains a non-finite tensor".format(label)
                )
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                self_label = "{}.{}".format(label, key)
                FormalIsaacEvaluator._assert_finite_tensor_tree(
                    child, label=self_label
                )
            return
        if isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                FormalIsaacEvaluator._assert_finite_tensor_tree(
                    child,
                    label="{}[{}]".format(label, index),
                )

    @staticmethod
    def _load_normalizer_state(
        runner: object,
        checkpoint: Mapping[str, object],
        *,
        attribute_name: str,
        checkpoint_key: str,
    ) -> None:
        """Load exactly one configured normalizer, with no absence fallback."""

        if not hasattr(runner, attribute_name):
            raise RuntimeError(
                "runner lacks configured normalizer attribute {}".format(
                    attribute_name
                )
            )
        normalizer = getattr(runner, attribute_name)
        has_checkpoint_state = checkpoint_key in checkpoint
        if normalizer is None:
            if has_checkpoint_state:
                raise RuntimeError(
                    "{} exists but reconstructed {} is disabled".format(
                        checkpoint_key, attribute_name
                    )
                )
            return
        if not has_checkpoint_state:
            raise RuntimeError(
                "checkpoint lacks enabled normalizer {}".format(
                    checkpoint_key
                )
            )
        state = checkpoint[checkpoint_key]
        if not isinstance(state, Mapping):
            raise RuntimeError(
                "{} must be a state mapping".format(checkpoint_key)
            )
        FormalIsaacEvaluator._assert_finite_tensor_tree(
            state, label=checkpoint_key
        )
        loader = getattr(normalizer, "load_state_dict", None)
        if not callable(loader):
            raise RuntimeError(
                "{} cannot load deterministic state".format(
                    attribute_name
                )
            )
        result = loader(state)
        # torch.nn.Module returns an _IncompatibleKeys object.  Treat any
        # reported mismatch as fatal while permitting normalizers whose
        # load_state_dict() convention returns None.
        if result is not None and (
            tuple(getattr(result, "missing_keys", ()))
            or tuple(getattr(result, "unexpected_keys", ()))
        ):
            raise RuntimeError(
                "{} did not load strictly".format(checkpoint_key)
            )
        evaluate = getattr(normalizer, "eval", None)
        if callable(evaluate):
            evaluate()

    def _load_frozen_snapshot(
        self,
        *,
        runner: object,
        checkpoint_receipt: Mapping[str, object],
        checkpoint_path: Path,
        expected_generation: int,
    ) -> None:
        """Verify a full exact checkpoint, then load only frozen inference state.

        The evaluator must not restore the trainer's live broker/pool/RNG/env
        state into its own simulator.  It nevertheless validates the complete
        exact-resume envelope and optimizer before applying actor/critic and
        both observation normalizers from the same bound bytes.
        """

        import torch

        checkpoint_bytes = inbox_protocol.read_artifact_receipt_bytes(
            checkpoint_receipt,
            label="formal sidecar checkpoint",
        )
        try:
            loaded = torch.load(
                io.BytesIO(checkpoint_bytes),
                # Keep the complete optimizer/RNG/exact-env envelope on CPU.
                # Only strict model/normalizer load_state_dict calls copy the
                # inference tensors to the evaluator GPU; otherwise a full
                # PPO optimizer snapshot needlessly doubles GPU residency.
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "formal sidecar could not decode the exact checkpoint"
            ) from exc
        if type(loaded) is not dict:
            raise RuntimeError(
                "formal sidecar checkpoint must be a dictionary"
            )
        preflight = getattr(
            runner,
            "_preflight_required_exact_resume_checkpoint",
            None,
        )
        if not callable(preflight):
            raise RuntimeError(
                "reconstructed runner lacks exact checkpoint preflight"
            )
        preflight(
            loaded,
            path=str(checkpoint_path),
            load_optimizer=True,
        )
        if (
            type(expected_generation) is not int
            or expected_generation < 0
            or loaded.get("iter") != expected_generation
        ):
            raise RuntimeError(
                "checkpoint iteration differs from frozen policy generation"
            )
        model_state = loaded.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            raise RuntimeError(
                "checkpoint lacks a model_state_dict mapping"
            )
        self._assert_finite_tensor_tree(
            model_state, label="model_state_dict"
        )
        policy = getattr(
            getattr(runner, "alg", None), "policy", None
        )
        load_policy = getattr(policy, "load_state_dict", None)
        if not callable(load_policy):
            raise RuntimeError(
                "reconstructed runner has no loadable policy"
            )
        incompatible = load_policy(model_state, strict=True)
        if (
            tuple(getattr(incompatible, "missing_keys", ()))
            or tuple(getattr(incompatible, "unexpected_keys", ()))
        ):
            raise RuntimeError(
                "checkpoint policy did not load strictly"
            )
        critic_attribute = self._critic_normalizer_name(runner)
        self._load_normalizer_state(
            runner,
            loaded,
            attribute_name="obs_normalizer",
            checkpoint_key="obs_norm_state_dict",
        )
        self._load_normalizer_state(
            runner,
            loaded,
            attribute_name=critic_attribute,
            checkpoint_key="privileged_obs_norm_state_dict",
        )
        evaluate = getattr(policy, "eval", None)
        if not callable(evaluate):
            raise RuntimeError(
                "reconstructed policy cannot enter eval mode"
            )
        evaluate()
        runner.current_learning_iteration = expected_generation + 1

    def _current_frozen_state_bindings(
        self, runner: object
    ) -> Dict[str, object]:
        critic_attribute = self._critic_normalizer_name(runner)
        return {
            "policy_state": runner._frozen_eval_state_binding(
                runner.alg.policy.state_dict()
            ),
            "actor_obs_normalizer": (
                runner._frozen_eval_state_binding(
                    self._normalizer_payload(
                        runner, "obs_normalizer"
                    )
                )
            ),
            "critic_obs_normalizer": (
                runner._frozen_eval_state_binding(
                    self._normalizer_payload(
                        runner, critic_attribute
                    )
                )
            ),
        }

    def _construct_bound_runtime(
        self,
        *,
        environment_config_pickle: Mapping[str, object],
        agent_config_pickle: Mapping[str, object],
        training_contract_path: Path,
        training_contract_sha256: str,
        training_launch_claim_sha256: str,
        expected_num_envs: Optional[int] = None,
        request_bindings: Optional[Mapping[str, object]] = None,
    ):
        """Construct the one shared pickle -> Gym -> exact runner runtime."""

        import pickle

        import gymnasium as gym

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from whole_body_tracking.utils.my_on_policy_runner import (
            MotionOnPolicyRunner,
        )

        env_bytes = inbox_protocol.read_artifact_receipt_bytes(
            environment_config_pickle,
            label="formal sidecar env.pkl",
        )
        agent_bytes = inbox_protocol.read_artifact_receipt_bytes(
            agent_config_pickle,
            label="formal sidecar agent.pkl",
        )
        env_cfg = pickle.loads(env_bytes)
        agent_cfg = pickle.loads(agent_bytes)
        scene_cfg = getattr(env_cfg, "scene", None)
        if scene_cfg is None or not hasattr(scene_cfg, "num_envs"):
            raise RuntimeError(
                "saved ActionBall env config has no scene.num_envs"
            )
        # Do not resize the environment before loading the exact checkpoint.
        # ActionBall's exact-resume payload binds the original environment
        # cardinality (including every active birth/task latch).  Replacing a
        # 4096-env training scene with a 100/1280-row evaluation scene makes
        # the strict loader reject the checkpoint, and bypassing that rejection
        # would leave the policy snapshot detached from its bound runtime
        # identity.  The proposal executor is required to batch the fixed
        # request through the saved non-zero environment count instead.
        if (
            type(scene_cfg.num_envs) is not int
            or scene_cfg.num_envs < 1
        ):
            raise RuntimeError(
                "saved ActionBall env config has an invalid scene.num_envs"
            )
        if (
            expected_num_envs is not None
            and (
                type(expected_num_envs) is not int
                or expected_num_envs < 1
                or scene_cfg.num_envs != expected_num_envs
            )
        ):
            raise RuntimeError(
                "saved ActionBall environment count differs from its claim"
            )
        sim_cfg = getattr(env_cfg, "sim", None)
        if sim_cfg is None or not hasattr(sim_cfg, "device"):
            raise RuntimeError(
                "saved ActionBall env config has no sim.device"
            )
        sim_cfg.device = self.device
        if not hasattr(agent_cfg, "device"):
            raise RuntimeError(
                "saved ActionBall agent config has no device"
            )
        agent_cfg.device = self.device
        to_dict = getattr(agent_cfg, "to_dict", None)
        if not callable(to_dict):
            raise RuntimeError(
                "saved ActionBall agent config lacks to_dict()"
            )

        self._close_environment()
        gym_env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
        wrapped = RslRlVecEnvWrapper(gym_env)
        self._active_env = wrapped
        contract = inbox_protocol.strict_read_json(
            training_contract_path,
            label="formal sidecar training contract",
        )
        if request_bindings is not None:
            self._validate_training_contract_bindings(
                contract, request_bindings
            )
        schema = contract.get("schema_version")
        if type(schema) is not int:
            raise RuntimeError(
                "formal sidecar training contract schema is invalid"
            )
        run_root = training_contract_path.parent.parent
        runner = MotionOnPolicyRunner(
            wrapped,
            to_dict(),
            log_dir=str(run_root),
            device=self.device,
            registry_name=None,
            training_contract_schema_version=schema,
            training_contract_sha256=training_contract_sha256,
            training_contract_lineage_exact=True,
            training_launch_claim_sha256=(
                training_launch_claim_sha256
            ),
            require_exact_resume_state=True,
        )
        self._active_runner = runner
        return wrapped, runner, contract

    def _build_runtime(
        self,
        request_document: Mapping[str, object],
    ):
        """Load only request-bound bytes; Hydra/default composition is absent."""

        request = inbox_protocol.validate_request_document(
            request_document
        )
        if self._progress is not None:
            self._progress.publish("runtime_building")
            self._progress.assert_before_deadline()
        bindings = request["bindings"]
        paths = self._validate_runtime_identity(request_document)
        checkpoint_path = self._artifact_path(bindings, "checkpoint")
        # Re-hash every ordered motion and immutable input before any pickle
        # is deserialized.
        inbox_protocol.verify_request_artifacts(request_document)
        wrapped, runner, _contract = self._construct_bound_runtime(
            environment_config_pickle=bindings[
                "environment_config_pickle"
            ],
            agent_config_pickle=bindings["agent_config_pickle"],
            training_contract_path=paths["training_contract_path"],
            training_contract_sha256=bindings[
                "training_contract"
            ]["sha256"],
            training_launch_claim_sha256=bindings[
                "training_launch_claim_sha256"
            ],
            request_bindings=bindings,
        )
        bind_runtime_bootstrap = getattr(
            runner,
            "bind_runtime_bootstrap_receipt",
            None,
        )
        if not callable(bind_runtime_bootstrap):
            raise RuntimeError(
                "reconstructed runner lacks runtime-bootstrap binding"
            )
        bind_runtime_bootstrap(
            content_sha256=bindings[
                "runtime_bootstrap_receipt_sha256"
            ],
            artifact_receipt=bindings[
                "runtime_bootstrap_receipt"
            ],
        )
        if (
            getattr(
                runner,
                "runtime_bootstrap_lineage_payload_sha256",
                None,
            )
            != bindings["runtime_bootstrap_lineage_payload_sha256"]
        ):
            raise RuntimeError(
                "request runtime-bootstrap lineage differs from live bytes"
            )
        expected_generation = int(bindings["policy_generation"])
        self._load_frozen_snapshot(
            runner=runner,
            checkpoint_receipt=bindings["checkpoint"],
            checkpoint_path=checkpoint_path,
            expected_generation=expected_generation,
        )
        current_bindings = self._current_frozen_state_bindings(
            runner
        )
        expected_bindings = {
            name: bindings[name] for name in current_bindings
        }
        if current_bindings != expected_bindings:
            raise RuntimeError(
                "loaded policy/normalizer state differs from frozen request"
            )
        policy = runner.get_inference_policy(
            device=wrapped.unwrapped.device
        )
        return wrapped, runner, policy

    def build_exact_resume_runtime_from_claim(
        self,
        *,
        claim_document: Mapping[str, object],
        final_checkpoint_path: object,
        _live_inventory_verification: Optional[
            Mapping[str, object]
        ] = None,
    ) -> ExactResumeRuntime:
        """Strictly restore a final checkpoint without a simulator step."""

        preimport_claim_payload = _preimport_claim_payload(
            claim_document
        )
        if _live_inventory_verification is None:
            live_inventory_verification = (
                _live_verify_runtime_inventory_cached(
                    claim_payload=preimport_claim_payload
                )
            )
        else:
            live_inventory_verification = (
                _install_live_inventory_cache(
                    claim_payload=preimport_claim_payload,
                    proof=_live_inventory_verification,
                )
            )
        import torch

        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_runtime_bootstrap as bootstrap_protocol,
        )

        if type(claim_document) is not dict:
            raise RuntimeError(
                "exact-resume runtime requires a plain launch claim"
            )
        claim_sha256 = claim_document.get("launch_claim_sha256")
        claim_payload = (
            bootstrap_protocol
            .validate_action_ball_launch_claim_document(
                claim_document,
                expected_sha256=claim_sha256,
            )
        )
        if claim_payload != preimport_claim_payload:
            raise RuntimeError(
                "pre-import/full launch claim validation differs"
            )
        checkout = Path(
            claim_payload.get("source_checkout", "")
        ).expanduser().resolve(strict=True)
        raw_checkpoint = Path(
            os.path.abspath(os.fspath(final_checkpoint_path))
        )
        if (
            not raw_checkpoint.is_absolute()
            or ".." in raw_checkpoint.parts
            or os.path.normpath(str(raw_checkpoint))
            != str(raw_checkpoint).rstrip(os.sep)
        ):
            raise RuntimeError(
                "exact-resume checkpoint path is not absolute/normalized"
            )
        checkpoint_path = raw_checkpoint.resolve(strict=True)
        if (
            not checkpoint_path.name.startswith("model_")
            or not checkpoint_path.name.endswith(".pt")
            or not checkpoint_path.name[6:-3].isdigit()
        ):
            raise RuntimeError(
                "exact-resume checkpoint is not model_<N>.pt"
            )
        allowed_root = (
            checkout
            / "hope_training"
            / "whole_body_tracking"
            / "logs"
            / "rsl_rl"
        )
        try:
            checkpoint_path.relative_to(allowed_root)
        except ValueError as exc:
            raise RuntimeError(
                "exact-resume checkpoint escaped the claim checkout log root"
            ) from exc
        raw_namespace = claim_payload.get("namespace")
        if type(raw_namespace) is not str:
            raise RuntimeError(
                "exact-resume launch namespace is not an absolute path"
            )
        namespace = Path(raw_namespace).expanduser()
        if (
            not namespace.is_absolute()
            or not namespace.name
            or ".." in namespace.parts
            or os.path.normpath(raw_namespace)
            != raw_namespace.rstrip(os.sep)
        ):
            raise RuntimeError(
                "exact-resume launch namespace is not absolute/normalized"
            )
        resolved_namespace = namespace.resolve(strict=True)
        if (
            resolved_namespace != namespace
            or not resolved_namespace.is_dir()
        ):
            raise RuntimeError(
                "exact-resume launch namespace is not one real directory"
            )
        checkpoint_receipt = inbox_protocol.artifact_receipt(
            checkpoint_path
        )
        checkpoint_bytes = inbox_protocol.read_artifact_receipt_bytes(
            checkpoint_receipt,
            label="final exact-resume checkpoint",
        )
        try:
            checkpoint = torch.load(
                io.BytesIO(checkpoint_bytes),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "final exact-resume checkpoint cannot be decoded"
            ) from exc
        if type(checkpoint) is not dict:
            raise RuntimeError(
                "final exact-resume checkpoint is not a dictionary"
            )
        filename_iteration = int(checkpoint_path.name[6:-3])
        if checkpoint.get("iter") != filename_iteration:
            raise RuntimeError(
                "final checkpoint filename/embedded iteration differ"
            )
        infos = checkpoint.get("infos")
        exact_state = (
            infos.get("hope_exact_resume_state")
            if type(infos) is dict
            else None
        )
        if type(infos) is not dict or type(exact_state) is not dict:
            raise RuntimeError(
                "final checkpoint lacks exact-resume infos/state"
            )
        binding_keys = (
            "runtime_bootstrap_receipt_sha256",
            "runtime_bootstrap_lineage_payload_sha256",
            "runtime_bootstrap_receipt",
        )
        try:
            info_bootstrap = {
                key: infos[key] for key in binding_keys
            }
            state_bootstrap = {
                key: exact_state[key] for key in binding_keys
            }
        except KeyError as exc:
            raise RuntimeError(
                "final checkpoint lacks runtime-bootstrap binding"
            ) from exc
        if info_bootstrap != state_bootstrap:
            raise RuntimeError(
                "checkpoint infos/exact-state bootstrap bindings differ"
            )
        try:
            inbox_protocol.verify_artifact_receipt(
                info_bootstrap["runtime_bootstrap_receipt"],
                label="exact-resume runtime bootstrap receipt",
            )
            raw_bootstrap_path = Path(
                info_bootstrap["runtime_bootstrap_receipt"]["path"]
            )
        except Exception as exc:
            raise RuntimeError(
                "checkpoint runtime-bootstrap artifact is invalid"
            ) from exc
        if (
            not raw_bootstrap_path.is_absolute()
            or ".." in raw_bootstrap_path.parts
            or os.path.normpath(str(raw_bootstrap_path))
            != str(raw_bootstrap_path).rstrip(os.sep)
        ):
            raise RuntimeError(
                "checkpoint runtime-bootstrap path is not normalized"
            )
        bootstrap_path = raw_bootstrap_path.resolve(strict=True)
        if (
            bootstrap_path != raw_bootstrap_path
            or bootstrap_path.name
            != bootstrap_protocol.RECEIPT_FILENAME
            or bootstrap_path.parent.name != "params"
        ):
            raise RuntimeError(
                "checkpoint runtime-bootstrap path is not its fixed run "
                "receipt"
            )
        params_dir = bootstrap_path.parent
        rsl_log_dir = params_dir.parent
        try:
            rsl_log_dir.relative_to(allowed_root)
        except ValueError as exc:
            raise RuntimeError(
                "checkpoint runtime-bootstrap run escaped the claim log root"
            ) from exc
        if (
            not rsl_log_dir.is_dir()
            or not rsl_log_dir.name.endswith("_" + namespace.name)
        ):
            raise RuntimeError(
                "runtime-bootstrap run does not bind the claim namespace"
            )
        roundtrip_dir = rsl_log_dir / (
            "exact_resume_roundtrip_" + claim_sha256[:16]
        )
        if checkpoint_path.parent == rsl_log_dir:
            pass
        elif checkpoint_path.parent == roundtrip_dir:
            entries = tuple(roundtrip_dir.iterdir())
            if (
                len(entries) != 1
                or entries[0] != checkpoint_path
                or entries[0].is_symlink()
                or not entries[0].is_file()
            ):
                raise RuntimeError(
                    "exact-resume verifier roundtrip directory is not "
                    "single-checkpoint"
                )
        else:
            raise RuntimeError(
                "exact-resume checkpoint is neither the final run checkpoint "
                "nor its exact verifier roundtrip"
            )
        checkpoint_iterations = []
        for candidate in rsl_log_dir.iterdir():
            name = candidate.name
            if (
                candidate.is_file()
                and not candidate.is_symlink()
                and name.startswith("model_")
                and name.endswith(".pt")
                and name[6:-3].isdigit()
            ):
                checkpoint_iterations.append(int(name[6:-3]))
        filename_iteration = int(checkpoint_path.name[6:-3])
        source_checkpoint_path = rsl_log_dir / checkpoint_path.name
        if (
            not checkpoint_iterations
            or filename_iteration != max(checkpoint_iterations)
            or not source_checkpoint_path.is_file()
            or source_checkpoint_path.is_symlink()
        ):
            raise RuntimeError(
                "exact-resume input does not correspond to the final "
                "checkpoint in its run"
            )
        expected_bootstrap_artifact = inbox_protocol.artifact_receipt(
            bootstrap_path
        )
        if (
            info_bootstrap["runtime_bootstrap_receipt"]
            != expected_bootstrap_artifact
        ):
            raise RuntimeError(
                "checkpoint bootstrap artifact is not its fixed run receipt"
            )
        bootstrap_document = inbox_protocol.strict_read_json(
            bootstrap_path,
            label="final runtime bootstrap receipt",
        )
        bootstrap_content = bootstrap_document.get("content")
        if (
            type(bootstrap_content) is not dict
            or bootstrap_document.get("content_sha256")
            != info_bootstrap[
                "runtime_bootstrap_receipt_sha256"
            ]
            or bootstrap_protocol
            .runtime_bootstrap_lineage_payload_sha256(
                bootstrap_content
            )
            != info_bootstrap[
                "runtime_bootstrap_lineage_payload_sha256"
            ]
        ):
            raise RuntimeError(
                "checkpoint runtime-bootstrap canonical binding differs"
            )
        claim_path = bootstrap_content["launch_claim"]["path"]
        persisted_claim = inbox_protocol.strict_read_json(
            claim_path,
            label="runtime bootstrap launch claim",
        )
        if persisted_claim != claim_document:
            raise RuntimeError(
                "factory launch claim differs from bootstrap-bound bytes"
            )
        training_contract_path = params_dir / "training_contract.json"
        env_pickle_path = params_dir / "env.pkl"
        agent_pickle_path = params_dir / "agent.pkl"
        runtime_identity_path = (
            params_dir / "action_ball_frozen_eval_runtime.json"
        )
        bootstrap_protocol.validate_runtime_bootstrap_receipt_document(
            bootstrap_document,
            expected_repo_root=checkout,
            expected_task_id=TASK_ID,
            expected_training_launch_claim_sha256=claim_sha256,
            expected_launch_claim_path=claim_path,
            expected_training_contract_path=training_contract_path,
            expected_environment_config_pickle_path=env_pickle_path,
            expected_agent_config_pickle_path=agent_pickle_path,
            expected_runtime_identity_path=runtime_identity_path,
            expected_runtime_inventory_path=bootstrap_content[
                "runtime_inventory"
            ]["path"],
            expected_source_commit_oid=claim_payload.get(
                "source_commit_sha"
            ),
        )
        training_contract_receipt = bootstrap_content[
            "training_contract"
        ]
        if (
            infos.get("training_contract_schema_version") != 3
            or infos.get("training_contract_sha256")
            != training_contract_receipt["sha256"]
            or infos.get("training_contract_lineage_exact") != 1
            or infos.get("training_launch_claim_sha256")
            != claim_sha256
        ):
            raise RuntimeError(
                "final checkpoint training/claim lineage is not exact"
            )
        stage_budget = claim_payload.get("stage_budget")
        if (
            type(stage_budget) is not dict
            or type(stage_budget.get("num_envs")) is not int
            or stage_budget["num_envs"] < 1
            or type(stage_budget.get("max_iterations")) is not int
            or stage_budget["max_iterations"] < 1
        ):
            raise RuntimeError(
                "launch claim lacks exact positive stage budget"
            )
        if filename_iteration not in (
            stage_budget["max_iterations"] - 1,
            stage_budget["max_iterations"],
        ):
            raise RuntimeError(
                "final checkpoint iteration differs from stage budget"
            )
        try:
            wrapped, runner, _contract = self._construct_bound_runtime(
                environment_config_pickle=bootstrap_content[
                    "environment_config_pickle"
                ],
                agent_config_pickle=bootstrap_content[
                    "agent_config_pickle"
                ],
                training_contract_path=training_contract_path,
                training_contract_sha256=training_contract_receipt[
                    "sha256"
                ],
                training_launch_claim_sha256=claim_sha256,
                expected_num_envs=stage_budget["num_envs"],
            )
            bind_runtime_bootstrap = getattr(
                runner,
                "bind_runtime_bootstrap_receipt",
                None,
            )
            if not callable(bind_runtime_bootstrap):
                raise RuntimeError(
                    "exact-resume runner lacks runtime-bootstrap binding"
                )
            bind_runtime_bootstrap(
                content_sha256=info_bootstrap[
                    "runtime_bootstrap_receipt_sha256"
                ],
                artifact_receipt=info_bootstrap[
                    "runtime_bootstrap_receipt"
                ],
            )
            if (
                getattr(
                    runner,
                    "runtime_bootstrap_lineage_payload_sha256",
                    None,
                )
                != info_bootstrap[
                    "runtime_bootstrap_lineage_payload_sha256"
                ]
            ):
                raise RuntimeError(
                    "exact-resume runner bootstrap lineage differs"
                )
            immutable_loader = getattr(
                runner,
                "load_formal_action_ball_checkpoint_bytes",
                None,
            )
            if not callable(immutable_loader):
                raise RuntimeError(
                    "runner lacks immutable safe formal checkpoint load"
                )
            immutable_loader(
                checkpoint_bytes,
                checkpoint_path=str(checkpoint_path),
                expected_sha256=checkpoint_receipt["sha256"],
                expected_size_bytes=checkpoint_receipt["size_bytes"],
                load_optimizer=True,
            )
            if (
                getattr(
                    runner,
                    "_exact_resume_roundtrip_pending",
                    None,
                )
                is not True
                or getattr(
                    runner,
                    "_action_ball_resume_reset_pending",
                    None,
                )
                is not True
                or getattr(
                    runner,
                    "_exact_resume_loaded_source_iteration",
                    None,
                )
                != filename_iteration
                or int(runner.current_learning_iteration)
                != filename_iteration + 1
            ):
                raise RuntimeError(
                    "runner did not enter the exact no-step restore window"
                )
            live_state_method = getattr(
                runner, "exact_resume_live_state_receipt", None
            )
            if not callable(live_state_method):
                raise RuntimeError(
                    "runner lacks exact-resume live-state proof"
                )
            exact_resume_live_state = live_state_method()
            if (
                type(exact_resume_live_state) is not dict
                or set(exact_resume_live_state)
                != {
                    "schema_version",
                    "kind",
                    "content",
                    "content_sha256",
                }
                or exact_resume_live_state["schema_version"] != 1
                or exact_resume_live_state["kind"]
                != "action_ball_exact_resume_live_state"
                or type(exact_resume_live_state["content"]) is not dict
                or set(exact_resume_live_state["content"])
                != {
                    "schema_version",
                    "kind",
                    "source_embedded_iteration",
                    "current_learning_iteration",
                    "roundtrip_pending",
                    "resume_reset_pending",
                    "model_state_sha256",
                    "optimizer_state_sha256",
                    "actor_normalizer_state_sha256",
                    "critic_normalizer_state_sha256",
                    "exact_resume_state_sha256",
                    "environment_resume_state_sha256",
                    "rng_state_sha256",
                    "runtime_bootstrap_binding_sha256",
                    "common_step_counter",
                    "common_step_counter_delta",
                    "live_core_sha256",
                }
                or exact_resume_live_state["content_sha256"]
                != inbox_protocol.canonical_sha256(
                    exact_resume_live_state["content"]
                )
            ):
                raise RuntimeError(
                    "runner exact-resume live-state proof is invalid"
                )
            live_content = exact_resume_live_state["content"]
            live_digests = (
                "model_state_sha256",
                "optimizer_state_sha256",
                "actor_normalizer_state_sha256",
                "critic_normalizer_state_sha256",
                "exact_resume_state_sha256",
                "environment_resume_state_sha256",
                "rng_state_sha256",
                "runtime_bootstrap_binding_sha256",
                "live_core_sha256",
            )
            if (
                live_content["schema_version"] != 1
                or live_content["kind"]
                != "action_ball_exact_resume_live_state"
                or live_content["source_embedded_iteration"]
                != filename_iteration
                or live_content["current_learning_iteration"]
                != filename_iteration + 1
                or live_content["roundtrip_pending"] is not True
                or live_content["resume_reset_pending"] is not True
                or type(live_content["common_step_counter"]) is not int
                or live_content["common_step_counter"] < 0
                or live_content["common_step_counter_delta"] != 0
                or any(
                    type(live_content[name]) is not str
                    or len(live_content[name]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in live_content[name]
                    )
                    for name in live_digests
                )
            ):
                raise RuntimeError(
                    "runner exact-resume live-state invariants differ"
                )
            # No env.reset(), env.step(), policy action, or optimizer update is
            # permitted before this receipt is returned.
            inbox_protocol.verify_artifact_receipt(
                checkpoint_receipt,
                label="post-load final checkpoint",
            )
            bootstrap_protocol.validate_runtime_bootstrap_receipt_document(
                bootstrap_document,
                expected_repo_root=checkout,
                expected_task_id=TASK_ID,
                expected_training_launch_claim_sha256=claim_sha256,
                expected_launch_claim_path=claim_path,
                expected_training_contract_path=training_contract_path,
                expected_environment_config_pickle_path=env_pickle_path,
                expected_agent_config_pickle_path=agent_pickle_path,
                expected_runtime_identity_path=runtime_identity_path,
                expected_runtime_inventory_path=bootstrap_content[
                    "runtime_inventory"
                ]["path"],
                expected_source_commit_oid=claim_payload.get(
                    "source_commit_sha"
                ),
            )
            receipt_content = {
                "schema_version": 1,
                "kind": "action_ball_exact_resume_runtime_construction",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_receipt["sha256"],
                "checkpoint_size_bytes": checkpoint_receipt[
                    "size_bytes"
                ],
                "checkpoint_iteration": filename_iteration,
                "load_optimizer": True,
                "bootstrap_content_sha256": info_bootstrap[
                    "runtime_bootstrap_receipt_sha256"
                ],
                "bootstrap_artifact_sha256": (
                    expected_bootstrap_artifact["sha256"]
                ),
                "bootstrap_artifact_size_bytes": (
                    expected_bootstrap_artifact["size_bytes"]
                ),
                "bootstrap_lineage_payload_sha256": info_bootstrap[
                    "runtime_bootstrap_lineage_payload_sha256"
                ],
                "runtime_inventory_live_verification": (
                    live_inventory_verification
                ),
                "exact_resume_live_state": exact_resume_live_state,
                "training_contract_sha256": (
                    training_contract_receipt["sha256"]
                ),
                "training_launch_claim_sha256": claim_sha256,
                "environment_count": stage_budget["num_envs"],
                "runner_current_learning_iteration": int(
                    runner.current_learning_iteration
                ),
            }
            construction_receipt = {
                "schema_version": 1,
                "kind": (
                    "action_ball_exact_resume_runtime_construction"
                ),
                "content": receipt_content,
                "content_sha256": inbox_protocol.canonical_sha256(
                    receipt_content
                ),
            }
            return ExactResumeRuntime(
                wrapped_env=wrapped,
                runner=runner,
                construction_receipt=construction_receipt,
                _owner=self,
            )
        except Exception:
            self._close_environment()
            raise

    def evaluate(
        self, request_document: Mapping[str, object]
    ) -> Dict[str, object]:
        request = inbox_protocol.validate_request_document(
            request_document
        )
        proposal_total = sum(
            int(window["proposal_count"])
            for window in request["windows"]
        )
        wrapped, runner, policy = self._build_runtime(
            request_document
        )
        runtime_env = wrapped.unwrapped
        manager = getattr(runtime_env, "command_manager", None)
        if manager is None:
            raise RuntimeError(
                "formal ActionBall evaluator has no command manager"
            )
        matches = []
        for raw_name in tuple(getattr(manager, "active_terms", ())):
            term = manager.get_term(raw_name)
            hook = getattr(term, self.runtime_hook_name, None)
            if callable(hook):
                matches.append(hook)
        if len(matches) != 1:
            raise RuntimeError(
                "formal evaluator requires exactly one ActionBall runtime "
                "hook"
            )
        self._validate_action_ball_term_identity(
            getattr(matches[0], "__self__", None),
            request,
        )
        if self._progress is not None:
            self._progress.publish("evaluating")
            self._progress.assert_before_deadline()

        def report_progress(
            attempts_completed: int, attempts_total: int
        ) -> None:
            if (
                type(attempts_total) is not int
                or attempts_total != proposal_total
            ):
                raise RuntimeError(
                    "formal runtime reported a wrong proposal total"
                )
            if self._progress is not None:
                self._progress.publish(
                    "evaluating",
                    attempts_completed=attempts_completed,
                )
                self._progress.assert_before_deadline()

        with __import__("torch").inference_mode():
            attempts = matches[0](
                request_document=request_document,
                vector_env=wrapped,
                runner=runner,
                deterministic_policy=policy,
                expected_task_id=TASK_ID,
                expected_policy_generation=request["bindings"][
                    "policy_generation"
                ],
                expected_proposal_sampler_contract_sha256=request[
                    "bindings"
                ]["proposal_sampler_contract_sha256"],
                progress_callback=report_progress,
                request_deadline_monotonic_ns=(
                    0
                    if self._progress is None
                    else self._progress
                    .request_deadline_monotonic_ns
                ),
            )
        if self._progress is not None:
            self._progress.publish(
                "validating_evidence",
                attempts_completed=proposal_total,
            )
            self._progress.assert_before_deadline()
        after_bindings = self._current_frozen_state_bindings(
            runner
        )
        expected_bindings = {
            name: request["bindings"][name]
            for name in after_bindings
        }
        if (
            after_bindings != expected_bindings
            or int(runner.current_learning_iteration)
            != request["bindings"]["policy_generation"] + 1
        ):
            raise RuntimeError(
                "formal evaluation mutated its frozen policy/normalizers/"
                "generation"
            )
        self._validate_proposal_receipt_bindings(
            request=request,
            attempts_by_role=attempts,
        )
        # Build once here as a complete schema/conservation preflight. The
        # caller will build the exact same evidence document for publication.
        inbox_protocol.build_evidence_document(
            request_document,
            sidecar_launch_sha256=request[
                "sidecar_launch_sha256"
            ],
            attempts_by_role=attempts,
        )
        return attempts


def build_exact_resume_runtime_from_claim(
    *,
    claim_document: Mapping[str, object],
    final_checkpoint_path: object,
    device: str = "cuda:0",
    _preimport_live_inventory_verification: Optional[
        Mapping[str, object]
    ] = None,
) -> ExactResumeRuntime:
    """Public shared constructor; AppLauncher must already be running."""

    preimport_claim_payload = _preimport_claim_payload(claim_document)
    if _preimport_live_inventory_verification is None:
        live_inventory_verification = (
            _live_verify_runtime_inventory_cached(
                claim_payload=preimport_claim_payload
            )
        )
    else:
        live_inventory_verification = _install_live_inventory_cache(
            claim_payload=preimport_claim_payload,
            proof=_preimport_live_inventory_verification,
        )
    evaluator = FormalIsaacEvaluator(device=device)
    try:
        return evaluator.build_exact_resume_runtime_from_claim(
            claim_document=claim_document,
            final_checkpoint_path=final_checkpoint_path,
            _live_inventory_verification=(
                live_inventory_verification
            ),
        )
    except Exception:
        evaluator._close_environment()
        raise


def process_one(
    *,
    inbox_root: object,
    owner_id: str,
    run_id: str,
    launch_document: Mapping[str, object],
    backend: object,
    progress: Optional[SidecarProgressPublisher] = None,
    request_deadline_s: Optional[float] = None,
) -> Optional[Path]:
    """Process the next request, or return ``None`` while awaiting work/ACK."""

    backend_contract_sha256 = getattr(
        backend, "backend_contract_sha256", None
    )
    if not isinstance(backend_contract_sha256, str):
        raise inbox_protocol.EvaluationInboxError(
            "sidecar backend lacks backend_contract_sha256"
        )
    inbox_protocol.validate_sidecar_launch_document(
        launch_document,
        actual_sidecar_code_sha256=sidecar_code_sha256(),
        backend_contract_sha256=backend_contract_sha256,
        require_trust=True,
    )
    queue = inbox_protocol.EvaluationInbox(inbox_root)
    queue.initialize()
    request = queue.next_pending_request(owner_id, run_id)
    if request is None:
        if progress is not None:
            progress.waiting_for_request_or_ack()
            progress.raise_if_failed()
        return None
    request_content = inbox_protocol.validate_request_document(
        request,
        expected_owner_id=owner_id,
        expected_run_id=run_id,
        expected_sidecar_launch_sha256=launch_document[
            "content_sha256"
        ],
    )
    inbox_protocol.verify_request_artifacts(request)
    evaluate = getattr(backend, "evaluate", None)
    if not callable(evaluate):
        raise inbox_protocol.EvaluationInboxError(
            "sidecar backend lacks evaluate()"
        )
    attempts_total = sum(
        int(window["proposal_count"])
        for window in request_content["windows"]
    )
    if progress is not None:
        if request_deadline_s is None:
            raise inbox_protocol.EvaluationInboxError(
                "formal sidecar progress requires a request deadline"
            )
        progress.begin_request(
            request_seq=request_content["request_seq"],
            request_sha256=request["content_sha256"],
            attempts_total=attempts_total,
            deadline_s=request_deadline_s,
        )
        progress.raise_if_failed()
    try:
        attempts_by_role = evaluate(request)
        if progress is not None:
            progress.assert_before_deadline()
        # Detect mutable checkpoint/motion/config bytes before any evidence can
        # leave the evaluator.  Formal launch uses a clean no-clobber checkout,
        # but the transport still fails closed if that external invariant breaks.
        inbox_protocol.verify_request_artifacts(request)
        evidence = inbox_protocol.build_evidence_document(
            request,
            sidecar_launch_sha256=request_content[
                "sidecar_launch_sha256"
            ],
            attempts_by_role=attempts_by_role,
        )
        if progress is not None:
            progress.assert_before_deadline()
        result = queue.publish_evidence(evidence)
        if progress is not None:
            progress.publish(
                "evidence_published",
                attempts_completed=attempts_total,
            )
            progress.raise_if_failed()
        return result
    except Exception as exc:
        if progress is not None:
            progress.publish(
                "request_failed",
                error_type=type(exc).__name__,
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process an append-only action-ball frozen-evaluation inbox"
        )
    )
    parser.add_argument("--inbox-root", required=True)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--launch", required=True)
    parser.add_argument(
        "--backend",
        choices=("formal",),
        required=True,
        help="the CLI exposes only the independent formal Isaac backend",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--heartbeat-interval-s",
        type=float,
        required=True,
        help=(
            "must exactly equal the code-pinned launch receipt heartbeat "
            "interval"
        ),
    )
    parser.add_argument(
        "--request-deadline-s",
        type=float,
        required=True,
        help=(
            "must exactly equal the code-pinned launch receipt per-request "
            "deadline"
        ),
    )
    return parser


def _launch_heartbeat_settings(
    launch_content: Mapping[str, object],
    *,
    cli_heartbeat_interval_s: float,
    cli_request_deadline_s: float,
) -> Dict[str, float]:
    contract = launch_content.get("heartbeat_contract")
    expected_keys = {
        "schema_version",
        "heartbeat_interval_seconds",
        "heartbeat_stale_after_seconds",
        "request_deadline_seconds",
    }
    if type(contract) is not dict or set(contract) != expected_keys:
        raise RuntimeError(
            "sidecar launch receipt lacks the exact heartbeat contract"
        )
    interval = contract["heartbeat_interval_seconds"]
    stale = contract["heartbeat_stale_after_seconds"]
    deadline = contract["request_deadline_seconds"]
    if (
        contract["schema_version"] != 1
        or type(interval) not in (int, float)
        or type(stale) not in (int, float)
        or type(deadline) not in (int, float)
        or not 0.25 <= float(interval) <= 60.0
        or not float(stale) > 2.0 * float(interval)
        or not 1.0 <= float(deadline) <= 86400.0
        or float(cli_heartbeat_interval_s) != float(interval)
        or float(cli_request_deadline_s) != float(deadline)
    ):
        raise RuntimeError(
            "sidecar CLI heartbeat/deadline differs from its launch receipt"
        )
    return {
        "heartbeat_interval_s": float(interval),
        "heartbeat_stale_after_s": float(stale),
        "request_deadline_s": float(deadline),
    }


def _ready_line(
    *,
    owner_id: str,
    run_id: str,
    device: str,
    launch_receipt_canonical_sha256: str,
) -> str:
    document = {
        "schema_version": 1,
        "kind": READY_KIND,
        "owner_id": owner_id,
        "run_id": run_id,
        "backend": "formal",
        "device": device,
        "launch_receipt_canonical_sha256": (
            launch_receipt_canonical_sha256
        ),
    }
    return "ACTION_BALL_SIDECAR_READY " + json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    launch = inbox_protocol.strict_read_json(
        args.launch, label="sidecar launch document"
    )
    launch_content = inbox_protocol.validate_sidecar_launch_document(
        launch,
        actual_sidecar_code_sha256=sidecar_code_sha256(),
        backend_contract_sha256=(
            FORMAL_ISAAC_BACKEND_CONTRACT_SHA256
        ),
        require_trust=True,
    )
    heartbeat_settings = _launch_heartbeat_settings(
        launch_content,
        cli_heartbeat_interval_s=args.heartbeat_interval_s,
        cli_request_deadline_s=args.request_deadline_s,
    )
    progress = SidecarProgressPublisher(
        inbox_root=args.inbox_root,
        owner_id=args.owner_id,
        run_id=args.run_id,
        launch_sha256=launch["content_sha256"],
        interval_s=heartbeat_settings["heartbeat_interval_s"],
    )
    progress.start()
    simulation_app = None
    backend = None
    try:
        # Kit must launch before any Isaac/Gym task imports in the evaluator.
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(
            headless=True,
            device=args.device,
            enable_cameras=False,
        )
        simulation_app = app_launcher.app
        backend = FormalIsaacEvaluator(
            device=args.device,
            progress=progress,
        )
        progress.publish("ready")
        progress.raise_if_failed()
        print(
            _ready_line(
                owner_id=args.owner_id,
                run_id=args.run_id,
                device=args.device,
                launch_receipt_canonical_sha256=launch[
                    "content_sha256"
                ],
            ),
            flush=True,
        )
        while simulation_app.is_running():
            progress.raise_if_failed()
            evidence_path = process_one(
                inbox_root=args.inbox_root,
                owner_id=args.owner_id,
                run_id=args.run_id,
                launch_document=launch,
                backend=backend,
                progress=progress,
                request_deadline_s=heartbeat_settings[
                    "request_deadline_s"
                ],
            )
            if evidence_path is not None:
                print(
                    "ACTION_BALL_SIDECAR_EVIDENCE "
                    + json.dumps(
                        {"path": str(evidence_path)},
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )
            time.sleep(1.0)
        return 0
    finally:
        try:
            if backend is not None:
                backend._close_environment()
        finally:
            try:
                if simulation_app is not None:
                    simulation_app.close()
            finally:
                progress.stop()


if __name__ == "__main__":
    raise SystemExit(main())
