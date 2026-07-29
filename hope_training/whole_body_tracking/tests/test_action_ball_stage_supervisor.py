from __future__ import annotations

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/action_ball_stage_supervisor.py"
SPEC = importlib.util.spec_from_file_location(
    "action_ball_stage_supervisor", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

EXACT_SOURCE = ROOT / "scripts/exact_process_group.py"
EXACT_SPEC = importlib.util.spec_from_file_location(
    "_test_action_ball_exact_process_group", EXACT_SOURCE
)
assert EXACT_SPEC is not None and EXACT_SPEC.loader is not None
EXACT = importlib.util.module_from_spec(EXACT_SPEC)
sys.modules[EXACT_SPEC.name] = EXACT
EXACT_SPEC.loader.exec_module(EXACT)

NOSITE_SOURCE = ROOT / "scripts/action_ball_python_nosite_bootstrap.py"
NOSITE_SPEC = importlib.util.spec_from_file_location(
    "_test_action_ball_python_nosite_bootstrap", NOSITE_SOURCE
)
assert NOSITE_SPEC is not None and NOSITE_SPEC.loader is not None
NOSITE = importlib.util.module_from_spec(NOSITE_SPEC)
sys.modules[NOSITE_SPEC.name] = NOSITE
NOSITE_SPEC.loader.exec_module(NOSITE)


class HostExact:
    """Small host-only identity double; production uses Linux /proc helper."""

    def __init__(self) -> None:
        self.identities: dict[int, SimpleNamespace] = {}

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        try:
            os.write(
                fd,
                (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode(),
            )
        finally:
            os.close(fd)

    @staticmethod
    def _read(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def bind_leader(
        self, _proc_root: Path, pid: int, pgid: int, output: Path
    ) -> dict:
        assert pid == pgid
        assert os.getpgid(pid) == pgid
        identity = SimpleNamespace(
            pid=pid,
            pgid=pgid,
            starttime_ticks=time.monotonic_ns(),
        )
        self.identities[pid] = identity
        value = {
            "schema_version": 1,
            "kind": "leader_identity",
            "leader": vars(identity),
        }
        self._write(output, value)
        return value

    @staticmethod
    def _leader_from(value: dict) -> SimpleNamespace:
        return SimpleNamespace(**value["leader"])

    def _group_exists(self, pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # A rapidly recycled host-only PID/PGID can now belong to an
            # unrelated process.  The production helper uses captured Linux
            # /proc starttime identity instead of this lightweight double.
            return False
        return True

    def group_snapshot(self, _proc_root: Path, pgid: int) -> list:
        identity = self.identities.get(pgid)
        if identity is None or not self._group_exists(pgid):
            return []
        return [identity]

    def term_group(
        self, _proc_root: Path, leader_evidence: Path, output: Path
    ) -> dict:
        leader = self._leader_from(self._read(leader_evidence))
        assert os.getpgid(leader.pid) == leader.pgid
        value = {
            "schema_version": 1,
            "kind": "pre_term_group_identity",
            "leader": vars(leader),
            "members": [vars(leader)],
        }
        self._write(output, value)
        os.killpg(leader.pgid, signal.SIGTERM)
        return value

    def verify_residual(
        self, _proc_root: Path, group_evidence: Path
    ) -> list:
        leader = self._leader_from(self._read(group_evidence))
        return self.group_snapshot(Path("/unused"), leader.pgid)

    def kill_residual(
        self, _proc_root: Path, term_evidence: Path, output: Path
    ) -> dict:
        leader = self._leader_from(self._read(term_evidence))
        members = self.verify_residual(Path("/unused"), term_evidence)
        value = {
            "schema_version": 1,
            "kind": "pre_kill_group_identity",
            "leader": vars(leader),
            "members": [vars(item) for item in members],
        }
        self._write(output, value)
        if members:
            os.killpg(leader.pgid, signal.SIGKILL)
        return value


def _locked_fd(path: Path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def _runtime_paths(tmp_path: Path) -> M.RuntimePaths:
    boot = tmp_path / "kit_boot.lock"
    trainer = tmp_path / "gpu0.lock"
    evaluator = tmp_path / "gpu1.lock"
    boot.write_bytes(b"do-not-truncate\n")
    trainer.touch()
    evaluator.touch()
    return M.RuntimePaths(
        boot_lock=boot,
        trainer_lock=trainer,
        evaluator_lock=evaluator,
    )


def _ready(owner: str = "Franco", run_id: str = "run-smoke") -> dict:
    return {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.formal_sidecar_ready",
        "owner_id": owner,
        "run_id": run_id,
        "backend": "formal",
        "device": "cuda:0",
        "launch_receipt_canonical_sha256": "c" * 64,
    }


def _setup_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkout = tmp_path / "checkout"
    wbt = checkout / "hope_training/whole_body_tracking"
    wbt.mkdir(parents=True)
    setup = wbt / "setup_train_env.sh"
    setup.write_text(
        "export ACTION_BALL_TEST_SETUP=ready\n",
        encoding="utf-8",
    )
    namespace = tmp_path / "namespace"
    namespace.mkdir()
    (namespace / "live_gpu_admission.json").write_text(
        "{}\n", encoding="utf-8"
    )
    events = tmp_path / "events.txt"
    return checkout, setup, namespace


def _sidecar_code(
    *,
    ready: dict,
    events: Path,
    trainer_fd: int,
    evaluator_fd: int,
    heartbeat_path: Path,
    exit_after: float | None = None,
    duplicate_ready: bool = False,
    emit_ready: bool = True,
) -> str:
    ready_line = (
        "ACTION_BALL_SIDECAR_READY "
        + json.dumps(ready, sort_keys=True, separators=(",", ":"))
    )
    exit_clause = (
        ""
        if exit_after is None
        else f"time.sleep({exit_after!r}); sys.exit(7)"
    )
    duplicate = (
        f"print({ready_line!r}, flush=True)"
        if duplicate_ready and emit_ready
        else ""
    )
    first_ready = (
        f"print({ready_line!r}, flush=True)" if emit_ready else ""
    )
    return textwrap.dedent(
        f"""
        import hashlib
        import json
        import os
        import pathlib
        import signal
        import sys
        import threading
        import time

        os.fstat({trainer_fd})
        os.fstat({evaluator_fd})
        heartbeat_path = pathlib.Path({str(heartbeat_path)!r})
        heartbeat_seq = 0
        heartbeat_stop = threading.Event()
        def publish_heartbeat():
            global heartbeat_seq
            while True:
                content = {{
                    "owner_id": {ready["owner_id"]!r},
                    "run_id": {ready["run_id"]!r},
                    "pid": os.getpid(),
                    "sidecar_code_sha256": {"f" * 64!r},
                    "launch_sha256": {"c" * 64!r},
                    "backend_contract_sha256": {"b" * 64!r},
                    "heartbeat_seq": heartbeat_seq,
                    "phase": "ready",
                    "request_seq": None,
                    "request_sha256": "",
                    "attempts_completed": 0,
                    "attempts_total": 0,
                    "request_started_unix_ns": 0,
                    "request_started_monotonic_ns": 0,
                    "request_deadline_unix_ns": 0,
                    "request_deadline_monotonic_ns": 0,
                    "heartbeat_unix_ns": time.time_ns(),
                    "heartbeat_monotonic_ns": time.monotonic_ns(),
                    "error_type": "",
                }}
                document = {{
                    "schema_version": 1,
                    "kind": (
                        "whole_body_tracking.action_ball."
                        "formal_sidecar_heartbeat"
                    ),
                    "content": content,
                    "content_sha256": hashlib.sha256(
                        json.dumps(
                            content, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                }}
                temporary = heartbeat_path.with_name(
                    ".heartbeat." + str(heartbeat_seq) + ".tmp"
                )
                temporary.write_text(
                    json.dumps(
                        document, sort_keys=True, separators=(",", ":")
                    ) + "\\n",
                    encoding="utf-8",
                )
                os.replace(temporary, heartbeat_path)
                heartbeat_seq += 1
                if heartbeat_stop.wait(0.02):
                    return
        heartbeat_thread = threading.Thread(
            target=publish_heartbeat, daemon=True
        )
        heartbeat_thread.start()
        heartbeat_deadline = time.monotonic() + 1.0
        while not heartbeat_path.is_file():
            if time.monotonic() >= heartbeat_deadline:
                raise RuntimeError("heartbeat fixture did not publish")
            time.sleep(0.001)
        with open({str(events)!r}, "a", encoding="utf-8") as handle:
            handle.write("sidecar_ready cuda=" + os.environ["CUDA_VISIBLE_DEVICES"] + "\\n")
        def stop(_sig, _frame):
            heartbeat_stop.set()
            with open({str(events)!r}, "a", encoding="utf-8") as handle:
                handle.write("sidecar_term\\n")
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        {first_ready}
        {duplicate}
        {exit_clause}
        while True:
            time.sleep(0.02)
        """
    )


def _trainer_code(
    *,
    events: Path,
    trainer_fd: int,
    evaluator_fd: int,
    checkout: Path,
    namespace: Path,
    print_learning: bool = True,
    exit_code: int = 0,
    sleep_seconds: float = 0.15,
    loop_until_term: bool = False,
) -> str:
    learning = (
        'print("Learning iteration 0/1", flush=True)'
        if print_learning
        else ""
    )
    if loop_until_term:
        body = textwrap.dedent(
            f"""
            def stop(_sig, _frame):
                with open({str(events)!r}, "a", encoding="utf-8") as handle:
                    handle.write("trainer_term\\n")
                raise SystemExit(0)
            signal.signal(signal.SIGTERM, stop)
            while True:
                time.sleep(0.02)
            """
        )
    else:
        body = f"time.sleep({sleep_seconds!r}); raise SystemExit({exit_code})"
    preamble = textwrap.dedent(
        f"""\
        import os
        import pathlib
        import signal
        import time

        os.fstat({trainer_fd})
        os.fstat({evaluator_fd})
        output = (
            pathlib.Path({str(checkout)!r})
            / "hope_training/whole_body_tracking/logs/rsl_rl"
            / {M.ACTION_BALL_EXPERIMENT_NAME!r}
            / ("2026-07-29_00-00-00_" + {namespace.name!r})
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "model_1.pt").write_bytes(b"exact checkpoint fixture\\n")
        print(
            "[INFO] Task: {M.ACTION_BALL_TASK_ID} | "
            "experiment: {M.ACTION_BALL_EXPERIMENT_NAME} | log: "
            + str(output),
            flush=True,
        )
        with open({str(events)!r}, "a", encoding="utf-8") as handle:
            handle.write("trainer_started cuda=" + os.environ["CUDA_VISIBLE_DEVICES"] + "\\n")
        """
    )
    return preamble + learning + "\n" + body + "\n"


def _verifier_code(
    *,
    events: Path,
    trainer_fd: int,
    evaluator_fd: int,
    boot_lock_path: Path,
    exit_code: int = 0,
) -> str:
    refusal = (
        ""
        if exit_code == 0
        else f"raise SystemExit({exit_code})"
    )
    return textwrap.dedent(
        f"""\
        import argparse
        import fcntl
        import hashlib
        import json
        import os
        import pathlib
        import re

        parser = argparse.ArgumentParser()
        parser.add_argument("--claim", required=True)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--out", required=True)
        args = parser.parse_args()
        os.fstat({trainer_fd})
        os.fstat({evaluator_fd})
        if os.environ["CUDA_VISIBLE_DEVICES"] != "0":
            raise SystemExit(12)
        boot_identity = os.stat({str(boot_lock_path)!r})
        inherited_boot = False
        for descriptor in range(3, 256):
            try:
                candidate = os.fstat(descriptor)
            except OSError:
                continue
            if (
                candidate.st_dev,
                candidate.st_ino,
            ) == (
                boot_identity.st_dev,
                boot_identity.st_ino,
            ):
                inherited_boot = True
                break
        if not inherited_boot:
            raise SystemExit(14)
        boot_descriptor = os.open(
            {str(boot_lock_path)!r}, os.O_RDWR | os.O_APPEND
        )
        try:
            try:
                fcntl.flock(
                    boot_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError:
                pass
            else:
                raise SystemExit(13)
        finally:
            os.close(boot_descriptor)
        with open({str(events)!r}, "a", encoding="utf-8") as handle:
            handle.write("verifier_started cuda=0\\n")
        {refusal}
        checkpoint = pathlib.Path(args.checkpoint)
        source_raw = checkpoint.read_bytes()
        source_sha = hashlib.sha256(source_raw).hexdigest()
        iteration = int(re.fullmatch(r"model_([0-9]+)\\.pt", checkpoint.name).group(1))
        roundtrip = (
            checkpoint.parent
            / ("exact_resume_roundtrip_" + {"d" * 16!r})
            / checkpoint.name
        )
        roundtrip.parent.mkdir()
        roundtrip.write_bytes(source_raw)
        verifier_sha = hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest()
        state_sha = "9" * 64
        receipt = {{
            "schema_version": 1,
            "kind": "action_ball_exact_resume_verification_v1",
            "status": "passed",
            "source_commit_sha": {"e" * 40!r},
            "launch_claim_sha256": {"d" * 64!r},
            "stage": "smoke",
            "namespace": str(pathlib.Path(args.out).parent),
            "verifier": {{
                "source_path": {M.EXACT_RESUME_VERIFIER_SOURCE!r},
                "source_sha256": verifier_sha,
                "runtime_factory_source_path": {M.SIDECAR_SOURCE!r},
                "runtime_factory_source_sha256": {"f" * 64!r},
            }},
            "source_checkpoint": {{
                "path": str(checkpoint),
                "sha256": source_sha,
                "size_bytes": len(source_raw),
                "embedded_iteration": iteration,
            }},
            "roundtrip_checkpoint": {{
                "path": str(roundtrip),
                "sha256": source_sha,
                "size_bytes": len(source_raw),
                "embedded_iteration": iteration,
            }},
            "runtime_bootstrap": {{
                "content_sha256": "1" * 64,
                "lineage_payload_sha256": "2" * 64,
            }},
            "restore": {{
                "factory_call_count": 2,
                "closed_runtime_count": 2,
                "load_optimizer": True,
                "fresh_strict_load_token_consumed": True,
                "roundtrip_save_api": "save_exact_resume_roundtrip",
                "roundtrip_save_receipt_sha256": "3" * 64,
                "source_construction_receipt_sha256": "4" * 64,
                "roundtrip_construction_receipt_sha256": "5" * 64,
                "runtime_inventory_live_verification_sha256": "6" * 64,
                "source_live_state_receipt_sha256": "7" * 64,
                "roundtrip_live_state_receipt_sha256": "7" * 64,
                "live_core_sha256": "8" * 64,
                "common_step_counter": 1,
                "common_step_counter_delta": 0,
            }},
            "state": {{
                "source_core_sha256": state_sha,
                "roundtrip_core_sha256": state_sha,
                "source_exact_resume_sha256": state_sha,
                "roundtrip_exact_resume_sha256": state_sha,
                "model_state_sha256": "a" * 64,
                "optimizer_state_sha256": "b" * 64,
                "normalizer_state_sha256": "c" * 64,
            }},
            "natural_exit": True,
        }}
        canonical = lambda value: json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        receipt["receipt_payload_sha256"] = hashlib.sha256(
            canonical(receipt)
        ).hexdigest()
        output = pathlib.Path(args.out)
        descriptor = os.open(
            output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            os.write(descriptor, canonical(receipt) + b"\\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with open({str(events)!r}, "a", encoding="utf-8") as handle:
            handle.write("verifier_done\\n")
        """
    )


def _validated_claim(
    *,
    checkout: Path,
    setup: Path,
    namespace: Path,
    runtime_paths: M.RuntimePaths,
    trainer_argv: tuple[str, ...],
    sidecar_argv: tuple[str, ...],
    ready: dict,
    exact_resume_verifier_path: Path | None = None,
) -> M.ValidatedClaim:
    verifier_path = (
        SOURCE
        if exact_resume_verifier_path is None
        else exact_resume_verifier_path
    )
    gpus = {
        "trainer": {
            "index": 0,
            "uuid": "GPU-trainer",
            "owner": "Franco",
            "lock_path": str(runtime_paths.trainer_lock),
            "boot_lock_path": str(runtime_paths.boot_lock),
            "require_empty": True,
            "owner_receipt_sha256": "a" * 64,
        },
        "evaluator": {
            "index": 1,
            "uuid": "GPU-evaluator",
            "owner": "Franco",
            "lock_path": str(runtime_paths.evaluator_lock),
            "boot_lock_path": str(runtime_paths.boot_lock),
            "require_empty": True,
            "owner_receipt_sha256": "b" * 64,
        },
    }
    import_root = namespace / "nosite-import-root"
    import_root.mkdir(exist_ok=True)
    import_roots = (NOSITE.bind_import_root(import_root),)
    return M.ValidatedClaim(
        claim_path=namespace / "launch_claim.json",
        claim_sha256="d" * 64,
        namespace=namespace,
        checkout=checkout,
        source_commit="e" * 40,
        stage="smoke",
        action_set_contract={
            "profile_id": "fixture_n1",
            "expected_n": 1,
            "experiment_name": M.ACTION_BALL_EXPERIMENT_NAME,
            "actor_obs_contract": "action_ball_n1",
            "actor_obs_width": 182,
            "namespace_identity": "n1-" + "0" * 12,
            "contract_sha256": "0" * 64,
        },
        trainer_argv=trainer_argv,
        sidecar_argv=sidecar_argv,
        runtime_code_sha256={
            M.EXACT_RESUME_VERIFIER_SOURCE: M._sha256_file(
                verifier_path
            ),
            M.NOSITE_BOOTSTRAP_SOURCE: M._sha256_file(
                NOSITE_SOURCE
            ),
            M.SIDECAR_SOURCE: "f" * 64,
        },
        gpus=gpus,
        setup_path=setup,
        exact_process_group_path=EXACT_SOURCE,
        exact_resume_verifier_path=verifier_path,
        nosite_bootstrap_path=NOSITE_SOURCE,
        nosite_import_roots=import_roots,
        max_iterations=1,
        expected_sidecar_ready=ready,
        heartbeat_path=namespace / "heartbeat.json",
        heartbeat_contract={
            "schema_version": 1,
            "heartbeat_interval_seconds": 5.0,
            "heartbeat_stale_after_seconds": 120.0,
            "request_deadline_seconds": 7200.0,
        },
        sidecar_code_sha256="f" * 64,
        sidecar_launch_content_sha256="c" * 64,
        sidecar_backend_contract_sha256="b" * 64,
    )


def _timing() -> M.Timing:
    return M.Timing(
        poll_seconds=0.01,
        boot_lock_timeout_seconds=1.0,
        sidecar_ready_timeout_seconds=2.0,
        trainer_ready_timeout_seconds=2.0,
        launcher_accept_timeout_seconds=2.0,
        publication_grace_seconds=0.5,
        exact_resume_timeout_seconds=2.0,
        term_grace_seconds=1.0,
        kill_grace_seconds=1.0,
    )


def _run_fixture(
    tmp_path: Path,
    *,
    sidecar_exit_after: float | None = None,
    trainer_learning: bool = True,
    trainer_exit_code: int = 0,
    trainer_loop: bool = False,
    duplicate_ready: bool = False,
    emit_ready: bool = True,
    verifier_exit_code: int = 0,
    stop_request: M.StopRequest | None = None,
    launcher_control_token: bytes | None = None,
) -> tuple[dict | None, Path, Path, M.RuntimePaths]:
    runtime_paths = _runtime_paths(tmp_path)
    trainer_fd = _locked_fd(runtime_paths.trainer_lock)
    evaluator_fd = _locked_fd(runtime_paths.evaluator_lock)
    checkout, setup, namespace = _setup_fixture(tmp_path)
    events = tmp_path / "events.txt"
    ready = _ready()
    verifier_path = checkout / M.EXACT_RESUME_VERIFIER_SOURCE
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.write_text(
        _verifier_code(
            events=events,
            trainer_fd=trainer_fd,
            evaluator_fd=evaluator_fd,
            boot_lock_path=runtime_paths.boot_lock,
            exit_code=verifier_exit_code,
        ),
        encoding="utf-8",
    )
    sidecar = (
        sys.executable,
        "-u",
        "-c",
        _sidecar_code(
            ready=ready,
            events=events,
            trainer_fd=trainer_fd,
            evaluator_fd=evaluator_fd,
            heartbeat_path=namespace / "heartbeat.json",
            exit_after=sidecar_exit_after,
            duplicate_ready=duplicate_ready,
            emit_ready=emit_ready,
        ),
    )
    trainer = (
        sys.executable,
        "-u",
        "-c",
        _trainer_code(
            events=events,
            trainer_fd=trainer_fd,
            evaluator_fd=evaluator_fd,
            checkout=checkout,
            namespace=namespace,
            print_learning=trainer_learning,
            exit_code=trainer_exit_code,
            loop_until_term=trainer_loop,
        ),
    )
    claim = _validated_claim(
        checkout=checkout,
        setup=setup,
        namespace=namespace,
        runtime_paths=runtime_paths,
        trainer_argv=trainer,
        sidecar_argv=sidecar,
        ready=ready,
        exact_resume_verifier_path=verifier_path,
    )
    control_read_fd: int | None = None
    control_write_fd: int | None = None
    if launcher_control_token is not None:
        control_read_fd, control_write_fd = os.pipe()

    def launcher_acceptance() -> None:
        deadline = time.monotonic() + 5.0
        ready_path = namespace / "supervisor_ready.json"
        failed_path = namespace / "supervisor_failed.json"
        intent_path = namespace / "launch_accept_intent.json"
        ack_path = namespace / "launch_accept_ack.json"
        while time.monotonic() < deadline:
            if failed_path.is_file():
                return
            if ready_path.is_file():
                try:
                    ready_document = M._read_strict_json(
                        ready_path, label="fixture supervisor ready"
                    )
                except M.SupervisorError:
                    time.sleep(0.005)
                    continue
                ready_sha = M._sha256_file(ready_path)
                intent = {
                    "schema_version": 1,
                    "kind": "action_ball_launcher_accept_intent",
                    "launch_claim_sha256": claim.claim_sha256,
                    "supervisor_ready_sha256": ready_sha,
                    "live_gpu_admission_sha256": M._sha256_file(
                        namespace / "live_gpu_admission.json"
                    ),
                }
                M._publish_exclusive_json(intent_path, intent)
                if control_write_fd is not None:
                    if launcher_control_token:
                        os.write(control_write_fd, launcher_control_token)
                    if launcher_control_token != b"A":
                        os.close(control_write_fd)
                        return
                while time.monotonic() < deadline:
                    if failed_path.is_file():
                        return
                    if ack_path.is_file():
                        try:
                            M._read_strict_json(
                                ack_path, label="fixture supervisor ack"
                            )
                        except M.SupervisorError:
                            time.sleep(0.005)
                            continue
                        ack_sha = M._sha256_file(ack_path)
                        accepted = {
                            "schema_version": 1,
                            "kind": "action_ball_launch_accepted",
                            "accepted_utc": "2026-07-29T00:00:00Z",
                            "stage": claim.stage,
                            "namespace": str(namespace),
                            "launch_claim_sha256": claim.claim_sha256,
                            "supervisor_ready": ready_document,
                            "accept_intent_sha256": M._sha256_file(
                                intent_path
                            ),
                            "supervisor_accept_ack_sha256": ack_sha,
                            "live_gpu_admission_sha256": M._sha256_file(
                                namespace / "live_gpu_admission.json"
                            ),
                        }
                        M._publish_exclusive_json(
                            namespace / "launch_accepted.json", accepted
                        )
                        if control_write_fd is not None:
                            commit_ack = namespace / "launch_commit_ack.json"
                            while time.monotonic() < deadline:
                                if commit_ack.is_file():
                                    os.close(control_write_fd)
                                    return
                                time.sleep(0.005)
                        return
                    time.sleep(0.005)
                return
            time.sleep(0.005)

    acceptance_thread = threading.Thread(
        target=launcher_acceptance, daemon=True
    )
    acceptance_thread.start()
    try:
        result = M.supervise_stage(
            claim,
            trainer_lock_fd=trainer_fd,
            evaluator_lock_fd=evaluator_fd,
            exact=HostExact(),
            timing=_timing(),
            runtime_paths=runtime_paths,
            stop_request=stop_request,
            launcher_control_fd=control_read_fd,
        )
        return result, namespace, events, runtime_paths
    finally:
        acceptance_thread.join(timeout=2.0)
        if control_write_fd is not None:
            try:
                os.close(control_write_fd)
            except OSError:
                pass
        os.close(trainer_fd)
        os.close(evaluator_fd)


def _gpu_row(
    role: str, runtime_paths: M.RuntimePaths
) -> dict:
    index = 0 if role == "trainer" else 1
    lock = (
        runtime_paths.trainer_lock
        if role == "trainer"
        else runtime_paths.evaluator_lock
    )
    return {
        "index": index,
        "uuid": f"GPU-{role}",
        "owner": "Franco",
        "lock_path": str(lock),
        "boot_lock_path": str(runtime_paths.boot_lock),
        "require_empty": True,
        "owner_receipt_sha256": ("a" if role == "trainer" else "b") * 64,
    }


def _claim_document(tmp_path: Path, runtime_paths: M.RuntimePaths) -> tuple[Path, str]:
    action_id = "fixture_action"
    manifest_sha = "9" * 64
    profile_id = "fixture_upper_nomove_n1_v1"
    order_uid_digest = M.canonical_sha256(
        {
            "schema_version": 1,
            "ordered_actions": [
                {
                    "index": 0,
                    "action_id": action_id,
                    "action_uid": 1001,
                }
            ],
        }
    )
    contract = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.action_set_contract",
        "profile_id": profile_id,
        "expected_n": 1,
        "scope": "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": [action_id],
        "ordered_action_uids": [1001],
        "order_uid_digest_sha256": order_uid_digest,
        "manifest_path": "configs/fixture-manifest.json",
        "manifest_sha256": manifest_sha,
        "experiment_name": M.ACTION_BALL_EXPERIMENT_NAME,
        "actor_obs_contract": "action_ball_n1",
        "actor_obs_width": 182,
        "namespace_identity": "n1-" + order_uid_digest[:12],
    }
    contract["contract_sha256"] = M.canonical_sha256(contract)
    namespace = tmp_path / (
        "claim-" + contract["namespace_identity"] + "-namespace"
    )
    namespace.mkdir()
    checkout = tmp_path / "claim-checkout"
    checkout.mkdir()
    bootstrap_path = checkout / M.NOSITE_BOOTSTRAP_SOURCE
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(NOSITE_SOURCE, bootstrap_path)
    trainer_entrypoint = checkout / "fixture_train_entrypoint.py"
    trainer_entrypoint.write_text(
        "# exact trainer entrypoint fixture\n", encoding="utf-8"
    )
    sidecar_entrypoint = checkout / M.SIDECAR_SOURCE
    sidecar_entrypoint.parent.mkdir(parents=True, exist_ok=True)
    sidecar_entrypoint.write_text(
        "# exact sidecar entrypoint fixture\n", encoding="utf-8"
    )
    import_root = checkout / "fixture_import_root"
    import_root.mkdir()
    import_roots = [NOSITE.bind_import_root(import_root)]
    owner = "Franco"
    run_id = "run-smoke"
    inbox = tmp_path / "inbox"
    trainer_args = [
        "train-entrypoint",
        "--",
        "device=cuda:0",
        f"task.experiment_name={contract['experiment_name']}",
        f"task.actor_obs_contract={contract['actor_obs_contract']}",
        (
            "task.racket.action_ball_manifest_path="
            f"{contract['manifest_path']}"
        ),
        (
            "task.racket.action_ball_manifest_sha256="
            f"{contract['manifest_sha256']}"
        ),
        "task.racket.clip_names=" + json.dumps([action_id]),
        f"task.racket.action_ball_evaluation_inbox_root={inbox}",
        f"task.racket.action_ball_evaluation_owner_id={owner}",
        f"task.racket.action_ball_evaluation_run_id={run_id}",
        "task.racket.action_ball_frozen_eval_interval_updates=100",
        f"++training_launch_claim_path={namespace / 'launch_claim.json'}",
    ]
    base_command = NOSITE.build_exact_nosite_argv(
        python=Path(sys.executable),
        bootstrap=bootstrap_path,
        bootstrap_sha256=M._sha256_file(bootstrap_path),
        entrypoint=trainer_entrypoint,
        entrypoint_sha256=M._sha256_file(trainer_entrypoint),
        import_roots=import_roots,
        entrypoint_argv=trainer_args,
    )
    argv_without = list(base_command.argv)
    sidecar_args = [
        "--inbox-root",
        str(inbox),
        "--owner-id",
        owner,
        "--run-id",
        run_id,
        "--launch",
        "/repo/sidecar-launch.json",
        "--backend",
        "formal",
        "--device",
        "cuda:0",
    ]
    sidecar_command = NOSITE.build_exact_nosite_argv(
        python=Path(sys.executable),
        bootstrap=bootstrap_path,
        bootstrap_sha256=M._sha256_file(bootstrap_path),
        entrypoint=sidecar_entrypoint,
        entrypoint_sha256=M._sha256_file(sidecar_entrypoint),
        import_roots=import_roots,
        entrypoint_argv=sidecar_args,
    )
    sidecar = list(sidecar_command.argv)
    payload = {
        "schema_version": 3,
        "kind": M.PAYLOAD_KIND,
        "launch_profile": profile_id,
        "action_set_contract": contract,
        "ordered_action_ids": [action_id],
        "manifest": {
            "path": contract["manifest_path"],
            "sha256": manifest_sha,
        },
        "stage": "smoke",
        "source_checkout": str(checkout),
        "source_commit_sha": "e" * 40,
        "runtime_code_sha256": {
            M.NOSITE_BOOTSTRAP_SOURCE: M._sha256_file(
                bootstrap_path
            ),
        },
        "namespace": str(namespace),
        "gpus": {
            "trainer": _gpu_row("trainer", runtime_paths),
            "evaluator": _gpu_row("evaluator", runtime_paths),
        },
        "argv_without_launch_claim": argv_without,
        "sidecar_argv": sidecar,
        "sidecar_launch_receipt": {
            "canonical_sha256": "c" * 64,
            "content_sha256": "c" * 64,
            "sidecar_code_sha256": "f" * 64,
            "backend_contract_sha256": "b" * 64,
            "heartbeat_contract": {
                "schema_version": 1,
                "heartbeat_interval_seconds": 5.0,
                "heartbeat_stale_after_seconds": 120.0,
                "request_deadline_seconds": 7200.0,
            },
        },
        "isaac_python_runtime": {
            "path": sys.executable,
            "runtime_inventory": {
                "import_roots": import_roots,
            },
        },
        "isolated_training_entrypoint": {
            "nosite_argv_contract_sha256": (
                base_command.contract_sha256
            ),
            "nosite_argv_contract": dict(base_command.contract),
        },
        "sidecar_nosite_execution": {
            "nosite_argv_contract_sha256": (
                sidecar_command.contract_sha256
            ),
            "nosite_argv_contract": dict(sidecar_command.contract),
        },
        "stage_budget": {
            "max_iterations": 1,
        },
        "frozen_evaluation_runtime": {
            "inbox_root": str(inbox),
            "owner_id": owner,
            "run_id": run_id,
            "interval_updates": 100,
        },
    }
    claim_sha = M.canonical_sha256(payload)
    final_command = NOSITE.build_exact_nosite_argv(
        python=Path(sys.executable),
        bootstrap=bootstrap_path,
        bootstrap_sha256=M._sha256_file(bootstrap_path),
        entrypoint=trainer_entrypoint,
        entrypoint_sha256=M._sha256_file(trainer_entrypoint),
        import_roots=import_roots,
        entrypoint_argv=[
            *trainer_args,
            f"++training_launch_claim_sha256={claim_sha}",
        ],
    )
    claim = {
        "schema_version": 3,
        "kind": M.CLAIM_KIND,
        "launch_claim_sha256": claim_sha,
        "canonical_payload": payload,
        "argv": list(final_command.argv),
        "confirmation_claim_sha256": claim_sha,
    }
    path = namespace / "launch_claim.json"
    path.write_text(
        json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path, claim_sha


def _rebuild_nosite_argv(argv: list[str], entrypoint_argv: list[str]) -> list[str]:
    contract = M._decode_nosite_argv(
        tuple(argv), label="fixture no-site argv"
    )
    command = NOSITE.build_exact_nosite_argv(
        python=Path(argv[0]),
        bootstrap=Path(contract["bootstrap"]["path"]),
        bootstrap_sha256=contract["bootstrap"]["sha256"],
        entrypoint=Path(contract["entrypoint"]["path"]),
        entrypoint_sha256=contract["entrypoint"]["sha256"],
        import_roots=contract["import_roots"],
        entrypoint_argv=entrypoint_argv,
    )
    return list(command.argv)


def _rebind_claim_final_argv(claim: dict, claim_sha: str) -> None:
    base = claim["canonical_payload"]["argv_without_launch_claim"]
    contract = M._decode_nosite_argv(
        tuple(base), label="fixture base trainer argv"
    )
    claim["argv"] = _rebuild_nosite_argv(
        base,
        [
            *contract["entrypoint_argv"],
            f"++training_launch_claim_sha256={claim_sha}",
        ],
    )


def _write_heartbeat(
    path: Path,
    *,
    pid: int,
    heartbeat_seq: int,
    heartbeat_monotonic_ns: int,
    phase: str = "ready",
    request_seq=None,
    request_sha256: str = "",
    attempts_completed: int = 0,
    attempts_total: int = 0,
    request_started_unix_ns: int = 0,
    request_started_monotonic_ns: int = 0,
    request_deadline_unix_ns: int = 0,
    request_deadline_monotonic_ns: int = 0,
    error_type: str = "",
) -> dict:
    content = {
        "owner_id": "Franco",
        "run_id": "run-smoke",
        "pid": pid,
        "sidecar_code_sha256": "f" * 64,
        "launch_sha256": "c" * 64,
        "backend_contract_sha256": "b" * 64,
        "heartbeat_seq": heartbeat_seq,
        "phase": phase,
        "request_seq": request_seq,
        "request_sha256": request_sha256,
        "attempts_completed": attempts_completed,
        "attempts_total": attempts_total,
        "request_started_unix_ns": request_started_unix_ns,
        "request_started_monotonic_ns": request_started_monotonic_ns,
        "request_deadline_unix_ns": request_deadline_unix_ns,
        "request_deadline_monotonic_ns": request_deadline_monotonic_ns,
        "heartbeat_unix_ns": 1_900_000_000_000_000_000 + heartbeat_seq,
        "heartbeat_monotonic_ns": heartbeat_monotonic_ns,
        "error_type": error_type,
    }
    document = {
        "schema_version": 1,
        "kind": M.HEARTBEAT_KIND,
        "content": content,
        "content_sha256": M.canonical_sha256(content),
    }
    path.write_bytes(M._canonical_bytes(document) + b"\n")
    return document


def test_heartbeat_strict_progress_and_deadlines(tmp_path: Path) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    checkout, setup, namespace = _setup_fixture(tmp_path)
    claim = _validated_claim(
        checkout=checkout,
        setup=setup,
        namespace=namespace,
        runtime_paths=runtime_paths,
        trainer_argv=("trainer",),
        sidecar_argv=("sidecar",),
        ready=_ready(),
    )
    evaluator = SimpleNamespace(identity=SimpleNamespace(pid=4242))
    now = 10_000_000_000
    _write_heartbeat(
        claim.heartbeat_path,
        pid=4242,
        heartbeat_seq=0,
        heartbeat_monotonic_ns=now,
    )
    first = M._observe_sidecar_heartbeat(
        claim, evaluator, None, now_monotonic_ns=now
    )
    assert M._observe_sidecar_heartbeat(
        claim, evaluator, first, now_monotonic_ns=now + 1
    ) is first
    with pytest.raises(M.SupervisorError, match="120s stale"):
        M._observe_sidecar_heartbeat(
            claim,
            evaluator,
            first,
            now_monotonic_ns=now + 120_000_000_001,
        )

    started_mono = now + 2
    started_unix = 1_800_000_000_000_000_000
    duration = 7_200_000_000_000
    _write_heartbeat(
        claim.heartbeat_path,
        pid=4242,
        heartbeat_seq=1,
        heartbeat_monotonic_ns=started_mono,
        phase="request_accepted",
        request_seq=0,
        request_sha256="a" * 64,
        attempts_total=10,
        request_started_unix_ns=started_unix,
        request_started_monotonic_ns=started_mono,
        request_deadline_unix_ns=started_unix + duration,
        request_deadline_monotonic_ns=started_mono + duration,
    )
    active = M._observe_sidecar_heartbeat(
        claim, evaluator, first, now_monotonic_ns=started_mono
    )
    _write_heartbeat(
        claim.heartbeat_path,
        pid=4242,
        heartbeat_seq=2,
        heartbeat_monotonic_ns=started_mono + 1,
        phase="evaluating",
        request_seq=0,
        request_sha256="a" * 64,
        attempts_completed=5,
        attempts_total=10,
        request_started_unix_ns=started_unix,
        request_started_monotonic_ns=started_mono,
        request_deadline_unix_ns=started_unix + duration,
        request_deadline_monotonic_ns=started_mono + duration,
    )
    progressed = M._observe_sidecar_heartbeat(
        claim, evaluator, active, now_monotonic_ns=started_mono + 1
    )
    assert progressed.attempts_completed == 5
    expired_now = started_mono + duration + 1
    _write_heartbeat(
        claim.heartbeat_path,
        pid=4242,
        heartbeat_seq=3,
        heartbeat_monotonic_ns=expired_now,
        phase="evaluating",
        request_seq=0,
        request_sha256="a" * 64,
        attempts_completed=5,
        attempts_total=10,
        request_started_unix_ns=started_unix,
        request_started_monotonic_ns=started_mono,
        request_deadline_unix_ns=started_unix + duration,
        request_deadline_monotonic_ns=started_mono + duration,
    )
    with pytest.raises(M.SupervisorError, match="7200s deadline"):
        M._observe_sidecar_heartbeat(
            claim,
            evaluator,
            progressed,
            now_monotonic_ns=expired_now,
        )


def test_heartbeat_rejects_noncanonical_and_attempt_regression(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    checkout, setup, namespace = _setup_fixture(tmp_path)
    claim = _validated_claim(
        checkout=checkout,
        setup=setup,
        namespace=namespace,
        runtime_paths=runtime_paths,
        trainer_argv=("trainer",),
        sidecar_argv=("sidecar",),
        ready=_ready(),
    )
    evaluator = SimpleNamespace(identity=SimpleNamespace(pid=4242))
    document = _write_heartbeat(
        claim.heartbeat_path,
        pid=4242,
        heartbeat_seq=0,
        heartbeat_monotonic_ns=100,
    )
    claim.heartbeat_path.write_text(json.dumps(document) + "\n")
    with pytest.raises(M.SupervisorError, match="canonical"):
        M._observe_sidecar_heartbeat(
            claim, evaluator, None, now_monotonic_ns=100
        )


def test_inflight_o_excl_publication_waits_for_complete_canonical_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inflight.json"
    expected = {"schema_version": 1, "kind": "fixture"}

    def publish_in_place() -> None:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            time.sleep(0.05)
            os.write(descriptor, M._canonical_bytes(expected) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    writer = threading.Thread(target=publish_in_place)
    writer.start()
    deadline = time.monotonic() + 1.0
    while not path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.001)
    observed = M._read_inflight_published_json(
        path,
        label="fixture inflight receipt",
        timing=_timing(),
        guard=lambda: None,
    )
    writer.join(timeout=1.0)
    assert observed == expected


def test_schema_v3_claim_binds_two_gpus_formal_sidecar_and_trainer(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    claim_path, claim_sha = _claim_document(tmp_path, runtime_paths)
    claim, payload, gpus = M._validate_claim_document(
        claim_path,
        claim_sha,
        runtime_paths=runtime_paths,
    )
    assert claim["launch_claim_sha256"] == claim_sha
    assert payload["stage"] == "smoke"
    assert gpus["trainer"]["index"] == 0
    assert gpus["evaluator"]["index"] == 1


def test_schema_rejects_equal_integer_heartbeat_contract_tamper(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    claim_path, _old_sha = _claim_document(tmp_path, runtime_paths)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    payload = claim["canonical_payload"]
    payload["sidecar_launch_receipt"]["heartbeat_contract"][
        "heartbeat_interval_seconds"
    ] = 5
    claim_sha = M.canonical_sha256(payload)
    claim["launch_claim_sha256"] = claim_sha
    claim["confirmation_claim_sha256"] = claim_sha
    _rebind_claim_final_argv(claim, claim_sha)
    claim_path.write_text(
        json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(M.SupervisorError, match="heartbeat contract"):
        M._validate_claim_document(
            claim_path, claim_sha, runtime_paths=runtime_paths
        )


def test_git_ignores_caller_path_and_git_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_git = shutil.which("git", path=os.defpath)
    assert trusted_git is not None
    checkout = tmp_path / "repo"
    subprocess.run(
        [trusted_git, "init", "-q", str(checkout)],
        check=True,
        env={
            "PATH": os.defpath,
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        },
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "fake-git-ran"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!/bin/sh\nprintf injected > {marker!s}\nexit 91\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "attacker-tree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "attacker-index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "objects"))
    monkeypatch.setenv(
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", str(tmp_path / "alternate")
    )
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.bare")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "inject.so"))
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", str(tmp_path / "inject.dylib"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    root = M._git(checkout, ["rev-parse", "--show-toplevel"]).strip()
    assert Path(root) == checkout
    assert not marker.exists()


def test_child_environment_uses_fixed_path_and_drops_loader_python_git_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poisoned = {
        "PATH": "/attacker/bin",
        "PYTHONPATH": "/attacker/python",
        "PYTHONHOME": "/attacker/home",
        "BASH_ENV": "/attacker/bash-env",
        "ENV": "/attacker/sh-env",
        "GIT_DIR": "/attacker/git",
        "GIT_CONFIG_GLOBAL": "/attacker/gitconfig",
        "LD_PRELOAD": "/attacker/lib.so",
        "DYLD_INSERT_LIBRARIES": "/attacker/lib.dylib",
        "XDG_CONFIG_HOME": "/attacker/xdg",
    }
    for key, value in poisoned.items():
        monkeypatch.setenv(key, value)
    env = M._child_env("trainer")
    assert env["PATH"] == os.defpath
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    for key in poisoned:
        if key != "PATH":
            assert key not in env


def test_verifier_inherits_boot_flock_across_parent_descriptor_close(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    trainer_fd = _locked_fd(runtime_paths.trainer_lock)
    evaluator_fd = _locked_fd(runtime_paths.evaluator_lock)
    checkout, setup, namespace = _setup_fixture(tmp_path)
    boot_fd = M._open_boot_lock(runtime_paths.boot_lock)
    M._acquire_boot_lock(
        boot_fd, timing=_timing(), guard=lambda: None
    )
    claim = _validated_claim(
        checkout=checkout,
        setup=setup,
        namespace=namespace,
        runtime_paths=runtime_paths,
        trainer_argv=("trainer",),
        sidecar_argv=("sidecar",),
        ready=_ready(),
    )
    code = textwrap.dedent(
        f"""\
        import os
        import signal
        import time
        os.fstat({boot_fd})
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
        while True:
            time.sleep(0.02)
        """
    )
    exact = HostExact()
    child = M._start_child(
        role="verifier",
        argv=(sys.executable, "-u", "-c", code),
        claim=claim,
        lock_fds=(trainer_fd, evaluator_fd, boot_fd),
        exact=exact,
    )
    os.close(boot_fd)
    contender = os.open(
        runtime_paths.boot_lock, os.O_RDWR | os.O_APPEND
    )
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped = M._stop_exact_child(
            child,
            claim=claim,
            exact=exact,
            timing=_timing(),
        )
        assert stopped["forced_kill"] is False
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        child.log.close()
        os.close(contender)
        os.close(trainer_fd)
        os.close(evaluator_fd)


def test_schema_refuses_cpu_fake_sidecar_even_when_claim_hash_is_recomputed(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    claim_path, _ = _claim_document(tmp_path, runtime_paths)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    argv = claim["canonical_payload"]["sidecar_argv"]
    sidecar_contract = M._decode_nosite_argv(
        tuple(argv), label="fixture sidecar argv"
    )
    sidecar_args = list(sidecar_contract["entrypoint_argv"])
    sidecar_args[sidecar_args.index("formal")] = "cpu-fake"
    claim["canonical_payload"]["sidecar_argv"] = _rebuild_nosite_argv(
        argv, sidecar_args
    )
    new_sha = M.canonical_sha256(claim["canonical_payload"])
    claim["launch_claim_sha256"] = new_sha
    claim["confirmation_claim_sha256"] = new_sha
    _rebind_claim_final_argv(claim, new_sha)
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    with pytest.raises(M.SupervisorError, match="CPU-fake"):
        M._validate_claim_document(
            claim_path, new_sha, runtime_paths=runtime_paths
        )


def test_schema_requires_exact_hydra_claim_path_then_final_sha(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    claim_path, claim_sha = _claim_document(tmp_path, runtime_paths)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    M._validate_claim_document(
        claim_path, claim_sha, runtime_paths=runtime_paths
    )

    wrong_path = json.loads(json.dumps(claim))
    final_contract = M._decode_nosite_argv(
        tuple(wrong_path["argv"]), label="fixture final trainer argv"
    )
    wrong_args = list(final_contract["entrypoint_argv"])
    path_index = next(
        index
        for index, token in enumerate(wrong_args)
        if token.startswith("++training_launch_claim_path=")
    )
    wrong_args[path_index] = (
        "++training_launch_claim_path=/tmp/other/launch_claim.json"
    )
    wrong_path["argv"] = _rebuild_nosite_argv(
        wrong_path["argv"], wrong_args
    )
    claim_path.write_text(json.dumps(wrong_path), encoding="utf-8")
    with pytest.raises(M.SupervisorError, match="exactly bound"):
        M._validate_claim_document(
            claim_path, claim_sha, runtime_paths=runtime_paths
        )

    bare_sha = json.loads(json.dumps(claim))
    final_contract = M._decode_nosite_argv(
        tuple(bare_sha["argv"]), label="fixture final trainer argv"
    )
    bare_args = list(final_contract["entrypoint_argv"])
    bare_args[-1] = f"training_launch_claim_sha256={claim_sha}"
    bare_sha["argv"] = _rebuild_nosite_argv(
        bare_sha["argv"], bare_args
    )
    claim_path.write_text(json.dumps(bare_sha), encoding="utf-8")
    with pytest.raises(M.SupervisorError, match="exactly bound"):
        M._validate_claim_document(
            claim_path, claim_sha, runtime_paths=runtime_paths
        )


def test_evaluator_ready_precedes_trainer_and_clean_shutdown_is_receipted(
    tmp_path: Path,
) -> None:
    result, namespace, events, runtime_paths = _run_fixture(tmp_path)
    assert result is not None
    assert result["status"] == "completed"
    assert result["trainer_returncode"] == 0
    assert result["evaluator_returncode"] in (0, -15)
    assert (namespace / "supervisor_ready.json").is_file()
    assert (namespace / "launch_commit_ack.json").is_file()
    assert (namespace / "supervisor_terminal.json").is_file()
    assert not (namespace / "supervisor_failed.json").exists()
    lines = events.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sidecar_ready cuda=1"
    assert lines[1] == "trainer_started cuda=0"
    assert lines[-3:] == [
        "sidecar_term",
        "verifier_started cuda=0",
        "verifier_done",
    ]
    assert (namespace / "exact_resume_verification.json").is_file()
    assert result["processes"]["verifier"]["returncode"] == 0
    assert result["cleanup"][-1]["receipt"]["payload_sha256"]
    assert runtime_paths.boot_lock.read_bytes() == b"do-not-truncate\n"
    post_verifier_boot_fd = _locked_fd(runtime_paths.boot_lock)
    os.close(post_verifier_boot_fd)
    for role in ("evaluator", "trainer"):
        identity = json.loads(
            (namespace / f"{role}_leader_identity.json").read_text(
                encoding="utf-8"
            )
        )["leader"]
        assert identity["pid"] == identity["pgid"]
        assert identity["starttime_ticks"] > 0
        assert len(
            result["processes"][role]["argv_sha256"]
        ) == 64
        assert len(
            result["processes"][role]["leader_receipt_sha256"]
        ) == 64


def test_launcher_control_a_and_final_commit_ack_close_two_phase_acceptance(
    tmp_path: Path,
) -> None:
    result, namespace, _events, _runtime_paths = _run_fixture(
        tmp_path,
        launcher_control_token=b"A",
    )
    assert result is not None and result["status"] == "completed"
    ready_path = namespace / "supervisor_ready.json"
    intent_path = namespace / "launch_accept_intent.json"
    precommit_ack_path = namespace / "launch_accept_ack.json"
    accepted_path = namespace / "launch_accepted.json"
    commit_ack = json.loads(
        (namespace / "launch_commit_ack.json").read_text(encoding="utf-8")
    )
    assert commit_ack == {
        "schema_version": 1,
        "kind": "action_ball_stage_supervisor_launch_commit_ack",
        "launch_claim_sha256": "d" * 64,
        "supervisor_ready_sha256": M._sha256_file(ready_path),
        "accept_intent_sha256": M._sha256_file(intent_path),
        "supervisor_accept_ack_sha256": M._sha256_file(precommit_ack_path),
        "launch_accepted_sha256": M._sha256_file(accepted_path),
        "live_gpu_admission_sha256": M._sha256_file(
            namespace / "live_gpu_admission.json"
        ),
        "processes": json.loads(
            ready_path.read_text(encoding="utf-8")
        )["processes"],
    }


def test_verifier_failure_blocks_terminal_and_publishes_closed_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        M.SupervisorError, match="exact-resume verifier exited nonzero"
    ):
        _run_fixture(tmp_path, verifier_exit_code=23)
    namespace = tmp_path / "namespace"
    assert not (namespace / "supervisor_terminal.json").exists()
    assert not (namespace / "exact_resume_verification.json").exists()
    failure = json.loads(
        (namespace / "supervisor_failed.json").read_text(encoding="utf-8")
    )
    assert failure["state"] == "waiting_exact_resume_verifier"
    assert failure["cleanup_status"] == "closed"
    assert failure["processes"]["verifier"]["returncode"] == 23
    post_failure_boot_fd = _locked_fd(tmp_path / "kit_boot.lock")
    os.close(post_failure_boot_fd)


def test_launcher_control_cancel_exactly_reaps_both_children(
    tmp_path: Path,
) -> None:
    with pytest.raises(M.SupervisorError, match="cancelled"):
        _run_fixture(
            tmp_path,
            trainer_loop=True,
            launcher_control_token=b"C",
        )
    namespace = tmp_path / "namespace"
    assert (namespace / "trainer_pre_term_identity.json").is_file()
    assert (namespace / "evaluator_pre_term_identity.json").is_file()
    failure = json.loads(
        (namespace / "supervisor_failed.json").read_text(encoding="utf-8")
    )
    assert failure["cleanup_status"] == "closed"


def test_evaluator_early_exit_terms_exact_trainer_group_and_fails(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        M.SupervisorError, match="evaluator exited while trainer was active"
    ):
        _run_fixture(
            tmp_path,
            sidecar_exit_after=0.5,
            trainer_loop=True,
        )
    namespace = tmp_path / "namespace"
    failure = json.loads(
        (namespace / "supervisor_failed.json").read_text(encoding="utf-8")
    )
    assert failure["claim_sha256"] == "d" * 64
    assert failure["state"] == "running"
    assert (namespace / "trainer_pre_term_identity.json").is_file()
    assert "trainer_term" in (tmp_path / "events.txt").read_text(
        encoding="utf-8"
    )


def test_trainer_exit_before_learning_terms_sidecar_and_never_publishes_ready(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        M.SupervisorError, match="trainer exited before Learning iteration"
    ):
        _run_fixture(
            tmp_path,
            trainer_learning=False,
            trainer_exit_code=9,
        )
    namespace = tmp_path / "namespace"
    assert not (namespace / "supervisor_ready.json").exists()
    assert (namespace / "supervisor_failed.json").is_file()
    assert (namespace / "evaluator_pre_term_identity.json").is_file()


def test_duplicate_sidecar_ready_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(
        M.SupervisorError, match="more than one ready line"
    ):
        _run_fixture(tmp_path, duplicate_ready=True)
    namespace = tmp_path / "namespace"
    assert (namespace / "supervisor_failed.json").is_file()
    assert not (namespace / "supervisor_ready.json").exists()


def test_no_clobber_refuses_before_spawning_children(tmp_path: Path) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    trainer_fd = _locked_fd(runtime_paths.trainer_lock)
    evaluator_fd = _locked_fd(runtime_paths.evaluator_lock)
    checkout, setup, namespace = _setup_fixture(tmp_path)
    sentinel = namespace / "evaluator.log"
    sentinel.write_text("belongs to earlier attempt\n", encoding="utf-8")
    claim = _validated_claim(
        checkout=checkout,
        setup=setup,
        namespace=namespace,
        runtime_paths=runtime_paths,
        trainer_argv=(sys.executable, "-c", "raise SystemExit(99)"),
        sidecar_argv=(sys.executable, "-c", "raise SystemExit(99)"),
        ready=_ready(),
    )
    try:
        with pytest.raises(M.SupervisorError, match="already exist"):
            M.supervise_stage(
                claim,
                trainer_lock_fd=trainer_fd,
                evaluator_lock_fd=evaluator_fd,
                exact=HostExact(),
                timing=_timing(),
                runtime_paths=runtime_paths,
            )
    finally:
        os.close(trainer_fd)
        os.close(evaluator_fd)
    assert sentinel.read_text(encoding="utf-8") == "belongs to earlier attempt\n"
    assert not (namespace / "train.log").exists()


def test_bind_failure_never_releases_workload_start_gate(tmp_path: Path) -> None:
    class RejectBind(HostExact):
        def bind_leader(self, *args, **kwargs):
            raise RuntimeError("synthetic bind refusal")

    runtime_paths = _runtime_paths(tmp_path)
    trainer_fd = _locked_fd(runtime_paths.trainer_lock)
    evaluator_fd = _locked_fd(runtime_paths.evaluator_lock)
    checkout, setup, namespace = _setup_fixture(tmp_path)
    events = tmp_path / "events.txt"
    ready = _ready()
    sidecar = (
        sys.executable,
        "-u",
        "-c",
        _sidecar_code(
            ready=ready,
                events=events,
                trainer_fd=trainer_fd,
                evaluator_fd=evaluator_fd,
                heartbeat_path=namespace / "heartbeat.json",
            ),
    )
    claim = _validated_claim(
        checkout=checkout,
        setup=setup,
        namespace=namespace,
        runtime_paths=runtime_paths,
        trainer_argv=(sys.executable, "-c", "raise SystemExit(0)"),
        sidecar_argv=sidecar,
        ready=ready,
    )
    try:
        with pytest.raises(RuntimeError, match="bind refusal"):
            M._start_child(
                role="evaluator",
                argv=sidecar,
                claim=claim,
                lock_fds=(trainer_fd, evaluator_fd),
                exact=RejectBind(),
            )
    finally:
        os.close(trainer_fd)
        os.close(evaluator_fd)
    assert not events.exists()
    assert not (namespace / "evaluator_leader_identity.json").exists()
    assert (namespace / "evaluator.log").read_bytes() == b""


def test_missing_boot_lock_refuses_instead_of_creating_coordination_domain(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-kit-boot.lock"
    with pytest.raises(FileNotFoundError):
        M._open_boot_lock(missing)
    assert not missing.exists()


def test_stop_request_breaks_boot_lock_wait_without_starting_child(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    blocker_fd = _locked_fd(runtime_paths.boot_lock)
    trainer_fd = _locked_fd(runtime_paths.trainer_lock)
    evaluator_fd = _locked_fd(runtime_paths.evaluator_lock)
    checkout, setup, namespace = _setup_fixture(tmp_path)
    stop = M.StopRequest()
    timer = threading.Timer(0.05, stop.handler, args=(signal.SIGTERM, None))
    claim = _validated_claim(
        checkout=checkout,
        setup=setup,
        namespace=namespace,
        runtime_paths=runtime_paths,
        trainer_argv=(sys.executable, "-c", "raise SystemExit(99)"),
        sidecar_argv=(sys.executable, "-c", "raise SystemExit(99)"),
        ready=_ready(),
    )
    timer.start()
    try:
        with pytest.raises(M.SupervisorError, match="SIGTERM"):
            M.supervise_stage(
                claim,
                trainer_lock_fd=trainer_fd,
                evaluator_lock_fd=evaluator_fd,
                exact=HostExact(),
                timing=_timing(),
                runtime_paths=runtime_paths,
                stop_request=stop,
            )
    finally:
        timer.join()
        os.close(blocker_fd)
        os.close(trainer_fd)
        os.close(evaluator_fd)
    assert (namespace / "supervisor_failed.json").is_file()
    assert not (namespace / "evaluator.log").exists()
    assert not (namespace / "train.log").exists()


def test_stop_request_during_sidecar_ready_wait_exactly_terms_sidecar(
    tmp_path: Path,
) -> None:
    stop = M.StopRequest()
    timer = threading.Timer(0.08, stop.handler, args=(signal.SIGTERM, None))
    timer.start()
    try:
        with pytest.raises(M.SupervisorError, match="SIGTERM"):
            _run_fixture(
                tmp_path,
                emit_ready=False,
                stop_request=stop,
            )
    finally:
        timer.join()
    namespace = tmp_path / "namespace"
    assert (namespace / "evaluator_pre_term_identity.json").is_file()
    assert not (namespace / "trainer.log").exists()
    assert (namespace / "supervisor_failed.json").is_file()


def test_stop_request_during_running_exactly_terms_both_groups(
    tmp_path: Path,
) -> None:
    stop = M.StopRequest()
    events = tmp_path / "events.txt"
    commit_ack = tmp_path / "namespace/launch_commit_ack.json"

    def request_after_trainer_started() -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if commit_ack.is_file():
                # The O_EXCL commit ACK is the last acceptance artifact; give
                # the supervisor one poll to enter its steady running loop.
                time.sleep(0.03)
                stop.handler(signal.SIGTERM, None)
                return
            time.sleep(0.01)
        stop.handler(signal.SIGTERM, None)

    watcher = threading.Thread(
        target=request_after_trainer_started, daemon=True
    )
    watcher.start()
    with pytest.raises(M.SupervisorError, match="SIGTERM"):
        _run_fixture(
            tmp_path,
            trainer_loop=True,
            stop_request=stop,
        )
    watcher.join(timeout=2.0)
    namespace = tmp_path / "namespace"
    assert (namespace / "trainer_pre_term_identity.json").is_file()
    assert (namespace / "evaluator_pre_term_identity.json").is_file()
    failure = json.loads(
        (namespace / "supervisor_failed.json").read_text(encoding="utf-8")
    )
    assert failure["state"] == "running"


def test_full_preflight_reopens_exact_clean_commit_and_refuses_dirty_source(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    claim_path, _old_sha = _claim_document(tmp_path, runtime_paths)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    checkout = Path(claim["canonical_payload"]["source_checkout"])
    sources = {
        M.SUPERVISOR_SOURCE: SOURCE,
        M.EXACT_PROCESS_GROUP_SOURCE: EXACT_SOURCE,
            M.EXACT_RESUME_VERIFIER_SOURCE: (
                ROOT / "scripts/action_ball_exact_resume_verifier.py"
            ),
            M.NOSITE_BOOTSTRAP_SOURCE: NOSITE_SOURCE,
            M.ACTION_SET_CONTRACT_SOURCE: (
                ROOT / "scripts/action_ball_action_set_contract.py"
            ),
    }
    for relative, source in sources.items():
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    contract_path = checkout / M.ACTION_SET_CONTRACT_SOURCE
    contract_identity = claim["canonical_payload"]["action_set_contract"]
    contract_keys = {
        "profile_id",
        "expected_n",
        "scope",
        "mobility_mode",
        "ordered_action_ids",
        "ordered_action_uids",
        "order_uid_digest_sha256",
        "manifest_path",
        "manifest_sha256",
        "experiment_name",
    }
    contract_row = {
        key: contract_identity[key] for key in contract_keys
    }
    contract_text = contract_path.read_text(encoding="utf-8").replace(
        "ACTION_SET_CONTRACTS = {}",
        "ACTION_SET_CONTRACTS = {!r}".format(
            {contract_identity["profile_id"]: contract_row}
        ),
    )
    contract_path.write_text(contract_text, encoding="utf-8")
    setup = checkout / M.SETUP_SOURCE
    setup.parent.mkdir(parents=True, exist_ok=True)
    setup.write_text("export TEST_EXACT_SETUP=1\n", encoding="utf-8")
    sidecar = checkout / M.SIDECAR_SOURCE
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("# exact formal sidecar fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "add", "."], check=True
    )
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-qm", "fixture"],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    pins = {
        relative: M._sha256_file(checkout / relative)
        for relative in (
            M.SUPERVISOR_SOURCE,
            M.EXACT_PROCESS_GROUP_SOURCE,
            M.EXACT_RESUME_VERIFIER_SOURCE,
            M.NOSITE_BOOTSTRAP_SOURCE,
            M.ACTION_SET_CONTRACT_SOURCE,
            M.SETUP_SOURCE,
            M.SIDECAR_SOURCE,
        )
    }
    sidecar_sha = pins[M.SIDECAR_SOURCE]
    payload = claim["canonical_payload"]
    payload["source_commit_sha"] = commit
    payload["runtime_code_sha256"] = pins
    payload["sidecar_launch_receipt"]["sidecar_code_sha256"] = sidecar_sha
    sidecar_contract = M._decode_nosite_argv(
        tuple(payload["sidecar_argv"]), label="fixture sidecar argv"
    )
    sidecar_command = NOSITE.build_exact_nosite_argv(
        python=Path(payload["sidecar_argv"][0]),
        bootstrap=checkout / M.NOSITE_BOOTSTRAP_SOURCE,
        bootstrap_sha256=pins[M.NOSITE_BOOTSTRAP_SOURCE],
        entrypoint=sidecar,
        entrypoint_sha256=sidecar_sha,
        import_roots=sidecar_contract["import_roots"],
        entrypoint_argv=sidecar_contract["entrypoint_argv"],
    )
    payload["sidecar_argv"] = list(sidecar_command.argv)
    payload["sidecar_nosite_execution"] = {
        "nosite_argv_contract_sha256": sidecar_command.contract_sha256,
        "nosite_argv_contract": dict(sidecar_command.contract),
    }
    claim_sha = M.canonical_sha256(payload)
    claim["launch_claim_sha256"] = claim_sha
    claim["confirmation_claim_sha256"] = claim_sha
    _rebind_claim_final_argv(claim, claim_sha)
    claim_path.write_text(
        json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    validated, exact = M.validate_claim_and_source(
        str(claim_path),
        claim_sha,
        runtime_paths=runtime_paths,
        self_path=checkout / M.SUPERVISOR_SOURCE,
    )
    assert validated.source_commit == commit
    assert validated.exact_resume_verifier_path == (
        checkout / M.EXACT_RESUME_VERIFIER_SOURCE
    )
    assert exact.__file__ == str(checkout / M.EXACT_PROCESS_GROUP_SOURCE)

    setup.write_text("export TEST_EXACT_SETUP=dirty\n", encoding="utf-8")
    with pytest.raises(M.SupervisorError, match="not exact-clean"):
        M.validate_claim_and_source(
            str(claim_path),
            claim_sha,
            runtime_paths=runtime_paths,
            self_path=checkout / M.SUPERVISOR_SOURCE,
        )
